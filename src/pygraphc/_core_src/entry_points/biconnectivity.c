/* Bridges, articulation points and biconnected components (iterative Tarjan). */
#include "entry_points.h"
#include "../helpers/parse.h"

/* ── Bridges (iterative Tarjan's) ── */

PyObject *py_bridges(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }
    if (el.edge_count == 0) { free_edges(&el); return result; }

    AdjacencyList al;
    if (build_adjacency(n, &el, &al, 0) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) {
        free_adjacency(&al); free_edges(&el); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, (size_t)n * sizeof(int));

    int timer = 0;
    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i];
                int eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else {
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        int eid = peid[u];
                        PyObject *tup = Py_BuildValue(
                            "(ii)", EDGE_FIRST_ENDPOINT(&el, eid), EDGE_SECOND_ENDPOINT(&el, eid));
                        if (!tup) {
                            free(buf5);
                            free_adjacency(&al); free_edges(&el);
                            Py_DECREF(result);
                            return NULL;
                        }
                        PyList_Append(result, tup);
                        Py_DECREF(tup);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5);
    free_adjacency(&al); free_edges(&el);
    return result;
}

/* ── Articulation points (iterative Tarjan's) ── */

PyObject *py_articulation_points(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }

    AdjacencyList al;
    if (build_adjacency(n, &el, &al, 0) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }
    free_edges(&el);

    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) {
        free_adjacency(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf6, *low = buf6 + n, *stk = buf6 + 2*n;
    int *sidx = buf6 + 3*n, *peid = buf6 + 4*n, *children = buf6 + 5*n;
    memset(disc, -1, (size_t)n * sizeof(int));
    memset(children, 0, (size_t)n * sizeof(int));

    int timer = 0;
    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;
        children[start] = 0;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i];
                int eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    children[v] = 0;
                    children[u]++;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else {
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    /* Non-root AP: low[child] >= disc[parent] */
                    if (peid[p] != -1 && low[u] >= disc[p]) {
                        PyObject *val = PyLong_FromLong(p);
                        PySet_Add(result, val);
                        Py_DECREF(val);
                    }
                }
                sp--;
            }
        }

        /* Root AP: 2+ DFS tree children */
        if (children[start] >= 2) {
            PyObject *val = PyLong_FromLong(start);
            PySet_Add(result, val);
            Py_DECREF(val);
        }
    }

    free(buf6);
    free_adjacency(&al);
    return result;
}

/* ── Biconnected components (iterative Tarjan's with edge stack) ── */

PyObject *py_biconnected_components(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }
    if (el.edge_count == 0) { free_edges(&el); return result; }

    Py_ssize_t m = el.edge_count;
    AdjacencyList al;
    if (build_adjacency(n, &el, &al, 0) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }
    free_edges(&el);

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *esbuf = (int *)malloc(2 * (size_t)m * sizeof(int));
    if (!buf5 || !esbuf) {
        free(buf5); free(esbuf);
        free_adjacency(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    int *esu = esbuf, *esv = esbuf + m;
    memset(disc, -1, (size_t)n * sizeof(int));

    int timer = 0;
    int esp = 0;  /* edge stack pointer */

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i];
                int eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    /* Tree edge */
                    esu[esp] = u; esv[esp] = v; esp++;
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else if (disc[v] < disc[u]) {
                    /* Back edge to ancestor */
                    esu[esp] = u; esv[esp] = v; esp++;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        /* Pop edges to form biconnected component */
                        PyObject *comp = PySet_New(NULL);
                        if (!comp) {
                            free(buf5); free(esbuf);
                            free_adjacency(&al); Py_DECREF(result);
                            return NULL;
                        }
                        while (esp > 0) {
                            esp--;
                            PyObject *pu = PyLong_FromLong(esu[esp]);
                            PyObject *pv = PyLong_FromLong(esv[esp]);
                            PySet_Add(comp, pu);
                            PySet_Add(comp, pv);
                            Py_DECREF(pu);
                            Py_DECREF(pv);
                            if (esu[esp] == p && esv[esp] == u) break;
                        }
                        PyList_Append(result, comp);
                        Py_DECREF(comp);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5); free(esbuf);
    free_adjacency(&al);
    return result;
}

/* ── bridges (node-ID edges) ── */

PyObject *py_bridges_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NodeIdContext ctx;
    if (node_id_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.node_count;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || ctx.edges.edge_count == 0) { node_id_context_free(&ctx); return result; }

    AdjacencyList al;
    if (build_adjacency(n, &ctx.edges, &al, 0) < 0) { node_id_context_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) {
        free_adjacency(&al); node_id_context_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i], eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        int ei = peid[u];
                        int su = EDGE_FIRST_ENDPOINT(&ctx.edges, ei), sv = EDGE_SECOND_ENDPOINT(&ctx.edges, ei);
                        PyObject *t = PyTuple_New(2);
                        Py_INCREF(ctx.node_id_items[su]); Py_INCREF(ctx.node_id_items[sv]);
                        PyTuple_SET_ITEM(t, 0, ctx.node_id_items[su]);
                        PyTuple_SET_ITEM(t, 1, ctx.node_id_items[sv]);
                        PyList_Append(result, t); Py_DECREF(t);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5);
    free_adjacency(&al); node_id_context_free(&ctx);
    return result;
}

/* ── articulation_points (node-ID edges) ── */

PyObject *py_ap_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NodeIdContext ctx;
    if (node_id_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.node_count;

    PyObject *result = PySet_New(NULL);
    if (!result || n == 0) { node_id_context_free(&ctx); return result; }

    AdjacencyList al;
    if (build_adjacency(n, &ctx.edges, &al, 0) < 0) { node_id_context_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) {
        free_adjacency(&al); node_id_context_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf6, *low = buf6 + n, *stk = buf6 + 2*n;
    int *sidx = buf6 + 3*n, *peid = buf6 + 4*n, *ch = buf6 + 5*n;
    memset(disc, -1, n * sizeof(int));
    memset(ch, 0, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1; ch[start] = 0;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i], eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid; ch[v] = 0; ch[u]++;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (peid[p] != -1 && low[u] >= disc[p])
                        PySet_Add(result, ctx.node_id_items[p]);
                }
                sp--;
            }
        }
        if (ch[start] >= 2)
            PySet_Add(result, ctx.node_id_items[start]);
    }

    free(buf6);
    free_adjacency(&al); node_id_context_free(&ctx);
    return result;
}

/* ── biconnected_components (node-ID edges) ── */

PyObject *py_bcc_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NodeIdContext ctx;
    if (node_id_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.node_count;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || ctx.edges.edge_count == 0) { node_id_context_free(&ctx); return result; }

    Py_ssize_t m = ctx.edges.edge_count;
    AdjacencyList al;
    if (build_adjacency(n, &ctx.edges, &al, 0) < 0) { node_id_context_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *esbuf = (int *)malloc(2 * (size_t)m * sizeof(int));
    if (!buf5 || !esbuf) {
        free(buf5); free(esbuf);
        free_adjacency(&al); node_id_context_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    int *esu = esbuf, *esv = esbuf + m;
    memset(disc, -1, n * sizeof(int));
    int timer = 0, esp = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.neighbors[i], eid = al.edge_indices[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else if (disc[v] < disc[u]) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        PyObject *comp = PySet_New(NULL);
                        while (esp > 0) {
                            esp--;
                            PySet_Add(comp, ctx.node_id_items[esu[esp]]);
                            PySet_Add(comp, ctx.node_id_items[esv[esp]]);
                            if (esu[esp] == p && esv[esp] == u) break;
                        }
                        PyList_Append(result, comp); Py_DECREF(comp);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5); free(esbuf);
    free_adjacency(&al); node_id_context_free(&ctx);
    return result;
}

PyObject *py_bridges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    int n = g->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || g->node_ids.edges.edge_count == 0 || !g->has_adjacency) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjacencyList *al = &g->adjacency;
    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (nmask && nmask[start]) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al->offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al->neighbors[i], eid = al->edge_indices[i];
                if (mask && mask[eid]) continue;
                if (nmask && nmask[v]) continue;
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al->offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        int ei = peid[u];
                        int su = EDGE_FIRST_ENDPOINT(&g->node_ids.edges, ei), sv = EDGE_SECOND_ENDPOINT(&g->node_ids.edges, ei);
                        PyObject *t = PyTuple_New(2);
                        Py_INCREF(g->node_ids.node_id_items[su]); Py_INCREF(g->node_ids.node_id_items[sv]);
                        PyTuple_SET_ITEM(t, 0, g->node_ids.node_id_items[su]);
                        PyTuple_SET_ITEM(t, 1, g->node_ids.node_id_items[sv]);
                        PyList_Append(result, t); Py_DECREF(t);
                    }
                }
                sp--;
            }
        }
    }
    free(buf5);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

PyObject *py_ap_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    int n = g->node_ids.node_count;

    PyObject *result = PySet_New(NULL);
    if (!result || n == 0 || !g->has_adjacency) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjacencyList *al = &g->adjacency;
    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *disc = buf6, *low = buf6 + n, *stk = buf6 + 2*n;
    int *sidx = buf6 + 3*n, *peid = buf6 + 4*n, *ch = buf6 + 5*n;
    memset(disc, -1, n * sizeof(int));
    memset(ch, 0, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (nmask && nmask[start]) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1; ch[start] = 0;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al->offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al->neighbors[i], eid = al->edge_indices[i];
                if (mask && mask[eid]) continue;
                if (nmask && nmask[v]) continue;
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid; ch[v] = 0; ch[u]++;
                    stk[++sp] = v; sidx[sp] = al->offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (peid[p] != -1 && low[u] >= disc[p])
                        PySet_Add(result, g->node_ids.node_id_items[p]);
                }
                sp--;
            }
        }
        if (ch[start] >= 2)
            PySet_Add(result, g->node_ids.node_id_items[start]);
    }
    free(buf6);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

PyObject *py_bcc_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    int n = g->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || g->node_ids.edges.edge_count == 0 || !g->has_adjacency) return result;

    Py_ssize_t m = g->node_ids.edges.edge_count;
    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, m, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjacencyList *al = &g->adjacency;
    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *esbuf = (int *)malloc(2 * (size_t)m * sizeof(int));
    if (!buf5 || !esbuf) {
        free(buf5); free(esbuf); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    int *esu = esbuf, *esv = esbuf + m;
    memset(disc, -1, n * sizeof(int));
    int timer = 0, esp = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (nmask && nmask[start]) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al->offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al->neighbors[i], eid = al->edge_indices[i];
                if (mask && mask[eid]) continue;
                if (nmask && nmask[v]) continue;
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al->offset[v];
                } else if (disc[v] < disc[u]) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        PyObject *comp = PySet_New(NULL);
                        while (esp > 0) {
                            esp--;
                            PySet_Add(comp, g->node_ids.node_id_items[esu[esp]]);
                            PySet_Add(comp, g->node_ids.node_id_items[esv[esp]]);
                            if (esu[esp] == p && esv[esp] == u) break;
                        }
                        PyList_Append(result, comp); Py_DECREF(comp);
                    }
                }
                sp--;
            }
        }
    }
    free(buf5); free(esbuf);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* bridges_as_edge_indices_ctx: return bridge edge indices for use in masking. */
PyObject *py_bridges_as_edge_indices_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    int n = g->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || g->node_ids.edges.edge_count == 0 || !g->has_adjacency) return result;

    if (g->directed) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError, "bridges is not defined for directed graphs");
        return NULL;
    }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjacencyList *al = &g->adjacency;
    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (nmask && nmask[start]) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al->offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al->neighbors[i], eid = al->edge_indices[i];
                if (mask && mask[eid]) continue;
                if (nmask && nmask[v]) continue;
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al->offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        PyObject *val = PyLong_FromLong(peid[u]);
                        PyList_Append(result, val);
                        Py_DECREF(val);
                    }
                }
                sp--;
            }
        }
    }
    free(buf5);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}
