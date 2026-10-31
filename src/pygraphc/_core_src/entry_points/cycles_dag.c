/* Directed acyclic graphs and cycles: topological sort, longest path, cycle basis. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/toposort.h"

PyObject *py_toposort_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    if (n == 0) {
        nid_free(&ctx);
        return PyList_New(0);
    }

    int *order;
    int count = toposort_core(n, &ctx.el, &order);
    if (count < 0) { nid_free(&ctx); return NULL; }

    if (count < n) {
        free(order); nid_free(&ctx);
        PyErr_Format(PyExc_ValueError,
            "Graph contains a cycle - topological sort is undefined "
            "(%d of %d nodes processed)", count, n);
        return NULL;
    }

    PyObject *result = PyList_New(n);
    if (result) {
        for (int i = 0; i < n; i++) {
            Py_INCREF(ctx.nid_items[order[i]]);
            PyList_SET_ITEM(result, i, ctx.nid_items[order[i]]);
        }
    }

    free(order); nid_free(&ctx);
    return result;
}

PyObject *py_toposort_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError,
            "topological_sort requires a directed graph");
        return NULL;
    }
    if (n == 0) return PyList_New(0);

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    /* Count active nodes */
    int active_n = n;
    if (nmask) { active_n = 0; for (int i = 0; i < n; i++) if (!nmask[i]) active_n++; }

    int *order;
    int count = toposort_core_masked(n, &g->nid.el, mask, nmask, &order);
    release_mask(&nmbuf); release_mask(&mbuf);
    if (count < 0) return NULL;

    if (count < active_n) {
        free(order);
        PyErr_Format(PyExc_ValueError,
            "Graph contains a cycle - topological sort is undefined "
            "(%d of %d nodes processed)", count, active_n);
        return NULL;
    }

    PyObject *result = PyList_New(count);
    if (result) {
        for (int i = 0; i < count; i++) {
            Py_INCREF(g->nid.nid_items[order[i]]);
            PyList_SET_ITEM(result, i, g->nid.nid_items[order[i]]);
        }
    }
    free(order);
    return result;
}

/* ── Cycle basis (fundamental cycles via DFS spanning tree) ── */

/*
 * Find a fundamental cycle basis of an undirected graph.
 * For each back edge in the DFS tree, traces the cycle through the tree.
 * Returns a Python list of lists of original node IDs.
 * Supports edge mask and node mask.
 */
PyObject *py_cycle_basis_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (n == 0 || !g->has_adj) return result;

    if (g->directed) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError,
            "cycle_basis is not defined for directed graphs");
        return NULL;
    }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjList *al = &g->al;
    EdgeList *el = &g->nid.el;

    /* Self-loops are trivial cycles */
    for (Py_ssize_t i = 0; i < el->m; i++) {
        if (mask && mask[i]) continue;
        int u = EDGE_SRC(el, i);
        if (nmask && nmask[u]) continue;
        if (u == EDGE_DST(el, i)) {
            PyObject *cycle = PyList_New(1);
            if (!cycle) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result); return NULL; }
            Py_INCREF(g->nid.nid_items[u]);
            PyList_SET_ITEM(cycle, 0, g->nid.nid_items[u]);
            PyList_Append(result, cycle);
            Py_DECREF(cycle);
        }
    }

    /* Track which non-tree edges have been processed (one bit per edge) */
    Py_ssize_t m = el->m;
    uint8_t *edge_seen = (uint8_t *)calloc(m > 0 ? (size_t)m : 1, 1);
    if (!edge_seen) {
        release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }

    /* Allocate: parent[n], depth[n], queue[n], peid[n], path_a[n], path_b[n] */
    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) {
        free(edge_seen);
        release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *parent = buf6;
    int *depth  = buf6 + n;
    int *queue  = buf6 + 2 * n;
    int *peid   = buf6 + 3 * n;  /* parent edge ID (to skip reverse tree edge) */
    int *path_a = buf6 + 4 * n;
    int *path_b = buf6 + 5 * n;
    memset(depth, -1, (size_t)n * sizeof(int));

    /* BFS spanning tree + non-tree edge cycle extraction.
     * BFS trees are shallow (O(log n) for random graphs), making the
     * LCA-based cycle trace O(log n) per back edge instead of O(n). */
    for (int start = 0; start < n; start++) {
        if (depth[start] != -1) continue;
        if (nmask && nmask[start]) continue;

        /* BFS to build spanning tree */
        int head = 0, tail = 0;
        queue[tail++] = start;
        depth[start] = 0;
        parent[start] = -1;
        peid[start] = -1;

        while (head < tail) {
            int u = queue[head++];
            for (int i = al->offset[u]; i < al->offset[u + 1]; i++) {
                int eid = al->eid[i];
                if (mask && mask[eid]) continue;
                int v = al->adj[i];
                if (nmask && nmask[v]) continue;
                if (v == u) continue;  /* skip self-loops (handled above) */
                if (depth[v] == -1) {
                    /* Tree edge */
                    depth[v] = depth[u] + 1;
                    parent[v] = u;
                    peid[v] = eid;
                    queue[tail++] = v;
                } else if (eid != peid[u] && !edge_seen[eid]) {
                    /* Non-tree edge: extract cycle via LCA.
                     * edge_seen prevents processing each undirected
                     * edge twice (CSR has both u->v and v->u). */
                    edge_seen[eid] = 1;
                    int a = u, b = v;
                    int la = 0, lb = 0;

                    /* Walk a and b up to the same depth */
                    while (depth[a] > depth[b]) { path_a[la++] = a; a = parent[a]; }
                    while (depth[b] > depth[a]) { path_b[lb++] = b; b = parent[b]; }
                    /* Walk both up to LCA */
                    while (a != b) {
                        path_a[la++] = a; a = parent[a];
                        path_b[lb++] = b; b = parent[b];
                    }
                    /* a == b == LCA */
                    int lca = a;
                    int plen = la + lb + 1;

                    PyObject *cycle = PyList_New(plen);
                    if (!cycle) {
                        free(buf6); free(edge_seen);
                        release_mask(&nmbuf); release_mask(&mbuf);
                        Py_DECREF(result); return NULL;
                    }
                    int idx = 0;
                    for (int j = 0; j < la; j++) {
                        Py_INCREF(g->nid.nid_items[path_a[j]]);
                        PyList_SET_ITEM(cycle, idx++, g->nid.nid_items[path_a[j]]);
                    }
                    Py_INCREF(g->nid.nid_items[lca]);
                    PyList_SET_ITEM(cycle, idx++, g->nid.nid_items[lca]);
                    for (int j = lb - 1; j >= 0; j--) {
                        Py_INCREF(g->nid.nid_items[path_b[j]]);
                        PyList_SET_ITEM(cycle, idx++, g->nid.nid_items[path_b[j]]);
                    }
                    PyList_Append(result, cycle);
                    Py_DECREF(cycle);
                }
            }
        }
    }

    free(buf6); free(edge_seen);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* ── DAG longest path (topological sort + DP) ── */

/*
 * Find the longest path in a weighted or unweighted DAG.
 * Uses topological sort + dynamic programming.
 * weights: if NULL, all edges have weight 1.
 * Returns Python list of original node IDs forming the longest path.
 */
PyObject *py_dag_longest_path_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *wobj = Py_None, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &wobj, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError,
            "dag_longest_path requires a directed graph");
        return NULL;
    }
    if (n == 0) return PyList_New(0);

    /* Parse optional weights */
    WeightList wl = {NULL, 0, 0};
    int has_weights = (wobj != Py_None);
    if (has_weights) {
        if (parse_weights(wobj, &wl, g->nid.el.m) < 0) return NULL;
    }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) {
        if (has_weights) free_weights(&wl); return NULL;
    }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) {
        release_mask(&mbuf); if (has_weights) free_weights(&wl); return NULL;
    }

    /* Get topological order (reuse masked toposort) */
    int *order;
    int active_n = n;
    if (nmask) { active_n = 0; for (int i = 0; i < n; i++) if (!nmask[i]) active_n++; }

    int count = toposort_core_masked(n, &g->nid.el, mask, nmask, &order);
    if (count < 0) {
        release_mask(&nmbuf); release_mask(&mbuf);
        if (has_weights) free_weights(&wl);
        return NULL;
    }
    if (count < active_n) {
        free(order);
        release_mask(&nmbuf); release_mask(&mbuf);
        if (has_weights) free_weights(&wl);
        PyErr_SetString(PyExc_ValueError,
            "Graph contains a cycle - dag_longest_path is undefined");
        return NULL;
    }

    /* DP: dist[v] = longest distance to reach v, prev[v] = predecessor on longest path */
    double *dist = (double *)malloc((size_t)n * sizeof(double));
    int *prev = (int *)malloc((size_t)n * sizeof(int));
    if (!dist || !prev) {
        free(dist); free(prev); free(order);
        release_mask(&nmbuf); release_mask(&mbuf);
        if (has_weights) free_weights(&wl);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i < n; i++) { dist[i] = 0.0; prev[i] = -1; }

    /* Use forward CSR to iterate outgoing edges */
    AdjList *al = &g->al;
    int use_csr = g->has_adj;

    /* Process nodes in topological order */
    for (int ti = 0; ti < count; ti++) {
        int u = order[ti];
        if (use_csr) {
            for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
                int eid = al->eid[e];
                if (mask && mask[eid]) continue;
                int v = al->adj[e];
                if (nmask && nmask[v]) continue;
                double w = has_weights ? wl.w[eid] : 1.0;
                if (dist[u] + w > dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                }
            }
        } else {
            /* No CSR (no edges) — nothing to relax */
        }
    }

    /* Find node with maximum distance */
    int best = -1;
    double best_dist = -1.0;
    for (int ti = 0; ti < count; ti++) {
        int u = order[ti];
        if (dist[u] > best_dist) {
            best_dist = dist[u];
            best = u;
        }
    }

    /* Trace back the path */
    if (best < 0 || best_dist == 0.0) {
        /* No edges or all isolated: return first active node */
        PyObject *res;
        if (count > 0) {
            res = PyList_New(1);
            if (res) {
                Py_INCREF(g->nid.nid_items[order[0]]);
                PyList_SET_ITEM(res, 0, g->nid.nid_items[order[0]]);
            }
        } else {
            res = PyList_New(0);
        }
        free(dist); free(prev); free(order);
        release_mask(&nmbuf); release_mask(&mbuf);
        if (has_weights) free_weights(&wl);
        return res;
    }

    /* Count path length by tracing back */
    int path_len = 0;
    for (int v = best; v >= 0; v = prev[v]) path_len++;

    PyObject *result = PyList_New(path_len);
    if (!result) {
        free(dist); free(prev); free(order);
        release_mask(&nmbuf); release_mask(&mbuf);
        if (has_weights) free_weights(&wl);
        return NULL;
    }
    /* Fill in reverse */
    int idx = path_len - 1;
    for (int v = best; v >= 0; v = prev[v]) {
        Py_INCREF(g->nid.nid_items[v]);
        PyList_SET_ITEM(result, idx--, g->nid.nid_items[v]);
    }

    free(dist); free(prev); free(order);
    release_mask(&nmbuf); release_mask(&mbuf);
    if (has_weights) free_weights(&wl);
    return result;
}
