/* Breadth-first search and edge-disjoint path enumeration. */
#include "entry_points.h"
#include "../helpers/parse.h"

/* ── BFS (unweighted, from source) ── */

PyObject *py_bfs(PyObject *self, PyObject *args) {
    int n, source;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iOi", &n, &edges_obj, &source)) return NULL;

    if (source < 0 || source >= n) {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return NULL;
    }

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    AdjList al;
    if (build_adj(n, &el, &al, 0) < 0) { free_edges(&el); return NULL; }
    free_edges(&el);

    int *visited = (int *)calloc((size_t)n, sizeof(int));
    int *queue   = (int *)malloc((size_t)n * sizeof(int));
    if (!visited || !queue) {
        free(visited); free(queue);
        free_adj(&al);
        PyErr_NoMemory();
        return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source;
    visited[source] = 1;

    while (head < tail) {
        int u = queue[head++];
        for (int i = al.offset[u]; i < al.offset[u + 1]; i++) {
            int v = al.adj[i];
            if (!visited[v]) {
                visited[v] = 1;
                queue[tail++] = v;
            }
        }
    }

    /* Build result list from queue (BFS order) */
    PyObject *result = PyList_New(tail);
    if (!result) {
        free(visited); free(queue); free_adj(&al);
        return NULL;
    }
    for (int i = 0; i < tail; i++)
        PyList_SET_ITEM(result, i, PyLong_FromLong(queue[i]));

    free(visited); free(queue);
    free_adj(&al);
    return result;
}

/* ── bfs (node-ID edges) ── */

PyObject *py_bfs_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges, *src_obj;
    if (!PyArg_ParseTuple(args, "OOO", &nids, &edges, &src_obj)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    long src_nid = PyLong_AsLong(src_obj);
    if (src_nid == -1 && PyErr_Occurred()) { nid_free(&ctx); return NULL; }
    int source = intmap_get(&ctx.im, src_nid);
    if (source < 0) {
        nid_free(&ctx);
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    AdjList al;
    if (build_adj(n, &ctx.el, &al, 0) < 0) { nid_free(&ctx); return NULL; }

    int *vis = calloc(n, sizeof(int)), *queue = malloc(n * sizeof(int));
    if (!vis || !queue) {
        free(vis); free(queue); free_adj(&al); nid_free(&ctx);
        PyErr_NoMemory(); return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source; vis[source] = 1;
    while (head < tail) {
        int u = queue[head++];
        for (int i = al.offset[u]; i < al.offset[u+1]; i++) {
            int v = al.adj[i];
            if (!vis[v]) { vis[v] = 1; queue[tail++] = v; }
        }
    }

    PyObject *result = PyList_New(tail);
    if (result) {
        for (int i = 0; i < tail; i++) {
            Py_INCREF(ctx.nid_items[queue[i]]);
            PyList_SET_ITEM(result, i, ctx.nid_items[queue[i]]);
        }
    }

    free(vis); free(queue); free_adj(&al); nid_free(&ctx);
    return result;
}

PyObject *py_bfs_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *src_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &src_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    long src_nid = PyLong_AsLong(src_obj);
    if (src_nid == -1 && PyErr_Occurred()) return NULL;
    int source = intmap_get(&g->nid.im, src_nid);
    if (source < 0) {
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) return NULL;
    if (nmask && nmask[source]) {
        release_mask(&nmbuf);
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        return NULL;
    }

    if (!g->has_adj) { release_mask(&nmbuf); return PyList_New(0); }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) { release_mask(&nmbuf); return NULL; }

    AdjList *al = &g->al;
    int *vis = calloc(n, sizeof(int)), *queue = malloc(n * sizeof(int));
    if (!vis || !queue) { free(vis); free(queue); release_mask(&mbuf); release_mask(&nmbuf); PyErr_NoMemory(); return NULL; }

    int head = 0, tail = 0;
    queue[tail++] = source; vis[source] = 1;
    while (head < tail) {
        int u = queue[head++];
        for (int i = al->offset[u]; i < al->offset[u+1]; i++) {
            if (mask && mask[al->eid[i]]) continue;
            int v = al->adj[i];
            if (nmask && nmask[v]) continue;
            if (!vis[v]) { vis[v] = 1; queue[tail++] = v; }
        }
    }

    PyObject *result = PyList_New(tail);
    if (result) {
        for (int i = 0; i < tail; i++) {
            Py_INCREF(g->nid.nid_items[queue[i]]);
            PyList_SET_ITEM(result, i, g->nid.nid_items[queue[i]]);
        }
    }
    free(vis); free(queue);
    release_mask(&mbuf); release_mask(&nmbuf);
    return result;
}

/* ── All-edge-paths (stack-based DFS, edge-disjoint) ── */

PyObject *py_all_edge_paths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *src_obj, *targets_obj;
    int cutoff = -1;
    PyObject *mask_obj = Py_None, *nmask_obj = Py_None;
    int node_simple = 0;
    int ignore_self_loops = 0;
    if (!PyArg_ParseTuple(args, "OOO|iOOpp", &capsule, &src_obj, &targets_obj,
                          &cutoff, &mask_obj, &nmask_obj, &node_simple, &ignore_self_loops))
        return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    Py_ssize_t m = g->nid.el.m;

    /* Parse source */
    long src_nid = PyLong_AsLong(src_obj);
    if (src_nid == -1 && PyErr_Occurred()) return NULL;
    int source = n > 0 ? intmap_get(&g->nid.im, src_nid) : -1;
    if (source < 0) {
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    /* Parse targets */
    PyObject *tgt_fast = PySequence_Fast(targets_obj, "targets must be a sequence");
    if (!tgt_fast) return NULL;
    Py_ssize_t ntgt = PySequence_Fast_GET_SIZE(tgt_fast);
    uint8_t *is_target = (uint8_t *)calloc(n, sizeof(uint8_t));
    if (!is_target) { Py_DECREF(tgt_fast); PyErr_NoMemory(); return NULL; }

    PyObject **tgt_items = PySequence_Fast_ITEMS(tgt_fast);
    for (Py_ssize_t i = 0; i < ntgt; i++) {
        long tnid = PyLong_AsLong(tgt_items[i]);
        if (tnid == -1 && PyErr_Occurred()) {
            free(is_target); Py_DECREF(tgt_fast); return NULL;
        }
        int tidx = intmap_get(&g->nid.im, tnid);
        if (tidx >= 0) is_target[tidx] = 1;
    }
    Py_DECREF(tgt_fast);

    /* Parse masks */
    const uint8_t *emask; Py_buffer embuf;
    if (parse_mask(mask_obj, m, &emask, &embuf) < 0) { free(is_target); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&embuf); free(is_target); return NULL; }

    if (nmask && nmask[source]) {
        release_mask(&nmbuf); release_mask(&embuf); free(is_target);
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        return NULL;
    }

    PyObject *result = PyList_New(0);
    if (!result || !g->has_adj || m == 0) {
        if (!result) { release_mask(&nmbuf); release_mask(&embuf); free(is_target); return NULL; }
        release_mask(&nmbuf); release_mask(&embuf); free(is_target);
        return result;
    }

    AdjList *al = &g->al;
    int max_depth = (cutoff > 0) ? cutoff : (int)(2 * m);  /* safety bound */

    /* Allocate DFS state */
    uint8_t *visited_edges = (uint8_t *)calloc(m, sizeof(uint8_t));
    uint8_t *visited_nodes = node_simple ? (uint8_t *)calloc(n, sizeof(uint8_t)) : NULL;
    int *stk_node = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *stk_pos  = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *path_eids = (int *)malloc((size_t)max_depth * sizeof(int));
    if (!visited_edges || !stk_node || !stk_pos || !path_eids ||
        (node_simple && !visited_nodes)) {
        free(visited_edges); free(visited_nodes);
        free(stk_node); free(stk_pos); free(path_eids);
        release_mask(&nmbuf); release_mask(&embuf); free(is_target);
        Py_DECREF(result); PyErr_NoMemory(); return NULL;
    }
    if (visited_nodes) visited_nodes[source] = 1;  /* source counts as visited */

    /* DFS */
    int sp = 0;
    stk_node[0] = source;
    stk_pos[0] = al->offset[source];

    while (sp >= 0) {
        int u = stk_node[sp];
        if (stk_pos[sp] < al->offset[u + 1]) {
            int idx = stk_pos[sp]++;
            int eid = al->eid[idx];
            int v = al->adj[idx];

            /* Self-loops: optionally never traversed. Otherwise an undirected
             * self-loop occupies two consecutive CSR slots and is visited
             * through the first slot only. */
            if (v == u && ignore_self_loops) continue;
            if (v == u && idx > al->offset[u] && al->eid[idx - 1] == eid) continue;

            /* Skip masked/visited edges */
            if (visited_edges[eid]) continue;
            if (emask && emask[eid]) continue;
            if (nmask && nmask[v]) continue;

            /* Mark edge as visited, record in path */
            visited_edges[eid] = 1;
            path_eids[sp] = eid;

            /* Check if v is a target BEFORE checking node_simple.
             * This matches the Python all_edge_paths_multigraph which yields
             * the path before checking if the node can be entered. This is
             * critical for self-loops: source -> self-loop -> source(=target)
             * should yield even though source is already visited. */
            if (is_target[v]) {
                /* Build path list */
                PyObject *path = PyList_New(sp + 1);
                if (!path) {
                    visited_edges[eid] = 0;
                    goto error;
                }
                for (int j = 0; j <= sp; j++)
                    PyList_SET_ITEM(path, j, PyLong_FromLong(path_eids[j]));
                if (PyList_Append(result, path) < 0) {
                    Py_DECREF(path);
                    visited_edges[eid] = 0;
                    goto error;
                }
                Py_DECREF(path);
            }

            /* Check node_simple constraint: skip entering already-visited nodes */
            if (visited_nodes && visited_nodes[v]) {
                visited_edges[eid] = 0;
                continue;
            }

            /* Mark node as visited (only when actually entering) */
            if (visited_nodes) visited_nodes[v] = 1;

            /* Push v if depth allows */
            if (sp + 1 < max_depth) {
                sp++;
                stk_node[sp] = v;
                stk_pos[sp] = al->offset[v];
            } else {
                /* At max depth, don't descend, unmark edge and node */
                visited_edges[eid] = 0;
                if (visited_nodes) visited_nodes[v] = 0;
            }
        } else {
            /* Pop: unmark the edge that brought us here and the node */
            if (sp > 0 && visited_nodes) {
                visited_nodes[stk_node[sp]] = 0;
            }
            sp--;
            if (sp >= 0) {
                visited_edges[path_eids[sp]] = 0;
            }
        }
    }

    free(visited_edges); free(visited_nodes);
    free(stk_node); free(stk_pos); free(path_eids);
    release_mask(&nmbuf); release_mask(&embuf); free(is_target);
    return result;

error:
    free(visited_edges); free(visited_nodes);
    free(stk_node); free(stk_pos); free(path_eids);
    release_mask(&nmbuf); release_mask(&embuf); free(is_target);
    Py_DECREF(result);
    return NULL;
}
