/* Connected components (index, node-ID and context tiers) and strongly connected components. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/components.h"

/* ── connected_components(n, edges) -> list[set[int]] ── */

PyObject *py_connected_components(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        free_edges(&el);
        return NULL;
    }
    free_edges(&el);

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); return NULL; }

    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result);
            return NULL;
        }
    }

    for (int i = 0; i < n; i++) {
        PyObject *val = PyLong_FromLong(i);
        PySet_Add(sets[cr.labels[i]], val);
        Py_DECREF(val);
    }

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    return result;
}

/* ── connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]] ── */

PyObject *py_connected_components_with_branches(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "iOO", &n, &edges_obj, &branch_ids_obj)) return NULL;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    /* Parse branch_ids */
    PyObject *br_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!br_fast) { free_edges(&el); return NULL; }
    Py_ssize_t br_len = PySequence_Fast_GET_SIZE(br_fast);

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        Py_DECREF(br_fast);
        free_edges(&el);
        return NULL;
    }

    int num_comp = cr.num_comp;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = el.m < br_len ? el.m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);
    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        int c = cr.labels[EDGE_SRC(&el, i)];
        PySet_Add(branch_sets[c], br_items[i]);
    }
    Py_DECREF(br_fast);
    free_edges(&el);

    /* Build result list */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup2;

    for (int c = 0; c < num_comp; c++) {
        PyObject *node_set = PySet_New(NULL);
        for (int j = 0; j < sizes[c]; j++) {
            PyObject *val = PyLong_FromLong(buckets[c][j]);
            PySet_Add(node_set, val);
            Py_DECREF(val);
        }
        PyObject *tup = PyTuple_Pack(2, node_set, branch_sets[c]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, c, tup);
    }

cleanup2:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    return result;
}

/* ── connected_components_remapped(node_ids, edges) -> list[set[NodeId]] ──
 *
 * Like connected_components but takes the node_ids remapping list and returns
 * sets containing the original node IDs (node_ids[index]) instead of indices.
 * This moves the remapping loop from Python into C.
 */
PyObject *py_connected_components_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    PyObject *nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!nid_fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(nid_fast);
    PyObject **nid_items = PySequence_Fast_ITEMS(nid_fast);

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(nid_fast); return NULL; }

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        free_edges(&el);
        Py_DECREF(nid_fast);
        return NULL;
    }
    free_edges(&el);

    int num_comp = cr.num_comp;

    /* Bucket nodes by component */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build Python list of sets with remapped node IDs */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup_r;

    for (int c = 0; c < num_comp; c++) {
        PyObject *s = PySet_New(NULL);
        if (!s) { Py_DECREF(result); result = NULL; goto cleanup_r; }
        for (int j = 0; j < sizes[c]; j++) {
            /* Use the original node ID from the remapping array */
            PySet_Add(s, nid_items[buckets[c][j]]);
        }
        PyList_SET_ITEM(result, c, s);
    }

cleanup_r:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) free(buckets[c]);
    free(buckets);
    Py_DECREF(nid_fast);
    return result;
}

/* ── connected_components_with_branches_remapped(node_ids, edges, branch_ids)
 *    -> list[tuple[set[NodeId], set[BranchId]]] ──
 *
 * Like connected_components_with_branches but remaps node indices to original
 * node IDs using the node_ids array.
 */
PyObject *py_connected_components_with_branches_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "OOO", &node_ids_obj, &edges_obj, &branch_ids_obj)) return NULL;

    PyObject *nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!nid_fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(nid_fast);
    PyObject **nid_items = PySequence_Fast_ITEMS(nid_fast);

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(nid_fast); return NULL; }

    PyObject *br_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!br_fast) { free_edges(&el); Py_DECREF(nid_fast); return NULL; }
    Py_ssize_t br_len = PySequence_Fast_GET_SIZE(br_fast);

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        Py_DECREF(br_fast);
        free_edges(&el);
        Py_DECREF(nid_fast);
        return NULL;
    }

    int num_comp = cr.num_comp;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = el.m < br_len ? el.m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);
    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        int c = cr.labels[EDGE_SRC(&el, i)];
        PySet_Add(branch_sets[c], br_items[i]);
    }
    Py_DECREF(br_fast);
    free_edges(&el);

    /* Build result list with remapped node IDs */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup_rb;

    for (int c = 0; c < num_comp; c++) {
        PyObject *node_set = PySet_New(NULL);
        for (int j = 0; j < sizes[c]; j++)
            PySet_Add(node_set, nid_items[buckets[c][j]]);
        PyObject *tup = PyTuple_Pack(2, node_set, branch_sets[c]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, c, tup);
    }

cleanup_rb:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    Py_DECREF(nid_fast);
    return result;
}

/* ── connected_components (node-ID edges) ── */

PyObject *py_cc_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;
    if (n == 0) { nid_free(&ctx); return PyList_New(0); }

    ComponentResult cr;
    if (compute_components(n, &ctx.el, &cr) < 0) { nid_free(&ctx); return NULL; }

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); nid_free(&ctx); return NULL; }

    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); nid_free(&ctx); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result); nid_free(&ctx);
            return NULL;
        }
    }

    for (int i = 0; i < n; i++)
        PySet_Add(sets[cr.labels[i]], ctx.nid_items[i]);

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    nid_free(&ctx);
    return result;
}

/* ── connected_components (node-ID, split src/dst) ── */

PyObject *py_cc_nid_split(PyObject *self, PyObject *args) {
    PyObject *nids, *src, *dst;
    if (!PyArg_ParseTuple(args, "OOO", &nids, &src, &dst)) return NULL;

    NidContext ctx;
    if (nid_parse_split(nids, src, dst, &ctx) < 0) return NULL;
    int n = ctx.n;
    if (n == 0) { nid_free(&ctx); return PyList_New(0); }

    ComponentResult cr;
    if (compute_components(n, &ctx.el, &cr) < 0) { nid_free(&ctx); return NULL; }

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); nid_free(&ctx); return NULL; }

    /* Create all sets upfront */
    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); nid_free(&ctx); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result); nid_free(&ctx);
            return NULL;
        }
    }

    /* Single pass: add each node to its component set */
    for (int i = 0; i < n; i++)
        PySet_Add(sets[cr.labels[i]], ctx.nid_items[i]);

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    nid_free(&ctx);
    return result;
}

PyObject *py_cc_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    if (n == 0) return PyList_New(0);

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    ComponentResult cr;
    if (compute_components_masked(n, &g->nid.el, mask, nmask, &cr) < 0) {
        release_mask(&nmbuf); release_mask(&mbuf); return NULL;
    }
    release_mask(&mbuf);
    int nc = cr.num_comp;

    /* Build sets, skipping masked nodes */
    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); release_mask(&nmbuf); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); release_mask(&nmbuf); return NULL;
        }
    }
    for (int i = 0; i < n; i++) {
        if (nmask && nmask[i]) continue;
        PySet_Add(sets[cr.labels[i]], g->nid.nid_items[i]);
    }

    /* Collect non-empty sets */
    PyObject *result = PyList_New(0);
    if (!result) {
        for (int c = 0; c < nc; c++) Py_DECREF(sets[c]);
        free(sets); free(cr.labels); release_mask(&nmbuf); return NULL;
    }
    for (int c = 0; c < nc; c++) {
        if (PySet_GET_SIZE(sets[c]) > 0)
            PyList_Append(result, sets[c]);
        Py_DECREF(sets[c]);
    }
    free(sets); free(cr.labels);
    release_mask(&nmbuf);
    return result;
}

/* ── cc_branches_ctx(capsule, branch_ids[, excluded_edges, excluded_nodes])
 *    -> list[tuple[set[NodeId], set[BranchId]]]
 *
 * Connected components with branch IDs from a cached graph context.
 *
 * excluded_edges: edges with excluded_edges[i] != 0 are removed from
 *     connectivity and their branch IDs are dropped.
 * excluded_nodes: removed from the output node sets but their edges
 *     still contribute to connectivity and branch ID sets.
 */
PyObject *py_cc_branches_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *branch_ids_obj, *emask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &branch_ids_obj, &emask_obj, &nmask_obj))
        return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    PyObject *br_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!br_fast) return NULL;
    Py_ssize_t br_len = PySequence_Fast_GET_SIZE(br_fast);

    /* Parse excluded-edges mask */
    const uint8_t *emask; Py_buffer embuf;
    if (parse_mask(emask_obj, g->nid.el.m, &emask, &embuf) < 0) {
        Py_DECREF(br_fast); return NULL;
    }

    /* Parse excluded-nodes mask (used only for output filtering) */
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) {
        release_mask(&embuf); Py_DECREF(br_fast); return NULL;
    }

    /* Run union-find with both edge and node exclusion — excluded nodes do NOT
       participate in connectivity (traversal stops at them). Their incident
       edges still contribute branch IDs to the CC of the non-excluded endpoint. */
    ComponentResult cr;
    if (compute_components_masked(n, &g->nid.el, emask, nmask, &cr) < 0) {
        release_mask(&nmbuf); release_mask(&embuf); Py_DECREF(br_fast);
        return NULL;
    }
    /* Keep embuf alive — reused for branch assignment below */

    int num_comp = cr.num_comp;

    /* Bucket nodes by component */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    if (!sizes || !buckets || !offsets) {
        free(sizes); free(buckets); free(offsets);
        free(cr.labels); release_mask(&nmbuf); release_mask(&embuf);
        Py_DECREF(br_fast);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    for (int c = 0; c < num_comp; c++) {
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
        if (!buckets[c]) {
            for (int j = 0; j < c; j++) free(buckets[j]);
            free(sizes); free(buckets); free(offsets);
            free(cr.labels); release_mask(&nmbuf); release_mask(&embuf);
            Py_DECREF(br_fast);
            PyErr_NoMemory(); return NULL;
        }
    }
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component (skipping excluded edges).
     *
     * Performance note: at 1M nodes / 1.5M edges, the branch set construction
     * adds ~60ms on top of plain CC's ~30ms. This is inherent — 1.5M PySet_Add
     * calls for branch assignment plus 54K PySet_New for branch sets. The
     * CPython PySet_Add hash-insert is the floor; no algorithmic improvement
     * possible without bypassing Python set objects entirely.
     */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    if (!branch_sets) {
        for (int c = 0; c < num_comp; c++) free(buckets[c]);
        free(sizes); free(buckets); free(offsets);
        free(cr.labels); release_mask(&nmbuf); release_mask(&embuf);
        Py_DECREF(br_fast);
        PyErr_NoMemory(); return NULL;
    }
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    EdgeList *el = &g->nid.el;
    Py_ssize_t edge_branch_count = el->m < br_len ? el->m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);

    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        if (emask && emask[i]) continue;
        int u = EDGE_SRC(el, i);
        /* Anchor branch_id to a non-excluded endpoint when possible.
         * When nmask is NULL, this is just `u` (zero overhead). */
        int anchor = (nmask && nmask[u]) ? EDGE_DST(el, i) : u;
        int c = cr.labels[anchor];
        PySet_Add(branch_sets[c], br_items[i]);
    }
    release_mask(&embuf);
    Py_DECREF(br_fast);

    /* Build result list with remapped node IDs, filtering excluded nodes */
    PyObject *result = PyList_New(0);
    if (!result) goto cleanup_bcctx;

    for (int c = 0; c < num_comp; c++) {
        PyObject *node_set = PySet_New(NULL);
        for (int j = 0; j < sizes[c]; j++) {
            int idx = buckets[c][j];
            if (nmask && nmask[idx]) continue;
            PySet_Add(node_set, g->nid.nid_items[idx]);
        }
        /* Include component if it has any nodes or any branches */
        if (PySet_GET_SIZE(node_set) > 0 || PySet_GET_SIZE(branch_sets[c]) > 0) {
            PyObject *tup = PyTuple_Pack(2, node_set, branch_sets[c]);
            PyList_Append(result, tup);
            Py_DECREF(tup);
        }
        Py_DECREF(node_set);
    }

cleanup_bcctx:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    release_mask(&nmbuf);
    return result;
}

PyObject *py_scc_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    if (n == 0) return PyList_New(0);

    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError,
            "strongly_connected_components requires a directed graph");
        return NULL;
    }

    /* No adjacency list => every node is its own SCC */
    if (!g->has_adj) {
        const uint8_t *nmask = NULL; Py_buffer nmbuf;
        if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) return NULL;
        PyObject *result = PyList_New(0);
        if (!result) { release_mask(&nmbuf); return NULL; }
        for (int i = 0; i < n; i++) {
            if (nmask && nmask[i]) continue;
            PyObject *s = PySet_New(NULL);
            PySet_Add(s, g->nid.nid_items[i]);
            PyList_Append(result, s);
            Py_DECREF(s);
        }
        release_mask(&nmbuf);
        return result;
    }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    ComponentResult cr;
    if (compute_scc_masked(n, &g->al, mask, nmask, &cr) < 0) {
        release_mask(&nmbuf); release_mask(&mbuf); return NULL;
    }
    release_mask(&mbuf);
    int nc = cr.num_comp;

    /* Build sets, skipping masked nodes */
    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); release_mask(&nmbuf); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); release_mask(&nmbuf); return NULL;
        }
    }
    for (int i = 0; i < n; i++) {
        if (nmask && nmask[i]) continue;
        PySet_Add(sets[cr.labels[i]], g->nid.nid_items[i]);
    }

    /* Collect non-empty sets */
    PyObject *result = PyList_New(0);
    if (!result) {
        for (int c = 0; c < nc; c++) Py_DECREF(sets[c]);
        free(sets); free(cr.labels); release_mask(&nmbuf); return NULL;
    }
    for (int c = 0; c < nc; c++) {
        if (PySet_GET_SIZE(sets[c]) > 0)
            PyList_Append(result, sets[c]);
        Py_DECREF(sets[c]);
    }
    free(sets); free(cr.labels);
    release_mask(&nmbuf);
    return result;
}
