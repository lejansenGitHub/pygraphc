#include "entry_points.h"
#include "../helpers/parse.h"

/* ── CSR-based query functions (O(degree) instead of O(m)) ── */

/* Helper: resolve a node ID to its internal index via the IntMap.
 * Returns -1 if not found. Sets *found to 0 if not found, 1 if found.
 * Does NOT set a Python exception on not-found (callers return empty). */
static int resolve_node(GraphContext *g, PyObject *nid_obj, int *found) {
    long nid = PyLong_AsLong(nid_obj);
    if (nid == -1 && PyErr_Occurred()) { *found = -1; return -1; }
    if (g->node_ids.node_count == 0 || !g->node_ids.node_index_by_node_id.keys) { *found = 0; return -1; }
    int idx = intmap_get(&g->node_ids.node_index_by_node_id, nid);
    *found = (idx >= 0) ? 1 : 0;
    return idx;
}

/* neighbors_ctx: O(degree) neighbor lookup via CSR. */
PyObject *py_neighbors_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;  /* conversion error */

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (!found || !g->has_adjacency) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return result; }

    AdjacencyList *al = &g->adjacency;
    for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
        int v = al->neighbors[e];
        if (v == u) continue;  /* skip self-loops */
        if (mask && mask[al->edge_indices[e]]) continue;
        if (nmask && nmask[v]) continue;
        PySet_Add(result, g->node_ids.node_id_items[v]);
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* predecessors_ctx: O(in-degree) predecessor lookup via reverse CSR. Directed only. */
PyObject *py_predecessors_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError, "predecessors requires a directed graph");
        return NULL;
    }

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (!found || !g->has_adjacency || !g->adjacency.reverse_offset) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return result; }

    AdjacencyList *al = &g->adjacency;
    for (int e = al->reverse_offset[u]; e < al->reverse_offset[u + 1]; e++) {
        int v = al->reverse_neighbors[e];
        if (v == u) continue;
        if (mask && mask[al->reverse_edge_indices[e]]) continue;
        if (nmask && nmask[v]) continue;
        PySet_Add(result, g->node_ids.node_id_items[v]);
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* degree_ctx: O(degree) degree count via CSR. For directed, returns out-degree. */
PyObject *py_degree_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;
    if (!found || !g->has_adjacency) return PyLong_FromLong(0);

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return PyLong_FromLong(0); }

    AdjacencyList *al = &g->adjacency;
    int count = 0;
    if (g->directed) {
        for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
            if (mask && mask[al->edge_indices[e]]) continue;
            if (nmask && nmask[al->neighbors[e]]) continue;
            count++;
        }
    } else {
        for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
            if (mask && mask[al->edge_indices[e]]) continue;
            int v = al->neighbors[e];
            if (nmask && nmask[v]) continue;
            count += (v == u) ? 1 : 1;  /* self-loops appear twice in CSR, each counts 1 */
        }
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return PyLong_FromLong(count);
}

/* in_degree_ctx: O(in-degree) via reverse CSR. Directed only. */
PyObject *py_in_degree_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError, "in_degree requires a directed graph");
        return NULL;
    }

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;
    if (!found || !g->has_adjacency || !g->adjacency.reverse_offset) return PyLong_FromLong(0);

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return PyLong_FromLong(0); }

    AdjacencyList *al = &g->adjacency;
    int count = 0;
    for (int e = al->reverse_offset[u]; e < al->reverse_offset[u + 1]; e++) {
        if (mask && mask[al->reverse_edge_indices[e]]) continue;
        if (nmask && nmask[al->reverse_neighbors[e]]) continue;
        count++;
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return PyLong_FromLong(count);
}

/* incident_edges_ctx: O(degree) edge index lookup via CSR. For directed, outgoing. */
PyObject *py_incident_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!found || !g->has_adjacency) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return result; }

    AdjacencyList *al = &g->adjacency;
    int prev_eid = -1;
    for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
        int eid = al->edge_indices[e];
        if (eid == prev_eid) continue;  /* deduplicate self-loops in undirected CSR */
        prev_eid = eid;
        if (mask && mask[eid]) continue;
        if (nmask && nmask[al->neighbors[e]]) continue;
        PyObject *val = PyLong_FromLong(eid);
        PyList_Append(result, val);
        Py_DECREF(val);
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* incoming_edges_ctx: O(in-degree) incoming edge indices via reverse CSR. Directed only. */
PyObject *py_incoming_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *nid_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &nid_obj, &mask_obj, &nmask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;
    if (!g->directed) {
        PyErr_SetString(PyExc_TypeError, "incoming_edge_indices requires a directed graph");
        return NULL;
    }

    int found;
    int u = resolve_node(g, nid_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!found || !g->has_adjacency || !g->adjacency.reverse_offset) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, g->node_ids.node_count, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    if (nmask && nmask[u]) { release_mask(&nmbuf); release_mask(&mbuf); return result; }

    AdjacencyList *al = &g->adjacency;
    for (int e = al->reverse_offset[u]; e < al->reverse_offset[u + 1]; e++) {
        int eid = al->reverse_edge_indices[e];
        if (mask && mask[eid]) continue;
        if (nmask && nmask[al->reverse_neighbors[e]]) continue;
        PyObject *val = PyLong_FromLong(eid);
        PyList_Append(result, val);
        Py_DECREF(val);
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* edge_indices_ctx: find edge indices between u and v via CSR. */
PyObject *py_edge_indices_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *u_obj, *v_obj, *mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|O", &capsule, &u_obj, &v_obj, &mask_obj)) return NULL;
    GraphContext *g = get_graph_context(capsule);
    if (!g) return NULL;

    int fu, fv;
    int u = resolve_node(g, u_obj, &fu);
    if (fu < 0) return NULL;
    int v = resolve_node(g, v_obj, &fv);
    if (fv < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!fu || !fv || !g->has_adjacency) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->node_ids.edges.edge_count, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }

    AdjacencyList *al = &g->adjacency;
    int prev_eid = -1;
    for (int e = al->offset[u]; e < al->offset[u + 1]; e++) {
        if (al->neighbors[e] != v) continue;
        int eid = al->edge_indices[e];
        if (eid == prev_eid) continue;  /* deduplicate self-loops in undirected CSR */
        prev_eid = eid;
        if (mask && mask[eid]) continue;
        PyObject *val = PyLong_FromLong(eid);
        PyList_Append(result, val);
        Py_DECREF(val);
    }
    release_mask(&mbuf);
    return result;
}
