#include "entry_points.h"
#include "../helpers/parse.h"

/* ── CSR-based query functions (O(degree) instead of O(m)) ── */

/* Helper: resolve a node ID to its internal index via the IntMap.
 * Returns -1 if not found. Sets *found to 0 if not found, 1 if found.
 * Does NOT set a Python exception on not-found (callers return empty). */
static int resolve_node(GraphContext *graph, PyObject *node_id_obj, int *found) {
    long node_id = PyLong_AsLong(node_id_obj);
    if (node_id == -1 && PyErr_Occurred()) { *found = -1; return -1; }
    if (graph->node_ids.node_count == 0 || !graph->node_ids.node_index_by_node_id.keys) { *found = 0; return -1; }
    int idx = intmap_get(&graph->node_ids.node_index_by_node_id, node_id);
    *found = (idx >= 0) ? 1 : 0;
    return idx;
}

/* neighbors_ctx: O(degree) neighbor lookup via CSR. */
PyObject *py_neighbors_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;  /* conversion error */

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (!found || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return result; }

    AdjacencyList *adjacency = &graph->adjacency;
    for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
        int neighbor = adjacency->neighbors[slot];
        if (neighbor == node) continue;  /* skip self-loops */
        if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
        if (node_mask && node_mask[neighbor]) continue;
        PySet_Add(result, graph->node_ids.node_id_items[neighbor]);
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* predecessors_ctx: O(in-degree) predecessor lookup via reverse CSR. Directed only. */
PyObject *py_predecessors_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "predecessors requires a directed graph");
        return NULL;
    }

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (!found || !graph->has_adjacency || !graph->adjacency.reverse_offset) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return result; }

    AdjacencyList *adjacency = &graph->adjacency;
    for (int slot = adjacency->reverse_offset[node]; slot < adjacency->reverse_offset[node + 1]; slot++) {
        int predecessor = adjacency->reverse_neighbors[slot];
        if (predecessor == node) continue;
        if (edge_mask && edge_mask[adjacency->reverse_edge_indices[slot]]) continue;
        if (node_mask && node_mask[predecessor]) continue;
        PySet_Add(result, graph->node_ids.node_id_items[predecessor]);
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* degree_ctx: O(degree) degree count via CSR. For directed, returns out-degree. */
PyObject *py_degree_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;
    if (!found || !graph->has_adjacency) return PyLong_FromLong(0);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return PyLong_FromLong(0); }

    AdjacencyList *adjacency = &graph->adjacency;
    int count = 0;
    if (graph->directed) {
        for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            if (node_mask && node_mask[adjacency->neighbors[slot]]) continue;
            count++;
        }
    } else {
        for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            int neighbor = adjacency->neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            count += (neighbor == node) ? 1 : 1;  /* self-loops appear twice in CSR, each counts 1 */
        }
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return PyLong_FromLong(count);
}

/* in_degree_ctx: O(in-degree) via reverse CSR. Directed only. */
PyObject *py_in_degree_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "in_degree requires a directed graph");
        return NULL;
    }

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;
    if (!found || !graph->has_adjacency || !graph->adjacency.reverse_offset) return PyLong_FromLong(0);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return PyLong_FromLong(0); }

    AdjacencyList *adjacency = &graph->adjacency;
    int count = 0;
    for (int slot = adjacency->reverse_offset[node]; slot < adjacency->reverse_offset[node + 1]; slot++) {
        if (edge_mask && edge_mask[adjacency->reverse_edge_indices[slot]]) continue;
        if (node_mask && node_mask[adjacency->reverse_neighbors[slot]]) continue;
        count++;
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return PyLong_FromLong(count);
}

/* incident_edges_ctx: O(degree) edge index lookup via CSR. For directed, outgoing. */
PyObject *py_incident_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!found || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return result; }

    AdjacencyList *adjacency = &graph->adjacency;
    int previous_edge_index = -1;
    for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
        int edge_index = adjacency->edge_indices[slot];
        if (edge_index == previous_edge_index) continue;  /* deduplicate self-loops in undirected CSR */
        previous_edge_index = edge_index;
        if (edge_mask && edge_mask[edge_index]) continue;
        if (node_mask && node_mask[adjacency->neighbors[slot]]) continue;
        PyObject *edge_index_obj = PyLong_FromLong(edge_index);
        PyList_Append(result, edge_index_obj);
        Py_DECREF(edge_index_obj);
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* incoming_edges_ctx: O(in-degree) incoming edge indices via reverse CSR. Directed only. */
PyObject *py_incoming_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "incoming_edge_indices requires a directed graph");
        return NULL;
    }

    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!found || !graph->has_adjacency || !graph->adjacency.reverse_offset) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    if (node_mask && node_mask[node]) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return result; }

    AdjacencyList *adjacency = &graph->adjacency;
    for (int slot = adjacency->reverse_offset[node]; slot < adjacency->reverse_offset[node + 1]; slot++) {
        int edge_index = adjacency->reverse_edge_indices[slot];
        if (edge_mask && edge_mask[edge_index]) continue;
        if (node_mask && node_mask[adjacency->reverse_neighbors[slot]]) continue;
        PyObject *edge_index_obj = PyLong_FromLong(edge_index);
        PyList_Append(result, edge_index_obj);
        Py_DECREF(edge_index_obj);
    }
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* edge_indices_ctx: find edge indices between first_node and second_node via CSR. */
PyObject *py_edge_indices_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *first_node_obj, *second_node_obj, *edge_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|O", &capsule, &first_node_obj, &second_node_obj, &edge_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;

    int first_found, second_found;
    int first_node = resolve_node(graph, first_node_obj, &first_found);
    if (first_found < 0) return NULL;
    int second_node = resolve_node(graph, second_node_obj, &second_found);
    if (second_found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!first_found || !second_found || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int previous_edge_index = -1;
    for (int slot = adjacency->offset[first_node]; slot < adjacency->offset[first_node + 1]; slot++) {
        if (adjacency->neighbors[slot] != second_node) continue;
        int edge_index = adjacency->edge_indices[slot];
        if (edge_index == previous_edge_index) continue;  /* deduplicate self-loops in undirected CSR */
        previous_edge_index = edge_index;
        if (edge_mask && edge_mask[edge_index]) continue;
        PyObject *edge_index_obj = PyLong_FromLong(edge_index);
        PyList_Append(result, edge_index_obj);
        Py_DECREF(edge_index_obj);
    }
    release_mask(&edge_mask_buffer);
    return result;
}
