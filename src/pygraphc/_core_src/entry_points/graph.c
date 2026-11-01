/* Graph construction: the parsed graph capsule, its counters, and edge translation. */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"

/*
 * Parse node ids and edges into a graph: one terminal parser for every edge
 * (parse_terminal_edges), or src/dst arrays of edge endpoints. Directed edges
 * are (u, v) or (in_nodes, out_nodes).
 */
PyObject *py_parse_graph(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_or_src, *dst_obj = Py_None;
    int directed = 0;
    if (!PyArg_ParseTuple(args, "OO|Op", &node_ids_obj, &edges_or_src, &dst_obj, &directed))
        return NULL;

    GraphContext *graph = (GraphContext *)calloc(1, sizeof(GraphContext));
    if (!graph) { PyErr_NoMemory(); return NULL; }
    graph->directed = directed;

    if (dst_obj != Py_None) {
        if (node_id_parse_split(node_ids_obj, edges_or_src, dst_obj, &graph->node_ids) < 0) { free(graph); return NULL; }
    } else {
        if (node_id_parse_nodes(node_ids_obj, &graph->node_ids) < 0) { free(graph); return NULL; }
        if (parse_terminal_edges(graph, edges_or_src) < 0) { free_graph_context(graph); return NULL; }
    }

    if (graph->node_ids.node_count > 0 && graph->node_ids.edges.edge_count > 0) {
        if (build_adjacency(graph->node_ids.node_count, &graph->node_ids.edges, graph->terminals, &graph->adjacency,
                            directed) < 0) {
            free_graph_context(graph);
            return NULL;
        }
        graph->has_adjacency = 1;
    }
    return PyCapsule_New(graph, "pygraphc.GraphCtx", graph_context_destructor);
}

PyObject *py_graph_shape(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    Py_ssize_t terminal_count = graph->terminals ? (Py_ssize_t)graph->terminals->terminal_count : 2 * edge_count;
    return Py_BuildValue("(inn)", graph->node_ids.node_count, edge_count, terminal_count);
}

PyObject *py_has_multi_terminal_edges(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    return PyBool_FromLong(graph->terminals != NULL);
}

PyObject *py_terminal_offsets(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    if (graph->terminals)
        return PyBytes_FromStringAndSize((const char *)graph->terminals->terminal_offsets,
                                         (edge_count + 1) * (Py_ssize_t)sizeof(int));
    PyObject *result = PyBytes_FromStringAndSize(NULL, (edge_count + 1) * (Py_ssize_t)sizeof(int));
    if (!result) return NULL;
    int *terminal_offsets = (int *)PyBytes_AS_STRING(result);
    for (Py_ssize_t edge_index = 0; edge_index <= edge_count; edge_index++) terminal_offsets[edge_index] = (int)(2 * edge_index);
    return result;
}

PyObject *py_terminal_nodes(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (graph->terminals)
        return PyBytes_FromStringAndSize((const char *)graph->terminals->terminal_nodes,
                                         (Py_ssize_t)graph->terminals->terminal_count * (Py_ssize_t)sizeof(int));
    const EdgeList *edges = &graph->node_ids.edges;
    PyObject *result = PyBytes_FromStringAndSize(NULL, 2 * (Py_ssize_t)edges->edge_count * (Py_ssize_t)sizeof(int));
    if (!result) return NULL;
    int *terminal_nodes = (int *)PyBytes_AS_STRING(result);
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        terminal_nodes[2 * edge_index] = EDGE_FIRST_ENDPOINT(edges, edge_index);
        terminal_nodes[2 * edge_index + 1] = EDGE_SECOND_ENDPOINT(edges, edge_index);
    }
    return result;
}

PyObject *py_graph_is_directed(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    return PyBool_FromLong(graph->directed);
}

PyObject *py_graph_edge_count(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    return PyLong_FromSsize_t(graph->node_ids.edges.edge_count);
}

PyObject *py_graph_node_count(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    return PyLong_FromLong(graph->node_ids.node_count);
}

/* ── translate_edges(node_ids, node_id_edges) -> index-based edges ── */

PyObject *py_translate_edges(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    PyObject *node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!node_ids_fast) return NULL;
    int node_count = (int)PySequence_Fast_GET_SIZE(node_ids_fast);
    PyObject **node_id_items = PySequence_Fast_ITEMS(node_ids_fast);

    /* Build hash map: node_id -> index */
    IntMap node_index_by_node_id;
    if (intmap_init(&node_index_by_node_id, node_count) < 0) { Py_DECREF(node_ids_fast); return NULL; }
    for (int node = 0; node < node_count; node++) {
        long node_id = PyLong_AsLong(node_id_items[node]);
        if (node_id == -1 && PyErr_Occurred()) {
            intmap_free(&node_index_by_node_id); Py_DECREF(node_ids_fast); return NULL;
        }
        intmap_put(&node_index_by_node_id, node_id, node);
    }
    Py_DECREF(node_ids_fast);

    /* Parse the node-ID edges and translate */
    PyObject *edges_fast = PySequence_Fast(edges_obj, "edges must be a sequence");
    if (!edges_fast) { intmap_free(&node_index_by_node_id); return NULL; }
    Py_ssize_t edge_count = PySequence_Fast_GET_SIZE(edges_fast);

    PyObject *result = PyList_New(edge_count);
    if (!result) { intmap_free(&node_index_by_node_id); Py_DECREF(edges_fast); return NULL; }

    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        PyObject *edge = edge_items[edge_index];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            intmap_free(&node_index_by_node_id); Py_DECREF(edges_fast); Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return NULL;
        }
        PyObject **endpoint_items = PySequence_Fast_ITEMS(edge_fast);
        long first_node_id = PyLong_AsLong(endpoint_items[0]);
        long second_node_id = PyLong_AsLong(endpoint_items[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            intmap_free(&node_index_by_node_id); Py_DECREF(edges_fast); Py_DECREF(result);
            return NULL;
        }
        int first_index = intmap_get(&node_index_by_node_id, first_node_id);
        int second_index = intmap_get(&node_index_by_node_id, second_node_id);
        if (first_index < 0 || second_index < 0) {
            intmap_free(&node_index_by_node_id); Py_DECREF(edges_fast); Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return NULL;
        }
        PyObject *pair = PyTuple_New(2);
        PyTuple_SET_ITEM(pair, 0, PyLong_FromLong(first_index));
        PyTuple_SET_ITEM(pair, 1, PyLong_FromLong(second_index));
        PyList_SET_ITEM(result, edge_index, pair);
    }

    intmap_free(&node_index_by_node_id);
    Py_DECREF(edges_fast);
    return result;
}
