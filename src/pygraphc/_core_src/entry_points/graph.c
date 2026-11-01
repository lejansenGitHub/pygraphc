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

PyObject *py_out_terminal_offsets(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "out-terminals exist on directed graphs only");
        return NULL;
    }
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    if (graph->terminals)
        return PyBytes_FromStringAndSize((const char *)graph->terminals->out_terminal_offsets,
                                         edge_count * (Py_ssize_t)sizeof(int));
    PyObject *result = PyBytes_FromStringAndSize(NULL, edge_count * (Py_ssize_t)sizeof(int));
    if (!result) return NULL;
    int *out_terminal_offsets = (int *)PyBytes_AS_STRING(result);
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) out_terminal_offsets[edge_index] = (int)(2 * edge_index + 1);
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

