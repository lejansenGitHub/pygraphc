/* Directed acyclic graphs and cycles: topological sort, longest path, cycle basis. */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../algorithms/toposort.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/cycle_basis.h"
#include "../algorithms/longest_path.h"

/* The node ids of an order's node states, in order; edge states are left out. NULL with an exception set. */
static PyObject *node_ids_in_order(const GraphContext *graph, const TopologicalOrder *order) {
    PyObject *result = PyList_New(order->nodes_in_order);
    if (!result) return NULL;
    int position = 0;
    for (int index = 0; index < order->state_count; index++) {
        int state = order->states[index];
        if (state >= graph->node_ids.node_count) continue;
        Py_INCREF(graph->node_ids.node_id_items[state]);
        PyList_SET_ITEM(result, position++, graph->node_ids.node_id_items[state]);
    }
    return result;
}

/* The topological sort of a parsed graph as a list of node ids; ValueError on a cycle. */
static PyObject *topological_sort_of(const GraphContext *graph, const uint8_t *edge_mask, const TerminalMask *terminal_mask,
                                     const uint8_t *node_mask) {
    TopologicalOrder order;
    if (topological_order(graph, edge_mask, terminal_mask, node_mask, &order) < 0) return NULL;
    PyObject *result = NULL;
    if (order.nodes_in_order < order.node_count)
        PyErr_Format(PyExc_ValueError,
                     "Graph contains a cycle - topological sort is undefined (%d of %d nodes processed)",
                     order.nodes_in_order, order.node_count);
    else
        result = node_ids_in_order(graph, &order);
    free(order.states);
    return result;
}

PyObject *py_topological_sort_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    GraphContext graph;
    if (parse_graph_by_node_id(node_ids_obj, edges_obj, 1, &graph) < 0) return NULL;
    TerminalMask no_terminal_mask = {NULL, NULL, {0}};
    PyObject *result = topological_sort_of(&graph, NULL, &no_terminal_mask, NULL);
    release_graph_by_node_id(&graph);
    return result;
}

PyObject *py_topological_sort_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "topological_sort requires a directed graph");
        return NULL;
    }
    if (graph->node_ids.node_count == 0) return PyList_New(0);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }
    PyObject *result = topological_sort_of(graph, edge_mask, &terminal_mask, node_mask);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* ── Cycle basis (fundamental cycles via BFS spanning tree) ── */

/* The list cycles are appended to, and the node ids they are named by. */
typedef struct {
    PyObject *result;
    PyObject *const *node_id_items;
} CycleList;

/* Append each cycle to the result list as a list of node ids, its reversed tail read backwards. */
static inline __attribute__((always_inline)) int append_cycle(void *visitor_state, const int *nodes, int length,
                                                              const int *reversed_tail, int reversed_length) {
    const CycleList *cycles = (const CycleList *)visitor_state;
    PyObject *const *node_id_items = cycles->node_id_items;
    PyObject *cycle = PyList_New(length + reversed_length);
    if (!cycle) return -1;
    for (int position = 0; position < length; position++) {
        PyObject *node_id = node_id_items[nodes[position]];
        Py_INCREF(node_id);
        PyList_SET_ITEM(cycle, position, node_id);
    }
    for (int position = 0; position < reversed_length; position++) {
        PyObject *node_id = node_id_items[reversed_tail[reversed_length - 1 - position]];
        Py_INCREF(node_id);
        PyList_SET_ITEM(cycle, length + position, node_id);
    }
    int status = PyList_Append(cycles->result, cycle);
    Py_DECREF(cycle);
    return status;
}

/*
 * Find a fundamental cycle basis of an undirected graph under its edge, node and
 * terminal masks (D25). Returns a Python list of lists of original node IDs.
 */
PyObject *py_cycle_basis_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (graph->directed) {
        PyErr_SetString(PyExc_TypeError, "cycle_basis is not defined for directed graphs");
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (!result || graph->node_ids.node_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) {
        release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL;
    }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL;
    }
    CycleList cycles = {result, graph->node_ids.node_id_items};
    int status = fundamental_cycle_basis(graph, two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask,
                                         terminal_mask.terminals, node_mask, append_cycle, &cycles);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
    return result;
}

/* ── DAG longest path (topological sort + DP) ── */

PyObject *py_dag_longest_path_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj = Py_None, *edge_mask_obj = Py_None, *node_mask_obj = Py_None,
             *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOOO", &capsule, &weights_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError, "dag_longest_path requires a directed graph");
        return NULL;
    }
    if (node_count == 0) return PyList_New(0);

    const TerminalList *terminals = graph->terminals;
    WeightList weights = {NULL, 0, 0, 0};
    int has_weights = weights_obj != Py_None;
    if (has_weights &&
        parse_counted_weights(weights_obj, &weights, terminals ? terminals->terminal_count : graph->node_ids.edges.edge_count,
                              terminals ? "terminal" : "edge", 0) < 0)
        return NULL;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) {
        if (has_weights) free_weights(&weights);
        return NULL;
    }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) {
        release_mask(&edge_mask_buffer); if (has_weights) free_weights(&weights);
        return NULL;
    }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); if (has_weights) free_weights(&weights);
        return NULL;
    }

    int *path = NULL, path_length = 0;
    PyObject *result = NULL;
    if (dag_longest_path(graph, has_weights ? weights.values : NULL, edge_mask, &terminal_mask, node_mask, &path,
                         &path_length) == 0 && (result = PyList_New(path_length))) {
        for (int position = 0; position < path_length; position++) {
            PyObject *node_id = graph->node_ids.node_id_items[path[position]];
            Py_INCREF(node_id);
            PyList_SET_ITEM(result, position, node_id);
        }
    }
    free(path);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (has_weights) free_weights(&weights);
    return result;
}
