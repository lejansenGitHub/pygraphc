/* Bridges, articulation points and biconnected components (iterative Tarjan). */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/biconnectivity.h"

/* The empty result of a mode: a set of critical nodes, (blocks, edge indices by block), or a list. */
static PyObject *empty_result(TarjanMode mode) {
    if (mode == TARJAN_CRITICAL_NODES) return PySet_New(NULL);
    if (mode == TARJAN_SIMPLE_PATH_BLOCKS) return Py_BuildValue("(NN)", PyList_New(0), PyDict_New());
    return PyList_New(0);
}

/* Critical edges as edge indices, or as (node id, node id) pairs of their endpoints, in the order found. */
static PyObject *critical_edge_list(const GraphContext *graph, const TarjanReport *report, int names_node_pairs) {
    PyObject *result = PyList_New(report->critical_edge_count);
    if (!result) return NULL;
    PyObject *const *node_id_items = graph->node_ids.node_id_items;
    for (int position = 0; position < report->critical_edge_count; position++) {
        int edge_index = report->critical_edges[position];
        PyObject *item = names_node_pairs
            ? PyTuple_Pack(2, node_id_items[EDGE_FIRST_ENDPOINT(&graph->node_ids.edges, edge_index)],
                           node_id_items[EDGE_SECOND_ENDPOINT(&graph->node_ids.edges, edge_index)])
            : PyLong_FromLong(edge_index);
        if (!item) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, position, item);
    }
    return result;
}

/* The node ids of the critical nodes. */
static PyObject *critical_node_set(const GraphContext *graph, const TarjanReport *report) {
    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    for (int node = 0; node < graph->node_ids.node_count; node++) {
        if (report->critical_nodes[node] && PySet_Add(result, graph->node_ids.node_id_items[node]) < 0) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/*
 * The node id set of one block and, when edge_indices_out is given, the set of
 * the edge indices of its edge states (NULL when it has none). Returns 0, or -1
 * with an exception set and nothing to release.
 */
static int block_sets(const GraphContext *graph, const TarjanReport *report, int block, PyObject **nodes_out,
                      PyObject **edge_indices_out) {
    int node_count = graph->node_ids.node_count;
    PyObject *nodes = PySet_New(NULL), *edge_indices = NULL;
    if (!nodes) return -1;
    for (int position = report->block_offsets[block]; position < report->block_offsets[block + 1]; position++) {
        int state = report->block_states[position];
        if (state < node_count) {
            if (PySet_Add(nodes, graph->node_ids.node_id_items[state]) < 0) goto failed;
            continue;
        }
        if (!edge_indices && !(edge_indices = PySet_New(NULL))) goto failed;
        PyObject *edge_index_obj = PyLong_FromLong(state - node_count);
        int status = edge_index_obj ? PySet_Add(edge_indices, edge_index_obj) : -1;
        Py_XDECREF(edge_index_obj);
        if (status < 0) goto failed;
    }
    *nodes_out = nodes;
    if (edge_indices_out) *edge_indices_out = edge_indices;
    return 0;
failed:
    Py_DECREF(nodes); Py_XDECREF(edge_indices);
    return -1;
}

/* Blocks as a list of node id sets. */
static PyObject *block_list(const GraphContext *graph, const TarjanReport *report) {
    PyObject *result = PyList_New(report->block_count);
    if (!result) return NULL;
    for (int block = 0; block < report->block_count; block++) {
        PyObject *nodes;
        if (block_sets(graph, report, block, &nodes, NULL) < 0) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, block, nodes);
    }
    return result;
}

/* Simple path blocks as (list of node id sets, block index -> set of its multi-terminal edge indices). */
static PyObject *simple_path_block_pair(const GraphContext *graph, const TarjanReport *report) {
    PyObject *blocks = PyList_New(report->block_count), *edge_indices_by_block = PyDict_New();
    if (!blocks || !edge_indices_by_block) goto failed;
    for (int block = 0; block < report->block_count; block++) {
        PyObject *nodes, *edge_indices;
        if (block_sets(graph, report, block, &nodes, &edge_indices) < 0) goto failed;
        PyList_SET_ITEM(blocks, block, nodes);
        if (!edge_indices) continue;
        PyObject *block_index_obj = PyLong_FromLong(block);
        int status = block_index_obj ? PyDict_SetItem(edge_indices_by_block, block_index_obj, edge_indices) : -1;
        Py_XDECREF(block_index_obj); Py_DECREF(edge_indices);
        if (status < 0) goto failed;
    }
    return Py_BuildValue("(NN)", blocks, edge_indices_by_block);
failed:
    Py_XDECREF(blocks); Py_XDECREF(edge_indices_by_block);
    return NULL;
}

/* Run tarjan_report and convert what it found into the result of the mode. */
static PyObject *tarjan_result(const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
                               const uint8_t *terminal_mask, const uint8_t *node_mask, TarjanMode mode,
                               int names_node_pairs) {
    if (graph->node_ids.node_count == 0 || !graph->has_adjacency) return empty_result(mode);
    TarjanReport report;
    if (tarjan_report(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, mode, &report) < 0) return NULL;
    PyObject *result = mode == TARJAN_CRITICAL_EDGES ? critical_edge_list(graph, &report, names_node_pairs)
                     : mode == TARJAN_CRITICAL_NODES ? critical_node_set(graph, &report)
                     : mode == TARJAN_SIMPLE_PATH_BLOCKS ? simple_path_block_pair(graph, &report)
                     : block_list(graph, &report);
    release_tarjan_report(&report);
    return result;
}

/* ── Node-ID edges, parsed per call ── */

/* Parse (node_ids, edges) into an undirected graph of two-terminal edges and run the search in the given mode. */
static PyObject *tarjan_by_node_id(PyObject *args, TarjanMode mode, int names_node_pairs) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;
    GraphContext graph;
    if (parse_graph_by_node_id(node_ids_obj, edges_obj, 0, &graph) < 0) return NULL;
    PyObject *result = tarjan_result(&graph, NULL, NULL, NULL, NULL, mode, names_node_pairs);
    release_graph_by_node_id(&graph);
    return result;
}

PyObject *py_bridges_by_node_id(PyObject *self, PyObject *args) { return tarjan_by_node_id(args, TARJAN_CRITICAL_EDGES, 1); }
PyObject *py_articulation_points_by_node_id(PyObject *self, PyObject *args) { return tarjan_by_node_id(args, TARJAN_CRITICAL_NODES, 0); }
PyObject *py_biconnected_components_by_node_id(PyObject *self, PyObject *args) { return tarjan_by_node_id(args, TARJAN_BLOCKS, 0); }

/* ── The parsed graph ── */

/*
 * Parse (capsule[, edge_mask, node_mask, terminal_mask]) and run the search in the
 * given mode; names_node_pairs reports critical edges as bridges, by node pair.
 */
static PyObject *tarjan_entry(PyObject *args, TarjanMode mode, int names_node_pairs) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (names_node_pairs && graph->terminals) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "bridges names edges by node pair, which cannot name an edge with more than two terminals; "
                        "use critical_edges()");
        return NULL;
    }
    if (graph->directed) {
        PyErr_SetString(PyExc_TypeError, "critical edges, critical nodes and blocks are not defined for directed graphs");
        return NULL;
    }
    if (graph->node_ids.node_count == 0 || !graph->has_adjacency) return empty_result(mode);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }

    PyObject *result = tarjan_result(graph, two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask,
                                     terminal_mask.terminals, node_mask, mode, names_node_pairs);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_bridges_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, TARJAN_CRITICAL_EDGES, 1); }
PyObject *py_critical_edges_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, TARJAN_CRITICAL_EDGES, 0); }
PyObject *py_articulation_points_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, TARJAN_CRITICAL_NODES, 0); }
PyObject *py_biconnected_components_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, TARJAN_BLOCKS, 0); }
PyObject *py_simple_path_blocks_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, TARJAN_SIMPLE_PATH_BLOCKS, 0); }
