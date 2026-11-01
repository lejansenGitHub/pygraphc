/* Dijkstra: source-target paths, single-source and multi-source lengths. */
#include "entry_points.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"
#include "../conversion/parse.h"
#include "../algorithms/bidirectional_dijkstra.h"
#include "../algorithms/dijkstra.h"

/* Turn the node indices of a bidirectional result into a list of node IDs. */
static PyObject *bidirectional_path_to_list(const BidirectionalPath *bidirectional_path, const NodeIdContext *context) {
    PyObject *path = PyList_New(bidirectional_path->path_length);
    if (!path) return NULL;
    for (int position = 0; position < bidirectional_path->path_length; position++) {
        PyObject *item = context->node_id_items[bidirectional_path->path[position]];
        Py_INCREF(item);
        PyList_SET_ITEM(path, position, item);
    }
    return path;
}

/* ── dijkstra (node-ID edges) ── */

PyObject *py_dijkstra_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *source_obj, *target_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &node_ids_obj, &edges_obj, &weights_obj, &source_obj, &target_obj))
        return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    long source_node_id = PyLong_AsLong(source_obj), target_node_id = PyLong_AsLong(target_obj);
    if (PyErr_Occurred()) { node_id_context_free(&context); return NULL; }
    int source = node_count > 0 ? intmap_get(&context.node_index_by_node_id, source_node_id) : -1;
    int target = node_count > 0 ? intmap_get(&context.node_index_by_node_id, target_node_id) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(context.node_id_items[source]);
        PyList_SET_ITEM(path, 0, context.node_id_items[source]);
        node_id_context_free(&context);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (node_count == 0 || source < 0 || target < 0) {
        node_id_context_free(&context);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, context.edges.edge_count) < 0) { node_id_context_free(&context); return NULL; }
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); free_weights(&weights); return NULL; }

    BidirectionalPath bidirectional_path;
    int failed = bidirectional_dijkstra(node_count, 0, &adjacency, weights.values, NULL, NULL, NULL, NULL, NULL, NULL, 0,
                                        source, target, &bidirectional_path) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &context);
    free(bidirectional_path.path);
    free_adjacency(&adjacency); free_weights(&weights);
    if (!path) { node_id_context_free(&context); return NULL; }
    PyObject *result = Py_BuildValue("(dN)", bidirectional_path.weight, path);
    node_id_context_free(&context);
    return result;
}

PyObject *py_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *target_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|OOO", &capsule, &weights_obj, &source_obj, &target_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    int edge_count = (int)graph->node_ids.edges.edge_count;
    const TerminalList *terminals = graph->terminals;

    long source_node_id = PyLong_AsLong(source_obj), target_node_id = PyLong_AsLong(target_obj);
    if (PyErr_Occurred()) return NULL;
    int source = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    int target = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, target_node_id) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(graph->node_ids.node_id_items[source]);
        PyList_SET_ITEM(path, 0, graph->node_ids.node_id_items[source]);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (node_count == 0 || source < 0 || target < 0) {
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && (node_mask[source] || node_mask[target])) {
        release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_counted_weights(weights_obj, &weights, terminals ? terminals->terminal_count : edge_count, terminals ? "terminal" : "edge", terminals != NULL) < 0) { release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }
    if (!graph->has_adjacency) { free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); PyObject *empty_path = PyList_New(0); return Py_BuildValue("(dN)", HUGE_VAL, empty_path); }

    PassWeights pass_weights = {NULL, NULL};
    if (terminals && weights.has_sign_bit && parse_pass_weights(terminals, edge_count, weights.values, graph->directed, &pass_weights) < 0) {
        free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        return NULL;
    }
    BidirectionalPath bidirectional_path;
    int failed = (pass_weights.shifts && !graph->directed)
        ? labeled_shortest_path(graph, weights.values, &pass_weights, two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask,
                                terminal_mask.terminals, node_mask, source, target, &bidirectional_path) < 0
        : bidirectional_dijkstra(node_count, edge_count, &graph->adjacency, weights.values, terminals, &pass_weights,
                                 two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask, terminal_mask.terminals,
                                 node_mask, graph->directed, source, target, &bidirectional_path) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &graph->node_ids);
    free(bidirectional_path.path);
    release_pass_weights(&pass_weights);
    free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (!path) return NULL;
    return Py_BuildValue("(dN)", bidirectional_path.weight, path);
}

/*
 * {node id: distance} from the given source indices (masked or repeated
 * sources skipped): one weight per edge on a graph of two-terminal edges, one per
 * terminal on a graph with multi-terminal edges. NULL with an exception set on error.
 */
static PyObject *shortest_path_lengths_from(const GraphContext *graph, PyObject *weights_obj, const int *sources,
                                            Py_ssize_t source_count, double cutoff, PyObject *edge_mask_obj,
                                            PyObject *node_mask_obj, PyObject *terminal_mask_obj) {
    int node_count = graph->node_ids.node_count;
    int use_cutoff = (cutoff >= 0.0);
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    const TerminalList *terminals = graph->terminals;
    WeightList weights;
    if (parse_counted_weights(weights_obj, &weights, terminals ? terminals->terminal_count : graph->node_ids.edges.edge_count,
                              terminals ? "terminal" : "edge", terminals != NULL) < 0) {
        release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL;
    }
    PassWeights pass_weights = {NULL, NULL};
    double *distance = NULL;
    int status = (terminals && weights.has_sign_bit &&
                  parse_pass_weights(terminals, graph->node_ids.edges.edge_count, weights.values, graph->directed, &pass_weights) < 0)
        ? -1
        : shortest_path_distances(graph, weights.values, &pass_weights, two_terminal_edge_mask(&terminal_mask, edge_mask),
                                  edge_mask, terminal_mask.terminals, node_mask, sources, source_count, cutoff, &distance);
    release_pass_weights(&pass_weights);
    for (int node = 0; status == 0 && node < node_count; node++) {
        if (node_mask && node_mask[node]) continue;
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            if (!value || PyDict_SetItem(result, graph->node_ids.node_id_items[node], value) < 0) status = -1;
            Py_XDECREF(value);
        }
    }
    free(distance);
    free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
    return result;
}

/* ── shortest_path_lengths and multi_source_dijkstra (node-ID edges) ── */

/*
 * Parse (node_ids, edges, weights, sources, cutoff) into an undirected graph and run
 * shortest_path_lengths_from; with single_source an unknown source gives {} before the weights are read.
 */
static PyObject *shortest_path_lengths_by_node_id(PyObject *node_ids_obj, PyObject *edges_obj, PyObject *weights_obj,
                                                  PyObject *sources_obj, double cutoff, int single_source) {
    GraphContext graph;
    if (parse_graph_by_node_id(node_ids_obj, edges_obj, 0, &graph) < 0) return NULL;
    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) { release_graph_by_node_id(&graph); return NULL; }
    Py_ssize_t source_count = PySequence_Fast_GET_SIZE(sources_fast);
    PyObject **source_items = PySequence_Fast_ITEMS(sources_fast);
    int *sources = malloc(((size_t)source_count + 1) * sizeof(int));
    PyObject *result = NULL;
    if (!sources) { PyErr_NoMemory(); goto done; }
    for (Py_ssize_t position = 0; position < source_count; position++) {
        long source_node_id = PyLong_AsLong(source_items[position]);
        if (PyErr_Occurred()) goto done;
        sources[position] = graph.node_ids.node_count > 0 ? intmap_get(&graph.node_ids.node_index_by_node_id, source_node_id) : -1;
    }
    result = (single_source && sources[0] < 0)
        ? PyDict_New()
        : shortest_path_lengths_from(&graph, weights_obj, sources, source_count, cutoff, Py_None, Py_None, Py_None);
done:
    free(sources);
    Py_DECREF(sources_fast);
    release_graph_by_node_id(&graph);
    return result;
}

PyObject *py_shortest_path_lengths_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *source_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &node_ids_obj, &edges_obj, &weights_obj, &source_obj, &cutoff)) return NULL;
    PyObject *sources_obj = PyTuple_Pack(1, source_obj);
    if (!sources_obj) return NULL;
    PyObject *result = shortest_path_lengths_by_node_id(node_ids_obj, edges_obj, weights_obj, sources_obj, cutoff, 1);
    Py_DECREF(sources_obj);
    return result;
}

PyObject *py_multi_source_dijkstra_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *sources_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &node_ids_obj, &edges_obj, &weights_obj, &sources_obj, &cutoff)) return NULL;
    return shortest_path_lengths_by_node_id(node_ids_obj, edges_obj, weights_obj, sources_obj, cutoff, 0);
}

PyObject *py_shortest_path_lengths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &source_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    long source_node_id = PyLong_AsLong(source_obj);
    if (PyErr_Occurred()) return NULL;
    int source = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (source < 0) return PyDict_New();
    return shortest_path_lengths_from(graph, weights_obj, &source, 1, cutoff, edge_mask_obj, node_mask_obj, terminal_mask_obj);
}

PyObject *py_multi_source_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *sources_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &sources_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) return NULL;
    Py_ssize_t source_count = PySequence_Fast_GET_SIZE(sources_fast);
    PyObject **source_items = PySequence_Fast_ITEMS(sources_fast);
    int *sources = malloc(((size_t)source_count + 1) * sizeof(int));
    if (!sources) { Py_DECREF(sources_fast); return PyErr_NoMemory(); }
    for (Py_ssize_t position = 0; position < source_count; position++) {
        long source_node_id = PyLong_AsLong(source_items[position]);
        if (PyErr_Occurred()) { free(sources); Py_DECREF(sources_fast); return NULL; }
        sources[position] = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    }
    Py_DECREF(sources_fast);
    PyObject *result = shortest_path_lengths_from(graph, weights_obj, sources, source_count, cutoff,
                                                  edge_mask_obj, node_mask_obj, terminal_mask_obj);
    free(sources);
    return result;
}
