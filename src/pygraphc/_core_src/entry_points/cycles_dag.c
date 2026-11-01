/* Directed acyclic graphs and cycles: topological sort, longest path, cycle basis. */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../algorithms/toposort.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"

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
    memset(&graph, 0, sizeof graph);
    graph.directed = 1;
    if (node_id_parse(node_ids_obj, edges_obj, &graph.node_ids) < 0) return NULL;
    if (graph.node_ids.node_count > 0 && graph.node_ids.edges.edge_count > 0) {
        if (build_adjacency(graph.node_ids.node_count, &graph.node_ids.edges, NULL, &graph.adjacency, 1) < 0) {
            node_id_context_free(&graph.node_ids);
            return NULL;
        }
        graph.has_adjacency = 1;
    }
    TerminalMask no_terminal_mask = {NULL, NULL, {0}};
    PyObject *result = topological_sort_of(&graph, NULL, &no_terminal_mask, NULL);
    if (graph.has_adjacency) free_adjacency(&graph.adjacency);
    node_id_context_free(&graph.node_ids);
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

/*
 * Find a fundamental cycle basis of an undirected graph.
 * For each non-tree edge of the BFS tree, traces the cycle through the tree.
 * Returns a Python list of lists of original node IDs.
 * Supports edge mask and node mask.
 */
PyObject *py_cycle_basis_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;

    if (graph->directed) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError,
            "cycle_basis is not defined for directed graphs");
        return NULL;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    EdgeList *edges = &graph->node_ids.edges;

    /* Self-loops are trivial cycles */
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        if (edge_mask && edge_mask[edge_index]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index);
        if (node_mask && node_mask[first_node]) continue;
        if (first_node == EDGE_SECOND_ENDPOINT(edges, edge_index)) {
            PyObject *cycle = PyList_New(1);
            if (!cycle) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
            Py_INCREF(graph->node_ids.node_id_items[first_node]);
            PyList_SET_ITEM(cycle, 0, graph->node_ids.node_id_items[first_node]);
            PyList_Append(result, cycle);
            Py_DECREF(cycle);
        }
    }

    /* Track which non-tree edges have been processed (one bit per edge) */
    Py_ssize_t edge_count = edges->edge_count;
    uint8_t *edge_seen = (uint8_t *)calloc(edge_count > 0 ? (size_t)edge_count : 1, 1);
    if (!edge_seen) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }

    /* Allocate: parent, depth, queue, parent_edge, path_a, path_b; node_count entries each */
    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free(edge_seen);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *parent      = scratch;
    int *depth       = scratch + node_count;
    int *queue       = scratch + 2 * node_count;
    int *parent_edge = scratch + 3 * node_count;  /* to skip the reverse tree edge */
    int *path_a      = scratch + 4 * node_count;
    int *path_b      = scratch + 5 * node_count;
    memset(depth, -1, (size_t)node_count * sizeof(int));

    /* BFS spanning tree + non-tree edge cycle extraction.
     * BFS trees are shallow (O(log n) for random graphs), making the
     * lowest-common-ancestor cycle trace O(log n) per non-tree edge instead of O(n). */
    for (int start = 0; start < node_count; start++) {
        if (depth[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;

        /* BFS to build spanning tree */
        int head = 0, tail = 0;
        queue[tail++] = start;
        depth[start] = 0;
        parent[start] = -1;
        parent_edge[start] = -1;

        while (head < tail) {
            int node = queue[head++];
            for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
                int edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                int neighbor = adjacency->neighbors[slot];
                if (node_mask && node_mask[neighbor]) continue;
                if (neighbor == node) continue;  /* skip self-loops (handled above) */
                if (depth[neighbor] == -1) {
                    /* Tree edge */
                    depth[neighbor] = depth[node] + 1;
                    parent[neighbor] = node;
                    parent_edge[neighbor] = edge_index;
                    queue[tail++] = neighbor;
                } else if (edge_index != parent_edge[node] && !edge_seen[edge_index]) {
                    /* Non-tree edge: extract cycle via the lowest common ancestor.
                     * edge_seen prevents processing each undirected
                     * edge twice (CSR has both node->neighbor and neighbor->node). */
                    edge_seen[edge_index] = 1;
                    int walker_a = node, walker_b = neighbor;
                    int path_a_length = 0, path_b_length = 0;

                    /* Walk walker_a and walker_b up to the same depth */
                    while (depth[walker_a] > depth[walker_b]) { path_a[path_a_length++] = walker_a; walker_a = parent[walker_a]; }
                    while (depth[walker_b] > depth[walker_a]) { path_b[path_b_length++] = walker_b; walker_b = parent[walker_b]; }
                    /* Walk both up to the lowest common ancestor */
                    while (walker_a != walker_b) {
                        path_a[path_a_length++] = walker_a; walker_a = parent[walker_a];
                        path_b[path_b_length++] = walker_b; walker_b = parent[walker_b];
                    }
                    /* walker_a == walker_b == lowest common ancestor */
                    int lowest_common_ancestor = walker_a;
                    int cycle_length = path_a_length + path_b_length + 1;

                    PyObject *cycle = PyList_New(cycle_length);
                    if (!cycle) {
                        free(scratch); free(edge_seen);
                        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
                        Py_DECREF(result); return NULL;
                    }
                    int idx = 0;
                    for (int position = 0; position < path_a_length; position++) {
                        Py_INCREF(graph->node_ids.node_id_items[path_a[position]]);
                        PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[path_a[position]]);
                    }
                    Py_INCREF(graph->node_ids.node_id_items[lowest_common_ancestor]);
                    PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[lowest_common_ancestor]);
                    for (int position = path_b_length - 1; position >= 0; position--) {
                        Py_INCREF(graph->node_ids.node_id_items[path_b[position]]);
                        PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[path_b[position]]);
                    }
                    PyList_Append(result, cycle);
                    Py_DECREF(cycle);
                }
            }
        }
    }

    free(scratch); free(edge_seen);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* ── DAG longest path (topological sort + DP) ── */

/*
 * Find the longest path in a weighted or unweighted DAG.
 * Uses topological sort + dynamic programming.
 * weights: if NULL, all edges have weight 1.
 * Returns Python list of original node IDs forming the longest path.
 */
/*
 * Longest distances over a topological order of states. A two-terminal edge
 * weighs its weight (with a terminal list, the sum of its two terminal
 * weights), 1 unweighted; entering a multi-terminal edge through an in-terminal
 * weighs that terminal's weight, 1 unweighted, and leaving it through an
 * out-terminal that terminal's weight, 0 unweighted, so a pass weighs entry
 * plus exit (D11) and counts 1 unweighted. An edge state relaxes its
 * out-terminals only once an in-terminal reached it. Instantiated for both
 * values of has_multi_terminal_edges.
 */
static inline __attribute__((always_inline)) void relax_in_order(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, const double *weights, const TopologicalOrder *order,
    double *distance, int *predecessor, const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    for (int position = 0; position < order->state_count; position++) {
        int state = order->states[position];
        if (has_multi_terminal_edges && state >= node_count) {
            int edge = state - node_count;
            if (predecessor[state] < 0) continue;
            for (int terminal = terminals->out_terminal_offsets[edge]; terminal < terminals->terminal_offsets[edge + 1]; terminal++) {
                if (!terminal_is_active(terminals, edge, terminal, edge_mask, terminal_mask, node_mask)) continue;
                int reached = terminals->terminal_nodes[terminal];
                double candidate = distance[state] + (weights ? weights[terminal] : 0.0);
                if (candidate > distance[reached]) { distance[reached] = candidate; predecessor[reached] = state; }
            }
            continue;
        }
        if (!graph->has_adjacency) continue;
        for (int slot = adjacency->offset[state]; slot < adjacency->offset[state + 1]; slot++) {
            int neighbor = adjacency->neighbors[slot], index = adjacency->edge_indices[slot];
            double weight;
            if (has_multi_terminal_edges && neighbor < 0) {
                int edge = -1 - neighbor;
                if (!terminal_is_active(terminals, edge, index, edge_mask, terminal_mask, node_mask)) continue;
                neighbor = node_count + edge;
                weight = weights ? weights[index] : 1.0;
                if (predecessor[neighbor] < 0 || distance[state] + weight > distance[neighbor]) {
                    distance[neighbor] = distance[state] + weight;
                    predecessor[neighbor] = state;
                }
                continue;
            }
            if (two_terminal_edge_mask && two_terminal_edge_mask[index]) continue;
            if (node_mask && node_mask[neighbor]) continue;
            weight = !weights ? 1.0
                   : has_multi_terminal_edges ? weights[terminals->terminal_offsets[index]] + weights[terminals->terminal_offsets[index] + 1]
                   : weights[index];
            if (distance[state] + weight > distance[neighbor]) {
                distance[neighbor] = distance[state] + weight;
                predecessor[neighbor] = state;
            }
        }
    }
}

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

    PyObject *result = NULL;
    TopologicalOrder order = {NULL, 0, 0, 0};
    size_t state_count = (size_t)node_count + (terminals ? (size_t)graph->node_ids.edges.edge_count : 0);
    double *distance = (double *)malloc(state_count * sizeof(double));
    int *predecessor = (int *)malloc(state_count * sizeof(int));
    if (!distance || !predecessor) { PyErr_NoMemory(); goto done; }
    if (topological_order(graph, edge_mask, &terminal_mask, node_mask, &order) < 0) goto done;
    if (order.nodes_in_order < order.node_count) {
        PyErr_SetString(PyExc_ValueError, "Graph contains a cycle - dag_longest_path is undefined");
        goto done;
    }
    for (size_t state = 0; state < state_count; state++) { distance[state] = 0.0; predecessor[state] = -1; }
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(&terminal_mask, edge_mask);
    const double *weight_values = has_weights ? weights.values : NULL;
    if (terminals)
        relax_in_order(graph, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask, weight_values, &order,
                       distance, predecessor, 1);
    else
        relax_in_order(graph, traversal_edge_mask, NULL, NULL, node_mask, weight_values, &order, distance, predecessor, 0);

    /* The node with the longest distance, the first in order on a tie */
    int best_node = -1, first_node = -1;
    double best_distance = -1.0;
    for (int position = 0; position < order.state_count; position++) {
        int state = order.states[position];
        if (state >= node_count) continue;
        if (first_node < 0) first_node = state;
        if (distance[state] > best_distance) { best_distance = distance[state]; best_node = state; }
    }
    if (best_node < 0 || best_distance == 0.0) {
        result = PyList_New(first_node >= 0 ? 1 : 0);
        if (result && first_node >= 0) {
            Py_INCREF(graph->node_ids.node_id_items[first_node]);
            PyList_SET_ITEM(result, 0, graph->node_ids.node_id_items[first_node]);
        }
        goto done;
    }
    int path_length = 0;
    for (int state = best_node; state >= 0; state = predecessor[state]) path_length += state < node_count;
    result = PyList_New(path_length);
    if (!result) goto done;
    int position = path_length - 1;
    for (int state = best_node; state >= 0; state = predecessor[state]) {
        if (state >= node_count) continue;
        Py_INCREF(graph->node_ids.node_id_items[state]);
        PyList_SET_ITEM(result, position--, graph->node_ids.node_id_items[state]);
    }

done:
    free(distance); free(predecessor); free(order.states);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (has_weights) free_weights(&weights);
    return result;
}
