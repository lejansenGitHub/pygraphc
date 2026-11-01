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

/* Whether a search state is a node; without multi-terminal edges every state is, and the check compiles away. */
static inline __attribute__((always_inline)) int is_node_state(int state, int node_count, const int has_multi_terminal_edges) {
    return !has_multi_terminal_edges || state < node_count;
}

/*
 * Append the cycle closed by a non-tree step between two discovered states:
 * both walk up to their lowest common ancestor, keeping only node states, so
 * the list holds the nodes of a Berge cycle in order. path_a and path_b hold
 * state_count entries. Returns 0, or -1 with an exception set.
 */
static inline __attribute__((always_inline)) int append_closed_cycle(
    const GraphContext *graph, const int *parent, const int *depth, int *path_a, int *path_b, int walker_a, int walker_b,
    PyObject *result, const int has_multi_terminal_edges) {
    int node_count = graph->node_ids.node_count, path_a_length = 0, path_b_length = 0;
    while (depth[walker_a] > depth[walker_b]) {
        if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path_a[path_a_length++] = walker_a;
        walker_a = parent[walker_a];
    }
    while (depth[walker_b] > depth[walker_a]) {
        if (is_node_state(walker_b, node_count, has_multi_terminal_edges)) path_b[path_b_length++] = walker_b;
        walker_b = parent[walker_b];
    }
    while (walker_a != walker_b) {
        if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path_a[path_a_length++] = walker_a;
        if (is_node_state(walker_b, node_count, has_multi_terminal_edges)) path_b[path_b_length++] = walker_b;
        walker_a = parent[walker_a]; walker_b = parent[walker_b];
    }
    if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path_a[path_a_length++] = walker_a;
    PyObject *cycle = PyList_New(path_a_length + path_b_length);
    if (!cycle) return -1;
    for (int position = 0; position < path_a_length; position++) {
        Py_INCREF(graph->node_ids.node_id_items[path_a[position]]);
        PyList_SET_ITEM(cycle, position, graph->node_ids.node_id_items[path_a[position]]);
    }
    for (int position = 0; position < path_b_length; position++) {
        PyObject *node_id = graph->node_ids.node_id_items[path_b[path_b_length - 1 - position]];
        Py_INCREF(node_id);
        PyList_SET_ITEM(cycle, path_a_length + position, node_id);
    }
    int status = PyList_Append(result, cycle);
    Py_DECREF(cycle);
    return status;
}

/*
 * A fundamental cycle basis over the incidence graph (D25): the states are the
 * nodes and, with has_multi_terminal_edges, one state per edge with three or
 * more terminals (node_count + e), joined to the node of each of its active
 * terminals when it has at least two; a two-terminal edge joins its endpoints
 * directly. Every step has an id (a two-terminal edge its index, terminal t
 * edge_count + t); a BFS tree over the states leaves one non-tree step per
 * independent cycle, so there are terminals - nodes - edges + components
 * cycles. A self-loop, and an edge with two terminals at one node, give the
 * cycle [node]. Instantiated for both values of has_multi_terminal_edges.
 */
static inline __attribute__((always_inline)) int fundamental_cycles(
    const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, PyObject *result, const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    const EdgeList *edges = &graph->node_ids.edges;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = edges->edge_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)edge_count : 0);
    size_t step_count = (size_t)edge_count + (has_multi_terminal_edges ? (size_t)terminals->terminal_count : 0);

    /* Self-loops of two-terminal edges are cycles of their own */
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        if (has_multi_terminal_edges && terminals->terminal_offsets[edge + 1] - terminals->terminal_offsets[edge] != 2) continue;
        if (two_terminal_edges && two_terminal_edges[edge]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge);
        if (first_node != EDGE_SECOND_ENDPOINT(edges, edge) || (node_mask && node_mask[first_node])) continue;
        PyObject *cycle = PyList_New(1);
        if (!cycle) return -1;
        Py_INCREF(graph->node_ids.node_id_items[first_node]);
        PyList_SET_ITEM(cycle, 0, graph->node_ids.node_id_items[first_node]);
        int appended = PyList_Append(result, cycle);
        Py_DECREF(cycle);
        if (appended < 0) return -1;
    }

    uint8_t *step_seen = (uint8_t *)calloc(step_count > 0 ? step_count : 1, 1);
    /* parent, depth, queue, parent_step, path_a, path_b: state_count entries each */
    int *scratch = (int *)malloc(6 * (state_count > 0 ? state_count : 1) * sizeof(int));
    /* the active terminals of each multi-terminal edge, stopping at two: an edge with fewer joins nothing */
    uint8_t *edge_joins = has_multi_terminal_edges ? (uint8_t *)calloc(edge_count > 0 ? (size_t)edge_count : 1, 1) : NULL;
    if (!step_seen || !scratch || (has_multi_terminal_edges && !edge_joins)) {
        free(step_seen); free(scratch); free(edge_joins); PyErr_NoMemory(); return -1;
    }
    int *parent = scratch, *depth = scratch + state_count, *queue = scratch + 2 * state_count;
    int *parent_step = scratch + 3 * state_count, *path_a = scratch + 4 * state_count, *path_b = scratch + 5 * state_count;
    memset(depth, -1, state_count * sizeof(int));
    if (has_multi_terminal_edges) {
        for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
            if (!is_multi_terminal_edge(terminals, (int)edge)) continue;
            int active = 0;
            for (int terminal = terminals->terminal_offsets[edge]; terminal < terminals->terminal_offsets[edge + 1] && active < 2; terminal++)
                active += terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask);
            edge_joins[edge] = active >= 2;
        }
    }

    int status = 0;
    for (int start = 0; start < node_count && status == 0; start++) {
        if (depth[start] != -1 || (node_mask && node_mask[start])) continue;
        int head = 0, tail = 0;
        queue[tail++] = start;
        depth[start] = 0; parent[start] = -1; parent_step[start] = -1;
        while (head < tail && status == 0) {
            int state = queue[head++];
            int is_edge_state = has_multi_terminal_edges && state >= node_count;
            int first_slot = is_edge_state ? terminals->terminal_offsets[state - node_count] : adjacency->offset[state];
            int end_slot = is_edge_state ? terminals->terminal_offsets[state - node_count + 1] : adjacency->offset[state + 1];
            for (int slot = first_slot; slot < end_slot && status == 0; slot++) {
                int next_state, step;
                if (is_edge_state) {
                    int edge = state - node_count;
                    if (!terminal_is_active(terminals, edge, slot, edge_mask, terminal_mask, node_mask)) continue;
                    next_state = terminals->terminal_nodes[slot];
                    step = (int)edge_count + slot;
                } else {
                    int neighbor = adjacency->neighbors[slot], index = adjacency->edge_indices[slot];
                    if (has_multi_terminal_edges && neighbor < 0) {
                        int edge = -1 - neighbor;
                        if (!edge_joins[edge] || !terminal_is_active(terminals, edge, index, edge_mask, terminal_mask, node_mask)) continue;
                        next_state = node_count + edge;
                        step = (int)edge_count + index;
                    } else {
                        if (two_terminal_edges && two_terminal_edges[index]) continue;
                        if ((node_mask && node_mask[neighbor]) || neighbor == state) continue;
                        next_state = neighbor;
                        step = index;
                    }
                }
                if (depth[next_state] == -1) {
                    depth[next_state] = depth[state] + 1;
                    parent[next_state] = state;
                    parent_step[next_state] = step;
                    queue[tail++] = next_state;
                } else if (step != parent_step[state] && !step_seen[step]) {
                    step_seen[step] = 1;
                    status = append_closed_cycle(graph, parent, depth, path_a, path_b, state, next_state, result,
                                                 has_multi_terminal_edges);
                }
            }
        }
    }
    free(step_seen); free(scratch); free(edge_joins);
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
    const uint8_t *two_terminal_edges = two_terminal_edge_mask(&terminal_mask, edge_mask);
    int status = graph->terminals
        ? fundamental_cycles(graph, two_terminal_edges, edge_mask, terminal_mask.terminals, node_mask, result, 1)
        : fundamental_cycles(graph, two_terminal_edges, NULL, NULL, node_mask, result, 0);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
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
