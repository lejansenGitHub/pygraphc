/* The longest path of a directed acyclic graph: topological order plus dynamic programming. */
#include "longest_path.h"
#include "toposort.h"

/*
 * Longest distances over a topological order of states. Entering a multi-terminal
 * edge through an in-terminal weighs that terminal's weight, 1 unweighted, and
 * leaving it through an out-terminal that terminal's weight, 0 unweighted; an edge
 * state relaxes its out-terminals only once an in-terminal reached it.
 * Instantiated for both values of has_multi_terminal_edges.
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

/*
 * The node states of the predecessor chain ending at end_node, from its start;
 * without predecessors end_node alone (nothing when it is -1). Returns -1 with
 * MemoryError set.
 */
static int write_path(const int *predecessor, int end_node, int node_count, int **path_out, int *path_length_out) {
    int path_length = 0;
    for (int state = end_node; state >= 0; state = predecessor ? predecessor[state] : -1) path_length += state < node_count;
    int *path = (int *)malloc((size_t)(path_length > 0 ? path_length : 1) * sizeof(int));
    if (!path) { PyErr_NoMemory(); return -1; }
    int position = path_length - 1;
    for (int state = end_node; state >= 0; state = predecessor ? predecessor[state] : -1)
        if (state < node_count) path[position--] = state;
    *path_out = path;
    *path_length_out = path_length;
    return 0;
}

int dag_longest_path(const GraphContext *graph, const double *weights, const uint8_t *edge_mask,
                     const TerminalMask *terminal_mask, const uint8_t *node_mask, int **path_out, int *path_length_out) {
    int node_count = graph->node_ids.node_count;
    size_t state_count = (size_t)node_count + (graph->terminals ? (size_t)graph->node_ids.edges.edge_count : 0);
    TopologicalOrder order = {NULL, 0, 0, 0};
    double *distance = (double *)malloc((state_count > 0 ? state_count : 1) * sizeof(double));
    int *predecessor = (int *)malloc((state_count > 0 ? state_count : 1) * sizeof(int));
    int status = -1;
    if (!distance || !predecessor) { PyErr_NoMemory(); goto done; }
    if (topological_order(graph, edge_mask, terminal_mask, node_mask, &order) < 0) goto done;
    if (order.nodes_in_order < order.node_count) {
        PyErr_SetString(PyExc_ValueError, "Graph contains a cycle - dag_longest_path is undefined");
        goto done;
    }
    for (size_t state = 0; state < state_count; state++) { distance[state] = 0.0; predecessor[state] = -1; }
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(terminal_mask, edge_mask);
    if (graph->terminals)
        relax_in_order(graph, traversal_edge_mask, edge_mask, terminal_mask->terminals, node_mask, weights, &order,
                       distance, predecessor, 1);
    else
        relax_in_order(graph, traversal_edge_mask, NULL, NULL, node_mask, weights, &order, distance, predecessor, 0);

    int best_node = -1, first_node = -1;
    double best_distance = -1.0;
    for (int position = 0; position < order.state_count; position++) {
        int state = order.states[position];
        if (state >= node_count) continue;
        if (first_node < 0) first_node = state;
        if (distance[state] > best_distance) { best_distance = distance[state]; best_node = state; }
    }
    status = (best_node < 0 || best_distance == 0.0)
        ? write_path(NULL, first_node, node_count, path_out, path_length_out)
        : write_path(predecessor, best_node, node_count, path_out, path_length_out);
done:
    free(distance); free(predecessor); free(order.states);
    return status;
}
