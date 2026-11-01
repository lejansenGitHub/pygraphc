#include "toposort.h"

/* ── Topological sort (Kahn's algorithm over the search states of a directed graph) ── */

/*
 * The in-degree of every state: a node counts its active incoming two-terminal
 * edges and the active out-terminals of multi-terminal edges that carry
 * something, an edge state its active in-terminals. An edge without an active
 * in-terminal transmits nothing (masking case 9), so its out-terminals count
 * for nobody.
 */
static inline __attribute__((always_inline)) void count_in_degrees(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, int *in_degree, const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    if (has_multi_terminal_edges) {
        for (Py_ssize_t edge = 0; edge < graph->node_ids.edges.edge_count; edge++) {
            int active_in_terminals = 0;
            for (int terminal = terminals->terminal_offsets[edge]; terminal < terminals->out_terminal_offsets[edge]; terminal++)
                active_in_terminals += terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask);
            in_degree[node_count + edge] = active_in_terminals;
        }
    }
    for (int node = 0; node < node_count; node++) {
        in_degree[node] = 0;
        if (node_mask && node_mask[node]) continue;
        for (int slot = adjacency->reverse_offset[node]; slot < adjacency->reverse_offset[node + 1]; slot++) {
            int predecessor = adjacency->reverse_neighbors[slot], index = adjacency->reverse_edge_indices[slot];
            if (has_multi_terminal_edges && predecessor < 0) {
                int edge = -1 - predecessor;
                if (in_degree[node_count + edge] > 0 && terminal_is_active(terminals, edge, index, edge_mask, terminal_mask, node_mask))
                    in_degree[node]++;
                continue;
            }
            if (two_terminal_edge_mask && two_terminal_edge_mask[index]) continue;
            if (node_mask && node_mask[predecessor]) continue;
            in_degree[node]++;
        }
    }
}

/* Lower a state's in-degree and queue it when it reaches zero. */
static inline __attribute__((always_inline)) void release_successor(int *in_degree, int *queue, int *queue_tail, int state) {
    if (--in_degree[state] == 0) queue[(*queue_tail)++] = state;
}

/*
 * Kahn's algorithm over the states (nodes, and node_count + e for an edge with
 * three or more terminals): a two-terminal edge leads from its first to its
 * second endpoint, a multi-terminal edge from each active in-terminal to its
 * state and from there to each active out-terminal. A node's successors are
 * taken in descending slot order, which is the order graphs of two-terminal
 * edges have always been sorted in. Instantiated for both values of
 * has_multi_terminal_edges.
 */
static inline __attribute__((always_inline)) int topological_states(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, TopologicalOrder *result,
    const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)graph->node_ids.edges.edge_count : 0);
    int *in_degree = (int *)malloc((state_count > 0 ? state_count : 1) * sizeof(int));
    int *queue = (int *)malloc((state_count > 0 ? state_count : 1) * sizeof(int));
    if (!in_degree || !queue) { free(in_degree); free(queue); PyErr_NoMemory(); return -1; }

    int queue_head = 0, queue_tail = 0, node_total = 0, nodes_in_order = 0;
    if (graph->has_adjacency) {
        count_in_degrees(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, in_degree, has_multi_terminal_edges);
    } else {
        for (int node = 0; node < node_count; node++) in_degree[node] = 0;
    }
    for (int node = 0; node < node_count; node++) {
        if (node_mask && node_mask[node]) continue;
        node_total++;
        if (in_degree[node] == 0) queue[queue_tail++] = node;
    }
    while (queue_head < queue_tail && graph->has_adjacency) {
        int state = queue[queue_head++];
        if (has_multi_terminal_edges && state >= node_count) {
            int edge = state - node_count;
            for (int terminal = terminals->out_terminal_offsets[edge]; terminal < terminals->terminal_offsets[edge + 1]; terminal++)
                if (terminal_is_active(terminals, edge, terminal, edge_mask, terminal_mask, node_mask))
                    release_successor(in_degree, queue, &queue_tail, terminals->terminal_nodes[terminal]);
            continue;
        }
        nodes_in_order++;
        for (int slot = adjacency->offset[state + 1] - 1; slot >= adjacency->offset[state]; slot--) {
            int successor = adjacency->neighbors[slot], index = adjacency->edge_indices[slot];
            if (has_multi_terminal_edges && successor < 0) {
                int edge = -1 - successor;
                if (terminal_is_active(terminals, edge, index, edge_mask, terminal_mask, node_mask))
                    release_successor(in_degree, queue, &queue_tail, node_count + edge);
                continue;
            }
            if (two_terminal_edge_mask && two_terminal_edge_mask[index]) continue;
            if (node_mask && node_mask[successor]) continue;
            release_successor(in_degree, queue, &queue_tail, successor);
        }
    }
    if (!graph->has_adjacency) nodes_in_order = queue_tail;
    free(in_degree);
    result->states = queue;
    result->state_count = queue_tail;
    result->node_count = node_total;
    result->nodes_in_order = nodes_in_order;
    return 0;
}

int topological_order(const GraphContext *graph, const uint8_t *edge_mask, const TerminalMask *terminal_mask,
                      const uint8_t *node_mask, TopologicalOrder *result) {
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(terminal_mask, edge_mask);
    return graph->terminals
        ? topological_states(graph, traversal_edge_mask, edge_mask, terminal_mask->terminals, node_mask, result, 1)
        : topological_states(graph, traversal_edge_mask, NULL, NULL, node_mask, result, 0);
}
