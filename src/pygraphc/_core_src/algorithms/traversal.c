/* Breadth-first order and edge-simple paths over the adjacency list. */
#include "traversal.h"
#include "terminals.h"

/*
 * Enter an edge with three or more terminals through the terminal of its slot and, the
 * first time that terminal is active, add the nodes of all its active exit terminals
 * (directed: out-terminals) to the queue. Returns the new queue tail.
 */
static inline int expand_multi_terminal_edge(const AdjacencyList *adjacency, const TerminalList *terminals, int directed,
                                             int slot, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                             const uint8_t *node_mask, int *visited, int *queue, int tail,
                                             uint8_t *expanded) {
    int edge = -1 - adjacency->neighbors[slot];
    if (expanded[edge] || !terminal_is_active(terminals, edge, adjacency->edge_indices[slot], edge_mask, terminal_mask, node_mask))
        return tail;
    expanded[edge] = 1;
    int first_exit = directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
    for (int exit = first_exit; exit < terminals->terminal_offsets[edge + 1]; exit++) {
        if (!terminal_is_active(terminals, edge, exit, edge_mask, terminal_mask, node_mask)) continue;
        int reached = terminals->terminal_nodes[exit];
        if (!visited[reached]) { visited[reached] = 1; queue[tail++] = reached; }
    }
    return tail;
}

/*
 * The loop of breadth_first_order, instantiated for both values of
 * has_multi_terminal_edges, so on a graph of two-terminal edges it runs without
 * the branch for multi-terminal entries.
 */
static inline __attribute__((always_inline)) int breadth_first_order_of(
    const AdjacencyList *adjacency, const TerminalList *terminals, int directed, int source, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, int *visited, int *queue, uint8_t *expanded,
    const int has_multi_terminal_edges) {
    int head = 0, tail = 0;
    queue[tail++] = source; visited[source] = 1;
    while (head < tail) {
        int node = queue[head++];
        for (int slot = adjacency->offset[node]; slot < adjacency->offset[node+1]; slot++) {
            if (has_multi_terminal_edges && adjacency->neighbors[slot] < 0) {
                tail = expand_multi_terminal_edge(adjacency, terminals, directed, slot, edge_mask, terminal_mask, node_mask,
                                                  visited, queue, tail, expanded);
                continue;
            }
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            int neighbor = adjacency->neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
        }
    }
    return tail;
}

int breadth_first_order(const AdjacencyList *adjacency, const TerminalList *terminals, int directed, int source,
                        const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask, int *visited,
                        int *queue, uint8_t *expanded) {
    return terminals
        ? breadth_first_order_of(adjacency, terminals, directed, source, edge_mask, terminal_mask, node_mask, visited, queue,
                                 expanded, 1)
        : breadth_first_order_of(adjacency, NULL, directed, source, edge_mask, NULL, node_mask, visited, queue, NULL, 0);
}
