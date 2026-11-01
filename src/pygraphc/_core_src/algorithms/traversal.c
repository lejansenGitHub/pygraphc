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

/*
 * An undirected self-loop occupies two consecutive CSR slots and is traversed
 * through the first only. Leaving a node unmarks the edge that entered it, and
 * with node_simple the node itself. The scratch is allocated here, next to the
 * loop, so the compiler knows the arrays do not alias.
 */
int all_edge_paths(const AdjacencyList *adjacency, int node_count, Py_ssize_t edge_count, int source,
                   const uint8_t *is_target, int cutoff, const uint8_t *edge_mask, const uint8_t *node_mask,
                   int node_simple, int ignore_self_loops, EdgePathVisitor visit, void *visitor_state) {
    int max_depth = cutoff > 0 ? cutoff : (int)(2 * edge_count);
    uint8_t *visited_edges = (uint8_t *)calloc((size_t)edge_count, sizeof(uint8_t));
    uint8_t *visited_nodes = node_simple ? (uint8_t *)calloc((size_t)node_count, sizeof(uint8_t)) : NULL;
    int *stack_node = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *stack_slot = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *path_edges = (int *)malloc((size_t)max_depth * sizeof(int));
    int status = 0;
    if (!visited_edges || !stack_node || !stack_slot || !path_edges || (node_simple && !visited_nodes)) {
        PyErr_NoMemory();
        status = -1;
        goto done;
    }
    if (visited_nodes) visited_nodes[source] = 1;
    int stack_top = 0;
    stack_node[0] = source;
    stack_slot[0] = adjacency->offset[source];
    while (stack_top >= 0) {
        int node = stack_node[stack_top];
        if (stack_slot[stack_top] < adjacency->offset[node + 1]) {
            int slot = stack_slot[stack_top]++;
            int edge_index = adjacency->edge_indices[slot];
            int neighbor = adjacency->neighbors[slot];
            if (neighbor == node && ignore_self_loops) continue;
            if (neighbor == node && slot > adjacency->offset[node] && adjacency->edge_indices[slot - 1] == edge_index) continue;
            if (visited_edges[edge_index]) continue;
            if (edge_mask && edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;
            visited_edges[edge_index] = 1;
            path_edges[stack_top] = edge_index;
            if (is_target[neighbor] && (status = visit(visitor_state, path_edges, stack_top + 1)) < 0) goto done;
            if (visited_nodes && visited_nodes[neighbor]) {
                visited_edges[edge_index] = 0;
                continue;
            }
            if (visited_nodes) visited_nodes[neighbor] = 1;
            if (stack_top + 1 < max_depth) {
                stack_top++;
                stack_node[stack_top] = neighbor;
                stack_slot[stack_top] = adjacency->offset[neighbor];
            } else {
                visited_edges[edge_index] = 0;
                if (visited_nodes) visited_nodes[neighbor] = 0;
            }
        } else {
            if (stack_top > 0 && visited_nodes) visited_nodes[stack_node[stack_top]] = 0;
            stack_top--;
            if (stack_top >= 0) visited_edges[path_edges[stack_top]] = 0;
        }
    }
done:
    free(visited_edges); free(visited_nodes);
    free(stack_node); free(stack_slot); free(path_edges);
    return status;
}
