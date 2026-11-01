/* Breadth-first order and edge-simple paths over the adjacency list. */
#ifndef PYGRAPHC_CORE_TRAVERSAL_H
#define PYGRAPHC_CORE_TRAVERSAL_H

#include "../models.h"

/*
 * Breadth-first order from source into queue; returns the number of nodes
 * visited. A two-terminal edge leads to its neighbor, an edge with three or
 * more terminals is entered through the terminal of its slot and, the first
 * time, leads to the nodes of all its active exit terminals (directed:
 * out-terminals). visited holds node_count zeroed ints, queue node_count ints;
 * expanded holds edge_count zeroed bytes and is needed only with a terminal list.
 */
PYGRAPHC_INTERNAL int breadth_first_order(const AdjacencyList *adjacency, const TerminalList *terminals, int directed,
                                          int source, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                          const uint8_t *node_mask, int *visited, int *queue, uint8_t *expanded);

/* Receives each path found as its edge indices in order; a negative return stops the search with that status. */
typedef int (*EdgePathVisitor)(void *visitor_state, const int *path_edges, int length);

/*
 * Every edge-simple path (no edge twice) from source to a node with is_target
 * set, depth-first, with at most cutoff edges (cutoff <= 0: up to 2 * edge_count).
 * node_simple forbids entering a node twice, ignore_self_loops never traverses a
 * self-loop. A path is reported when it reaches a target, before node_simple
 * decides whether the target can be entered. Returns 0, -1 with MemoryError set,
 * or the first negative return of visit. An undirected self-loop occupies two
 * consecutive CSR slots and is traversed through the first only. Defined here, so
 * the caller's visitor inlines into the loop and the scratch, allocated next to
 * the loop, is known not to alias.
 */
static inline __attribute__((always_inline)) int all_edge_paths(
    const AdjacencyList *adjacency, int node_count, Py_ssize_t edge_count, int source, const uint8_t *is_target, int cutoff,
    const uint8_t *edge_mask, const uint8_t *node_mask, int node_simple, int ignore_self_loops, EdgePathVisitor visit,
    void *visitor_state) {
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

#endif
