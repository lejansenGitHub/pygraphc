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

/* Flags per terminal of a multi-terminal edge, fixed for one call: which terminals a path enters and leaves by. */
enum { ENTRY_TERMINAL = 1, EXIT_TERMINAL = 2, SHARES_ITS_NODE = 4 };

/*
 * Flag the terminals of every multi-terminal edge a path steps through (D28) into
 * flags (terminal_count zeroed bytes). A path enters at an active terminal and
 * leaves at another (directed: an in-terminal to an out-terminal); terminals of one
 * edge at one node lead to the same node, so only the first active one per node and
 * side enters or leaves, and SHARES_ITS_NODE marks a first terminal with another
 * active one at its node, the step back to it. O(terminals). Returns 0, or -1 with
 * MemoryError set.
 */
PYGRAPHC_INTERNAL int flag_path_terminals(const GraphContext *graph, const uint8_t *edge_mask,
                                          const uint8_t *terminal_mask, const uint8_t *node_mask, uint8_t *flags);

/* Receives each path found as its edge indices in order; a negative return stops the search with that status. */
typedef int (*EdgePathVisitor)(void *visitor_state, const int *path_edges, int length);

typedef struct {
    int source, cutoff, node_simple, ignore_self_loops;
    const uint8_t *is_target, *two_terminal_edge_mask, *node_mask;
    const uint8_t *terminal_flags;   /* flag_path_terminals, with multi-terminal edges only */
} EdgePathQuery;

/*
 * Every edge-simple path (no edge twice) from source to a node with is_target
 * set, depth-first, with at most cutoff edges (cutoff <= 0: up to 2 * edge_count).
 * node_simple forbids entering a node twice, ignore_self_loops never traverses a
 * self-loop. A path is reported when it reaches a target, before node_simple
 * decides whether the target can be entered. Returns 0, -1 with MemoryError set,
 * or the first negative return of visit. An undirected self-loop occupies two
 * consecutive CSR slots and is traversed through the first only.
 *
 * Instantiated per value of has_multi_terminal_edges (D17): with multi-terminal
 * edges a slot entering such an edge first walks the edge's exit terminals, the
 * edge staying used until they are done. Defined here, so the caller's visitor
 * inlines into the loop and the scratch, allocated next to the loop, is known not
 * to alias.
 */
static inline __attribute__((always_inline)) int all_edge_paths(const GraphContext *graph, const EdgePathQuery *query,
                                                                EdgePathVisitor visit, void *visitor_state,
                                                                const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    int max_depth = query->cutoff > 0 ? query->cutoff : (int)(2 * edge_count);
    const uint8_t *edge_mask = query->two_terminal_edge_mask, *node_mask = query->node_mask, *is_target = query->is_target;
    const uint8_t *terminal_flags = query->terminal_flags;
    const int ignore_self_loops = query->ignore_self_loops, directed = graph->directed;

    uint8_t *visited_edges = (uint8_t *)calloc((size_t)edge_count, sizeof(uint8_t));
    uint8_t *visited_nodes = query->node_simple ? (uint8_t *)calloc((size_t)node_count, sizeof(uint8_t)) : NULL;
    int *stack_node = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *stack_slot = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *path_edges = (int *)malloc((size_t)max_depth * sizeof(int));
    int *stack_edge = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int *stack_entry = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int *stack_exit = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int status = 0;
    if (!visited_edges || !stack_node || !stack_slot || !path_edges || (query->node_simple && !visited_nodes) ||
        (has_multi_terminal_edges && (!stack_edge || !stack_entry || !stack_exit))) {
        PyErr_NoMemory();
        status = -1;
        goto done;
    }
    if (visited_nodes) visited_nodes[query->source] = 1;
    int stack_top = 0;
    stack_node[0] = query->source;
    stack_slot[0] = adjacency->offset[query->source];
    if (has_multi_terminal_edges) stack_edge[0] = -1;
    while (stack_top >= 0) {
        int node = stack_node[stack_top];
        int edge_index, neighbor, in_multi_terminal_edge = 0;
        if (has_multi_terminal_edges && stack_edge[stack_top] >= 0) {
            int edge = stack_edge[stack_top], entry = stack_entry[stack_top], end = terminals->terminal_offsets[edge + 1];
            int exit = -1;
            while (stack_exit[stack_top] < end) {
                int terminal = stack_exit[stack_top]++;
                if ((terminal_flags[terminal] & EXIT_TERMINAL) &&
                    (terminal != entry || (terminal_flags[entry] & SHARES_ITS_NODE))) {
                    exit = terminal;
                    break;
                }
            }
            if (exit < 0) {
                visited_edges[edge] = 0;
                stack_edge[stack_top] = -1;
                continue;
            }
            in_multi_terminal_edge = 1;
            edge_index = edge;
            neighbor = terminals->terminal_nodes[exit];
            if (neighbor == node && ignore_self_loops) continue;
            goto step;
        }
        if (stack_slot[stack_top] < adjacency->offset[node + 1]) {
            int slot = stack_slot[stack_top]++;
            edge_index = adjacency->edge_indices[slot];
            neighbor = adjacency->neighbors[slot];
            if (has_multi_terminal_edges && neighbor < 0) {
                int edge = -1 - neighbor;
                if (visited_edges[edge] || !(terminal_flags[edge_index] & ENTRY_TERMINAL)) continue;
                visited_edges[edge] = 1;
                stack_edge[stack_top] = edge;
                stack_entry[stack_top] = edge_index;
                stack_exit[stack_top] = directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
                continue;
            }
            if (neighbor == node) {
                if (ignore_self_loops) continue;
                if (slot > adjacency->offset[node] && adjacency->edge_indices[slot - 1] == edge_index &&
                    (!has_multi_terminal_edges || adjacency->neighbors[slot - 1] >= 0)) continue;
            }
            if (visited_edges[edge_index]) continue;
            if (edge_mask && edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;
            visited_edges[edge_index] = 1;
        step:
            path_edges[stack_top] = edge_index;
            if (is_target[neighbor]) {
                int visit_status = visit(visitor_state, path_edges, stack_top + 1);
                if (visit_status < 0) { status = visit_status; break; }
            }
            if (visited_nodes && visited_nodes[neighbor]) {
                if (!in_multi_terminal_edge) visited_edges[edge_index] = 0;
                continue;
            }
            if (visited_nodes) visited_nodes[neighbor] = 1;
            if (stack_top + 1 < max_depth) {
                stack_top++;
                stack_node[stack_top] = neighbor;
                stack_slot[stack_top] = adjacency->offset[neighbor];
                if (has_multi_terminal_edges) stack_edge[stack_top] = -1;
            } else {
                if (!in_multi_terminal_edge) visited_edges[edge_index] = 0;
                if (visited_nodes) visited_nodes[neighbor] = 0;
            }
        } else {
            if (stack_top > 0 && visited_nodes) visited_nodes[stack_node[stack_top]] = 0;
            stack_top--;
            if (stack_top >= 0 && !(has_multi_terminal_edges && stack_edge[stack_top] >= 0))
                visited_edges[path_edges[stack_top]] = 0;
        }
    }
done:
    free(visited_edges); free(visited_nodes);
    free(stack_node); free(stack_slot); free(path_edges);
    free(stack_edge); free(stack_entry); free(stack_exit);
    return status;
}

#endif
