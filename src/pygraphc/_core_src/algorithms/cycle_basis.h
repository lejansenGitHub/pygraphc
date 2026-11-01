/* A fundamental cycle basis over the incidence graph of an undirected graph. */
#ifndef PYGRAPHC_CORE_CYCLE_BASIS_H
#define PYGRAPHC_CORE_CYCLE_BASIS_H

#include "terminals.h"

/* Receives each cycle as its node indices in order; a negative return stops the search with that status. */
typedef int (*CycleVisitor)(void *visitor_state, const int *nodes, int length);

/* Whether a search state is a node; without multi-terminal edges every state is, and the check compiles away. */
static inline __attribute__((always_inline)) int is_node_state(int state, int node_count, const int has_multi_terminal_edges) {
    return !has_multi_terminal_edges || state < node_count;
}

/*
 * Report the cycle closed by a non-tree step between two discovered states: both
 * walk up to their lowest common ancestor, keeping only node states, and the
 * second walk is appended reversed, so the cycle lists its nodes in order. path
 * and reversed_path hold state_count entries; a cycle visits each state once.
 */
static inline __attribute__((always_inline)) int report_closed_cycle(
    const int *parent, const int *depth, int *path, int *reversed_path, int walker_a, int walker_b, int node_count,
    CycleVisitor visit, void *visitor_state, const int has_multi_terminal_edges) {
    int path_length = 0, reversed_length = 0;
    while (depth[walker_a] > depth[walker_b]) {
        if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path[path_length++] = walker_a;
        walker_a = parent[walker_a];
    }
    while (depth[walker_b] > depth[walker_a]) {
        if (is_node_state(walker_b, node_count, has_multi_terminal_edges)) reversed_path[reversed_length++] = walker_b;
        walker_b = parent[walker_b];
    }
    while (walker_a != walker_b) {
        if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path[path_length++] = walker_a;
        if (is_node_state(walker_b, node_count, has_multi_terminal_edges)) reversed_path[reversed_length++] = walker_b;
        walker_a = parent[walker_a]; walker_b = parent[walker_b];
    }
    if (is_node_state(walker_a, node_count, has_multi_terminal_edges)) path[path_length++] = walker_a;
    while (reversed_length > 0) path[path_length++] = reversed_path[--reversed_length];
    return visit(visitor_state, path, path_length);
}

/* The basis, instantiated for both values of has_multi_terminal_edges. */
static inline __attribute__((always_inline)) int fundamental_cycles(
    const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, CycleVisitor visit, void *visitor_state, const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    const EdgeList *edges = &graph->node_ids.edges;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = edges->edge_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)edge_count : 0);
    size_t step_count = (size_t)edge_count + (has_multi_terminal_edges ? (size_t)terminals->terminal_count : 0);

    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        if (has_multi_terminal_edges && terminals->terminal_offsets[edge + 1] - terminals->terminal_offsets[edge] != 2) continue;
        if (two_terminal_edges && two_terminal_edges[edge]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge);
        if (first_node != EDGE_SECOND_ENDPOINT(edges, edge) || (node_mask && node_mask[first_node])) continue;
        int status = visit(visitor_state, &first_node, 1);
        if (status < 0) return status;
    }

    uint8_t *step_seen = (uint8_t *)calloc(step_count > 0 ? step_count : 1, 1);
    int *scratch = (int *)malloc(6 * (state_count > 0 ? state_count : 1) * sizeof(int));
    uint8_t *edge_joins = has_multi_terminal_edges ? (uint8_t *)calloc(edge_count > 0 ? (size_t)edge_count : 1, 1) : NULL;
    if (!step_seen || !scratch || (has_multi_terminal_edges && !edge_joins)) {
        free(step_seen); free(scratch); free(edge_joins); PyErr_NoMemory(); return -1;
    }
    int *parent = scratch, *depth = scratch + state_count, *queue = scratch + 2 * state_count;
    int *parent_step = scratch + 3 * state_count, *path = scratch + 4 * state_count, *reversed_path = scratch + 5 * state_count;
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
                    status = report_closed_cycle(parent, depth, path, reversed_path, state, next_state, node_count, visit,
                                                 visitor_state, has_multi_terminal_edges);
                }
            }
        }
    }
    free(step_seen); free(scratch); free(edge_joins);
    return status;
}

/*
 * A fundamental cycle basis (D25): the states are the nodes and, on a graph with
 * multi-terminal edges, one state per edge with three or more terminals
 * (node_count + e), joined to the node of each of its active terminals when it has
 * at least two; a two-terminal edge joins its endpoints directly. Every step has an
 * id (a two-terminal edge its index, terminal t edge_count + t); a BFS tree over
 * the states leaves one non-tree step per independent cycle, so there are
 * terminals - nodes - edges + components cycles. A cycle holds the nodes of a
 * Berge cycle in order. A self-loop, and an edge with two terminals at one node,
 * give the cycle [node]. Returns 0, -1 with MemoryError set, or the first
 * negative return of visit. Defined here, so the caller's visitor inlines into
 * the search.
 */
static inline __attribute__((always_inline)) int fundamental_cycle_basis(
    const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, CycleVisitor visit, void *visitor_state) {
    return graph->terminals
        ? fundamental_cycles(graph, two_terminal_edges, edge_mask, terminal_mask, node_mask, visit, visitor_state, 1)
        : fundamental_cycles(graph, two_terminal_edges, NULL, NULL, node_mask, visit, visitor_state, 0);
}

#endif
