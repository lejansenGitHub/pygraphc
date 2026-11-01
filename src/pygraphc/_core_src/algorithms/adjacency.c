/* The adjacency list of a graph: CSR arrays built from its edge list. */
#include "adjacency.h"

/*
 * Count (fills == 0) or fill one direction of the adjacency list. Forward, a
 * two-terminal edge sits at its first endpoint (and, undirected, its second);
 * with a terminal list, an edge with three or more terminals sits as -1 - e at
 * each in-terminal (undirected: each terminal) with that terminal in
 * edge_indices, and a single-terminal edge nowhere. In reverse: at the second
 * endpoint and the out-terminals.
 */
static inline __attribute__((always_inline)) void place_slots(
    const EdgeList *edges, const TerminalList *terminals, int directed, int reverse, int *next_slot, int *neighbors,
    int *edge_indices, const int fills, const int has_terminal_list) {
    for (Py_ssize_t edge = 0; edge < edges->edge_count; edge++) {
        if (has_terminal_list) {
            int first_terminal = terminals->terminal_offsets[edge], end_terminal = terminals->terminal_offsets[edge + 1];
            if (end_terminal - first_terminal != 2) {
                if (end_terminal - first_terminal < 3) continue;
                int from_terminal = directed && reverse ? terminals->out_terminal_offsets[edge] : first_terminal;
                int to_terminal = directed && !reverse ? terminals->out_terminal_offsets[edge] : end_terminal;
                for (int terminal = from_terminal; terminal < to_terminal; terminal++) {
                    int at = terminals->terminal_nodes[terminal];
                    if (fills) { neighbors[next_slot[at]] = -1 - (int)edge; edge_indices[next_slot[at]] = terminal; }
                    next_slot[at]++;
                }
                continue;
            }
        }
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge), second_node = EDGE_SECOND_ENDPOINT(edges, edge);
        int at = reverse ? second_node : first_node, neighbor = reverse ? first_node : second_node;
        if (fills) { neighbors[next_slot[at]] = neighbor; edge_indices[next_slot[at]] = (int)edge; }
        next_slot[at]++;
        if (!directed) {
            if (fills) { neighbors[next_slot[second_node]] = first_node; edge_indices[next_slot[second_node]] = (int)edge; }
            next_slot[second_node]++;
        }
    }
}

/* One direction as CSR arrays; returns 0, or -1 with an exception set and nothing allocated. */
static inline __attribute__((always_inline)) int build_direction(
    int node_count, const EdgeList *edges, const TerminalList *terminals, int directed, int reverse, int **offset_out,
    int **neighbors_out, int **edge_indices_out, const int has_terminal_list) {
    int *offset = (int *)calloc((size_t)node_count + 1, sizeof(int));
    if (!offset) { PyErr_NoMemory(); return -1; }
    place_slots(edges, terminals, directed, reverse, offset + 1, NULL, NULL, 0, has_terminal_list);
    for (int node = 1; node <= node_count; node++) offset[node] += offset[node - 1];
    size_t slot_count = (size_t)offset[node_count];
    int *neighbors = (int *)malloc((slot_count > 0 ? slot_count : 1) * sizeof(int));
    int *edge_indices = (int *)malloc((slot_count > 0 ? slot_count : 1) * sizeof(int));
    int *next_slot = (int *)malloc(((size_t)node_count + 1) * sizeof(int));
    if (!neighbors || !edge_indices || !next_slot) {
        free(offset); free(neighbors); free(edge_indices); free(next_slot);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(next_slot, offset, (size_t)node_count * sizeof(int));
    place_slots(edges, terminals, directed, reverse, next_slot, neighbors, edge_indices, 1, has_terminal_list);
    free(next_slot);
    *offset_out = offset;
    *neighbors_out = neighbors;
    *edge_indices_out = edge_indices;
    return 0;
}

static inline __attribute__((always_inline)) int build_both_directions(
    int node_count, const EdgeList *edges, const TerminalList *terminals, AdjacencyList *adjacency, int directed,
    const int has_terminal_list) {
    adjacency->reverse_offset = NULL;
    adjacency->reverse_neighbors = NULL;
    adjacency->reverse_edge_indices = NULL;
    if (build_direction(node_count, edges, terminals, directed, 0, &adjacency->offset, &adjacency->neighbors,
                        &adjacency->edge_indices, has_terminal_list) < 0)
        return -1;
    if (directed && build_direction(node_count, edges, terminals, directed, 1, &adjacency->reverse_offset,
                                    &adjacency->reverse_neighbors, &adjacency->reverse_edge_indices, has_terminal_list) < 0) {
        free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
        return -1;
    }
    return 0;
}

int build_adjacency(int node_count, const EdgeList *edges, const TerminalList *terminals, AdjacencyList *adjacency,
                    int directed) {
    return terminals ? build_both_directions(node_count, edges, terminals, adjacency, directed, 1)
                     : build_both_directions(node_count, edges, NULL, adjacency, directed, 0);
}
