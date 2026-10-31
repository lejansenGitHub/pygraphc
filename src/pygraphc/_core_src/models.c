/* Init and free helpers of the data models. */
#include "models.h"

void free_edges(EdgeList *edges) {
    if (edges->owns_memory) {
        free(edges->first_endpoints);
        free(edges->second_endpoints);
    }
}

int intmap_init(IntMap *map, int expected_count) {
    int capacity = 16;
    while (capacity < expected_count * 2) capacity <<= 1;
    map->capacity = capacity;
    map->capacity_mask = capacity - 1;
    map->keys = (long *)malloc((size_t)capacity * sizeof(long));
    map->values = (int *)malloc((size_t)capacity * sizeof(int));
    if (!map->keys || !map->values) {
        free(map->keys); free(map->values);
        PyErr_NoMemory();
        return -1;
    }
    memset(map->keys, 0xFF, (size_t)capacity * sizeof(long)); /* -1 = empty sentinel */
    return 0;
}

void intmap_free(IntMap *map) {
    free(map->keys);
    free(map->values);
}

void node_id_context_free(NodeIdContext *context) {
    free_edges(&context->edges);
    if (context->node_index_by_node_id.keys) intmap_free(&context->node_index_by_node_id);
    Py_DECREF(context->node_ids_fast);
}

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

void free_adjacency(AdjacencyList *adjacency) {
    free(adjacency->offset);
    free(adjacency->neighbors);
    free(adjacency->edge_indices);
    free(adjacency->reverse_offset);
    free(adjacency->reverse_neighbors);
    free(adjacency->reverse_edge_indices);
}

void graph_context_destructor(PyObject *capsule) {
    GraphContext *graph = (GraphContext *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!graph) { PyErr_Clear(); return; }
    free_graph_context(graph);
}

void free_graph_context(GraphContext *graph) {
    if (graph->has_adjacency) free_adjacency(&graph->adjacency);
    if (graph->terminals) { free_terminal_list(graph->terminals); free(graph->terminals); }
    free_edges(&graph->node_ids.edges);
    if (graph->node_ids.node_index_by_node_id.keys) intmap_free(&graph->node_ids.node_index_by_node_id);
    Py_DECREF(graph->node_ids.node_ids_fast);
    free(graph);
}

void free_terminal_list(TerminalList *terminals) {
    free(terminals->terminal_offsets);
    free(terminals->terminal_nodes);
    free(terminals->out_terminal_offsets);
}

void free_weights(WeightList *weights) {
    if (weights->owns_memory) free(weights->values);
}
