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
