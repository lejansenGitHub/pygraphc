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

int build_adjacency(int node_count, EdgeList *edges, AdjacencyList *adjacency, int directed) {
    Py_ssize_t edge_count = edges->edge_count;
    Py_ssize_t slot_count = directed ? edge_count : 2 * edge_count;
    adjacency->offset = (int *)calloc((size_t)(node_count + 1), sizeof(int));
    adjacency->neighbors = slot_count > 0 ? (int *)malloc((size_t)slot_count * sizeof(int)) : NULL;
    adjacency->edge_indices = slot_count > 0 ? (int *)malloc((size_t)slot_count * sizeof(int)) : NULL;
    adjacency->reverse_offset = NULL;
    adjacency->reverse_neighbors = NULL;
    adjacency->reverse_edge_indices = NULL;
    if (!adjacency->offset || (slot_count > 0 && (!adjacency->neighbors || !adjacency->edge_indices))) {
        free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
        PyErr_NoMemory();
        return -1;
    }

    /* Count degrees for forward CSR */
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        adjacency->offset[EDGE_FIRST_ENDPOINT(edges, edge_index) + 1]++;
        if (!directed)
            adjacency->offset[EDGE_SECOND_ENDPOINT(edges, edge_index) + 1]++;
    }
    for (int node = 1; node <= node_count; node++) adjacency->offset[node] += adjacency->offset[node - 1];

    /* Fill forward CSR */
    if (slot_count > 0) {
        int *next_slot = (int *)malloc((size_t)node_count * sizeof(int));
        if (!next_slot) {
            free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(next_slot, adjacency->offset, (size_t)node_count * sizeof(int));
        for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
            int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index), second_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
            adjacency->neighbors[next_slot[first_node]] = second_node; adjacency->edge_indices[next_slot[first_node]++] = (int)edge_index;
            if (!directed) {
                adjacency->neighbors[next_slot[second_node]] = first_node; adjacency->edge_indices[next_slot[second_node]++] = (int)edge_index;
            }
        }
        free(next_slot);
    }

    /* Build reverse CSR for directed graphs */
    if (directed && edge_count > 0) {
        adjacency->reverse_offset = (int *)calloc((size_t)(node_count + 1), sizeof(int));
        adjacency->reverse_neighbors = (int *)malloc((size_t)edge_count * sizeof(int));
        adjacency->reverse_edge_indices = (int *)malloc((size_t)edge_count * sizeof(int));
        if (!adjacency->reverse_offset || !adjacency->reverse_neighbors || !adjacency->reverse_edge_indices) {
            free(adjacency->reverse_offset); free(adjacency->reverse_neighbors); free(adjacency->reverse_edge_indices);
            adjacency->reverse_offset = NULL; adjacency->reverse_neighbors = NULL; adjacency->reverse_edge_indices = NULL;
            free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++)
            adjacency->reverse_offset[EDGE_SECOND_ENDPOINT(edges, edge_index) + 1]++;
        for (int node = 1; node <= node_count; node++) adjacency->reverse_offset[node] += adjacency->reverse_offset[node - 1];

        int *next_slot = (int *)malloc((size_t)node_count * sizeof(int));
        if (!next_slot) {
            free(adjacency->reverse_offset); free(adjacency->reverse_neighbors); free(adjacency->reverse_edge_indices);
            adjacency->reverse_offset = NULL; adjacency->reverse_neighbors = NULL; adjacency->reverse_edge_indices = NULL;
            free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(next_slot, adjacency->reverse_offset, (size_t)node_count * sizeof(int));
        for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
            int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index), second_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
            adjacency->reverse_neighbors[next_slot[second_node]] = first_node; adjacency->reverse_edge_indices[next_slot[second_node]++] = (int)edge_index;
        }
        free(next_slot);
    }

    return 0;
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
    if (graph->has_adjacency) free_adjacency(&graph->adjacency);
    if (graph->ports) { free_port_list(graph->ports); free(graph->ports); }
    free_edges(&graph->node_ids.edges);
    if (graph->node_ids.node_index_by_node_id.keys) intmap_free(&graph->node_ids.node_index_by_node_id);
    Py_DECREF(graph->node_ids.node_ids_fast);
    free(graph);
}

void free_port_list(PortList *ports) {
    free(ports->port_offsets);
    free(ports->port_nodes);
    free(ports->out_port_offsets);
}

void free_weights(WeightList *weights) {
    if (weights->owns_memory) free(weights->values);
}
