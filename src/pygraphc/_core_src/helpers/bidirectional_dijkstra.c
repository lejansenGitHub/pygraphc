#include "bidirectional_dijkstra.h"
#include "min_heap.h"

/* Walk the two predecessor chains from the meeting edge into a single path.
   meet_forward and meet_backward are the endpoints of the meeting edge; they
   are the same node when the searches met on a node rather than an edge. */
static int bidirectional_build_path(const int *predecessor_forward, const int *predecessor_backward,
                                    int meet_forward, int meet_backward, BidirectionalPath *out) {
    int forward_length = 0;
    for (int node = meet_forward; node != -1; node = predecessor_forward[node]) forward_length++;
    int backward_start = (meet_forward == meet_backward) ? predecessor_backward[meet_backward]
                                                         : meet_backward;
    int backward_length = 0;
    for (int node = backward_start; node != -1; node = predecessor_backward[node]) backward_length++;

    int total = forward_length + backward_length;
    int *path = (int *)malloc((size_t)total * sizeof(int));
    if (!path) { PyErr_NoMemory(); return -1; }
    int node = meet_forward;
    for (int position = forward_length - 1; position >= 0; position--) { path[position] = node; node = predecessor_forward[node]; }
    node = backward_start;
    for (int position = forward_length; position < total; position++) { path[position] = node; node = predecessor_backward[node]; }

    out->path = path;
    out->path_length = total;
    return 0;
}

/* Whether a search has a valid tentative distance for a node. distance entries are
   left uninitialised, so this is what tells a fresh node from a reached one. */
static inline int bidirectional_reached(const int *heap_position, const uint8_t *settled, int node) {
    return heap_position[node] != -1 || settled[node];
}

/* Bidirectional Dijkstra over the CSR adjacency. The forward search follows
   outgoing edges from source, the backward search incoming edges into target;
   undirected graphs use the same adjacency for both. The frontier with the
   smaller top key is expanded next, the best source-target weight seen over
   relaxed edges is tracked as the searches proceed, and the search stops once
   a node has been settled by both. Returns -1 on error. */
int bidirectional_dijkstra(int node_count, const AdjacencyList *adjacency, const double *edge_weights,
                           const uint8_t *edge_mask, const uint8_t *node_mask,
                           int directed, int source, int target, BidirectionalPath *out) {
    out->weight = HUGE_VAL;
    out->path = NULL;
    out->path_length = 0;

    const int *backward_offset = directed ? adjacency->reverse_offset : adjacency->offset;
    const int *backward_neighbors = directed ? adjacency->reverse_neighbors : adjacency->neighbors;
    const int *backward_edge_indices = directed ? adjacency->reverse_edge_indices : adjacency->edge_indices;
    if (!backward_offset) return 0;  /* directed graph without edges */

    double *distance = (double *)malloc(2 * (size_t)node_count * sizeof(double));
    int *predecessor = (int *)malloc(2 * (size_t)node_count * sizeof(int));
    int *heap = (int *)malloc(2 * (size_t)node_count * sizeof(int));
    int *heap_position = (int *)malloc(2 * (size_t)node_count * sizeof(int));
    uint8_t *settled = (uint8_t *)calloc(2 * (size_t)node_count, 1);
    if (!distance || !predecessor || !heap || !heap_position || !settled) {
        free(distance); free(predecessor); free(heap); free(heap_position); free(settled);
        PyErr_NoMemory();
        return -1;
    }
    memset(heap_position, -1, 2 * (size_t)node_count * sizeof(int));

    double *distance_forward = distance, *distance_backward = distance + node_count;
    int *predecessor_forward = predecessor, *predecessor_backward = predecessor + node_count;
    int *heap_position_forward = heap_position, *heap_position_backward = heap_position + node_count;
    uint8_t *settled_forward = settled, *settled_backward = settled + node_count;

    MinHeap heap_forward = {distance_forward, heap, heap_position_forward, 1};
    MinHeap heap_backward = {distance_backward, heap + node_count, heap_position_backward, 1};
    distance_forward[source] = 0.0; predecessor_forward[source] = -1;
    heap_forward.nodes[0] = source; heap_position_forward[source] = 0;
    distance_backward[target] = 0.0; predecessor_backward[target] = -1;
    heap_backward.nodes[0] = target; heap_position_backward[target] = 0;

    double best_weight = HUGE_VAL;
    int meet_forward = -1, meet_backward = -1;

    while (heap_forward.size > 0 && heap_backward.size > 0) {
        int expand_forward = distance_forward[heap_forward.nodes[0]]
                          <= distance_backward[heap_backward.nodes[0]];
        MinHeap *frontier = expand_forward ? &heap_forward : &heap_backward;
        double *distance_near = expand_forward ? distance_forward : distance_backward;
        const double *distance_far = expand_forward ? distance_backward : distance_forward;
        int *predecessor_near = expand_forward ? predecessor_forward : predecessor_backward;
        uint8_t *settled_near = expand_forward ? settled_forward : settled_backward;
        const uint8_t *settled_far = expand_forward ? settled_backward : settled_forward;
        const int *heap_position_near = expand_forward ? heap_position_forward : heap_position_backward;
        const int *heap_position_far = expand_forward ? heap_position_backward : heap_position_forward;
        const int *offset = expand_forward ? adjacency->offset : backward_offset;
        const int *neighbors = expand_forward ? adjacency->neighbors : backward_neighbors;
        const int *edge_indices = expand_forward ? adjacency->edge_indices : backward_edge_indices;

        int node = min_heap_pop(frontier);
        settled_near[node] = 1;
        if (bidirectional_reached(heap_position_far, settled_far, node) && distance_near[node] + distance_far[node] < best_weight) {
            best_weight = distance_near[node] + distance_far[node];
            meet_forward = node; meet_backward = node;
        }
        if (settled_far[node]) break;

        for (int slot = offset[node]; slot < offset[node + 1]; slot++) {
            if (edge_mask && edge_mask[edge_indices[slot]]) continue;
            int neighbor = neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            double reach = distance_near[node] + edge_weights[edge_indices[slot]];
            if (bidirectional_reached(heap_position_far, settled_far, neighbor) && reach + distance_far[neighbor] < best_weight) {
                best_weight = reach + distance_far[neighbor];
                meet_forward = expand_forward ? node : neighbor;
                meet_backward = expand_forward ? neighbor : node;
            }
            if (!bidirectional_reached(heap_position_near, settled_near, neighbor)) {
                distance_near[neighbor] = reach;
                predecessor_near[neighbor] = node;
                int heap_slot = frontier->size++;
                frontier->nodes[heap_slot] = neighbor;
                frontier->heap_position[neighbor] = heap_slot;
                min_heap_sift_up(frontier, heap_slot);
            } else if (!settled_near[neighbor] && reach < distance_near[neighbor]) {
                distance_near[neighbor] = reach;
                predecessor_near[neighbor] = node;
                min_heap_sift_up(frontier, frontier->heap_position[neighbor]);
            }
        }
    }

    int status = 0;
    if (best_weight < HUGE_VAL) {
        out->weight = best_weight;
        status = bidirectional_build_path(predecessor_forward, predecessor_backward, meet_forward, meet_backward, out);
    }

    free(distance); free(predecessor); free(heap); free(heap_position); free(settled);
    return status;
}

/* Turn the node indices of a bidirectional result into a list of node IDs. */
PyObject *bidirectional_path_to_list(const BidirectionalPath *bidirectional_path, const NodeIdContext *context) {
    PyObject *path = PyList_New(bidirectional_path->path_length);
    if (!path) return NULL;
    for (int position = 0; position < bidirectional_path->path_length; position++) {
        PyObject *item = context->node_id_items[bidirectional_path->path[position]];
        Py_INCREF(item);
        PyList_SET_ITEM(path, position, item);
    }
    return path;
}
