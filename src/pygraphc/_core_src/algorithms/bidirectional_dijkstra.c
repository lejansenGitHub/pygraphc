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

/* Relax the search state `to`, reached from `from` at distance `reach`, on the
   side being expanded, and record a better meeting with the other side. */
static inline __attribute__((always_inline)) void bidirectional_relax(
    MinHeap *frontier, double *distance_near, int *predecessor_near, const uint8_t *settled_near,
    const int *heap_position_near, const double *distance_far, const uint8_t *settled_far,
    const int *heap_position_far, int expand_forward, int from, int to, double reach, double *best_weight,
    int *meet_forward, int *meet_backward) {
    if (bidirectional_reached(heap_position_far, settled_far, to) && reach + distance_far[to] < *best_weight) {
        *best_weight = reach + distance_far[to];
        *meet_forward = expand_forward ? from : to;
        *meet_backward = expand_forward ? to : from;
    }
    if (!bidirectional_reached(heap_position_near, settled_near, to)) {
        distance_near[to] = reach;
        predecessor_near[to] = from;
        int heap_slot = frontier->size++;
        frontier->nodes[heap_slot] = to;
        frontier->heap_position[to] = heap_slot;
        min_heap_sift_up(frontier, heap_slot);
    } else if (!settled_near[to] && reach < distance_near[to]) {
        distance_near[to] = reach;
        predecessor_near[to] = from;
        min_heap_sift_up(frontier, frontier->heap_position[to]);
    }
}

/* Drop the edge states (indices from node_count on) from a path of search states. */
static void bidirectional_keep_nodes(BidirectionalPath *out, int node_count) {
    int kept = 0;
    for (int position = 0; position < out->path_length; position++)
        if (out->path[position] < node_count) out->path[kept++] = out->path[position];
    out->path_length = kept;
}

/* Bidirectional Dijkstra over the CSR adjacency. The forward search follows
   outgoing edges from source, the backward search incoming edges into target;
   undirected graphs use the same adjacency for both. The frontier with the
   smaller top key is expanded next, the best source-target weight seen over
   relaxed edges is tracked as the searches proceed, and the search stops once
   a state has been settled by both. With has_multi_terminal_edges, each edge with
   three or more terminals is a search state (node_count + e), entered through an
   active terminal and left through each active exit terminal (directed: forward in to
   out, backward out to in), never back to the node it was entered from; a
   two-terminal edge then costs the sum of its two terminal weights. With a negative
   terminal weight the steps are shifted as PassWeights describes, the backward
   search running on the reverse of the forward one, so that a meeting at an
   edge state adds up to the true pass cost. Returns -1 on error. */
static inline __attribute__((always_inline)) int bidirectional_search(
    int node_count, int edge_count, const AdjacencyList *adjacency, const double *weights, const TerminalList *terminals,
    const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask,
    const PassWeights *pass_weights, int directed, int source, int target, BidirectionalPath *out,
    const int has_multi_terminal_edges) {
    const double *shifts = has_multi_terminal_edges ? pass_weights->shifts : NULL;
    out->weight = HUGE_VAL;
    out->path = NULL;
    out->path_length = 0;

    const int *backward_offset = directed ? adjacency->reverse_offset : adjacency->offset;
    const int *backward_neighbors = directed ? adjacency->reverse_neighbors : adjacency->neighbors;
    const int *backward_edge_indices = directed ? adjacency->reverse_edge_indices : adjacency->edge_indices;
    if (!backward_offset) return 0;  /* directed graph without edges */

    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)edge_count : 0);
    double *distance = (double *)malloc(2 * state_count * sizeof(double));
    int *predecessor = (int *)malloc(2 * state_count * sizeof(int));
    int *heap = (int *)malloc(2 * state_count * sizeof(int));
    int *heap_position = (int *)malloc(2 * state_count * sizeof(int));
    uint8_t *settled = (uint8_t *)calloc(2 * state_count, 1);
    if (!distance || !predecessor || !heap || !heap_position || !settled) {
        free(distance); free(predecessor); free(heap); free(heap_position); free(settled);
        PyErr_NoMemory();
        return -1;
    }
    memset(heap_position, -1, 2 * state_count * sizeof(int));

    double *distance_forward = distance, *distance_backward = distance + state_count;
    int *predecessor_forward = predecessor, *predecessor_backward = predecessor + state_count;
    int *heap_position_forward = heap_position, *heap_position_backward = heap_position + state_count;
    uint8_t *settled_forward = settled, *settled_backward = settled + state_count;

    MinHeap heap_forward = {distance_forward, heap, heap_position_forward, 1};
    MinHeap heap_backward = {distance_backward, heap + state_count, heap_position_backward, 1};
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

        int state = min_heap_pop(frontier);
        settled_near[state] = 1;
        if (bidirectional_reached(heap_position_far, settled_far, state) && distance_near[state] + distance_far[state] < best_weight) {
            best_weight = distance_near[state] + distance_far[state];
            meet_forward = state; meet_backward = state;
        }
        if (settled_far[state]) break;

        if (has_multi_terminal_edges && state >= node_count) {
            int edge = state - node_count;
            int first_exit = (directed && expand_forward) ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
            int end_exit = (directed && !expand_forward) ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge + 1];
            int entry_node = predecessor_near[state];
            double exit_shift = shifts ? (directed == expand_forward ? shifts[edge] : -shifts[edge]) : 0.0;
            int closed_exit = (shifts && !directed && !expand_forward) ? pass_weights->negative_terminals[edge] : -1;
            for (int exit = first_exit; exit < end_exit; exit++) {
                if (terminals->terminal_nodes[exit] == entry_node || exit == closed_exit) continue;
                if (!terminal_is_active(terminals, edge, exit, edge_mask, terminal_mask, node_mask)) continue;
                bidirectional_relax(frontier, distance_near, predecessor_near, settled_near, heap_position_near,
                                    distance_far, settled_far, heap_position_far, expand_forward, state,
                                    terminals->terminal_nodes[exit], distance_near[state] + weights[exit] + exit_shift,
                                    &best_weight, &meet_forward, &meet_backward);
            }
            continue;
        }

        int node = state;
        for (int slot = offset[node]; slot < offset[node + 1]; slot++) {
            int neighbor = neighbors[slot];
            int edge_index = edge_indices[slot];
            double edge_weight;
            if (has_multi_terminal_edges && neighbor < 0) {
                int edge = -1 - neighbor;
                if (!terminal_is_active(terminals, edge, edge_index, edge_mask, terminal_mask, node_mask)) continue;
                int negative_terminal = (shifts && !directed) ? pass_weights->negative_terminals[edge] : -1;
                if (negative_terminal >= 0 && (expand_forward ? edge_index == negative_terminal : edge_index != negative_terminal)) {
                    int first_pass = expand_forward ? terminals->terminal_offsets[edge] : negative_terminal;
                    int end_pass = expand_forward ? terminals->terminal_offsets[edge + 1] : negative_terminal + 1;
                    for (int pass = first_pass; pass < end_pass; pass++) {
                        if (pass == edge_index || !terminal_is_active(terminals, edge, pass, edge_mask, terminal_mask, node_mask)) continue;
                        bidirectional_relax(frontier, distance_near, predecessor_near, settled_near, heap_position_near,
                                            distance_far, settled_far, heap_position_far, expand_forward, node,
                                            terminals->terminal_nodes[pass], distance_near[node] + weights[edge_index] + weights[pass],
                                            &best_weight, &meet_forward, &meet_backward);
                    }
                    if (expand_forward) continue;
                }
                neighbor = node_count + edge;
                edge_weight = weights[edge_index] + (shifts ? (directed == expand_forward ? -shifts[edge] : shifts[edge]) : 0.0);
            } else {
                if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                edge_weight = has_multi_terminal_edges
                    ? weights[terminals->terminal_offsets[edge_index]] + weights[terminals->terminal_offsets[edge_index] + 1]
                    : weights[edge_index];
            }
            bidirectional_relax(frontier, distance_near, predecessor_near, settled_near, heap_position_near,
                                distance_far, settled_far, heap_position_far, expand_forward, node, neighbor,
                                distance_near[node] + edge_weight, &best_weight, &meet_forward, &meet_backward);
        }
    }

    int status = 0;
    if (best_weight < HUGE_VAL) {
        out->weight = best_weight;
        status = bidirectional_build_path(predecessor_forward, predecessor_backward, meet_forward, meet_backward, out);
        if (has_multi_terminal_edges && status == 0) bidirectional_keep_nodes(out, node_count);
    }

    free(distance); free(predecessor); free(heap); free(heap_position); free(settled);
    return status;
}

int bidirectional_dijkstra(int node_count, int edge_count, const AdjacencyList *adjacency, const double *weights,
                           const TerminalList *terminals, const PassWeights *pass_weights, const uint8_t *two_terminal_edge_mask,
                           const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask, int directed,
                           int source, int target, BidirectionalPath *out) {
    return terminals
        ? bidirectional_search(node_count, edge_count, adjacency, weights, terminals, two_terminal_edge_mask, edge_mask,
                               terminal_mask, node_mask, pass_weights, directed, source, target, out, 1)
        : bidirectional_search(node_count, 0, adjacency, weights, NULL, two_terminal_edge_mask, NULL, NULL, node_mask,
                               NULL, directed, source, target, out, 0);
}
