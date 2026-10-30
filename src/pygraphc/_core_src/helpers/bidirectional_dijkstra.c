#include "bidirectional_dijkstra.h"
#include "min_heap.h"

/* Walk the two predecessor chains from the meeting edge into a single path.
   meet_forward and meet_backward are the endpoints of the meeting edge; they
   are the same node when the searches met on a node rather than an edge. */
static int bidi_build_path(const int *prev_forward, const int *prev_backward,
                           int meet_forward, int meet_backward, BidiPath *out) {
    int forward_length = 0;
    for (int node = meet_forward; node != -1; node = prev_forward[node]) forward_length++;
    int backward_start = (meet_forward == meet_backward) ? prev_backward[meet_backward]
                                                         : meet_backward;
    int backward_length = 0;
    for (int node = backward_start; node != -1; node = prev_backward[node]) backward_length++;

    int total = forward_length + backward_length;
    int *path = (int *)malloc((size_t)total * sizeof(int));
    if (!path) { PyErr_NoMemory(); return -1; }
    int node = meet_forward;
    for (int i = forward_length - 1; i >= 0; i--) { path[i] = node; node = prev_forward[node]; }
    node = backward_start;
    for (int i = forward_length; i < total; i++) { path[i] = node; node = prev_backward[node]; }

    out->path = path;
    out->path_length = total;
    return 0;
}

/* Whether a search has a valid tentative distance for a node. dist entries are
   left uninitialised, so this is what tells a fresh node from a reached one. */
static inline int bidi_reached(const int *pos, const uint8_t *settled, int node) {
    return pos[node] != -1 || settled[node];
}

/* Bidirectional Dijkstra over the CSR adjacency. The forward search follows
   outgoing edges from source, the backward search incoming edges into target;
   undirected graphs use the same adjacency for both. The frontier with the
   smaller top key is expanded next, the best source-target weight seen over
   relaxed edges is tracked as the searches proceed, and the search stops once
   a node has been settled by both. Returns -1 on error. */
int bidi_dijkstra(int n, const AdjList *al, const double *edge_weights,
                  const uint8_t *edge_mask, const uint8_t *node_mask,
                  int directed, int source, int target, BidiPath *out) {
    out->weight = HUGE_VAL;
    out->path = NULL;
    out->path_length = 0;

    const int *back_offset = directed ? al->rev_offset : al->offset;
    const int *back_adj = directed ? al->rev_adj : al->adj;
    const int *back_eid = directed ? al->rev_eid : al->eid;
    if (!back_offset) return 0;  /* directed graph without edges */

    double *dist = (double *)malloc(2 * (size_t)n * sizeof(double));
    int *prev = (int *)malloc(2 * (size_t)n * sizeof(int));
    int *heap = (int *)malloc(2 * (size_t)n * sizeof(int));
    int *pos = (int *)malloc(2 * (size_t)n * sizeof(int));
    uint8_t *settled = (uint8_t *)calloc(2 * (size_t)n, 1);
    if (!dist || !prev || !heap || !pos || !settled) {
        free(dist); free(prev); free(heap); free(pos); free(settled);
        PyErr_NoMemory();
        return -1;
    }
    memset(pos, -1, 2 * (size_t)n * sizeof(int));

    double *dist_forward = dist, *dist_backward = dist + n;
    int *prev_forward = prev, *prev_backward = prev + n;
    int *pos_forward = pos, *pos_backward = pos + n;
    uint8_t *settled_forward = settled, *settled_backward = settled + n;

    MinHeap heap_forward = {dist_forward, heap, pos_forward, 1};
    MinHeap heap_backward = {dist_backward, heap + n, pos_backward, 1};
    dist_forward[source] = 0.0; prev_forward[source] = -1;
    heap_forward.heap[0] = source; pos_forward[source] = 0;
    dist_backward[target] = 0.0; prev_backward[target] = -1;
    heap_backward.heap[0] = target; pos_backward[target] = 0;

    double best_weight = HUGE_VAL;
    int meet_forward = -1, meet_backward = -1;

    while (heap_forward.size > 0 && heap_backward.size > 0) {
        int expand_forward = dist_forward[heap_forward.heap[0]]
                          <= dist_backward[heap_backward.heap[0]];
        MinHeap *frontier = expand_forward ? &heap_forward : &heap_backward;
        double *dist_near = expand_forward ? dist_forward : dist_backward;
        const double *dist_far = expand_forward ? dist_backward : dist_forward;
        int *prev_near = expand_forward ? prev_forward : prev_backward;
        uint8_t *settled_near = expand_forward ? settled_forward : settled_backward;
        const uint8_t *settled_far = expand_forward ? settled_backward : settled_forward;
        const int *pos_near = expand_forward ? pos_forward : pos_backward;
        const int *pos_far = expand_forward ? pos_backward : pos_forward;
        const int *offset = expand_forward ? al->offset : back_offset;
        const int *adj = expand_forward ? al->adj : back_adj;
        const int *eid = expand_forward ? al->eid : back_eid;

        int u = mh_pop(frontier);
        settled_near[u] = 1;
        if (bidi_reached(pos_far, settled_far, u) && dist_near[u] + dist_far[u] < best_weight) {
            best_weight = dist_near[u] + dist_far[u];
            meet_forward = u; meet_backward = u;
        }
        if (settled_far[u]) break;

        for (int i = offset[u]; i < offset[u + 1]; i++) {
            if (edge_mask && edge_mask[eid[i]]) continue;
            int v = adj[i];
            if (node_mask && node_mask[v]) continue;
            double reach = dist_near[u] + edge_weights[eid[i]];
            if (bidi_reached(pos_far, settled_far, v) && reach + dist_far[v] < best_weight) {
                best_weight = reach + dist_far[v];
                meet_forward = expand_forward ? u : v;
                meet_backward = expand_forward ? v : u;
            }
            if (!bidi_reached(pos_near, settled_near, v)) {
                dist_near[v] = reach;
                prev_near[v] = u;
                int slot = frontier->size++;
                frontier->heap[slot] = v;
                frontier->pos[v] = slot;
                mh_sift_up(frontier, slot);
            } else if (!settled_near[v] && reach < dist_near[v]) {
                dist_near[v] = reach;
                prev_near[v] = u;
                mh_sift_up(frontier, frontier->pos[v]);
            }
        }
    }

    int status = 0;
    if (best_weight < HUGE_VAL) {
        out->weight = best_weight;
        status = bidi_build_path(prev_forward, prev_backward, meet_forward, meet_backward, out);
    }

    free(dist); free(prev); free(heap); free(pos); free(settled);
    return status;
}

/* Turn the node indices of a bidirectional result into a list of node IDs. */
PyObject *bidi_path_to_list(const BidiPath *bp, const NidContext *ctx) {
    PyObject *path = PyList_New(bp->path_length);
    if (!path) return NULL;
    for (int i = 0; i < bp->path_length; i++) {
        PyObject *item = ctx->nid_items[bp->path[i]];
        Py_INCREF(item);
        PyList_SET_ITEM(path, i, item);
    }
    return path;
}
