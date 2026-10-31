#ifndef PYGRAPHC_CORE_SERIES_PARALLEL_H
#define PYGRAPHC_CORE_SERIES_PARALLEL_H

#include "../models.h"

/* ── Terminal-preserving series-parallel reduction ──
 *
 * The structural loop of pygraphc.reduction: the three moves applied to a
 * fixpoint against a terminal set, emitting a flat operation log instead of
 * provenance trees. The caller folds the log into its own algebra.
 *
 * Edge slots 0..m-1 are the input edges in edge index order, the slots above
 * them are the virtual edges in creation order. The Python tier hands the
 * edges over sorted by edge id, so slot order is edge id order and "the
 * lowest edge id" is "the lowest slot" everywhere below.
 *
 * Incidences are a doubly linked list of half-edges per node, so a move
 * unlinks in constant time and degrees are read, never recomputed. Self-loops
 * live on a separate singly linked list per node, take part in no move and
 * die with their node.
 */

#define SP_OP_LEAF     0
#define SP_OP_SERIES   1
#define SP_OP_PARALLEL 2
#define SP_OP_PENDANT  3

typedef struct {
    int kind;
    int left;        /* child operation id, -1 when absent */
    int right;
    int endpoint_u;  /* oriented endpoints of the resulting virtual edge, -1 when none */
    int endpoint_v;
    int interior;    /* eliminated node of a series move, removed node of a pendant move */
    int leaf_edge;   /* input edge index of a leaf */
    int absorber;    /* neighbour that absorbs the payload of a pendant move */
} SPOperation;

typedef struct {
    int *edge_first_node;
    int *edge_second_node;
    int *edge_operation;
    int *loop_next;     /* next self-loop of the same node, -1 at the end */
    uint8_t *edge_alive;
    int *half_next;     /* half 2e sits at edge_first_node[e], half 2e+1 at edge_second_node[e] */
    int *half_prev;
    int *node_head;     /* first half-edge of the node, -1 when isolated */
    int *degree;        /* non-loop incidences */
    int *loop_head;
    uint8_t *node_alive;
    int *heap;          /* min-heap of candidate node indices, deduplicated by queued */
    uint8_t *queued;
    int *pair_scratch;  /* edges between one endpoint pair, gathered per series move */
    SPOperation *operations;
    Py_ssize_t operation_count;
    Py_ssize_t operation_capacity;
    int heap_size;
    int edge_slots;
    Py_ssize_t edge_capacity;
} SPState;

typedef struct { int other; int slot; } SPPairEntry;

typedef struct { int first_slot; int start; int count; int from_node; int to_node; } SPParallelCandidate;

/* The four masks of a reduction, parsed together and released together. */
typedef struct {
    const uint8_t *terminal; const uint8_t *protect; const uint8_t *keep; const uint8_t *blocked;
    const uint8_t *edge_mask; const uint8_t *node_mask;
    Py_buffer terminal_buffer, protect_buffer, keep_buffer, blocked_buffer, edge_mask_buffer, node_mask_buffer;
    uint8_t *no_members;
} SPMasks;

/* ── The same reduction as primitives over an opaque state handle ──
 *
 * The incidence structure above, the terminal and protected masks and the
 * candidate queue live in a capsule, so a Python caller drives the fixpoint
 * itself: it asks for a move, applies it, and reads the log at the end. The
 * handle survives every move, so no graph is ever rebuilt between them.
 *
 * The handle goes inert on ``sp_state_free`` rather than dangling, and it
 * keeps a reference to the graph capsule it was built from, so the node and
 * edge indices its buffers report always belong to a graph that is still
 * alive.
 */

typedef struct {
    SPState state;
    PyObject *owner;         /* graph capsule the state was built from */
    uint8_t *terminal_mask;  /* owned copies, one byte per node index; a non-zero byte marks membership */
    uint8_t *protected_mask;
    uint8_t *pendant_keep_mask;
    uint8_t *series_blocked_mask;
    int *candidates;         /* node indices of the batch under construction */
    uint8_t *admitted;       /* nodes already in that batch */
    int node_count;
    int live;                /* 0 after sp_state_free, which makes the handle inert */
} SPHandle;

PYGRAPHC_INTERNAL void sp_free(SPState *state);
PYGRAPHC_INTERNAL int sp_alloc(SPState *state, int n, Py_ssize_t edge_capacity);
PYGRAPHC_INTERNAL void sp_push(SPState *state, int node);
PYGRAPHC_INTERNAL int sp_merge_parallel(SPState *state, int from_node, int to_node, const int *members, int count);
PYGRAPHC_INTERNAL int sp_collect_parallels(SPState *state, int n, const uint8_t *protected_mask,
                                           SPPairEntry **entries_out, SPParallelCandidate **candidates_out,
                                           int *count_out);
PYGRAPHC_INTERNAL int sp_merge_all_parallels(SPState *state, int n, const uint8_t *protected_mask);
PYGRAPHC_INTERNAL int sp_describe_move(SPState *state, int node, const uint8_t *protected_mask,
                                       const uint8_t *pendant_keep_mask, const uint8_t *series_blocked_mask,
                                       int *parts);
PYGRAPHC_INTERNAL int sp_apply_pendant(SPState *state, int node, int edge);
PYGRAPHC_INTERNAL int sp_apply_series(SPState *state, int node, int first, int second,
                                      int first_neighbour, int second_neighbour);
PYGRAPHC_INTERNAL int sp_gather_pair(SPState *state, int from_node, int to_node,
                                     const uint8_t *protected_mask, int *members);
PYGRAPHC_INTERNAL int sp_move(SPState *state, int node, const uint8_t *terminal_mask, const uint8_t *protected_mask,
                              const uint8_t *pendant_keep_mask, const uint8_t *series_blocked_mask);
PYGRAPHC_INTERNAL int sp_load_edges(SPState *state, GraphContext *graph, const uint8_t *edge_mask, const uint8_t *node_mask);
PYGRAPHC_INTERNAL int sp_drop_terminal_free(SPState *state, GraphContext *graph, int n, const uint8_t *edge_mask,
                                            const uint8_t *node_mask, const uint8_t *terminal_mask);
PYGRAPHC_INTERNAL void sp_enqueue_all_candidates(SPState *state, int n, const uint8_t *terminal_mask);
PYGRAPHC_INTERNAL PyObject *sp_build_log(SPState *state, int n);
PYGRAPHC_INTERNAL void sp_release_masks(SPMasks *masks);
PYGRAPHC_INTERNAL int sp_parse_masks(PyObject *terminal_obj, PyObject *protected_obj, PyObject *emask_obj,
                                     PyObject *nmask_obj, PyObject *keep_obj, PyObject *blocked_obj,
                                     int n, Py_ssize_t m, SPMasks *masks);
PYGRAPHC_INTERNAL void sp_handle_release(SPHandle *handle);
PYGRAPHC_INTERNAL void sp_handle_destructor(PyObject *capsule);
PYGRAPHC_INTERNAL SPHandle *get_sp_handle(PyObject *capsule);
PYGRAPHC_INTERNAL int sp_check_edge(SPHandle *handle, int edge, int node, int neighbour);
PYGRAPHC_INTERNAL int sp_check_distinct_edges(int edge_a, int edge_b);
PYGRAPHC_INTERNAL int sp_check_node_removable(SPHandle *handle, int node);
PYGRAPHC_INTERNAL int sp_check_members(SPHandle *handle, const int32_t *members, Py_ssize_t count,
                                       int from_node, int to_node);
PYGRAPHC_INTERNAL int sp_collect_independent(SPHandle *handle, int kind, int *count_out);

static inline int sp_pop(SPState *state) {
    int smallest = state->heap[0];
    state->heap[0] = state->heap[--state->heap_size];
    int parent = 0;
    for (;;) {
        int left = 2 * parent + 1, right = left + 1, best = parent;
        if (left < state->heap_size && state->heap[left] < state->heap[best]) best = left;
        if (right < state->heap_size && state->heap[right] < state->heap[best]) best = right;
        if (best == parent) break;
        int swap = state->heap[best]; state->heap[best] = state->heap[parent]; state->heap[parent] = swap;
        parent = best;
    }
    state->queued[smallest] = 0;
    return smallest;
}

#endif
