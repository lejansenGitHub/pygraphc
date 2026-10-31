/* The series-parallel reduction engine and the state handle of its primitives. */
#include "series_parallel.h"
#include "parse.h"
#include "components.h"

void sp_free(SPState *state) {
    free(state->edge_first_node); free(state->edge_second_node); free(state->edge_operation);
    free(state->loop_next); free(state->edge_alive);
    free(state->half_next); free(state->half_prev);
    free(state->node_head); free(state->degree); free(state->loop_head);
    free(state->node_alive); free(state->heap); free(state->queued);
    free(state->pair_scratch); free(state->operations);
}

int sp_alloc(SPState *state, int node_count, Py_ssize_t edge_capacity) {
    size_t nodes = (size_t)(node_count > 0 ? node_count : 1);
    size_t slots = (size_t)edge_capacity;
    memset(state, 0, sizeof(*state));
    state->edge_first_node = (int *)malloc(slots * sizeof(int));
    state->edge_second_node = (int *)malloc(slots * sizeof(int));
    state->edge_operation = (int *)malloc(slots * sizeof(int));
    state->loop_next = (int *)malloc(slots * sizeof(int));
    state->edge_alive = (uint8_t *)calloc(slots, 1);
    state->half_next = (int *)malloc(2 * slots * sizeof(int));
    state->half_prev = (int *)malloc(2 * slots * sizeof(int));
    state->node_head = (int *)malloc(nodes * sizeof(int));
    state->degree = (int *)calloc(nodes, sizeof(int));
    state->loop_head = (int *)malloc(nodes * sizeof(int));
    state->node_alive = (uint8_t *)calloc(nodes, 1);
    state->heap = (int *)malloc(nodes * sizeof(int));
    state->queued = (uint8_t *)calloc(nodes, 1);
    state->pair_scratch = (int *)malloc(slots * sizeof(int));
    state->operation_capacity = 64;
    state->operations = (SPOperation *)malloc((size_t)state->operation_capacity * sizeof(SPOperation));
    if (!state->edge_first_node || !state->edge_second_node || !state->edge_operation || !state->loop_next
        || !state->edge_alive || !state->half_next || !state->half_prev
        || !state->node_head || !state->degree || !state->loop_head
        || !state->node_alive || !state->heap || !state->queued
        || !state->pair_scratch || !state->operations) {
        sp_free(state);
        PyErr_NoMemory();
        return -1;
    }
    state->edge_capacity = edge_capacity;
    for (size_t node = 0; node < nodes; node++) { state->node_head[node] = -1; state->loop_head[node] = -1; }
    return 0;
}

static int sp_emit(SPState *state, int kind, int left, int right,
                   int endpoint_u, int endpoint_v, int interior, int leaf_edge, int absorber) {
    if (state->operation_count == state->operation_capacity) {
        Py_ssize_t grown = state->operation_capacity * 2;
        SPOperation *moved = (SPOperation *)realloc(state->operations, (size_t)grown * sizeof(SPOperation));
        if (!moved) { PyErr_NoMemory(); return -1; }
        state->operations = moved;
        state->operation_capacity = grown;
    }
    SPOperation *operation = &state->operations[state->operation_count];
    operation->kind = kind; operation->left = left; operation->right = right;
    operation->endpoint_u = endpoint_u; operation->endpoint_v = endpoint_v;
    operation->interior = interior; operation->leaf_edge = leaf_edge; operation->absorber = absorber;
    return (int)state->operation_count++;
}

#define SP_OTHER_END(state, half) (((half) & 1) ? (state)->edge_first_node[(half) >> 1] : (state)->edge_second_node[(half) >> 1])

static void sp_link(SPState *state, int edge) {
    for (int side = 0; side < 2; side++) {
        int half = 2 * edge + side;
        int node = side ? state->edge_second_node[edge] : state->edge_first_node[edge];
        int first = state->node_head[node];
        state->half_prev[half] = -1;
        state->half_next[half] = first;
        if (first >= 0) state->half_prev[first] = half;
        state->node_head[node] = half;
        state->degree[node]++;
    }
}

static void sp_unlink(SPState *state, int edge) {
    for (int side = 0; side < 2; side++) {
        int half = 2 * edge + side;
        int node = side ? state->edge_second_node[edge] : state->edge_first_node[edge];
        int previous = state->half_prev[half], following = state->half_next[half];
        if (previous >= 0) state->half_next[previous] = following;
        else state->node_head[node] = following;
        if (following >= 0) state->half_prev[following] = previous;
        state->degree[node]--;
    }
    state->edge_alive[edge] = 0;
}

/* A new virtual edge, or -1 with an exception set when the slots are spent.
 *
 * Every move that reduces the live edge count keeps ``edge_slots`` inside the
 * allocation, so the bound holding is an invariant rather than a case to
 * handle; it is checked anyway, because the array writes below are what a
 * broken invariant would run past.
 */
static int sp_add_edge(SPState *state, int from_node, int to_node, int operation) {
    if (state->edge_slots >= state->edge_capacity) {
        PyErr_SetString(PyExc_ValueError, "the reduction state has no edge slot left");
        return -1;
    }
    int edge = state->edge_slots++;
    state->edge_first_node[edge] = from_node;
    state->edge_second_node[edge] = to_node;
    state->edge_operation[edge] = operation;
    state->edge_alive[edge] = 1;
    state->loop_next[edge] = -1;
    sp_link(state, edge);
    return edge;
}

static void sp_eliminate(SPState *state, int node) {
    for (int loop = state->loop_head[node]; loop >= 0; loop = state->loop_next[loop])
        state->edge_alive[loop] = 0;
    state->loop_head[node] = -1;
    state->node_alive[node] = 0;
}

/* Queue a node for another look, at most once.
 *
 * ``heap`` is sized for the node count and ``sp_push`` has no bound check, so
 * the ``queued`` test is what keeps the heap in bounds, not just what keeps
 * the work down. Python's worklist has no such test; matching it here would
 * overflow the heap, not merely repeat work. */
void sp_push(SPState *state, int node) {
    if (state->queued[node]) return;
    state->queued[node] = 1;
    int child = state->heap_size++;
    state->heap[child] = node;
    while (child > 0) {
        int parent = (child - 1) / 2;
        if (state->heap[parent] <= state->heap[child]) break;
        int swap = state->heap[parent]; state->heap[parent] = state->heap[child]; state->heap[child] = swap;
        child = parent;
    }
}

/* Replace every edge of one endpoint pair by a single virtual edge.
 *
 * ``members`` holds at least two live edges between the two nodes in slot
 * order. A k-ary merge emits k-1 binary parallel operations in a left-deep
 * chain; only the last one carries the endpoints of the resulting virtual
 * edge, so the Python fold recognises the earlier ones as chain links of one
 * k-ary parallel node and flattens them.
 */
int sp_merge_parallel(SPState *state, int from_node, int to_node, const int *members, int count) {
    int combined = state->edge_operation[members[0]];
    for (int index = 1; index < count; index++) {
        int last = index == count - 1;
        combined = sp_emit(state, SP_OP_PARALLEL, combined, state->edge_operation[members[index]],
                           last ? from_node : -1, last ? to_node : -1, -1, -1, -1);
        if (combined < 0) return -1;
    }
    for (int index = 0; index < count; index++) sp_unlink(state, members[index]);
    return sp_add_edge(state, from_node, to_node, combined) < 0 ? -1 : 0;
}

static int sp_compare_ints(const void *left, const void *right) {
    int left_value = *(const int *)left, right_value = *(const int *)right;
    return left_value < right_value ? -1 : (left_value > right_value ? 1 : 0);
}

static int sp_compare_pair_entries(const void *left, const void *right) {
    const SPPairEntry *left_entry = (const SPPairEntry *)left, *right_entry = (const SPPairEntry *)right;
    if (left_entry->other != right_entry->other) return left_entry->other < right_entry->other ? -1 : 1;
    return left_entry->slot < right_entry->slot ? -1 : 1;
}

static int sp_compare_candidates(const void *left, const void *right) {
    const SPParallelCandidate *left_candidate = (const SPParallelCandidate *)left;
    const SPParallelCandidate *right_candidate = (const SPParallelCandidate *)right;
    return left_candidate->first_slot < right_candidate->first_slot ? -1 : 1;
}

/* Every unprotected endpoint pair carrying two or more live edges, in order of
 * its lowest edge id, with the members of each pair in slot order.
 *
 * The live non-loop edges are bucketed by their smaller endpoint in slot
 * order, every bucket is sorted by (larger endpoint, slot), which makes the
 * edges of one pair a run whose first member is the pair's lowest edge id.
 * ``entries`` owns the member slots the candidates point into, so the caller
 * frees both. Pairs at distinct node pairs never share an edge, so the whole
 * list is mutually independent.
 */
int sp_collect_parallels(SPState *state, int node_count, const uint8_t *protected_mask,
                         SPPairEntry **entries_out, SPParallelCandidate **candidates_out,
                         int *count_out) {
    *entries_out = NULL; *candidates_out = NULL; *count_out = 0;
    int live_edge_count = 0;
    for (int edge = 0; edge < state->edge_slots; edge++)
        if (state->edge_alive[edge] && state->edge_first_node[edge] != state->edge_second_node[edge]) live_edge_count++;
    if (live_edge_count < 2) return 0;

    int *bucket_offset = (int *)calloc((size_t)node_count + 1, sizeof(int));
    int *fill = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    SPPairEntry *entries = (SPPairEntry *)malloc((size_t)live_edge_count * sizeof(SPPairEntry));
    SPParallelCandidate *candidates = (SPParallelCandidate *)malloc((size_t)(live_edge_count / 2) * sizeof(SPParallelCandidate));
    if (!bucket_offset || !fill || !entries || !candidates) {
        free(bucket_offset); free(fill); free(entries); free(candidates);
        PyErr_NoMemory();
        return -1;
    }
    for (int edge = 0; edge < state->edge_slots; edge++) {
        if (!state->edge_alive[edge] || state->edge_first_node[edge] == state->edge_second_node[edge]) continue;
        int smaller = state->edge_first_node[edge] < state->edge_second_node[edge] ? state->edge_first_node[edge] : state->edge_second_node[edge];
        bucket_offset[smaller + 1]++;
    }
    for (int node = 1; node <= node_count; node++) bucket_offset[node] += bucket_offset[node - 1];
    memcpy(fill, bucket_offset, (size_t)node_count * sizeof(int));
    for (int edge = 0; edge < state->edge_slots; edge++) {
        if (!state->edge_alive[edge] || state->edge_first_node[edge] == state->edge_second_node[edge]) continue;
        int from_node = state->edge_first_node[edge], to_node = state->edge_second_node[edge];
        int smaller = from_node < to_node ? from_node : to_node;
        entries[fill[smaller]].other = from_node < to_node ? to_node : from_node;
        entries[fill[smaller]].slot = edge;
        fill[smaller]++;
    }
    free(fill);

    int candidate_count = 0;
    for (int node = 0; node < node_count; node++) {
        int start = bucket_offset[node], end = bucket_offset[node + 1];
        if (end - start < 2) continue;
        qsort(entries + start, (size_t)(end - start), sizeof(SPPairEntry), sp_compare_pair_entries);
        int run = start;
        while (run < end) {
            int run_end = run + 1;
            while (run_end < end && entries[run_end].other == entries[run].other) run_end++;
            if (run_end - run >= 2 && !protected_mask[node] && !protected_mask[entries[run].other]) {
                candidates[candidate_count].first_slot = entries[run].slot;
                candidates[candidate_count].start = run;
                candidates[candidate_count].count = run_end - run;
                candidates[candidate_count].from_node = node;
                candidates[candidate_count].to_node = entries[run].other;
                candidate_count++;
            }
            run = run_end;
        }
    }
    free(bucket_offset);
    qsort(candidates, (size_t)candidate_count, sizeof(SPParallelCandidate), sp_compare_candidates);
    *entries_out = entries; *candidates_out = candidates; *count_out = candidate_count;
    return 0;
}

/* The initial parallel sweep: merge every collected pair, lowest edge id first. */
int sp_merge_all_parallels(SPState *state, int node_count, const uint8_t *protected_mask) {
    SPPairEntry *entries; SPParallelCandidate *candidates; int candidate_count;
    if (sp_collect_parallels(state, node_count, protected_mask, &entries, &candidates, &candidate_count) < 0) return -1;
    int status = 0;
    for (int index = 0; index < candidate_count && status == 0; index++) {
        const SPParallelCandidate *candidate = &candidates[index];
        for (int member = 0; member < candidate->count; member++)
            state->pair_scratch[member] = entries[candidate->start + member].slot;
        status = sp_merge_parallel(state, candidate->from_node, candidate->to_node,
                                   state->pair_scratch, candidate->count);
    }
    free(entries); free(candidates);
    return status;
}

/* The move that applies at one node, pendant before series, or 0 for none.
 *
 * ``parts`` receives the two edges and the two neighbors involved, the
 * second of each -1 for a pendant move. A degree-2 node whose two incidences
 * run to the same neighbor is a parallel pair, not a series candidate, and
 * no move applies to it.
 */
int sp_describe_move(SPState *state, int node, const uint8_t *protected_mask,
                     const uint8_t *pendant_keep_mask, const uint8_t *series_blocked_mask,
                     int *parts) {
    if (state->degree[node] == 1 && !pendant_keep_mask[node]) {
        int half = state->node_head[node];
        parts[0] = half >> 1; parts[1] = -1;
        parts[2] = SP_OTHER_END(state, half); parts[3] = -1;
        return SP_OP_PENDANT;
    }
    if (state->degree[node] != 2 || protected_mask[node] || series_blocked_mask[node]) return 0;
    int first_half = state->node_head[node], second_half = state->half_next[first_half];
    int first = first_half >> 1, second = second_half >> 1;
    int first_neighbor = SP_OTHER_END(state, first_half), second_neighbor = SP_OTHER_END(state, second_half);
    if (first > second) {
        int swap = first; first = second; second = swap;
        swap = first_neighbor; first_neighbor = second_neighbor; second_neighbor = swap;
    }
    if (first_neighbor == second_neighbor) return 0;
    parts[0] = first; parts[1] = second;
    parts[2] = first_neighbor; parts[3] = second_neighbor;
    return SP_OP_SERIES;
}

/* Delete a degree-1 node, recording the neighbor that absorbs its payload. */
int sp_apply_pendant(SPState *state, int node, int edge) {
    int subtree = state->edge_operation[edge];
    int neighbor = state->edge_first_node[edge] == node ? state->edge_second_node[edge] : state->edge_first_node[edge];
    sp_unlink(state, edge);
    if (sp_emit(state, SP_OP_PENDANT, subtree, -1, -1, -1, node, -1, neighbor) < 0) return -1;
    sp_eliminate(state, node);
    return 0;
}

/* Replace a degree-2 node by one edge between its two neighbors, returning that edge. */
int sp_apply_series(SPState *state, int node, int first, int second,
                    int first_neighbor, int second_neighbor) {
    int operation = sp_emit(state, SP_OP_SERIES, state->edge_operation[first], state->edge_operation[second],
                     first_neighbor, second_neighbor, node, -1, -1);
    if (operation < 0) return -1;
    sp_unlink(state, first);
    sp_unlink(state, second);
    int edge = sp_add_edge(state, first_neighbor, second_neighbor, operation);
    sp_eliminate(state, node);
    return edge;
}

/* The live edges between one unprotected node pair in slot order, into ``members``.
 *
 * The smaller incidence set is scanned, so a hub pair costs the degree of the
 * other endpoint, and the slots are insertion sorted because a pair holds few
 * edges. Returns the count, or 0 when the pair carries fewer than two edges
 * or either endpoint is protected.
 */
int sp_gather_pair(SPState *state, int from_node, int to_node,
                   const uint8_t *protected_mask, int *members) {
    if (protected_mask[from_node] || protected_mask[to_node]) return 0;
    int scanned = state->degree[from_node] <= state->degree[to_node] ? from_node : to_node;
    int target = scanned == from_node ? to_node : from_node;
    int count = 0;
    for (int half = state->node_head[scanned]; half >= 0; half = state->half_next[half])
        if (SP_OTHER_END(state, half) == target) members[count++] = half >> 1;
    if (count < 2) return 0;
    for (int index = 1; index < count; index++) {
        int slot = members[index], position = index;
        while (position > 0 && members[position - 1] > slot) {
            members[position] = members[position - 1];
            position--;
        }
        members[position] = slot;
    }
    return count;
}

/* One move at the popped node and the parallel pair a series move may create.
 *
 * That follow-up merge is the only parallel merge after the initial sweep.
 */
int sp_move(SPState *state, int node, const uint8_t *preserved_mask, const uint8_t *protected_mask,
            const uint8_t *pendant_keep_mask, const uint8_t *series_blocked_mask) {
    int parts[4];
    int kind = sp_describe_move(state, node, protected_mask, pendant_keep_mask, series_blocked_mask, parts);
    if (kind == SP_OP_PENDANT) {
        if (sp_apply_pendant(state, node, parts[0]) < 0) return -1;
        if (!preserved_mask[parts[2]]) sp_push(state, parts[2]);
        return 0;
    }
    if (kind != SP_OP_SERIES) return 0;
    if (sp_apply_series(state, node, parts[0], parts[1], parts[2], parts[3]) < 0) return -1;
    int count = sp_gather_pair(state, parts[2], parts[3], protected_mask, state->pair_scratch);
    if (count >= 2) {
        int smaller = parts[2] < parts[3] ? parts[2] : parts[3];
        int larger = parts[2] < parts[3] ? parts[3] : parts[2];
        if (sp_merge_parallel(state, smaller, larger, state->pair_scratch, count) < 0) return -1;
    }
    if (!preserved_mask[parts[2]]) sp_push(state, parts[2]);
    if (!preserved_mask[parts[3]]) sp_push(state, parts[3]);
    return 0;
}

/* One leaf operation and one edge slot per included input edge, in edge index order. */
int sp_load_edges(SPState *state, GraphContext *graph, const uint8_t *edge_mask, const uint8_t *node_mask) {
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    for (Py_ssize_t index = 0; index < edge_count; index++) {
        if (edge_mask && edge_mask[index]) continue;
        int from_node = EDGE_FIRST_ENDPOINT(&graph->node_ids.edges, index), to_node = EDGE_SECOND_ENDPOINT(&graph->node_ids.edges, index);
        if (node_mask && (node_mask[from_node] || node_mask[to_node])) continue;
        int operation = sp_emit(state, SP_OP_LEAF, -1, -1, -1, -1, -1, (int)index, -1);
        if (operation < 0) return -1;
        int edge = state->edge_slots++;
        state->edge_first_node[edge] = from_node;
        state->edge_second_node[edge] = to_node;
        state->edge_operation[edge] = operation;
        state->edge_alive[edge] = 1;
        state->loop_next[edge] = -1;
        if (from_node == to_node) {
            state->loop_next[edge] = state->loop_head[from_node];
            state->loop_head[from_node] = edge;
        } else {
            sp_link(state, edge);
        }
    }
    return 0;
}

/* A component without a preserved node carries no question and goes whole, before
 * any move, on the connectivity of the input graph. */
int sp_drop_components_without_preserved_nodes(SPState *state, GraphContext *graph, int node_count, const uint8_t *edge_mask,
                          const uint8_t *node_mask, const uint8_t *preserved_mask) {
    if (node_count == 0) return 0;
    ComponentResult components;
    if (compute_components_masked(node_count, &graph->node_ids.edges, edge_mask, node_mask, &components) < 0) return -1;
    uint8_t *has_preserved_node = (uint8_t *)calloc((size_t)components.component_count, 1);
    if (!has_preserved_node) {
        free(components.labels);
        PyErr_NoMemory();
        return -1;
    }
    for (int node = 0; node < node_count; node++)
        if (state->node_alive[node] && preserved_mask[node]) has_preserved_node[components.labels[node]] = 1;
    for (int node = 0; node < node_count; node++) {
        if (!state->node_alive[node] || has_preserved_node[components.labels[node]]) continue;
        while (state->node_head[node] >= 0) sp_unlink(state, state->node_head[node] >> 1);
        sp_eliminate(state, node);
    }
    free(has_preserved_node);
    free(components.labels);
    return 0;
}

/* Every surviving non-preserved node into the candidate queue, ascending, which is already a valid min-heap. */
void sp_enqueue_all_candidates(SPState *state, int node_count, const uint8_t *preserved_mask) {
    for (int node = 0; node < node_count; node++) {
        if (!state->node_alive[node] || preserved_mask[node]) continue;
        state->heap[state->heap_size++] = node;
        state->queued[node] = 1;
    }
}

/* The operation log, the residual edges and the surviving nodes as twelve int32 byte buffers. */
PyObject *sp_build_log(SPState *state, int node_count) {
    Py_ssize_t residual_count = 0, surviving_count = 0;
    for (int edge = 0; edge < state->edge_slots; edge++) if (state->edge_alive[edge]) residual_count++;
    for (int node = 0; node < node_count; node++) if (state->node_alive[node]) surviving_count++;

    PyObject *result = PyTuple_New(12);
    if (!result) return NULL;
    const SPOperation *operations = state->operations;
    Py_ssize_t operation_count = state->operation_count;

#define SP_OP_FIELD(position, field)                                                       \
    do {                                                                                   \
        PyObject *buffer = PyBytes_FromStringAndSize(NULL, operation_count * (Py_ssize_t)4); \
        if (!buffer) { Py_DECREF(result); return NULL; }                                   \
        int32_t *values = (int32_t *)PyBytes_AS_STRING(buffer);                            \
        for (Py_ssize_t index = 0; index < operation_count; index++)                       \
            values[index] = (int32_t)operations[index].field;                              \
        PyTuple_SET_ITEM(result, position, buffer);                                        \
    } while (0)

    SP_OP_FIELD(0, kind);
    SP_OP_FIELD(1, left);
    SP_OP_FIELD(2, right);
    SP_OP_FIELD(3, endpoint_u);
    SP_OP_FIELD(4, endpoint_v);
    SP_OP_FIELD(5, interior);
    SP_OP_FIELD(6, leaf_edge);
    SP_OP_FIELD(7, absorber);
#undef SP_OP_FIELD

    PyObject *residual_op = PyBytes_FromStringAndSize(NULL, residual_count * 4);
    PyObject *residual_u = PyBytes_FromStringAndSize(NULL, residual_count * 4);
    PyObject *residual_v = PyBytes_FromStringAndSize(NULL, residual_count * 4);
    PyObject *surviving = PyBytes_FromStringAndSize(NULL, surviving_count * 4);
    if (!residual_op || !residual_u || !residual_v || !surviving) {
        Py_XDECREF(residual_op); Py_XDECREF(residual_u); Py_XDECREF(residual_v); Py_XDECREF(surviving);
        Py_DECREF(result);
        return NULL;
    }
    int32_t *operation_values = (int32_t *)PyBytes_AS_STRING(residual_op);
    int32_t *from_values = (int32_t *)PyBytes_AS_STRING(residual_u);
    int32_t *to_values = (int32_t *)PyBytes_AS_STRING(residual_v);
    Py_ssize_t written = 0;
    for (int edge = 0; edge < state->edge_slots; edge++) {
        if (!state->edge_alive[edge]) continue;
        operation_values[written] = (int32_t)state->edge_operation[edge];
        from_values[written] = (int32_t)state->edge_first_node[edge];
        to_values[written] = (int32_t)state->edge_second_node[edge];
        written++;
    }
    int32_t *node_values = (int32_t *)PyBytes_AS_STRING(surviving);
    written = 0;
    for (int node = 0; node < node_count; node++)
        if (state->node_alive[node]) node_values[written++] = (int32_t)node;

    PyTuple_SET_ITEM(result, 8, residual_op);
    PyTuple_SET_ITEM(result, 9, residual_u);
    PyTuple_SET_ITEM(result, 10, residual_v);
    PyTuple_SET_ITEM(result, 11, surviving);
    return result;
}

void sp_release_masks(SPMasks *masks) {
    free(masks->no_members);
    release_mask(&masks->node_mask_buffer); release_mask(&masks->edge_mask_buffer);
    release_mask(&masks->blocked_buffer); release_mask(&masks->keep_buffer);
    release_mask(&masks->protect_buffer); release_mask(&masks->preserved_buffer);
}

/* An absent protected, edge or node mask becomes a shared all-zero block.
 *
 * An absent preserved node mask is refused instead: with no preserved node every component
 * has no preserved node, and both callers drop those before applying a single move,
 * so the neutral reading of None would delete the whole graph. */
int sp_parse_masks(PyObject *preserved_obj, PyObject *protected_obj, PyObject *edge_mask_obj,
                   PyObject *node_mask_obj, PyObject *keep_obj, PyObject *blocked_obj,
                   int node_count, Py_ssize_t edge_count, SPMasks *masks) {
    memset(masks, 0, sizeof(*masks));
    if (preserved_obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "series_parallel_reduce requires a preserved node mask: with no preserved node every "
                        "component has no preserved node and the whole graph is deleted");
        return -1;
    }
    if (parse_mask(preserved_obj, node_count, &masks->preserved, &masks->preserved_buffer) < 0) return -1;
    if (parse_mask(protected_obj, node_count, &masks->protect, &masks->protect_buffer) < 0) {
        release_mask(&masks->preserved_buffer); return -1;
    }
    if (parse_mask(edge_mask_obj, edge_count, &masks->edge_mask, &masks->edge_mask_buffer) < 0) {
        release_mask(&masks->protect_buffer); release_mask(&masks->preserved_buffer); return -1;
    }
    if (parse_mask(node_mask_obj, node_count, &masks->node_mask, &masks->node_mask_buffer) < 0) {
        release_mask(&masks->edge_mask_buffer); release_mask(&masks->protect_buffer);
        release_mask(&masks->preserved_buffer); return -1;
    }
    if (parse_mask(keep_obj, node_count, &masks->keep, &masks->keep_buffer) < 0) {
        release_mask(&masks->node_mask_buffer); release_mask(&masks->edge_mask_buffer);
        release_mask(&masks->protect_buffer); release_mask(&masks->preserved_buffer); return -1;
    }
    if (parse_mask(blocked_obj, node_count, &masks->blocked, &masks->blocked_buffer) < 0) {
        release_mask(&masks->keep_buffer); release_mask(&masks->node_mask_buffer); release_mask(&masks->edge_mask_buffer);
        release_mask(&masks->protect_buffer); release_mask(&masks->preserved_buffer); return -1;
    }
    if (masks->preserved && masks->protect && masks->keep && masks->blocked) return 0;
    masks->no_members = (uint8_t *)calloc((size_t)(node_count > 0 ? node_count : 1), 1);
    if (!masks->no_members) {
        sp_release_masks(masks);
        PyErr_NoMemory();
        return -1;
    }
    if (!masks->preserved) masks->preserved = masks->no_members;
    if (!masks->protect) masks->protect = masks->no_members;
    if (!masks->keep) masks->keep = masks->no_members;
    if (!masks->blocked) masks->blocked = masks->no_members;
    return 0;
}

void sp_handle_release(SPHandle *handle) {
    if (!handle->live) return;
    handle->live = 0;
    sp_free(&handle->state);
    free(handle->preserved_mask);
    free(handle->protected_mask);
    free(handle->pendant_keep_mask);
    free(handle->series_blocked_mask);
    free(handle->candidates);
    free(handle->admitted);
    Py_CLEAR(handle->owner);
}

void sp_handle_destructor(PyObject *capsule) {
    SPHandle *handle = (SPHandle *)PyCapsule_GetPointer(capsule, "pygraphc.SPHandle");
    if (!handle) { PyErr_Clear(); return; }
    sp_handle_release(handle);
    free(handle);
}

SPHandle *get_sp_handle(PyObject *capsule) {
    SPHandle *handle = (SPHandle *)PyCapsule_GetPointer(capsule, "pygraphc.SPHandle");
    if (!handle) {
        PyErr_SetString(PyExc_TypeError, "expected a reduction state capsule");
        return NULL;
    }
    if (!handle->live) {
        PyErr_SetString(PyExc_ValueError, "reduction state is already freed");
        return NULL;
    }
    return handle;
}

/* A live edge between two distinct live nodes, or -1 with an exception set.
 *
 * Indices a caller hands back are checked before anything is written, so a
 * wrong one raises instead of corrupting the incidence structure or reading
 * past an array. The edge must run between exactly the two given nodes, and
 * the two must differ: a self-loop is never linked into the incidence lists,
 * so its half-edge slots hold nothing and unlinking one would read them.
 */
int sp_check_edge(SPHandle *handle, int edge, int node, int neighbor) {
    SPState *state = &handle->state;
    int node_count = handle->node_count;
    if (node != neighbor
        && node >= 0 && node < node_count && neighbor >= 0 && neighbor < node_count && edge >= 0 && edge < state->edge_slots
        && state->edge_alive[edge] && state->node_alive[node]
        && ((state->edge_first_node[edge] == node && state->edge_second_node[edge] == neighbor)
            || (state->edge_second_node[edge] == node && state->edge_first_node[edge] == neighbor)))
        return 0;
    PyErr_SetString(PyExc_ValueError, "the move must name a live edge between two live nodes of the state");
    return -1;
}

/* The two edges of a series move must be distinct, or -1 with an exception set.
 *
 * One edge named twice is unlinked twice and replaced by one new edge, which
 * leaves the live edge count unchanged while consuming a slot, so a caller
 * could repeat it until the slot allocation is spent.
 */
int sp_check_distinct_edges(int edge_a, int edge_b) {
    if (edge_a != edge_b) return 0;
    PyErr_SetString(PyExc_ValueError, "a series move must name two distinct edges");
    return -1;
}

/* The node a pendant or series move removes must not be a preserved node.
 *
 * Preserved nodes survive the reduction by contract and neither reporting
 * primitive ever offers one, so a move naming one comes from the caller.
 */
int sp_check_node_removable(SPHandle *handle, int node) {
    if (node >= 0 && node < handle->node_count && !handle->preserved_mask[node]) return 0;
    PyErr_SetString(PyExc_ValueError, "a pendant or series move must name a live non-preserved node");
    return -1;
}

/* The members of a parallel merge, checked and copied into the pair scratch.
 *
 * Slots must be strictly increasing, which both reporting primitives produce
 * and which rules out unlinking one edge twice.
 */
int sp_check_members(SPHandle *handle, const int32_t *members, Py_ssize_t count,
                     int from_node, int to_node) {
    SPState *state = &handle->state;
    if (count < 2 || count > state->edge_slots) {
        PyErr_SetString(PyExc_ValueError, "a parallel merge needs at least two edges of the state");
        return -1;
    }
    for (Py_ssize_t index = 0; index < count; index++) {
        int edge = (int)members[index];
        if (sp_check_edge(handle, edge, from_node, to_node) < 0) return -1;
        if (index > 0 && edge <= state->pair_scratch[index - 1]) {
            PyErr_SetString(PyExc_ValueError, "the edges of a parallel merge must be in increasing slot order");
            return -1;
        }
        state->pair_scratch[index] = edge;
    }
    return 0;
}

/* Every currently applicable move of one node kind, greedily independent in increasing node index.
 *
 * A move of one kind changes only the incidences at the node it removes and
 * at that node's neighbors, so two eligible nodes conflict exactly when one
 * is a neighbor of the other. Scanning eligible nodes in increasing index
 * and admitting one only when none of its neighbors is already admitted
 * therefore yields a set whose moves may be applied in any order for the same
 * result. The candidate queue is compacted on the way, dropping nodes that no
 * longer survive.
 */
int sp_collect_independent(SPHandle *handle, int kind, int *count_out) {
    SPState *state = &handle->state;
    int kept = 0;
    for (int index = 0; index < state->heap_size; index++) {
        int node = state->heap[index];
        if (state->node_alive[node] && !handle->preserved_mask[node]) handle->candidates[kept++] = node;
        else state->queued[node] = 0;
    }
    qsort(handle->candidates, (size_t)kept, sizeof(int), sp_compare_ints);
    memcpy(state->heap, handle->candidates, (size_t)kept * sizeof(int));
    state->heap_size = kept;  /* ascending, which is already a valid min-heap */

    int parts[4], admitted_count = 0;
    for (int index = 0; index < kept; index++) {
        int node = handle->candidates[index];
        if (sp_describe_move(state, node, handle->protected_mask, handle->pendant_keep_mask,
                                    handle->series_blocked_mask, parts) != kind) continue;
        if (handle->admitted[parts[2]] || (parts[3] >= 0 && handle->admitted[parts[3]])) continue;
        handle->admitted[node] = 1;
        handle->candidates[admitted_count++] = node;
    }
    for (int index = 0; index < admitted_count; index++) handle->admitted[handle->candidates[index]] = 0;
    *count_out = admitted_count;
    return 0;
}
