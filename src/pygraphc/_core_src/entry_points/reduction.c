/* The series-parallel reduction: the one-shot reduce and the primitives over a state handle. */
#include "entry_points.h"
#include "../conversion/series_parallel.h"

/* series_parallel_reduce_ctx(capsule, preserved_mask, protected_mask
 *                           [, edge_mask, node_mask,
 *                            pendant_keep_mask, series_blocked_mask])
 *     -> twelve int32 byte buffers
 *
 * ``preserved_mask``, ``protected_mask``, ``pendant_keep_mask`` and
 * ``series_blocked_mask`` hold one byte per node index and mark membership by
 * a non-zero byte; ``edge_mask`` and ``node_mask`` exclude by a non-zero byte
 * like every other context call. A kept node takes no pendant move, a blocked
 * one no series move. The buffers are the ones ``sp_build_log`` documents.
 *
 * There is no ``fold_leaves``: the log reports which neighbor absorbs a
 * pendant payload either way and the Python fold decides whether to apply it.
 */
PyObject *py_series_parallel_reduce_ctx(PyObject *self, PyObject *args, PyObject *keywords) {
    static char *keyword_names[] = {"capsule", "preserved_mask", "protected_mask", "edge_mask", "node_mask",
                            "pendant_keep_mask", "series_blocked_mask", NULL};
    PyObject *capsule, *preserved_obj, *protected_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    PyObject *keep_obj = Py_None, *blocked_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOO|OOOO", keyword_names, &capsule, &preserved_obj,
                                     &protected_obj, &edge_mask_obj, &node_mask_obj, &keep_obj, &blocked_obj))
        return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
    if (!graph) return NULL;
    if (graph->directed) {
        PyErr_SetString(PyExc_TypeError, "series_parallel_reduce requires an undirected graph");
        return NULL;
    }
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    SPMasks masks;
    if (sp_parse_masks(preserved_obj, protected_obj, edge_mask_obj, node_mask_obj, keep_obj, blocked_obj, node_count, edge_count, &masks) < 0)
        return NULL;

    /* Every virtual edge consumes at least two live edges and leaves one, so
     * there are never more virtual edges than input edges. */
    SPState state;
    if (sp_alloc(&state, node_count, 2 * edge_count + 8) < 0) {
        sp_release_masks(&masks);
        return NULL;
    }
    for (int node = 0; node < node_count; node++)
        state.node_alive[node] = !(masks.node_mask && masks.node_mask[node]);

    int status = sp_load_edges(&state, graph, masks.edge_mask, masks.node_mask);
    if (status == 0) status = sp_drop_components_without_preserved_nodes(&state, graph, node_count, masks.edge_mask, masks.node_mask, masks.preserved);
    if (status == 0) status = sp_merge_all_parallels(&state, node_count, masks.protect);
    if (status == 0) {
        sp_enqueue_all_candidates(&state, node_count, masks.preserved);
        while (state.heap_size > 0 && status == 0) {
            int node = sp_pop(&state);
            if (!state.node_alive[node]) continue;
            status = sp_move(&state, node, masks.preserved, masks.protect, masks.keep, masks.blocked);
        }
    }
    sp_release_masks(&masks);
    if (status < 0) { sp_free(&state); return NULL; }

    PyObject *result = sp_build_log(&state, node_count);
    sp_free(&state);
    return result;
}

/* sp_state_new(capsule, preserved_mask, protected_mask[, edge_mask, node_mask]) -> state capsule
 *
 * The incidence structure of the masked graph with one leaf operation per
 * included edge, the components without a preserved node already gone, and every
 * surviving non-preserved node queued as a move candidate. No move is applied.
 */
PyObject *py_sp_state_new(PyObject *self, PyObject *args) {
    PyObject *capsule, *preserved_obj, *protected_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    PyObject *keep_obj = Py_None, *blocked_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|OOOO", &capsule, &preserved_obj, &protected_obj, &edge_mask_obj,
                          &node_mask_obj, &keep_obj, &blocked_obj))
        return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
    if (!graph) return NULL;
    if (graph->directed) {
        PyErr_SetString(PyExc_TypeError, "series_parallel_reduce requires an undirected graph");
        return NULL;
    }
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    SPMasks masks;
    if (sp_parse_masks(preserved_obj, protected_obj, edge_mask_obj, node_mask_obj, keep_obj, blocked_obj, node_count, edge_count, &masks) < 0)
        return NULL;

    SPHandle *handle = (SPHandle *)calloc(1, sizeof(SPHandle));
    if (!handle) { sp_release_masks(&masks); return PyErr_NoMemory(); }
    size_t nodes = (size_t)(node_count > 0 ? node_count : 1);
    handle->node_count = node_count;
    if (sp_alloc(&handle->state, node_count, 2 * edge_count + 8) < 0) {
        free(handle);
        sp_release_masks(&masks);
        return NULL;
    }
    handle->live = 1;
    handle->preserved_mask = (uint8_t *)malloc(nodes);
    handle->protected_mask = (uint8_t *)malloc(nodes);
    handle->pendant_keep_mask = (uint8_t *)malloc(nodes);
    handle->series_blocked_mask = (uint8_t *)malloc(nodes);
    handle->candidates = (int *)malloc(nodes * sizeof(int));
    handle->admitted = (uint8_t *)calloc(nodes, 1);
    if (!handle->preserved_mask || !handle->protected_mask || !handle->pendant_keep_mask
        || !handle->series_blocked_mask || !handle->candidates || !handle->admitted) {
        sp_handle_release(handle);
        free(handle);
        sp_release_masks(&masks);
        return PyErr_NoMemory();
    }
    memcpy(handle->preserved_mask, masks.preserved, nodes);
    memcpy(handle->protected_mask, masks.protect, nodes);
    memcpy(handle->pendant_keep_mask, masks.keep, nodes);
    memcpy(handle->series_blocked_mask, masks.blocked, nodes);
    for (int node = 0; node < node_count; node++)
        handle->state.node_alive[node] = !(masks.node_mask && masks.node_mask[node]);

    int status = sp_load_edges(&handle->state, graph, masks.edge_mask, masks.node_mask);
    if (status == 0)
        status = sp_drop_components_without_preserved_nodes(&handle->state, graph, node_count, masks.edge_mask, masks.node_mask, handle->preserved_mask);
    sp_release_masks(&masks);
    if (status < 0) { sp_handle_release(handle); free(handle); return NULL; }
    sp_enqueue_all_candidates(&handle->state, node_count, handle->preserved_mask);

    PyObject *result = PyCapsule_New(handle, "pygraphc.SPHandle", sp_handle_destructor);
    if (!result) { sp_handle_release(handle); free(handle); return NULL; }
    handle->owner = Py_NewRef(capsule);
    return result;
}

/* sp_state_free(state) -> None: release the arrays and make the handle inert. */
PyObject *py_sp_state_free(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    sp_handle_release(handle);
    Py_RETURN_NONE;
}

/* sp_next_move(state) -> (kind, node, edge_a, edge_b, neighbor_a, neighbor_b) or None
 *
 * The move that applies at the next candidate node, pendant before series,
 * in increasing node index. Candidates that admit no move are dropped from
 * the queue, and the reported node is dequeued whether or not the caller
 * applies the move.
 */
PyObject *py_sp_next_move(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    SPState *state = &handle->state;
    int parts[4];
    while (state->heap_size > 0) {
        int node = sp_pop(state);
        if (!state->node_alive[node]) continue;
        int kind = sp_describe_move(state, node, handle->protected_mask, handle->pendant_keep_mask,
                                    handle->series_blocked_mask, parts);
        if (kind == 0) continue;
        return Py_BuildValue("(iiiiii)", kind, node, parts[0], parts[1], parts[2], parts[3]);
    }
    Py_RETURN_NONE;
}

/* sp_apply_move(state, move) -> (created_edge, parallel_pending)
 *
 * One move as ``sp_next_move`` reported it. ``created_edge`` is the edge a
 * series move produced, -1 for a pendant move, and ``parallel_pending`` is 1
 * when the two endpoints of that edge now carry a mergeable parallel pair.
 * The neighbors the move touched re-enter the candidate queue.
 */
PyObject *py_sp_apply_move(PyObject *self, PyObject *args) {
    PyObject *capsule;
    int kind, node, edge_a, edge_b, neighbor_a, neighbor_b;
    if (!PyArg_ParseTuple(args, "O(iiiiii)", &capsule, &kind, &node, &edge_a, &edge_b,
                          &neighbor_a, &neighbor_b))
        return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    SPState *state = &handle->state;
    if (sp_check_edge(handle, edge_a, node, neighbor_a) < 0) return NULL;
    if (kind == SP_OP_SERIES && sp_check_edge(handle, edge_b, node, neighbor_b) < 0) return NULL;
    if (kind == SP_OP_SERIES && sp_check_distinct_edges(edge_a, edge_b) < 0) return NULL;
    if (sp_check_node_removable(handle, node) < 0) return NULL;
    if (kind == SP_OP_PENDANT) {
        if (sp_apply_pendant(state, node, edge_a) < 0) return NULL;
        if (!handle->preserved_mask[neighbor_a]) sp_push(state, neighbor_a);
        return Py_BuildValue("(ii)", -1, 0);
    }
    if (kind != SP_OP_SERIES) {
        PyErr_SetString(PyExc_ValueError, "move kind must be a pendant or a series move");
        return NULL;
    }
    int created = sp_apply_series(state, node, edge_a, edge_b, neighbor_a, neighbor_b);
    if (created < 0) return NULL;
    if (!handle->preserved_mask[neighbor_a]) sp_push(state, neighbor_a);
    if (!handle->preserved_mask[neighbor_b]) sp_push(state, neighbor_b);
    int count = sp_gather_pair(state, neighbor_a, neighbor_b, handle->protected_mask, state->pair_scratch);
    return Py_BuildValue("(ii)", created, count >= 2 ? 1 : 0);
}

/* sp_pair_edges(state, from_node, to_node) -> int32 buffer of edges, or None
 *
 * The live edges between one unprotected node pair in slot order, or None
 * when the pair carries fewer than two or either endpoint is protected.
 */
PyObject *py_sp_pair_edges(PyObject *self, PyObject *args) {
    PyObject *capsule;
    int from_node, to_node;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &from_node, &to_node)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    SPState *state = &handle->state;
    if (from_node < 0 || from_node >= handle->node_count || to_node < 0 || to_node >= handle->node_count) {
        PyErr_SetString(PyExc_ValueError, "both nodes must be node indices of the state");
        return NULL;
    }
    int count = sp_gather_pair(state, from_node, to_node, handle->protected_mask, state->pair_scratch);
    if (count < 2) Py_RETURN_NONE;
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)count * 4);
    if (!result) return NULL;
    int32_t *values = (int32_t *)PyBytes_AS_STRING(result);
    for (int index = 0; index < count; index++) values[index] = (int32_t)state->pair_scratch[index];
    return result;
}

/* sp_apply_parallel(state, from_node, to_node, edges) -> created_edge
 *
 * Replace the given live edges between one node pair by a single edge. The
 * caller passes the members ``sp_pair_edges`` reported, in slot order.
 */
PyObject *py_sp_apply_parallel(PyObject *self, PyObject *args) {
    PyObject *capsule, *edges_obj;
    int from_node, to_node;
    if (!PyArg_ParseTuple(args, "OiiO", &capsule, &from_node, &to_node, &edges_obj)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    Py_buffer edges_view;
    if (PyObject_GetBuffer(edges_obj, &edges_view, PyBUF_SIMPLE) < 0) return NULL;
    Py_ssize_t count = edges_view.len / 4;
    int status = sp_check_members(handle, (const int32_t *)edges_view.buf, count, from_node, to_node);
    PyBuffer_Release(&edges_view);
    if (status < 0) return NULL;
    SPState *state = &handle->state;
    int smaller = from_node < to_node ? from_node : to_node;
    int larger = from_node < to_node ? to_node : from_node;
    if (sp_merge_parallel(state, smaller, larger, state->pair_scratch, (int)count) < 0) return NULL;
    return PyLong_FromLong(state->edge_slots - 1);
}

/* sp_batch_moves(state, kind) -> int32 buffer, or None when no move of that kind applies
 *
 * A pendant or series kind reports five values per move, ``kind``, the node
 * and the two edges and two neighbors of ``sp_next_move``. A parallel kind
 * reports one variable-length record per endpoint pair: the two endpoints,
 * the member count and the member edges. Either way the moves in the buffer
 * are mutually independent and the buffer is what ``sp_apply_batch`` takes.
 */
PyObject *py_sp_batch_moves(PyObject *self, PyObject *args) {
    PyObject *capsule;
    int kind;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &kind)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    SPState *state = &handle->state;

    if (kind == SP_OP_PARALLEL) {
        SPPairEntry *entries; SPParallelCandidate *candidates; int candidate_count;
        if (sp_collect_parallels(state, handle->node_count, handle->protected_mask,
                                 &entries, &candidates, &candidate_count) < 0)
            return NULL;
        if (candidate_count == 0) { free(entries); free(candidates); Py_RETURN_NONE; }
        Py_ssize_t value_count = 0;
        for (int index = 0; index < candidate_count; index++) value_count += 3 + candidates[index].count;
        PyObject *result = PyBytes_FromStringAndSize(NULL, value_count * 4);
        if (!result) { free(entries); free(candidates); return NULL; }
        int32_t *out = (int32_t *)PyBytes_AS_STRING(result);
        for (int index = 0; index < candidate_count; index++) {
            const SPParallelCandidate *candidate = &candidates[index];
            *out++ = (int32_t)candidate->from_node;
            *out++ = (int32_t)candidate->to_node;
            *out++ = (int32_t)candidate->count;
            for (int member = 0; member < candidate->count; member++)
                *out++ = (int32_t)entries[candidate->start + member].slot;
        }
        free(entries); free(candidates);
        return result;
    }
    if (kind != SP_OP_PENDANT && kind != SP_OP_SERIES) {
        PyErr_SetString(PyExc_ValueError, "move kind must be a pendant, series or parallel move");
        return NULL;
    }
    int admitted_count;
    if (sp_collect_independent(handle, kind, &admitted_count) < 0) return NULL;
    if (admitted_count == 0) Py_RETURN_NONE;
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)admitted_count * 5 * 4);
    if (!result) return NULL;
    int32_t *out = (int32_t *)PyBytes_AS_STRING(result);
    int parts[4];
    for (int index = 0; index < admitted_count; index++) {
        int node = handle->candidates[index];
        sp_describe_move(state, node, handle->protected_mask, handle->pendant_keep_mask,
                                    handle->series_blocked_mask, parts);
        *out++ = (int32_t)node;
        *out++ = (int32_t)parts[0]; *out++ = (int32_t)parts[1];
        *out++ = (int32_t)parts[2]; *out++ = (int32_t)parts[3];
    }
    return result;
}

/* sp_apply_batch(state, kind, batch) -> None
 *
 * Every move of one ``sp_batch_moves`` buffer, in buffer order. The
 * neighbors the moves touched re-enter the candidate queue; the parallel
 * pairs a series batch may create are left for the next report.
 */
PyObject *py_sp_apply_batch(PyObject *self, PyObject *args) {
    PyObject *capsule, *batch_obj;
    int kind;
    if (!PyArg_ParseTuple(args, "OiO", &capsule, &kind, &batch_obj)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    Py_buffer batch_view;
    if (PyObject_GetBuffer(batch_obj, &batch_view, PyBUF_SIMPLE) < 0) return NULL;
    SPState *state = &handle->state;
    const int32_t *values = (const int32_t *)batch_view.buf;
    Py_ssize_t length = batch_view.len / 4, position = 0;
    int status = 0;

    if (kind == SP_OP_PARALLEL) {
        while (position + 3 <= length && status == 0) {
            int from_node = (int)values[position], to_node = (int)values[position + 1];
            Py_ssize_t count = (Py_ssize_t)values[position + 2];
            position += 3;
            if (count < 2 || position + count > length) {
                PyErr_SetString(PyExc_ValueError, "a parallel batch record holds two endpoints, a count and that many edges");
                status = -1;
                break;
            }
            status = sp_check_members(handle, values + position, count, from_node, to_node);
            position += count;
            if (status == 0) status = sp_merge_parallel(state, from_node, to_node, state->pair_scratch, (int)count);
            if (status == 0 && !handle->preserved_mask[from_node]) sp_push(state, from_node);
            if (status == 0 && !handle->preserved_mask[to_node]) sp_push(state, to_node);
        }
        if (status == 0 && position != length) {
            PyErr_SetString(PyExc_ValueError, "a parallel batch record holds two endpoints, a count and that many edges");
            status = -1;
        }
    } else if (kind == SP_OP_PENDANT || kind == SP_OP_SERIES) {
        while (position + 5 <= length && status == 0) {
            int node = (int)values[position], edge_a = (int)values[position + 1];
            int edge_b = (int)values[position + 2];
            int neighbor_a = (int)values[position + 3], neighbor_b = (int)values[position + 4];
            position += 5;
            status = sp_check_edge(handle, edge_a, node, neighbor_a);
            if (status == 0 && kind == SP_OP_SERIES) status = sp_check_edge(handle, edge_b, node, neighbor_b);
            if (status == 0 && kind == SP_OP_SERIES) status = sp_check_distinct_edges(edge_a, edge_b);
            if (status == 0) status = sp_check_node_removable(handle, node);
            if (status < 0) break;
            if (kind == SP_OP_PENDANT) {
                status = sp_apply_pendant(state, node, edge_a);
            } else {
                status = sp_apply_series(state, node, edge_a, edge_b, neighbor_a, neighbor_b) < 0 ? -1 : 0;
                if (!handle->preserved_mask[neighbor_b]) sp_push(state, neighbor_b);
            }
            if (!handle->preserved_mask[neighbor_a]) sp_push(state, neighbor_a);
        }
        if (status == 0 && position != length) {
            PyErr_SetString(PyExc_ValueError, "a pendant or series batch holds five int32 per move");
            status = -1;
        }
    } else {
        PyErr_SetString(PyExc_ValueError, "move kind must be a pendant, series or parallel move");
        status = -1;
    }
    PyBuffer_Release(&batch_view);
    if (status < 0) return NULL;
    Py_RETURN_NONE;
}

/* sp_state_log(state) -> twelve int32 byte buffers, as ``series_parallel_reduce_ctx`` returns. */
PyObject *py_sp_state_log(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SPHandle *handle = get_sp_handle(capsule);
    if (!handle) return NULL;
    return sp_build_log(&handle->state, handle->node_count);
}
