/* The series-parallel reduction state from and to Python. */
#include "series_parallel.h"
#include "parse.h"

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
