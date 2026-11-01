/* The series-parallel reduction state from and to Python: masks, the operation log as buffers, the state capsule. */
#ifndef PYGRAPHC_CORE_CONVERSION_SERIES_PARALLEL_H
#define PYGRAPHC_CORE_CONVERSION_SERIES_PARALLEL_H

#include "../algorithms/series_parallel.h"

/* The four masks of a reduction, parsed together and released together. */
typedef struct {
    const uint8_t *preserved; const uint8_t *protect; const uint8_t *keep; const uint8_t *blocked;
    const uint8_t *edge_mask; const uint8_t *node_mask;
    Py_buffer preserved_buffer, protect_buffer, keep_buffer, blocked_buffer, edge_mask_buffer, node_mask_buffer;
    uint8_t *no_members;
} SPMasks;

PYGRAPHC_INTERNAL PyObject *sp_build_log(SPState *state, int node_count);
PYGRAPHC_INTERNAL void sp_release_masks(SPMasks *masks);
PYGRAPHC_INTERNAL int sp_parse_masks(PyObject *preserved_obj, PyObject *protected_obj, PyObject *edge_mask_obj,
                                     PyObject *node_mask_obj, PyObject *keep_obj, PyObject *blocked_obj,
                                     int node_count, Py_ssize_t edge_count, SPMasks *masks);
PYGRAPHC_INTERNAL void sp_handle_release(SPHandle *handle);
PYGRAPHC_INTERNAL void sp_handle_destructor(PyObject *capsule);
PYGRAPHC_INTERNAL SPHandle *get_sp_handle(PyObject *capsule);

#endif
