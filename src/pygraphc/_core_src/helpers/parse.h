/* Parsing helpers: Python objects and buffers into the data models. */
#ifndef PYGRAPHC_CORE_PARSE_H
#define PYGRAPHC_CORE_PARSE_H

#include "../models.h"

/* ── Buffer-format helpers ── */

static inline int is_int32_format(const char *format, Py_ssize_t itemsize) {
    if (itemsize != 4 || !format) return 0;
    while (*format == '<' || *format == '>' || *format == '=' || *format == '!') format++;
    return *format == 'i';
}

static inline int is_int64_format(const char *format, Py_ssize_t itemsize) {
    if (itemsize != 8 || !format) return 0;
    while (*format == '<' || *format == '>' || *format == '=' || *format == '!') format++;
    return *format == 'l' || *format == 'q';
}

PYGRAPHC_INTERNAL int parse_edges(PyObject *edges_obj, EdgeList *edges);
PYGRAPHC_INTERNAL int node_id_parse(PyObject *node_ids_obj, PyObject *edges_obj, NodeIdContext *context);
PYGRAPHC_INTERNAL int node_id_parse_split(PyObject *node_ids_obj, PyObject *first_endpoints_obj, PyObject *second_endpoints_obj,
                                      NodeIdContext *context);
PYGRAPHC_INTERNAL int parse_mask(PyObject *mask_obj, Py_ssize_t expected_length,
                                 const uint8_t **mask_out, Py_buffer *mask_buffer);
PYGRAPHC_INTERNAL int parse_weights(PyObject *weights_obj, WeightList *weights, Py_ssize_t expected_edge_count);
/* parse_weights for a count of other things than edges, named in the error message; tracks_sign fills has_sign_bit. */
PYGRAPHC_INTERNAL int parse_counted_weights(PyObject *weights_obj, WeightList *weights, Py_ssize_t expected_count,
                                            const char *counted, int tracks_sign);
PYGRAPHC_INTERNAL int parse_labels(PyObject *labels_obj, int node_count, const int32_t **labels_out, Py_buffer *labels_buffer);

#endif
