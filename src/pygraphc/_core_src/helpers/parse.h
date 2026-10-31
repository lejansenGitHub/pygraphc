/* Parsing helpers: Python objects and buffers into the data models. */
#ifndef PYGRAPHC_CORE_PARSE_H
#define PYGRAPHC_CORE_PARSE_H

#include "../models.h"

/* ── Buffer-format helpers ── */

static inline int is_int32_fmt(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 4 || !fmt) return 0;
    while (*fmt == '<' || *fmt == '>' || *fmt == '=' || *fmt == '!') fmt++;
    return *fmt == 'i';
}

static inline int is_int64_fmt(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 8 || !fmt) return 0;
    while (*fmt == '<' || *fmt == '>' || *fmt == '=' || *fmt == '!') fmt++;
    return *fmt == 'l' || *fmt == 'q';
}

PYGRAPHC_INTERNAL int parse_edges(PyObject *edges_obj, EdgeList *el);
PYGRAPHC_INTERNAL int nid_parse(PyObject *node_ids_obj, PyObject *edges_obj, NidContext *ctx);
PYGRAPHC_INTERNAL int nid_parse_split(PyObject *node_ids_obj, PyObject *src_obj, PyObject *dst_obj,
                                      NidContext *ctx);
PYGRAPHC_INTERNAL int parse_mask(PyObject *mask_obj, Py_ssize_t expected_len,
                                 const uint8_t **mask_out, Py_buffer *mask_buf);
PYGRAPHC_INTERNAL int parse_weights(PyObject *obj, WeightList *wl, Py_ssize_t expected_m);
PYGRAPHC_INTERNAL int parse_labels(PyObject *labels_obj, int n, const int32_t **labels_out, Py_buffer *labels_buf);

#endif
