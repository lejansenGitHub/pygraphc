/* Parsing helpers: Python objects and buffers into the data models. */
#include "parse.h"

/*
 * Try the buffer protocol first (numpy int32 array with shape (m,2)).
 * Falls back to sequence-of-tuples parsing.
 * Returns 0 on success, -1 on error.
 */
int parse_edges(PyObject *edges_obj, EdgeList *el) {
    Py_buffer buf;

    /* Fast path: buffer protocol (numpy array) */
    if (PyObject_GetBuffer(edges_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        /* Validate: must be int32 with shape (m, 2) */
        int is_int32 = (buf.format != NULL && (
            strcmp(buf.format, "i") == 0 ||
            strcmp(buf.format, "<i") == 0 ||
            strcmp(buf.format, "=i") == 0
        ) && buf.itemsize == 4);

        if (is_int32 && buf.ndim == 2 && buf.shape[1] == 2) {
            Py_ssize_t m = buf.shape[0];
            int *data = (int *)buf.buf;
            el->edge_count = m;
            /* Point directly into the numpy buffer — zero copy for edges */
            el->first_endpoints = NULL;
            el->second_endpoints = NULL;
            el->owns_memory = 0;

            /* We need separate src/dst arrays for the union-find loop,
               but we can read directly from the contiguous buffer:
               data[i*2] = src, data[i*2+1] = dst */

            /* Actually, just store the raw pointer and read inline.
               We'll use a special code path. Store raw ptr in src. */
            el->first_endpoints = data;  /* raw (m,2) int32 buffer */
            el->second_endpoints = NULL;  /* sentinel: means "interleaved" */

            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();  /* buffer protocol not supported, try sequence */
    }

    /* Slow path: list of tuples */
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!fast) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);
    el->edge_count = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->first_endpoints = NULL;
        el->second_endpoints = NULL;
        Py_DECREF(fast);
        return 0;
    }

    el->first_endpoints = (int *)malloc(m * sizeof(int));
    el->second_endpoints = (int *)malloc(m * sizeof(int));
    if (!el->first_endpoints || !el->second_endpoints) {
        free(el->first_endpoints); free(el->second_endpoints);
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *edge = items[i];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(el->first_endpoints); free(el->second_endpoints);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **ep = PySequence_Fast_ITEMS(edge_fast);
        el->first_endpoints[i] = (int)PyLong_AsLong(ep[0]);
        el->second_endpoints[i] = (int)PyLong_AsLong(ep[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(el->first_endpoints); free(el->second_endpoints);
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    return 0;
}

/* ── Edge parsing with node-ID translation via C hash map ── */

static int parse_edges_mapped(IntMap *im, PyObject *edges_obj, EdgeList *el) {
    /* Fast path: buffer protocol (numpy int32/int64 array with shape (m,2)) */
    Py_buffer buf;
    if (PyObject_GetBuffer(edges_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_i32 = is_int32_format(buf.format, buf.itemsize);
        int is_i64 = is_int64_format(buf.format, buf.itemsize);

        if ((is_i32 || is_i64) && buf.ndim == 2 && buf.shape[1] == 2) {
            Py_ssize_t m = buf.shape[0];
            el->edge_count = m;
            el->owns_memory = 1;
            if (m == 0) {
                el->first_endpoints = NULL; el->second_endpoints = NULL;
                PyBuffer_Release(&buf);
                return 0;
            }
            el->first_endpoints = (int *)malloc((size_t)m * sizeof(int));
            el->second_endpoints = (int *)malloc((size_t)m * sizeof(int));
            if (!el->first_endpoints || !el->second_endpoints) {
                free(el->first_endpoints); free(el->second_endpoints);
                PyBuffer_Release(&buf);
                PyErr_NoMemory();
                return -1;
            }
            if (is_i32) {
                int *data = (int *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int u_idx = intmap_get(im, (long)data[i * 2]);
                    int v_idx = intmap_get(im, (long)data[i * 2 + 1]);
                    if (u_idx < 0 || v_idx < 0) {
                        free(el->first_endpoints); free(el->second_endpoints);
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    el->first_endpoints[i] = u_idx;
                    el->second_endpoints[i] = v_idx;
                }
            } else {  /* int64 */
                long long *data = (long long *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int u_idx = intmap_get(im, (long)data[i * 2]);
                    int v_idx = intmap_get(im, (long)data[i * 2 + 1]);
                    if (u_idx < 0 || v_idx < 0) {
                        free(el->first_endpoints); free(el->second_endpoints);
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    el->first_endpoints[i] = u_idx;
                    el->second_endpoints[i] = v_idx;
                }
            }
            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();
    }

    /* Slow path: list of tuples */
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!fast) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);
    el->edge_count = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->first_endpoints = NULL; el->second_endpoints = NULL;
        Py_DECREF(fast);
        return 0;
    }

    el->first_endpoints = (int *)malloc((size_t)m * sizeof(int));
    el->second_endpoints = (int *)malloc((size_t)m * sizeof(int));
    if (!el->first_endpoints || !el->second_endpoints) {
        free(el->first_endpoints); free(el->second_endpoints);
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *edge = items[i];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(el->first_endpoints); free(el->second_endpoints);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **ep = PySequence_Fast_ITEMS(edge_fast);
        long u_nid = PyLong_AsLong(ep[0]);
        long v_nid = PyLong_AsLong(ep[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(el->first_endpoints); free(el->second_endpoints);
            Py_DECREF(fast);
            return -1;
        }
        int u_idx = intmap_get(im, u_nid);
        int v_idx = intmap_get(im, v_nid);
        if (u_idx < 0 || v_idx < 0) {
            free(el->first_endpoints); free(el->second_endpoints);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        el->first_endpoints[i] = u_idx;
        el->second_endpoints[i] = v_idx;
    }
    Py_DECREF(fast);
    return 0;
}

/* ── Edge parsing from two separate src/dst sequences or 1D arrays ── */

static int parse_endpoint_array(IntMap *im, PyObject *obj, int *out, Py_ssize_t m,
                            const char *label) {
    /* Try buffer protocol: 1D int32 or int64 array */
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_i32 = is_int32_format(buf.format, buf.itemsize);
        int is_i64 = is_int64_format(buf.format, buf.itemsize);
        if ((is_i32 || is_i64) && buf.ndim == 1 && buf.shape[0] == m) {
            if (is_i32) {
                int *data = (int *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int idx = intmap_get(im, (long)data[i]);
                    if (idx < 0) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[i] = idx;
                }
            } else {
                long long *data = (long long *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int idx = intmap_get(im, (long)data[i]);
                    if (idx < 0) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[i] = idx;
                }
            }
            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
    } else {
        PyErr_Clear();
    }

    /* Slow path: sequence of ints */
    PyObject *fast = PySequence_Fast(obj, label);
    if (!fast) return -1;
    if (PySequence_Fast_GET_SIZE(fast) != m) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "src and dst must have the same length");
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        long nid = PyLong_AsLong(items[i]);
        if (nid == -1 && PyErr_Occurred()) { Py_DECREF(fast); return -1; }
        int idx = intmap_get(im, nid);
        if (idx < 0) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        out[i] = idx;
    }
    Py_DECREF(fast);
    return 0;
}

static int parse_edges_mapped_split(IntMap *im, PyObject *src_obj, PyObject *dst_obj,
                                    EdgeList *el) {
    /* Determine m from src */
    Py_ssize_t m;
    Py_buffer buf;
    if (PyObject_GetBuffer(src_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        m = (buf.ndim == 1) ? buf.shape[0] : -1;
        PyBuffer_Release(&buf);
        if (m < 0) {
            PyErr_SetString(PyExc_ValueError, "src must be a 1D array or sequence");
            return -1;
        }
    } else {
        PyErr_Clear();
        PyObject *fast = PySequence_Fast(src_obj, "src must be a sequence");
        if (!fast) return -1;
        m = PySequence_Fast_GET_SIZE(fast);
        Py_DECREF(fast);
    }

    el->edge_count = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->first_endpoints = NULL; el->second_endpoints = NULL;
        return 0;
    }

    el->first_endpoints = (int *)malloc((size_t)m * sizeof(int));
    el->second_endpoints = (int *)malloc((size_t)m * sizeof(int));
    if (!el->first_endpoints || !el->second_endpoints) {
        free(el->first_endpoints); free(el->second_endpoints);
        PyErr_NoMemory();
        return -1;
    }

    if (parse_endpoint_array(im, src_obj, el->first_endpoints, m, "src must be a sequence") < 0) {
        free(el->first_endpoints); free(el->second_endpoints);
        return -1;
    }
    if (parse_endpoint_array(im, dst_obj, el->second_endpoints, m, "dst must be a sequence") < 0) {
        free(el->first_endpoints); free(el->second_endpoints);
        return -1;
    }
    return 0;
}

int node_id_parse(PyObject *node_ids_obj, PyObject *edges_obj, NodeIdContext *ctx) {
    ctx->node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!ctx->node_ids_fast) return -1;
    ctx->node_count = (int)PySequence_Fast_GET_SIZE(ctx->node_ids_fast);
    ctx->node_id_items = PySequence_Fast_ITEMS(ctx->node_ids_fast);
    ctx->node_index_by_node_id.keys = NULL;
    ctx->node_index_by_node_id.values = NULL;
    ctx->edges.first_endpoints = NULL; ctx->edges.second_endpoints = NULL;
    ctx->edges.edge_count = 0; ctx->edges.owns_memory = 0;

    if (ctx->node_count == 0) return 0;

    if (intmap_init(&ctx->node_index_by_node_id, ctx->node_count) < 0) {
        Py_DECREF(ctx->node_ids_fast);
        return -1;
    }
    for (int i = 0; i < ctx->node_count; i++) {
        long nid = PyLong_AsLong(ctx->node_id_items[i]);
        if (nid == -1 && PyErr_Occurred()) {
            intmap_free(&ctx->node_index_by_node_id);
            Py_DECREF(ctx->node_ids_fast);
            return -1;
        }
        if (intmap_get(&ctx->node_index_by_node_id, nid) != -1) {
            PyErr_Format(PyExc_ValueError, "duplicate node id %ld", nid);
            intmap_free(&ctx->node_index_by_node_id);
            Py_DECREF(ctx->node_ids_fast);
            return -1;
        }
        intmap_put(&ctx->node_index_by_node_id, nid, i);
    }

    if (parse_edges_mapped(&ctx->node_index_by_node_id, edges_obj, &ctx->edges) < 0) {
        intmap_free(&ctx->node_index_by_node_id);
        Py_DECREF(ctx->node_ids_fast);
        return -1;
    }
    return 0;
}

int node_id_parse_split(PyObject *node_ids_obj, PyObject *src_obj, PyObject *dst_obj,
                    NodeIdContext *ctx) {
    ctx->node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!ctx->node_ids_fast) return -1;
    ctx->node_count = (int)PySequence_Fast_GET_SIZE(ctx->node_ids_fast);
    ctx->node_id_items = PySequence_Fast_ITEMS(ctx->node_ids_fast);
    ctx->node_index_by_node_id.keys = NULL;
    ctx->node_index_by_node_id.values = NULL;
    ctx->edges.first_endpoints = NULL; ctx->edges.second_endpoints = NULL;
    ctx->edges.edge_count = 0; ctx->edges.owns_memory = 0;

    if (ctx->node_count == 0) return 0;

    if (intmap_init(&ctx->node_index_by_node_id, ctx->node_count) < 0) {
        Py_DECREF(ctx->node_ids_fast);
        return -1;
    }
    for (int i = 0; i < ctx->node_count; i++) {
        long nid = PyLong_AsLong(ctx->node_id_items[i]);
        if (nid == -1 && PyErr_Occurred()) {
            intmap_free(&ctx->node_index_by_node_id);
            Py_DECREF(ctx->node_ids_fast);
            return -1;
        }
        if (intmap_get(&ctx->node_index_by_node_id, nid) != -1) {
            PyErr_Format(PyExc_ValueError, "duplicate node id %ld", nid);
            intmap_free(&ctx->node_index_by_node_id);
            Py_DECREF(ctx->node_ids_fast);
            return -1;
        }
        intmap_put(&ctx->node_index_by_node_id, nid, i);
    }

    if (parse_edges_mapped_split(&ctx->node_index_by_node_id, src_obj, dst_obj, &ctx->edges) < 0) {
        intmap_free(&ctx->node_index_by_node_id);
        Py_DECREF(ctx->node_ids_fast);
        return -1;
    }
    return 0;
}

/* ── Edge mask helper ── */

int parse_mask(PyObject *mask_obj, Py_ssize_t expected_len,
               const uint8_t **mask_out, Py_buffer *mask_buf) {
    if (mask_obj == NULL || mask_obj == Py_None) {
        *mask_out = NULL;
        mask_buf->obj = NULL;
        return 0;
    }
    if (PyObject_GetBuffer(mask_obj, mask_buf, PyBUF_SIMPLE) < 0)
        return -1;
    if (mask_buf->len < expected_len) {
        PyBuffer_Release(mask_buf);
        PyErr_SetString(PyExc_ValueError, "mask length must be >= edge count");
        return -1;
    }
    *mask_out = (const uint8_t *)mask_buf->buf;
    return 0;
}

static int check_weight_count(WeightList *wl, Py_ssize_t expected_m) {
    if (wl->count == expected_m) return 0;
    PyErr_Format(PyExc_ValueError,
                 "weights length %zd does not match edge count %zd",
                 wl->count, expected_m);
    if (wl->owns_memory) free(wl->values);
    wl->values = NULL;
    return -1;
}

int parse_weights(PyObject *obj, WeightList *wl, Py_ssize_t expected_m) {
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        if (buf.format && strcmp(buf.format, "d") == 0 &&
            buf.ndim == 1 && buf.itemsize == 8) {
            wl->values = (double *)buf.buf;
            wl->count = buf.shape[0];
            wl->owns_memory = 0;
            PyBuffer_Release(&buf);
            return check_weight_count(wl, expected_m);
        }
        PyBuffer_Release(&buf);
    } else {
        PyErr_Clear();
    }
    PyObject *fast = PySequence_Fast(obj, "weights must be a sequence");
    if (!fast) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    wl->count = n;
    wl->owns_memory = 1;
    if (n == 0) { wl->values = NULL; Py_DECREF(fast); return check_weight_count(wl, expected_m); }
    wl->values = (double *)malloc((size_t)n * sizeof(double));
    if (!wl->values) { Py_DECREF(fast); PyErr_NoMemory(); return -1; }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        wl->values[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) { free(wl->values); Py_DECREF(fast); return -1; }
    }
    Py_DECREF(fast);
    return check_weight_count(wl, expected_m);
}

/*
 * Labels buffer with exactly one int32 per node index: an int32 formatted
 * buffer (memoryview cast to 'i', array('i'), numpy int32) or raw bytes of
 * length 4 * n. Returns 0 on success, -1 with an exception set otherwise.
 */
int parse_labels(PyObject *labels_obj, int n, const int32_t **labels_out, Py_buffer *labels_buf) {
    if (PyObject_GetBuffer(labels_obj, labels_buf, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0)
        return -1;
    int byte_format = labels_buf->itemsize == 1;
    if (!byte_format && !is_int32_format(labels_buf->format, labels_buf->itemsize)) {
        PyBuffer_Release(labels_buf);
        PyErr_SetString(PyExc_TypeError, "labels must be an int32 buffer");
        return -1;
    }
    if (labels_buf->len != (Py_ssize_t)n * (Py_ssize_t)sizeof(int32_t)) {
        PyBuffer_Release(labels_buf);
        PyErr_Format(PyExc_ValueError,
                     "labels must hold one int32 per node: expected %d entries, got %zd bytes",
                     n, labels_buf->len);
        return -1;
    }
    *labels_out = (const int32_t *)labels_buf->buf;
    return 0;
}
