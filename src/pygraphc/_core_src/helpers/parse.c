/* Parsing helpers: Python objects and buffers into the data models. */
#include "parse.h"

/*
 * Try the buffer protocol first (numpy int32 array with shape (edge_count, 2)).
 * Falls back to sequence-of-tuples parsing.
 * Returns 0 on success, -1 on error.
 */
int parse_edges(PyObject *edges_obj, EdgeList *edges) {
    Py_buffer buffer;

    /* Fast path: buffer protocol (numpy array) */
    if (PyObject_GetBuffer(edges_obj, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        /* Validate: must be int32 with shape (edge_count, 2) */
        int is_int32 = (buffer.format != NULL && (
            strcmp(buffer.format, "i") == 0 ||
            strcmp(buffer.format, "<i") == 0 ||
            strcmp(buffer.format, "=i") == 0
        ) && buffer.itemsize == 4);

        if (is_int32 && buffer.ndim == 2 && buffer.shape[1] == 2) {
            Py_ssize_t edge_count = buffer.shape[0];
            int *pairs = (int *)buffer.buf;
            edges->edge_count = edge_count;
            /* Point directly into the numpy buffer — zero copy for edges */
            edges->first_endpoints = NULL;
            edges->second_endpoints = NULL;
            edges->owns_memory = 0;

            /* We need separate endpoint arrays for the union-find loop,
               but we can read directly from the contiguous buffer:
               pairs[edge_index*2] = first endpoint, pairs[edge_index*2+1] = second endpoint */

            /* Actually, just store the raw pointer and read inline.
               We'll use a special code path. Store raw ptr in first_endpoints. */
            edges->first_endpoints = pairs;  /* raw (edge_count, 2) int32 buffer */
            edges->second_endpoints = NULL;  /* sentinel: means "interleaved" */

            PyBuffer_Release(&buffer);
            return 0;
        }
        PyBuffer_Release(&buffer);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();  /* buffer protocol not supported, try sequence */
    }

    /* Slow path: list of tuples */
    PyObject *edges_fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!edges_fast) return -1;

    Py_ssize_t edge_count = PySequence_Fast_GET_SIZE(edges_fast);
    edges->edge_count = edge_count;
    edges->owns_memory = 1;
    if (edge_count == 0) {
        edges->first_endpoints = NULL;
        edges->second_endpoints = NULL;
        Py_DECREF(edges_fast);
        return 0;
    }

    edges->first_endpoints = (int *)malloc(edge_count * sizeof(int));
    edges->second_endpoints = (int *)malloc(edge_count * sizeof(int));
    if (!edges->first_endpoints || !edges->second_endpoints) {
        free(edges->first_endpoints); free(edges->second_endpoints);
        Py_DECREF(edges_fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        PyObject *edge = edge_items[edge_index];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(edges->first_endpoints); free(edges->second_endpoints);
            Py_DECREF(edges_fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **endpoint_items = PySequence_Fast_ITEMS(edge_fast);
        edges->first_endpoints[edge_index] = (int)PyLong_AsLong(endpoint_items[0]);
        edges->second_endpoints[edge_index] = (int)PyLong_AsLong(endpoint_items[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(edges->first_endpoints); free(edges->second_endpoints);
            Py_DECREF(edges_fast);
            return -1;
        }
    }
    Py_DECREF(edges_fast);
    return 0;
}

/* ── Edge parsing with node-ID translation via C hash map ── */

static int parse_edges_mapped(IntMap *node_index_by_node_id, PyObject *edges_obj, EdgeList *edges) {
    /* Fast path: buffer protocol (numpy int32/int64 array with shape (edge_count, 2)) */
    Py_buffer buffer;
    if (PyObject_GetBuffer(edges_obj, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_int32 = is_int32_format(buffer.format, buffer.itemsize);
        int is_int64 = is_int64_format(buffer.format, buffer.itemsize);

        if ((is_int32 || is_int64) && buffer.ndim == 2 && buffer.shape[1] == 2) {
            Py_ssize_t edge_count = buffer.shape[0];
            edges->edge_count = edge_count;
            edges->owns_memory = 1;
            if (edge_count == 0) {
                edges->first_endpoints = NULL; edges->second_endpoints = NULL;
                PyBuffer_Release(&buffer);
                return 0;
            }
            edges->first_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
            edges->second_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
            if (!edges->first_endpoints || !edges->second_endpoints) {
                free(edges->first_endpoints); free(edges->second_endpoints);
                PyBuffer_Release(&buffer);
                PyErr_NoMemory();
                return -1;
            }
            if (is_int32) {
                int *pairs = (int *)buffer.buf;
                for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
                    int first_index = intmap_get(node_index_by_node_id, (long)pairs[edge_index * 2]);
                    int second_index = intmap_get(node_index_by_node_id, (long)pairs[edge_index * 2 + 1]);
                    if (first_index < 0 || second_index < 0) {
                        free(edges->first_endpoints); free(edges->second_endpoints);
                        PyBuffer_Release(&buffer);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    edges->first_endpoints[edge_index] = first_index;
                    edges->second_endpoints[edge_index] = second_index;
                }
            } else {  /* int64 */
                long long *pairs = (long long *)buffer.buf;
                for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
                    int first_index = intmap_get(node_index_by_node_id, (long)pairs[edge_index * 2]);
                    int second_index = intmap_get(node_index_by_node_id, (long)pairs[edge_index * 2 + 1]);
                    if (first_index < 0 || second_index < 0) {
                        free(edges->first_endpoints); free(edges->second_endpoints);
                        PyBuffer_Release(&buffer);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    edges->first_endpoints[edge_index] = first_index;
                    edges->second_endpoints[edge_index] = second_index;
                }
            }
            PyBuffer_Release(&buffer);
            return 0;
        }
        PyBuffer_Release(&buffer);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();
    }

    /* Slow path: list of tuples */
    PyObject *edges_fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!edges_fast) return -1;

    Py_ssize_t edge_count = PySequence_Fast_GET_SIZE(edges_fast);
    edges->edge_count = edge_count;
    edges->owns_memory = 1;
    if (edge_count == 0) {
        edges->first_endpoints = NULL; edges->second_endpoints = NULL;
        Py_DECREF(edges_fast);
        return 0;
    }

    edges->first_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
    edges->second_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
    if (!edges->first_endpoints || !edges->second_endpoints) {
        free(edges->first_endpoints); free(edges->second_endpoints);
        Py_DECREF(edges_fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        PyObject *edge = edge_items[edge_index];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(edges->first_endpoints); free(edges->second_endpoints);
            Py_DECREF(edges_fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **endpoint_items = PySequence_Fast_ITEMS(edge_fast);
        long first_node_id = PyLong_AsLong(endpoint_items[0]);
        long second_node_id = PyLong_AsLong(endpoint_items[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(edges->first_endpoints); free(edges->second_endpoints);
            Py_DECREF(edges_fast);
            return -1;
        }
        int first_index = intmap_get(node_index_by_node_id, first_node_id);
        int second_index = intmap_get(node_index_by_node_id, second_node_id);
        if (first_index < 0 || second_index < 0) {
            free(edges->first_endpoints); free(edges->second_endpoints);
            Py_DECREF(edges_fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        edges->first_endpoints[edge_index] = first_index;
        edges->second_endpoints[edge_index] = second_index;
    }
    Py_DECREF(edges_fast);
    return 0;
}

/* ── Edge parsing from two separate first/second endpoint sequences or 1D arrays ── */

static int parse_endpoint_array(IntMap *node_index_by_node_id, PyObject *endpoints_obj, int *out, Py_ssize_t edge_count,
                            const char *error_message) {
    /* Try buffer protocol: 1D int32 or int64 array */
    Py_buffer buffer;
    if (PyObject_GetBuffer(endpoints_obj, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_int32 = is_int32_format(buffer.format, buffer.itemsize);
        int is_int64 = is_int64_format(buffer.format, buffer.itemsize);
        if ((is_int32 || is_int64) && buffer.ndim == 1 && buffer.shape[0] == edge_count) {
            if (is_int32) {
                int *values = (int *)buffer.buf;
                for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
                    int idx = intmap_get(node_index_by_node_id, (long)values[edge_index]);
                    if (idx < 0) {
                        PyBuffer_Release(&buffer);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[edge_index] = idx;
                }
            } else {
                long long *values = (long long *)buffer.buf;
                for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
                    int idx = intmap_get(node_index_by_node_id, (long)values[edge_index]);
                    if (idx < 0) {
                        PyBuffer_Release(&buffer);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[edge_index] = idx;
                }
            }
            PyBuffer_Release(&buffer);
            return 0;
        }
        PyBuffer_Release(&buffer);
    } else {
        PyErr_Clear();
    }

    /* Slow path: sequence of ints */
    PyObject *endpoints_fast = PySequence_Fast(endpoints_obj, error_message);
    if (!endpoints_fast) return -1;
    if (PySequence_Fast_GET_SIZE(endpoints_fast) != edge_count) {
        Py_DECREF(endpoints_fast);
        PyErr_SetString(PyExc_ValueError, "src and dst must have the same length");
        return -1;
    }
    PyObject **endpoint_items = PySequence_Fast_ITEMS(endpoints_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        long node_id = PyLong_AsLong(endpoint_items[edge_index]);
        if (node_id == -1 && PyErr_Occurred()) { Py_DECREF(endpoints_fast); return -1; }
        int idx = intmap_get(node_index_by_node_id, node_id);
        if (idx < 0) {
            Py_DECREF(endpoints_fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        out[edge_index] = idx;
    }
    Py_DECREF(endpoints_fast);
    return 0;
}

static int parse_edges_mapped_split(IntMap *node_index_by_node_id, PyObject *first_endpoints_obj, PyObject *second_endpoints_obj,
                                    EdgeList *edges) {
    /* Determine edge_count from the first endpoints */
    Py_ssize_t edge_count;
    Py_buffer buffer;
    if (PyObject_GetBuffer(first_endpoints_obj, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        edge_count = (buffer.ndim == 1) ? buffer.shape[0] : -1;
        PyBuffer_Release(&buffer);
        if (edge_count < 0) {
            PyErr_SetString(PyExc_ValueError, "src must be a 1D array or sequence");
            return -1;
        }
    } else {
        PyErr_Clear();
        PyObject *endpoints_fast = PySequence_Fast(first_endpoints_obj, "src must be a sequence");
        if (!endpoints_fast) return -1;
        edge_count = PySequence_Fast_GET_SIZE(endpoints_fast);
        Py_DECREF(endpoints_fast);
    }

    edges->edge_count = edge_count;
    edges->owns_memory = 1;
    if (edge_count == 0) {
        edges->first_endpoints = NULL; edges->second_endpoints = NULL;
        return 0;
    }

    edges->first_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
    edges->second_endpoints = (int *)malloc((size_t)edge_count * sizeof(int));
    if (!edges->first_endpoints || !edges->second_endpoints) {
        free(edges->first_endpoints); free(edges->second_endpoints);
        PyErr_NoMemory();
        return -1;
    }

    if (parse_endpoint_array(node_index_by_node_id, first_endpoints_obj, edges->first_endpoints, edge_count, "src must be a sequence") < 0) {
        free(edges->first_endpoints); free(edges->second_endpoints);
        return -1;
    }
    if (parse_endpoint_array(node_index_by_node_id, second_endpoints_obj, edges->second_endpoints, edge_count, "dst must be a sequence") < 0) {
        free(edges->first_endpoints); free(edges->second_endpoints);
        return -1;
    }
    return 0;
}

int node_id_parse(PyObject *node_ids_obj, PyObject *edges_obj, NodeIdContext *context) {
    context->node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!context->node_ids_fast) return -1;
    context->node_count = (int)PySequence_Fast_GET_SIZE(context->node_ids_fast);
    context->node_id_items = PySequence_Fast_ITEMS(context->node_ids_fast);
    context->node_index_by_node_id.keys = NULL;
    context->node_index_by_node_id.values = NULL;
    context->edges.first_endpoints = NULL; context->edges.second_endpoints = NULL;
    context->edges.edge_count = 0; context->edges.owns_memory = 0;

    if (context->node_count == 0) return 0;

    if (intmap_init(&context->node_index_by_node_id, context->node_count) < 0) {
        Py_DECREF(context->node_ids_fast);
        return -1;
    }
    for (int node = 0; node < context->node_count; node++) {
        long node_id = PyLong_AsLong(context->node_id_items[node]);
        if (node_id == -1 && PyErr_Occurred()) {
            intmap_free(&context->node_index_by_node_id);
            Py_DECREF(context->node_ids_fast);
            return -1;
        }
        if (intmap_get(&context->node_index_by_node_id, node_id) != -1) {
            PyErr_Format(PyExc_ValueError, "duplicate node id %ld", node_id);
            intmap_free(&context->node_index_by_node_id);
            Py_DECREF(context->node_ids_fast);
            return -1;
        }
        intmap_put(&context->node_index_by_node_id, node_id, node);
    }

    if (parse_edges_mapped(&context->node_index_by_node_id, edges_obj, &context->edges) < 0) {
        intmap_free(&context->node_index_by_node_id);
        Py_DECREF(context->node_ids_fast);
        return -1;
    }
    return 0;
}

int node_id_parse_split(PyObject *node_ids_obj, PyObject *first_endpoints_obj, PyObject *second_endpoints_obj,
                    NodeIdContext *context) {
    context->node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!context->node_ids_fast) return -1;
    context->node_count = (int)PySequence_Fast_GET_SIZE(context->node_ids_fast);
    context->node_id_items = PySequence_Fast_ITEMS(context->node_ids_fast);
    context->node_index_by_node_id.keys = NULL;
    context->node_index_by_node_id.values = NULL;
    context->edges.first_endpoints = NULL; context->edges.second_endpoints = NULL;
    context->edges.edge_count = 0; context->edges.owns_memory = 0;

    if (context->node_count == 0) return 0;

    if (intmap_init(&context->node_index_by_node_id, context->node_count) < 0) {
        Py_DECREF(context->node_ids_fast);
        return -1;
    }
    for (int node = 0; node < context->node_count; node++) {
        long node_id = PyLong_AsLong(context->node_id_items[node]);
        if (node_id == -1 && PyErr_Occurred()) {
            intmap_free(&context->node_index_by_node_id);
            Py_DECREF(context->node_ids_fast);
            return -1;
        }
        if (intmap_get(&context->node_index_by_node_id, node_id) != -1) {
            PyErr_Format(PyExc_ValueError, "duplicate node id %ld", node_id);
            intmap_free(&context->node_index_by_node_id);
            Py_DECREF(context->node_ids_fast);
            return -1;
        }
        intmap_put(&context->node_index_by_node_id, node_id, node);
    }

    if (parse_edges_mapped_split(&context->node_index_by_node_id, first_endpoints_obj, second_endpoints_obj, &context->edges) < 0) {
        intmap_free(&context->node_index_by_node_id);
        Py_DECREF(context->node_ids_fast);
        return -1;
    }
    return 0;
}

/* ── Edge mask helper ── */

int parse_mask(PyObject *mask_obj, Py_ssize_t expected_length,
               const uint8_t **mask_out, Py_buffer *mask_buffer) {
    if (mask_obj == NULL || mask_obj == Py_None) {
        *mask_out = NULL;
        mask_buffer->obj = NULL;
        return 0;
    }
    if (PyObject_GetBuffer(mask_obj, mask_buffer, PyBUF_SIMPLE) < 0)
        return -1;
    if (mask_buffer->len < expected_length) {
        PyBuffer_Release(mask_buffer);
        PyErr_SetString(PyExc_ValueError, "mask length must be >= edge count");
        return -1;
    }
    *mask_out = (const uint8_t *)mask_buffer->buf;
    return 0;
}

static int check_weight_count(WeightList *weights, Py_ssize_t expected_edge_count) {
    if (weights->count == expected_edge_count) return 0;
    PyErr_Format(PyExc_ValueError,
                 "weights length %zd does not match edge count %zd",
                 weights->count, expected_edge_count);
    if (weights->owns_memory) free(weights->values);
    weights->values = NULL;
    return -1;
}

int parse_weights(PyObject *weights_obj, WeightList *weights, Py_ssize_t expected_edge_count) {
    Py_buffer buffer;
    if (PyObject_GetBuffer(weights_obj, &buffer, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        if (buffer.format && strcmp(buffer.format, "d") == 0 &&
            buffer.ndim == 1 && buffer.itemsize == 8) {
            weights->values = (double *)buffer.buf;
            weights->count = buffer.shape[0];
            weights->owns_memory = 0;
            PyBuffer_Release(&buffer);
            return check_weight_count(weights, expected_edge_count);
        }
        PyBuffer_Release(&buffer);
    } else {
        PyErr_Clear();
    }
    PyObject *weights_fast = PySequence_Fast(weights_obj, "weights must be a sequence");
    if (!weights_fast) return -1;
    Py_ssize_t weight_count = PySequence_Fast_GET_SIZE(weights_fast);
    weights->count = weight_count;
    weights->owns_memory = 1;
    if (weight_count == 0) { weights->values = NULL; Py_DECREF(weights_fast); return check_weight_count(weights, expected_edge_count); }
    weights->values = (double *)malloc((size_t)weight_count * sizeof(double));
    if (!weights->values) { Py_DECREF(weights_fast); PyErr_NoMemory(); return -1; }
    PyObject **weight_items = PySequence_Fast_ITEMS(weights_fast);
    for (Py_ssize_t weight_index = 0; weight_index < weight_count; weight_index++) {
        weights->values[weight_index] = PyFloat_AsDouble(weight_items[weight_index]);
        if (PyErr_Occurred()) { free(weights->values); Py_DECREF(weights_fast); return -1; }
    }
    Py_DECREF(weights_fast);
    return check_weight_count(weights, expected_edge_count);
}

/*
 * Labels buffer with exactly one int32 per node index: an int32 formatted
 * buffer (memoryview cast to 'i', array('i'), numpy int32) or raw bytes of
 * length 4 * node_count. Returns 0 on success, -1 with an exception set otherwise.
 */
int parse_labels(PyObject *labels_obj, int node_count, const int32_t **labels_out, Py_buffer *labels_buffer) {
    if (PyObject_GetBuffer(labels_obj, labels_buffer, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0)
        return -1;
    int byte_format = labels_buffer->itemsize == 1;
    if (!byte_format && !is_int32_format(labels_buffer->format, labels_buffer->itemsize)) {
        PyBuffer_Release(labels_buffer);
        PyErr_SetString(PyExc_TypeError, "labels must be an int32 buffer");
        return -1;
    }
    if (labels_buffer->len != (Py_ssize_t)node_count * (Py_ssize_t)sizeof(int32_t)) {
        PyBuffer_Release(labels_buffer);
        PyErr_Format(PyExc_ValueError,
                     "labels must hold one int32 per node: expected %d entries, got %zd bytes",
                     node_count, labels_buffer->len);
        return -1;
    }
    *labels_out = (const int32_t *)labels_buffer->buf;
    return 0;
}
