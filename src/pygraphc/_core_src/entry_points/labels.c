#include "entry_points.h"
#include "../conversion/parse_terminals.h"
#include "../conversion/parse.h"
#include "../algorithms/biconnectivity.h"
#include "../algorithms/labels.h"

/* ── Label kernels: int32 byte arrays instead of Python containers ──
 *
 * These return raw int32 buffers indexed by node or edge index, so a result
 * over a million nodes is one allocation and no Python object per node.
 */

/* component_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[node_count]
 *
 * labels[i] is the smallest node index of the connected component of node i
 * under the masks, -1 for an excluded node.
 */
PyObject *py_component_labels_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)node_count * (Py_ssize_t)sizeof(int32_t));
    if (!result || node_count == 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return result; }

    ComponentResult components;
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        Py_DECREF(result); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }
    int status = compute_graph_components(graph, edge_mask, &terminal_mask, node_mask, &components);
    release_terminal_mask(&terminal_mask);
    release_mask(&edge_mask_buffer);
    if (status == 0) {
        status = smallest_node_labels(&components, node_count, node_mask, (int32_t *)PyBytes_AS_STRING(result));
        free(components.labels);
    }
    release_mask(&node_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
    return result;
}

/* quotient_edges_ctx(capsule, labels[, edge_mask])
 *     -> (from_labels, to_labels, edge_indices, internal_edge_indices), each bytes of int32
 *
 * Every edge is reported by its index; nothing is keyed by endpoint pair.
 */
PyObject *py_quotient_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *labels_obj, *edge_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &capsule, &labels_obj, &edge_mask_obj)) return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;

    const int32_t *labels; Py_buffer labels_buffer;
    if (parse_labels(labels_obj, node_count, &labels, &labels_buffer) < 0) return NULL;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { PyBuffer_Release(&labels_buffer); return NULL; }

    size_t capacity = (size_t)(edge_count > 0 ? edge_count : 1);
    int32_t *scratch = (int32_t *)malloc(4 * capacity * sizeof(int32_t));
    if (!scratch) { release_mask(&edge_mask_buffer); PyBuffer_Release(&labels_buffer); PyErr_NoMemory(); return NULL; }
    int32_t *from_labels = scratch, *to_labels = scratch + capacity;
    int32_t *crossing_edges = scratch + 2 * capacity, *internal_edges = scratch + 3 * capacity;
    Py_ssize_t crossing_count, internal_count;
    split_quotient_edges(&graph->node_ids.edges, labels, edge_mask, from_labels, to_labels, crossing_edges, internal_edges,
                         &crossing_count, &internal_count);
    release_mask(&edge_mask_buffer); PyBuffer_Release(&labels_buffer);

    Py_ssize_t crossing_bytes = crossing_count * (Py_ssize_t)sizeof(int32_t);
    Py_ssize_t internal_bytes = internal_count * (Py_ssize_t)sizeof(int32_t);
    PyObject *result = Py_BuildValue("(y#y#y#y#)",
                                     (const char *)from_labels, crossing_bytes,
                                     (const char *)to_labels, crossing_bytes,
                                     (const char *)crossing_edges, crossing_bytes,
                                     (const char *)internal_edges, internal_bytes);
    free(scratch);
    return result;
}

/* degrees_ctx(capsule[, edge_mask, node_mask, terminal_mask]) -> bytes of int32[node_count]
 *
 * The degree of every node under the masks, as degree_ctx counts it; for a
 * directed graph the out-degree. One pass over the terminals.
 */
PyObject *py_degrees_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }

    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)node_count * (Py_ssize_t)sizeof(int32_t));
    if (result && node_count > 0)
        joined_terminal_degrees(graph, two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask, terminal_mask.terminals,
                                node_mask, (int32_t *)PyBytes_AS_STRING(result));
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* biconnected_component_edge_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[edge_count] */
PyObject *py_biconnected_component_edge_labels_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;

    PyObject *result = PyBytes_FromStringAndSize(NULL, edge_count * (Py_ssize_t)sizeof(int32_t));
    if (!result) return NULL;
    int32_t *labels = (int32_t *)PyBytes_AS_STRING(result);
    if (node_count == 0 || edge_count == 0 || !graph->has_adjacency) {
        memset(labels, -1, (size_t)edge_count * sizeof(int32_t));
        return result;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    int status = biconnected_edge_labels(graph, edge_mask, node_mask, labels);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
    return result;
}
