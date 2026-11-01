#include "entry_points.h"
#include "../helpers/terminals.h"
#include "../helpers/parse.h"
#include "../helpers/components.h"

/* ── Label kernels: int32 byte arrays instead of Python containers ──
 *
 * These return raw int32 buffers indexed by node or edge index, so a result
 * over a million nodes is one allocation and no Python object per node.
 */

/* component_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[node_count]
 *
 * labels[i] is the smallest node index of the connected component of node i
 * under the masks, -1 for an excluded node. Component ids of
 * compute_components_masked are numbered in order of first appearance while
 * scanning node indices upwards, so the first index carrying an id is the
 * smallest index of its component.
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
    if (status < 0) {
        Py_DECREF(result); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }
    release_mask(&edge_mask_buffer);

    int *smallest_index = (int *)malloc((size_t)components.component_count * sizeof(int));
    if (!smallest_index) {
        free(components.labels); Py_DECREF(result); release_mask(&node_mask_buffer); PyErr_NoMemory(); return NULL;
    }
    memset(smallest_index, -1, (size_t)components.component_count * sizeof(int));
    int32_t *labels = (int32_t *)PyBytes_AS_STRING(result);
    for (int node = 0; node < node_count; node++) {
        int component = components.labels[node];
        if (smallest_index[component] == -1) smallest_index[component] = node;
        labels[node] = (node_mask && node_mask[node]) ? -1 : (int32_t)smallest_index[component];
    }
    free(smallest_index); free(components.labels);
    release_mask(&node_mask_buffer);
    return result;
}

/* quotient_edges_ctx(capsule, labels[, edge_mask])
 *     -> (from_labels, to_labels, edge_indices, internal_edge_indices), each bytes of int32
 *
 * One pass over the edges in index order. An unmasked edge whose endpoints
 * carry different labels is a crossing edge and fills the three parallel
 * arrays; one whose endpoints carry the same label is internal. Edges at a
 * node labelled -1 (excluded) are neither. Every edge is reported by its
 * index; nothing is keyed by endpoint pair.
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

    /* Four scratch arrays of edge_count entries each; at least one entry so the pointers are never NULL. */
    size_t capacity = (size_t)(edge_count > 0 ? edge_count : 1);
    int32_t *scratch = (int32_t *)malloc(4 * capacity * sizeof(int32_t));
    if (!scratch) { release_mask(&edge_mask_buffer); PyBuffer_Release(&labels_buffer); PyErr_NoMemory(); return NULL; }
    int32_t *from_labels = scratch, *to_labels = scratch + capacity;
    int32_t *crossing_edges = scratch + 2 * capacity, *internal_edges = scratch + 3 * capacity;
    Py_ssize_t crossing_count = 0, internal_count = 0;

    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        if (edge_mask && edge_mask[edge_index]) continue;
        int32_t from_label = labels[EDGE_FIRST_ENDPOINT(&graph->node_ids.edges, edge_index)];
        int32_t to_label = labels[EDGE_SECOND_ENDPOINT(&graph->node_ids.edges, edge_index)];
        if (from_label < 0 || to_label < 0) continue;
        if (from_label == to_label) { internal_edges[internal_count++] = (int32_t)edge_index; continue; }
        from_labels[crossing_count] = from_label;
        to_labels[crossing_count] = to_label;
        crossing_edges[crossing_count++] = (int32_t)edge_index;
    }
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

/*
 * Add each joined terminal of every edge to the degree of its node: undirected
 * the active terminals of an edge with at least two, directed the active
 * in-terminals of an edge with an active out-terminal (the out-degree). A
 * two-terminal edge without a terminal list has terminals 2e and 2e + 1 at its
 * endpoints, a masked one folded into two_terminal_edges. Instantiated for both
 * values of has_terminal_list.
 */
static inline __attribute__((always_inline)) void add_joined_terminals(
    const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, int32_t *degrees, const int has_terminal_list) {
    const TerminalList *terminals = graph->terminals;
    const EdgeList *edges = &graph->node_ids.edges;
    for (Py_ssize_t edge = 0; edge < edges->edge_count; edge++) {
        if (!has_terminal_list) {
            if (two_terminal_edges && two_terminal_edges[edge]) continue;
            int first_node = EDGE_FIRST_ENDPOINT(edges, edge), second_node = EDGE_SECOND_ENDPOINT(edges, edge);
            if (node_mask && (node_mask[first_node] || node_mask[second_node])) continue;
            degrees[first_node]++;
            if (!graph->directed) degrees[second_node]++;
            continue;
        }
        int first = terminals->terminal_offsets[edge], end = terminals->terminal_offsets[edge + 1];
        int out_start = graph->directed ? terminals->out_terminal_offsets[edge] : end;
        int active_in = 0, active_out = 0;
        for (int terminal = first; terminal < end; terminal++) {
            if (!terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask)) continue;
            if (terminal < out_start) active_in++;
            else active_out++;
        }
        if (graph->directed ? (active_in == 0 || active_out == 0) : active_in < 2) continue;
        for (int terminal = first; terminal < out_start; terminal++)
            if (terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask))
                degrees[terminals->terminal_nodes[terminal]]++;
    }
}

/* degrees_ctx(capsule[, edge_mask, node_mask, terminal_mask]) -> bytes of int32[node_count]
 *
 * The degree of every node under the masks, as degree_ctx counts it: the joined
 * terminals at the node, a self-loop counting twice, 0 for an excluded node. For
 * a directed graph the out-degree. One pass over the terminals.
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
    if (result && node_count > 0) {
        int32_t *degrees = (int32_t *)PyBytes_AS_STRING(result);
        memset(degrees, 0, (size_t)node_count * sizeof(int32_t));
        const uint8_t *two_terminal_edges = two_terminal_edge_mask(&terminal_mask, edge_mask);
        if (graph->terminals)
            add_joined_terminals(graph, two_terminal_edges, edge_mask, terminal_mask.terminals, node_mask, degrees, 1);
        else
            add_joined_terminals(graph, two_terminal_edges, NULL, NULL, node_mask, degrees, 0);
    }
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* biconnected_component_edge_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[edge_count]
 *
 * Biconnected component id of every edge, numbered in the order the
 * components complete in the depth-first search. A bridge is a singleton
 * component. Masked edges, edges at an excluded node and self-loops get -1.
 * Same iterative Tarjan as biconnected_components_ctx with an edge stack of edge indices.
 */
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
    memset(labels, -1, (size_t)edge_count * sizeof(int32_t));
    if (node_count == 0 || edge_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    int *edge_stack = (int *)malloc((size_t)edge_count * sizeof(int));
    if (!scratch || !edge_stack) {
        free(scratch); free(edge_stack); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));
    int timer = 0, edge_stack_top = 0, component_id = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    edge_stack[edge_stack_top++] = edge_index;
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else if (discovery_time[neighbor] < discovery_time[node]) {
                    edge_stack[edge_stack_top++] = edge_index;
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] >= discovery_time[parent]) {
                        while (edge_stack_top > 0) {
                            int popped_edge = edge_stack[--edge_stack_top];
                            labels[popped_edge] = (int32_t)component_id;
                            if (popped_edge == parent_edge[node]) break;
                        }
                        component_id++;
                    }
                }
                stack_top--;
            }
        }
    }
    free(scratch); free(edge_stack);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}
