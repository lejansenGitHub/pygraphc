/* Graphs with multiport edges: parsing, shape, and connected components over active ports. */
#include "entry_points.h"
#include "../_core_src/helpers/parse.h"
#include "port_list.h"

/* ── Parsing ── */

typedef struct {
    PortList *ports;
    int capacity;
} PortBuilder;

static int append_port(PortBuilder *builder, int node, int edge, int is_out) {
    PortList *ports = builder->ports;
    if (ports->port_count == builder->capacity) {
        int capacity = builder->capacity > 0 ? builder->capacity * 2 : 64;
        int *port_nodes = (int *)realloc(ports->port_nodes, (size_t)capacity * sizeof(int));
        if (port_nodes) ports->port_nodes = port_nodes;
        int *port_edges = (int *)realloc(ports->port_edges, (size_t)capacity * sizeof(int));
        if (port_edges) ports->port_edges = port_edges;
        if (!port_nodes || !port_edges) { PyErr_NoMemory(); return -1; }
        if (ports->port_is_out) {
            uint8_t *port_is_out = (uint8_t *)realloc(ports->port_is_out, (size_t)capacity);
            if (!port_is_out) { PyErr_NoMemory(); return -1; }
            ports->port_is_out = port_is_out;
        }
        builder->capacity = capacity;
    }
    ports->port_nodes[ports->port_count] = node;
    ports->port_edges[ports->port_count] = edge;
    if (ports->port_is_out) ports->port_is_out[ports->port_count] = (uint8_t)is_out;
    ports->port_count++;
    return 0;
}

static int node_of(GraphContext *graph, PyObject *item, int *node_out) {
    long node_id = PyLong_AsLong(item);
    if (node_id == -1 && PyErr_Occurred()) return -1;
    int node = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, node_id) : -1;
    if (node < 0) { PyErr_SetString(PyExc_ValueError, "edge references unknown node ID"); return -1; }
    *node_out = node;
    return 0;
}

static int append_ports_of(PortBuilder *builder, GraphContext *graph, PyObject *node_ids_obj, int edge, int is_out) {
    PyObject *node_ids_fast = PySequence_Fast(node_ids_obj, "the ports of an edge must be a sequence of node ids");
    if (!node_ids_fast) return -1;
    Py_ssize_t count = PySequence_Fast_GET_SIZE(node_ids_fast);
    if (count == 0) {
        Py_DECREF(node_ids_fast);
        PyErr_SetString(PyExc_ValueError, "an edge needs at least one port");
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(node_ids_fast);
    for (Py_ssize_t position = 0; position < count; position++) {
        int node;
        if (node_of(graph, items[position], &node) < 0 || append_port(builder, node, edge, is_out) < 0) {
            Py_DECREF(node_ids_fast);
            return -1;
        }
    }
    Py_DECREF(node_ids_fast);
    return 0;
}

/* A directed edge is (u, v), from u to v, or (in_nodes, out_nodes) with at least one of each. */
static int append_directed_edge(PortBuilder *builder, GraphContext *graph, PyObject *edge_obj, int edge) {
    PyObject *edge_fast = PySequence_Fast(edge_obj, "a directed edge is (u, v) or (in_nodes, out_nodes)");
    if (!edge_fast) return -1;
    if (PySequence_Fast_GET_SIZE(edge_fast) != 2) {
        Py_DECREF(edge_fast);
        PyErr_SetString(PyExc_ValueError, "a directed edge is (u, v) or (in_nodes, out_nodes)");
        return -1;
    }
    PyObject **sides = PySequence_Fast_ITEMS(edge_fast);
    int status;
    if (PyLong_Check(sides[0]) && PyLong_Check(sides[1])) {
        int first_node, second_node;
        status = (node_of(graph, sides[0], &first_node) < 0 || node_of(graph, sides[1], &second_node) < 0 ||
                  append_port(builder, first_node, edge, 0) < 0 || append_port(builder, second_node, edge, 1) < 0)
                     ? -1 : 0;
    } else if (!PyLong_Check(sides[0]) && !PyLong_Check(sides[1])) {
        status = (append_ports_of(builder, graph, sides[0], edge, 0) < 0 ||
                  append_ports_of(builder, graph, sides[1], edge, 1) < 0)
                     ? -1 : 0;
    } else {
        PyErr_SetString(PyExc_ValueError, "a directed edge is (u, v) or (in_nodes, out_nodes)");
        status = -1;
    }
    Py_DECREF(edge_fast);
    return status;
}

static void free_port_graph(GraphContext *graph) {
    if (graph->ports) { free_port_list(graph->ports); free(graph->ports); }
    node_id_context_free(&graph->node_ids);
    free(graph);
}

PyObject *py_parse_port_graph(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    int directed = 0;
    if (!PyArg_ParseTuple(args, "OO|p", &node_ids_obj, &edges_obj, &directed)) return NULL;

    GraphContext *graph = (GraphContext *)calloc(1, sizeof(GraphContext));
    if (!graph) { PyErr_NoMemory(); return NULL; }
    graph->directed = directed;
    PyObject *no_line_edges = PyTuple_New(0);
    if (!no_line_edges) { free(graph); return NULL; }
    int status = node_id_parse(node_ids_obj, no_line_edges, &graph->node_ids);
    Py_DECREF(no_line_edges);
    if (status < 0) { free(graph); return NULL; }

    graph->ports = (PortList *)calloc(1, sizeof(PortList));
    PyObject *edges_fast = graph->ports ? PySequence_Fast(edges_obj, "edges must be a sequence") : NULL;
    if (!edges_fast) {
        if (!graph->ports) PyErr_NoMemory();
        free_port_graph(graph);
        return NULL;
    }
    PortList *ports = graph->ports;
    ports->edge_count = (int)PySequence_Fast_GET_SIZE(edges_fast);
    ports->port_offsets = (int *)malloc(((size_t)ports->edge_count + 1) * sizeof(int));
    if (directed) ports->port_is_out = (uint8_t *)malloc(1);
    if (!ports->port_offsets || (directed && !ports->port_is_out)) {
        Py_DECREF(edges_fast); free_port_graph(graph); PyErr_NoMemory(); return NULL;
    }
    PortBuilder builder = {ports, 0};
    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (int edge = 0; edge < ports->edge_count; edge++) {
        ports->port_offsets[edge] = ports->port_count;
        status = directed ? append_directed_edge(&builder, graph, edge_items[edge], edge)
                          : append_ports_of(&builder, graph, edge_items[edge], edge, 0);
        if (status < 0) { Py_DECREF(edges_fast); free_port_graph(graph); return NULL; }
    }
    ports->port_offsets[ports->edge_count] = ports->port_count;
    Py_DECREF(edges_fast);
    if (build_port_incidence(graph->node_ids.node_count, ports) < 0) { free_port_graph(graph); return NULL; }
    return PyCapsule_New(graph, "pygraphc.GraphCtx", graph_context_destructor);
}

static GraphContext *port_graph_of(PyObject *capsule) {
    GraphContext *graph = get_graph_context(capsule);
    if (graph && !graph->ports) {
        PyErr_SetString(PyExc_TypeError, "expected a parsed graph with multiport edges");
        return NULL;
    }
    return graph;
}

/* ── Shape and port addressing ── */

PyObject *py_port_graph_shape(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = port_graph_of(capsule);
    if (!graph) return NULL;
    return Py_BuildValue("(iii)", graph->node_ids.node_count, graph->ports->edge_count, graph->ports->port_count);
}

PyObject *py_port_offsets(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphContext *graph = port_graph_of(capsule);
    if (!graph) return NULL;
    return PyBytes_FromStringAndSize((const char *)graph->ports->port_offsets,
                                     ((Py_ssize_t)graph->ports->edge_count + 1) * (Py_ssize_t)sizeof(int));
}

/* ── Masks and labels ── */

typedef struct {
    const uint8_t *edges;
    const uint8_t *ports;
    const uint8_t *nodes;
    Py_buffer edges_buffer;
    Py_buffer ports_buffer;
    Py_buffer nodes_buffer;
} PortMasks;

static int parse_port_masks(GraphContext *graph, PyObject *edge_mask_obj, PyObject *port_mask_obj,
                            PyObject *node_mask_obj, PortMasks *masks) {
    if (parse_mask(edge_mask_obj, graph->ports->edge_count, &masks->edges, &masks->edges_buffer) < 0) return -1;
    if (parse_mask(port_mask_obj, graph->ports->port_count, &masks->ports, &masks->ports_buffer) < 0) {
        release_mask(&masks->edges_buffer);
        return -1;
    }
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &masks->nodes, &masks->nodes_buffer) < 0) {
        release_mask(&masks->ports_buffer);
        release_mask(&masks->edges_buffer);
        return -1;
    }
    return 0;
}

static void release_port_masks(PortMasks *masks) {
    release_mask(&masks->nodes_buffer);
    release_mask(&masks->ports_buffer);
    release_mask(&masks->edges_buffer);
}

static int *labels_of(GraphContext *graph, PortMasks *masks) {
    int node_count = graph->node_ids.node_count;
    int *labels = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    if (!labels) { PyErr_NoMemory(); return NULL; }
    if (port_component_labels(node_count, graph->ports, masks->edges, masks->ports, masks->nodes, labels) < 0) {
        free(labels);
        return NULL;
    }
    return labels;
}

PyObject *py_port_component_labels(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *port_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &port_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = port_graph_of(capsule);
    if (!graph) return NULL;
    PortMasks masks;
    if (parse_port_masks(graph, edge_mask_obj, port_mask_obj, node_mask_obj, &masks) < 0) return NULL;
    int *labels = labels_of(graph, &masks);
    release_port_masks(&masks);
    if (!labels) return NULL;
    PyObject *result = PyBytes_FromStringAndSize((const char *)labels,
                                                 (Py_ssize_t)graph->node_ids.node_count * (Py_ssize_t)sizeof(int));
    free(labels);
    return result;
}

/* ── Components and membership ── */

/*
 * One node set per component, in the order of each component's smallest node;
 * with branch_ids, also the branch ids of the edges that belong to it. An edge
 * belongs to the component of its active ports (all in one component, since
 * the edge joins them) and to none without an active port.
 */
static PyObject *components_of(GraphContext *graph, PortMasks *masks, PyObject *branch_ids_obj) {
    int node_count = graph->node_ids.node_count;
    PortList *ports = graph->ports;
    PyObject *branch_ids_fast = NULL;
    if (branch_ids_obj) {
        branch_ids_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
        if (!branch_ids_fast) return NULL;
        if (PySequence_Fast_GET_SIZE(branch_ids_fast) != ports->edge_count) {
            Py_DECREF(branch_ids_fast);
            PyErr_SetString(PyExc_ValueError, "branch_ids needs one id per edge");
            return NULL;
        }
    }
    int *labels = labels_of(graph, masks);
    PyObject **node_sets = labels ? (PyObject **)calloc((size_t)(node_count > 0 ? node_count : 1), sizeof(PyObject *)) : NULL;
    PyObject **branch_sets = node_sets && branch_ids_fast
        ? (PyObject **)calloc((size_t)(node_count > 0 ? node_count : 1), sizeof(PyObject *)) : NULL;
    PyObject *result = (node_sets && (!branch_ids_fast || branch_sets)) ? PyList_New(0) : NULL;
    if (!result) {
        if (labels && (!node_sets || (branch_ids_fast && !branch_sets))) PyErr_NoMemory();
        goto cleanup;
    }
    for (int node = 0; node < node_count; node++) {
        int label = labels[node];
        if (label < 0) continue;
        if (!node_sets[label]) {
            node_sets[label] = PySet_New(NULL);
            if (!node_sets[label]) goto failure;
            if (branch_sets) {
                branch_sets[label] = PySet_New(NULL);
                if (!branch_sets[label]) goto failure;
            }
        }
        if (PySet_Add(node_sets[label], graph->node_ids.node_id_items[node]) < 0) goto failure;
    }
    if (branch_sets) {
        PyObject **branch_ids = PySequence_Fast_ITEMS(branch_ids_fast);
        for (int edge = 0; edge < ports->edge_count; edge++) {
            for (int port = ports->port_offsets[edge]; port < ports->port_offsets[edge + 1]; port++) {
                if (!port_is_active(ports, port, masks->edges, masks->ports, masks->nodes)) continue;
                if (PySet_Add(branch_sets[labels[ports->port_nodes[port]]], branch_ids[edge]) < 0) goto failure;
                break;
            }
        }
    }
    for (int node = 0; node < node_count; node++) {
        if (!node_sets[node]) continue;
        PyObject *entry = branch_sets ? PyTuple_Pack(2, node_sets[node], branch_sets[node]) : node_sets[node];
        if (!entry) goto failure;
        int appended = PyList_Append(result, entry);
        if (branch_sets) Py_DECREF(entry);
        if (appended < 0) goto failure;
    }
    goto cleanup;

failure:
    Py_CLEAR(result);
cleanup:
    if (node_sets) for (int node = 0; node < node_count; node++) Py_XDECREF(node_sets[node]);
    if (branch_sets) for (int node = 0; node < node_count; node++) Py_XDECREF(branch_sets[node]);
    free(node_sets);
    free(branch_sets);
    free(labels);
    Py_XDECREF(branch_ids_fast);
    return result;
}

PyObject *py_port_connected_components(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *port_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &port_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = port_graph_of(capsule);
    if (!graph) return NULL;
    PortMasks masks;
    if (parse_port_masks(graph, edge_mask_obj, port_mask_obj, node_mask_obj, &masks) < 0) return NULL;
    PyObject *result = components_of(graph, &masks, NULL);
    release_port_masks(&masks);
    return result;
}

PyObject *py_port_connected_components_with_branch_ids(PyObject *self, PyObject *args) {
    PyObject *capsule, *branch_ids_obj, *edge_mask_obj = Py_None, *port_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OOO", &capsule, &branch_ids_obj, &edge_mask_obj, &port_mask_obj, &node_mask_obj))
        return NULL;
    GraphContext *graph = port_graph_of(capsule);
    if (!graph) return NULL;
    PortMasks masks;
    if (parse_port_masks(graph, edge_mask_obj, port_mask_obj, node_mask_obj, &masks) < 0) return NULL;
    PyObject *result = components_of(graph, &masks, branch_ids_obj);
    release_port_masks(&masks);
    return result;
}
