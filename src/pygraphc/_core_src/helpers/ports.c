/* Edges with ports: parsing, the adjacency list with multiport entries and port masks. */
#include "ports.h"
#include "parse.h"

/* ── Parsing ── */

typedef struct {
    PortList *ports;
    int capacity;
} PortBuilder;

static int append_port(PortBuilder *builder, int node) {
    PortList *ports = builder->ports;
    if (ports->port_count == builder->capacity) {
        int capacity = builder->capacity > 0 ? builder->capacity * 2 : 64;
        int *port_nodes = (int *)realloc(ports->port_nodes, (size_t)capacity * sizeof(int));
        if (!port_nodes) { PyErr_NoMemory(); return -1; }
        ports->port_nodes = port_nodes;
        builder->capacity = capacity;
    }
    ports->port_nodes[ports->port_count++] = node;
    return 0;
}

static int node_of(const GraphContext *graph, PyObject *item, int *node_out) {
    long node_id = PyLong_AsLong(item);
    if (node_id == -1 && PyErr_Occurred()) return -1;
    int node = graph->node_ids.node_count > 0
        ? intmap_get((IntMap *)&graph->node_ids.node_index_by_node_id, node_id) : -1;
    if (node < 0) { PyErr_SetString(PyExc_ValueError, "edge references unknown node ID"); return -1; }
    *node_out = node;
    return 0;
}

static int append_ports_of(PortBuilder *builder, const GraphContext *graph, PyObject *node_ids_obj) {
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
        if (node_of(graph, items[position], &node) < 0 || append_port(builder, node) < 0) {
            Py_DECREF(node_ids_fast);
            return -1;
        }
    }
    Py_DECREF(node_ids_fast);
    return 0;
}

/* A directed edge is (u, v), from u to v, or (in_nodes, out_nodes) with at least one of each. */
static int append_directed_edge(PortBuilder *builder, const GraphContext *graph, PyObject *edge_obj, int edge) {
    PyObject *edge_fast = PySequence_Fast(edge_obj, "a directed edge is (u, v) or (in_nodes, out_nodes)");
    if (!edge_fast) return -1;
    if (PySequence_Fast_GET_SIZE(edge_fast) != 2) {
        Py_DECREF(edge_fast);
        PyErr_SetString(PyExc_ValueError, "a directed edge is (u, v) or (in_nodes, out_nodes)");
        return -1;
    }
    PyObject **sides = PySequence_Fast_ITEMS(edge_fast);
    int status = -1;
    if (PyLong_Check(sides[0]) && PyLong_Check(sides[1])) {
        int first_node, second_node;
        if (node_of(graph, sides[0], &first_node) == 0 && node_of(graph, sides[1], &second_node) == 0 &&
            append_port(builder, first_node) == 0) {
            builder->ports->out_port_offsets[edge] = builder->ports->port_count;
            status = append_port(builder, second_node);
        }
    } else if (!PyLong_Check(sides[0]) && !PyLong_Check(sides[1])) {
        if (append_ports_of(builder, graph, sides[0]) == 0) {
            builder->ports->out_port_offsets[edge] = builder->ports->port_count;
            status = append_ports_of(builder, graph, sides[1]);
        }
    } else {
        PyErr_SetString(PyExc_ValueError, "a directed edge is (u, v) or (in_nodes, out_nodes)");
    }
    Py_DECREF(edge_fast);
    return status;
}

/* The edge list: endpoints of a two-port edge, (node of the first port, same node) for any other edge. */
static int fill_edge_list(EdgeList *edges, const PortList *ports) {
    Py_ssize_t edge_count = edges->edge_count;
    edges->first_endpoints = (int *)malloc((size_t)(edge_count > 0 ? edge_count : 1) * sizeof(int));
    edges->second_endpoints = (int *)malloc((size_t)(edge_count > 0 ? edge_count : 1) * sizeof(int));
    edges->owns_memory = 1;
    if (!edges->first_endpoints || !edges->second_endpoints) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first_port = ports->port_offsets[edge];
        int two_ports = ports->port_offsets[edge + 1] - first_port == 2;
        edges->first_endpoints[edge] = ports->port_nodes[first_port];
        edges->second_endpoints[edge] = ports->port_nodes[two_ports ? first_port + 1 : first_port];
    }
    return 0;
}

int parse_port_edges(GraphContext *graph, PyObject *edges_obj) {
    PyObject *edges_fast = PySequence_Fast(edges_obj, "edges must be a sequence");
    if (!edges_fast) return -1;
    int edge_count = (int)PySequence_Fast_GET_SIZE(edges_fast);
    PortList *ports = (PortList *)calloc(1, sizeof(PortList));
    graph->ports = ports;
    graph->node_ids.edges.edge_count = edge_count;
    if (ports) {
        ports->port_offsets = (int *)malloc(((size_t)edge_count + 1) * sizeof(int));
        if (graph->directed) ports->out_port_offsets = (int *)malloc(((size_t)edge_count + 1) * sizeof(int));
    }
    if (!ports || !ports->port_offsets || (graph->directed && !ports->out_port_offsets)) {
        Py_DECREF(edges_fast);
        PyErr_NoMemory();
        return -1;
    }
    PortBuilder builder = {ports, 0};
    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (int edge = 0; edge < edge_count; edge++) {
        ports->port_offsets[edge] = ports->port_count;
        int status = graph->directed ? append_directed_edge(&builder, graph, edge_items[edge], edge)
                                     : append_ports_of(&builder, graph, edge_items[edge]);
        if (status < 0) { Py_DECREF(edges_fast); return -1; }
    }
    ports->port_offsets[edge_count] = ports->port_count;
    Py_DECREF(edges_fast);
    return fill_edge_list(&graph->node_ids.edges, ports);
}

/* ── Adjacency list ── */

/*
 * Count (slots == NULL) or fill one direction of the adjacency list. Forward,
 * a two-port edge sits at its first endpoint (and, undirected, its second)
 * and an edge with three or more ports at its in-ports (undirected: all its
 * ports); in reverse, at its second endpoint and its out-ports.
 */
static void place_slots(Py_ssize_t edge_count, const EdgeList *edges, const PortList *ports, int directed, int reverse,
                        int *next_slot, int *neighbors, int *edge_indices) {
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first_port = ports->port_offsets[edge], end_port = ports->port_offsets[edge + 1];
        int port_count = end_port - first_port;
        if (port_count == 2) {
            int first_node = EDGE_FIRST_ENDPOINT(edges, edge), second_node = EDGE_SECOND_ENDPOINT(edges, edge);
            int at = reverse ? second_node : first_node, neighbor = reverse ? first_node : second_node;
            if (neighbors) { neighbors[next_slot[at]] = neighbor; edge_indices[next_slot[at]] = (int)edge; }
            next_slot[at]++;
            if (!directed) {
                if (neighbors) { neighbors[next_slot[second_node]] = first_node; edge_indices[next_slot[second_node]] = (int)edge; }
                next_slot[second_node]++;
            }
        } else if (port_count > 2) {
            int from_port = directed && reverse ? ports->out_port_offsets[edge] : first_port;
            int to_port = directed && !reverse ? ports->out_port_offsets[edge] : end_port;
            for (int port = from_port; port < to_port; port++) {
                int at = ports->port_nodes[port];
                if (neighbors) { neighbors[next_slot[at]] = -1 - (int)edge; edge_indices[next_slot[at]] = port; }
                next_slot[at]++;
            }
        }
    }
}

/* One direction as CSR arrays; returns 0, or -1 with an exception set and nothing allocated. */
static int build_direction(int node_count, const EdgeList *edges, const PortList *ports, int directed, int reverse,
                           int **offset_out, int **neighbors_out, int **edge_indices_out) {
    int *offset = (int *)calloc((size_t)node_count + 1, sizeof(int));
    if (!offset) { PyErr_NoMemory(); return -1; }
    place_slots(edges->edge_count, edges, ports, directed, reverse, offset + 1, NULL, NULL);
    for (int node = 1; node <= node_count; node++) offset[node] += offset[node - 1];
    size_t slot_count = (size_t)offset[node_count];
    int *neighbors = (int *)malloc((slot_count > 0 ? slot_count : 1) * sizeof(int));
    int *edge_indices = (int *)malloc((slot_count > 0 ? slot_count : 1) * sizeof(int));
    int *next_slot = (int *)malloc(((size_t)node_count + 1) * sizeof(int));
    if (!neighbors || !edge_indices || !next_slot) {
        free(offset); free(neighbors); free(edge_indices); free(next_slot);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(next_slot, offset, (size_t)node_count * sizeof(int));
    place_slots(edges->edge_count, edges, ports, directed, reverse, next_slot, neighbors, edge_indices);
    free(next_slot);
    *offset_out = offset;
    *neighbors_out = neighbors;
    *edge_indices_out = edge_indices;
    return 0;
}

int build_port_adjacency(int node_count, const EdgeList *edges, const PortList *ports, int directed,
                         AdjacencyList *adjacency) {
    adjacency->reverse_offset = NULL;
    adjacency->reverse_neighbors = NULL;
    adjacency->reverse_edge_indices = NULL;
    if (build_direction(node_count, edges, ports, directed, 0,
                        &adjacency->offset, &adjacency->neighbors, &adjacency->edge_indices) < 0)
        return -1;
    if (directed && build_direction(node_count, edges, ports, directed, 1, &adjacency->reverse_offset,
                                    &adjacency->reverse_neighbors, &adjacency->reverse_edge_indices) < 0) {
        free(adjacency->offset); free(adjacency->neighbors); free(adjacency->edge_indices);
        return -1;
    }
    return 0;
}

/* ── Port masks ── */

int parse_port_mask(const GraphContext *graph, PyObject *port_mask_obj, const uint8_t *edge_mask, PortMask *mask) {
    mask->ports = NULL;
    mask->two_port_edges = NULL;
    mask->ports_buffer.obj = NULL;
    if (port_mask_obj == NULL || port_mask_obj == Py_None) return 0;
    const PortList *ports = graph->ports;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    Py_ssize_t port_count = ports ? ports->port_count : 2 * edge_count;
    if (parse_mask(port_mask_obj, port_count, &mask->ports, &mask->ports_buffer) < 0) return -1;
    mask->two_port_edges = (uint8_t *)malloc((size_t)(edge_count > 0 ? edge_count : 1));
    if (!mask->two_port_edges) { release_mask(&mask->ports_buffer); PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        Py_ssize_t first_port = ports ? ports->port_offsets[edge] : 2 * edge;
        int two_ports = ports ? ports->port_offsets[edge + 1] - first_port == 2 : 1;
        mask->two_port_edges[edge] = (uint8_t)((edge_mask && edge_mask[edge]) ||
                                               (two_ports && (mask->ports[first_port] || mask->ports[first_port + 1])));
    }
    return 0;
}

void release_port_mask(PortMask *mask) {
    free(mask->two_port_edges);
    mask->two_port_edges = NULL;
    release_mask(&mask->ports_buffer);
}

int parse_pass_weights(const PortList *ports, Py_ssize_t edge_count, const double *weights, int directed,
                       PassWeights *pass_weights) {
    pass_weights->shifts = NULL;
    pass_weights->negative_ports = NULL;
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first = ports->port_offsets[edge], end = ports->port_offsets[edge + 1];
        int first_out = directed ? ports->out_port_offsets[edge] : end;
        double lowest = HUGE_VAL, second_lowest = HUGE_VAL, lowest_out = HUGE_VAL;
        int lowest_port = -1;
        for (int port = first; port < end; port++) {
            double weight = weights[port];
            if (port >= first_out) { if (weight < lowest_out) lowest_out = weight; }
            else if (weight < lowest) { second_lowest = lowest; lowest = weight; lowest_port = port; }
            else if (weight < second_lowest) second_lowest = weight;
        }
        double cheapest_pass = directed ? lowest + lowest_out : lowest + second_lowest;
        if (cheapest_pass < 0.0) {
            release_pass_weights(pass_weights);
            char message[200];
            snprintf(message, sizeof message, "the port weights of edge %zd allow a pass costing %g; %s must sum to at least 0",
                     edge, cheapest_pass, directed ? "every in-port and out-port of an edge" : "every two ports of an edge");
            PyErr_SetString(PyExc_ValueError, message);
            return -1;
        }
        if (lowest >= 0.0) continue;
        if (!pass_weights->shifts) {
            pass_weights->shifts = (double *)calloc((size_t)edge_count, sizeof(double));
            pass_weights->negative_ports = (int *)malloc((size_t)edge_count * sizeof(int));
            if (!pass_weights->shifts || !pass_weights->negative_ports) {
                release_pass_weights(pass_weights);
                PyErr_NoMemory();
                return -1;
            }
            memset(pass_weights->negative_ports, -1, (size_t)edge_count * sizeof(int));
        }
        pass_weights->shifts[edge] = lowest;
        pass_weights->negative_ports[edge] = lowest_port;
    }
    return 0;
}

void release_pass_weights(PassWeights *pass_weights) {
    free(pass_weights->shifts);
    free(pass_weights->negative_ports);
    pass_weights->shifts = NULL;
    pass_weights->negative_ports = NULL;
}
