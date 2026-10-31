/* Edges with terminals: parsing and terminal masks. */
#include "terminals.h"
#include "parse.h"

/* ── Parsing ── */

/*
 * The terminal nodes parsed so far. While every edge has two terminals
 * (directed: one in, one out), edge e has terminals 2e and 2e + 1, the
 * offsets stay NULL and nodes is the edge list itself, interleaved; at the
 * first edge that differs the offsets are written out and kept from then on.
 * The state travels by value, so the parse loop keeps it in registers.
 */
typedef struct {
    int *nodes;
    int capacity;
    int count;
} TerminalNodes;

/* Room for `more` terminals; nodes is NULL (the old array freed) when memory runs out. */
static __attribute__((noinline)) TerminalNodes grow_terminal_nodes(TerminalNodes terminal_nodes, Py_ssize_t more) {
    Py_ssize_t capacity = terminal_nodes.capacity;
    while (capacity < terminal_nodes.count + more) capacity *= 2;
    int *nodes = (int *)realloc(terminal_nodes.nodes, (size_t)capacity * sizeof(int));
    if (!nodes) { free(terminal_nodes.nodes); PyErr_NoMemory(); }
    terminal_nodes.nodes = nodes;
    terminal_nodes.capacity = (int)capacity;
    return terminal_nodes;
}

/* Write the nodes of `length` node ids from nodes[at] on; the caller has made room. Returns 0, or -1 with an exception set. */
static inline __attribute__((always_inline)) int write_nodes(int *nodes, int at, const IntMap *node_index_by_node_id,
                                                             PyObject **items, Py_ssize_t length) {
    for (Py_ssize_t position = 0; position < length; position++) {
        long node_id = PyLong_AsLong(items[position]);
        if (node_id == -1 && PyErr_Occurred()) return -1;
        int node = intmap_get((IntMap *)node_index_by_node_id, node_id);
        if (node < 0) { PyErr_SetString(PyExc_ValueError, "edge references unknown node ID"); return -1; }
        nodes[at + position] = node;
    }
    return 0;
}

/*
 * Append a sequence of node ids, at least one. Returns the new state; nodes
 * is NULL (everything freed) or count is -1 (nodes kept) on error, with an
 * exception set.
 */
static inline __attribute__((always_inline)) TerminalNodes append_sequence(TerminalNodes terminal_nodes,
                                                                           const IntMap *node_index_by_node_id,
                                                                           PyObject *node_ids_obj) {
    PyObject *node_ids_fast = PySequence_Fast(node_ids_obj, "the terminals of an edge must be a sequence of node ids");
    if (!node_ids_fast) { terminal_nodes.count = -1; return terminal_nodes; }
    Py_ssize_t length = PySequence_Fast_GET_SIZE(node_ids_fast);
    if (length == 0) {
        PyErr_SetString(PyExc_ValueError, "an edge needs at least one terminal");
        terminal_nodes.count = -1;
    } else {
        if (terminal_nodes.count + length > terminal_nodes.capacity) terminal_nodes = grow_terminal_nodes(terminal_nodes, length);
        if (!terminal_nodes.nodes) { Py_DECREF(node_ids_fast); return terminal_nodes; }
        if (write_nodes(terminal_nodes.nodes, terminal_nodes.count, node_index_by_node_id,
                        PySequence_Fast_ITEMS(node_ids_fast), length) < 0)
            terminal_nodes.count = -1;
        else
            terminal_nodes.count += (int)length;
    }
    Py_DECREF(node_ids_fast);
    return terminal_nodes;
}

/* The terminals of a directed edge given as (in_nodes, out_nodes); *out_start is its first out-terminal. */
static __attribute__((noinline)) TerminalNodes append_in_and_out_terminals(TerminalNodes terminal_nodes,
                                                                            const IntMap *node_index_by_node_id,
                                                                            PyObject **sides, int *out_start) {
    terminal_nodes = append_sequence(terminal_nodes, node_index_by_node_id, sides[0]);
    if (!terminal_nodes.nodes || terminal_nodes.count < 0) return terminal_nodes;
    *out_start = terminal_nodes.count;
    return append_sequence(terminal_nodes, node_index_by_node_id, sides[1]);
}

/* A directed edge is (u, v), from u to v, or (in_nodes, out_nodes); *out_start is its first out-terminal. */
static inline __attribute__((always_inline)) TerminalNodes append_directed_edge(TerminalNodes terminal_nodes,
                                                                                const IntMap *node_index_by_node_id,
                                                                                PyObject *edge_obj, int *out_start) {
    PyObject *edge_fast = PySequence_Fast(edge_obj, "a directed edge is (u, v) or (in_nodes, out_nodes)");
    if (!edge_fast) { terminal_nodes.count = -1; return terminal_nodes; }
    PyObject **sides = PySequence_Fast_ITEMS(edge_fast);
    int pair = PySequence_Fast_GET_SIZE(edge_fast) == 2;
    if (pair && PyLong_Check(sides[0]) && PyLong_Check(sides[1])) {
        if (terminal_nodes.count + 2 > terminal_nodes.capacity) terminal_nodes = grow_terminal_nodes(terminal_nodes, 2);
        if (terminal_nodes.nodes) {
            *out_start = terminal_nodes.count + 1;
            if (write_nodes(terminal_nodes.nodes, terminal_nodes.count, node_index_by_node_id, sides, 2) < 0)
                terminal_nodes.count = -1;
            else
                terminal_nodes.count += 2;
        }
    } else if (pair && !PyLong_Check(sides[0]) && !PyLong_Check(sides[1])) {
        terminal_nodes = append_in_and_out_terminals(terminal_nodes, node_index_by_node_id, sides, out_start);
    } else {
        PyErr_SetString(PyExc_ValueError, "a directed edge is (u, v) or (in_nodes, out_nodes)");
        terminal_nodes.count = -1;
    }
    Py_DECREF(edge_fast);
    return terminal_nodes;
}

/* Write out the implicit offsets of the edges before `edge`: edge e at 2e, its out-terminal at 2e + 1. */
static __attribute__((noinline)) int write_out_offsets(Py_ssize_t edge_count, int edge, int directed, int **offsets_out,
                                                       int **out_offsets_out) {
    int *offsets = (int *)malloc(((size_t)edge_count + 1) * sizeof(int));
    int *out_offsets = directed ? (int *)malloc(((size_t)edge_count + 1) * sizeof(int)) : NULL;
    if (!offsets || (directed && !out_offsets)) { free(offsets); free(out_offsets); PyErr_NoMemory(); return -1; }
    for (int earlier = 0; earlier < edge; earlier++) {
        offsets[earlier] = 2 * earlier;
        if (directed) out_offsets[earlier] = 2 * earlier + 1;
    }
    *offsets_out = offsets;
    *out_offsets_out = out_offsets;
    return 0;
}

/* The edge list of a graph with a terminal list: a two-terminal edge's endpoints, any other edge at the node of its first terminal twice. */
static int write_edge_list(EdgeList *edges, const int *nodes, const int *offsets, Py_ssize_t edge_count) {
    int *endpoints = (int *)malloc((size_t)(edge_count > 0 ? 2 * edge_count : 1) * sizeof(int));
    if (!endpoints) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first_terminal = offsets[edge];
        int two_terminals = offsets[edge + 1] - first_terminal == 2;
        endpoints[2 * edge] = nodes[first_terminal];
        endpoints[2 * edge + 1] = nodes[two_terminals ? first_terminal + 1 : first_terminal];
    }
    edges->first_endpoints = endpoints;
    edges->second_endpoints = NULL;
    return 0;
}

/* The parsed terminals of all edges: the nodes, and the offsets once some edge is not a pair. */
typedef struct {
    TerminalNodes terminal_nodes;
    int *offsets;
    int *out_offsets;
} ParsedTerminals;

/*
 * The parse loop, instantiated per value of `directed`. On error the result's
 * nodes is NULL, everything freed, with an exception set.
 */
static inline __attribute__((always_inline)) ParsedTerminals parse_edge_items(
    const IntMap *node_index_by_node_id, PyObject **edge_items, Py_ssize_t edge_count, const int directed) {
    ParsedTerminals parsed = {{NULL, (int)(edge_count > 0 ? 2 * edge_count : 2), 0}, NULL, NULL};
    parsed.terminal_nodes.nodes = (int *)malloc((size_t)parsed.terminal_nodes.capacity * sizeof(int));
    if (!parsed.terminal_nodes.nodes) { PyErr_NoMemory(); return parsed; }
    TerminalNodes terminal_nodes = parsed.terminal_nodes;
    int *offsets = NULL, *out_offsets = NULL;
    for (int edge = 0; edge < edge_count; edge++) {
        int start = terminal_nodes.count, out_start = start + 1;
        terminal_nodes = directed ? append_directed_edge(terminal_nodes, node_index_by_node_id, edge_items[edge], &out_start)
                                  : append_sequence(terminal_nodes, node_index_by_node_id, edge_items[edge]);
        if (!terminal_nodes.nodes || terminal_nodes.count < 0) break;
        if (!offsets && (terminal_nodes.count - start != 2 || (directed && out_start != start + 1))
            && write_out_offsets(edge_count, edge, directed, &offsets, &out_offsets) < 0) {
            terminal_nodes.count = -1;
            break;
        }
        if (offsets) {
            offsets[edge] = start;
            if (directed) out_offsets[edge] = out_start;
        }
    }
    if (!terminal_nodes.nodes || terminal_nodes.count < 0) {
        free(terminal_nodes.nodes); free(offsets); free(out_offsets);
        parsed.terminal_nodes.nodes = NULL;
        return parsed;
    }
    parsed.terminal_nodes = terminal_nodes;
    parsed.offsets = offsets;
    parsed.out_offsets = out_offsets;
    return parsed;
}

int parse_terminal_edges(GraphContext *graph, PyObject *edges_obj) {
    EdgeList *edges = &graph->node_ids.edges;
    const IntMap *node_index_by_node_id = &graph->node_ids.node_index_by_node_id;
    if (graph->node_ids.node_count == 0) return 0;
    int buffer_status = parse_edge_pair_buffer((IntMap *)node_index_by_node_id, edges_obj, edges);
    if (buffer_status != 0) return buffer_status < 0 ? -1 : 0;

    PyObject *edges_fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!edges_fast) return -1;
    Py_ssize_t edge_count = PySequence_Fast_GET_SIZE(edges_fast);
    PyObject **edge_items = PySequence_Fast_ITEMS(edges_fast);
    ParsedTerminals parsed = graph->directed ? parse_edge_items(node_index_by_node_id, edge_items, edge_count, 1)
                                             : parse_edge_items(node_index_by_node_id, edge_items, edge_count, 0);
    Py_DECREF(edges_fast);
    if (!parsed.terminal_nodes.nodes) return -1;

    int *nodes = parsed.terminal_nodes.nodes, count = parsed.terminal_nodes.count;
    edges->edge_count = edge_count;
    edges->owns_memory = 1;
    if (!parsed.offsets) {
        edges->first_endpoints = nodes;
        edges->second_endpoints = NULL;
        return 0;
    }
    parsed.offsets[edge_count] = count;
    TerminalList *terminals = (TerminalList *)malloc(sizeof(TerminalList));
    if (!terminals || write_edge_list(edges, nodes, parsed.offsets, edge_count) < 0) {
        free(terminals); free(nodes); free(parsed.offsets); free(parsed.out_offsets);
        edges->edge_count = 0;
        if (!PyErr_Occurred()) PyErr_NoMemory();
        return -1;
    }
    terminals->terminal_count = count;
    terminals->terminal_offsets = parsed.offsets;
    terminals->terminal_nodes = nodes;
    terminals->out_terminal_offsets = parsed.out_offsets;
    graph->terminals = terminals;
    return 0;
}

/* ── Terminal masks ── */

int parse_terminal_mask(const GraphContext *graph, PyObject *terminal_mask_obj, const uint8_t *edge_mask, TerminalMask *mask) {
    mask->terminals = NULL;
    mask->two_terminal_edges = NULL;
    mask->terminals_buffer.obj = NULL;
    if (terminal_mask_obj == NULL || terminal_mask_obj == Py_None) return 0;
    const TerminalList *terminals = graph->terminals;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    Py_ssize_t terminal_count = terminals ? terminals->terminal_count : 2 * edge_count;
    if (parse_mask(terminal_mask_obj, terminal_count, &mask->terminals, &mask->terminals_buffer) < 0) return -1;
    mask->two_terminal_edges = (uint8_t *)malloc((size_t)(edge_count > 0 ? edge_count : 1));
    if (!mask->two_terminal_edges) { release_mask(&mask->terminals_buffer); PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        Py_ssize_t first_terminal = terminals ? terminals->terminal_offsets[edge] : 2 * edge;
        int two_terminals = terminals ? terminals->terminal_offsets[edge + 1] - first_terminal == 2 : 1;
        mask->two_terminal_edges[edge] = (uint8_t)((edge_mask && edge_mask[edge]) ||
                                               (two_terminals && (mask->terminals[first_terminal] || mask->terminals[first_terminal + 1])));
    }
    return 0;
}

void release_terminal_mask(TerminalMask *mask) {
    free(mask->two_terminal_edges);
    mask->two_terminal_edges = NULL;
    release_mask(&mask->terminals_buffer);
}

int parse_pass_weights(const TerminalList *terminals, Py_ssize_t edge_count, const double *weights, int directed,
                       PassWeights *pass_weights) {
    pass_weights->shifts = NULL;
    pass_weights->negative_terminals = NULL;
    for (Py_ssize_t edge = 0; edge < edge_count; edge++) {
        int first = terminals->terminal_offsets[edge], end = terminals->terminal_offsets[edge + 1];
        int first_out = directed ? terminals->out_terminal_offsets[edge] : end;
        double lowest = HUGE_VAL, second_lowest = HUGE_VAL, lowest_out = HUGE_VAL;
        int lowest_terminal = -1;
        for (int terminal = first; terminal < end; terminal++) {
            double weight = weights[terminal];
            if (terminal >= first_out) { if (weight < lowest_out) lowest_out = weight; }
            else if (weight < lowest) { second_lowest = lowest; lowest = weight; lowest_terminal = terminal; }
            else if (weight < second_lowest) second_lowest = weight;
        }
        double cheapest_pass = directed ? lowest + lowest_out : lowest + second_lowest;
        if (cheapest_pass < 0.0) {
            release_pass_weights(pass_weights);
            char message[200];
            snprintf(message, sizeof message, "the terminal weights of edge %zd allow a pass costing %g; %s must sum to at least 0",
                     edge, cheapest_pass, directed ? "every in-terminal and out-terminal of an edge" : "every two terminals of an edge");
            PyErr_SetString(PyExc_ValueError, message);
            return -1;
        }
        if (lowest >= 0.0) continue;
        if (!pass_weights->shifts) {
            pass_weights->shifts = (double *)calloc((size_t)edge_count, sizeof(double));
            pass_weights->negative_terminals = (int *)malloc((size_t)edge_count * sizeof(int));
            if (!pass_weights->shifts || !pass_weights->negative_terminals) {
                release_pass_weights(pass_weights);
                PyErr_NoMemory();
                return -1;
            }
            memset(pass_weights->negative_terminals, -1, (size_t)edge_count * sizeof(int));
        }
        pass_weights->shifts[edge] = lowest;
        pass_weights->negative_terminals[edge] = lowest_terminal;
    }
    return 0;
}

void release_pass_weights(PassWeights *pass_weights) {
    free(pass_weights->shifts);
    free(pass_weights->negative_terminals);
    pass_weights->shifts = NULL;
    pass_weights->negative_terminals = NULL;
}
