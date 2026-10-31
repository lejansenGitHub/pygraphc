/* Edges with terminals: parsing, the adjacency list with multi-terminal entries and terminal masks. */
#ifndef PYGRAPHC_CORE_TERMINALS_H
#define PYGRAPHC_CORE_TERMINALS_H

#include "../models.h"

/*
 * Parse edges given as tuples of node ids, one per terminal, into graph->terminals and
 * graph->node_ids.edges; directed edges are (u, v) or (in_nodes, out_nodes).
 * A two-terminal edge keeps its endpoints in the edge list; any other edge sits
 * there as (node of its first terminal, same node), which joins nothing. The node
 * ids must already be parsed. Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int parse_terminal_edges(GraphContext *graph, PyObject *edges_obj);

/*
 * Build the adjacency list of a graph with multi-terminal edges: a two-terminal edge as
 * in build_adjacency, an edge with three or more terminals as -1 - e at each of its
 * terminals (directed: in-terminals forward, out-terminals in reverse) with the terminal in
 * edge_indices, and a single-terminal edge not at all. Returns 0, or -1 with an
 * exception set.
 */
PYGRAPHC_INTERNAL int build_terminal_adjacency(int node_count, const EdgeList *edges, const TerminalList *terminals, int directed,
                                           AdjacencyList *adjacency);

/* A parsed terminal mask and the edge mask two-terminal edges are traversed under. */
typedef struct {
    const uint8_t *terminals;        /* one byte per terminal, NULL when no terminal mask is given */
    uint8_t *two_terminal_edges;     /* the given edge mask plus every two-terminal edge with a masked terminal; NULL without */
    Py_buffer terminals_buffer;
} TerminalMask;

/*
 * Parse an optional terminal mask (None masks nothing), one byte per terminal. Every
 * graph numbers its terminals: with a terminal list as listed there, without one edge e
 * has terminals 2e and 2e + 1. A two-terminal edge with a masked terminal joins nothing
 * (masking case 5), so it is folded into a copy of the edge mask once per call
 * and the kernels check two-terminal edges by edge mask alone. Returns 0, or -1
 * with an exception set.
 */
PYGRAPHC_INTERNAL int parse_terminal_mask(const GraphContext *graph, PyObject *terminal_mask_obj, const uint8_t *edge_mask,
                                      TerminalMask *mask);
PYGRAPHC_INTERNAL void release_terminal_mask(TerminalMask *mask);

/*
 * How searches pass through edges under per-terminal weights. A pass goes from one
 * terminal to another (directed: an in-terminal to an out-terminal) and costs both terminal
 * weights, so every such pair must sum to at least 0. With a negative weight a
 * pass is split into an entry and an exit step that are both non-negative,
 * using the edge's shift s (its lowest terminal weight, directed: in-terminal weight,
 * when negative, else 0):
 * - directed: an in-terminal step costs w - s, an out-terminal step w + s;
 * - undirected, forward: entering through p costs w_p + s, leaving through q
 *   costs w_q - s; the negative terminal n (at most one per edge) is never
 *   entered but passes directly to each other terminal at w_n + w_q, so no path
 *   leaves an edge through the terminal it entered by. A backward search runs on
 *   the reverse: entering costs w - s, leaving w + s but never through n, and
 *   each other terminal passes directly to n at w_q + w_n.
 * Both arrays are NULL when no terminal weight is negative, and searches then
 * pass through edges as they did before.
 */
typedef struct {
    double *shifts;        /* edge_count, or NULL */
    int *negative_terminals;   /* edge_count: the negative terminal of an undirected edge, or -1; or NULL */
} PassWeights;

/*
 * Validate per-terminal weights and fill pass_weights; callers skip it when no
 * weight has its sign bit set (WeightList.has_sign_bit). Returns 0, or -1
 * with ValueError or MemoryError set.
 */
PYGRAPHC_INTERNAL int parse_pass_weights(const TerminalList *terminals, Py_ssize_t edge_count, const double *weights,
                                         int directed, PassWeights *pass_weights);
PYGRAPHC_INTERNAL void release_pass_weights(PassWeights *pass_weights);

/* The edge mask under which two-terminal edges are traversed: the given one, unless a terminal mask is set. */
static inline const uint8_t *two_terminal_edge_mask(const TerminalMask *mask, const uint8_t *edge_mask) {
    return mask->two_terminal_edges ? mask->two_terminal_edges : edge_mask;
}

/*
 * The node of the first active terminal of an edge, or -1 when it has none: the
 * membership rule (an edge belongs to the component of its active terminals, which
 * are always joined). Without a terminal list, edge e has terminals 2e and 2e + 1 at
 * its two endpoints.
 */
static inline int first_active_terminal_node(const GraphContext *graph, int edge, const uint8_t *edge_mask,
                                         const uint8_t *terminal_mask, const uint8_t *node_mask) {
    const TerminalList *terminals = graph->terminals;
    if (terminals) {
        for (int terminal = terminals->terminal_offsets[edge]; terminal < terminals->terminal_offsets[edge + 1]; terminal++)
            if (terminal_is_active(terminals, edge, terminal, edge_mask, terminal_mask, node_mask)) return terminals->terminal_nodes[terminal];
        return -1;
    }
    if (edge_mask && edge_mask[edge]) return -1;
    const EdgeList *edges = &graph->node_ids.edges;
    int first_node = EDGE_FIRST_ENDPOINT(edges, edge);
    if (!(terminal_mask && terminal_mask[2 * edge]) && !(node_mask && node_mask[first_node])) return first_node;
    int second_node = EDGE_SECOND_ENDPOINT(edges, edge);
    if (!(terminal_mask && terminal_mask[2 * edge + 1]) && !(node_mask && node_mask[second_node])) return second_node;
    return -1;
}

#endif
