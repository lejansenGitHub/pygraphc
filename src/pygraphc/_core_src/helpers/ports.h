/* Edges with ports: parsing, the adjacency list with multiport entries and port masks. */
#ifndef PYGRAPHC_CORE_PORTS_H
#define PYGRAPHC_CORE_PORTS_H

#include "../models.h"

/*
 * Parse edges given as tuples of node ids, one per port, into graph->ports and
 * graph->node_ids.edges; directed edges are (u, v) or (in_nodes, out_nodes).
 * A two-port edge keeps its endpoints in the edge list; any other edge sits
 * there as (node of its first port, same node), which joins nothing. The node
 * ids must already be parsed. Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int parse_port_edges(GraphContext *graph, PyObject *edges_obj);

/*
 * Build the adjacency list of a graph with multiport edges: a two-port edge as
 * in build_adjacency, an edge with three or more ports as -1 - e at each of its
 * ports (directed: in-ports forward, out-ports in reverse) with the port in
 * edge_indices, and a single-port edge not at all. Returns 0, or -1 with an
 * exception set.
 */
PYGRAPHC_INTERNAL int build_port_adjacency(int node_count, const EdgeList *edges, const PortList *ports, int directed,
                                           AdjacencyList *adjacency);

/* A parsed port mask and the edge mask two-port edges are traversed under. */
typedef struct {
    const uint8_t *ports;        /* one byte per port, NULL when no port mask is given */
    uint8_t *two_port_edges;     /* the given edge mask plus every two-port edge with a masked port; NULL without */
    Py_buffer ports_buffer;
} PortMask;

/*
 * Parse an optional port mask (None masks nothing), one byte per port. Every
 * graph numbers its ports: with a port list as listed there, without one edge e
 * has ports 2e and 2e + 1. A two-port edge with a masked port joins nothing
 * (masking case 5), so it is folded into a copy of the edge mask once per call
 * and the kernels check two-port edges by edge mask alone. Returns 0, or -1
 * with an exception set.
 */
PYGRAPHC_INTERNAL int parse_port_mask(const GraphContext *graph, PyObject *port_mask_obj, const uint8_t *edge_mask,
                                      PortMask *mask);
PYGRAPHC_INTERNAL void release_port_mask(PortMask *mask);

/*
 * How searches pass through edges under per-port weights. A pass goes from one
 * port to another (directed: an in-port to an out-port) and costs both port
 * weights, so every such pair must sum to at least 0. With a negative weight a
 * pass is split into an entry and an exit step that are both non-negative,
 * using the edge's shift s (its lowest port weight, directed: in-port weight,
 * when negative, else 0):
 * - directed: an in-port step costs w - s, an out-port step w + s;
 * - undirected, forward: entering through p costs w_p + s, leaving through q
 *   costs w_q - s; the negative port n (at most one per edge) is never
 *   entered but passes directly to each other port at w_n + w_q, so no path
 *   leaves an edge through the port it entered by. A backward search runs on
 *   the reverse: entering costs w - s, leaving w + s but never through n, and
 *   each other port passes directly to n at w_q + w_n.
 * Both arrays are NULL when no port weight is negative, and searches then
 * pass through edges as they did before.
 */
typedef struct {
    double *shifts;        /* edge_count, or NULL */
    int *negative_ports;   /* edge_count: the negative port of an undirected edge, or -1; or NULL */
} PassWeights;

/*
 * Validate per-port weights and fill pass_weights; callers skip it when no
 * weight has its sign bit set (WeightList.has_sign_bit). Returns 0, or -1
 * with ValueError or MemoryError set.
 */
PYGRAPHC_INTERNAL int parse_pass_weights(const PortList *ports, Py_ssize_t edge_count, const double *weights,
                                         int directed, PassWeights *pass_weights);
PYGRAPHC_INTERNAL void release_pass_weights(PassWeights *pass_weights);

/* The edge mask under which two-port edges are traversed: the given one, unless a port mask is set. */
static inline const uint8_t *two_port_edge_mask(const PortMask *mask, const uint8_t *edge_mask) {
    return mask->two_port_edges ? mask->two_port_edges : edge_mask;
}

/*
 * The node of the first active port of an edge, or -1 when it has none: the
 * membership rule (an edge belongs to the component of its active ports, which
 * are always joined). Without a port list, edge e has ports 2e and 2e + 1 at
 * its two endpoints.
 */
static inline int first_active_port_node(const GraphContext *graph, int edge, const uint8_t *edge_mask,
                                         const uint8_t *port_mask, const uint8_t *node_mask) {
    const PortList *ports = graph->ports;
    if (ports) {
        for (int port = ports->port_offsets[edge]; port < ports->port_offsets[edge + 1]; port++)
            if (port_is_active(ports, edge, port, edge_mask, port_mask, node_mask)) return ports->port_nodes[port];
        return -1;
    }
    if (edge_mask && edge_mask[edge]) return -1;
    const EdgeList *edges = &graph->node_ids.edges;
    int first_node = EDGE_FIRST_ENDPOINT(edges, edge);
    if (!(port_mask && port_mask[2 * edge]) && !(node_mask && node_mask[first_node])) return first_node;
    int second_node = EDGE_SECOND_ENDPOINT(edges, edge);
    if (!(port_mask && port_mask[2 * edge + 1]) && !(node_mask && node_mask[second_node])) return second_node;
    return -1;
}

#endif
