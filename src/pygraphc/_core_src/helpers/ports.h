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
 * Parse an optional port mask (None masks nothing). A two-port edge with a
 * masked port joins nothing (masking case 5), so it is folded into a copy of
 * the edge mask once per call and the kernels check two-port edges by edge
 * mask alone. Only a graph with multiport edges numbers its ports. Returns 0,
 * or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int parse_port_mask(const GraphContext *graph, PyObject *port_mask_obj, const uint8_t *edge_mask,
                                      PortMask *mask);
PYGRAPHC_INTERNAL void release_port_mask(PortMask *mask);

/* The edge mask under which two-port edges are traversed: the given one, unless a port mask is set. */
static inline const uint8_t *two_port_edge_mask(const PortMask *mask, const uint8_t *edge_mask) {
    return mask->two_port_edges ? mask->two_port_edges : edge_mask;
}

#endif
