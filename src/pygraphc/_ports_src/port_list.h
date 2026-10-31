/* Edges with ports: the activity rule, the node incidence of the ports, and components over active ports. */
#ifndef PYGRAPHC_PORTS_PORT_LIST_H
#define PYGRAPHC_PORTS_PORT_LIST_H

#include "../_core_src/models.h"

/*
 * The one rule every algorithm on edges with ports asks: a port is active iff
 * its edge, the port itself and its node are all unmasked. A NULL mask masks
 * nothing.
 */
static inline int stage_one_port_is_active(const PortList *ports, int port, const uint8_t *edge_mask,
                                 const uint8_t *port_mask, const uint8_t *node_mask) {
    if (edge_mask && edge_mask[ports->port_edges[port]]) return 0;
    if (port_mask && port_mask[port]) return 0;
    if (node_mask && node_mask[ports->port_nodes[port]]) return 0;
    return 1;
}

/* Fill incidence_offsets and incidence_ports from port_nodes. Returns 0, or -1 with an exception set. */
PYGRAPHC_INTERNAL int build_port_incidence(int node_count, PortList *ports);

/*
 * Label every node with the smallest node index of its component, where an
 * edge joins all of its active ports; masked nodes are labelled -1. Direction
 * is ignored. Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int port_component_labels(int node_count, const PortList *ports, const uint8_t *edge_mask,
                                            const uint8_t *port_mask, const uint8_t *node_mask, int *labels);

#endif
