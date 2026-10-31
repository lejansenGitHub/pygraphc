/* Component labelling kernels: union-find components and Tarjan strongly connected components. */
#ifndef PYGRAPHC_CORE_COMPONENTS_H
#define PYGRAPHC_CORE_COMPONENTS_H

#include "../models.h"
#include "ports.h"

PYGRAPHC_INTERNAL int compute_components_masked(int node_count, EdgeList *edges, const uint8_t *edge_mask,
                                                const uint8_t *node_mask,
                                                ComponentResult *components);
PYGRAPHC_INTERNAL int compute_strongly_connected_components_masked(int node_count, AdjacencyList *adjacency,
                                                                   const uint8_t *edge_mask, const uint8_t *node_mask,
                                                                   ComponentResult *components);

/*
 * Components of a graph with multiport edges: two-port edges as in
 * compute_components_masked, every other edge joining its active ports.
 * Returns 0 on success, -1 with an exception set.
 */
PYGRAPHC_INTERNAL int compute_components_with_ports(int node_count, EdgeList *edges, const PortList *ports,
                                                    const uint8_t *edge_mask, const PortMask *port_mask,
                                                    const uint8_t *node_mask, ComponentResult *components);

static inline int compute_components(int node_count, EdgeList *edges, ComponentResult *components) {
    return compute_components_masked(node_count, edges, NULL, NULL, components);
}

#endif
