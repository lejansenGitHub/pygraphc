/* Component labelling kernels: union-find components and Tarjan strongly connected components. */
#ifndef PYGRAPHC_CORE_COMPONENTS_H
#define PYGRAPHC_CORE_COMPONENTS_H

#include "../models.h"

PYGRAPHC_INTERNAL int compute_components_masked(int node_count, EdgeList *edges, const uint8_t *edge_mask,
                                                const uint8_t *node_mask,
                                                ComponentResult *components);
PYGRAPHC_INTERNAL int compute_strongly_connected_components_masked(int node_count, AdjacencyList *adjacency,
                                                                   const uint8_t *edge_mask, const uint8_t *node_mask,
                                                                   ComponentResult *components);

static inline int compute_components(int node_count, EdgeList *edges, ComponentResult *components) {
    return compute_components_masked(node_count, edges, NULL, NULL, components);
}

#endif
