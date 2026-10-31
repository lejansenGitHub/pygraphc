/* Component labelling kernels: union-find components and Tarjan strongly connected components. */
#ifndef PYGRAPHC_CORE_COMPONENTS_H
#define PYGRAPHC_CORE_COMPONENTS_H

#include "../models.h"

PYGRAPHC_INTERNAL int compute_components_masked(int n, EdgeList *el, const uint8_t *mask,
                                                const uint8_t *node_mask,
                                                ComponentResult *cr);
PYGRAPHC_INTERNAL int compute_strongly_connected_components_masked(int n, AdjacencyList *al, const uint8_t *mask,
                                         const uint8_t *node_mask, ComponentResult *cr);

static inline int compute_components(int n, EdgeList *el, ComponentResult *cr) {
    return compute_components_masked(n, el, NULL, NULL, cr);
}

#endif
