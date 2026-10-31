/* Topological sort kernels (Kahn's algorithm). */
#ifndef PYGRAPHC_CORE_TOPOSORT_H
#define PYGRAPHC_CORE_TOPOSORT_H

#include "../models.h"

PYGRAPHC_INTERNAL int topological_order(int node_count, EdgeList *edges, int **out_order);
PYGRAPHC_INTERNAL int topological_order_masked(int node_count, EdgeList *edges, const uint8_t *edge_mask,
                                               const uint8_t *node_mask, int **out_order);

#endif
