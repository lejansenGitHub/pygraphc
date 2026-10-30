/* Topological sort kernels (Kahn's algorithm). */
#ifndef PYGRAPHC_CORE_TOPOSORT_H
#define PYGRAPHC_CORE_TOPOSORT_H

#include "../models.h"

PYGRAPHC_INTERNAL int toposort_core(int n, EdgeList *el, int **out_order);
PYGRAPHC_INTERNAL int toposort_core_masked(int n, EdgeList *el, const uint8_t *mask,
                                           const uint8_t *node_mask, int **out_order);

#endif
