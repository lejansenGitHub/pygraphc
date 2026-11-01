#ifndef PYGRAPHC_CORE_BIDIRECTIONAL_DIJKSTRA_H
#define PYGRAPHC_CORE_BIDIRECTIONAL_DIJKSTRA_H

#include "../models.h"
#include "terminals.h"

/* ── Bidirectional Dijkstra (source -> target) ── */

/* Result of a bidirectional search: the path weight and the node indices
   from source to target. path is NULL when the target is unreachable. */
typedef struct {
    double weight;
    int *path;
    int path_length;
} BidirectionalPath;

/* Cheapest source-target path. weights are one per edge without a terminal list,
   one per terminal with one (terminals non-NULL, with their pass_weights);
   two_terminal_edge_mask is the edge mask with terminal-masked two-terminal edges folded
   in. Returns -1 with an exception set. */
PYGRAPHC_INTERNAL int bidirectional_dijkstra(int node_count, int edge_count, const AdjacencyList *adjacency,
                                             const double *weights, const TerminalList *terminals,
                                             const PassWeights *pass_weights,
                                             const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
                                             const uint8_t *terminal_mask, const uint8_t *node_mask, int directed,
                                             int source, int target, BidirectionalPath *out);

#endif
