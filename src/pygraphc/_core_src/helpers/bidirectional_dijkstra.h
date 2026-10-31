#ifndef PYGRAPHC_CORE_BIDIRECTIONAL_DIJKSTRA_H
#define PYGRAPHC_CORE_BIDIRECTIONAL_DIJKSTRA_H

#include "../models.h"

/* ── Bidirectional Dijkstra (source -> target) ── */

/* Result of a bidirectional search: the path weight and the node indices
   from source to target. path is NULL when the target is unreachable. */
typedef struct {
    double weight;
    int *path;
    int path_length;
} BidirectionalPath;

PYGRAPHC_INTERNAL int bidirectional_dijkstra(int node_count, const AdjacencyList *adjacency, const double *edge_weights,
                                             const uint8_t *edge_mask, const uint8_t *node_mask,
                                             int directed, int source, int target, BidirectionalPath *out);
PYGRAPHC_INTERNAL PyObject *bidirectional_path_to_list(const BidirectionalPath *bidirectional_path, const NodeIdContext *context);

#endif
