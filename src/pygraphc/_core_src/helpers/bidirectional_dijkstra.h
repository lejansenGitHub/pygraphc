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

/* Cheapest source-target path. weights are one per edge without a port list,
   one per port with one (ports non-NULL); two_port_edge_mask is the edge mask
   with port-masked two-port edges folded in. Returns -1 with an exception set. */
PYGRAPHC_INTERNAL int bidirectional_dijkstra(int node_count, int edge_count, const AdjacencyList *adjacency,
                                             const double *weights, const PortList *ports,
                                             const uint8_t *two_port_edge_mask, const uint8_t *edge_mask,
                                             const uint8_t *port_mask, const uint8_t *node_mask, int directed,
                                             int source, int target, BidirectionalPath *out);
PYGRAPHC_INTERNAL PyObject *bidirectional_path_to_list(const BidirectionalPath *bidirectional_path, const NodeIdContext *context);

#endif
