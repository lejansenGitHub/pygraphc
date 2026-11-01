/* Dijkstra over the search states of a graph: distances from sources, and the labelled point-to-point search. */
#ifndef PYGRAPHC_CORE_DIJKSTRA_H
#define PYGRAPHC_CORE_DIJKSTRA_H

#include "terminals.h"
#include "bidirectional_dijkstra.h"

/*
 * Distances from the given source indices (negative, masked or repeated sources
 * skipped) into *distance_out, one per search state with the nodes first;
 * HUGE_VAL when unreached, and with cutoff >= 0 no distance beyond it. weights
 * holds one weight per edge, or per terminal on a graph with a terminal list.
 * The caller frees *distance_out. Returns 0, or -1 with MemoryError set.
 */
PYGRAPHC_INTERNAL int shortest_path_distances(const GraphContext *graph, const double *weights,
                                              const PassWeights *pass_weights, const uint8_t *two_terminal_edge_mask,
                                              const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                              const uint8_t *node_mask, const int *sources, Py_ssize_t source_count,
                                              double cutoff, double **distance_out);

/*
 * Cheapest path from source to target on an undirected graph with a negative
 * terminal weight, into out (weight HUGE_VAL and no path when unreachable).
 * Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int labeled_shortest_path(const GraphContext *graph, const double *weights,
                                            const PassWeights *pass_weights, const uint8_t *two_terminal_edge_mask,
                                            const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                            const uint8_t *node_mask, int source, int target, BidirectionalPath *out);

#endif
