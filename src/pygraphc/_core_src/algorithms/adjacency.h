/* The adjacency list of a graph: CSR arrays built from its edge list. */
#ifndef PYGRAPHC_CORE_ADJACENCY_H
#define PYGRAPHC_CORE_ADJACENCY_H

#include "../models.h"

/*
 * The adjacency list of a graph (CSR, reverse CSR when directed). A
 * two-terminal edge sits at both endpoints (directed: forward at its first,
 * reverse at its second); with a terminal list, an edge with three or more
 * terminals sits as -1 - e at each of its terminals (directed: in-terminals
 * forward, out-terminals in reverse) with that terminal in edge_indices, and a
 * single-terminal edge nowhere. terminals is NULL when every edge has two
 * terminals. Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int build_adjacency(int node_count, const EdgeList *edges, const TerminalList *terminals,
                                      AdjacencyList *adjacency, int directed);

#endif
