/* Breadth-first order and edge-simple paths over the adjacency list. */
#ifndef PYGRAPHC_CORE_TRAVERSAL_H
#define PYGRAPHC_CORE_TRAVERSAL_H

#include "../models.h"

/*
 * Breadth-first order from source into queue; returns the number of nodes
 * visited. A two-terminal edge leads to its neighbor, an edge with three or
 * more terminals is entered through the terminal of its slot and, the first
 * time, leads to the nodes of all its active exit terminals (directed:
 * out-terminals). visited holds node_count zeroed ints, queue node_count ints;
 * expanded holds edge_count zeroed bytes and is needed only with a terminal list.
 */
PYGRAPHC_INTERNAL int breadth_first_order(const AdjacencyList *adjacency, const TerminalList *terminals, int directed,
                                          int source, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                          const uint8_t *node_mask, int *visited, int *queue, uint8_t *expanded);

/* Receives each path found as its edge indices in order; a negative return stops the search with that status. */
typedef int (*EdgePathVisitor)(void *visitor_state, const int *path_edges, int length);

/*
 * Every edge-simple path (no edge twice) from source to a node with is_target
 * set, depth-first, with at most cutoff edges (cutoff <= 0: up to 2 * edge_count).
 * node_simple forbids entering a node twice, ignore_self_loops never traverses a
 * self-loop. A path is reported when it reaches a target, before node_simple
 * decides whether the target can be entered. Returns 0, -1 with MemoryError set,
 * or the first negative return of visit.
 */
PYGRAPHC_INTERNAL int all_edge_paths(const AdjacencyList *adjacency, int node_count, Py_ssize_t edge_count, int source,
                                     const uint8_t *is_target, int cutoff, const uint8_t *edge_mask,
                                     const uint8_t *node_mask, int node_simple, int ignore_self_loops,
                                     EdgePathVisitor visit, void *visitor_state);

#endif
