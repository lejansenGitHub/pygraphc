/* A fundamental cycle basis over the incidence graph of an undirected graph. */
#ifndef PYGRAPHC_CORE_CYCLE_BASIS_H
#define PYGRAPHC_CORE_CYCLE_BASIS_H

#include "terminals.h"

/* Receives each cycle as its node indices in order; a negative return stops the search with that status. */
typedef int (*CycleVisitor)(void *visitor_state, const int *nodes, int length);

/*
 * A fundamental cycle basis (D25): the states are the nodes and, on a graph with
 * multi-terminal edges, one state per edge with three or more terminals
 * (node_count + e), joined to the node of each of its active terminals when it has
 * at least two; a two-terminal edge joins its endpoints directly. Every step has an
 * id (a two-terminal edge its index, terminal t edge_count + t); a BFS tree over
 * the states leaves one non-tree step per independent cycle, so there are
 * terminals - nodes - edges + components cycles. A cycle holds the nodes of a
 * Berge cycle in order. A self-loop, and an edge with two terminals at one node,
 * give the cycle [node]. Returns 0, -1 with MemoryError set, or the first
 * negative return of visit.
 */
PYGRAPHC_INTERNAL int fundamental_cycle_basis(const GraphContext *graph, const uint8_t *two_terminal_edges,
                                              const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                              const uint8_t *node_mask, CycleVisitor visit, void *visitor_state);

#endif
