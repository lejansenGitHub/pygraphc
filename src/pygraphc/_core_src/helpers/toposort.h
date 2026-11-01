/* Topological order of a directed graph (Kahn's algorithm over its search states). */
#ifndef PYGRAPHC_CORE_TOPOSORT_H
#define PYGRAPHC_CORE_TOPOSORT_H

#include "../models.h"
#include "terminals.h"

/*
 * A topological order of states: nodes, and node_count + e for an edge with
 * three or more terminals. The order is complete (the graph acyclic under its
 * masks) when nodes_in_order equals node_count, the number of unmasked nodes.
 */
typedef struct {
    int *states;         /* state_count states in order; the caller frees it */
    int state_count;
    int node_count;
    int nodes_in_order;
} TopologicalOrder;

/*
 * Topological order of a parsed directed graph under its edge, terminal and
 * node masks; a multi-terminal edge orders every active in-terminal node
 * before every active out-terminal node (D5). Returns 0, or -1 with an
 * exception set.
 */
PYGRAPHC_INTERNAL int topological_order(const GraphContext *graph, const uint8_t *edge_mask,
                                        const TerminalMask *terminal_mask, const uint8_t *node_mask,
                                        TopologicalOrder *result);

#endif
