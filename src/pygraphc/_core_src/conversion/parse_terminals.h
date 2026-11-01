/* Edges with terminals from Python: parsing edges and terminal masks. */
#ifndef PYGRAPHC_CORE_PARSE_TERMINALS_H
#define PYGRAPHC_CORE_PARSE_TERMINALS_H

#include "../algorithms/terminals.h"

/*
 * Parse edges given as tuples of node ids, one per terminal, or as an
 * (edge_count, 2) int buffer, into graph->node_ids.edges and, when some edge
 * has other than two terminals, graph->terminals; directed edges are (u, v) or
 * (in_nodes, out_nodes). While every edge has two terminals the terminal list
 * stays implicit (edge e at terminals 2e and 2e + 1, the edge list holding its
 * nodes); otherwise any edge of other than two terminals sits in the edge list
 * as (node of its first terminal, same node), which joins nothing. The node
 * ids must already be parsed. Returns 0, or -1 with an exception set.
 */
PYGRAPHC_INTERNAL int parse_terminal_edges(GraphContext *graph, PyObject *edges_obj);

/*
 * Parse an optional terminal mask (None masks nothing), one byte per terminal. Every
 * graph numbers its terminals: with a terminal list as listed there, without one edge e
 * has terminals 2e and 2e + 1. A two-terminal edge with a masked terminal joins nothing
 * (masking case 5), so it is folded into a copy of the edge mask once per call
 * and the kernels check two-terminal edges by edge mask alone. Returns 0, or -1
 * with an exception set.
 */
PYGRAPHC_INTERNAL int parse_terminal_mask(const GraphContext *graph, PyObject *terminal_mask_obj, const uint8_t *edge_mask,
                                      TerminalMask *mask);
PYGRAPHC_INTERNAL void release_terminal_mask(TerminalMask *mask);

#endif
