/* The longest path of a directed acyclic graph: topological order plus dynamic programming. */
#ifndef PYGRAPHC_CORE_LONGEST_PATH_H
#define PYGRAPHC_CORE_LONGEST_PATH_H

#include "terminals.h"

/*
 * The node indices of a longest path into *path_out (caller frees) and its length
 * into *path_length_out. A two-terminal edge weighs its weight (with a terminal
 * list, the sum of its two terminal weights), 1 when weights is NULL; a pass
 * through a multi-terminal edge weighs its entry plus its exit terminal (D11) and
 * counts 1 unweighted. The end is the node with the longest distance, the first in
 * topological order on a tie; with no positive distance the path is that first
 * node alone, or empty without nodes. Returns 0, or -1 with ValueError set on a
 * cycle or MemoryError.
 */
PYGRAPHC_INTERNAL int dag_longest_path(const GraphContext *graph, const double *weights, const uint8_t *edge_mask,
                                       const TerminalMask *terminal_mask, const uint8_t *node_mask, int **path_out,
                                       int *path_length_out);

#endif
