/* Per-node and per-edge labels written into int32 arrays: component labels, quotient edges, degrees. */
#ifndef PYGRAPHC_CORE_LABELS_H
#define PYGRAPHC_CORE_LABELS_H

#include "components.h"

/*
 * labels[node] is the smallest node index of its component, -1 for a masked node.
 * Component ids of compute_graph_components are numbered in order of first
 * appearance while scanning node indices upwards, so the first index carrying an
 * id is the smallest of its component. Returns 0, or -1 with MemoryError set.
 */
PYGRAPHC_INTERNAL int smallest_node_labels(const ComponentResult *components, int node_count, const uint8_t *node_mask,
                                           int32_t *labels);

/*
 * One pass over the edges in index order: an unmasked edge whose endpoints carry
 * different labels is crossing (its two labels and its index into the three
 * crossing arrays), one whose endpoints carry the same label is internal. Edges at
 * a node labelled -1 are neither. Each array holds edge_count entries; the counts
 * go to *crossing_count and *internal_count.
 */
PYGRAPHC_INTERNAL void split_quotient_edges(const EdgeList *edges, const int32_t *labels, const uint8_t *edge_mask,
                                            int32_t *from_labels, int32_t *to_labels, int32_t *crossing_edges,
                                            int32_t *internal_edges, Py_ssize_t *crossing_count,
                                            Py_ssize_t *internal_count);

/*
 * The degree of every node under the masks into degrees (node_count entries): the
 * joined terminals at the node, undirected the active terminals of an edge with at
 * least two, directed the active in-terminals of an edge with an active
 * out-terminal (the out-degree); a self-loop counts twice, a masked node 0.
 */
PYGRAPHC_INTERNAL void joined_terminal_degrees(const GraphContext *graph, const uint8_t *two_terminal_edges,
                                               const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                               const uint8_t *node_mask, int32_t *degrees);

#endif
