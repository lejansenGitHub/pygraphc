/* Iterative Tarjan over the search states of a graph: critical edges, critical nodes and blocks. */
#ifndef PYGRAPHC_CORE_BICONNECTIVITY_H
#define PYGRAPHC_CORE_BICONNECTIVITY_H

#include "terminals.h"

/*
 * The search states of a graph are its nodes and, when it has multi-terminal edges,
 * one state per edge with three or more terminals (node_count + e), linked to the
 * node of each of its active terminals. A link is a two-terminal edge (its edge
 * index) or one terminal (edge_count + terminal), so a search never walks back
 * over the link it came in by, while a second terminal of one edge at the same
 * node is a back link (case 7).
 *
 * - critical edges: a two-terminal edge whose tree link has low[child] >
 *   discovery[parent] (a bridge), and a multi-terminal edge whose state has a child
 *   node c with low[c] >= discovery[edge state], i.e. removing the edge separates its
 *   terminals; in the order the search finds them;
 * - critical nodes: as in Tarjan, counting only children whose subtree holds a
 *   node, so an edge with one active terminal never makes its node critical;
 * - blocks: closed only below a node, never below an edge state, so the blocks
 *   meeting at a multi-terminal edge merge into one (D14); a block with fewer than
 *   two nodes joins nothing and is left out; edge states are not members;
 * - simple path blocks: the blocks of the search states themselves, closed below
 *   every state, edge states included, so a multi-terminal edge can separate them
 *   as a node does; without multi-terminal edges they are the blocks. A simple path
 *   visits each node and each multi-terminal edge at most once, so the block-cut
 *   tree of these blocks gives the nodes on simple paths;
 * - edge labels: the blocks, as a block id per edge in the order the blocks close
 *   (D27). The link stack holds (link, edge) instead of (state, neighbor); a block
 *   closed below an edge state without a child node holds one node and labels
 *   nothing, so an edge whose active terminals all sit at one node keeps -1 like a
 *   self-loop.
 */
typedef enum {
    TARJAN_CRITICAL_EDGES, TARJAN_CRITICAL_NODES, TARJAN_BLOCKS, TARJAN_SIMPLE_PATH_BLOCKS, TARJAN_EDGE_LABELS
} TarjanMode;

/* What one search found; only the arrays of its mode are set, the others are NULL. */
typedef struct {
    int *critical_edges;        /* edge indices, in the order found */
    int critical_edge_count;
    int *critical_nodes;        /* critical nodes in the order found, a node once per reason */
    int critical_node_count;
    int *block_offsets;         /* block_count + 1: the states of block b are block_states[block_offsets[b] ..] */
    int *block_states;          /* the distinct states of each block: a node, or node_count + e for an edge state */
    int block_count;
} TarjanReport;

/*
 * Run the search in a mode other than edge labels on an undirected graph with an
 * adjacency list. Two-terminal edges are traversed under two_terminal_edge_mask;
 * edge_mask and terminal_mask decide which terminals of multi-terminal edges are
 * active. Returns 0, or -1 with MemoryError set and nothing to release.
 */
PYGRAPHC_INTERNAL int tarjan_report(const GraphContext *graph, const uint8_t *two_terminal_edge_mask,
                                    const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask,
                                    TarjanMode mode, TarjanReport *report);
PYGRAPHC_INTERNAL void release_tarjan_report(TarjanReport *report);

/*
 * The edge labels of the search into labels (edge_count entries): a block id per
 * edge, -1 for a masked edge, an edge at a masked node and a self-loop. Masks as for
 * tarjan_report. Returns 0, or -1 with MemoryError set.
 */
PYGRAPHC_INTERNAL int tarjan_edge_labels(const GraphContext *graph, const uint8_t *two_terminal_edge_mask,
                                         const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask,
                                         int32_t *labels);

#endif
