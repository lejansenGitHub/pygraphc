/* Data models of the _core extension: the structs the kernels share and their small helpers. */
#ifndef PYGRAPHC_CORE_MODELS_H
#define PYGRAPHC_CORE_MODELS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Symbols shared between the translation units of the extension, kept out of its export table. */
#if defined(__GNUC__) || defined(__clang__)
#define PYGRAPHC_INTERNAL __attribute__((visibility("hidden")))
#else
#define PYGRAPHC_INTERNAL
#endif

/* ── Edge parsing: supports both list-of-tuples and numpy int32 array ── */

typedef struct {
    int *first_endpoints;   /* interleaved (edge_count, 2) pairs when second_endpoints is NULL */
    int *second_endpoints;
    Py_ssize_t edge_count;
    int owns_memory;  /* 1 = we malloc'd the endpoint arrays, 0 = points into numpy buffer */
} EdgeList;

/* Inline edge access macros — handles both interleaved and split layouts */
#define EDGE_FIRST_ENDPOINT(edges, edge_index) ((edges)->second_endpoints == NULL ? (edges)->first_endpoints[(edge_index)*2]   : (edges)->first_endpoints[edge_index])
#define EDGE_SECOND_ENDPOINT(edges, edge_index) ((edges)->second_endpoints == NULL ? (edges)->first_endpoints[(edge_index)*2+1] : (edges)->second_endpoints[edge_index])

PYGRAPHC_INTERNAL void free_edges(EdgeList *edges);

/* ── Integer hash map (open addressing, power-of-2) ── */

typedef struct {
    long *keys;
    int *values;
    int capacity;       /* always power of 2 */
    int capacity_mask;  /* capacity - 1, a hash bit mask */
} IntMap;

PYGRAPHC_INTERNAL int intmap_init(IntMap *map, int expected_count);
PYGRAPHC_INTERNAL void intmap_free(IntMap *map);

static inline void intmap_put(IntMap *map, long key, int value) {
    int slot = (int)((unsigned long)key & (unsigned long)map->capacity_mask);
    while (map->keys[slot] != -1) {
        if (map->keys[slot] == key) { map->values[slot] = value; return; }
        slot = (slot + 1) & map->capacity_mask;
    }
    map->keys[slot] = key;
    map->values[slot] = value;
}

static inline int intmap_get(IntMap *map, long key) {
    int slot = (int)((unsigned long)key & (unsigned long)map->capacity_mask);
    while (map->keys[slot] != -1) {
        if (map->keys[slot] == key) return map->values[slot];
        slot = (slot + 1) & map->capacity_mask;
    }
    return -1;  /* not found */
}

/* ── NodeIdContext: shared setup for node-ID-based API ── */

typedef struct {
    PyObject *node_ids_fast;
    PyObject **node_id_items;
    int node_count;
    IntMap node_index_by_node_id;
    EdgeList edges;
} NodeIdContext;

PYGRAPHC_INTERNAL void node_id_context_free(NodeIdContext *context);

/* ── Helper: run union-find, return component_id array + num_components ── */

typedef struct {
    int *labels;    /* labels[node] = component_id (sequential 0..component_count-1) */
    int component_count;
} ComponentResult;

/* ── Adjacency list (CSR format) for graph traversal ──
 *
 * In a graph with multi-terminal edges, an edge e with three or more terminals appears
 * as neighbors[slot] = -1 - e at the node of each of its terminals (directed: of
 * each in-terminal in the forward lists, of each out-terminal in the reverse lists),
 * and edge_indices[slot] then holds the terminal that slot stands for.
 */

typedef struct {
    int *offset;                /* size node_count+1, CSR row pointers */
    int *neighbors;             /* size 2*edge_count (undirected) or edge_count (directed), neighbor indices */
    int *edge_indices;          /* same size, original edge index (the terminal, where neighbors is negative) */
    int *reverse_offset;        /* directed only: size node_count+1, reverse CSR (NULL if undirected) */
    int *reverse_neighbors;     /* directed only: size edge_count, predecessor indices */
    int *reverse_edge_indices;  /* directed only: size edge_count, original edge indices */
} AdjacencyList;

PYGRAPHC_INTERNAL void free_adjacency(AdjacencyList *adjacency);

/* ── Edges with terminals: an edge is its list of terminals, every terminal sits at a node ──
 *
 * Kept for graphs with an edge of other than two terminals. Terminals are numbered over
 * all edges in edge order, a two-terminal edge included (terminal 0 at its first
 * endpoint, terminal 1 at its second), so that a terminal mask has one entry per terminal.
 */

typedef struct {
    int terminal_count;
    int *terminal_offsets;      /* edge_count + 1: the terminals of edge e are terminal_offsets[e] .. terminal_offsets[e + 1] - 1 */
    int *terminal_nodes;        /* terminal_count: the node index each terminal sits at */
    int *out_terminal_offsets;  /* edge_count, directed only (NULL otherwise): out-terminals of e start here, in-terminals before */
} TerminalList;

PYGRAPHC_INTERNAL void free_terminal_list(TerminalList *terminals);

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

/* Whether edge e has three or more terminals and so appears as -1 - e in the adjacency list. */
static inline int is_multi_terminal_edge(const TerminalList *terminals, int edge) {
    return terminals->terminal_offsets[edge + 1] - terminals->terminal_offsets[edge] > 2;
}

/*
 * The one rule every algorithm on edges with terminals asks: a terminal is active iff
 * its edge, the terminal itself and its node are all unmasked. A NULL mask masks
 * nothing.
 */
static inline int terminal_is_active(const TerminalList *terminals, int edge, int terminal, const uint8_t *edge_mask,
                                 const uint8_t *terminal_mask, const uint8_t *node_mask) {
    if (edge_mask && edge_mask[edge]) return 0;
    if (terminal_mask && terminal_mask[terminal]) return 0;
    if (node_mask && node_mask[terminals->terminal_nodes[terminal]]) return 0;
    return 1;
}

/* ── GraphContext: cached parsed graph for reuse across algorithms ── */

typedef struct {
    NodeIdContext node_ids;
    AdjacencyList adjacency;
    int           has_adjacency;
    int           directed;  /* 0 = undirected, 1 = directed */
    TerminalList     *terminals;     /* NULL for a graph of two-terminal edges; set when an edge has other than two terminals */
} GraphContext;

PYGRAPHC_INTERNAL void graph_context_destructor(PyObject *capsule);
/* Free a parsed graph: adjacency, terminal list, edges, node ids. */
PYGRAPHC_INTERNAL void free_graph_context(GraphContext *graph);

static inline GraphContext *get_graph_context(PyObject *capsule) {
    GraphContext *graph = (GraphContext *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!graph) PyErr_SetString(PyExc_TypeError, "expected a parsed Graph capsule");
    return graph;
}

/* The parsed graph of a kernel that reads only two-terminal edges; a graph with multi-terminal edges raises. */
static inline GraphContext *require_two_terminal_edges(PyObject *capsule) {
    GraphContext *graph = get_graph_context(capsule);
    if (graph && graph->terminals) {
        PyErr_SetString(PyExc_NotImplementedError, "not available on graphs with multi-terminal edges");
        return NULL;
    }
    return graph;
}

/* ── Weight parsing: list of floats or numpy float64 array ── */

typedef struct {
    double *values;
    Py_ssize_t count;
    int owns_memory;
    int has_sign_bit;  /* parse_counted_weights with tracks_sign: some value has its sign bit set (negative or -0.0) */
} WeightList;

PYGRAPHC_INTERNAL void free_weights(WeightList *weights);

/* ── Edge and node masks: borrowed byte buffers, released after use ── */

static inline void release_mask(Py_buffer *mask_buffer) {
    if (mask_buffer->obj) PyBuffer_Release(mask_buffer);
}

#endif
