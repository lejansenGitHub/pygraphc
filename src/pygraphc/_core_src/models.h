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
 * In a graph with multiport edges, an edge e with three or more ports appears
 * as neighbors[slot] = -1 - e at the node of each of its ports (directed: of
 * each in-port in the forward lists, of each out-port in the reverse lists),
 * and edge_indices[slot] then holds the port that slot stands for.
 */

typedef struct {
    int *offset;                /* size node_count+1, CSR row pointers */
    int *neighbors;             /* size 2*edge_count (undirected) or edge_count (directed), neighbor indices */
    int *edge_indices;          /* same size, original edge index (the port, where neighbors is negative) */
    int *reverse_offset;        /* directed only: size node_count+1, reverse CSR (NULL if undirected) */
    int *reverse_neighbors;     /* directed only: size edge_count, predecessor indices */
    int *reverse_edge_indices;  /* directed only: size edge_count, original edge indices */
} AdjacencyList;

PYGRAPHC_INTERNAL int build_adjacency(int node_count, EdgeList *edges, AdjacencyList *adjacency, int directed);
PYGRAPHC_INTERNAL void free_adjacency(AdjacencyList *adjacency);

/* ── Edges with ports: an edge is its list of ports, every port sits at a node ──
 *
 * Kept for graphs with an edge of other than two ports. Ports are numbered over
 * all edges in edge order, a two-port edge included (port 0 at its first
 * endpoint, port 1 at its second), so that a port mask has one entry per port.
 */

typedef struct {
    int port_count;
    int *port_offsets;      /* edge_count + 1: the ports of edge e are port_offsets[e] .. port_offsets[e + 1] - 1 */
    int *port_nodes;        /* port_count: the node index each port sits at */
    int *out_port_offsets;  /* edge_count, directed only (NULL otherwise): out-ports of e start here, in-ports before */
} PortList;

PYGRAPHC_INTERNAL void free_port_list(PortList *ports);

/* Whether edge e has three or more ports and so appears as -1 - e in the adjacency list. */
static inline int is_multiport_edge(const PortList *ports, int edge) {
    return ports->port_offsets[edge + 1] - ports->port_offsets[edge] > 2;
}

/*
 * The one rule every algorithm on edges with ports asks: a port is active iff
 * its edge, the port itself and its node are all unmasked. A NULL mask masks
 * nothing.
 */
static inline int port_is_active(const PortList *ports, int edge, int port, const uint8_t *edge_mask,
                                 const uint8_t *port_mask, const uint8_t *node_mask) {
    if (edge_mask && edge_mask[edge]) return 0;
    if (port_mask && port_mask[port]) return 0;
    if (node_mask && node_mask[ports->port_nodes[port]]) return 0;
    return 1;
}

/* ── GraphContext: cached parsed graph for reuse across algorithms ── */

typedef struct {
    NodeIdContext node_ids;
    AdjacencyList adjacency;
    int           has_adjacency;
    int           directed;  /* 0 = undirected, 1 = directed */
    PortList     *ports;     /* NULL for a graph of two-port edges; set when an edge has other than two ports */
} GraphContext;

PYGRAPHC_INTERNAL void graph_context_destructor(PyObject *capsule);

static inline GraphContext *get_graph_context(PyObject *capsule) {
    GraphContext *graph = (GraphContext *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!graph) PyErr_SetString(PyExc_TypeError, "expected a parsed Graph capsule");
    return graph;
}

/* The parsed graph of a kernel that reads only two-port edges; a graph with multiport edges raises. */
static inline GraphContext *get_two_port_graph_context(PyObject *capsule) {
    GraphContext *graph = get_graph_context(capsule);
    if (graph && graph->ports) {
        PyErr_SetString(PyExc_NotImplementedError, "not available on graphs with multiport edges");
        return NULL;
    }
    return graph;
}

/* ── Weight parsing: list of floats or numpy float64 array ── */

typedef struct {
    double *values;
    Py_ssize_t count;
    int owns_memory;
} WeightList;

PYGRAPHC_INTERNAL void free_weights(WeightList *weights);

/* ── Edge and node masks: borrowed byte buffers, released after use ── */

static inline void release_mask(Py_buffer *mask_buffer) {
    if (mask_buffer->obj) PyBuffer_Release(mask_buffer);
}

#endif
