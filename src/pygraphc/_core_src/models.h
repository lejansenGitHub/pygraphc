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

/* ── Adjacency list (CSR format) for graph traversal ── */

typedef struct {
    int *offset;                /* size node_count+1, CSR row pointers */
    int *neighbors;             /* size 2*edge_count (undirected) or edge_count (directed), neighbor indices */
    int *edge_indices;          /* same size, original edge index */
    int *reverse_offset;        /* directed only: size node_count+1, reverse CSR (NULL if undirected) */
    int *reverse_neighbors;     /* directed only: size edge_count, predecessor indices */
    int *reverse_edge_indices;  /* directed only: size edge_count, original edge indices */
} AdjacencyList;

PYGRAPHC_INTERNAL int build_adjacency(int node_count, EdgeList *edges, AdjacencyList *adjacency, int directed);
PYGRAPHC_INTERNAL void free_adjacency(AdjacencyList *adjacency);

/* ── Edges with ports: an edge is its list of ports, every port sits at a node ── */

typedef struct {
    int edge_count;
    int port_count;
    int *port_offsets;       /* edge_count + 1: the ports of edge e are port_offsets[e] .. port_offsets[e + 1] - 1 */
    int *port_nodes;         /* port_count: the node index each port sits at */
    int *port_edges;         /* port_count: the edge each port belongs to */
    uint8_t *port_is_out;    /* port_count, directed only (NULL otherwise): 1 for an out-port, 0 for an in-port */
    int *incidence_offsets;  /* node_count + 1: the ports at node v are incidence_ports[incidence_offsets[v] ..] */
    int *incidence_ports;    /* port_count: port indices grouped by node */
} PortList;

PYGRAPHC_INTERNAL void free_port_list(PortList *ports);

/* ── GraphContext: cached parsed graph for reuse across algorithms ── */

typedef struct {
    NodeIdContext node_ids;
    AdjacencyList adjacency;
    int           has_adjacency;
    int           directed;  /* 0 = undirected, 1 = directed */
    PortList     *ports;     /* NULL for a graph of lines; set when an edge has other than two ports */
} GraphContext;

PYGRAPHC_INTERNAL void graph_context_destructor(PyObject *capsule);

static inline GraphContext *get_graph_context(PyObject *capsule) {
    GraphContext *graph = (GraphContext *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!graph) PyErr_SetString(PyExc_TypeError, "expected a parsed Graph capsule");
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
