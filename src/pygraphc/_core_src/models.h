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
    int *src;
    int *dst;
    Py_ssize_t m;
    int owns_memory;  /* 1 = we malloc'd src/dst, 0 = points into numpy buffer */
} EdgeList;

/* Inline edge access macros — handles both interleaved and split layouts */
#define EDGE_SRC(el, i) ((el)->dst == NULL ? (el)->src[(i)*2]   : (el)->src[i])
#define EDGE_DST(el, i) ((el)->dst == NULL ? (el)->src[(i)*2+1] : (el)->dst[i])

PYGRAPHC_INTERNAL void free_edges(EdgeList *el);

/* ── Integer hash map (open addressing, power-of-2) ── */

typedef struct {
    long *keys;
    int *vals;
    int cap;      /* always power of 2 */
    int mask;     /* cap - 1 */
} IntMap;

PYGRAPHC_INTERNAL int intmap_init(IntMap *m, int n);
PYGRAPHC_INTERNAL void intmap_free(IntMap *m);

static inline void intmap_put(IntMap *m, long key, int val) {
    int i = (int)((unsigned long)key & (unsigned long)m->mask);
    while (m->keys[i] != -1) {
        if (m->keys[i] == key) { m->vals[i] = val; return; }
        i = (i + 1) & m->mask;
    }
    m->keys[i] = key;
    m->vals[i] = val;
}

static inline int intmap_get(IntMap *m, long key) {
    int i = (int)((unsigned long)key & (unsigned long)m->mask);
    while (m->keys[i] != -1) {
        if (m->keys[i] == key) return m->vals[i];
        i = (i + 1) & m->mask;
    }
    return -1;  /* not found */
}

/* ── NidContext: shared setup for node-ID-based API ── */

typedef struct {
    PyObject *nid_fast;
    PyObject **nid_items;
    int n;
    IntMap im;
    EdgeList el;
} NidContext;

PYGRAPHC_INTERNAL void nid_free(NidContext *ctx);

/* ── Helper: run union-find, return component_id array + num_components ── */

typedef struct {
    int *labels;    /* labels[node] = component_id (sequential 0..num_comp-1) */
    int num_comp;
} ComponentResult;

/* ── Adjacency list (CSR format) for graph traversal ── */

typedef struct {
    int *offset;      /* size n+1, CSR row pointers */
    int *adj;         /* size 2*m (undirected) or m (directed), neighbor indices */
    int *eid;         /* same size, original edge index */
    int *rev_offset;  /* directed only: size n+1, reverse CSR (NULL if undirected) */
    int *rev_adj;     /* directed only: size m, predecessor indices */
    int *rev_eid;     /* directed only: size m, original edge indices */
} AdjList;

PYGRAPHC_INTERNAL int build_adj(int n, EdgeList *el, AdjList *al, int directed);
PYGRAPHC_INTERNAL void free_adj(AdjList *al);

/* ── GraphCtx: cached parsed graph for reuse across algorithms ── */

typedef struct {
    NidContext nid;
    AdjList    al;
    int        has_adj;
    int        directed;  /* 0 = undirected, 1 = directed */
} GraphCtx;

PYGRAPHC_INTERNAL void graphctx_destructor(PyObject *capsule);

static inline GraphCtx *get_graphctx(PyObject *capsule) {
    GraphCtx *g = (GraphCtx *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!g) PyErr_SetString(PyExc_TypeError, "expected a parsed Graph capsule");
    return g;
}

/* ── Weight parsing: list of floats or numpy float64 array ── */

typedef struct {
    double *w;
    Py_ssize_t n;
    int owns_memory;
} WeightList;

PYGRAPHC_INTERNAL void free_weights(WeightList *wl);

/* ── Edge and node masks: borrowed byte buffers, released after use ── */

static inline void release_mask(Py_buffer *mask_buf) {
    if (mask_buf->obj) PyBuffer_Release(mask_buf);
}

#endif
