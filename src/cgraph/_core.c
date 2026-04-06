#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Buffer-format helpers ── */

static inline int is_int32_fmt(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 4 || !fmt) return 0;
    while (*fmt == '<' || *fmt == '>' || *fmt == '=' || *fmt == '!') fmt++;
    return *fmt == 'i';
}

static inline int is_int64_fmt(const char *fmt, Py_ssize_t itemsize) {
    if (itemsize != 8 || !fmt) return 0;
    while (*fmt == '<' || *fmt == '>' || *fmt == '=' || *fmt == '!') fmt++;
    return *fmt == 'l' || *fmt == 'q';
}

/* ── Union-Find with path compression + union by rank ── */

typedef struct {
    int *parent;
    int *rank;
    int n;
} UF;

static inline int uf_init(UF *uf, int n) {
    uf->n = n;
    uf->parent = (int *)malloc(2 * (size_t)n * sizeof(int));
    if (!uf->parent) return -1;
    uf->rank = uf->parent + n;
    for (int i = 0; i < n; i++) { uf->parent[i] = i; uf->rank[i] = 0; }
    return 0;
}

static inline void uf_free(UF *uf) {
    free(uf->parent);  /* rank is part of same allocation */
}

static inline int uf_find(UF *uf, int x) {
    while (uf->parent[x] != x) {
        int next = uf->parent[uf->parent[x]];
        uf->parent[x] = next;
        x = next;
    }
    return x;
}

static inline void uf_union(UF *uf, int a, int b) {
    a = uf_find(uf, a);
    b = uf_find(uf, b);
    if (a == b) return;
    if (uf->rank[a] < uf->rank[b]) { int t = a; a = b; b = t; }
    uf->parent[b] = a;
    if (uf->rank[a] == uf->rank[b]) uf->rank[a]++;
}

/* ── Edge parsing: supports both list-of-tuples and numpy int32 array ── */

typedef struct {
    int *src;
    int *dst;
    Py_ssize_t m;
    int owns_memory;  /* 1 = we malloc'd src/dst, 0 = points into numpy buffer */
} EdgeList;

/*
 * Try the buffer protocol first (numpy int32 array with shape (m,2)).
 * Falls back to sequence-of-tuples parsing.
 * Returns 0 on success, -1 on error.
 */
static int parse_edges(PyObject *edges_obj, EdgeList *el) {
    Py_buffer buf;

    /* Fast path: buffer protocol (numpy array) */
    if (PyObject_GetBuffer(edges_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        /* Validate: must be int32 with shape (m, 2) */
        int is_int32 = (buf.format != NULL && (
            strcmp(buf.format, "i") == 0 ||
            strcmp(buf.format, "<i") == 0 ||
            strcmp(buf.format, "=i") == 0
        ) && buf.itemsize == 4);

        if (is_int32 && buf.ndim == 2 && buf.shape[1] == 2) {
            Py_ssize_t m = buf.shape[0];
            int *data = (int *)buf.buf;
            el->m = m;
            /* Point directly into the numpy buffer — zero copy for edges */
            el->src = NULL;
            el->dst = NULL;
            el->owns_memory = 0;

            /* We need separate src/dst arrays for the union-find loop,
               but we can read directly from the contiguous buffer:
               data[i*2] = src, data[i*2+1] = dst */

            /* Actually, just store the raw pointer and read inline.
               We'll use a special code path. Store raw ptr in src. */
            el->src = data;  /* raw (m,2) int32 buffer */
            el->dst = NULL;  /* sentinel: means "interleaved" */

            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();  /* buffer protocol not supported, try sequence */
    }

    /* Slow path: list of tuples */
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!fast) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);
    el->m = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->src = NULL;
        el->dst = NULL;
        Py_DECREF(fast);
        return 0;
    }

    el->src = (int *)malloc(m * sizeof(int));
    el->dst = (int *)malloc(m * sizeof(int));
    if (!el->src || !el->dst) {
        free(el->src); free(el->dst);
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *edge = items[i];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(el->src); free(el->dst);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **ep = PySequence_Fast_ITEMS(edge_fast);
        el->src[i] = (int)PyLong_AsLong(ep[0]);
        el->dst[i] = (int)PyLong_AsLong(ep[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(el->src); free(el->dst);
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    return 0;
}

/* Inline edge access macros — handles both interleaved and split layouts */
#define EDGE_SRC(el, i) ((el)->dst == NULL ? (el)->src[(i)*2]   : (el)->src[i])
#define EDGE_DST(el, i) ((el)->dst == NULL ? (el)->src[(i)*2+1] : (el)->dst[i])

static void free_edges(EdgeList *el) {
    if (el->owns_memory) {
        free(el->src);
        free(el->dst);
    }
}

/* ── Integer hash map (open addressing, power-of-2) ── */

typedef struct {
    long *keys;
    int *vals;
    int cap;      /* always power of 2 */
    int mask;     /* cap - 1 */
} IntMap;

static int intmap_init(IntMap *m, int n) {
    int cap = 16;
    while (cap < n * 2) cap <<= 1;
    m->cap = cap;
    m->mask = cap - 1;
    m->keys = (long *)malloc((size_t)cap * sizeof(long));
    m->vals = (int *)malloc((size_t)cap * sizeof(int));
    if (!m->keys || !m->vals) {
        free(m->keys); free(m->vals);
        PyErr_NoMemory();
        return -1;
    }
    memset(m->keys, 0xFF, (size_t)cap * sizeof(long)); /* -1 = empty sentinel */
    return 0;
}

static void intmap_free(IntMap *m) {
    free(m->keys);
    free(m->vals);
}

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

/* ── Edge parsing with node-ID translation via C hash map ── */

static int parse_edges_mapped(IntMap *im, PyObject *edges_obj, EdgeList *el) {
    /* Fast path: buffer protocol (numpy int32/int64 array with shape (m,2)) */
    Py_buffer buf;
    if (PyObject_GetBuffer(edges_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_i32 = is_int32_fmt(buf.format, buf.itemsize);
        int is_i64 = is_int64_fmt(buf.format, buf.itemsize);

        if ((is_i32 || is_i64) && buf.ndim == 2 && buf.shape[1] == 2) {
            Py_ssize_t m = buf.shape[0];
            el->m = m;
            el->owns_memory = 1;
            if (m == 0) {
                el->src = NULL; el->dst = NULL;
                PyBuffer_Release(&buf);
                return 0;
            }
            el->src = (int *)malloc((size_t)m * sizeof(int));
            el->dst = (int *)malloc((size_t)m * sizeof(int));
            if (!el->src || !el->dst) {
                free(el->src); free(el->dst);
                PyBuffer_Release(&buf);
                PyErr_NoMemory();
                return -1;
            }
            if (is_i32) {
                int *data = (int *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int u_idx = intmap_get(im, (long)data[i * 2]);
                    int v_idx = intmap_get(im, (long)data[i * 2 + 1]);
                    if (u_idx < 0 || v_idx < 0) {
                        free(el->src); free(el->dst);
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    el->src[i] = u_idx;
                    el->dst[i] = v_idx;
                }
            } else {  /* int64 */
                long long *data = (long long *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int u_idx = intmap_get(im, (long)data[i * 2]);
                    int v_idx = intmap_get(im, (long)data[i * 2 + 1]);
                    if (u_idx < 0 || v_idx < 0) {
                        free(el->src); free(el->dst);
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    el->src[i] = u_idx;
                    el->dst[i] = v_idx;
                }
            }
            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
        /* Fall through to slow path */
    } else {
        PyErr_Clear();
    }

    /* Slow path: list of tuples */
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence or numpy array");
    if (!fast) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);
    el->m = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->src = NULL; el->dst = NULL;
        Py_DECREF(fast);
        return 0;
    }

    el->src = (int *)malloc((size_t)m * sizeof(int));
    el->dst = (int *)malloc((size_t)m * sizeof(int));
    if (!el->src || !el->dst) {
        free(el->src); free(el->dst);
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *edge = items[i];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            free(el->src); free(el->dst);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return -1;
        }
        PyObject **ep = PySequence_Fast_ITEMS(edge_fast);
        long u_nid = PyLong_AsLong(ep[0]);
        long v_nid = PyLong_AsLong(ep[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            free(el->src); free(el->dst);
            Py_DECREF(fast);
            return -1;
        }
        int u_idx = intmap_get(im, u_nid);
        int v_idx = intmap_get(im, v_nid);
        if (u_idx < 0 || v_idx < 0) {
            free(el->src); free(el->dst);
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        el->src[i] = u_idx;
        el->dst[i] = v_idx;
    }
    Py_DECREF(fast);
    return 0;
}

/* ── Edge parsing from two separate src/dst sequences or 1D arrays ── */

static int _parse_one_array(IntMap *im, PyObject *obj, int *out, Py_ssize_t m,
                            const char *label) {
    /* Try buffer protocol: 1D int32 or int64 array */
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        int is_i32 = is_int32_fmt(buf.format, buf.itemsize);
        int is_i64 = is_int64_fmt(buf.format, buf.itemsize);
        if ((is_i32 || is_i64) && buf.ndim == 1 && buf.shape[0] == m) {
            if (is_i32) {
                int *data = (int *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int idx = intmap_get(im, (long)data[i]);
                    if (idx < 0) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[i] = idx;
                }
            } else {
                long long *data = (long long *)buf.buf;
                for (Py_ssize_t i = 0; i < m; i++) {
                    int idx = intmap_get(im, (long)data[i]);
                    if (idx < 0) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
                        return -1;
                    }
                    out[i] = idx;
                }
            }
            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
    } else {
        PyErr_Clear();
    }

    /* Slow path: sequence of ints */
    PyObject *fast = PySequence_Fast(obj, label);
    if (!fast) return -1;
    if (PySequence_Fast_GET_SIZE(fast) != m) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "src and dst must have the same length");
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        long nid = PyLong_AsLong(items[i]);
        if (nid == -1 && PyErr_Occurred()) { Py_DECREF(fast); return -1; }
        int idx = intmap_get(im, nid);
        if (idx < 0) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return -1;
        }
        out[i] = idx;
    }
    Py_DECREF(fast);
    return 0;
}

static int parse_edges_mapped_split(IntMap *im, PyObject *src_obj, PyObject *dst_obj,
                                    EdgeList *el) {
    /* Determine m from src */
    Py_ssize_t m;
    Py_buffer buf;
    if (PyObject_GetBuffer(src_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        m = (buf.ndim == 1) ? buf.shape[0] : -1;
        PyBuffer_Release(&buf);
        if (m < 0) {
            PyErr_SetString(PyExc_ValueError, "src must be a 1D array or sequence");
            return -1;
        }
    } else {
        PyErr_Clear();
        PyObject *fast = PySequence_Fast(src_obj, "src must be a sequence");
        if (!fast) return -1;
        m = PySequence_Fast_GET_SIZE(fast);
        Py_DECREF(fast);
    }

    el->m = m;
    el->owns_memory = 1;
    if (m == 0) {
        el->src = NULL; el->dst = NULL;
        return 0;
    }

    el->src = (int *)malloc((size_t)m * sizeof(int));
    el->dst = (int *)malloc((size_t)m * sizeof(int));
    if (!el->src || !el->dst) {
        free(el->src); free(el->dst);
        PyErr_NoMemory();
        return -1;
    }

    if (_parse_one_array(im, src_obj, el->src, m, "src must be a sequence") < 0) {
        free(el->src); free(el->dst);
        return -1;
    }
    if (_parse_one_array(im, dst_obj, el->dst, m, "dst must be a sequence") < 0) {
        free(el->src); free(el->dst);
        return -1;
    }
    return 0;
}

/* ── NidContext: shared setup for node-ID-based API ── */

typedef struct {
    PyObject *nid_fast;
    PyObject **nid_items;
    int n;
    IntMap im;
    EdgeList el;
} NidContext;

static int nid_parse(PyObject *node_ids_obj, PyObject *edges_obj, NidContext *ctx) {
    ctx->nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!ctx->nid_fast) return -1;
    ctx->n = (int)PySequence_Fast_GET_SIZE(ctx->nid_fast);
    ctx->nid_items = PySequence_Fast_ITEMS(ctx->nid_fast);
    ctx->im.keys = NULL;
    ctx->im.vals = NULL;
    ctx->el.src = NULL; ctx->el.dst = NULL;
    ctx->el.m = 0; ctx->el.owns_memory = 0;

    if (ctx->n == 0) return 0;

    if (intmap_init(&ctx->im, ctx->n) < 0) {
        Py_DECREF(ctx->nid_fast);
        return -1;
    }
    for (int i = 0; i < ctx->n; i++) {
        long nid = PyLong_AsLong(ctx->nid_items[i]);
        if (nid == -1 && PyErr_Occurred()) {
            intmap_free(&ctx->im);
            Py_DECREF(ctx->nid_fast);
            return -1;
        }
        intmap_put(&ctx->im, nid, i);
    }

    if (parse_edges_mapped(&ctx->im, edges_obj, &ctx->el) < 0) {
        intmap_free(&ctx->im);
        Py_DECREF(ctx->nid_fast);
        return -1;
    }
    return 0;
}

static int nid_parse_split(PyObject *node_ids_obj, PyObject *src_obj, PyObject *dst_obj,
                           NidContext *ctx) {
    ctx->nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!ctx->nid_fast) return -1;
    ctx->n = (int)PySequence_Fast_GET_SIZE(ctx->nid_fast);
    ctx->nid_items = PySequence_Fast_ITEMS(ctx->nid_fast);
    ctx->im.keys = NULL;
    ctx->im.vals = NULL;
    ctx->el.src = NULL; ctx->el.dst = NULL;
    ctx->el.m = 0; ctx->el.owns_memory = 0;

    if (ctx->n == 0) return 0;

    if (intmap_init(&ctx->im, ctx->n) < 0) {
        Py_DECREF(ctx->nid_fast);
        return -1;
    }
    for (int i = 0; i < ctx->n; i++) {
        long nid = PyLong_AsLong(ctx->nid_items[i]);
        if (nid == -1 && PyErr_Occurred()) {
            intmap_free(&ctx->im);
            Py_DECREF(ctx->nid_fast);
            return -1;
        }
        intmap_put(&ctx->im, nid, i);
    }

    if (parse_edges_mapped_split(&ctx->im, src_obj, dst_obj, &ctx->el) < 0) {
        intmap_free(&ctx->im);
        Py_DECREF(ctx->nid_fast);
        return -1;
    }
    return 0;
}

static void nid_free(NidContext *ctx) {
    free_edges(&ctx->el);
    if (ctx->im.keys) intmap_free(&ctx->im);
    Py_DECREF(ctx->nid_fast);
}

/* ── Helper: run union-find, return component_id array + num_components ── */

typedef struct {
    int *labels;    /* labels[node] = component_id (sequential 0..num_comp-1) */
    int num_comp;
} ComponentResult;

/*
 * Core routine: parse edges, run union-find, assign sequential component IDs.
 * Caller must free result.labels.
 * Returns 0 on success, -1 on error.
 */
static int compute_components(int n, EdgeList *el, ComponentResult *cr) {
    UF uf;
    if (uf_init(&uf, n) < 0) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t i = 0; i < el->m; i++)
        uf_union(&uf, EDGE_SRC(el, i), EDGE_DST(el, i));

    /* labels + root_to_id in one allocation */
    int *buf = (int *)malloc(2 * (size_t)n * sizeof(int));
    if (!buf) {
        uf_free(&uf);
        PyErr_NoMemory();
        return -1;
    }
    cr->labels = buf;
    int *root_to_id = buf + n;
    memset(root_to_id, -1, (size_t)n * sizeof(int));

    int num_comp = 0;
    for (int i = 0; i < n; i++) {
        int r = uf_find(&uf, i);
        if (root_to_id[r] == -1)
            root_to_id[r] = num_comp++;
        cr->labels[i] = root_to_id[r];
    }
    cr->num_comp = num_comp;

    uf_free(&uf);
    return 0;
}

/* ── connected_components(n, edges) -> list[set[int]] ── */

static PyObject *py_connected_components(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        free_edges(&el);
        return NULL;
    }
    free_edges(&el);

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); return NULL; }

    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result);
            return NULL;
        }
    }

    for (int i = 0; i < n; i++) {
        PyObject *val = PyLong_FromLong(i);
        PySet_Add(sets[cr.labels[i]], val);
        Py_DECREF(val);
    }

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    return result;
}

/* ── connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]] ── */

static PyObject *py_connected_components_with_branches(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "iOO", &n, &edges_obj, &branch_ids_obj)) return NULL;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    /* Parse branch_ids */
    PyObject *br_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!br_fast) { free_edges(&el); return NULL; }
    Py_ssize_t br_len = PySequence_Fast_GET_SIZE(br_fast);

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        Py_DECREF(br_fast);
        free_edges(&el);
        return NULL;
    }

    int num_comp = cr.num_comp;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = el.m < br_len ? el.m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);
    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        int c = cr.labels[EDGE_SRC(&el, i)];
        PySet_Add(branch_sets[c], br_items[i]);
    }
    Py_DECREF(br_fast);
    free_edges(&el);

    /* Build result list */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup2;

    for (int c = 0; c < num_comp; c++) {
        PyObject *node_set = PySet_New(NULL);
        for (int j = 0; j < sizes[c]; j++) {
            PyObject *val = PyLong_FromLong(buckets[c][j]);
            PySet_Add(node_set, val);
            Py_DECREF(val);
        }
        PyObject *tup = PyTuple_Pack(2, node_set, branch_sets[c]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, c, tup);
    }

cleanup2:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    return result;
}

/* ── connected_components_remapped(node_ids, edges) -> list[set[NodeId]] ──
 *
 * Like connected_components but takes the node_ids remapping list and returns
 * sets containing the original node IDs (node_ids[index]) instead of indices.
 * This moves the remapping loop from Python into C.
 */
static PyObject *py_connected_components_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    PyObject *nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!nid_fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(nid_fast);
    PyObject **nid_items = PySequence_Fast_ITEMS(nid_fast);

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(nid_fast); return NULL; }

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        free_edges(&el);
        Py_DECREF(nid_fast);
        return NULL;
    }
    free_edges(&el);

    int num_comp = cr.num_comp;

    /* Bucket nodes by component */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build Python list of sets with remapped node IDs */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup_r;

    for (int c = 0; c < num_comp; c++) {
        PyObject *s = PySet_New(NULL);
        if (!s) { Py_DECREF(result); result = NULL; goto cleanup_r; }
        for (int j = 0; j < sizes[c]; j++) {
            /* Use the original node ID from the remapping array */
            PySet_Add(s, nid_items[buckets[c][j]]);
        }
        PyList_SET_ITEM(result, c, s);
    }

cleanup_r:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) free(buckets[c]);
    free(buckets);
    Py_DECREF(nid_fast);
    return result;
}

/* ── connected_components_with_branches_remapped(node_ids, edges, branch_ids)
 *    -> list[tuple[set[NodeId], set[BranchId]]] ──
 *
 * Like connected_components_with_branches but remaps node indices to original
 * node IDs using the node_ids array.
 */
static PyObject *py_connected_components_with_branches_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "OOO", &node_ids_obj, &edges_obj, &branch_ids_obj)) return NULL;

    PyObject *nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!nid_fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(nid_fast);
    PyObject **nid_items = PySequence_Fast_ITEMS(nid_fast);

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(nid_fast); return NULL; }

    PyObject *br_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!br_fast) { free_edges(&el); Py_DECREF(nid_fast); return NULL; }
    Py_ssize_t br_len = PySequence_Fast_GET_SIZE(br_fast);

    ComponentResult cr;
    if (compute_components(n, &el, &cr) < 0) {
        Py_DECREF(br_fast);
        free_edges(&el);
        Py_DECREF(nid_fast);
        return NULL;
    }

    int num_comp = cr.num_comp;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[cr.labels[i]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = cr.labels[i];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = el.m < br_len ? el.m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);
    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        int c = cr.labels[EDGE_SRC(&el, i)];
        PySet_Add(branch_sets[c], br_items[i]);
    }
    Py_DECREF(br_fast);
    free_edges(&el);

    /* Build result list with remapped node IDs */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup_rb;

    for (int c = 0; c < num_comp; c++) {
        PyObject *node_set = PySet_New(NULL);
        for (int j = 0; j < sizes[c]; j++)
            PySet_Add(node_set, nid_items[buckets[c][j]]);
        PyObject *tup = PyTuple_Pack(2, node_set, branch_sets[c]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, c, tup);
    }

cleanup_rb:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    Py_DECREF(nid_fast);
    return result;
}

/* ── Adjacency list (CSR format) for graph traversal ── */

typedef struct {
    int *offset;  /* size n+1, CSR row pointers */
    int *adj;     /* size 2*m, neighbor node indices */
    int *eid;     /* size 2*m, original edge index */
} AdjList;

static int build_adj(int n, EdgeList *el, AdjList *al) {
    Py_ssize_t m = el->m;
    al->offset = (int *)calloc((size_t)(n + 1), sizeof(int));
    al->adj = m > 0 ? (int *)malloc((size_t)(2 * m) * sizeof(int)) : NULL;
    al->eid = m > 0 ? (int *)malloc((size_t)(2 * m) * sizeof(int)) : NULL;
    if (!al->offset || (m > 0 && (!al->adj || !al->eid))) {
        free(al->offset); free(al->adj); free(al->eid);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        al->offset[EDGE_SRC(el, i) + 1]++;
        al->offset[EDGE_DST(el, i) + 1]++;
    }
    for (int i = 1; i <= n; i++) al->offset[i] += al->offset[i - 1];
    if (m > 0) {
        int *pos = (int *)malloc((size_t)n * sizeof(int));
        if (!pos) {
            free(al->offset); free(al->adj); free(al->eid);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(pos, al->offset, (size_t)n * sizeof(int));
        for (Py_ssize_t i = 0; i < m; i++) {
            int u = EDGE_SRC(el, i), v = EDGE_DST(el, i);
            al->adj[pos[u]] = v; al->eid[pos[u]++] = (int)i;
            al->adj[pos[v]] = u; al->eid[pos[v]++] = (int)i;
        }
        free(pos);
    }
    return 0;
}

static void free_adj(AdjList *al) {
    free(al->offset);
    free(al->adj);
    free(al->eid);
}

/* ── Weight parsing: list of floats or numpy float64 array ── */

typedef struct {
    double *w;
    Py_ssize_t n;
    int owns_memory;
} WeightList;

static int parse_weights(PyObject *obj, WeightList *wl) {
    Py_buffer buf;
    if (PyObject_GetBuffer(obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) == 0) {
        if (buf.format && strcmp(buf.format, "d") == 0 &&
            buf.ndim == 1 && buf.itemsize == 8) {
            wl->w = (double *)buf.buf;
            wl->n = buf.shape[0];
            wl->owns_memory = 0;
            PyBuffer_Release(&buf);
            return 0;
        }
        PyBuffer_Release(&buf);
    } else {
        PyErr_Clear();
    }
    PyObject *fast = PySequence_Fast(obj, "weights must be a sequence");
    if (!fast) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    wl->n = n;
    wl->owns_memory = 1;
    if (n == 0) { wl->w = NULL; Py_DECREF(fast); return 0; }
    wl->w = (double *)malloc((size_t)n * sizeof(double));
    if (!wl->w) { Py_DECREF(fast); PyErr_NoMemory(); return -1; }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; i++) {
        wl->w[i] = PyFloat_AsDouble(items[i]);
        if (PyErr_Occurred()) { free(wl->w); Py_DECREF(fast); return -1; }
    }
    Py_DECREF(fast);
    return 0;
}

static void free_weights(WeightList *wl) {
    if (wl->owns_memory) free(wl->w);
}

/* ── Bridges (iterative Tarjan's) ── */

static PyObject *py_bridges(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }
    if (el.m == 0) { free_edges(&el); return result; }

    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) {
        free_adj(&al); free_edges(&el); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, (size_t)n * sizeof(int));

    int timer = 0;
    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i];
                int eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else {
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        int eid = peid[u];
                        PyObject *tup = Py_BuildValue(
                            "(ii)", EDGE_SRC(&el, eid), EDGE_DST(&el, eid));
                        if (!tup) {
                            free(buf5);
                            free_adj(&al); free_edges(&el);
                            Py_DECREF(result);
                            return NULL;
                        }
                        PyList_Append(result, tup);
                        Py_DECREF(tup);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5);
    free_adj(&al); free_edges(&el);
    return result;
}

/* ── Articulation points (iterative Tarjan's) ── */

static PyObject *py_articulation_points(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }

    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }
    free_edges(&el);

    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) {
        free_adj(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf6, *low = buf6 + n, *stk = buf6 + 2*n;
    int *sidx = buf6 + 3*n, *peid = buf6 + 4*n, *children = buf6 + 5*n;
    memset(disc, -1, (size_t)n * sizeof(int));
    memset(children, 0, (size_t)n * sizeof(int));

    int timer = 0;
    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;
        children[start] = 0;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i];
                int eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    children[v] = 0;
                    children[u]++;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else {
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    /* Non-root AP: low[child] >= disc[parent] */
                    if (peid[p] != -1 && low[u] >= disc[p]) {
                        PyObject *val = PyLong_FromLong(p);
                        PySet_Add(result, val);
                        Py_DECREF(val);
                    }
                }
                sp--;
            }
        }

        /* Root AP: 2+ DFS tree children */
        if (children[start] >= 2) {
            PyObject *val = PyLong_FromLong(start);
            PySet_Add(result, val);
            Py_DECREF(val);
        }
    }

    free(buf6);
    free_adj(&al);
    return result;
}

/* ── Biconnected components (iterative Tarjan's with edge stack) ── */

static PyObject *py_biconnected_components(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (n == 0) return result;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result); return NULL; }
    if (el.m == 0) { free_edges(&el); return result; }

    Py_ssize_t m = el.m;
    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); Py_DECREF(result); return NULL;
    }
    free_edges(&el);

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *esbuf = (int *)malloc(2 * (size_t)m * sizeof(int));
    if (!buf5 || !esbuf) {
        free(buf5); free(esbuf);
        free_adj(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    int *esu = esbuf, *esv = esbuf + m;
    memset(disc, -1, (size_t)n * sizeof(int));

    int timer = 0;
    int esp = 0;  /* edge stack pointer */

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start;
        sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++;
        peid[start] = -1;

        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i];
                int eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    /* Tree edge */
                    esu[esp] = u; esv[esp] = v; esp++;
                    disc[v] = low[v] = timer++;
                    peid[v] = eid;
                    sp++;
                    stk[sp] = v;
                    sidx[sp] = al.offset[v];
                } else if (disc[v] < disc[u]) {
                    /* Back edge to ancestor */
                    esu[esp] = u; esv[esp] = v; esp++;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        /* Pop edges to form biconnected component */
                        PyObject *comp = PySet_New(NULL);
                        if (!comp) {
                            free(buf5); free(esbuf);
                            free_adj(&al); Py_DECREF(result);
                            return NULL;
                        }
                        while (esp > 0) {
                            esp--;
                            PyObject *pu = PyLong_FromLong(esu[esp]);
                            PyObject *pv = PyLong_FromLong(esv[esp]);
                            PySet_Add(comp, pu);
                            PySet_Add(comp, pv);
                            Py_DECREF(pu);
                            Py_DECREF(pv);
                            if (esu[esp] == p && esv[esp] == u) break;
                        }
                        PyList_Append(result, comp);
                        Py_DECREF(comp);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5); free(esbuf);
    free_adj(&al);
    return result;
}

/* ── BFS (unweighted, from source) ── */

static PyObject *py_bfs(PyObject *self, PyObject *args) {
    int n, source;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iOi", &n, &edges_obj, &source)) return NULL;

    if (source < 0 || source >= n) {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return NULL;
    }

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    AdjList al;
    if (build_adj(n, &el, &al) < 0) { free_edges(&el); return NULL; }
    free_edges(&el);

    int *visited = (int *)calloc((size_t)n, sizeof(int));
    int *queue   = (int *)malloc((size_t)n * sizeof(int));
    if (!visited || !queue) {
        free(visited); free(queue);
        free_adj(&al);
        PyErr_NoMemory();
        return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source;
    visited[source] = 1;

    while (head < tail) {
        int u = queue[head++];
        for (int i = al.offset[u]; i < al.offset[u + 1]; i++) {
            int v = al.adj[i];
            if (!visited[v]) {
                visited[v] = 1;
                queue[tail++] = v;
            }
        }
    }

    /* Build result list from queue (BFS order) */
    PyObject *result = PyList_New(tail);
    if (!result) {
        free(visited); free(queue); free_adj(&al);
        return NULL;
    }
    for (int i = 0; i < tail; i++)
        PyList_SET_ITEM(result, i, PyLong_FromLong(queue[i]));

    free(visited); free(queue);
    free_adj(&al);
    return result;
}

/* ── Binary min-heap for Dijkstra ── */

typedef struct {
    double *dist;
    int *heap;
    int *pos;
    int size;
} MinHeap;

static inline void mh_swap(MinHeap *h, int i, int j) {
    int ni = h->heap[i], nj = h->heap[j];
    h->heap[i] = nj; h->heap[j] = ni;
    h->pos[ni] = j; h->pos[nj] = i;
}

/* 4-ary heap: shallower tree = fewer cache misses in sift_down */
#define MH_ARITY 4

static inline void mh_sift_up(MinHeap *h, int i) {
    while (i > 0) {
        int p = (i - 1) / MH_ARITY;
        if (h->dist[h->heap[i]] < h->dist[h->heap[p]]) {
            mh_swap(h, i, p);
            i = p;
        } else break;
    }
}

static inline void mh_sift_down(MinHeap *h, int i) {
    while (1) {
        int first = MH_ARITY * i + 1;
        if (first >= h->size) break;
        int last = first + MH_ARITY;
        if (last > h->size) last = h->size;
        int s = i;
        double sd = h->dist[h->heap[s]];
        for (int c = first; c < last; c++) {
            double cd = h->dist[h->heap[c]];
            if (cd < sd) { s = c; sd = cd; }
        }
        if (s != i) { mh_swap(h, i, s); i = s; }
        else break;
    }
}

static inline int mh_pop(MinHeap *h) {
    int node = h->heap[0];
    h->pos[node] = -1;
    h->size--;
    if (h->size > 0) {
        h->heap[0] = h->heap[h->size];
        h->pos[h->heap[0]] = 0;
        mh_sift_down(h, 0);
    }
    return node;
}

/* ── Dijkstra (source -> target, returns distance + path) ── */

static PyObject *py_dijkstra(PyObject *self, PyObject *args) {
    int n, source, target;
    PyObject *edges_obj, *weights_obj;
    if (!PyArg_ParseTuple(args, "iOOii", &n, &edges_obj, &weights_obj,
                          &source, &target))
        return NULL;

    if (n == 0 || source < 0 || source >= n || target < 0 || target >= n) {
        PyObject *path = PyList_New(0);
        if (!path) return NULL;
        return Py_BuildValue("(dN)", HUGE_VAL, path);
    }
    if (source == target) {
        PyObject *path = PyList_New(1);
        if (!path) return NULL;
        PyList_SET_ITEM(path, 0, PyLong_FromLong(source));
        return Py_BuildValue("(dN)", 0.0, path);
    }

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;
    WeightList wl;
    if (parse_weights(weights_obj, &wl) < 0) { free_edges(&el); return NULL; }

    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); free_weights(&wl); return NULL;
    }
    free_edges(&el);

    double *dist = (double *)malloc((size_t)n * sizeof(double));
    int *prev    = (int *)malloc((size_t)n * sizeof(int));
    int *heap    = (int *)malloc((size_t)n * sizeof(int));
    int *pos     = (int *)malloc((size_t)n * sizeof(int));

    if (!dist || !prev || !heap || !pos) {
        free(dist); free(prev); free(heap); free(pos);
        free_adj(&al); free_weights(&wl);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        dist[i] = HUGE_VAL;
        prev[i] = -1;
        pos[i] = -1;
    }

    MinHeap mh = {dist, heap, pos, 0};
    dist[source] = 0.0;
    heap[0] = source;
    pos[source] = 0;
    mh.size = 1;

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (u == target) break;
        if (dist[u] == HUGE_VAL) break;

        for (int i = al.offset[u]; i < al.offset[u + 1]; i++) {
            int v = al.adj[i];
            double w = wl.w[al.eid[i]];
            double nd = dist[u] + w;
            if (nd < dist[v]) {
                prev[v] = u;
                if (pos[v] == -1) {
                    dist[v] = nd;
                    int p = mh.size++;
                    heap[p] = v;
                    pos[v] = p;
                    mh_sift_up(&mh, p);
                } else {
                    dist[v] = nd;
                    mh_sift_up(&mh, pos[v]);
                }
            }
        }
    }

    /* Build result */
    double final_dist = dist[target];
    PyObject *path_list;

    if (final_dist < HUGE_VAL) {
        int plen = 0;
        for (int v = target; v != -1; v = prev[v]) plen++;
        path_list = PyList_New(plen);
        if (!path_list) {
            free(dist); free(prev); free(heap); free(pos);
            free_adj(&al); free_weights(&wl);
            return NULL;
        }
        int v = target;
        for (int i = plen - 1; i >= 0; i--) {
            PyList_SET_ITEM(path_list, i, PyLong_FromLong(v));
            v = prev[v];
        }
    } else {
        path_list = PyList_New(0);
        if (!path_list) {
            free(dist); free(prev); free(heap); free(pos);
            free_adj(&al); free_weights(&wl);
            return NULL;
        }
    }

    PyObject *res = Py_BuildValue("(dN)", final_dist, path_list);
    free(dist); free(prev); free(heap); free(pos);
    free_adj(&al); free_weights(&wl);
    return res;
}

/* ── SSSP lengths (single source, all reachable, optional cutoff) ── */

static PyObject *py_sssp_lengths(PyObject *self, PyObject *args) {
    int n, source;
    double cutoff;
    PyObject *edges_obj, *weights_obj;
    if (!PyArg_ParseTuple(args, "iOOid", &n, &edges_obj, &weights_obj,
                          &source, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);

    PyObject *result_dict = PyDict_New();
    if (!result_dict) return NULL;
    if (n == 0 || source < 0 || source >= n) return result_dict;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) { Py_DECREF(result_dict); return NULL; }
    WeightList wl;
    if (parse_weights(weights_obj, &wl) < 0) {
        free_edges(&el); Py_DECREF(result_dict); return NULL;
    }

    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); free_weights(&wl);
        Py_DECREF(result_dict); return NULL;
    }
    free_edges(&el);

    double *dist = (double *)malloc((size_t)n * sizeof(double));
    int *heap    = (int *)malloc((size_t)n * sizeof(int));
    int *pos     = (int *)malloc((size_t)n * sizeof(int));

    if (!dist || !heap || !pos) {
        free(dist); free(heap); free(pos);
        free_adj(&al); free_weights(&wl); Py_DECREF(result_dict);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        dist[i] = HUGE_VAL;
        pos[i] = -1;
    }

    MinHeap mh = {dist, heap, pos, 0};
    dist[source] = 0.0;
    heap[0] = source;
    pos[source] = 0;
    mh.size = 1;

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u] == HUGE_VAL) break;
        if (use_cutoff && dist[u] > cutoff) break;

        for (int i = al.offset[u]; i < al.offset[u + 1]; i++) {
            int v = al.adj[i];
            double w = wl.w[al.eid[i]];
            double nd = dist[u] + w;
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v] == -1) {
                    dist[v] = nd;
                    int p = mh.size++;
                    heap[p] = v;
                    pos[v] = p;
                    mh_sift_up(&mh, p);
                } else {
                    dist[v] = nd;
                    mh_sift_up(&mh, pos[v]);
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *key = PyLong_FromLong(i);
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(result_dict, key, val);
            Py_DECREF(key); Py_DECREF(val);
        }
    }

    free(dist); free(heap); free(pos);
    free_adj(&al); free_weights(&wl);
    return result_dict;
}

/* ── Multi-source Dijkstra (multiple start nodes, optional cutoff) ── */

static PyObject *py_multi_source_dijkstra(PyObject *self, PyObject *args) {
    int n;
    double cutoff;
    PyObject *edges_obj, *weights_obj, *sources_obj;
    if (!PyArg_ParseTuple(args, "iOOOd", &n, &edges_obj, &weights_obj,
                          &sources_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);

    PyObject *result_dict = PyDict_New();
    if (!result_dict) return NULL;
    if (n == 0) return result_dict;

    PyObject *src_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!src_fast) { Py_DECREF(result_dict); return NULL; }
    Py_ssize_t nsrc = PySequence_Fast_GET_SIZE(src_fast);

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) {
        Py_DECREF(src_fast); Py_DECREF(result_dict); return NULL;
    }
    WeightList wl;
    if (parse_weights(weights_obj, &wl) < 0) {
        free_edges(&el); Py_DECREF(src_fast); Py_DECREF(result_dict);
        return NULL;
    }

    AdjList al;
    if (build_adj(n, &el, &al) < 0) {
        free_edges(&el); free_weights(&wl);
        Py_DECREF(src_fast); Py_DECREF(result_dict);
        return NULL;
    }
    free_edges(&el);

    double *dist = (double *)malloc((size_t)n * sizeof(double));
    int *heap    = (int *)malloc((size_t)n * sizeof(int));
    int *pos     = (int *)malloc((size_t)n * sizeof(int));

    if (!dist || !heap || !pos) {
        free(dist); free(heap); free(pos);
        free_adj(&al); free_weights(&wl);
        Py_DECREF(src_fast); Py_DECREF(result_dict);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        dist[i] = HUGE_VAL;
        pos[i] = -1;
    }

    MinHeap mh = {dist, heap, pos, 0};

    /* Initialize all source nodes at distance 0 */
    PyObject **src_items = PySequence_Fast_ITEMS(src_fast);
    for (Py_ssize_t i = 0; i < nsrc; i++) {
        int s = (int)PyLong_AsLong(src_items[i]);
        if (PyErr_Occurred()) {
            free(dist); free(heap); free(pos);
            free_adj(&al); free_weights(&wl);
            Py_DECREF(src_fast); Py_DECREF(result_dict);
            return NULL;
        }
        if (s >= 0 && s < n && dist[s] > 0.0) {
            dist[s] = 0.0;
            int p = mh.size++;
            heap[p] = s;
            pos[s] = p;
            mh_sift_up(&mh, p);
        }
    }
    Py_DECREF(src_fast);

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u] == HUGE_VAL) break;
        if (use_cutoff && dist[u] > cutoff) break;

        for (int i = al.offset[u]; i < al.offset[u + 1]; i++) {
            int v = al.adj[i];
            double w = wl.w[al.eid[i]];
            double nd = dist[u] + w;
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v] == -1) {
                    dist[v] = nd;
                    int p = mh.size++;
                    heap[p] = v;
                    pos[v] = p;
                    mh_sift_up(&mh, p);
                } else {
                    dist[v] = nd;
                    mh_sift_up(&mh, pos[v]);
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *key = PyLong_FromLong(i);
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(result_dict, key, val);
            Py_DECREF(key); Py_DECREF(val);
        }
    }

    free(dist); free(heap); free(pos);
    free_adj(&al); free_weights(&wl);
    return result_dict;
}

/* ── translate_edges(node_ids, nid_edges) -> index-based edges ── */

static PyObject *py_translate_edges(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    PyObject *nid_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!nid_fast) return NULL;
    int n = (int)PySequence_Fast_GET_SIZE(nid_fast);
    PyObject **nid_items = PySequence_Fast_ITEMS(nid_fast);

    /* Build hash map: node_id -> index */
    IntMap im;
    if (intmap_init(&im, n) < 0) { Py_DECREF(nid_fast); return NULL; }
    for (int i = 0; i < n; i++) {
        long nid = PyLong_AsLong(nid_items[i]);
        if (nid == -1 && PyErr_Occurred()) {
            intmap_free(&im); Py_DECREF(nid_fast); return NULL;
        }
        intmap_put(&im, nid, i);
    }
    Py_DECREF(nid_fast);

    /* Parse nid_edges and translate */
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence");
    if (!fast) { intmap_free(&im); return NULL; }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);

    PyObject *result = PyList_New(m);
    if (!result) { intmap_free(&im); Py_DECREF(fast); return NULL; }

    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *edge = items[i];
        PyObject *edge_fast = PySequence_Fast(edge, "each edge must be a 2-tuple");
        if (!edge_fast || PySequence_Fast_GET_SIZE(edge_fast) != 2) {
            Py_XDECREF(edge_fast);
            intmap_free(&im); Py_DECREF(fast); Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "each edge must be a 2-tuple");
            return NULL;
        }
        PyObject **ep = PySequence_Fast_ITEMS(edge_fast);
        long u_nid = PyLong_AsLong(ep[0]);
        long v_nid = PyLong_AsLong(ep[1]);
        Py_DECREF(edge_fast);
        if (PyErr_Occurred()) {
            intmap_free(&im); Py_DECREF(fast); Py_DECREF(result);
            return NULL;
        }
        int u_idx = intmap_get(&im, u_nid);
        int v_idx = intmap_get(&im, v_nid);
        if (u_idx < 0 || v_idx < 0) {
            intmap_free(&im); Py_DECREF(fast); Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "edge references unknown node ID");
            return NULL;
        }
        PyObject *tup = PyTuple_New(2);
        PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(u_idx));
        PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(v_idx));
        PyList_SET_ITEM(result, i, tup);
    }

    intmap_free(&im);
    Py_DECREF(fast);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Node-ID API: edges use original node IDs, C does hash map + algo + remap.
 * Zero Python object creation for intermediate index-based edges.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── connected_components (node-ID edges) ── */

static PyObject *py_cc_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;
    if (n == 0) { nid_free(&ctx); return PyList_New(0); }

    ComponentResult cr;
    if (compute_components(n, &ctx.el, &cr) < 0) { nid_free(&ctx); return NULL; }

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); nid_free(&ctx); return NULL; }

    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); nid_free(&ctx); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result); nid_free(&ctx);
            return NULL;
        }
    }

    for (int i = 0; i < n; i++)
        PySet_Add(sets[cr.labels[i]], ctx.nid_items[i]);

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    nid_free(&ctx);
    return result;
}

/* ── connected_components (node-ID, split src/dst) ── */

static PyObject *py_cc_nid_split(PyObject *self, PyObject *args) {
    PyObject *nids, *src, *dst;
    if (!PyArg_ParseTuple(args, "OOO", &nids, &src, &dst)) return NULL;

    NidContext ctx;
    if (nid_parse_split(nids, src, dst, &ctx) < 0) return NULL;
    int n = ctx.n;
    if (n == 0) { nid_free(&ctx); return PyList_New(0); }

    ComponentResult cr;
    if (compute_components(n, &ctx.el, &cr) < 0) { nid_free(&ctx); return NULL; }

    int nc = cr.num_comp;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(nc);
    if (!result) { free(cr.labels); nid_free(&ctx); return NULL; }

    /* Create all sets upfront */
    PyObject **sets = (PyObject **)malloc((size_t)nc * sizeof(PyObject *));
    if (!sets) { free(cr.labels); Py_DECREF(result); nid_free(&ctx); PyErr_NoMemory(); return NULL; }
    for (int c = 0; c < nc; c++) {
        sets[c] = PySet_New(NULL);
        if (!sets[c]) {
            for (int j = 0; j < c; j++) Py_DECREF(sets[j]);
            free(sets); free(cr.labels); Py_DECREF(result); nid_free(&ctx);
            return NULL;
        }
    }

    /* Single pass: add each node to its component set */
    for (int i = 0; i < n; i++)
        PySet_Add(sets[cr.labels[i]], ctx.nid_items[i]);

    for (int c = 0; c < nc; c++)
        PyList_SET_ITEM(result, c, sets[c]);

    free(sets);
    free(cr.labels);
    nid_free(&ctx);
    return result;
}

/* ── bridges (node-ID edges) ── */

static PyObject *py_bridges_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || ctx.el.m == 0) { nid_free(&ctx); return result; }

    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    if (!buf5) {
        free_adj(&al); nid_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i], eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] > disc[p]) {
                        int ei = peid[u];
                        int su = EDGE_SRC(&ctx.el, ei), sv = EDGE_DST(&ctx.el, ei);
                        PyObject *t = PyTuple_New(2);
                        Py_INCREF(ctx.nid_items[su]); Py_INCREF(ctx.nid_items[sv]);
                        PyTuple_SET_ITEM(t, 0, ctx.nid_items[su]);
                        PyTuple_SET_ITEM(t, 1, ctx.nid_items[sv]);
                        PyList_Append(result, t); Py_DECREF(t);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5);
    free_adj(&al); nid_free(&ctx);
    return result;
}

/* ── articulation_points (node-ID edges) ── */

static PyObject *py_ap_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    PyObject *result = PySet_New(NULL);
    if (!result || n == 0) { nid_free(&ctx); return result; }

    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf6 = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf6) {
        free_adj(&al); nid_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf6, *low = buf6 + n, *stk = buf6 + 2*n;
    int *sidx = buf6 + 3*n, *peid = buf6 + 4*n, *ch = buf6 + 5*n;
    memset(disc, -1, n * sizeof(int));
    memset(ch, 0, n * sizeof(int));
    int timer = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1; ch[start] = 0;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i], eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++; peid[v] = eid; ch[v] = 0; ch[u]++;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else { if (low[u] > disc[v]) low[u] = disc[v]; }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (peid[p] != -1 && low[u] >= disc[p])
                        PySet_Add(result, ctx.nid_items[p]);
                }
                sp--;
            }
        }
        if (ch[start] >= 2)
            PySet_Add(result, ctx.nid_items[start]);
    }

    free(buf6);
    free_adj(&al); nid_free(&ctx);
    return result;
}

/* ── biconnected_components (node-ID edges) ── */

static PyObject *py_bcc_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nids, &edges)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    PyObject *result = PyList_New(0);
    if (!result || n == 0 || ctx.el.m == 0) { nid_free(&ctx); return result; }

    Py_ssize_t m = ctx.el.m;
    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); Py_DECREF(result); return NULL; }

    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *esbuf = (int *)malloc(2 * (size_t)m * sizeof(int));
    if (!buf5 || !esbuf) {
        free(buf5); free(esbuf);
        free_adj(&al); nid_free(&ctx); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    int *esu = esbuf, *esv = esbuf + m;
    memset(disc, -1, n * sizeof(int));
    int timer = 0, esp = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al.offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al.offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al.adj[i], eid = al.eid[i];
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al.offset[v];
                } else if (disc[v] < disc[u]) {
                    esu[esp] = u; esv[esp] = v; esp++;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        PyObject *comp = PySet_New(NULL);
                        while (esp > 0) {
                            esp--;
                            PySet_Add(comp, ctx.nid_items[esu[esp]]);
                            PySet_Add(comp, ctx.nid_items[esv[esp]]);
                            if (esu[esp] == p && esv[esp] == u) break;
                        }
                        PyList_Append(result, comp); Py_DECREF(comp);
                    }
                }
                sp--;
            }
        }
    }

    free(buf5); free(esbuf);
    free_adj(&al); nid_free(&ctx);
    return result;
}

/* ── bfs (node-ID edges) ── */

static PyObject *py_bfs_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges, *src_obj;
    if (!PyArg_ParseTuple(args, "OOO", &nids, &edges, &src_obj)) return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    long src_nid = PyLong_AsLong(src_obj);
    if (src_nid == -1 && PyErr_Occurred()) { nid_free(&ctx); return NULL; }
    int source = intmap_get(&ctx.im, src_nid);
    if (source < 0) {
        nid_free(&ctx);
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); return NULL; }

    int *vis = calloc(n, sizeof(int)), *queue = malloc(n * sizeof(int));
    if (!vis || !queue) {
        free(vis); free(queue); free_adj(&al); nid_free(&ctx);
        PyErr_NoMemory(); return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source; vis[source] = 1;
    while (head < tail) {
        int u = queue[head++];
        for (int i = al.offset[u]; i < al.offset[u+1]; i++) {
            int v = al.adj[i];
            if (!vis[v]) { vis[v] = 1; queue[tail++] = v; }
        }
    }

    PyObject *result = PyList_New(tail);
    if (result) {
        for (int i = 0; i < tail; i++) {
            Py_INCREF(ctx.nid_items[queue[i]]);
            PyList_SET_ITEM(result, i, ctx.nid_items[queue[i]]);
        }
    }

    free(vis); free(queue); free_adj(&al); nid_free(&ctx);
    return result;
}

/* ── dijkstra (node-ID edges) ── */

static PyObject *py_dijkstra_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges, *wobj, *src_obj, *tgt_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &nids, &edges, &wobj, &src_obj, &tgt_obj))
        return NULL;

    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    long sn = PyLong_AsLong(src_obj), tn = PyLong_AsLong(tgt_obj);
    if (PyErr_Occurred()) { nid_free(&ctx); return NULL; }
    int source = n > 0 ? intmap_get(&ctx.im, sn) : -1;
    int target = n > 0 ? intmap_get(&ctx.im, tn) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(ctx.nid_items[source]);
        PyList_SET_ITEM(path, 0, ctx.nid_items[source]);
        nid_free(&ctx);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (n == 0 || source < 0 || target < 0) {
        nid_free(&ctx);
        PyObject *p = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, p);
    }

    WeightList wl;
    if (parse_weights(wobj, &wl) < 0) { nid_free(&ctx); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); free_weights(&wl); return NULL; }

    double *dist = malloc(n*sizeof(double));
    int *prev = malloc(n*sizeof(int)), *heap = malloc(n*sizeof(int)), *pos = malloc(n*sizeof(int));
    if (!dist||!prev||!heap||!pos) {
        free(dist);free(prev);free(heap);free(pos);
        free_adj(&al); free_weights(&wl); nid_free(&ctx);
        PyErr_NoMemory(); return NULL;
    }
    for (int i=0;i<n;i++) { dist[i]=HUGE_VAL; prev[i]=-1; pos[i]=-1; }
    MinHeap mh = {dist, heap, pos, 0};
    dist[source]=0.0; heap[0]=source; pos[source]=0; mh.size=1;

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (u == target || dist[u] == HUGE_VAL) break;
        for (int i=al.offset[u]; i<al.offset[u+1]; i++) {
            int v=al.adj[i]; double nd=dist[u]+wl.w[al.eid[i]];
            if (nd < dist[v]) {
                prev[v] = u;
                if (pos[v]==-1) { dist[v]=nd; int p=mh.size++; heap[p]=v; pos[v]=p; mh_sift_up(&mh,p); }
                else { dist[v]=nd; mh_sift_up(&mh, pos[v]); }
            }
        }
    }

    double fd = dist[target];
    PyObject *path;
    if (fd < HUGE_VAL) {
        int plen=0; for (int v=target;v!=-1;v=prev[v]) plen++;
        path = PyList_New(plen);
        int v=target;
        for (int i=plen-1;i>=0;i--) {
            Py_INCREF(ctx.nid_items[v]);
            PyList_SET_ITEM(path, i, ctx.nid_items[v]);
            v = prev[v];
        }
    } else { path = PyList_New(0); }

    PyObject *res = Py_BuildValue("(dN)", fd, path);
    free(dist);free(prev);free(heap);free(pos);
    free_adj(&al); free_weights(&wl); nid_free(&ctx);
    return res;
}

/* ── sssp_lengths (node-ID edges) ── */

static PyObject *py_sssp_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges, *wobj, *src_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &nids, &edges, &wobj, &src_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);
    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    PyObject *rd = PyDict_New();
    if (!rd) { nid_free(&ctx); return NULL; }

    long sn = PyLong_AsLong(src_obj);
    if (PyErr_Occurred()) { nid_free(&ctx); Py_DECREF(rd); return NULL; }
    int source = n > 0 ? intmap_get(&ctx.im, sn) : -1;
    if (n == 0 || source < 0) { nid_free(&ctx); return rd; }

    WeightList wl;
    if (parse_weights(wobj, &wl) < 0) { nid_free(&ctx); Py_DECREF(rd); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { nid_free(&ctx); free_weights(&wl); Py_DECREF(rd); return NULL; }

    double *dist = malloc(n*sizeof(double));
    int *heap = malloc(n*sizeof(int)), *pos = malloc(n*sizeof(int));
    if (!dist||!heap||!pos) {
        free(dist);free(heap);free(pos);
        free_adj(&al); free_weights(&wl); nid_free(&ctx); Py_DECREF(rd);
        PyErr_NoMemory(); return NULL;
    }
    for (int i=0;i<n;i++) { dist[i]=HUGE_VAL; pos[i]=-1; }
    MinHeap mh = {dist, heap, pos, 0};
    dist[source]=0.0; heap[0]=source; pos[source]=0; mh.size=1;

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u]==HUGE_VAL) break;
        if (use_cutoff && dist[u]>cutoff) break;
        for (int i=al.offset[u]; i<al.offset[u+1]; i++) {
            int v=al.adj[i]; double nd=dist[u]+wl.w[al.eid[i]];
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v]==-1) { dist[v]=nd; int p=mh.size++; heap[p]=v; pos[v]=p; mh_sift_up(&mh,p); }
                else { dist[v]=nd; mh_sift_up(&mh, pos[v]); }
            }
        }
    }

    for (int i=0;i<n;i++) {
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(rd, ctx.nid_items[i], val);
            Py_DECREF(val);
        }
    }

    free(dist);free(heap);free(pos);
    free_adj(&al); free_weights(&wl); nid_free(&ctx);
    return rd;
}

/* ── multi_source_dijkstra (node-ID edges) ── */

static PyObject *py_msdijk_nid(PyObject *self, PyObject *args) {
    PyObject *nids, *edges, *wobj, *srcs_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &nids, &edges, &wobj, &srcs_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);
    NidContext ctx;
    if (nid_parse(nids, edges, &ctx) < 0) return NULL;
    int n = ctx.n;

    PyObject *rd = PyDict_New();
    if (!rd) { nid_free(&ctx); return NULL; }
    if (n == 0) { nid_free(&ctx); return rd; }

    PyObject *sf = PySequence_Fast(srcs_obj, "sources must be a sequence");
    if (!sf) { nid_free(&ctx); Py_DECREF(rd); return NULL; }

    WeightList wl;
    if (parse_weights(wobj, &wl) < 0) { Py_DECREF(sf); nid_free(&ctx); Py_DECREF(rd); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al) < 0) { Py_DECREF(sf); free_weights(&wl); nid_free(&ctx); Py_DECREF(rd); return NULL; }

    double *dist = malloc(n*sizeof(double));
    int *heap = malloc(n*sizeof(int)), *pos = malloc(n*sizeof(int));
    if (!dist||!heap||!pos) {
        free(dist);free(heap);free(pos);
        free_adj(&al); free_weights(&wl); Py_DECREF(sf); nid_free(&ctx); Py_DECREF(rd);
        PyErr_NoMemory(); return NULL;
    }
    for (int i=0;i<n;i++) { dist[i]=HUGE_VAL; pos[i]=-1; }
    MinHeap mh = {dist, heap, pos, 0};

    Py_ssize_t nsrc = PySequence_Fast_GET_SIZE(sf);
    PyObject **si = PySequence_Fast_ITEMS(sf);
    for (Py_ssize_t i=0;i<nsrc;i++) {
        long sn = PyLong_AsLong(si[i]);
        if (PyErr_Occurred()) {
            free(dist);free(heap);free(pos);
            free_adj(&al); free_weights(&wl); Py_DECREF(sf); nid_free(&ctx); Py_DECREF(rd);
            return NULL;
        }
        int s = intmap_get(&ctx.im, sn);
        if (s >= 0 && dist[s] > 0.0) {
            dist[s]=0.0; int p=mh.size++; heap[p]=s; pos[s]=p; mh_sift_up(&mh,p);
        }
    }
    Py_DECREF(sf);

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u]==HUGE_VAL) break;
        if (use_cutoff && dist[u]>cutoff) break;
        for (int i=al.offset[u]; i<al.offset[u+1]; i++) {
            int v=al.adj[i]; double nd=dist[u]+wl.w[al.eid[i]];
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v]==-1) { dist[v]=nd; int p=mh.size++; heap[p]=v; pos[v]=p; mh_sift_up(&mh,p); }
                else { dist[v]=nd; mh_sift_up(&mh, pos[v]); }
            }
        }
    }

    for (int i=0;i<n;i++) {
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(rd, ctx.nid_items[i], val);
            Py_DECREF(val);
        }
    }

    free(dist);free(heap);free(pos);
    free_adj(&al); free_weights(&wl); nid_free(&ctx);
    return rd;
}

/* ── Module definition ── */

static PyMethodDef methods[] = {
    {"connected_components", py_connected_components, METH_VARARGS,
     "connected_components(n, edges) -> list[set[int]]\n\n"
     "Find connected components using union-find with path compression.\n"
     "edges can be a list of 2-tuples or a numpy int32 array of shape (m, 2)."},
    {"connected_components_with_branches", py_connected_components_with_branches, METH_VARARGS,
     "connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]]\n\n"
     "Find connected components with associated branch IDs.\n"
     "edges can be a list of 2-tuples or a numpy int32 array of shape (m, 2)."},
    {"connected_components_remapped", py_connected_components_remapped, METH_VARARGS,
     "connected_components_remapped(node_ids, edges) -> list[set[NodeId]]\n\n"
     "Like connected_components but remaps indices to original node IDs.\n"
     "node_ids[i] is the original ID for internal index i."},
    {"connected_components_with_branches_remapped", py_connected_components_with_branches_remapped, METH_VARARGS,
     "connected_components_with_branches_remapped(node_ids, edges, branch_ids) -> list[tuple[set[NodeId], set[BranchId]]]\n\n"
     "Like connected_components_with_branches but remaps indices to original node IDs."},
    {"bridges", py_bridges, METH_VARARGS,
     "bridges(n, edges) -> list[tuple[int, int]]\n\n"
     "Find bridge edges using iterative Tarjan's algorithm."},
    {"articulation_points", py_articulation_points, METH_VARARGS,
     "articulation_points(n, edges) -> set[int]\n\n"
     "Find articulation points using iterative Tarjan's algorithm."},
    {"biconnected_components", py_biconnected_components, METH_VARARGS,
     "biconnected_components(n, edges) -> list[set[int]]\n\n"
     "Find biconnected components as node sets."},
    {"bfs", py_bfs, METH_VARARGS,
     "bfs(n, edges, source) -> list[int]\n\n"
     "BFS traversal from source, returns visited nodes in BFS order."},
    {"dijkstra", py_dijkstra, METH_VARARGS,
     "dijkstra(n, edges, weights, source, target) -> (float, list[int])\n\n"
     "Shortest path from source to target via Dijkstra with binary heap.\n"
     "Returns (distance, path). If unreachable, returns (inf, [])."},
    {"sssp_lengths", py_sssp_lengths, METH_VARARGS,
     "sssp_lengths(n, edges, weights, source, cutoff) -> dict[int, float]\n\n"
     "Single-source shortest path lengths. cutoff < 0 means no cutoff."},
    {"multi_source_dijkstra", py_multi_source_dijkstra, METH_VARARGS,
     "multi_source_dijkstra(n, edges, weights, sources, cutoff) -> dict[int, float]\n\n"
     "Multi-source shortest path lengths. cutoff < 0 means no cutoff."},
    {"translate_edges", py_translate_edges, METH_VARARGS,
     "translate_edges(node_ids, nid_edges) -> list[tuple[int, int]]\n\n"
     "Translate edges from node-ID-based to index-based using a C hash map."},
    {"cc_nid", py_cc_nid, METH_VARARGS,
     "cc_nid(node_ids, nid_edges) -> list[set[NodeId]]\n\n"
     "Connected components with node-ID-based edges."},
    {"cc_nid_split", py_cc_nid_split, METH_VARARGS,
     "cc_nid_split(node_ids, src, dst) -> list[set[NodeId]]\n\n"
     "Connected components with separate src/dst sequences or 1D arrays."},
    {"bridges_nid", py_bridges_nid, METH_VARARGS,
     "bridges_nid(node_ids, nid_edges) -> list[tuple[NodeId, NodeId]]\n\n"
     "Bridges with node-ID-based edges."},
    {"ap_nid", py_ap_nid, METH_VARARGS,
     "ap_nid(node_ids, nid_edges) -> set[NodeId]\n\n"
     "Articulation points with node-ID-based edges."},
    {"bcc_nid", py_bcc_nid, METH_VARARGS,
     "bcc_nid(node_ids, nid_edges) -> list[set[NodeId]]\n\n"
     "Biconnected components with node-ID-based edges."},
    {"bfs_nid", py_bfs_nid, METH_VARARGS,
     "bfs_nid(node_ids, nid_edges, source_nid) -> list[NodeId]\n\n"
     "BFS with node-ID-based edges and source."},
    {"dijkstra_nid", py_dijkstra_nid, METH_VARARGS,
     "dijkstra_nid(node_ids, nid_edges, weights, source_nid, target_nid) -> (float, list)\n\n"
     "Dijkstra with node-ID-based edges."},
    {"sssp_nid", py_sssp_nid, METH_VARARGS,
     "sssp_nid(node_ids, nid_edges, weights, source_nid, cutoff) -> dict[NodeId, float]\n\n"
     "SSSP lengths with node-ID-based edges."},
    {"msdijk_nid", py_msdijk_nid, METH_VARARGS,
     "msdijk_nid(node_ids, nid_edges, weights, source_nids, cutoff) -> dict[NodeId, float]\n\n"
     "Multi-source Dijkstra with node-ID-based edges."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core",
    "Fast graph algorithms: union-find, Tarjan's, BFS, Dijkstra.", -1, methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
