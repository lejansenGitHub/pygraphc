#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Union-Find with path compression + union by rank ── */

typedef struct {
    int *parent;
    int *rank;
    int n;
} UF;

static inline void uf_init(UF *uf, int n) {
    uf->n = n;
    uf->parent = (int *)malloc(n * sizeof(int));
    uf->rank   = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) uf->parent[i] = i;
}

static inline void uf_free(UF *uf) {
    free(uf->parent);
    free(uf->rank);
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
    uf_init(&uf, n);
    for (Py_ssize_t i = 0; i < el->m; i++)
        uf_union(&uf, EDGE_SRC(el, i), EDGE_DST(el, i));

    cr->labels = (int *)malloc(n * sizeof(int));
    if (!cr->labels) {
        uf_free(&uf);
        PyErr_NoMemory();
        return -1;
    }

    /* Assign sequential component IDs using a root->id mapping */
    int *root_to_id = (int *)malloc(n * sizeof(int));
    if (!root_to_id) {
        free(cr->labels);
        uf_free(&uf);
        PyErr_NoMemory();
        return -1;
    }
    memset(root_to_id, -1, n * sizeof(int));

    int num_comp = 0;
    for (int i = 0; i < n; i++) {
        int r = uf_find(&uf, i);
        if (root_to_id[r] == -1)
            root_to_id[r] = num_comp++;
        cr->labels[i] = root_to_id[r];
    }
    cr->num_comp = num_comp;

    free(root_to_id);
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

    /* Build Python list of sets */
    PyObject *result = PyList_New(num_comp);
    if (!result) goto cleanup;

    for (int c = 0; c < num_comp; c++) {
        PyObject *s = PySet_New(NULL);
        if (!s) { Py_DECREF(result); result = NULL; goto cleanup; }
        for (int j = 0; j < sizes[c]; j++) {
            PyObject *val = PyLong_FromLong(buckets[c][j]);
            PySet_Add(s, val);
            Py_DECREF(val);
        }
        PyList_SET_ITEM(result, c, s);
    }

cleanup:
    free(cr.labels); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) free(buckets[c]);
    free(buckets);
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

    int *disc = (int *)malloc((size_t)n * sizeof(int));
    int *low  = (int *)malloc((size_t)n * sizeof(int));
    int *stk  = (int *)malloc((size_t)n * sizeof(int));
    int *sidx = (int *)malloc((size_t)n * sizeof(int));
    int *peid = (int *)malloc((size_t)n * sizeof(int));

    if (!disc || !low || !stk || !sidx || !peid) {
        free(disc); free(low); free(stk); free(sidx); free(peid);
        free_adj(&al); free_edges(&el); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
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
                            free(disc); free(low); free(stk);
                            free(sidx); free(peid);
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

    free(disc); free(low); free(stk); free(sidx); free(peid);
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

    int *disc     = (int *)malloc((size_t)n * sizeof(int));
    int *low      = (int *)malloc((size_t)n * sizeof(int));
    int *stk      = (int *)malloc((size_t)n * sizeof(int));
    int *sidx     = (int *)malloc((size_t)n * sizeof(int));
    int *peid     = (int *)malloc((size_t)n * sizeof(int));
    int *children = (int *)calloc((size_t)n, sizeof(int));

    if (!disc || !low || !stk || !sidx || !peid || !children) {
        free(disc); free(low); free(stk); free(sidx);
        free(peid); free(children);
        free_adj(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    memset(disc, -1, (size_t)n * sizeof(int));

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

    free(disc); free(low); free(stk); free(sidx);
    free(peid); free(children);
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

    int *disc = (int *)malloc((size_t)n * sizeof(int));
    int *low  = (int *)malloc((size_t)n * sizeof(int));
    int *stk  = (int *)malloc((size_t)n * sizeof(int));
    int *sidx = (int *)malloc((size_t)n * sizeof(int));
    int *peid = (int *)malloc((size_t)n * sizeof(int));
    int *esu  = (int *)malloc((size_t)m * sizeof(int));
    int *esv  = (int *)malloc((size_t)m * sizeof(int));

    if (!disc || !low || !stk || !sidx || !peid || !esu || !esv) {
        free(disc); free(low); free(stk); free(sidx);
        free(peid); free(esu); free(esv);
        free_adj(&al); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
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
                            free(disc); free(low); free(stk); free(sidx);
                            free(peid); free(esu); free(esv);
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

    free(disc); free(low); free(stk); free(sidx);
    free(peid); free(esu); free(esv);
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

static inline void mh_sift_up(MinHeap *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->dist[h->heap[i]] < h->dist[h->heap[p]]) {
            mh_swap(h, i, p);
            i = p;
        } else break;
    }
}

static inline void mh_sift_down(MinHeap *h, int i) {
    while (1) {
        int s = i, l = 2 * i + 1, r = 2 * i + 2;
        if (l < h->size && h->dist[h->heap[l]] < h->dist[h->heap[s]]) s = l;
        if (r < h->size && h->dist[h->heap[r]] < h->dist[h->heap[s]]) s = r;
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
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core",
    "Fast graph algorithms: union-find, Tarjan's, BFS, Dijkstra.", -1, methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
