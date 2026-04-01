#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

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
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core",
    "Fast union-find connected components.", -1, methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
