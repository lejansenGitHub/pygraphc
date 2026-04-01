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
    /* Path splitting – every node points to its grandparent.
       Avoids recursion, cache-friendly. */
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

/* ── Parse edge list once, shared by both public functions ── */

typedef struct {
    int *src;   /* source node of each edge */
    int *dst;   /* destination node of each edge */
    Py_ssize_t m;  /* number of edges */
} EdgeList;

/* Returns 0 on success, -1 on error (exception set). */
static int parse_edges(PyObject *edges_obj, EdgeList *el) {
    PyObject *fast = PySequence_Fast(edges_obj, "edges must be a sequence");
    if (!fast) return -1;

    Py_ssize_t m = PySequence_Fast_GET_SIZE(fast);
    el->m = m;
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

static void free_edges(EdgeList *el) {
    free(el->src);
    free(el->dst);
}

/* ── connected_components(n, edges) -> list[set[int]] ── */

static PyObject *py_connected_components(PyObject *self, PyObject *args) {
    int n;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &n, &edges_obj)) return NULL;

    EdgeList el;
    if (parse_edges(edges_obj, &el) < 0) return NULL;

    /* Build union-find */
    UF uf;
    uf_init(&uf, n);
    for (Py_ssize_t i = 0; i < el.m; i++)
        uf_union(&uf, el.src[i], el.dst[i]);
    free_edges(&el);

    /* Map root -> component index, collect into buckets.
       We do two passes to avoid Python overhead per-node. */

    /* Pass 1: find roots, assign component indices, count sizes */
    int *root_of = (int *)malloc(n * sizeof(int));
    int *comp_id = (int *)malloc(n * sizeof(int));  /* maps root -> sequential id */
    int *sizes   = NULL;
    int num_comp = 0;

    memset(comp_id, -1, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int r = uf_find(&uf, i);
        root_of[i] = r;
        if (comp_id[r] == -1) {
            comp_id[r] = num_comp++;
        }
    }
    uf_free(&uf);

    /* Pass 2: bucket nodes by component */
    sizes = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[comp_id[root_of[i]]]++;

    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++)
        buckets[c] = (int *)malloc(sizes[c] * sizeof(int));

    for (int i = 0; i < n; i++) {
        int c = comp_id[root_of[i]];
        buckets[c][offsets[c]++] = i;
    }

    /* Build Python list of frozensets */
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
        PyList_SET_ITEM(result, c, s);  /* steals ref */
    }

cleanup:
    free(root_of); free(comp_id); free(sizes); free(offsets);
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

    /* Build union-find */
    UF uf;
    uf_init(&uf, n);
    for (Py_ssize_t i = 0; i < el.m; i++)
        uf_union(&uf, el.src[i], el.dst[i]);

    /* Assign component IDs */
    int *root_of = (int *)malloc(n * sizeof(int));
    int *comp_id = (int *)malloc(n * sizeof(int));
    int num_comp = 0;
    memset(comp_id, -1, n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int r = uf_find(&uf, i);
        root_of[i] = r;
        if (comp_id[r] == -1) comp_id[r] = num_comp++;
    }
    uf_free(&uf);

    /* Bucket nodes */
    int *sizes = (int *)calloc(num_comp, sizeof(int));
    for (int i = 0; i < n; i++) sizes[comp_id[root_of[i]]]++;
    int **buckets = (int **)malloc(num_comp * sizeof(int *));
    int *offsets  = (int *)calloc(num_comp, sizeof(int));
    for (int c = 0; c < num_comp; c++) buckets[c] = (int *)malloc(sizes[c] * sizeof(int));
    for (int i = 0; i < n; i++) {
        int c = comp_id[root_of[i]];
        buckets[c][offsets[c]++] = i;
    }

    /* Build branch sets per component using Python sets directly.
       This avoids allocating C-side branch buckets. */
    PyObject **branch_sets = (PyObject **)calloc(num_comp, sizeof(PyObject *));
    for (int c = 0; c < num_comp; c++)
        branch_sets[c] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = el.m < br_len ? el.m : br_len;
    PyObject **br_items = PySequence_Fast_ITEMS(br_fast);
    for (Py_ssize_t i = 0; i < edge_branch_count; i++) {
        int c = comp_id[root_of[el.src[i]]];
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
        PyList_SET_ITEM(result, c, tup);  /* steals ref */
    }

cleanup2:
    free(root_of); free(comp_id); free(sizes); free(offsets);
    for (int c = 0; c < num_comp; c++) {
        free(buckets[c]);
        Py_DECREF(branch_sets[c]);
    }
    free(buckets); free(branch_sets);
    return result;
}

/* ── Module definition ── */

static PyMethodDef methods[] = {
    {"connected_components", py_connected_components, METH_VARARGS,
     "connected_components(n, edges) -> list[set[int]]\n\n"
     "Find connected components using union-find with path compression."},
    {"connected_components_with_branches", py_connected_components_with_branches, METH_VARARGS,
     "connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]]\n\n"
     "Find connected components with associated branch IDs."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core",
    "Fast union-find connected components.", -1, methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
