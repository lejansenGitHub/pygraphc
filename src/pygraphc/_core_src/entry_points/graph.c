/* Graph construction: the parsed graph capsule, its counters, and edge translation. */
#include "entry_points.h"
#include "../helpers/parse.h"

PyObject *py_parse_graph(PyObject *self, PyObject *args) {
    PyObject *nids, *edges_or_src, *dst_obj = Py_None;
    int directed = 0;
    if (!PyArg_ParseTuple(args, "OO|Op", &nids, &edges_or_src, &dst_obj, &directed))
        return NULL;

    GraphCtx *g = (GraphCtx *)malloc(sizeof(GraphCtx));
    if (!g) { PyErr_NoMemory(); return NULL; }
    g->has_adj = 0;
    g->directed = directed;

    int rc;
    if (dst_obj != Py_None)
        rc = nid_parse_split(nids, edges_or_src, dst_obj, &g->nid);
    else
        rc = nid_parse(nids, edges_or_src, &g->nid);
    if (rc < 0) { free(g); return NULL; }

    if (g->nid.n > 0 && g->nid.el.m > 0) {
        if (build_adj(g->nid.n, &g->nid.el, &g->al, directed) < 0) {
            nid_free(&g->nid);
            free(g);
            return NULL;
        }
        g->has_adj = 1;
    }

    return PyCapsule_New(g, "pygraphc.GraphCtx", graphctx_destructor);
}

PyObject *py_graph_is_directed(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    return PyBool_FromLong(g->directed);
}

PyObject *py_graph_edge_count(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    return PyLong_FromSsize_t(g->nid.el.m);
}

PyObject *py_graph_node_count(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    return PyLong_FromLong(g->nid.n);
}

/* ── translate_edges(node_ids, nid_edges) -> index-based edges ── */

PyObject *py_translate_edges(PyObject *self, PyObject *args) {
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
