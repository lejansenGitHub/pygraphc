/* Dijkstra: source-target paths, single-source and multi-source lengths. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/min_heap.h"
#include "../helpers/bidirectional_dijkstra.h"

/* ── Dijkstra (source -> target, returns distance + path) ── */

PyObject *py_dijkstra(PyObject *self, PyObject *args) {
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
    if (parse_weights(weights_obj, &wl, el.m) < 0) { free_edges(&el); return NULL; }

    AdjList al;
    if (build_adj(n, &el, &al, 0) < 0) {
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

PyObject *py_sssp_lengths(PyObject *self, PyObject *args) {
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
    if (parse_weights(weights_obj, &wl, el.m) < 0) {
        free_edges(&el); Py_DECREF(result_dict); return NULL;
    }

    AdjList al;
    if (build_adj(n, &el, &al, 0) < 0) {
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

PyObject *py_multi_source_dijkstra(PyObject *self, PyObject *args) {
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
    if (parse_weights(weights_obj, &wl, el.m) < 0) {
        free_edges(&el); Py_DECREF(src_fast); Py_DECREF(result_dict);
        return NULL;
    }

    AdjList al;
    if (build_adj(n, &el, &al, 0) < 0) {
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

/* ── dijkstra (node-ID edges) ── */

PyObject *py_dijkstra_nid(PyObject *self, PyObject *args) {
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
    if (parse_weights(wobj, &wl, ctx.el.m) < 0) { nid_free(&ctx); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al, 0) < 0) { nid_free(&ctx); free_weights(&wl); return NULL; }

    BidiPath bp;
    int failed = bidi_dijkstra(n, &al, wl.w, NULL, NULL, 0, source, target, &bp) < 0;
    PyObject *path = failed ? NULL : bidi_path_to_list(&bp, &ctx);
    free(bp.path);
    free_adj(&al); free_weights(&wl);
    if (!path) { nid_free(&ctx); return NULL; }
    PyObject *res = Py_BuildValue("(dN)", bp.weight, path);
    nid_free(&ctx);
    return res;
}

/* ── sssp_lengths (node-ID edges) ── */

PyObject *py_sssp_nid(PyObject *self, PyObject *args) {
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
    if (parse_weights(wobj, &wl, ctx.el.m) < 0) { nid_free(&ctx); Py_DECREF(rd); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al, 0) < 0) { nid_free(&ctx); free_weights(&wl); Py_DECREF(rd); return NULL; }

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

PyObject *py_msdijk_nid(PyObject *self, PyObject *args) {
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
    if (parse_weights(wobj, &wl, ctx.el.m) < 0) { Py_DECREF(sf); nid_free(&ctx); Py_DECREF(rd); return NULL; }
    AdjList al;
    if (build_adj(n, &ctx.el, &al, 0) < 0) { Py_DECREF(sf); free_weights(&wl); nid_free(&ctx); Py_DECREF(rd); return NULL; }

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

PyObject *py_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *wobj, *src_obj, *tgt_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|OO", &capsule, &wobj, &src_obj, &tgt_obj, &mask_obj, &nmask_obj))
        return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    long sn = PyLong_AsLong(src_obj), tn = PyLong_AsLong(tgt_obj);
    if (PyErr_Occurred()) return NULL;
    int source = n > 0 ? intmap_get(&g->nid.im, sn) : -1;
    int target = n > 0 ? intmap_get(&g->nid.im, tn) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(g->nid.nid_items[source]);
        PyList_SET_ITEM(path, 0, g->nid.nid_items[source]);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (n == 0 || source < 0 || target < 0) {
        PyObject *p = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, p);
    }

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    if (nmask && (nmask[source] || nmask[target])) {
        release_mask(&nmbuf); release_mask(&mbuf);
        PyObject *p = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, p);
    }

    WeightList wl;
    if (parse_weights(wobj, &wl, g->nid.el.m) < 0) { release_mask(&nmbuf); release_mask(&mbuf); return NULL; }
    if (!g->has_adj) { free_weights(&wl); release_mask(&nmbuf); release_mask(&mbuf); PyObject *p = PyList_New(0); return Py_BuildValue("(dN)", HUGE_VAL, p); }

    BidiPath bp;
    int failed = bidi_dijkstra(n, &g->al, wl.w, mask, nmask, g->directed, source, target, &bp) < 0;
    PyObject *path = failed ? NULL : bidi_path_to_list(&bp, &g->nid);
    free(bp.path);
    free_weights(&wl); release_mask(&nmbuf); release_mask(&mbuf);
    if (!path) return NULL;
    return Py_BuildValue("(dN)", bp.weight, path);
}

PyObject *py_sssp_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *wobj, *src_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OO", &capsule, &wobj, &src_obj, &cutoff, &mask_obj, &nmask_obj))
        return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    int use_cutoff = (cutoff >= 0.0);

    PyObject *rd = PyDict_New();
    if (!rd) return NULL;

    long sn = PyLong_AsLong(src_obj);
    if (PyErr_Occurred()) { Py_DECREF(rd); return NULL; }
    int source = n > 0 ? intmap_get(&g->nid.im, sn) : -1;
    if (n == 0 || source < 0 || !g->has_adj) return rd;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) { Py_DECREF(rd); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(rd); return NULL; }

    if (nmask && nmask[source]) {
        release_mask(&nmbuf); release_mask(&mbuf);
        return rd;  /* source masked out — empty result */
    }

    WeightList wl;
    if (parse_weights(wobj, &wl, g->nid.el.m) < 0) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd); return NULL; }
    AdjList *al = &g->al;

    double *dist = malloc(n*sizeof(double));
    int *heap = malloc(n*sizeof(int)), *pos = malloc(n*sizeof(int));
    if (!dist||!heap||!pos) {
        free(dist);free(heap);free(pos);
        free_weights(&wl); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd); PyErr_NoMemory(); return NULL;
    }
    for (int i=0;i<n;i++) { dist[i]=HUGE_VAL; pos[i]=-1; }
    MinHeap mh = {dist, heap, pos, 0};
    dist[source]=0.0; heap[0]=source; pos[source]=0; mh.size=1;

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u]==HUGE_VAL) break;
        if (use_cutoff && dist[u]>cutoff) break;
        for (int i=al->offset[u]; i<al->offset[u+1]; i++) {
            if (mask && mask[al->eid[i]]) continue;
            int v=al->adj[i];
            if (nmask && nmask[v]) continue;
            double nd=dist[u]+wl.w[al->eid[i]];
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v]==-1) { dist[v]=nd; int p=mh.size++; heap[p]=v; pos[v]=p; mh_sift_up(&mh,p); }
                else { dist[v]=nd; mh_sift_up(&mh, pos[v]); }
            }
        }
    }

    for (int i=0;i<n;i++) {
        if (nmask && nmask[i]) continue;
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(rd, g->nid.nid_items[i], val);
            Py_DECREF(val);
        }
    }
    free(dist);free(heap);free(pos);
    free_weights(&wl); release_mask(&nmbuf); release_mask(&mbuf);
    return rd;
}

PyObject *py_msdijk_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *wobj, *srcs_obj, *mask_obj = Py_None, *nmask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OO", &capsule, &wobj, &srcs_obj, &cutoff, &mask_obj, &nmask_obj))
        return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    int use_cutoff = (cutoff >= 0.0);

    PyObject *rd = PyDict_New();
    if (!rd) return NULL;
    if (n == 0 || !g->has_adj) return rd;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) { Py_DECREF(rd); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(rd); return NULL; }

    PyObject *sf = PySequence_Fast(srcs_obj, "sources must be a sequence");
    if (!sf) { release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd); return NULL; }

    WeightList wl;
    if (parse_weights(wobj, &wl, g->nid.el.m) < 0) { Py_DECREF(sf); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd); return NULL; }
    AdjList *al = &g->al;

    double *dist = malloc(n*sizeof(double));
    int *heap = malloc(n*sizeof(int)), *pos = malloc(n*sizeof(int));
    if (!dist||!heap||!pos) {
        free(dist);free(heap);free(pos);
        free_weights(&wl); Py_DECREF(sf); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd);
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
            free_weights(&wl); Py_DECREF(sf); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(rd);
            return NULL;
        }
        int s = intmap_get(&g->nid.im, sn);
        if (s >= 0 && !(nmask && nmask[s]) && dist[s] > 0.0) {
            dist[s]=0.0; int p=mh.size++; heap[p]=s; pos[s]=p; mh_sift_up(&mh,p);
        }
    }
    Py_DECREF(sf);

    while (mh.size > 0) {
        int u = mh_pop(&mh);
        if (dist[u]==HUGE_VAL) break;
        if (use_cutoff && dist[u]>cutoff) break;
        for (int i=al->offset[u]; i<al->offset[u+1]; i++) {
            if (mask && mask[al->eid[i]]) continue;
            int v=al->adj[i];
            if (nmask && nmask[v]) continue;
            double nd=dist[u]+wl.w[al->eid[i]];
            if (nd < dist[v] && (!use_cutoff || nd <= cutoff)) {
                if (pos[v]==-1) { dist[v]=nd; int p=mh.size++; heap[p]=v; pos[v]=p; mh_sift_up(&mh,p); }
                else { dist[v]=nd; mh_sift_up(&mh, pos[v]); }
            }
        }
    }

    for (int i=0;i<n;i++) {
        if (nmask && nmask[i]) continue;
        if (dist[i] < HUGE_VAL && (!use_cutoff || dist[i] <= cutoff)) {
            PyObject *val = PyFloat_FromDouble(dist[i]);
            PyDict_SetItem(rd, g->nid.nid_items[i], val);
            Py_DECREF(val);
        }
    }
    free(dist);free(heap);free(pos);
    free_weights(&wl); release_mask(&nmbuf); release_mask(&mbuf);
    return rd;
}
