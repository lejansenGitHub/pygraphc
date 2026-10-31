/* Init and free helpers of the data models. */
#include "models.h"

void free_edges(EdgeList *el) {
    if (el->owns_memory) {
        free(el->src);
        free(el->dst);
    }
}

int intmap_init(IntMap *m, int n) {
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

void intmap_free(IntMap *m) {
    free(m->keys);
    free(m->vals);
}

void nid_free(NidContext *ctx) {
    free_edges(&ctx->el);
    if (ctx->im.keys) intmap_free(&ctx->im);
    Py_DECREF(ctx->nid_fast);
}

int build_adj(int n, EdgeList *el, AdjList *al, int directed) {
    Py_ssize_t m = el->m;
    Py_ssize_t total = directed ? m : 2 * m;
    al->offset = (int *)calloc((size_t)(n + 1), sizeof(int));
    al->adj = total > 0 ? (int *)malloc((size_t)total * sizeof(int)) : NULL;
    al->eid = total > 0 ? (int *)malloc((size_t)total * sizeof(int)) : NULL;
    al->rev_offset = NULL;
    al->rev_adj = NULL;
    al->rev_eid = NULL;
    if (!al->offset || (total > 0 && (!al->adj || !al->eid))) {
        free(al->offset); free(al->adj); free(al->eid);
        PyErr_NoMemory();
        return -1;
    }

    /* Count degrees for forward CSR */
    for (Py_ssize_t i = 0; i < m; i++) {
        al->offset[EDGE_SRC(el, i) + 1]++;
        if (!directed)
            al->offset[EDGE_DST(el, i) + 1]++;
    }
    for (int i = 1; i <= n; i++) al->offset[i] += al->offset[i - 1];

    /* Fill forward CSR */
    if (total > 0) {
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
            if (!directed) {
                al->adj[pos[v]] = u; al->eid[pos[v]++] = (int)i;
            }
        }
        free(pos);
    }

    /* Build reverse CSR for directed graphs */
    if (directed && m > 0) {
        al->rev_offset = (int *)calloc((size_t)(n + 1), sizeof(int));
        al->rev_adj = (int *)malloc((size_t)m * sizeof(int));
        al->rev_eid = (int *)malloc((size_t)m * sizeof(int));
        if (!al->rev_offset || !al->rev_adj || !al->rev_eid) {
            free(al->rev_offset); free(al->rev_adj); free(al->rev_eid);
            al->rev_offset = NULL; al->rev_adj = NULL; al->rev_eid = NULL;
            free(al->offset); free(al->adj); free(al->eid);
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t i = 0; i < m; i++)
            al->rev_offset[EDGE_DST(el, i) + 1]++;
        for (int i = 1; i <= n; i++) al->rev_offset[i] += al->rev_offset[i - 1];

        int *pos = (int *)malloc((size_t)n * sizeof(int));
        if (!pos) {
            free(al->rev_offset); free(al->rev_adj); free(al->rev_eid);
            al->rev_offset = NULL; al->rev_adj = NULL; al->rev_eid = NULL;
            free(al->offset); free(al->adj); free(al->eid);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(pos, al->rev_offset, (size_t)n * sizeof(int));
        for (Py_ssize_t i = 0; i < m; i++) {
            int u = EDGE_SRC(el, i), v = EDGE_DST(el, i);
            al->rev_adj[pos[v]] = u; al->rev_eid[pos[v]++] = (int)i;
        }
        free(pos);
    }

    return 0;
}

void free_adj(AdjList *al) {
    free(al->offset);
    free(al->adj);
    free(al->eid);
    free(al->rev_offset);
    free(al->rev_adj);
    free(al->rev_eid);
}

void graphctx_destructor(PyObject *capsule) {
    GraphCtx *g = (GraphCtx *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!g) { PyErr_Clear(); return; }
    if (g->has_adj) free_adj(&g->al);
    free_edges(&g->nid.el);
    if (g->nid.im.keys) intmap_free(&g->nid.im);
    Py_DECREF(g->nid.nid_fast);
    free(g);
}

void free_weights(WeightList *wl) {
    if (wl->owns_memory) free(wl->w);
}
