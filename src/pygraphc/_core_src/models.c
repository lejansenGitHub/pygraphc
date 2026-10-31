/* Init and free helpers of the data models. */
#include "models.h"

void free_edges(EdgeList *el) {
    if (el->owns_memory) {
        free(el->first_endpoints);
        free(el->second_endpoints);
    }
}

int intmap_init(IntMap *m, int n) {
    int cap = 16;
    while (cap < n * 2) cap <<= 1;
    m->capacity = cap;
    m->capacity_mask = cap - 1;
    m->keys = (long *)malloc((size_t)cap * sizeof(long));
    m->values = (int *)malloc((size_t)cap * sizeof(int));
    if (!m->keys || !m->values) {
        free(m->keys); free(m->values);
        PyErr_NoMemory();
        return -1;
    }
    memset(m->keys, 0xFF, (size_t)cap * sizeof(long)); /* -1 = empty sentinel */
    return 0;
}

void intmap_free(IntMap *m) {
    free(m->keys);
    free(m->values);
}

void node_id_context_free(NodeIdContext *ctx) {
    free_edges(&ctx->edges);
    if (ctx->node_index_by_node_id.keys) intmap_free(&ctx->node_index_by_node_id);
    Py_DECREF(ctx->node_ids_fast);
}

int build_adjacency(int n, EdgeList *el, AdjacencyList *al, int directed) {
    Py_ssize_t m = el->edge_count;
    Py_ssize_t total = directed ? m : 2 * m;
    al->offset = (int *)calloc((size_t)(n + 1), sizeof(int));
    al->neighbors = total > 0 ? (int *)malloc((size_t)total * sizeof(int)) : NULL;
    al->edge_indices = total > 0 ? (int *)malloc((size_t)total * sizeof(int)) : NULL;
    al->reverse_offset = NULL;
    al->reverse_neighbors = NULL;
    al->reverse_edge_indices = NULL;
    if (!al->offset || (total > 0 && (!al->neighbors || !al->edge_indices))) {
        free(al->offset); free(al->neighbors); free(al->edge_indices);
        PyErr_NoMemory();
        return -1;
    }

    /* Count degrees for forward CSR */
    for (Py_ssize_t i = 0; i < m; i++) {
        al->offset[EDGE_FIRST_ENDPOINT(el, i) + 1]++;
        if (!directed)
            al->offset[EDGE_SECOND_ENDPOINT(el, i) + 1]++;
    }
    for (int i = 1; i <= n; i++) al->offset[i] += al->offset[i - 1];

    /* Fill forward CSR */
    if (total > 0) {
        int *pos = (int *)malloc((size_t)n * sizeof(int));
        if (!pos) {
            free(al->offset); free(al->neighbors); free(al->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(pos, al->offset, (size_t)n * sizeof(int));
        for (Py_ssize_t i = 0; i < m; i++) {
            int u = EDGE_FIRST_ENDPOINT(el, i), v = EDGE_SECOND_ENDPOINT(el, i);
            al->neighbors[pos[u]] = v; al->edge_indices[pos[u]++] = (int)i;
            if (!directed) {
                al->neighbors[pos[v]] = u; al->edge_indices[pos[v]++] = (int)i;
            }
        }
        free(pos);
    }

    /* Build reverse CSR for directed graphs */
    if (directed && m > 0) {
        al->reverse_offset = (int *)calloc((size_t)(n + 1), sizeof(int));
        al->reverse_neighbors = (int *)malloc((size_t)m * sizeof(int));
        al->reverse_edge_indices = (int *)malloc((size_t)m * sizeof(int));
        if (!al->reverse_offset || !al->reverse_neighbors || !al->reverse_edge_indices) {
            free(al->reverse_offset); free(al->reverse_neighbors); free(al->reverse_edge_indices);
            al->reverse_offset = NULL; al->reverse_neighbors = NULL; al->reverse_edge_indices = NULL;
            free(al->offset); free(al->neighbors); free(al->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t i = 0; i < m; i++)
            al->reverse_offset[EDGE_SECOND_ENDPOINT(el, i) + 1]++;
        for (int i = 1; i <= n; i++) al->reverse_offset[i] += al->reverse_offset[i - 1];

        int *pos = (int *)malloc((size_t)n * sizeof(int));
        if (!pos) {
            free(al->reverse_offset); free(al->reverse_neighbors); free(al->reverse_edge_indices);
            al->reverse_offset = NULL; al->reverse_neighbors = NULL; al->reverse_edge_indices = NULL;
            free(al->offset); free(al->neighbors); free(al->edge_indices);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(pos, al->reverse_offset, (size_t)n * sizeof(int));
        for (Py_ssize_t i = 0; i < m; i++) {
            int u = EDGE_FIRST_ENDPOINT(el, i), v = EDGE_SECOND_ENDPOINT(el, i);
            al->reverse_neighbors[pos[v]] = u; al->reverse_edge_indices[pos[v]++] = (int)i;
        }
        free(pos);
    }

    return 0;
}

void free_adjacency(AdjacencyList *al) {
    free(al->offset);
    free(al->neighbors);
    free(al->edge_indices);
    free(al->reverse_offset);
    free(al->reverse_neighbors);
    free(al->reverse_edge_indices);
}

void graph_context_destructor(PyObject *capsule) {
    GraphContext *g = (GraphContext *)PyCapsule_GetPointer(capsule, "pygraphc.GraphCtx");
    if (!g) { PyErr_Clear(); return; }
    if (g->has_adjacency) free_adjacency(&g->adjacency);
    free_edges(&g->node_ids.edges);
    if (g->node_ids.node_index_by_node_id.keys) intmap_free(&g->node_ids.node_index_by_node_id);
    Py_DECREF(g->node_ids.node_ids_fast);
    free(g);
}

void free_weights(WeightList *wl) {
    if (wl->owns_memory) free(wl->values);
}
