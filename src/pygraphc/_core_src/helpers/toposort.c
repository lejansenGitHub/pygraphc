#include "toposort.h"

/* ── Topological sort (Kahn's algorithm, directed edges) ── */

/*
 * Core: Kahn's algorithm on internal indices.
 * Edges are directed: src[i] -> dst[i].
 * Returns topological order in *out_order (caller must free).
 * Returns number of nodes in order (== n if acyclic).
 */
int topological_order(int n, EdgeList *el, int **out_order) {
    Py_ssize_t m = el->edge_count;
    int *in_deg = (int *)calloc(n, sizeof(int));
    int *head   = (int *)calloc(n, sizeof(int));  /* per-node adjacency list head */
    int *nxt    = m > 0 ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    int *adj_dst = m > 0 ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    int *queue  = (int *)malloc((size_t)n * sizeof(int));

    if (!in_deg || !head || (m > 0 && (!nxt || !adj_dst)) || !queue) {
        free(in_deg); free(head); free(nxt); free(adj_dst); free(queue);
        PyErr_NoMemory();
        return -1;
    }

    /* Initialize heads to -1 (empty list) */
    for (int i = 0; i < n; i++) head[i] = -1;

    /* Build directed adjacency list + compute in-degrees */
    for (Py_ssize_t i = 0; i < m; i++) {
        int u = EDGE_FIRST_ENDPOINT(el, i), v = EDGE_SECOND_ENDPOINT(el, i);
        in_deg[v]++;
        adj_dst[i] = v;
        nxt[i] = head[u];
        head[u] = (int)i;
    }

    /* Seed queue with zero in-degree nodes */
    int qh = 0, qt = 0;
    for (int i = 0; i < n; i++) {
        if (in_deg[i] == 0) queue[qt++] = i;
    }

    /* Process queue */
    while (qh < qt) {
        int u = queue[qh++];
        for (int e = head[u]; e >= 0; e = nxt[e]) {
            int v = adj_dst[e];
            if (--in_deg[v] == 0) queue[qt++] = v;
        }
    }

    free(in_deg); free(head); free(nxt); free(adj_dst);
    *out_order = queue;
    return qt;  /* number of nodes processed; < n means cycle */
}

/* ── toposort_ctx: topological sort from cached graph context ── */

int topological_order_masked(int n, EdgeList *el, const uint8_t *mask,
                         const uint8_t *node_mask, int **out_order) {
    Py_ssize_t m = el->edge_count;
    int *in_deg  = (int *)calloc(n, sizeof(int));
    int *head    = (int *)malloc((size_t)n * sizeof(int));
    int *nxt     = m > 0 ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    int *adj_dst = m > 0 ? (int *)malloc((size_t)m * sizeof(int)) : NULL;
    int *queue   = (int *)malloc((size_t)n * sizeof(int));

    if (!in_deg || !head || (m > 0 && (!nxt || !adj_dst)) || !queue) {
        free(in_deg); free(head); free(nxt); free(adj_dst); free(queue);
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < n; i++) head[i] = -1;

    /* Build directed adjacency list + compute in-degrees, respecting masks */
    for (Py_ssize_t i = 0; i < m; i++) {
        if (mask && mask[i]) continue;
        int u = EDGE_FIRST_ENDPOINT(el, i), v = EDGE_SECOND_ENDPOINT(el, i);
        if (node_mask && (node_mask[u] || node_mask[v])) continue;
        in_deg[v]++;
        adj_dst[i] = v;
        nxt[i] = head[u];
        head[u] = (int)i;
    }

    /* Seed queue with zero in-degree nodes (skip masked nodes) */
    int qh = 0, qt = 0;
    for (int i = 0; i < n; i++) {
        if (node_mask && node_mask[i]) continue;
        if (in_deg[i] == 0) queue[qt++] = i;
    }

    while (qh < qt) {
        int u = queue[qh++];
        for (int e = head[u]; e >= 0; e = nxt[e]) {
            if (mask && mask[e]) continue;
            int v = adj_dst[e];
            if (node_mask && node_mask[v]) continue;
            if (--in_deg[v] == 0) queue[qt++] = v;
        }
    }

    free(in_deg); free(head); free(nxt); free(adj_dst);
    *out_order = queue;
    return qt;
}
