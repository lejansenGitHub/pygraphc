#include "toposort.h"

/* ── Topological sort (Kahn's algorithm, directed edges) ── */

/*
 * Core: Kahn's algorithm on internal indices.
 * Edges are directed: first endpoint -> second endpoint.
 * Returns topological order in *out_order (caller must free).
 * Returns number of nodes in order (== node_count if acyclic).
 */
int topological_order(int node_count, EdgeList *edges, int **out_order) {
    Py_ssize_t edge_count = edges->edge_count;
    int *in_degree   = (int *)calloc(node_count, sizeof(int));
    int *head        = (int *)calloc(node_count, sizeof(int));  /* per-node adjacency list head */
    int *next_edge   = edge_count > 0 ? (int *)malloc((size_t)edge_count * sizeof(int)) : NULL;
    int *edge_target = edge_count > 0 ? (int *)malloc((size_t)edge_count * sizeof(int)) : NULL;
    int *queue       = (int *)malloc((size_t)node_count * sizeof(int));

    if (!in_degree || !head || (edge_count > 0 && (!next_edge || !edge_target)) || !queue) {
        free(in_degree); free(head); free(next_edge); free(edge_target); free(queue);
        PyErr_NoMemory();
        return -1;
    }

    /* Initialize heads to -1 (empty list) */
    for (int node = 0; node < node_count; node++) head[node] = -1;

    /* Build directed adjacency list + compute in-degrees */
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        int from_node = EDGE_FIRST_ENDPOINT(edges, edge_index), to_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
        in_degree[to_node]++;
        edge_target[edge_index] = to_node;
        next_edge[edge_index] = head[from_node];
        head[from_node] = (int)edge_index;
    }

    /* Enqueue every node of in-degree zero */
    int queue_head = 0, queue_tail = 0;
    for (int node = 0; node < node_count; node++) {
        if (in_degree[node] == 0) queue[queue_tail++] = node;
    }

    /* Process queue */
    while (queue_head < queue_tail) {
        int node = queue[queue_head++];
        for (int edge_index = head[node]; edge_index >= 0; edge_index = next_edge[edge_index]) {
            int successor = edge_target[edge_index];
            if (--in_degree[successor] == 0) queue[queue_tail++] = successor;
        }
    }

    free(in_degree); free(head); free(next_edge); free(edge_target);
    *out_order = queue;
    return queue_tail;  /* number of nodes processed; < node_count means cycle */
}

/* ── topological_sort_ctx: topological sort from cached graph context ── */

int topological_order_masked(int node_count, EdgeList *edges, const uint8_t *edge_mask,
                             const uint8_t *node_mask, int **out_order) {
    Py_ssize_t edge_count = edges->edge_count;
    int *in_degree   = (int *)calloc(node_count, sizeof(int));
    int *head        = (int *)malloc((size_t)node_count * sizeof(int));
    int *next_edge   = edge_count > 0 ? (int *)malloc((size_t)edge_count * sizeof(int)) : NULL;
    int *edge_target = edge_count > 0 ? (int *)malloc((size_t)edge_count * sizeof(int)) : NULL;
    int *queue       = (int *)malloc((size_t)node_count * sizeof(int));

    if (!in_degree || !head || (edge_count > 0 && (!next_edge || !edge_target)) || !queue) {
        free(in_degree); free(head); free(next_edge); free(edge_target); free(queue);
        PyErr_NoMemory();
        return -1;
    }

    for (int node = 0; node < node_count; node++) head[node] = -1;

    /* Build directed adjacency list + compute in-degrees, respecting masks */
    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        if (edge_mask && edge_mask[edge_index]) continue;
        int from_node = EDGE_FIRST_ENDPOINT(edges, edge_index), to_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
        if (node_mask && (node_mask[from_node] || node_mask[to_node])) continue;
        in_degree[to_node]++;
        edge_target[edge_index] = to_node;
        next_edge[edge_index] = head[from_node];
        head[from_node] = (int)edge_index;
    }

    /* Enqueue every node of in-degree zero (skip masked nodes) */
    int queue_head = 0, queue_tail = 0;
    for (int node = 0; node < node_count; node++) {
        if (node_mask && node_mask[node]) continue;
        if (in_degree[node] == 0) queue[queue_tail++] = node;
    }

    while (queue_head < queue_tail) {
        int node = queue[queue_head++];
        for (int edge_index = head[node]; edge_index >= 0; edge_index = next_edge[edge_index]) {
            if (edge_mask && edge_mask[edge_index]) continue;
            int successor = edge_target[edge_index];
            if (node_mask && node_mask[successor]) continue;
            if (--in_degree[successor] == 0) queue[queue_tail++] = successor;
        }
    }

    free(in_degree); free(head); free(next_edge); free(edge_target);
    *out_order = queue;
    return queue_tail;
}
