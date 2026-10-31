#include "components.h"
#include "union_find.h"

/*
 * Core routine: parse edges, run union-find, assign sequential component IDs.
 * mask: if non-NULL, skip edges where mask[i] != 0.
 * Caller must free result.labels.
 * Returns 0 on success, -1 on error.
 */
int compute_components_masked(int n, EdgeList *el, const uint8_t *mask,
                              const uint8_t *node_mask,
                              ComponentResult *cr) {
    UnionFind uf;
    if (union_find_init(&uf, n) < 0) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t i = 0; i < el->edge_count; i++) {
        if (mask && mask[i]) continue;
        int u = EDGE_FIRST_ENDPOINT(el, i), v = EDGE_SECOND_ENDPOINT(el, i);
        if (node_mask && (node_mask[u] || node_mask[v])) continue;
        union_find_union(&uf, u, v);
    }

    /* labels + root_to_id in one allocation */
    int *buf = (int *)malloc(2 * (size_t)n * sizeof(int));
    if (!buf) {
        union_find_free(&uf);
        PyErr_NoMemory();
        return -1;
    }
    cr->labels = buf;
    int *root_to_id = buf + n;
    memset(root_to_id, -1, (size_t)n * sizeof(int));

    int num_comp = 0;
    for (int i = 0; i < n; i++) {
        int r = union_find_find(&uf, i);
        if (root_to_id[r] == -1)
            root_to_id[r] = num_comp++;
        cr->labels[i] = root_to_id[r];
    }
    cr->component_count = num_comp;

    union_find_free(&uf);
    return 0;
}

/* ── Strongly connected components (iterative Tarjan's, directed CSR) ── */

/*
 * Core routine: iterative Tarjan's SCC on a directed graph.
 * Uses the forward CSR (al->offset/adj/eid).
 * Supports edge mask and node mask.
 * Writes component labels into cr (same shape as compute_components_masked).
 * Returns 0 on success, -1 on error.
 */
int compute_strongly_connected_components_masked(int n, AdjacencyList *al, const uint8_t *mask,
                       const uint8_t *node_mask, ComponentResult *cr) {
    /* 6n ints: disc, low, on_stack, tarjan_stk, dfs_stk, dfs_sidx */
    int *buf = (int *)malloc(6 * (size_t)n * sizeof(int));
    if (!buf) { PyErr_NoMemory(); return -1; }
    int *disc      = buf;
    int *low       = buf + n;
    int *on_stack  = buf + 2 * n;
    int *tarjan_stk= buf + 3 * n;
    int *dfs_stk   = buf + 4 * n;
    int *dfs_sidx  = buf + 5 * n;
    memset(disc, -1, (size_t)n * sizeof(int));
    memset(on_stack, 0, (size_t)n * sizeof(int));

    cr->labels = (int *)malloc((size_t)n * sizeof(int));
    if (!cr->labels) { free(buf); PyErr_NoMemory(); return -1; }
    memset(cr->labels, -1, (size_t)n * sizeof(int));

    int timer = 0, tsp = 0, num_comp = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;

        int sp = 0;
        dfs_stk[0] = start;
        dfs_sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++;
        on_stack[start] = 1;
        tarjan_stk[tsp++] = start;

        while (sp >= 0) {
            int u = dfs_stk[sp];
            if (dfs_sidx[sp] < al->offset[u + 1]) {
                int i = dfs_sidx[sp]++;
                int eid = al->edge_indices[i];
                if (mask && mask[eid]) continue;
                int v = al->neighbors[i];
                if (node_mask && node_mask[v]) continue;
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    on_stack[v] = 1;
                    tarjan_stk[tsp++] = v;
                    dfs_stk[++sp] = v;
                    dfs_sidx[sp] = al->offset[v];
                } else if (on_stack[v]) {
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                /* Backtrack: propagate low to parent */
                if (sp > 0) {
                    int p = dfs_stk[sp - 1];
                    if (low[p] > low[u]) low[p] = low[u];
                }
                /* If u is a root of an SCC, pop the Tarjan stack */
                if (low[u] == disc[u]) {
                    int comp_id = num_comp++;
                    int w;
                    do {
                        w = tarjan_stk[--tsp];
                        on_stack[w] = 0;
                        cr->labels[w] = comp_id;
                    } while (w != u);
                }
                sp--;
            }
        }
    }

    /* Assign isolated masked-out nodes to their own components */
    for (int i = 0; i < n; i++) {
        if (cr->labels[i] == -1)
            cr->labels[i] = num_comp++;
    }

    cr->component_count = num_comp;
    free(buf);
    return 0;
}
