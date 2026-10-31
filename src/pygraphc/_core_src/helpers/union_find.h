#ifndef PYGRAPHC_CORE_UNION_FIND_H
#define PYGRAPHC_CORE_UNION_FIND_H

#include <stdlib.h>

/* ── Union-Find with path compression + union by rank ── */

typedef struct {
    int *parent;
    int *rank;
    int node_count;
} UnionFind;

static inline int union_find_init(UnionFind *uf, int n) {
    uf->node_count = n;
    uf->parent = (int *)malloc(2 * (size_t)n * sizeof(int));
    if (!uf->parent) return -1;
    uf->rank = uf->parent + n;
    for (int i = 0; i < n; i++) { uf->parent[i] = i; uf->rank[i] = 0; }
    return 0;
}

static inline void union_find_free(UnionFind *uf) {
    free(uf->parent);  /* rank is part of same allocation */
}

static inline int union_find_find(UnionFind *uf, int x) {
    while (uf->parent[x] != x) {
        int next = uf->parent[uf->parent[x]];
        uf->parent[x] = next;
        x = next;
    }
    return x;
}

static inline void union_find_union(UnionFind *uf, int a, int b) {
    a = union_find_find(uf, a);
    b = union_find_find(uf, b);
    if (a == b) return;
    if (uf->rank[a] < uf->rank[b]) { int t = a; a = b; b = t; }
    uf->parent[b] = a;
    if (uf->rank[a] == uf->rank[b]) uf->rank[a]++;
}

#endif
