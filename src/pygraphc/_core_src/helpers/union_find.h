#ifndef PYGRAPHC_CORE_UNION_FIND_H
#define PYGRAPHC_CORE_UNION_FIND_H

#include <stdlib.h>

/* ── Union-Find with path compression + union by rank ── */

typedef struct {
    int *parent;
    int *rank;
    int n;
} UF;

static inline int uf_init(UF *uf, int n) {
    uf->n = n;
    uf->parent = (int *)malloc(2 * (size_t)n * sizeof(int));
    if (!uf->parent) return -1;
    uf->rank = uf->parent + n;
    for (int i = 0; i < n; i++) { uf->parent[i] = i; uf->rank[i] = 0; }
    return 0;
}

static inline void uf_free(UF *uf) {
    free(uf->parent);  /* rank is part of same allocation */
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

#endif
