#ifndef PYGRAPHC_CORE_UNION_FIND_H
#define PYGRAPHC_CORE_UNION_FIND_H

#include <stdlib.h>

/* ── Union-Find with path compression + union by rank ── */

typedef struct {
    int *parent;
    int *rank;
    int node_count;
} UnionFind;

static inline int union_find_init(UnionFind *union_find, int node_count) {
    union_find->node_count = node_count;
    union_find->parent = (int *)malloc(2 * (size_t)node_count * sizeof(int));
    if (!union_find->parent) return -1;
    union_find->rank = union_find->parent + node_count;
    for (int node = 0; node < node_count; node++) { union_find->parent[node] = node; union_find->rank[node] = 0; }
    return 0;
}

static inline void union_find_free(UnionFind *union_find) {
    free(union_find->parent);  /* rank is part of same allocation */
}

static inline int union_find_find(UnionFind *union_find, int node) {
    while (union_find->parent[node] != node) {
        int next = union_find->parent[union_find->parent[node]];
        union_find->parent[node] = next;
        node = next;
    }
    return node;
}

static inline void union_find_union(UnionFind *union_find, int first_node, int second_node) {
    first_node = union_find_find(union_find, first_node);
    second_node = union_find_find(union_find, second_node);
    if (first_node == second_node) return;
    if (union_find->rank[first_node] < union_find->rank[second_node]) { int swap = first_node; first_node = second_node; second_node = swap; }
    union_find->parent[second_node] = first_node;
    if (union_find->rank[first_node] == union_find->rank[second_node]) union_find->rank[first_node]++;
}

#endif
