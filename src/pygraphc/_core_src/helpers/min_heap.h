#ifndef PYGRAPHC_CORE_MIN_HEAP_H
#define PYGRAPHC_CORE_MIN_HEAP_H

/* ── Binary min-heap for Dijkstra ── */

typedef struct {
    double *distance;
    int *nodes;
    int *heap_position;
    int size;
} MinHeap;

static inline void min_heap_swap(MinHeap *h, int i, int j) {
    int ni = h->nodes[i], nj = h->nodes[j];
    h->nodes[i] = nj; h->nodes[j] = ni;
    h->heap_position[ni] = j; h->heap_position[nj] = i;
}

/* 4-ary heap: shallower tree = fewer cache misses in sift_down */
#define MIN_HEAP_ARITY 4

static inline void min_heap_sift_up(MinHeap *h, int i) {
    while (i > 0) {
        int p = (i - 1) / MIN_HEAP_ARITY;
        if (h->distance[h->nodes[i]] < h->distance[h->nodes[p]]) {
            min_heap_swap(h, i, p);
            i = p;
        } else break;
    }
}

static inline void min_heap_sift_down(MinHeap *h, int i) {
    while (1) {
        int first = MIN_HEAP_ARITY * i + 1;
        if (first >= h->size) break;
        int last = first + MIN_HEAP_ARITY;
        if (last > h->size) last = h->size;
        int s = i;
        double sd = h->distance[h->nodes[s]];
        for (int c = first; c < last; c++) {
            double cd = h->distance[h->nodes[c]];
            if (cd < sd) { s = c; sd = cd; }
        }
        if (s != i) { min_heap_swap(h, i, s); i = s; }
        else break;
    }
}

static inline int min_heap_pop(MinHeap *h) {
    int node = h->nodes[0];
    h->heap_position[node] = -1;
    h->size--;
    if (h->size > 0) {
        h->nodes[0] = h->nodes[h->size];
        h->heap_position[h->nodes[0]] = 0;
        min_heap_sift_down(h, 0);
    }
    return node;
}

#endif
