#ifndef PYGRAPHC_CORE_MIN_HEAP_H
#define PYGRAPHC_CORE_MIN_HEAP_H

/* ── Binary min-heap for Dijkstra ── */

typedef struct {
    double *dist;
    int *heap;
    int *pos;
    int size;
} MinHeap;

static inline void mh_swap(MinHeap *h, int i, int j) {
    int ni = h->heap[i], nj = h->heap[j];
    h->heap[i] = nj; h->heap[j] = ni;
    h->pos[ni] = j; h->pos[nj] = i;
}

/* 4-ary heap: shallower tree = fewer cache misses in sift_down */
#define MH_ARITY 4

static inline void mh_sift_up(MinHeap *h, int i) {
    while (i > 0) {
        int p = (i - 1) / MH_ARITY;
        if (h->dist[h->heap[i]] < h->dist[h->heap[p]]) {
            mh_swap(h, i, p);
            i = p;
        } else break;
    }
}

static inline void mh_sift_down(MinHeap *h, int i) {
    while (1) {
        int first = MH_ARITY * i + 1;
        if (first >= h->size) break;
        int last = first + MH_ARITY;
        if (last > h->size) last = h->size;
        int s = i;
        double sd = h->dist[h->heap[s]];
        for (int c = first; c < last; c++) {
            double cd = h->dist[h->heap[c]];
            if (cd < sd) { s = c; sd = cd; }
        }
        if (s != i) { mh_swap(h, i, s); i = s; }
        else break;
    }
}

static inline int mh_pop(MinHeap *h) {
    int node = h->heap[0];
    h->pos[node] = -1;
    h->size--;
    if (h->size > 0) {
        h->heap[0] = h->heap[h->size];
        h->pos[h->heap[0]] = 0;
        mh_sift_down(h, 0);
    }
    return node;
}

#endif
