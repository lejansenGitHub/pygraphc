#ifndef PYGRAPHC_CORE_MIN_HEAP_H
#define PYGRAPHC_CORE_MIN_HEAP_H

/* ── Binary min-heap for Dijkstra ── */

typedef struct {
    double *distance;
    int *nodes;
    int *heap_position;
    int size;
} MinHeap;

static inline void min_heap_swap(MinHeap *heap, int first_position, int second_position) {
    int first_node = heap->nodes[first_position], second_node = heap->nodes[second_position];
    heap->nodes[first_position] = second_node; heap->nodes[second_position] = first_node;
    heap->heap_position[first_node] = second_position; heap->heap_position[second_node] = first_position;
}

/* 4-ary heap: shallower tree = fewer cache misses in sift_down */
#define MIN_HEAP_ARITY 4

static inline void min_heap_sift_up(MinHeap *heap, int position) {
    while (position > 0) {
        int parent_position = (position - 1) / MIN_HEAP_ARITY;
        if (heap->distance[heap->nodes[position]] < heap->distance[heap->nodes[parent_position]]) {
            min_heap_swap(heap, position, parent_position);
            position = parent_position;
        } else break;
    }
}

static inline void min_heap_sift_down(MinHeap *heap, int position) {
    while (1) {
        int first_child = MIN_HEAP_ARITY * position + 1;
        if (first_child >= heap->size) break;
        int end_child = first_child + MIN_HEAP_ARITY;
        if (end_child > heap->size) end_child = heap->size;
        int smallest_position = position;
        double smallest_distance = heap->distance[heap->nodes[smallest_position]];
        for (int child_position = first_child; child_position < end_child; child_position++) {
            double child_distance = heap->distance[heap->nodes[child_position]];
            if (child_distance < smallest_distance) { smallest_position = child_position; smallest_distance = child_distance; }
        }
        if (smallest_position != position) { min_heap_swap(heap, position, smallest_position); position = smallest_position; }
        else break;
    }
}

static inline int min_heap_pop(MinHeap *heap) {
    int node = heap->nodes[0];
    heap->heap_position[node] = -1;
    heap->size--;
    if (heap->size > 0) {
        heap->nodes[0] = heap->nodes[heap->size];
        heap->heap_position[heap->nodes[0]] = 0;
        min_heap_sift_down(heap, 0);
    }
    return node;
}

#endif
