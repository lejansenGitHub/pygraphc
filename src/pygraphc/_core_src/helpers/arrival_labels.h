/* Arrival labels: the second distances an undirected search keeps under a negative terminal weight. */
#ifndef PYGRAPHC_CORE_ARRIVAL_LABELS_H
#define PYGRAPHC_CORE_ARRIVAL_LABELS_H

#include "../models.h"
#include "min_heap.h"

/*
 * A pass goes from one terminal of an edge to another. Under a negative terminal
 * weight an undirected search must also forbid passing edge E again from its
 * negative terminal n right after leaving E through n (that would count w_n
 * twice). Both rules need one second label per search state:
 * - a node keeps its best distance, tagged with the edge whose negative terminal
 *   it arrived through (or -1), and the best distance with another tag in
 *   state second_node_states + node; settled at its best label it relaxes
 *   everything except the pass its tag forbids, its second state only that
 *   pass;
 * - an edge state keeps its cheapest entry and that entry's terminal, and the
 *   cheapest entry through another terminal in state second_edge_states + e;
 *   settled at its best entry it leaves through every terminal but the entry
 *   terminal, its second state only through that one.
 * Without a negative terminal weight no search uses labels: a U-turn then costs
 * at least 0 and never shortens a path.
 */
typedef struct LabelledMeeting LabelledMeeting;

typedef struct ArrivalLabels {
    int *arrival_edges;       /* node_count: tag of a node's best label; its second label's tag always differs */
    int *entry_terminals;         /* edge_count: entry terminal of an edge state's best label; its second label's differs */
    int *predecessors;        /* state count, or NULL: the state each label was reached from */
    int second_node_states;   /* search state of node 0's second label */
    int second_edge_states;   /* search state of edge 0's second label */
    LabelledMeeting *meeting; /* one side of a point-to-point search, or NULL */
} ArrivalLabels;

/* The cheapest meeting of the two sides of a point-to-point search: the label each side's part ends at. */
typedef struct {
    double weight;
    int forward_state;
    int backward_state;
} PathMeeting;

/*
 * One side's view of a point-to-point search. D21's rules read the same on a
 * reversed path, so the backward side is the same search run from the target:
 * its tag at a node names the edge whose negative terminal the rest of the path
 * enters first, its edge state's key the terminal the path leaves through.
 */
struct LabelledMeeting {
    const MinHeap *far_heap;
    const ArrivalLabels *far_labels;
    PathMeeting *best;
    int is_backward;
};

/*
 * Allocate the labels of a search whose other states number plain_state_count
 * (nodes and edge states): the second labels follow them. Returns the total
 * state count, or -1 with MemoryError set.
 */
static inline int allocate_arrival_labels(ArrivalLabels *labels, int node_count, int edge_count, int plain_state_count,
                                          int with_predecessors) {
    int state_count = plain_state_count + node_count + edge_count;
    labels->arrival_edges = (int *)malloc((size_t)node_count * sizeof(int) + 1);
    labels->entry_terminals = (int *)malloc((size_t)edge_count * sizeof(int) + 1);
    labels->predecessors = with_predecessors ? (int *)malloc((size_t)state_count * sizeof(int)) : NULL;
    labels->second_node_states = plain_state_count;
    labels->second_edge_states = plain_state_count + node_count;
    labels->meeting = NULL;
    if (!labels->arrival_edges || !labels->entry_terminals || (with_predecessors && !labels->predecessors)) {
        free(labels->arrival_edges); free(labels->entry_terminals); free(labels->predecessors);
        labels->arrival_edges = labels->entry_terminals = labels->predecessors = NULL;
        PyErr_NoMemory();
        return -1;
    }
    memset(labels->arrival_edges, -1, (size_t)node_count * sizeof(int));
    memset(labels->entry_terminals, -1, (size_t)edge_count * sizeof(int));
    if (with_predecessors) memset(labels->predecessors, -1, (size_t)state_count * sizeof(int));
    return state_count;
}

static inline void free_arrival_labels(ArrivalLabels *labels) {
    free(labels->arrival_edges); free(labels->entry_terminals); free(labels->predecessors);
    labels->arrival_edges = labels->entry_terminals = labels->predecessors = NULL;
}

/* Lower a state's distance in the heap; the caller has checked that it decreases. */
static inline __attribute__((always_inline)) void lower_state_distance(MinHeap *min_heap, int state, double new_distance) {
    min_heap->distance[state] = new_distance;
    if (min_heap->heap_position[state] == -1) {
        int heap_slot = min_heap->size++;
        min_heap->nodes[heap_slot] = state;
        min_heap->heap_position[state] = heap_slot;
        min_heap_sift_up(min_heap, heap_slot);
    } else {
        min_heap_sift_up(min_heap, min_heap->heap_position[state]);
    }
}

/*
 * Offer the far side a meeting: the near search reaches `state` (second label
 * second_state) from `from` at candidate_distance with key `key`. The far label
 * that may continue is its best one unless the keys conflict (equal and not -1:
 * a U-turn, or a second pass right at a negative terminal), else its second one;
 * only a settled far label counts, so the meeting's chains never change afterwards.
 */
static inline __attribute__((always_inline)) void record_meeting(LabelledMeeting *meeting, int far_best_key, int state,
                                                                 int second_state, int key, double candidate_distance, int from) {
    int far_state = (key >= 0 && key == far_best_key) ? second_state : state;
    const MinHeap *far_heap = meeting->far_heap;
    double weight = candidate_distance + far_heap->distance[far_state];
    if (far_heap->heap_position[far_state] != -1 || weight >= meeting->best->weight) return;
    meeting->best->weight = weight;
    meeting->best->forward_state = meeting->is_backward ? far_state : from;
    meeting->best->backward_state = meeting->is_backward ? from : far_state;
}

/*
 * Relax a two-label state reached from state `from` at candidate_distance
 * with key `key` (a node's arrival tag, an edge state's entry terminal), keeping
 * the second label the best one whose key differs from the best label's.
 * best_key points at the best label's key; second_state is the state of the
 * second label.
 */
static inline __attribute__((always_inline)) void relax_labeled(MinHeap *min_heap, int *best_key, int *predecessors, int state, int second_state,
                                 double candidate_distance, int key, int from) {
    double *distance = min_heap->distance;
    if (candidate_distance < distance[state]) {
        if (*best_key != key && distance[state] < HUGE_VAL) {
            if (distance[state] < distance[second_state]) lower_state_distance(min_heap, second_state, distance[state]);
            if (predecessors) predecessors[second_state] = predecessors[state];
        }
        *best_key = key;
        if (predecessors) predecessors[state] = from;
        lower_state_distance(min_heap, state, candidate_distance);
    } else if (key != *best_key && candidate_distance < distance[second_state]) {
        if (predecessors) predecessors[second_state] = from;
        lower_state_distance(min_heap, second_state, candidate_distance);
    }
}

#endif
