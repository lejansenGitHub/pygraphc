/* Iterative Tarjan over the search states of a graph: critical edges, critical nodes and blocks. */
#include "biconnectivity.h"

void release_tarjan_report(TarjanReport *report) {
    free(report->critical_edges);
    free(report->critical_nodes);
    free(report->block_offsets);
    free(report->block_states);
    memset(report, 0, sizeof *report);
}

/*
 * The arrays of a mode, sized for the most a search can report: every edge once,
 * a block per finished state, and two states per link popped. Returns 0, or -1
 * with nothing allocated.
 */
static int allocate_report(TarjanReport *report, TarjanMode mode, int node_count, int edge_count, size_t state_count,
                           size_t link_capacity) {
    memset(report, 0, sizeof *report);
    if (mode == TARJAN_CRITICAL_EDGES) {
        report->critical_edges = (int *)malloc((size_t)(edge_count > 0 ? edge_count : 1) * sizeof(int));
        if (report->critical_edges) return 0;
    } else if (mode == TARJAN_CRITICAL_NODES) {
        report->critical_nodes = (uint8_t *)calloc((size_t)(node_count > 0 ? node_count : 1), 1);
        if (report->critical_nodes) return 0;
    } else {
        report->block_offsets = (int *)malloc((state_count + 1) * sizeof(int));
        report->block_states = (int *)malloc((link_capacity > 0 ? 2 * link_capacity : 1) * sizeof(int));
        if (report->block_offsets && report->block_states) {
            report->block_offsets[0] = 0;
            return 0;
        }
    }
    release_tarjan_report(report);
    return -1;
}

/*
 * Pop the links of the block below (parent, state) and append its distinct states,
 * edge states only with keeps_edge_states. A block of fewer than minimum_nodes
 * nodes is dropped. block_stamp marks the states already in this closure by its
 * number, which differs per closure even when a block is dropped.
 */
static inline __attribute__((always_inline)) void close_block(
    TarjanReport *report, const int *link_from, const int *link_to, int *link_top, int parent, int state, int node_count,
    int *block_stamp, int closure, const int keeps_edge_states, const int minimum_nodes) {
    int end = report->block_offsets[report->block_count], block_nodes = 0;
    while (*link_top > 0) {
        (*link_top)--;
        int from = link_from[*link_top], to = link_to[*link_top];
        int ends[2] = {from, to};
        for (int side = 0; side < 2; side++) {
            int member = ends[side];
            if ((!keeps_edge_states && member >= node_count) || block_stamp[member] == closure) continue;
            block_stamp[member] = closure;
            report->block_states[end++] = member;
            if (member < node_count) block_nodes++;
        }
        if (from == parent && to == state) break;
    }
    if (block_nodes >= minimum_nodes) report->block_offsets[++report->block_count] = end;
}

/*
 * One search in one mode, instantiated for every mode and both values of
 * has_multi_terminal_edges, so graphs of two-terminal edges run it without the
 * edge-state branches.
 */
static inline __attribute__((always_inline)) int tarjan_search(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, TarjanReport *report, const TarjanMode mode, const int has_multi_terminal_edges) {
    const int *slot_offsets = graph->adjacency.offset, *slot_neighbors = graph->adjacency.neighbors;
    const int *slot_edge_indices = graph->adjacency.edge_indices;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    int edge_count = (int)graph->node_ids.edges.edge_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)edge_count : 0);
    size_t link_capacity = (size_t)edge_count + (has_multi_terminal_edges ? (size_t)terminals->terminal_count : 0);

    const int collects_links = mode == TARJAN_BLOCKS || mode == TARJAN_SIMPLE_PATH_BLOCKS;
    const int marks_edge_states = has_multi_terminal_edges && mode == TARJAN_CRITICAL_EDGES;
    int *scratch = (int *)malloc((collects_links ? 7 : 6) * state_count * sizeof(int));
    int *link_stack = collects_links ? (int *)malloc(2 * (link_capacity > 0 ? link_capacity : 1) * sizeof(int)) : NULL;
    uint8_t *critical_edge_states = marks_edge_states ? (uint8_t *)calloc((size_t)edge_count + 1, 1) : NULL;
    if (!scratch || (collects_links && !link_stack) || (marks_edge_states && !critical_edge_states)
        || allocate_report(report, mode, node_count, edge_count, state_count, link_capacity) < 0) {
        free(scratch); free(link_stack); free(critical_edge_states);
        PyErr_NoMemory();
        return -1;
    }
    int *discovery_time = scratch, *low_link = scratch + state_count, *stack = scratch + 2 * state_count;
    int *stack_next_slot = scratch + 3 * state_count, *parent_link = scratch + 4 * state_count;
    int *child_count = scratch + 5 * state_count;
    int *block_stamp = collects_links ? scratch + 6 * state_count : NULL;
    int *link_from = link_stack, *link_to = link_stack ? link_stack + link_capacity : NULL;
    memset(discovery_time, -1, state_count * sizeof(int));
    if (collects_links) memset(block_stamp, -1, state_count * sizeof(int));
    const int counts_children = mode == TARJAN_CRITICAL_NODES || has_multi_terminal_edges;
    int timer = 0, link_top = 0, closure = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = slot_offsets[start];
        discovery_time[start] = low_link[start] = timer++; parent_link[start] = -1;
        if (counts_children) child_count[start] = 0;
        while (stack_top >= 0) {
            int state = stack[stack_top];
            int is_edge_state = has_multi_terminal_edges && state >= node_count;
            int end_slot = is_edge_state ? terminals->terminal_offsets[state - node_count + 1] : slot_offsets[state + 1];
            if (stack_next_slot[stack_top] < end_slot) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor, link;
                if (is_edge_state) {
                    int edge = state - node_count;
                    if (!terminal_is_active(terminals, edge, slot, edge_mask, terminal_mask, node_mask)) continue;
                    neighbor = terminals->terminal_nodes[slot];
                    link = edge_count + slot;
                } else {
                    neighbor = slot_neighbors[slot];
                    int edge_index = slot_edge_indices[slot];
                    if (has_multi_terminal_edges && neighbor < 0) {
                        int edge = -1 - neighbor;
                        if (!terminal_is_active(terminals, edge, edge_index, edge_mask, terminal_mask, node_mask)) continue;
                        neighbor = node_count + edge;
                        link = edge_count + edge_index;
                    } else {
                        if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
                        if (node_mask && node_mask[neighbor]) continue;
                        link = edge_index;
                    }
                }
                if (link == parent_link[state]) continue;
                if (discovery_time[neighbor] == -1) {
                    if (collects_links) { link_from[link_top] = state; link_to[link_top] = neighbor; link_top++; }
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_link[neighbor] = link;
                    if (counts_children) { child_count[neighbor] = 0; if (!has_multi_terminal_edges) child_count[state]++; }
                    stack[++stack_top] = neighbor;
                    stack_next_slot[stack_top] = (has_multi_terminal_edges && neighbor >= node_count)
                        ? terminals->terminal_offsets[neighbor - node_count] : slot_offsets[neighbor];
                } else if (!collects_links || discovery_time[neighbor] < discovery_time[state]) {
                    if (collects_links) { link_from[link_top] = state; link_to[link_top] = neighbor; link_top++; }
                    if (low_link[state] > discovery_time[neighbor]) low_link[state] = discovery_time[neighbor];
                }
                continue;
            }
            if (stack_top == 0) {
                if (mode == TARJAN_CRITICAL_NODES && child_count[state] >= 2) report->critical_nodes[state] = 1;
                stack_top--;
                continue;
            }
            int parent = stack[stack_top - 1];
            int parent_is_edge_state = has_multi_terminal_edges && parent >= node_count;
            int subtree_holds_node = !is_edge_state || child_count[state] > 0;
            if (low_link[parent] > low_link[state]) low_link[parent] = low_link[state];
            if (has_multi_terminal_edges && subtree_holds_node) child_count[parent]++;
            if (mode == TARJAN_CRITICAL_EDGES) {
                if (!is_edge_state && !parent_is_edge_state && low_link[state] > discovery_time[parent])
                    report->critical_edges[report->critical_edge_count++] = parent_link[state];
                if (marks_edge_states) {
                    if (parent_is_edge_state && low_link[state] >= discovery_time[parent]) critical_edge_states[parent - node_count] = 1;
                    if (is_edge_state && critical_edge_states[state - node_count])
                        report->critical_edges[report->critical_edge_count++] = state - node_count;
                }
            } else if (mode == TARJAN_CRITICAL_NODES) {
                if (!parent_is_edge_state && parent_link[parent] != -1 && subtree_holds_node &&
                    low_link[state] >= discovery_time[parent])
                    report->critical_nodes[parent] = 1;
            } else if (mode == TARJAN_SIMPLE_PATH_BLOCKS && has_multi_terminal_edges) {
                if (low_link[state] >= discovery_time[parent])
                    close_block(report, link_from, link_to, &link_top, parent, state, node_count, block_stamp, closure++, 1, 0);
            } else if (!parent_is_edge_state && low_link[state] >= discovery_time[parent]) {
                close_block(report, link_from, link_to, &link_top, parent, state, node_count, block_stamp, closure++, 0,
                            mode == TARJAN_BLOCKS && has_multi_terminal_edges ? 2 : 0);
            }
            stack_top--;
        }
    }
    free(scratch); free(link_stack); free(critical_edge_states);
    return 0;
}

int tarjan_report(const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
                  const uint8_t *terminal_mask, const uint8_t *node_mask, TarjanMode mode, TarjanReport *report) {
    memset(report, 0, sizeof *report);
    switch (mode) {
    case TARJAN_CRITICAL_EDGES:
        return graph->terminals
            ? tarjan_search(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, report, TARJAN_CRITICAL_EDGES, 1)
            : tarjan_search(graph, two_terminal_edge_mask, NULL, NULL, node_mask, report, TARJAN_CRITICAL_EDGES, 0);
    case TARJAN_CRITICAL_NODES:
        return graph->terminals
            ? tarjan_search(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, report, TARJAN_CRITICAL_NODES, 1)
            : tarjan_search(graph, two_terminal_edge_mask, NULL, NULL, node_mask, report, TARJAN_CRITICAL_NODES, 0);
    case TARJAN_SIMPLE_PATH_BLOCKS:
        return graph->terminals
            ? tarjan_search(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, report, TARJAN_SIMPLE_PATH_BLOCKS, 1)
            : tarjan_search(graph, two_terminal_edge_mask, NULL, NULL, node_mask, report, TARJAN_SIMPLE_PATH_BLOCKS, 0);
    default:
        return graph->terminals
            ? tarjan_search(graph, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, report, TARJAN_BLOCKS, 1)
            : tarjan_search(graph, two_terminal_edge_mask, NULL, NULL, node_mask, report, TARJAN_BLOCKS, 0);
    }
}

int biconnected_edge_labels(const GraphContext *graph, const uint8_t *edge_mask, const uint8_t *node_mask, int32_t *labels) {
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    memset(labels, -1, (size_t)edge_count * sizeof(int32_t));
    if (!graph->has_adjacency) return 0;
    const AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    int *edge_stack = (int *)malloc((size_t)edge_count * sizeof(int));
    if (!scratch || !edge_stack) { free(scratch); free(edge_stack); PyErr_NoMemory(); return -1; }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));
    int timer = 0, edge_stack_top = 0, component_id = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    edge_stack[edge_stack_top++] = edge_index;
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else if (discovery_time[neighbor] < discovery_time[node]) {
                    edge_stack[edge_stack_top++] = edge_index;
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] >= discovery_time[parent]) {
                        while (edge_stack_top > 0) {
                            int popped_edge = edge_stack[--edge_stack_top];
                            labels[popped_edge] = (int32_t)component_id;
                            if (popped_edge == parent_edge[node]) break;
                        }
                        component_id++;
                    }
                }
                stack_top--;
            }
        }
    }
    free(scratch); free(edge_stack);
    return 0;
}
