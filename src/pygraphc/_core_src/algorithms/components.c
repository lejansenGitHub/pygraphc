#include "components.h"
#include "terminals.h"
#include "union_find.h"

/*
 * Union-find over the edges and sequential component ids per node. Two-terminal
 * edges are unioned from the edge list under two_terminal_edge_mask; with
 * has_multi_terminal_edges, every edge with other than two terminals also joins its
 * active terminals (terminal_is_active). The function is instantiated for both values
 * of has_multi_terminal_edges, so graphs of two-terminal edges run the loop as before.
 */
static inline __attribute__((always_inline)) int components_of(
    int node_count, EdgeList *edges, const uint8_t *two_terminal_edge_mask, const uint8_t *node_mask,
    const TerminalList *terminals, const uint8_t *edge_mask, const uint8_t *terminal_mask, ComponentResult *components,
    const int has_multi_terminal_edges) {
    UnionFind union_find;
    if (union_find_init(&union_find, node_count) < 0) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index), second_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
        if (node_mask && (node_mask[first_node] || node_mask[second_node])) continue;
        union_find_union(&union_find, first_node, second_node);
    }
    if (has_multi_terminal_edges) {
        for (Py_ssize_t edge = 0; edge < edges->edge_count; edge++) {
            int first_terminal = terminals->terminal_offsets[edge], end_terminal = terminals->terminal_offsets[edge + 1];
            if (end_terminal - first_terminal == 2) continue;
            int first_active_node = -1;
            for (int terminal = first_terminal; terminal < end_terminal; terminal++) {
                if (!terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask)) continue;
                if (first_active_node < 0) first_active_node = terminals->terminal_nodes[terminal];
                else union_find_union(&union_find, first_active_node, terminals->terminal_nodes[terminal]);
            }
        }
    }

    /* labels + component_id_by_root in one allocation */
    int *buffer = (int *)malloc(2 * (size_t)node_count * sizeof(int));
    if (!buffer) {
        union_find_free(&union_find);
        PyErr_NoMemory();
        return -1;
    }
    components->labels = buffer;
    int *component_id_by_root = buffer + node_count;
    memset(component_id_by_root, -1, (size_t)node_count * sizeof(int));

    int component_count = 0;
    for (int node = 0; node < node_count; node++) {
        int root = union_find_find(&union_find, node);
        if (component_id_by_root[root] == -1)
            component_id_by_root[root] = component_count++;
        components->labels[node] = component_id_by_root[root];
    }
    components->component_count = component_count;

    union_find_free(&union_find);
    return 0;
}

/*
 * Core routine: union-find over the edge list, sequential component IDs.
 * edge_mask: if non-NULL, skip edges where edge_mask[edge_index] != 0.
 * Caller must free result.labels.
 * Returns 0 on success, -1 on error.
 */
int compute_components_masked(int node_count, EdgeList *edges, const uint8_t *edge_mask,
                              const uint8_t *node_mask,
                              ComponentResult *components) {
    return components_of(node_count, edges, edge_mask, node_mask, NULL, NULL, NULL, components, 0);
}

int compute_graph_components(GraphContext *graph, const uint8_t *edge_mask, const TerminalMask *terminal_mask,
                             const uint8_t *node_mask, ComponentResult *components) {
    int node_count = graph->node_ids.node_count;
    EdgeList *edges = &graph->node_ids.edges;
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(terminal_mask, edge_mask);
    return graph->terminals
        ? components_of(node_count, edges, traversal_edge_mask, node_mask, graph->terminals, edge_mask, terminal_mask->terminals,
                        components, 1)
        : components_of(node_count, edges, traversal_edge_mask, node_mask, NULL, NULL, NULL, components, 0);
}

/* ── Strongly connected components (iterative Tarjan's, directed CSR) ── */

/*
 * Core routine: iterative Tarjan's SCC on a directed graph.
 * Uses the forward CSR (adjacency->offset/neighbors/edge_indices).
 * Supports edge mask and node mask.
 * Writes component labels into components (same shape as compute_components_masked).
 * Returns 0 on success, -1 on error.
 */
/*
 * Tarjan's strongly connected components over the search states of a directed
 * graph: its nodes and, with has_multi_terminal_edges, one state per edge with
 * three or more terminals (node_count + e), entered from the node of each
 * active in-terminal and leaving to the node of each active out-terminal (OR
 * semantics, D5). Labels cover every state; an edge state shares the label of
 * the nodes it lies on a cycle with. Instantiated for both values of
 * has_multi_terminal_edges, so graphs of two-terminal edges run the loop as
 * before.
 */
static inline __attribute__((always_inline)) int strongly_connected_states(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, ComponentResult *components,
    const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)graph->node_ids.edges.edge_count : 0);
    /* 6 * state_count ints: discovery_time, low_link, on_stack, tarjan_stack, dfs_stack, stack_next_slot */
    int *scratch = (int *)malloc(6 * state_count * sizeof(int));
    components->labels = (int *)malloc(state_count * sizeof(int));
    if (!scratch || !components->labels) { free(scratch); free(components->labels); PyErr_NoMemory(); return -1; }
    int *discovery_time = scratch, *low_link = scratch + state_count, *on_stack = scratch + 2 * state_count;
    int *tarjan_stack = scratch + 3 * state_count, *dfs_stack = scratch + 4 * state_count;
    int *stack_next_slot = scratch + 5 * state_count;
    memset(discovery_time, -1, state_count * sizeof(int));
    memset(on_stack, 0, state_count * sizeof(int));
    memset(components->labels, -1, state_count * sizeof(int));

    int timer = 0, tarjan_stack_top = 0, component_count = 0;
    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;

        int stack_top = 0;
        dfs_stack[0] = start;
        stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++;
        on_stack[start] = 1;
        tarjan_stack[tarjan_stack_top++] = start;

        while (stack_top >= 0) {
            int state = dfs_stack[stack_top];
            int is_edge_state = has_multi_terminal_edges && state >= node_count;
            int end_slot = is_edge_state ? terminals->terminal_offsets[state - node_count + 1] : adjacency->offset[state + 1];
            if (stack_next_slot[stack_top] < end_slot) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor;
                if (is_edge_state) {
                    int edge = state - node_count;
                    if (!terminal_is_active(terminals, edge, slot, edge_mask, terminal_mask, node_mask)) continue;
                    neighbor = terminals->terminal_nodes[slot];
                } else {
                    neighbor = adjacency->neighbors[slot];
                    int edge_index = adjacency->edge_indices[slot];
                    if (has_multi_terminal_edges && neighbor < 0) {
                        int edge = -1 - neighbor;
                        if (!terminal_is_active(terminals, edge, edge_index, edge_mask, terminal_mask, node_mask)) continue;
                        neighbor = node_count + edge;
                    } else {
                        if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
                        if (node_mask && node_mask[neighbor]) continue;
                    }
                }
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++;
                    on_stack[neighbor] = 1;
                    tarjan_stack[tarjan_stack_top++] = neighbor;
                    dfs_stack[++stack_top] = neighbor;
                    stack_next_slot[stack_top] = (has_multi_terminal_edges && neighbor >= node_count)
                        ? terminals->out_terminal_offsets[neighbor - node_count] : adjacency->offset[neighbor];
                } else if (on_stack[neighbor]) {
                    if (low_link[state] > discovery_time[neighbor]) low_link[state] = discovery_time[neighbor];
                }
            } else {
                /* Backtrack: propagate low_link to parent */
                if (stack_top > 0) {
                    int parent = dfs_stack[stack_top - 1];
                    if (low_link[parent] > low_link[state]) low_link[parent] = low_link[state];
                }
                /* If state is a root of an SCC, pop the Tarjan stack */
                if (low_link[state] == discovery_time[state]) {
                    int component_id = component_count++;
                    int popped_state;
                    do {
                        popped_state = tarjan_stack[--tarjan_stack_top];
                        on_stack[popped_state] = 0;
                        components->labels[popped_state] = component_id;
                    } while (popped_state != state);
                }
                stack_top--;
            }
        }
    }

    /* Masked nodes get components of their own */
    for (int node = 0; node < node_count; node++) {
        if (components->labels[node] == -1)
            components->labels[node] = component_count++;
    }

    components->component_count = component_count;
    free(scratch);
    return 0;
}

int compute_strongly_connected_components(const GraphContext *graph, const uint8_t *edge_mask,
                                          const TerminalMask *terminal_mask, const uint8_t *node_mask,
                                          ComponentResult *components) {
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(terminal_mask, edge_mask);
    return graph->terminals
        ? strongly_connected_states(graph, traversal_edge_mask, edge_mask, terminal_mask->terminals, node_mask, components, 1)
        : strongly_connected_states(graph, traversal_edge_mask, NULL, NULL, node_mask, components, 0);
}
