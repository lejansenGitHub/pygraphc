/* Per-node and per-edge labels written into int32 arrays: component labels, quotient edges, degrees. */
#include "labels.h"
#include "terminals.h"

int smallest_node_labels(const ComponentResult *components, int node_count, const uint8_t *node_mask, int32_t *labels) {
    int *smallest_index = (int *)malloc((size_t)(components->component_count > 0 ? components->component_count : 1) * sizeof(int));
    if (!smallest_index) { PyErr_NoMemory(); return -1; }
    memset(smallest_index, -1, (size_t)components->component_count * sizeof(int));
    for (int node = 0; node < node_count; node++) {
        int component = components->labels[node];
        if (smallest_index[component] == -1) smallest_index[component] = node;
        labels[node] = (node_mask && node_mask[node]) ? -1 : (int32_t)smallest_index[component];
    }
    free(smallest_index);
    return 0;
}

void split_quotient_edges(const EdgeList *edges, const int32_t *labels, const uint8_t *edge_mask, int32_t *from_labels,
                          int32_t *to_labels, int32_t *crossing_edges, int32_t *internal_edges, Py_ssize_t *crossing_count,
                          Py_ssize_t *internal_count) {
    Py_ssize_t crossing = 0, internal = 0;
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        if (edge_mask && edge_mask[edge_index]) continue;
        int32_t from_label = labels[EDGE_FIRST_ENDPOINT(edges, edge_index)];
        int32_t to_label = labels[EDGE_SECOND_ENDPOINT(edges, edge_index)];
        if (from_label < 0 || to_label < 0) continue;
        if (from_label == to_label) { internal_edges[internal++] = (int32_t)edge_index; continue; }
        from_labels[crossing] = from_label;
        to_labels[crossing] = to_label;
        crossing_edges[crossing++] = (int32_t)edge_index;
    }
    *crossing_count = crossing;
    *internal_count = internal;
}

/*
 * The degree count, instantiated for both values of has_terminal_list. A
 * two-terminal edge without a terminal list has terminals 2e and 2e + 1 at its
 * endpoints, a masked one folded into two_terminal_edges.
 */
static inline __attribute__((always_inline)) void add_joined_terminals(
    const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, int32_t *degrees, const int has_terminal_list) {
    const TerminalList *terminals = graph->terminals;
    const EdgeList *edges = &graph->node_ids.edges;
    for (Py_ssize_t edge = 0; edge < edges->edge_count; edge++) {
        if (!has_terminal_list) {
            if (two_terminal_edges && two_terminal_edges[edge]) continue;
            int first_node = EDGE_FIRST_ENDPOINT(edges, edge), second_node = EDGE_SECOND_ENDPOINT(edges, edge);
            if (node_mask && (node_mask[first_node] || node_mask[second_node])) continue;
            degrees[first_node]++;
            if (!graph->directed) degrees[second_node]++;
            continue;
        }
        int first = terminals->terminal_offsets[edge], end = terminals->terminal_offsets[edge + 1];
        int out_start = graph->directed ? terminals->out_terminal_offsets[edge] : end;
        int active_in = 0, active_out = 0;
        for (int terminal = first; terminal < end; terminal++) {
            if (!terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask)) continue;
            if (terminal < out_start) active_in++;
            else active_out++;
        }
        if (graph->directed ? (active_in == 0 || active_out == 0) : active_in < 2) continue;
        for (int terminal = first; terminal < out_start; terminal++)
            if (terminal_is_active(terminals, (int)edge, terminal, edge_mask, terminal_mask, node_mask))
                degrees[terminals->terminal_nodes[terminal]]++;
    }
}

void joined_terminal_degrees(const GraphContext *graph, const uint8_t *two_terminal_edges, const uint8_t *edge_mask,
                             const uint8_t *terminal_mask, const uint8_t *node_mask, int32_t *degrees) {
    memset(degrees, 0, (size_t)graph->node_ids.node_count * sizeof(int32_t));
    if (graph->terminals)
        add_joined_terminals(graph, two_terminal_edges, edge_mask, terminal_mask, node_mask, degrees, 1);
    else
        add_joined_terminals(graph, two_terminal_edges, NULL, NULL, node_mask, degrees, 0);
}
