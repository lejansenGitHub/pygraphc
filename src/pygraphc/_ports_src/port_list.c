/* Edges with ports: node incidence and components over active ports. */
#include "port_list.h"
#include "../_core_src/helpers/union_find.h"

int build_port_incidence(int node_count, PortList *ports) {
    ports->incidence_offsets = (int *)calloc((size_t)node_count + 1, sizeof(int));
    ports->incidence_ports = (int *)malloc((size_t)(ports->port_count > 0 ? ports->port_count : 1) * sizeof(int));
    if (!ports->incidence_offsets || !ports->incidence_ports) { PyErr_NoMemory(); return -1; }
    for (int port = 0; port < ports->port_count; port++) ports->incidence_offsets[ports->port_nodes[port] + 1]++;
    for (int node = 0; node < node_count; node++) ports->incidence_offsets[node + 1] += ports->incidence_offsets[node];
    int *next_slot = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    if (!next_slot) { PyErr_NoMemory(); return -1; }
    memcpy(next_slot, ports->incidence_offsets, (size_t)node_count * sizeof(int));
    for (int port = 0; port < ports->port_count; port++)
        ports->incidence_ports[next_slot[ports->port_nodes[port]]++] = port;
    free(next_slot);
    return 0;
}

int port_component_labels(int node_count, const PortList *ports, const uint8_t *edge_mask,
                          const uint8_t *port_mask, const uint8_t *node_mask, int *labels) {
    UnionFind union_find;
    if (union_find_init(&union_find, node_count) < 0) { PyErr_NoMemory(); return -1; }
    for (int edge = 0; edge < ports->edge_count; edge++) {
        int first_active_node = -1;
        for (int port = ports->port_offsets[edge]; port < ports->port_offsets[edge + 1]; port++) {
            if (!port_is_active(ports, port, edge_mask, port_mask, node_mask)) continue;
            int node = ports->port_nodes[port];
            if (first_active_node < 0) first_active_node = node;
            else union_find_union(&union_find, first_active_node, node);
        }
    }
    int *smallest_node_by_root = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    if (!smallest_node_by_root) { union_find_free(&union_find); PyErr_NoMemory(); return -1; }
    memset(smallest_node_by_root, -1, (size_t)node_count * sizeof(int));
    for (int node = 0; node < node_count; node++) {
        if (node_mask && node_mask[node]) { labels[node] = -1; continue; }
        int root = union_find_find(&union_find, node);
        if (smallest_node_by_root[root] < 0) smallest_node_by_root[root] = node;
        labels[node] = smallest_node_by_root[root];
    }
    free(smallest_node_by_root);
    union_find_free(&union_find);
    return 0;
}
