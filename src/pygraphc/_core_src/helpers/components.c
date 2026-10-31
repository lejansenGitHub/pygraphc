#include "components.h"
#include "ports.h"
#include "union_find.h"

/*
 * Union-find over the edges and sequential component ids per node. Two-port
 * edges are unioned from the edge list under two_port_edge_mask; with
 * has_multiport_edges, every edge with other than two ports also joins its
 * active ports (port_is_active). The function is instantiated for both values
 * of has_multiport_edges, so graphs of two-port edges run the loop as before.
 */
static inline __attribute__((always_inline)) int components_of(
    int node_count, EdgeList *edges, const uint8_t *two_port_edge_mask, const uint8_t *node_mask,
    const PortList *ports, const uint8_t *edge_mask, const uint8_t *port_mask, ComponentResult *components,
    const int has_multiport_edges) {
    UnionFind union_find;
    if (union_find_init(&union_find, node_count) < 0) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        if (two_port_edge_mask && two_port_edge_mask[edge_index]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index), second_node = EDGE_SECOND_ENDPOINT(edges, edge_index);
        if (node_mask && (node_mask[first_node] || node_mask[second_node])) continue;
        union_find_union(&union_find, first_node, second_node);
    }
    if (has_multiport_edges) {
        for (Py_ssize_t edge = 0; edge < edges->edge_count; edge++) {
            int first_port = ports->port_offsets[edge], end_port = ports->port_offsets[edge + 1];
            if (end_port - first_port == 2) continue;
            int first_active_node = -1;
            for (int port = first_port; port < end_port; port++) {
                if (!port_is_active(ports, (int)edge, port, edge_mask, port_mask, node_mask)) continue;
                if (first_active_node < 0) first_active_node = ports->port_nodes[port];
                else union_find_union(&union_find, first_active_node, ports->port_nodes[port]);
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

int compute_graph_components(GraphContext *graph, const uint8_t *edge_mask, const PortMask *port_mask,
                             const uint8_t *node_mask, ComponentResult *components) {
    int node_count = graph->node_ids.node_count;
    EdgeList *edges = &graph->node_ids.edges;
    const uint8_t *traversal_edge_mask = two_port_edge_mask(port_mask, edge_mask);
    return graph->ports
        ? components_of(node_count, edges, traversal_edge_mask, node_mask, graph->ports, edge_mask, port_mask->ports,
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
int compute_strongly_connected_components_masked(int node_count, AdjacencyList *adjacency, const uint8_t *edge_mask,
                                                 const uint8_t *node_mask, ComponentResult *components) {
    /* 6 * node_count ints: discovery_time, low_link, on_stack, tarjan_stack, dfs_stack, stack_next_slot */
    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) { PyErr_NoMemory(); return -1; }
    int *discovery_time  = scratch;
    int *low_link        = scratch + node_count;
    int *on_stack        = scratch + 2 * node_count;
    int *tarjan_stack    = scratch + 3 * node_count;
    int *dfs_stack       = scratch + 4 * node_count;
    int *stack_next_slot = scratch + 5 * node_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));
    memset(on_stack, 0, (size_t)node_count * sizeof(int));

    components->labels = (int *)malloc((size_t)node_count * sizeof(int));
    if (!components->labels) { free(scratch); PyErr_NoMemory(); return -1; }
    memset(components->labels, -1, (size_t)node_count * sizeof(int));

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
            int node = dfs_stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                int neighbor = adjacency->neighbors[slot];
                if (node_mask && node_mask[neighbor]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++;
                    on_stack[neighbor] = 1;
                    tarjan_stack[tarjan_stack_top++] = neighbor;
                    dfs_stack[++stack_top] = neighbor;
                    stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else if (on_stack[neighbor]) {
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                /* Backtrack: propagate low_link to parent */
                if (stack_top > 0) {
                    int parent = dfs_stack[stack_top - 1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                }
                /* If node is a root of an SCC, pop the Tarjan stack */
                if (low_link[node] == discovery_time[node]) {
                    int component_id = component_count++;
                    int popped_node;
                    do {
                        popped_node = tarjan_stack[--tarjan_stack_top];
                        on_stack[popped_node] = 0;
                        components->labels[popped_node] = component_id;
                    } while (popped_node != node);
                }
                stack_top--;
            }
        }
    }

    /* Assign isolated masked-out nodes to their own components */
    for (int node_index = 0; node_index < node_count; node_index++) {
        if (components->labels[node_index] == -1)
            components->labels[node_index] = component_count++;
    }

    components->component_count = component_count;
    free(scratch);
    return 0;
}
