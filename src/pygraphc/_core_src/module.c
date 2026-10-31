#include "entry_points/entry_points.h"

/* ── Module definition ── */

static PyMethodDef methods[] = {
    {"connected_components", py_connected_components, METH_VARARGS,
     "connected_components(n, edges) -> list[set[int]]\n\n"
     "Find connected components using union-find with path compression.\n"
     "edges can be a list of 2-tuples or a numpy int32 array of shape (m, 2)."},
    {"connected_components_with_branches", py_connected_components_with_branches, METH_VARARGS,
     "connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]]\n\n"
     "Find connected components with associated branch IDs.\n"
     "edges can be a list of 2-tuples or a numpy int32 array of shape (m, 2)."},
    {"connected_components_remapped", py_connected_components_remapped, METH_VARARGS,
     "connected_components_remapped(node_ids, edges) -> list[set[NodeId]]\n\n"
     "Like connected_components but remaps indices to original node IDs.\n"
     "node_ids[i] is the original ID for internal index i."},
    {"connected_components_with_branches_remapped", py_connected_components_with_branches_remapped, METH_VARARGS,
     "connected_components_with_branches_remapped(node_ids, edges, branch_ids) -> list[tuple[set[NodeId], set[BranchId]]]\n\n"
     "Like connected_components_with_branches but remaps indices to original node IDs."},
    {"bridges", py_bridges, METH_VARARGS,
     "bridges(n, edges) -> list[tuple[int, int]]\n\n"
     "Find bridge edges using iterative Tarjan's algorithm."},
    {"articulation_points", py_articulation_points, METH_VARARGS,
     "articulation_points(n, edges) -> set[int]\n\n"
     "Find articulation points using iterative Tarjan's algorithm."},
    {"biconnected_components", py_biconnected_components, METH_VARARGS,
     "biconnected_components(n, edges) -> list[set[int]]\n\n"
     "Find biconnected components as node sets."},
    {"bfs", py_bfs, METH_VARARGS,
     "bfs(n, edges, source) -> list[int]\n\n"
     "BFS traversal from source, returns visited nodes in BFS order."},
    {"dijkstra", py_dijkstra, METH_VARARGS,
     "dijkstra(n, edges, weights, source, target) -> (float, list[int])\n\n"
     "Shortest path from source to target via Dijkstra with binary heap.\n"
     "Returns (distance, path). If unreachable, returns (inf, [])."},
    {"shortest_path_lengths", py_shortest_path_lengths, METH_VARARGS,
     "shortest_path_lengths(n, edges, weights, source, cutoff) -> dict[int, float]\n\n"
     "Single-source shortest path lengths. cutoff < 0 means no cutoff."},
    {"multi_source_dijkstra", py_multi_source_dijkstra, METH_VARARGS,
     "multi_source_dijkstra(n, edges, weights, sources, cutoff) -> dict[int, float]\n\n"
     "Multi-source shortest path lengths. cutoff < 0 means no cutoff."},
    {"translate_edges", py_translate_edges, METH_VARARGS,
     "translate_edges(node_ids, node_id_edges) -> list[tuple[int, int]]\n\n"
     "Translate edges from node-ID-based to index-based using a C hash map."},
    {"connected_components_by_node_id", py_connected_components_by_node_id, METH_VARARGS,
     "connected_components_by_node_id(node_ids, node_id_edges) -> list[set[NodeId]]\n\n"
     "Connected components with node-ID-based edges."},
    {"connected_components_by_node_id_split", py_connected_components_by_node_id_split, METH_VARARGS,
     "connected_components_by_node_id_split(node_ids, first_endpoints, second_endpoints) -> list[set[NodeId]]\n\n"
     "Connected components with separate endpoint sequences or 1D arrays."},
    {"bridges_by_node_id", py_bridges_by_node_id, METH_VARARGS,
     "bridges_by_node_id(node_ids, node_id_edges) -> list[tuple[NodeId, NodeId]]\n\n"
     "Bridges with node-ID-based edges."},
    {"articulation_points_by_node_id", py_articulation_points_by_node_id, METH_VARARGS,
     "articulation_points_by_node_id(node_ids, node_id_edges) -> set[NodeId]\n\n"
     "Articulation points with node-ID-based edges."},
    {"biconnected_components_by_node_id", py_biconnected_components_by_node_id, METH_VARARGS,
     "biconnected_components_by_node_id(node_ids, node_id_edges) -> list[set[NodeId]]\n\n"
     "Biconnected components with node-ID-based edges."},
    {"bfs_by_node_id", py_bfs_by_node_id, METH_VARARGS,
     "bfs_by_node_id(node_ids, node_id_edges, source_node_id) -> list[NodeId]\n\n"
     "BFS with node-ID-based edges and source."},
    {"dijkstra_by_node_id", py_dijkstra_by_node_id, METH_VARARGS,
     "dijkstra_by_node_id(node_ids, node_id_edges, weights, source_node_id, target_node_id) -> (float, list)\n\n"
     "Bidirectional Dijkstra with node-ID-based edges. Weights as a float64\n"
     "buffer are used in place; a sequence is converted and validated first."},
    {"shortest_path_lengths_by_node_id", py_shortest_path_lengths_by_node_id, METH_VARARGS,
     "shortest_path_lengths_by_node_id(node_ids, node_id_edges, weights, source_node_id, cutoff) -> dict[NodeId, float]\n\n"
     "SSSP lengths with node-ID-based edges."},
    {"multi_source_dijkstra_by_node_id", py_multi_source_dijkstra_by_node_id, METH_VARARGS,
     "multi_source_dijkstra_by_node_id(node_ids, node_id_edges, weights, source_node_ids, cutoff) -> dict[NodeId, float]\n\n"
     "Multi-source Dijkstra with node-ID-based edges."},
    {"parse_graph", py_parse_graph, METH_VARARGS,
     "parse_graph(node_ids, edges[, dst]) -> capsule\n\n"
     "Parse graph once, return opaque capsule for reuse across algorithms."},
    {"parse_port_graph", py_parse_port_graph, METH_VARARGS,
     "parse_port_graph(node_ids, edges[, directed]) -> capsule\n\n"
     "Parse a graph whose edges are tuples of ports; directed edges are (u, v) or (in_nodes, out_nodes)."},
    {"port_graph_shape", py_port_graph_shape, METH_VARARGS,
     "port_graph_shape(capsule) -> (node_count, edge_count, port_count) as stored in C."},
    {"port_offsets", py_port_offsets, METH_VARARGS,
     "port_offsets(capsule) -> bytes\n\nint32 offsets: the ports of edge e are offsets[e] .. offsets[e + 1] - 1."},
    {"port_nodes", py_port_nodes, METH_VARARGS,
     "port_nodes(capsule) -> bytes\n\nint32 node index of every port, in port order; edge e of a graph of "
     "two-port edges has ports 2e and 2e + 1."},
    {"graph_edge_count", py_graph_edge_count, METH_VARARGS,
     "graph_edge_count(capsule) -> int\n\nReturn edge count from parsed graph."},
    {"graph_node_count", py_graph_node_count, METH_VARARGS,
     "graph_node_count(capsule) -> int\n\nReturn node count from parsed graph."},
    {"connected_components_ctx", py_connected_components_ctx, METH_VARARGS, "Connected components from cached graph."},
    {"connected_components_with_branches_ctx", py_connected_components_with_branches_ctx, METH_VARARGS,
     "connected_components_with_branches_ctx(capsule, branch_ids[, excluded_edges, excluded_nodes])\n\n"
     "Connected components with branch IDs from cached graph.\n"
     "Excluded edges are dropped from connectivity and branch sets.\n"
     "Excluded nodes are removed from output node sets but edges still connect."},
    {"bridges_ctx", py_bridges_ctx, METH_VARARGS, "Bridges from cached graph."},
    {"articulation_points_ctx", py_articulation_points_ctx, METH_VARARGS, "Articulation points from cached graph."},
    {"biconnected_components_ctx", py_biconnected_components_ctx, METH_VARARGS, "Biconnected components from cached graph."},
    {"bfs_ctx", py_bfs_ctx, METH_VARARGS, "BFS from cached graph."},
    {"dijkstra_ctx", py_dijkstra_ctx, METH_VARARGS,
     "Bidirectional Dijkstra from cached graph. Weights as a float64 buffer are\n"
     "used in place; a sequence is converted and validated first."},
    {"shortest_path_lengths_ctx", py_shortest_path_lengths_ctx, METH_VARARGS, "SSSP lengths from cached graph."},
    {"multi_source_dijkstra_ctx", py_multi_source_dijkstra_ctx, METH_VARARGS, "Multi-source Dijkstra from cached graph."},
    {"topological_sort_by_node_id", py_topological_sort_by_node_id, METH_VARARGS,
     "topological_sort_by_node_id(node_ids, node_id_edges) -> list[NodeId]\n\n"
     "Topological sort (Kahn's algorithm). Edges are directed: (u, v) means u -> v.\n"
     "Raises ValueError if the graph contains a cycle."},
    {"all_edge_paths_ctx", py_all_edge_paths_ctx, METH_VARARGS,
     "all_edge_paths_ctx(capsule, source, targets[, cutoff, edge_mask, node_mask, node_simple]) -> list[list[int]]\n\n"
     "Find all paths from source to targets using each edge at most once.\n"
     "If node_simple is true, each node may be visited at most once per path."},
    {"strongly_connected_components_ctx", py_strongly_connected_components_ctx, METH_VARARGS,
     "strongly_connected_components_ctx(capsule[, edge_mask, node_mask]) -> list[set[NodeId]]\n\n"
     "Strongly connected components from a cached directed graph (Tarjan's algorithm)."},
    {"topological_sort_ctx", py_topological_sort_ctx, METH_VARARGS,
     "topological_sort_ctx(capsule[, edge_mask, node_mask]) -> list[NodeId]\n\n"
     "Topological sort from a cached directed graph (Kahn's algorithm)."},
    {"graph_is_directed", py_graph_is_directed, METH_VARARGS,
     "graph_is_directed(capsule) -> bool\n\n"
     "Return True if the parsed graph is directed."},
    {"neighbors_ctx", py_neighbors_ctx, METH_VARARGS,
     "neighbors_ctx(capsule, node_id[, edge_mask, node_mask]) -> set[NodeId]\n\n"
     "Neighbor node IDs via CSR. For directed graphs, returns successors."},
    {"predecessors_ctx", py_predecessors_ctx, METH_VARARGS,
     "predecessors_ctx(capsule, node_id[, edge_mask, node_mask]) -> set[NodeId]\n\n"
     "Predecessor node IDs via reverse CSR. Directed graphs only."},
    {"degree_ctx", py_degree_ctx, METH_VARARGS,
     "degree_ctx(capsule, node_id[, edge_mask, node_mask]) -> int\n\n"
     "Node degree via CSR. For directed graphs, returns out-degree."},
    {"in_degree_ctx", py_in_degree_ctx, METH_VARARGS,
     "in_degree_ctx(capsule, node_id[, edge_mask, node_mask]) -> int\n\n"
     "In-degree via reverse CSR. Directed graphs only."},
    {"incident_edges_ctx", py_incident_edges_ctx, METH_VARARGS,
     "incident_edges_ctx(capsule, node_id[, edge_mask, node_mask]) -> list[int]\n\n"
     "Incident edge indices via CSR. For directed graphs, returns outgoing."},
    {"incoming_edges_ctx", py_incoming_edges_ctx, METH_VARARGS,
     "incoming_edges_ctx(capsule, node_id[, edge_mask, node_mask]) -> list[int]\n\n"
     "Incoming edge indices via reverse CSR. Directed graphs only."},
    {"edge_indices_ctx", py_edge_indices_ctx, METH_VARARGS,
     "edge_indices_ctx(capsule, u, v[, edge_mask]) -> list[int]\n\n"
     "Edge indices between u and v via CSR."},
    {"critical_edges_ctx", py_critical_edges_ctx, METH_VARARGS,
     "critical_edges_ctx(capsule[, edge_mask, node_mask, port_mask]) -> list[int]\n\n"
     "Indices of the edges whose removal disconnects their component."},
    {"simple_path_blocks_ctx", py_simple_path_blocks_ctx, METH_VARARGS,
     "simple_path_blocks_ctx(capsule[, edge_mask, node_mask, port_mask]) -> (list[set], dict[int, set[int]])\n\n"
     "Blocks of nodes and multiport edges, as a simple path visits each at most once."},
    {"cycle_basis_ctx", py_cycle_basis_ctx, METH_VARARGS,
     "cycle_basis_ctx(capsule[, edge_mask, node_mask]) -> list[list[NodeId]]\n\n"
     "Fundamental cycle basis of an undirected graph."},
    {"dag_longest_path_ctx", py_dag_longest_path_ctx, METH_VARARGS,
     "dag_longest_path_ctx(capsule[, weights, edge_mask, node_mask]) -> list[NodeId]\n\n"
     "Longest path in a directed acyclic graph."},
    {"component_labels_ctx", py_component_labels_ctx, METH_VARARGS,
     "component_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes\n\n"
     "int32 per node index: the smallest node index of its connected component, -1 for an excluded node."},
    {"quotient_edges_ctx", py_quotient_edges_ctx, METH_VARARGS,
     "quotient_edges_ctx(capsule, labels[, edge_mask]) -> (from_labels, to_labels, edge_indices, internal_edge_indices)\n\n"
     "Four int32 byte arrays over the unmasked edges in index order: the crossing edges (endpoint labels\n"
     "differ) as three parallel arrays and the internal edges (endpoint labels equal) as edge indices."},
    {"degrees_ctx", py_degrees_ctx, METH_VARARGS,
     "degrees_ctx(capsule[, edge_mask, node_mask]) -> bytes\n\n"
     "int32 incidence degree per node index under the masks, self-loops counted twice, 0 for an excluded node."},
    {"biconnected_component_edge_labels_ctx", py_biconnected_component_edge_labels_ctx, METH_VARARGS,
     "biconnected_component_edge_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes\n\n"
     "int32 biconnected component id per edge index; bridges are singleton components,\n"
     "masked edges, edges at excluded nodes and self-loops get -1."},
    {"sp_state_new", py_sp_state_new, METH_VARARGS,
     "sp_state_new(capsule, terminal_mask, protected_mask[, edge_mask, node_mask,\n"
     "             pendant_keep_mask, series_blocked_mask]) -> state capsule\n\n"
     "Reduction state of the masked graph with the terminal-free components already gone."},
    {"sp_state_free", py_sp_state_free, METH_VARARGS,
     "sp_state_free(state) -> None. Release the state and make the handle inert."},
    {"sp_state_graph", py_sp_state_graph, METH_VARARGS,
     "sp_state_graph(state) -> the graph capsule the state was built from."},
    {"sp_next_move", py_sp_next_move, METH_VARARGS,
     "sp_next_move(state) -> (kind, node, edge_a, edge_b, neighbor_a, neighbor_b) or None."},
    {"sp_apply_move", py_sp_apply_move, METH_VARARGS,
     "sp_apply_move(state, move) -> (created_edge, parallel_pending)."},
    {"sp_pair_edges", py_sp_pair_edges, METH_VARARGS,
     "sp_pair_edges(state, from_node, to_node) -> int32 buffer of the live edges, or None."},
    {"sp_apply_parallel", py_sp_apply_parallel, METH_VARARGS,
     "sp_apply_parallel(state, from_node, to_node, edges) -> created edge slot."},
    {"sp_batch_moves", py_sp_batch_moves, METH_VARARGS,
     "sp_batch_moves(state, kind) -> int32 buffer of independent moves, or None."},
    {"sp_apply_batch", py_sp_apply_batch, METH_VARARGS,
     "sp_apply_batch(state, kind, batch) -> None. Apply one whole batch, in buffer order."},
    {"sp_state_log", py_sp_state_log, METH_VARARGS,
     "sp_state_log(state) -> twelve int32 byte buffers, as series_parallel_reduce_ctx returns."},
    {"series_parallel_reduce_ctx", (PyCFunction)py_series_parallel_reduce_ctx, METH_VARARGS | METH_KEYWORDS,
     "series_parallel_reduce_ctx(capsule, terminal_mask, protected_mask[, edge_mask, node_mask])\n"
     "    -> twelve int32 byte buffers\n\n"
     "Terminal-preserving series-parallel reduction as a flat operation log: op_kind, left, right,\n"
     "endpoint_u, endpoint_v, interior_node, leaf_edge_index and pendant_absorber per operation, then\n"
     "the residual as operation id and both endpoint node indices per surviving edge, then the\n"
     "surviving node indices. terminal_mask and protected_mask mark membership by a non-zero byte."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core",
    "Fast graph algorithms: union-find, Tarjan's, BFS, Dijkstra.", -1, methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&module);
}
