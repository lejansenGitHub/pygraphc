/* The Python entry points of the _core extension, registered in module.c.
 *
 * Most algorithms come in up to three tiers: an index-based one over node indices
 * 0..n-1, a node-ID one (`_nid`) and a context one (`_ctx`) over a parsed graph capsule.
 */
#ifndef PYGRAPHC_CORE_ENTRY_POINTS_H
#define PYGRAPHC_CORE_ENTRY_POINTS_H

#include "../models.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Node-ID API: edges use original node IDs, C does hash map + algo + remap.
 * Zero Python object creation for intermediate index-based edges.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Context-based algorithm variants (reuse parsed graph, optional mask) ── */

/* graph.c */
PYGRAPHC_INTERNAL PyObject *py_parse_graph(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_parse_terminal_graph(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_terminal_graph_shape(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_terminal_offsets(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_terminal_nodes(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_graph_is_directed(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_graph_edge_count(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_graph_node_count(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_translate_edges(PyObject *self, PyObject *args);

/* connectivity.c */
PYGRAPHC_INTERNAL PyObject *py_connected_components(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_with_branches(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_remapped(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_with_branches_remapped(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_by_node_id_split(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_connected_components_with_branches_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_strongly_connected_components_ctx(PyObject *self, PyObject *args);

/* biconnectivity.c */
PYGRAPHC_INTERNAL PyObject *py_bridges(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_articulation_points(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_biconnected_components(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_bridges_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_articulation_points_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_biconnected_components_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_bridges_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_articulation_points_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_biconnected_components_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_critical_edges_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_simple_path_blocks_ctx(PyObject *self, PyObject *args);

/* traversal.c */
PYGRAPHC_INTERNAL PyObject *py_bfs(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_bfs_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_bfs_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_all_edge_paths_ctx(PyObject *self, PyObject *args);

/* shortest_paths.c */
PYGRAPHC_INTERNAL PyObject *py_dijkstra(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_shortest_path_lengths(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_multi_source_dijkstra(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_dijkstra_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_shortest_path_lengths_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_multi_source_dijkstra_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_dijkstra_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_shortest_path_lengths_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_multi_source_dijkstra_ctx(PyObject *self, PyObject *args);

/* cycles_dag.c */
PYGRAPHC_INTERNAL PyObject *py_topological_sort_by_node_id(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_topological_sort_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_cycle_basis_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_dag_longest_path_ctx(PyObject *self, PyObject *args);

/* queries.c */
PYGRAPHC_INTERNAL PyObject *py_neighbors_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_predecessors_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_degree_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_in_degree_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_incident_edges_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_incoming_edges_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_edge_indices_ctx(PyObject *self, PyObject *args);

/* labels.c */
PYGRAPHC_INTERNAL PyObject *py_component_labels_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_quotient_edges_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_degrees_ctx(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_biconnected_component_edge_labels_ctx(PyObject *self, PyObject *args);

/* reduction.c */
PYGRAPHC_INTERNAL PyObject *py_series_parallel_reduce_ctx(PyObject *self, PyObject *args, PyObject *keywords);
PYGRAPHC_INTERNAL PyObject *py_sp_state_new(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_state_free(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_state_graph(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_next_move(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_apply_move(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_pair_edges(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_apply_parallel(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_batch_moves(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_apply_batch(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_sp_state_log(PyObject *self, PyObject *args);

#endif
