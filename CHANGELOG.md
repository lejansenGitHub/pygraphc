# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `all_edge_paths(..., ignore_self_loops=True)` on `Graph` and `GraphView`
  never traverses self-loops, so no returned path contains one.

### Fixed
- Weighted algorithms (`shortest_path`, `shortest_path_lengths`,
  `multi_source_shortest_path_lengths`, `eccentricity`, `dag_longest_path`)
  now raise `ValueError` when the weights length differs from the edge count.
  Previously a shorter list was read out of bounds and an empty list segfaulted.
- `GraphView.with_edges` keeps the view's excluded edges and excluded nodes
  on the rebuilt view. Previously node masks were always dropped and edge
  exclusions were dropped after a second rebuild, which also made
  `split_node` on a rebuilt view keep the original edge next to the rerouted one.
- `with_edges` on a split-list-constructed `Graph` raises `ValueError` instead
  of silently dropping every base edge.
- `all_edge_paths` reports each path containing an undirected self-loop once.
- `edge_indices(u, u)` reports a self-loop once.
- `branch_ids` must have exactly one entry per edge; a mismatch raises
  `ValueError` at construction. Several edges may share a branch id, and
  `without_branches` excludes every edge carrying the id.
- Duplicate node ids raise `ValueError` instead of creating a phantom isolated node.

### Changed
- Rebuilds through `with_edges` and `split_node` keep every base edge at its
  original index, keep excluded edges masked, and append added edges and new
  nodes. Edge indices and branch ids of the base graph therefore stay valid on
  the rebuilt view. Weights passed to a rebuilt view must cover every edge,
  masked ones included.
- `with_edges` takes `added_branch_ids`, required when the base graph carries
  `branch_ids`. `split_node` gives a rerouted edge the branch id of the edge it
  replaces.

## [0.2.0] - 2026-04-28

### Added
- `Graph` class: parse node ids and edges once, run many algorithms on the
  shared C structures (int map, edge list, CSR adjacency).
- `GraphView` masks: `without_edges`, `without_branches`, `without_nodes`,
  `with_edges`, chainable and without rebuilding the parsed graph;
  `for_each_edge_excluded` helper.
- `split_node` on `Graph` and `GraphView`: reroute selected edges to a new node.
- Structural algorithms: `bridges`, `articulation_points`,
  `biconnected_components`, `bfs`, `two_edge_connected_components`,
  `nodes_on_simple_paths`, `bridges_with_branch_ids`.
- Weighted algorithms with a C binary heap: `shortest_path`,
  `shortest_path_lengths` (with cutoff), `multi_source_shortest_path_lengths`,
  `eccentricity`.
- Directed graphs (`directed=True`): `strongly_connected_components`,
  `weakly_connected_components`, `topological_sort`; BFS and Dijkstra follow
  edge direction.
- `cycle_basis` (fundamental cycles) and `dag_longest_path` (optionally weighted).
- `all_edge_paths`: enumerate edge-disjoint paths from a source to targets,
  with `cutoff` and `node_simple` options; MultiGraph (parallel edge) support.
- Query primitives: `neighbors`, `successors`, `predecessors`, `degree`,
  `in_degree`, `out_degree`, `edge_indices`, `incident_edge_indices`,
  `outgoing_edge_indices`, `incoming_edge_indices`.
- DAG structure learning in C: `hill_climb_k2`, `k2_local_score`, `estimate_cpds`.
- `connected_components_with_branch_ids` excludes branches and nodes by domain
  id rather than by internal index.
- Split `src`/`dst` list calling convention for `connected_components`.

### Changed
- Renamed the package from `cgraph` to `networkc`, then to `pygraphc` (import
  name and distribution name).
- Query methods run in C on the CSR adjacency (O(degree)) instead of
  scanning the edge list in Python (O(m)).

### Fixed
- `without_nodes` stops traversal at excluded nodes, so it breaks connectivity.
- Query methods returned empty results on graphs constructed from split
  `src`/`dst` lists.

## [0.1.0] - 2026-04-04

### Added
- C union-find with path compression and union by rank
- `connected_components(node_ids, edges)` — yields component sets with original node IDs
- `connected_components_with_branch_ids(node_ids, edges, branch_ids)` — yields (node_set, branch_set) tuples
- Accepts edges as `list[tuple[int, int]]` or numpy `int32` array
- Node ID remapping performed entirely in C (no Python loop)
- Performance benchmarks vs scipy (4-13x speedup)
- PEP 561 typed package (`py.typed`)

### Changed
- Renamed package from `connected-component` to `cgraph`
- Public API uses clean names without `igp_` prefix
- Remapped API is the only public interface (pass `node_ids`, not `num_nodes`)
