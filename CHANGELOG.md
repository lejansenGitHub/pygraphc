# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `all_edge_paths(..., ignore_self_loops=True)` on `Graph` and `GraphView`
  never traverses self-loops, so no returned path contains one.
- `tests/performance_tests/test_networkx_baselines.py` adds a guarded networkx
  baseline for every operation the other performance tests left uncovered:
  `connected_components`, `articulation_points`, `biconnected_components`,
  `shortest_path`, `multi_source_shortest_path_lengths`, `eccentricity`,
  `two_edge_connected_components`, `nodes_on_simple_paths` and masked
  connected components. Each test times both sides best-of-3, prints the
  ratio, asserts identical results and asserts pygraphc is faster, so a fast
  wrong answer fails too.

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
- README benchmark table re-measured on one machine with networkx 3.6.1 and
  now states the hardware, the networkx version and the timing discipline.
  Several speedups are lower than the previously published ones (connected
  components at 1M: 20x, not 46x; articulation points at 1M: 17x, not 26x;
  Dijkstra at 1M: 14x, not 29x). The table also records the one operation
  where networkx is faster: for a single source-target pair `nx.shortest_path`
  uses bidirectional Dijkstra and beats `shortest_path` by about 6x, while
  against the one-directional `nx.dijkstra_path` pygraphc stays 23x ahead.
- Rebuilds through `with_edges` and `split_node` keep every base edge at its
  original index, keep excluded edges masked, and append added edges and new
  nodes. Edge indices and branch ids of the base graph therefore stay valid on
  the rebuilt view. Weights passed to a rebuilt view must cover every edge,
  masked ones included.
- `with_edges` takes `added_branch_ids`, required when the base graph carries
  `branch_ids`. `split_node` gives a rerouted edge the branch id of the edge it
  replaces.

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
