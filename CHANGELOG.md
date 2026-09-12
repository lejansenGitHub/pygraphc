# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Graph` and `GraphView` are generic over the caller's node and branch id
  types (`Graph[NodeIdT, BranchIdT]`, both bound to `int`). Results carry the
  id types that were passed in, so `NewType` ids round-trip without casts
  under mypy strict. Edge indices come back as `EdgeIndex`, a `NewType` over
  `int`, while parameters taking edge indices still accept any `int`. Id
  parameters accept any `Sequence`. `NodeIdT`, `BranchIdT` and `EdgeIndex` are
  exported; `NodeId` and `BranchId` stay as `int` aliases.
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
- Continuous integration runs for every pull request, not only for those
  targeting `main`. The `pull_request` trigger was restricted to
  `branches: [main]`, so a pull request stacked on another branch ran no job
  at all and reported no checks, which reads as results pending rather than
  as nothing having run. The `push` trigger stays on `main`.
- The coverage job's `diff-cover` step compares against the branch the pull
  request targets instead of always against `origin/main`. On a stacked pull
  request the hardcoded base measured the whole stack's diff and reported a
  number about work the pull request did not contain.
- The `test` and `performance` matrices set `fail-fast: false`. A failure in one
  matrix leg cancelled the rest, so legs that never finished were reported as
  failures on results they had not produced.
- The `performance` job reads its matrix from the contents of
  `tests/performance_tests/` instead of a hand-written list of file names. The
  list had fallen two files behind the directory, so `test_cycle_dag_perf.py`
  and `test_directed_perf.py` had never run. A file now runs because it is
  there. The matrix stays one job per file rather than one job for the whole
  directory: the nine jobs finish in about three minutes while their durations
  sum to about sixteen, so folding them into one job would cost some thirteen
  extra minutes of critical path. Per-file jobs cost the slowest single file
  plus the few seconds the listing job takes.
- `networkx>=3.4` is part of the `dev` extra. Every comparison test in the
  performance suite opens with `pytest.importorskip("networkx")`, so with no
  extra providing it they all skipped and the comparison they exist for was
  never made. They now run: the speedup measurements against networkx in the
  BFS, bridges, Dijkstra, cycle/DAG and directed suites, and the path-count
  equality assertion in `test_phase6_perf.py`, the one of them that gates a job.
  `test_cycle_basis_speedup_vs_networkx` stops at 100K, because networkx's
  `cycle_basis` is quadratic on that generator and the 1M point would cost tens
  of minutes per call; `test_cycle_basis_performance` keeps 1M covered on the
  pygraphc side, where no networkx call is involved.
- The masking performance tests assert the cost of a mask per returned
  component instead of a ratio of total times. Excluding half the nodes breaks
  the graph into 40,500 components where the unmasked graph has 5,333, so the
  ratio of totals was a measure of the output size and tripped its threshold
  whenever the baseline round happened to be fast. Timing is best of five on
  both sides rather than a mean of three, and the 10,000-node cases are
  asserted too.

### Changed
- README benchmark table re-measured on one machine with networkx 3.6.1 and
  now states the hardware, the networkx version and the timing discipline.
  Several speedups are lower than the previously published ones (connected
  components at 1M: 20x, not 46x; articulation points at 1M: 17x, not 26x;
  Dijkstra at 1M: 14x, not 29x). The table also records the one operation
  where networkx appeared to be faster: for a single source-target pair
  `nx.shortest_path` uses bidirectional Dijkstra and beat `shortest_path` by
  about 6x. That row has since been withdrawn — it compared pygraphc building
  its graph inside the timer against a networkx graph built outside it, and the
  entry below replaces it with two symmetric measurements.
- `shortest_path` (free function, `Graph` and `GraphView`) runs a bidirectional
  Dijkstra instead of a full single-source Dijkstra from the source. The two
  searches meet in the middle, so only a small part of a large graph is
  settled. On the 100k-node sparse weighted graph of the networkx baseline, a
  prepared `Graph` with weights as a float64 buffer (numpy array or
  `array("d", ...)`) answers a single source-target query in 0.18 ms against
  11.0 ms for the full single-source search the old code ran first, so 62x; with
  weights as a list of floats it is 0.7 ms, where converting the list dominates
  the query. Against `nx.shortest_path`, itself bidirectional, the comparison is
  now made under two symmetric disciplines: both sides prepared gives 7.5x, both
  sides building their graph inside the timer gives 69x. The 0.16x previously
  published for this operation was neither — it timed pygraphc building its
  graph against a networkx graph built beforehand.
  Weights are validated the same way for both input forms. The returned path
  may be a different path of the same total weight than before when several
  shortest paths exist. `shortest_path_lengths`,
  `multi_source_shortest_path_lengths` and `eccentricity` keep using the
  single-source search.
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
