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
- `series_parallel_reduce(terminal_mask, protected_mask, *, fold_leaves=True)`
- `reduce(..., pendant=PendantPolicy(default, exceptions))` in
  `pygraphc.reduction`: the pendant move's decision per node instead of once
  per graph. `"absorb"` hands the removed pendant's material to the
  neighbour, `"discard"` drops it, `"keep"` makes the node ineligible for the
  pendant move so it survives with its one incidence. The action of a pendant
  move is the action of the node it removes. A `keep` node is not a terminal:
  it is still merged away in series. The policy is frozen and hashable, and
  copies its exceptions, so it keeps the actions it was validated with.
- `reduce(..., series_ineligible=frozenset())` in `pygraphc.reduction`: blocks
  the series move at a node while its parallel merges stay allowed, which is
  the half of `protected` that answers the series question. `protected` keeps
  its meaning and the two compose by union on that question.
- `series_parallel_reduce(..., pendant_keep_mask=None, series_blocked_mask=None)`
  on `Graph` and `GraphView`: the two new per-node decisions as two more byte
  masks over node indices, read by the C loop next to the terminal and
  protected masks. A missing one of these two is the empty set; the terminal
  mask has no such default and `None` in its place still raises `TypeError`.

- `Graph.series_parallel_state(terminal_mask, protected_mask)` and the
  `ReductionState` handle it returns: the mutable half-edge incidence
  structure of the reduction behind an opaque capsule, with the
  terminal-free components already removed and one leaf operation per edge
  emitted, plus one short primitive per step over it. `next_move()` reports
  the move at the next candidate node, `apply_move(move)` applies it and
  says whether it created a mergeable parallel pair, `pair_edges(u, v)` and
  `apply_parallel(u, v, edges)` are that merge, `batch_moves(kind)` reports
  every currently applicable and mutually independent move of one kind into
  one flat int32 buffer and `apply_batch(kind, batch)` applies the whole
  buffer, `log()` reads the same `ReductionLog` the monolithic loop returns.
  The handle carries the graph it was built from and goes inert on `free()`,
  so a released state raises `ValueError` instead of dangling. Every index a
  caller hands back is validated before anything is written: an edge must be
  live and run between the two distinct live nodes named, a series move must
  name two distinct edges, and a pendant or series move must not name a
  terminal.
- `reduce(..., engine="moves" | "rounds")`: the reduction fixpoint driven
  from Python over those primitives, one boundary crossing per move and one
  per kind per round. The state survives every move, so no graph is rebuilt
  between them. On 20 000 nodes, 25 000 edges and 200 terminals the monolithic
  `"c"` engine takes 0.022 s, `"moves"` 0.025 s and `"rounds"` 0.023 s against
  0.087 s for `"python"` and 0.153 s for networkx; at a million nodes 2.02 s,
  2.19 s and 1.92 s. Roughly four fifths of every engine's time is the fold
  that builds the provenance trees, which is why the boundary barely shows.
  `"moves"` applies the moves in the monolith's order and reproduces its
  `Reduced` exactly, tree shapes included; `"rounds"` changes the order the
  merges happen in, so its residual, the leaves of every residual edge, the
  folded payload of every surviving node as a set and the material that left
  the graph are the same, while the split of `dropped` across pendant moves
  and the nesting of a series chain may differ.
- `series_parallel_reduce(terminal_mask, protected_mask)`
  on `Graph` and `GraphView`: the structural loop of the terminal-preserving
  reduction in C, returning a `ReductionLog` of twelve int32 `memoryview`s.
  Eight hold one entry per operation in move order (`op_kind` as leaf,
  series, parallel or pendant, the two child operation ids, the oriented
  endpoints of the virtual edge an operation produces, the eliminated or
  removed node, the input edge index of a leaf and the neighbour a pendant
  payload moves to); three hold the residual as operation id and both
  endpoint node indices per surviving edge; one holds the surviving node
  indices. A parallel merge of more than two edges is a left-deep chain of
  binary operations of which only the last carries endpoints. The two masks
  mark membership by a non-zero byte, one byte per node index, and are read as
  buffers rather than iterated, so a list of node ids is not a mask. `None` in
  place of the terminal mask raises `TypeError`: with no terminal every
  component is terminal-free and the whole graph would be deleted.
- `reduce(..., engine="c" | "python")` in `pygraphc.reduction`. The new
  default `"c"` runs the loop above and folds its log into the same
  `Leaf`/`Series`/`Parallel` trees and the same `Reduced`; `"python"` runs
  the worklist, which stays the reference semantics and is what a
  caller-supplied `order` selects. On 20 000 nodes, 25 000 edges and 200
  terminals the C engine takes 0.022 s against 0.087 s for the Python engine
  and 0.146 s for a straightforward networkx implementation of the same three
  moves, so 6.7x networkx and 3.9x the Python engine, and it reduces a million
  nodes with 1.25 million edges in 3.3 s, of which 0.29 s is the C loop and
  the rest the fold that builds the trees.
- `minimal_toggles(tree, state, *, target_closed, togglable_leaves=None)` in
  `pygraphc.reduction` restricts the fold to the leaves that may be flipped;
  every leaf may by default, so existing callers are unaffected. A leaf outside
  the set keeps its state, so a node that needs every child is unreachable as
  soon as one child is and a node that needs one child picks the cheapest
  reachable one. The return type is now `frozenset[EdgeId] | None`: the empty
  set means the tree already takes the target state, `None` means no subset of
  the permitted leaves reaches it. Ties among equally small candidate sets are
  broken by edge id order, unchanged.
- `series_chain(tree, edge_endpoints, start_node)` and `SeriesStep` in
  `pygraphc.reduction` expose the ordered walk of a series chain as one step
  per position, each naming the node stepped from, the subtree crossed and the
  node reached. `edge_endpoints` is `MultiGraph.endpoints` of the graph the tree
  was reduced from: the direction each subtree runs in is derived from the nodes
  it spans, because the tree does not record it — a series node stores its
  children in the direction of the merge that created it and a later merge can
  reach it from either end, so a walk that trusts the stored order crosses a
  nested chain backwards. Walking from the other endpoint returns the reversed
  sequence with every step reversed, a `Leaf` is a chain of one step, a series
  node nested in a series node is flattened into the sequence so positions along
  the whole chain are addressable by index, and a `Parallel` child is one
  position. A `Parallel` tree, a start node that is not an endpoint, a leaf that
  is not an edge of the graph, a subtree that does not span exactly two nodes, a
  series node the walk reaches at a node neither of its outer children touches
  and a subtree whose spanned pair is not the pair of chain nodes its position
  sits between raise `ValueError`; the last of those is what makes the refusal sound,
  since subtrees that each span two nodes can still fail to meet end to end.
- `tests/integration_tests/`: three domain integration cases behind a new
- `tests/integration_tests/`: four domain integration cases behind a new
  `integration` pytest marker, each modelling a problem from a field the
  reduction kernel was not designed for and asserting facts about that field
  rather than implementation output — the seven bridges of Konigsberg
  (`Partition`, `quotient` with parallel edges, `lift`, `minimal_toggles`, the
  irreducible series-parallel core), the caffeine molecular scaffold
  (`cycle_basis` plus `reduce` with `fold_leaves` as a Bemis-Murcko side-chain
  strip), a ten by ten perfect maze (pendant and series moves down to one
  edge, `paths`, `scenario`) and the same maze braided with three extra
  passages (the parallel move, `paths` returning eight routes, `scenario` on a
  passage that lies on a loop, `minimal_toggles` as the minimal cut that makes
  the maze unsolvable — one passage at the one-door entrance, two across a ring
  — dead ends pruned into the folded material). The two maze
  cases share the cell geometry and the drawing parser through
  `tests/integration_tests/maze_geometry.py`.
- `connected_component(node_id)` on `Graph` and `GraphView`: the connected
  component containing one node, as a set of node ids, the single-component
  counterpart of `connected_components()`. It reads the `component_labels`
  kernel and collects only the requested component, choosing between a
  `bytes.find` scan and one pass over every label by how many members a
  `bytes.count` probe reports. At 1,000,000 nodes that makes it the cheaper
  call on every shape except a graph that is itself one component: 7.2 ms and
  3.8 MiB against 738.4 ms and 214.1 MiB on all singletons, 63.2 ms against
  143.3 ms on a sparse graph with a giant component and a tail, but 40.1 ms
  against 9.9 ms when the whole graph is one component. On smaller graphs the
  fixed cost of the probe weighs more: the same sparse mixture at 100,000 nodes
  is 0.85x rather than 2.3x. Each call
  repeats the O(n + m) label pass, so more than a handful of components is a
  job for `connected_components()`. The node is resolved through the graph's
  cached node id to index map when another method has already built it and by
  a scan of the node ids when it has not, so a single lookup never builds a
  map it would use once. Both masks of a view are respected; an unknown node
  and a node excluded from the view raise `ValueError`.
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
- Four C label kernels on `Graph` and `GraphView`, each returning an int32
  `memoryview` instead of Python containers: `component_labels()` (smallest
  node index of every node's connected component, -1 for an excluded node),
  `quotient_edges(labels)` (one pass over the unmasked edges in index order
  returning crossing edges as three parallel arrays of source label,
  destination label and edge index plus the internal edge indices; the
  labels buffer must hold one int32 per node), `degrees()` (incidence degrees
  under the masks, self-loops counted twice) and `bcc_edge_labels()`
  (biconnected component id per edge, bridges as singleton components, -1
  for masked edges and self-loops). `Partition.from_components` and
  `quotient` in `pygraphc.reduction` now run on these kernels with identical
  results; the partition no longer materialises one set per block.
- `pygraphc.reduction`: the terminal-preserving graph reduction kernel.
  `MultiGraph` with edge identity, `Partition` from masked connected
  components (`from_components`, `from_groups`, `compose`, `refines`,
  `blocks`, constant-time `block_of`), `quotient` with internal edges,
  `lift` with a fixed combination order, `reduce` (pendant, series and
  parallel moves with terminals, protected nodes and leaf fold or drop)
  producing `Leaf`/`Series`/`Parallel` provenance trees with
  `folded_nodes`, `folded_interior` and `dropped` (neighbour and tree of
  every pendant-deleted edge) bookkeeping, the folds `leaves`,
  `paths(cutoff)`, `closed` and `minimal_toggles`, the iterative canonical
  serialisation `tree_records` / `tree_from_records`, and `scenario` (mask
  and re-partition). `Series` and `Parallel` compare by identity with a
  cached structural hash, so deep chains never recurse. `MultiGraph`
  validates node ids (non-negative ints, no bools, no duplicates, known
  endpoints) and `reduce` rejects unknown terminals and protected nodes.
  Virtual edge ids start above any present in the input, so a residual can
  be reduced again. Parallel candidates come from a pair index, so hub-heavy
  inputs reduce in linear time. All names are re-exported from `pygraphc`.
- `benchmarks/profile_workflows.py`: a phase profiler over a registry of the
  end-to-end workflows the library supports (graph construction, connected
  components, a scenario sweep, the full partition/quotient/lift/reduce
  pipeline, the same reduction through the raw operation log, the weighted and
  structural families, path enumeration and the DAG hill climb). Each workflow
  is instrumented with explicit phases, timed best-of-three after a warm-up
  with the profiler detached, and written to `profiles/` as a `.prof` per
  workflow, a `.txt` of its top entries by cumulative and total time, a shared
  `summary.md` with time, share and `tracemalloc` peak per phase plus a
  networkx reference column, and a `summary.json`. Phases rather than a call
  tree, because the split that matters — C kernel against Python object
  construction — is invisible in a flat profile: the reduction spends about a
  tenth of its `reduce` step in the C loop and the rest folding the operation
  log into provenance trees.
- `benchmarks/baseline.json` and `tests/unit_tests/test_workflow_profiles.py`:
  a deliberately coarse guard that runs the harness at small sizes and fails
  when a phase's share of its workflow's library time moves by more than 20
  percentage points, when a phase above 50 microseconds or a workflow's library
  time exceeds eight times its baseline, or when the phases stop accounting for
  95% of the measured total; the whole table is printed on failure. Shares are
  taken over the library time — the total without the harness's own input
  generation, which is a third to nine tenths of these workflows at any size —
  and the absolute per-phase gate is what covers the C kernels, which are single
  digit percentages of their workflows and which a share can only catch once
  they are an order of magnitude slower. No gate fails on one run: a workflow
  that trips is measured again and only what survives the fastest of three
  fresh runs is reported, which the share gate needs as much as the absolute
  ones because the harness reports the phases of the round with the best total.
  The baseline records the machine, interpreter, date and the revision the tree
  was at when the numbers were taken. A CI job runs the harness at the
  small sizes and uploads `profiles/` as an artifact without gating a merge.
- One workflow per `reduce` engine in the profile harness —
  `reduce_engine_c`, `reduce_engine_moves`, `reduce_engine_rounds`,
  `reduce_engine_python` — identical in the seeded graph, the terminal set,
  the sizes and the phase names, differing in the engine and in nothing else,
  so the comparison between them is not confounded. Phases are `generate
  input`, `construct kernel graph`, `structural moves`, `fold provenance` and
  `release`; an engine without a counterpart to a phase records it as zero
  rather than omitting it, so the four tables line up row by row. The `python`
  engine interleaves the structural work and the fold, and the harness prints
  that caveat wherever it prints those phases — the console table, `summary.md`
  and `engine_comparison.md` — rather than in the artifact alone.
- `python benchmarks/profile_workflows.py --compare-engines` and the
  `profiles/engine_comparison.md` it writes: the phase table for the four
  engines side by side at 20 000, 100 000 and 1 000 000 nodes, a `pstats`
  diff of each interesting pair naming the functions whose time differs and
  the functions only one profile has, the number of Python-to-C boundary
  crossings each engine makes, and the answers with numbers to what the fold
  costs, where `"moves"` loses its margin and why `"rounds"` matches the
  monolith. The two smaller sizes are profiled under `cProfile`, the largest
  is timed only, and the artifact says which is which. The measured numbers
  land in `profiles/engine_comparison.json`, and `--rebuild-comparison`
  rewrites the artifact from them without re-measuring. Two earlier claims do
  not survive the profile: the fold is about nine tenths of a warm `reduce`
  and about three quarters of a cold one rather than four fifths of either,
  and the fold-phase noise floor — taken from two engines that fold a
  byte-identical log — is several percent, the same order as the differences
  between the three C-backed engines.

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
- `reduce(..., fold_leaves=True | False)` in `pygraphc.reduction` is deprecated
  in favour of `pendant=PendantPolicy("absorb" | "discard")`, which is what it
  now maps to. It still works, the default is unchanged, and it does not warn
  yet; giving both it and `pendant` raises `ValueError`.
- `reduce` rejects unknown node ids in `series_ineligible` and in the pendant
  policy's exceptions the way it already rejected them in `terminals` and
  `protected`; the message names all four.
- `Leaf` in `pygraphc.reduction` no longer caches its hash. A leaf has no
  children, so neither its equality nor its hash can recurse and the generated
  ones over the edge id are enough; `Series` and `Parallel` keep their cached
  structural hash. One leaf exists per input edge, which makes this the
  reduction's busiest constructor and cuts about a seventh off the C engine's
  time on 25 000 edges.
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
