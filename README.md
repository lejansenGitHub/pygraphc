# pygraphc

Fast, general-purpose graph algorithm library implemented in C, callable from Python. Zero runtime dependencies.

Edges use **original node IDs** — no manual index mapping needed. A C-side hash map translates IDs to internal indices automatically.

## Benchmarks

### vs networkx (graph algorithms)

Sparse random graphs, ~3 edges per node. Algorithm time only: node ids, edge
lists and the networkx graph are built before the timer starts. Measured on an
Apple M3 Pro (12 cores, macOS 26.6), CPython 3.11.11, networkx 3.6.1, best of 3
runs after one warm-up. `tests/performance_tests/test_networkx_baselines.py`
guards the connected-components rows from 10K up and every row of the second
table: it asserts that pygraphc and networkx return the same result and that
pygraphc is faster. The per-algorithm files in the same directory re-measure
the remaining rows and print the ratio, but assert an absolute time limit
rather than the ratio.

| Algorithm | Nodes | pygraphc | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Connected Components | 1K | 0.00007s | 0.0005s | **6.5x** |
| Connected Components | 10K | 0.0004s | 0.003s | **6.0x** |
| Connected Components | 100K | 0.004s | 0.061s | **14x** |
| Connected Components | 1M | 0.059s | 1.15s | **20x** |
| Bridges | 1M | 0.223s | 12.88s | **58x** |
| Articulation Points | 1M | 0.204s | 3.55s | **17x** |
| BFS | 1M | 0.073s | 6.94s | **95x** |
| Dijkstra (single-source lengths) | 1M | 0.356s | 5.00s | **14x** |
| Shortest path, single pair, `Graph` + float64 weights | 100K | 0.0002s | 0.0015s | **7.5x** |
| Shortest path, single pair, `Graph` + list weights | 100K | 0.0007s | 0.0015s | **2.2x** |
| Edge paths (cutoff=5) | 80 | 0.000001s | 0.0001s | **91x** |
| SCC (directed) | 1M | 0.129s | 4.64s | **36x** |
| WCC (directed) | 1M | 0.029s | 1.97s | **69x** |
| Topological sort | 1M | 0.070s | 1.86s | **27x** |
| DAG longest path | 1M | 0.090s | 4.49s | **50x** |
| Cycle basis | 100K | 0.018s | 17.89s | **979x** |

Further networkx baselines, same machine and discipline, from
`tests/performance_tests/test_networkx_baselines.py`:

| Algorithm | Nodes | pygraphc | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Articulation Points | 100K | 0.007s | 0.153s | **21x** |
| Biconnected Components | 100K | 0.015s | 0.276s | **18x** |
| Multi-source Dijkstra lengths | 100K | 0.014s | 0.262s | **18x** |
| Eccentricity (weighted) | 100K | 0.015s | 0.257s | **17x** |
| Two-edge-connected components | 10K | 0.005s | 0.087s | **18x** |
| `nodes_on_simple_paths` | 24 | 0.00001s | 0.008s | **677x** |
| Connected Components, edge-masked view | 100K | 0.003s | 0.059s | **21x** |
| Single-pair `shortest_path` vs `nx.dijkstra_path` | 100K | 0.0002s | 0.206s | **1000x** |
| Single-pair `shortest_path` vs `nx.shortest_path` | 100K | 0.0002s | 0.0015s | **7.5x** |
| the same, both sides building their graph inside the timer | 100K | 0.0033s | 0.227s | **69x** |

`nx.shortest_path` dispatches to bidirectional Dijkstra for a single
source-target pair and pygraphc does the same, so the last two rows are one
algorithm measured under two fair disciplines: both sides prepared, and both
sides building their graph inside the timer. Prepared, with weights as a float64
buffer, pygraphc is 7.5x ahead; building inside the timer it is 69x ahead,
because constructing an `nx.Graph` over 150,000 weighted edges costs about
225 ms against 3 ms for the whole pygraphc call. The same prepared call with a
list of floats instead of a buffer gives 2.2x — the element-by-element
conversion of the weights costs several times the search.

An earlier version of this table reported **0.16x, "the one operation where
networkx wins"**. That number compared pygraphc building its graph *inside* the
timer against a networkx graph built outside it. There is no operation here
where networkx wins; there was an unfair measurement.

`shortest_path` with a target runs a bidirectional Dijkstra: two searches, one forward from the source and one backward from the target, meet in the middle, so only a small part of a large graph is settled. It is compared against `nx.shortest_path`, which dispatches to networkx's own bidirectional Dijkstra; against the one-directional `nx.dijkstra_path` the same query is ~500x (list weights) to ~12,000x (float64 weights) faster.

**Pass the weights as a float64 buffer** — a numpy `float64` array or `array.array("d", ...)` — to get the fast row. Such a buffer is handed to C as is, so the search reads only the edges it inspects, a few hundred on the graph above. A list of floats has to be converted and validated element by element first, which is proportional to the whole edge list and costs more than the search itself (1.0 ms of the 1.1 ms). Both rows are correct and both beat networkx; only the buffer row shows what the search actually costs. This is the same advice as passing edges as numpy arrays.

`shortest_path_lengths`, `multi_source_shortest_path_lengths` and `eccentricity` need every distance and keep using the single-source search, where the weight conversion is a small part of the total either way.

### vs pgmpy (DAG structure learning)

Hill-climb with K2 scoring on binary variables. Both produce identical DAGs.

| Scenario | pygraphc | pgmpy | Speedup |
|----------|-------:|------:|--------:|
| 5 vars, 100 samples | 0.00003s | 0.021s | **~700x** |
| 5 vars, 1000 samples | 0.00007s | 0.012s | **~170x** |
| 10 vars, 100 samples | 0.00005s | 0.044s | **~900x** |
| 10 vars, 500 samples | 0.0001s | 0.045s | **~410x** |
| 10 vars, 1000 samples | 0.0002s | 0.046s | **~220x** |
| 15 vars, 500 samples | 0.0002s | 0.103s | **~430x** |
| 15 vars, 1000 samples | 0.0004s | 0.104s | **~270x** |
| 20 vars, 500 samples | 0.0005s | 0.192s | **~420x** |
| 20 vars, 1000 samples | 0.0007s | 0.192s | **~290x** |

**2x–980x faster** than networkx and **~170x–900x faster** than pgmpy, with identical results. Zero construction overhead for directed graphs (forward + reverse CSR uses the same 2m memory as undirected).

## Installation

```bash
pip install -e .
```

Or from GitHub:

```bash
pip install git+https://github.com/lejansenGitHub/pygraphc.git
```

Requires a C compiler (the extension is compiled at install time with `-O3`).

The default build is portable across CPUs of the same architecture. To tune the
extension for the machine it is built on, opt in with `PYGRAPHC_NATIVE=1`; the
resulting binary is not portable:

```bash
PYGRAPHC_NATIVE=1 pip install -e .
```

### Development setup

Create a virtual environment, then install the package with the dev extras and
the vendored `blueprint-linters` wheel (not on PyPI):

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" vendor/blueprint_linters-0.9.0-py3-none-any.whl
```

## API

### Structural algorithms

```python
from pygraphc import (
    connected_components,
    bridges,
    articulation_points,
    biconnected_components,
    bfs,
    two_edge_connected_components,
    nodes_on_simple_paths,
)

node_ids = [100, 200, 300, 400]
edges = [(100, 200), (300, 400)]  # pairs of original node IDs

# Connected components
for component in connected_components(node_ids, edges):
    print(component)  # {100, 200}, {300, 400}

# Bridge edges
bridges(node_ids, edges)  # [(100, 200), (300, 400)]

# Articulation points
articulation_points(node_ids, edges)  # set()

# Biconnected components
list(biconnected_components(node_ids, edges))  # [{100, 200}, {300, 400}]

# BFS traversal
bfs(node_ids, edges, 100)  # [100, 200]

# 2-edge-connected components (bridges removed, then CC)
list(two_edge_connected_components(node_ids, edges))

# Nodes on any simple path from source to targets (block-cut tree)
nodes_on_simple_paths(node_ids, edges, source=100, targets=[400])
```

### Connected components with branch IDs

Track which branch IDs belong to each connected component:

```python
from pygraphc import connected_components_with_branch_ids

node_ids = [100, 200, 300, 400]
edges = [(100, 200), (300, 400)]
branch_ids = [901, 902]  # one per edge

for nodes, branches in connected_components_with_branch_ids(node_ids, edges, branch_ids):
    print(nodes, branches)
# {100, 200} {901}
# {300, 400} {902}
```

Also available on `Graph` and `GraphView`. Pass `branch_ids` at construction, then exclude by domain ID:

```python
g = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100, 200])
list(g.connected_components_with_branch_ids())
# [({1, 2, 3}, {100, 200})]

# Exclude branches by ID — dropped from connectivity and branch sets
view = g.without_branches([200])  # exclude branch 200 (edge 2--3)
list(view.connected_components_with_branch_ids())
# [({1, 2}, {100}), ({3}, set())]

# Exclude nodes by ID — breaks connectivity, but incident edge branch IDs
# are still collected in the non-excluded endpoint's CC
view = g.without_nodes([2])
list(view.connected_components_with_branch_ids())
# [({1}, {100}), ({3}, {200})]  — node 2 breaks the chain

# Combine both — exclude branches and nodes by their domain IDs
view = g.without_branches([200]).without_nodes([1])
list(view.connected_components_with_branch_ids())
# [({2}, {100}), ({3}, set())]
```

### Weighted algorithms

```python
from pygraphc import (
    shortest_path,
    shortest_path_lengths,
    multi_source_shortest_path_lengths,
    eccentricity,
)

node_ids = [0, 1, 2, 3]
edges = [(0, 1), (1, 2), (2, 3)]
weights = [1.0, 2.0, 3.0]

# Shortest path (bidirectional Dijkstra with C 4-ary heap)
# Pass weights as a numpy float64 array or array('d') to skip the per-element conversion
shortest_path(node_ids, edges, weights, source=0, target=3)  # [0, 1, 2, 3]

# Single-source shortest path lengths (with optional cutoff)
shortest_path_lengths(node_ids, edges, weights, source=0)
# {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}

# Multi-source shortest path lengths
multi_source_shortest_path_lengths(node_ids, edges, weights, sources=[0, 3])

# Eccentricity (max shortest-path distance from source)
eccentricity(node_ids, edges, weights, source=0)  # 6.0
```

### Graph class (parse once, run many algorithms)

When running multiple algorithms on the same graph, use the `Graph` class to avoid re-parsing the input each time:

```python
from pygraphc import Graph

node_ids = [0, 1, 2, 3, 4, 5]
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]

g = Graph(node_ids, edges)

# All calls reuse the same parsed C structures (IntMap + EdgeList + CSR adjacency list)
g.bridges()                    # [(2, 3)]
g.articulation_points()        # {2, 3}
list(g.connected_components()) # [{0, 1, 2, 3, 4, 5}]
g.bfs(0)                       # [0, 1, 2, ...]

# Weighted algorithms (weights still passed per-call)
weights = [1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0]
g.shortest_path(weights, source=0, target=5)
g.shortest_path_lengths(weights, source=0)

# Composite algorithms
list(g.two_edge_connected_components())
g.nodes_on_simple_paths(source=0, targets=[5])
```

Split-list constructor is also supported: `Graph(node_ids, src, dst)`.

On a 100K-node graph, running 3 algorithms via `Graph` is ~2x faster than 3 separate free-function calls, since input parsing and adjacency-list construction happen only once.

### Typed ids

`Graph` and `GraphView` are generic over the caller's node and branch id types, both bound to `int`. Every result hands back the id objects that were passed in, so branded `NewType` ids keep their brand and mypy strict (including `disallow_any_explicit`) needs no casts on either side:

```python
from typing import NewType

from pygraphc import EdgeIndex, Graph

NodeId = NewType("NodeId", int)
BranchId = NewType("BranchId", int)

node_ids = [NodeId(1), NodeId(2), NodeId(3)]
edges = [(NodeId(1), NodeId(2)), (NodeId(2), NodeId(3))]
branch_ids = [BranchId(10), BranchId(20)]

g = Graph(node_ids, edges, branch_ids=branch_ids)  # inferred: Graph[NodeId, BranchId]
components: list[set[NodeId]] = list(g.connected_components())
bridges: list[tuple[NodeId, NodeId, BranchId]] = g.bridges_with_branch_ids()
indices: list[EdgeIndex] = g.edge_indices(NodeId(1), NodeId(2))
view = g.without_branches([BranchId(20)])          # GraphView[NodeId, BranchId]
```

Edge indices (positions in the edge list) come back as `EdgeIndex`, a `NewType` over `int`, so a returned edge index is never mistaken for a branch id. Parameters that take edge indices stay `Collection[int]`, so both an `EdgeIndex` from a query and a plain literal are accepted. Id parameters take any `Sequence`, so tuples work as well as lists. A graph built without `branch_ids` has `int` as its branch type. `NodeId` and `BranchId` stay exported as `int` aliases.

### Edge-masked views (exclude edges without rebuilding)

Create lightweight views that exclude edges from the graph without rebuilding the CSR. The base graph is never mutated — views overlay a byte mask on the shared adjacency structure.

```python
from pygraphc import Graph, GraphView, for_each_edge_excluded

g = Graph(node_ids, edges)

# Exclude edges by index (position in the original edge list)
view = g.without_edges([3])          # exclude edge at index 3
list(view.connected_components())    # runs on the masked graph
view.bridges()
view.bfs(0)
view.shortest_path(weights, source=0, target=5)

# Look up edge indices by node pair
idx = g.edge_indices(2, 3)           # -> [3]  (list, for multigraph support)
view = g.without_edges(idx)

# Run an algorithm for each edge excluded, one at a time
for edge_idx, components in for_each_edge_excluded(g, "connected_components"):
    print(f"Without edge {edge_idx}: {len(components)} components")
```

**Mask overhead** is 2–8% vs an unmasked `Graph` call — negligible in absolute terms. The check compiles to a single byte load + branch per edge, trivially predicted. When no mask is used (`Graph` methods), the `NULL` pointer check is predicted away at zero cost.

Sparse random graph, 100K nodes (~3 edges per node), best of 20 runs:

| Algorithm | No mask | With mask | Overhead |
|-----------|--------:|----------:|---------:|
| Connected Components | 1.4ms | 1.5ms | +3% |
| Bridges | 1.5ms | 1.6ms | +7% |
| Articulation Points | 1.6ms | 1.7ms | +8% |
| Biconnected Components | 9.9ms | 10.6ms | +7% |
| BFS | 2.3ms | 2.5ms | +8% |
| Dijkstra (single pair, bidirectional) | 0.55ms | 0.55ms | +0% |
| SSSP lengths | 13.6ms | 13.8ms | +2% |

The single-pair row was re-measured after `shortest_path` became bidirectional; the other rows are from the earlier run.

The key win is avoiding O(V + E) graph rebuild per edge modification.

Parallel edges are supported — each edge is tracked by ID, so two edges between the same pair of nodes are handled correctly (e.g. for bridges, Dijkstra weight selection).

### MultiGraph support

pygraphc natively supports parallel edges (multigraphs). Each duplicate edge gets a unique index — no deduplication. All algorithms handle them correctly:

```python
from pygraphc import Graph

g = Graph([1, 2, 3], [(1, 2), (1, 2), (2, 3)])

g.is_multigraph                  # True
g.edge_count                     # 3
g.edge_indices(1, 2)             # [0, 1] — both parallel edges

# Bridges: (1,2) is NOT a bridge because a parallel edge exists
g.bridges()                      # [(2, 3)]

# Masking one parallel edge keeps the other active
view = g.without_edges([0])
list(view.connected_components())  # [{1, 2, 3}] — still connected via edge 1
```

### Node-masked views (exclude nodes without rebuilding)

Exclude nodes without rebuilding the CSR. Traversal stops at excluded nodes — they break connectivity. For `connected_components_with_branch_ids`, incident edge branch IDs are still collected in the non-excluded endpoint's CC. Chainable with edge masks.

```python
g = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])

# Exclude node 2 — breaks connectivity at node 2
view = g.without_nodes([2])
list(view.connected_components())  # [{0, 1}, {3, 4}]

# With branch IDs: incident edges tracked in non-excluded endpoint's CC
g2 = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100, 200])
view2 = g2.without_nodes([2])
list(view2.connected_components_with_branch_ids())
# [({1}, {100}), ({3}, {200})]  — connectivity broken, branch IDs preserved

# Chain with edge masks
view = g.without_edges([0]).without_nodes([3])
view.bfs(1)                        # [1, 2]

# BFS/Dijkstra raise ValueError if source is excluded
view = g.without_nodes([0])
view.bfs(0)                        # ValueError: source node is excluded
```

### Node splitting (reroute edges to a new node)

Split a node into two by choosing which edges stay on the original and which move to a new node. Composes `without_edges` + `with_edges` internally — no new C code needed.

```python
g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (1, 3)])
# edges: 0:(0,1), 1:(1,2), 2:(1,3)

# Split node 1: reroute edge 2 (1,3) to new node 99
view = g.split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[2])
# Result: edges (0,1), (1,2), (99,3) — node 1 keeps edges 0 and 1
list(view.connected_components())  # [{0, 1, 2, 99, 3}] — still connected

# Split the middle of a chain to disconnect
g2 = Graph([0, 1, 2], [(0, 1), (1, 2)])
view2 = g2.split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[1])
sorted(view2.connected_components(), key=min)  # [{0, 1}, {2, 99}]

# Works on views too (chainable)
view3 = g.without_edges([0]).split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[2])
```

**Rebuild contract.** `with_edges` and `split_node` rebuild the CSR, but the rebuilt view keeps every base edge at its original index, keeps excluded edges and nodes masked, and appends added edges and new nodes after the base ones. Edge indices, node masks and `branch_ids` of the base graph therefore stay valid on the rebuilt view. Weights passed to a rebuilt view must cover every edge, masked ones included. When the base graph carries `branch_ids`, `with_edges` needs `added_branch_ids`, one per added edge; `split_node` gives a rerouted edge the branch id of the edge it replaces. Both require edge-pair construction and raise `ValueError` on a split-list graph.

```python
g = Graph([0, 1, 2], [(0, 1)], branch_ids=[10])
view = g.with_edges([(1, 2)], added_branch_ids=[11])
view.incident_edge_indices(1)                      # [0, 1] — base edge keeps index 0
list(view.without_branches([11]).connected_components())  # [{0, 1}, {2}]
```

### Graph reduction kernel

`pygraphc.reduction` implements a terminal-preserving reduction kernel for multigraphs:
partition by an edge mask, quotient over the blocks, and reduce to the edges that matter
between a set of terminals while every input edge keeps its identity in a series-parallel
provenance tree. The partition step runs in C through
masked connected components; quotient, lift, reduction, provenance folds and scenario
application are the Python tier on top. Node ids are non-negative integers (validated
at construction: `TypeError` for a non-int, `ValueError` for negatives, duplicates and
unknown endpoints), edge ids are opaque hashable values that keep their identity through
every operation, and every result is deterministic by id order.

```python
from pygraphc import MultiGraph, Parallel, Partition, minimal_toggles, paths, quotient, reduce, scenario

# Edges are identified by id, never by endpoint pair: parallel edges survive.
graph = MultiGraph([1, 2, 3, 4], {"e1": (1, 2), "p1": (2, 3), "p2": (2, 3), "e2": (3, 4)})
active = set(graph.endpoints)

# Partition by masked connected components; block ids are the minimum member id.
base = Partition.from_components(graph, active)        # blocks: {1: [1, 2, 3, 4]}
base.block_of[4]                                        # 1, constant-time lookup

# Scenario = mask and re-partition. Removing both parallel edges splits the
# block although neither edge is a bridge. The C graph is built once
# per MultiGraph; scenarios only change the byte mask.
after = scenario(graph, active, removed={"p1", "p2"})
after.blocks()                                          # {1: [1, 2], 3: [3, 4]}
after.refines(base)                                     # True

# Quotient over blocks: every crossing edge keeps its id; edges inside a block are internal.
meta, internal = quotient(after, graph, crossing={"p1", "p2", "e1"})
meta.endpoints                                          # {"p1": (1, 3), "p2": (1, 3)}
internal                                                # {1: ["e1"]}

# Terminal-preserving reduction: pendant deletion, series merge, parallel merge until
# nothing applies. Every residual edge carries a series-parallel provenance tree.
reduced = reduce(meta, terminals={1, 3})
(tree,) = reduced.provenance.values()
isinstance(tree, Parallel)                              # True: Parallel({Leaf("p1"), Leaf("p2")})
paths(tree)                                             # {frozenset({"p1"}), frozenset({"p2"})}
minimal_toggles(tree, {"p1": True, "p2": True}, target_closed=False)
# frozenset({"p1", "p2"}): opening the residual edge needs both leaves
```

`reduce(graph, terminals, protected=frozenset(), *, fold_leaves=True, order=None)`:

- Terminals and protected nodes must be nodes of the graph; unknown ids raise `ValueError`.
- Components without a terminal are removed whole before any move.
- Terminals always survive. Protected nodes keep their incident edges unmerged
  (no series merge at them, no parallel merge of edges touching them); they may
  still be deleted as pendants.
- Self-loops take part in no move and leave with their node. Degree counts
  non-loop incidences, so a node with a loop and one further edge is a pendant,
  and a series merge needs exactly two non-loop incidences to two distinct neighbours.
- With `fold_leaves`, every eliminated node appears exactly once in the output:
  in `Reduced.folded_nodes` of a surviving node, as `Series.interior_nodes` of a
  residual tree, or in `Reduced.folded_interior` of such an interior node (the
  material folded into it before its series move). A pendant move passes the
  pendant node, its folded material and the interior nodes of the dropped edge's
  tree with their folded material on to the neighbour. With `fold_leaves=False`
  pendant material is dropped.
- Every pendant move also records `(neighbour, tree)` in `Reduced.dropped`, the
  provenance tree of the removed edge, so edge material merged before its attachment
  became pendant (a dead-end cycle returning to one node) stays available.
- Edges produced by moves get `VirtualEdgeId`s numbered in creation order; they
  sort after every input edge id and start above any virtual id already present in
  the input, so a residual can be reduced again (with the same terminals it is a fixpoint).
- Deterministic: candidates are processed in increasing node id (or in `order`),
  ties among edges by id order. Without protected nodes the residual and the folded
  material are the same for every order; a protected node can make them order
  dependent, in which case the id order is the documented answer.

Tree folds: `leaves(tree)`, `paths(tree, cutoff=None)` (series is the product,
parallel the union, the cutoff prunes inside the product), `closed(tree, state)`
(series is AND, parallel is OR) and `minimal_toggles(tree, state, target_closed=...)`
(union where the node type needs every child, cheapest child otherwise, ties by
edge id). `Series` and `Parallel` compare by identity (every tree node is created once,
by the move that produces it) and hash by cached structure; the folds, `repr` and
`tree_records(tree)` / `tree_from_records(records)` are iterative, so chains deeper than
the recursion limit are fine. `tree_records` is the canonical serialisable form (a
post-order log with parallel children ordered by smallest leaf id, independent of the
hash seed) and the way to compare two trees structurally; `pickle` and `copy.deepcopy`
recurse and are not suitable for deep trees.

`lift(partition, attribute, combine)` combines node attributes per block in
increasing node order, so a non-commutative `combine` still gives reproducible
results. `Partition.compose(finer)` expresses a second quotient level.

#### Label kernels (the C tier under the reduction)

`Graph` and `GraphView` expose the four array kernels the reduction runs on.
Each returns an int32 `memoryview` indexed by node or edge index, so a result
over a million nodes is one buffer and not one Python object per node. They are
usable on their own, masks included.

```python
graph = Graph([10, 20, 30, 40], [(10, 20), (20, 30), (30, 10), (30, 40), (20, 20)])

graph.component_labels().tolist()      # [0, 0, 0, 0] — smallest node index of each component
graph.without_edges([3]).component_labels().tolist()   # [0, 0, 0, 3]
graph.without_nodes([30]).component_labels().tolist()  # [0, 0, -1, 3] — excluded nodes get -1

graph.degrees().tolist()               # [2, 4, 3, 1] — self-loops count twice
graph.bcc_edge_labels().tolist()       # [1, 1, 1, 0, -1] — bridge is its own block, self-loop -1

labels = graph.without_edges([3]).component_labels()
src, dst, edge_indices, internal = graph.quotient_edges(labels)
# src/dst/edge_indices are parallel: crossing edge 3 goes from label 0 to label 3
[part.tolist() for part in (src, dst, edge_indices, internal)]  # [[0], [3], [3], [0, 1, 2, 4]]
```

- `component_labels()` labels a node with the smallest node index of its
  component, so a caller whose node ids are sorted reads the numerically
  smallest member as `node_ids[label]`. Excluded nodes get -1.
- `quotient_edges(labels)` walks the unmasked edges once in index order:
  endpoints with different labels are crossing edges and fill the three
  parallel arrays, equal labels make an edge internal. Every edge keeps its
  index, so nothing is keyed by endpoint pair and parallel edges stay
  distinct. The labels buffer must hold one int32 per node
  (`ValueError` otherwise, `TypeError` for a non-int32 buffer).
- `degrees()` gives incidence degrees under both masks with self-loops counted
  twice (out-degrees for a directed graph); `bcc_edge_labels()` gives the
  biconnected component id of every edge, bridges being singleton blocks and
  masked edges, edges at an excluded node and self-loops -1.

`Partition.from_components` builds `block_of` straight from the label array and
`quotient` from the edge split, so neither materialises a Python set per block.

### DAG structure learning (Bayesian networks)

Learn directed acyclic graph (DAG) structures from discrete data using greedy hill-climb search with K2 Bayesian scoring. Implemented in C — drop-in replacement for pgmpy's `HillClimbSearch` with identical results and orders of magnitude faster.

The algorithm:
1. Start with an empty DAG (no edges)
2. Each iteration, evaluate all legal single-edge operations (add, remove, flip) and pick the one with the highest K2 score improvement
3. A tabu list prevents cycling by forbidding recently reversed operations
4. Stop when no operation improves the score by more than `epsilon`

```python
from pygraphc import hill_climb_k2, estimate_cpds, k2_local_score

# Dataset: rows of discrete observations, values in [0, cardinality)
data = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 0],
]
cardinalities = [2, 2, 2]  # all binary

# Learn DAG structure
edges = hill_climb_k2(data, cardinalities, max_indegree=2)
# e.g. [(0, 2), (1, 2)] — variable 2 depends on variables 0 and 1

# Estimate conditional probability distributions (Laplace smoothing)
cpds = estimate_cpds(data, cardinalities, edges)
# {0: [[0.43, 0.57]], 2: [[0.75, 0.25], [0.25, 0.75], ...]}
# cpds[var] = list of distributions, one per parent configuration

# Compute K2 local score for a single variable given its parents
score = k2_local_score(data, cardinalities, child=2, parents=[0, 1])
```

Parameters for `hill_climb_k2`:
- `max_indegree` — maximum parents per node (default 1)
- `tabu_length` — circular buffer size for forbidden operations (default 100)
- `epsilon` — minimum score improvement to continue (default 1e-4)
- `max_iter` — iteration cap (default 1,000,000)

Complexity per iteration: O(n^2 * n_samples) where n = number of variables.

### Directed graphs

All graph algorithms support directed graphs via `directed=True`. Edges `(u, v)` are treated as `u -> v`. A forward and reverse CSR are built, using the same total memory as undirected (2m entries).

```python
from pygraphc import Graph, strongly_connected_components, weakly_connected_components

node_ids = [1, 2, 3, 4, 5]
edges = [(1, 2), (2, 3), (3, 1), (4, 5)]  # 1->2->3->1 cycle, 4->5

g = Graph(node_ids, edges, directed=True)

# Strongly connected components (mutually reachable nodes)
list(g.strongly_connected_components())
# [{1, 2, 3}, {4}, {5}]

# Weakly connected components (ignoring direction)
list(g.weakly_connected_components())
# [{1, 2, 3}, {4, 5}]

# Topological sort (DAGs only — raises ValueError on cycles)
dag = Graph([1, 2, 3, 4], [(1, 2), (1, 3), (3, 4)], directed=True)
dag.topological_sort()  # [1, 3, 4, 2] (one valid ordering)

# BFS follows outgoing edges only
g.bfs(source=1)  # [1, 2, 3] — cannot reach 4 or 5

# Dijkstra respects direction
dag = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
dag.shortest_path(weights=[1.0, 2.0], source=1, target=3)  # [1, 2, 3]
dag.shortest_path(weights=[1.0, 2.0], source=3, target=1)  # [] — no path

# Free-function shorthand
list(strongly_connected_components(node_ids, edges))
list(weakly_connected_components(node_ids, edges))
```

**Query methods** on directed graphs:

```python
g = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)

g.neighbors(1)       # {2} — successors only
g.successors(1)      # {2}
g.predecessors(1)    # {3}
g.degree(1)          # 1 (out-degree)
g.out_degree(1)      # 1
g.in_degree(1)       # 1
g.edge_indices(1, 2) # [0] — direction matters: edge_indices(2, 1) == []
```

**GraphView masking** works identically on directed graphs:

```python
g = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
view = g.without_edges([2])  # remove 3->1
list(view.strongly_connected_components())  # [{1}, {2}, {3}]
```

**Method availability:**

| Method | Undirected | Directed |
|--------|:---------:|:--------:|
| `connected_components` | yes | TypeError |
| `strongly_connected_components` | TypeError | yes |
| `weakly_connected_components` | TypeError | yes |
| `topological_sort` | TypeError | yes |
| `bridges` / `articulation_points` / `biconnected_components` | yes | TypeError |
| `bfs` / `shortest_path` / `dijkstra` | yes | yes |
| `all_edge_paths` | yes | yes |
| `successors` / `predecessors` / `in_degree` / `out_degree` | TypeError | yes |

### Cycle basis (fundamental cycles)

Detect all fundamental cycles in an undirected graph. The number of cycles equals the circuit rank: `m - n + c` where `c` is the number of connected components.

```python
from pygraphc import Graph, cycle_basis

# Triangle
g = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
g.cycle_basis()  # [[3, 2, 1]]

# Figure-8 (two cycles sharing node 3)
g = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3)])
g.cycle_basis()  # [[3, 2, 1], [5, 4, 3]]

# Tree (no cycles)
g = Graph([1, 2, 3], [(1, 2), (2, 3)])
g.cycle_basis()  # []

# Free function
cycle_basis([1, 2, 3], [(1, 2), (2, 3), (3, 1)])  # [[3, 2, 1]]
```

Useful for checking whether a graph is a forest (an empty basis means acyclic). Self-loops are detected as single-node cycles. Works with GraphView masks — removing a cycle-closing edge eliminates that cycle.

### DAG longest path

Find the longest path in a directed acyclic graph. Supports optional edge weights.

```python
from pygraphc import Graph, dag_longest_path

# Unweighted: longest by hop count
g = Graph([1, 2, 3, 4], [(1, 2), (1, 3), (3, 4)], directed=True)
g.dag_longest_path()  # [1, 3, 4]

# Weighted: longest by total weight
g = Graph([1, 2, 3], [(1, 2), (1, 3)], directed=True)
g.dag_longest_path(weights=[1.0, 100.0])  # [1, 3]

# Free function
dag_longest_path([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])  # [1, 2, 3, 4]
```

Uses topological sort + dynamic programming. Raises `ValueError` on cyclic graphs. 32-80x faster than networkx.

### Edge-path enumeration (edge-disjoint paths)

Find all paths from source to targets where each edge is used at most once. Returns paths as lists of edge indices. By default, nodes may be revisited (critical for multigraphs). Use `node_simple=True` to restrict to node-simple paths where each node is visited at most once. Use `ignore_self_loops=True` to never traverse self-loops, so no returned path contains one.

```python
g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (0, 2), (2, 3)])

# All edge-disjoint paths from 0 to 3
paths = g.all_edge_paths(source=0, targets=3)
# [[0, 1, 3], [2, 3]] — two paths using different edges

# With cutoff (max edges per path)
paths = g.all_edge_paths(source=0, targets=3, cutoff=2)
# [[2, 3]] — only paths with ≤2 edges

# Node-simple paths (no node revisits, like networkx all_simple_edge_paths)
g2 = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 1), (1, 4)])
g2.all_edge_paths(1, 4)                     # [[3], [0, 1, 2, 3]] — includes revisit of node 1
g2.all_edge_paths(1, 4, node_simple=True)    # [[3]] — only the direct edge

# Works with masks — respects excluded edges and nodes
view = g.without_edges([2])
paths = view.all_edge_paths(source=0, targets=3)
# [[0, 1, 3]] — only the path through edges 0→1→3
```

## Performance details

### Cost breakdown

Every pygraphc call has three cost phases. Connected components at 1M nodes, sparse random graph (~3 edges per node):

```python
class Branch:
    def __init__(self, branch_id: int, node_a: int, node_b: int):
        self.branch_id = branch_id
        self.node_a = node_a
        self.node_b = node_b
```

| Phase | Tuples | Split lists | numpy (m,2) |
|-------|-------:|------------:|------------:|
| 1. Gather (Python loop over objects) | 71ms | **35ms** | 91ms |
| 2. Parse (hash map + edge translation) | 23ms | 22ms | **5ms** |
| 3. C algorithm (union-find + results) | 62ms | 62ms | 62ms |
| **Total** | **156ms** | **119ms** | **159ms** |
| vs tuples | baseline | **1.31x** | ~same |

- **Gather** is pure Python — extracting `node_a`/`node_b` from your objects. pygraphc can't optimize this, but the interface choice affects it (tuples cost 71ms, split lists cost 35ms).
- **Parse** is the C-side input handling — building the node-ID hash map and translating edges to internal indices. Numpy skips per-element Python unpacking via buffer reads.
- **C algorithm** is the actual computation (union-find + building Python `set` results). Fixed cost, identical across all interfaces. The `set` construction via `PySet_Add` accounts for ~50ms of the 62ms — this is the CPython floor for creating hash-based sets of 1M elements.

### Performance across graph topologies

End-to-end from `Branch` objects using split lists. Connected components at 1M nodes:

| Scenario | Components | Gather | pygraphc | Total |
|----------|-----------:|-------:|-------:|------:|
| 1 component (all connected) | 1 | 25ms | 30ms | 55ms |
| 10 components (100K each) | 10 | 22ms | 27ms | 48ms |
| 1K components (1K each) | 1,000 | 21ms | 23ms | 44ms |
| 1 dominant + 100K isolated | 100,001 | 19ms | 54ms | 73ms |
| Random sparse (avg degree 3) | 54,266 | 35ms | 74ms | 109ms |
| 1M isolated (0 edges) | 1,000,000 | 0ms | 212ms | 212ms |

The pygraphc call splits into union-find algorithm and Python set construction:

| Scenario | Components | Union-find | Set construction | Set % of pygraphc |
|----------|-----------:|-----------:|-----------------:|----------------:|
| 1 component (all connected) | 1 | 4ms | 24ms | 86% |
| 10 components (100K each) | 10 | 4ms | 26ms | 87% |
| 1K components (1K each) | 1,000 | 4ms | 20ms | 84% |
| 1 dominant + 100K isolated | 100,001 | 4ms | 41ms | 92% |
| Random sparse (avg degree 3) | 54,266 | 11ms | 50ms | 82% |
| 1M isolated (0 edges) | 1,000,000 | 2ms | 160ms | 99% |

The union-find algorithm is 2-11ms — already near-optimal. **82-99% of the pygraphc time is Python set construction** (`PySet_New` + `PySet_Add`), which is the CPython floor for building hash-based sets. Many small components are expensive because each `PySet_New()` allocates a Python set object. Gather cost scales with edge count, not component count.

### Calling conventions

Two ways to pass edges — both accept Python lists or numpy arrays:

**Tuples** — simplest:
```python
edges = [(b.node_a, b.node_b) for b in branches]
connected_components(node_ids, edges)
```

**Split lists** — **1.3x faster** end-to-end:
```python
src = [b.node_a for b in branches]
dst = [b.node_b for b in branches]
connected_components(node_ids, src, dst)
```

Split lists is faster because building two flat lists avoids creating 1.5M tuple objects. The C parsing cost is similar — `PyLong_AsLong` per element dominates regardless of container shape.

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pip install networkx pgmpy
pytest tests/performance_tests/ -v -s -k speedup
pytest tests/performance_tests/test_networkx_baselines.py -v -s  # networkx baseline per operation
python benchmarks/bench_all.py       # structured cost breakdown + topology scenarios
python benchmarks/bench_dag_learn.py  # hill-climb K2: pygraphc vs pgmpy
```

## Tests

After the [development setup](#development-setup):

```bash
pytest tests/unit_tests/ -v
```

`tests/performance_tests/test_networkx_baselines.py` holds one networkx
baseline per graph operation, each asserting identical results and a speedup.
It needs `networkx` installed and is skipped otherwise; it carries the
`performance` marker like the other files there, so `-m "not performance"`
excludes it.
