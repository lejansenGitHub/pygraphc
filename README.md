# pygraphc

Fast, general-purpose graph algorithm library implemented in C, callable from Python. Zero runtime dependencies.

**Multi-terminal edges are first-class.** An edge may join any number of nodes
through its terminals, and every algorithm works on that edge as one edge: no hidden
hub nodes and no star expansion. Terminals can be masked one by one, and weighted
paths can take a weight per terminal. A graph of ordinary two-terminal edges runs as
fast as before. See [Concepts](#concepts) and
[Edges with terminals](#edges-with-terminals-multi-terminal-edges).

Edges use **original node IDs** — no manual index mapping needed. A C-side hash map translates IDs to internal indices automatically.

## Concepts

Every algorithm and every mask is defined in terms of these words, building on
each other in this order.

- **Node:** a vertex of the graph, given by its id. Node ids are integers of
  your choice; the C core maps them to internal indices.
- **Terminal:** one attachment of an edge to a node. A terminal belongs to exactly one
  edge and sits at exactly one node. An edge's terminals are numbered 0, 1, 2, …
  in the order they are written, and a terminal is addressed as
  `(edge_index, terminal_index)`. Two terminals of one edge may sit at the same node
  and stay distinct.
- **Edge:** an element of the graph that joins its terminals, written as the tuple
  of its terminals' nodes. The edge index is the position in the edge list.
  - A **two-terminal edge** `(u, v)` is the ordinary edge between `u` and `v`.
  - A **multi-terminal edge** has three or more terminals, e.g. `(u, v, w)`.
  - An undirected edge may also have a single terminal; it then joins nothing.
- **Directed edge:** each terminal is an **in-terminal** or an **out-terminal**. `(u, v)`
  has its in-terminal at `u` and its out-terminal at `v`. A multi-terminal edge is written
  `(in_nodes, out_nodes)`. Every directed edge has at least one in-terminal and one
  out-terminal.
- **Mask:** a view that excludes nodes (`without_nodes`), edges
  (`without_edges`) or single terminals (`without_terminals`) without rebuilding the
  graph. Masks combine as a union.
- **Active terminal:** a terminal whose edge, whose node and which itself are all
  unmasked. Activity is the only thing an algorithm asks about masks, so each
  of the following terms is defined through active terminals alone.
- **Connection:** an edge connects all of its active terminals to each other.
  On a directed edge, every active in-terminal leads to every active out-terminal.
  An edge with fewer than two active terminals connects nothing.
- **Connected component:** a maximal set of unmasked nodes that are linked
  through connections.
- **Membership:** an edge belongs to the component of its active terminals. Since
  these are connected, there is exactly one such component. An edge without
  an active terminal belongs to no component.
- **Critical edge / critical node:** an edge or node whose removal splits the
  nodes of its component (a bridge, an articulation point). Only active terminals
  count.

## Benchmarks

### vs networkx (graph algorithms)

Sparse random graphs, ~3 edges per node. Algorithm time only: node ids, edge
lists and the networkx graph are built before the timer starts. Measured on an
Apple M3 Pro (12 cores, macOS 26.7.2), CPython 3.11.13, networkx 3.6.1, best of 3
runs after one warm-up. `benchmarks/networkx_baselines.py` measures every row
of both tables and only reports; `tests/unit_tests/test_networkx_differential.py`
checks that pygraphc and networkx return the same results. Speed itself is
guarded by the three performance tiers described under [Tests](#tests).

| Algorithm | Nodes | pygraphc | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Connected Components | 1K | 0.000012s | 0.00017s | **14x** |
| Connected Components | 10K | 0.00012s | 0.0019s | **15x** |
| Connected Components | 100K | 0.0023s | 0.048s | **21x** |
| Connected Components | 1M | 0.028s | 1.11s | **39x** |
| Bridges | 1M | 0.16s | 12.83s | **83x** |
| Articulation Points | 1M | 0.13s | 3.26s | **25x** |
| BFS | 1M | 0.035s | 8.56s | **246x** |
| Dijkstra (single-source lengths) | 1M | 0.29s | 4.01s | **14x** |
| Shortest path, single pair, `Graph` + float64 weights | 100K | 0.0002s | 0.0015s | **7.5x** |
| Shortest path, single pair, `Graph` + list weights | 100K | 0.00062s | 0.0012s | **1.9x** |
| Edge paths (cutoff=5) | 80 | 0.000001s | 0.0001s | **91x** |
| SCC (directed) | 1M | 0.28s | 2.64s | **9.3x** |
| WCC (directed) | 1M | 0.027s | 2.01s | **74x** |
| Topological sort | 1M | 0.084s | 2.08s | **25x** |
| DAG longest path | 1M | 0.10s | 4.73s | **47x** |
| Cycle basis | 100K | 0.015s | 14.70s | **996x** |

Further networkx baselines, same machine and discipline, from
`benchmarks/networkx_baselines.py`:

| Algorithm | Nodes | pygraphc | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Articulation Points | 100K | 0.0041s | 0.13s | **32x** |
| Biconnected Components | 100K | 0.010s | 0.18s | **17x** |
| Multi-source Dijkstra lengths | 100K | 0.011s | 0.27s | **25x** |
| Eccentricity (weighted) | 100K | 0.011s | 0.23s | **20x** |
| Two-edge-connected components | 10K | 0.00049s | 0.071s | **145x** |
| `nodes_on_simple_paths` | 24 | 0.0000063s | 0.0060s | **953x** |
| Connected Components, edge-masked view | 100K | 0.0024s | 0.050s | **20x** |
| Single-pair `shortest_path` vs `nx.dijkstra_path` | 100K | 0.00019s | 0.22s | **1160x** |
| Single-pair `shortest_path` vs `nx.shortest_path` | 100K | 0.00016s | 0.0013s | **7.9x** |
| the same, both sides building their graph inside the timer | 100K | 0.0025s | 0.14s | **58x** |

`nx.shortest_path` dispatches to bidirectional Dijkstra for a single
source-target pair and pygraphc does the same, so the last two rows are one
algorithm measured under two fair disciplines: both sides prepared, and both
sides building their graph inside the timer. Prepared, with weights as a float64
buffer, pygraphc is 7.9x ahead; building inside the timer it is 58x ahead,
because constructing an `nx.Graph` over 150,000 weighted edges costs about
140 ms against 2.5 ms for the whole pygraphc call. The same prepared call with a
list of floats instead of a buffer gives 1.9x — the element-by-element
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

### One component instead of all of them

`connected_component(node_id)` is the single-component counterpart of the
`connected_components()` generator: it returns the component containing one
node as a set of node ids, so nothing about how a caller uses the result
changes. It reads the `component_labels` kernel and builds only the requested
component, instead of a set per component with every node added to one. Both
masks of a view are respected; an unknown node and a node excluded from the
view raise `ValueError`.

```python
g = Graph([10, 20, 30, 40, 50, 60], [(10, 20), (20, 30), (40, 50)])

g.connected_component(30)                            # {10, 20, 30}
g.connected_component(60)                            # {60}
g.without_edges([1]).connected_component(30)         # {30}
```

Both paths run the same O(n + m) label pass; the accessor then collects only
the requested component, picking between a `bytes.find` scan and one pass over
every label according to a `bytes.count` probe of how many members there are.
At 1,000,000 nodes that makes it the cheaper call on every shape except a graph
that is itself a single component, where `connected_components()` builds its one
set in C and a Python collection cannot beat it. The advantage is size-dependent
as well as shape-dependent: on the same sparse mixture it is 2.3x at a million
nodes and 0.85x at a hundred thousand, because the fixed cost of the probe and
the label pass is a larger share of a smaller graph.

Measured over four shapes with `benchmarks/bench_single_component.py` (fastest
of five calls, `tracemalloc` peak of one call). The accessor uses the graph's
cached node id to index map when another method has already built it and scans
the node ids when it has not, so `cold` is what a single lookup on a fresh
graph costs and `warm` what a lookup costs once the map exists:

| shape | nodes | members | cold | warm | all, then pick one | ratio | peak, one | peak, all |
|---|---|---|---|---|---|---|---|---|
| one giant component | 100,000 | 100,000 | 4.3 ms | 4.0 ms | 1.0 ms | 0.24x | 6.38 MiB | 6.00 MiB |
| 1000 equal components | 100,000 | 100 | 1.7 ms | 1.4 ms | 1.9 ms | 1.4x | 0.39 MiB | 8.03 MiB |
| all singletons | 100,000 | 1 | 1.1 ms | 0.8 ms | 26.5 ms | 32x | 0.38 MiB | 21.36 MiB |
| sparse mixture, giant member | 100,000 | 64,162 | 5.9 ms | 5.7 ms | 4.8 ms | 0.85x | 2.88 MiB | 7.76 MiB |
| sparse mixture, smallest member | 100,000 | 1 | 3.6 ms | 3.2 ms | 5.2 ms | 1.6x | 0.38 MiB | 7.76 MiB |
| one giant component | 1,000,000 | 1,000,000 | 43.3 ms | 40.1 ms | 9.9 ms | 0.25x | 51.82 MiB | 48.00 MiB |
| 1000 equal components | 1,000,000 | 1,000 | 11.8 ms | 9.0 ms | 11.7 ms | 1.3x | 3.85 MiB | 31.47 MiB |
| all singletons | 1,000,000 | 1 | 10.5 ms | 7.2 ms | 738.4 ms | 103x | 3.82 MiB | 214.05 MiB |
| sparse mixture, giant member | 1,000,000 | 642,061 | 64.5 ms | 63.2 ms | 143.3 ms | 2.3x | 51.82 MiB | 103.47 MiB |
| sparse mixture, smallest member | 1,000,000 | 1 | 28.8 ms | 25.8 ms | 132.3 ms | 5.1x | 3.82 MiB | 103.47 MiB |

The sparse mixture is a random graph at average degree 1.6, so it has one
component holding about two thirds of the nodes and a long tail of small ones —
the shape most real graphs have. The `count` probe costs one pass over the
label buffer, about 3 ms at 1,000,000 nodes, which is why the singleton rows
are slower than a bare scan would be; it buys the four rows where the
requested component is large.

There is no single break-even count, because there are two break-evens:

- **Shape.** Its advantage grows as the requested component shrinks relative
  to the graph: at a million nodes, 2.3x at two thirds of the graph, 5.1x for a
  small component of the same graph, 103x for a singleton. On a graph that is
  one single component it is 4x slower. Two rows of the table are below 1x, and
  the second is the one to keep in mind: the sparse mixture's giant component at
  a hundred thousand nodes is 0.85x, the same shape that is 2.3x at a million.
- **Count.** Each call repeats the O(n + m) label pass, so *k* lookups cost
  *k* warm calls, and one `connected_components()` call is cheaper above the
  ratio in the table — one lookup on a single-component graph, two on the
  sparse mixture's giant component, five on its small components, 103 on all
  singletons.

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

### Edges with terminals (multi-terminal edges)

> **Specification for the multi-terminal core, implemented in this pull request.**
> Everything below is the contract the implementation and its tests follow; it
> is written down first so that the semantics can be reviewed, and changed on
> request, before the code exists.

An edge joins its **terminals**, and every terminal sits at a node. An edge is written
as a tuple of node ids: a pair is a two-terminal edge, exactly as before, and a longer tuple
is an edge with that many terminals. The position in the tuple is the terminal index,
so terminals are addressed as `(edge_index, terminal_index)`, never by node, and two
terminals of one edge at the same node stay distinguishable. pygraphc knows nothing
about what a terminal is; which terminal is a transformer's primary is the caller's
mapping.

```python
graph = Graph(
    [1, 2, 3, 4, 5],
    [(1, 2), (2, 3, 4), (4, 5)],   # a two-terminal edge, a three-terminal edge, a two-terminal edge
)
```

**Directed graphs** keep the pair form unchanged: `(u, v)` is an edge from `u`
to `v`, with no extra direction input. An edge with more terminals is written as
two tuples, its in-nodes and its out-nodes; its terminal indices run through the
in-nodes first, then the out-nodes. Every edge needs at least one in-terminal and
one out-terminal. Reaching the node of any active in-terminal reaches the nodes of all
active out-terminals (OR semantics: each input feeds every output on its own).

```python
Graph(node_ids, [(1, 2), ((1,), (2, 3)), ((4, 5), (6,))], directed=True)
#                 1 -> 2   1 feeds 2 and 3   4 and 5 both feed 6
```

**Graphs of two-terminal edges behave as before.** Every existing call, input
format and result keeps its meaning; edges with more terminals only add to the
interface. The one change is membership under node masks: an edge whose two
endpoints are both masked has no active terminal and so belongs to no component,
where it used to form a component of its own with no nodes. There is one
`Graph` class and one `GraphView` for every kind of edge.

#### One rule decides everything: active terminals

Connection, membership and critical elements are defined through **active
terminals** only (see [Concepts](#concepts)). Every algorithm asks only this rule,
so a change in masking semantics is a change in one place.

#### Masking cases

Masks combine as a union and their order does not matter. Each case has a
named test.

| # | Case | Connection | Membership |
|---|------|------------|------------|
| 1 | Edge masked | joins nothing; its nodes keep their other edges | none |
| 2 | Some terminals masked, two or more active | the active terminals stay joined; a masked terminal only detaches its node from this edge | the component of the active terminals |
| 3 | Exactly one active terminal | joins nothing, never critical | the component of that terminal |
| 4 | No active terminal | as case 1 | none |
| 5 | Two-terminal edge with one terminal masked | case 3: joins nothing, as if the edge were masked | the component of the active end |
| 6 | Node masked | the node leaves every result; its terminals are inactive, the other terminals of those edges stay joined | as cases 2 to 4 for the remaining terminals |
| 7 | Two terminals of one edge at the same node, one masked | the node stays attached through the other terminal | unchanged |
| 8 | All terminals at one node | joins nothing, with or without masks | that node's component |
| 9 | Directed edge left without an active in-terminal or out-terminal | transmits nothing | as for undirected edges |
| 10 | Weighted path through an edge | enters through an active in-terminal (or any active terminal when undirected) and leaves through another active terminal; the cost is the entry terminal's weight plus the exit terminal's weight | — |
| 11 | Critical edges, critical nodes, biconnected blocks | computed on active terminals only; a masked terminal is never critical | — |
| 12 | Tree-shaped edge with k terminals in biconnected blocks | one block holding its k terminal nodes, as a bridge is its own two-node block | — |

#### Masking terminals, edges and nodes

Masks are views, as before, and they chain:

```python
graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])

graph.without_edges([1])          # the three-terminal edge joins nothing (case 1)
graph.without_terminals([(1, 0)])     # its terminal at node 2 is detached; 3 and 4 stay joined (case 2)
graph.without_nodes([3])          # node 3 leaves; terminals at 2 and 4 stay joined (case 6)
graph.without_terminals([(1, 0)]).without_nodes([4])   # one active terminal left: joins nothing (case 3)
```

`without_terminals` takes `(edge_index, terminal_index)` pairs on any graph. A two-terminal
edge's terminal 0 is its first node and terminal 1 its second, so masking one of its
terminals detaches it like masking the edge (case 5), while it still belongs to the
component of its active end.

#### Weights per terminal

Weighted algorithms take one weight per terminal, in edge order and, within an
edge, in terminal order. A **pass** through an edge goes from one terminal to another
terminal (directed: from an in-terminal to an out-terminal) and costs the entry terminal's
weight plus the exit terminal's weight, which is how the star equivalent of a
three-winding transformer is written down. Two-terminal edges keep their single
weight per edge: weights with one entry per edge are read as before, and the
per-terminal form is only needed when a graph has an edge with more than two terminals.

A terminal weight may be negative, as one arm of a star equivalent can be, as long
as every pass costs at least 0: every two terminals of an edge (directed: every
in-terminal and out-terminal) must sum to at least 0, otherwise the call raises
`ValueError`. A path never leaves an edge through the terminal it entered by, and
after leaving an edge through its negative terminal it does not pass that edge
again right there, since either would count the negative weight twice:

```python
star = Graph([1, 2, 3], [(1, 2, 3)])
star.shortest_path_lengths([3, -1, 2], 1)   # {1: 0.0, 2: 2.0, 3: 5.0}: 1 -> 3 costs 3 + 2, not (3 - 1) + (-1 + 2)
```

#### Edges with a single terminal

An undirected edge may have one terminal: it joins nothing and belongs to the
component of its node (case 8), which is how a load or a feed-in modelled as
an edge behaves. A directed edge needs at least one in-terminal and one out-terminal,
so it has at least two terminals.

#### Methods that name edges by node pair

`bridges()` returns node pairs and `edge_indices(u, v)` takes one. A pair cannot
name an edge with more than two terminals, so on a graph that has one, these
methods raise an error naming their replacement instead of answering
incompletely: `critical_edges()` returns the edge indices of every edge whose
removal disconnects its component, two-terminal and multi-terminal edges alike, and
`edge_indices_at(node_ids)` returns the edges whose terminals sit at exactly
these nodes. On a graph of two-terminal edges they behave as before, and the
`*_with_branch_ids` methods stay as they are. `critical_edges()` works on every
graph and view, `edge_indices_at` on every graph.

```python
graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])
graph.edge_indices_at([4, 2, 3])   # [1]: order and repeats do not matter
graph.critical_edges()             # [1, 0]: both edges are critical
```

#### Directed algorithms

Strongly connected components, topological sort and DAG longest path read
multi-terminal edges with the OR rule: an edge leads from each active
in-terminal to each active out-terminal, and is never part of a result.

```python
graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4)), (4, 1)], directed=True)
list(graph.strongly_connected_components())   # [{3}, {1, 4}, {2}]: 1 -> 4 -> 1; 3 has no way back, 2 only feeds

dag = Graph([1, 2, 3, 4, 5], [((1, 2), (3, 4)), (4, 5)], directed=True)
dag.topological_sort()       # 1 and 2 before 3 and 4, 4 before 5
dag.dag_longest_path()       # [1, 4, 5]: a pass through the edge counts 1
dag.dag_longest_path([1.0, 5.0, 2.0, 7.0, 0.5, 0.5])   # [2, 4, 5]: a pass weighs entry plus exit
```

An edge with an in- and an out-terminal at the same node is a cycle, like a
self-loop, and an edge without an active in-terminal passes nothing (case 9).

#### Neighborhood queries and cycles

The neighborhood queries follow the hypergraph libraries (XGI, HyperNetX) and,
for direction, the forward and backward stars of Gallo et al. An edge at a
node counts when its terminal there is active and it joins that terminal to
another active one: any other terminal undirected, an out-terminal from an
in-terminal directed.

```python
graph = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3, 4), (4, 5)])
graph.neighbors(2)               # {1, 3, 4}: every other terminal node of the edges at 2
graph.degree(2)                  # 2: one per terminal at the node
graph.incident_edge_indices(2)   # [0, 1]
graph.without_terminals([(1, 1)]).neighbors(3)   # set(): its only terminal is masked

dag = Graph([1, 2, 3, 4, 5], [((1, 2), (3, 4)), (4, 5)], directed=True)
dag.successors(1)                # {3, 4}
dag.predecessors(4)              # {1, 2}
```

`degree` counts terminals, so an edge with two terminals at a node counts twice,
as a self-loop does (case 26), and a node is never its own neighbor (case 27).

`cycle_basis` returns Berge cycles, the standard hypergraph cycle: it passes
through an edge from one terminal to another and uses every node and edge at
most once. The number of cycles is the cycle rank of the incidence graph,
terminals − nodes − edges + components, which is m − n + c on two-terminal
edges. A single multi-terminal edge is a tree, never a cycle.

```python
Graph([1, 2, 3, 4], [(1, 2, 3), (3, 4), (4, 1)]).cycle_basis()   # [[4, 1, 3]]
Graph([1, 2, 3, 4], [(1, 2, 3), (2, 3, 4)]).cycle_basis()        # [[3, 2]]: two edges sharing two nodes
Graph([1, 2], [(1, 1, 2)]).cycle_basis()                         # [[1]]: two terminals at one node
```

#### Critical edges, critical nodes and blocks

Computed on the active terminals only (case 11), by one depth-first search that
treats an edge with three or more terminals as a single step between its terminal
nodes:

- an edge is **critical** when removing it disconnects nodes of its terminals: a
  bridge for a two-terminal edge;
- a node is **critical** (`articulation_points`) when removing it disconnects
  other nodes; an edge with one active terminal never makes its node critical;
- **blocks** (`biconnected_components`) never split at a multi-terminal edge, so a
  tree-shaped edge with k terminals is one block of its k nodes (case 12), and an
  edge on a cycle merges with the cycle's block;
- `two_edge_connected_components` are the nodes that no single edge removal
  separates; a critical multi-terminal edge can lie on a cycle, so it is not
  simply removed.

`nodes_on_simple_paths` uses its own blocks: a simple path visits each node
and each multi-terminal edge at most once, and enters and leaves an edge through one
terminal each. A multi-terminal edge can therefore separate these blocks the way a
node does. In the tree-shaped edge `(1, 2, 3)`, the paths from 1 to 2 do not
visit 3.

#### Methods not yet available on terminals

Methods whose kernel does not read terminals yet raise `NotImplementedError` on a
graph with a multi-terminal edge and on a view with a terminal mask, instead of
answering on part of the graph; `docs/multi_terminal-decisions.md` lists the stage
that brings each. On graphs of two-terminal edges without a terminal mask they run as
before, with no added cost per call.

#### Open points

- **A "touching" relation** (an edge with a terminal, active or not, at a node of
  a component it is not a member of, such as an open switch at the boundary)
  is not part of this pull request.
- Every answer that could reasonably go another way is listed in
  [Multi-terminal edge case behavior](#multi-terminal-edge-case-behavior-up-for-discussion)
  at the end of this README.

### Node-masked views (exclude nodes without rebuilding)

Exclude nodes without rebuilding the CSR. Traversal stops at excluded nodes — they break connectivity. For `connected_components_with_branch_ids`, incident edge branch IDs are still collected in the non-excluded endpoint's CC; an edge with both endpoints excluded belongs to no component. Chainable with edge masks.

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

`pygraphc.reduction` implements a node-preserving reduction kernel for multigraphs:
partition by an edge mask, quotient over the blocks, and reduce to the edges that matter
between a set of preserved nodes while every input edge keeps its identity in a series-parallel
provenance tree. The partition step runs in C through
masked connected components; quotient, lift, reduction, provenance folds and scenario
application are the Python tier on top. Node ids are non-negative integers (validated
at construction: `TypeError` for a non-int, `ValueError` for negatives, duplicates and
unknown endpoints), edge ids are opaque hashable values that keep their identity through
every operation, and every result is deterministic by id order.

```python
from pygraphc import (
    MultiGraph,
    Parallel,
    Partition,
    PendantPolicy,
    minimal_toggles,
    paths,
    quotient,
    reduce,
    scenario,
)

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

# Node-preserving reduction: pendant deletion, series merge, parallel merge until
# nothing applies. Every residual edge carries a series-parallel provenance tree.
reduced = reduce(meta, preserved_nodes={1, 3})
(tree,) = reduced.provenance.values()
isinstance(tree, Parallel)                              # True: Parallel({Leaf("p1"), Leaf("p2")})
paths(tree)                                             # {frozenset({"p1"}), frozenset({"p2"})}
minimal_toggles(tree, {"p1": True, "p2": True}, target_closed=False)
# frozenset({"p1", "p2"}): opening the residual edge needs both leaves

# Per-node knobs. A keep action saves a pendant node, a series-ineligible node
# is not merged away between its two neighbours.
star = MultiGraph([0, 1, 2], {"a": (0, 1), "b": (0, 2)})
reduce(star, preserved_nodes={0}, pendant=PendantPolicy("absorb", {2: "keep"})).graph.nodes
# [0, 2]: node 1 folds into node 0, node 2 stays with its edge
path = MultiGraph([0, 1, 2], {"a": (0, 1), "b": (1, 2)})
reduce(path, preserved_nodes={0, 2}).graph.nodes                  # [0, 2]: node 1 merged in series
reduce(path, preserved_nodes={0, 2}, series_ineligible={1}).graph.nodes
# [0, 1, 2]: both input edges survive, node 1 keeps them apart
```

`reduce(graph, preserved_nodes, protected=frozenset(), *, pendant=None, series_ineligible=frozenset(), order=None, engine="c")`:
`reduce(graph, preserved_nodes, protected=frozenset(), *, fold_leaves=True, order=None, engine="c")`
(`engine` is one of `"c"`, `"moves"`, `"rounds"` and `"python"`):

- Every node named in `preserved_nodes`, `protected`, `series_ineligible` or the pendant
  policy's exceptions must be a node of the graph; unknown ids raise `ValueError`.
- Components without a preserved node are removed whole before any move, whatever the
  pendant policy says: a component with no preserved node carries no question.
- Four per-node notions answer four different questions:

  | node is | pendant move at it | series merge at it | parallel merge of its edges |
  | --- | --- | --- | --- |
  | a preserved node | no | no | yes |
  | `protected` | yes | no | no |
  | in `series_ineligible` | yes | no | yes |
  | a `keep` node of `pendant` | no | yes | yes |

  So a preserved node is the only notion that survives every move; `protected` is
  `series_ineligible` plus the parallel block, and the two compose by union on
  the series question; a `keep` node is not a preserved node, since it may still be
  merged away in series, and a protected node is not preserved, since it may
  still be deleted as a pendant.
- `pendant=PendantPolicy(default, exceptions)` decides per node what the pendant
  move does: `"absorb"` hands the removed node's material to the neighbour,
  `"discard"` drops it, `"keep"` leaves the node in the graph with its one
  incidence. The action of a pendant move is the action of the node it removes,
  so material absorbed into a node earlier is discarded with it if that node
  discards. A default plus exceptions rather than a mapping over every node,
  because the decision is uniform for most of a graph. The exceptions are copied
  into a read-only mapping, so a policy hashes and keeps the actions it was
  validated with.
- `fold_leaves=True | False` is the deprecated shorthand for a uniform `"absorb"`
  or `"discard"` policy. Giving both it and `pendant` raises `ValueError`.
- Self-loops take part in no move and leave with their node. Degree counts
  non-loop incidences, so a node with a loop and one further edge is a pendant,
  and a series merge needs exactly two non-loop incidences to two distinct neighbours.
- Under a uniform `"absorb"` policy, every eliminated node appears exactly once in the output:
  in `Reduced.folded_nodes` of a surviving node, as `Series.interior_nodes` of a
  residual tree, or in `Reduced.folded_interior` of such an interior node (the
  material folded into it before its series move). A pendant move passes the
  pendant node, its folded material and the interior nodes of the dropped edge's
  tree with their folded material on to the neighbour. A `"discard"` action drops
  that material, a `"keep"` action leaves the node in the residual with its
  `folded_nodes` entry.
- Every pendant move also records `(neighbour, tree)` in `Reduced.dropped`, the
  provenance tree of the removed edge, so edge material merged before its attachment
  became pendant (a dead-end cycle returning to one node) stays available.
- Edges produced by moves get `VirtualEdgeId`s numbered in creation order; they
  sort after every input edge id and start above any virtual id already present in
  the input, so a residual can be reduced again (with the same preserved nodes it is a fixpoint).
- Deterministic: candidates are processed in increasing node id (or in `order`),
  ties among edges by id order. With no protected node and no `keep` action the
  residual and the folded material are the same for every order. A protected node
  makes them order dependent through its parallel block, which can leave one of two
  competing nodes inert; a `keep` action makes them order dependent because it
  blocks the pendant move while leaving the series move that removes the node, so
  whether the node is reached at degree one or at degree two decides its fate. In
  both cases the id order is the documented answer. A series-ineligible node does
  not make them order dependent: a node offers the pendant move at degree one and
  the series move at degree two, never both, and degrees only fall, so removing the
  series move leaves a degree-two node inert rather than creating a competing pair.

Tree folds: `leaves(tree)`, `paths(tree, cutoff=None)` (series is the product,
parallel the union, the cutoff prunes inside the product), `closed(tree, state)`
(series is AND, parallel is OR) and `minimal_toggles(tree, state, target_closed=...)`
(union where the node type needs every child, cheapest child otherwise, ties by
edge id).

`minimal_toggles(..., togglable_leaves=None)` restricts the fold to the leaves that
may be flipped; the default is every leaf, which is the behaviour without the
parameter. A leaf outside the set keeps the state `edge_closed` gives it, so a node
that needs every child (series to closed, parallel to open) is out of reach as soon
as one child is, and a node that needs one child picks the cheapest among the
reachable children only. The result distinguishes two answers: `frozenset()` means
the tree already takes the target state and nothing has to be toggled, `None` means
no subset of the permitted leaves makes it take the target state.

`series_chain(tree, edge_endpoints, start_node)` walks a series chain and returns one
`SeriesStep(from_node, subtree, to_node)` per position, in order. `edge_endpoints` is
`MultiGraph.endpoints` of the graph the tree was reduced from, and every leaf of the
tree is one of its edges. The two nodes each subtree spans follow from it — a leaf
spans its edge's endpoints, a series node spans what its children span minus the
nodes its merges ate, which are its interior nodes — and with them the direction each
child runs in. The tree does not record that direction itself: the stored order of a
series node is the direction of the merge that created it, and a later merge can
reach it from either end, so a walk that trusts the stored order crosses a nested
chain backwards. A `Leaf` is a chain of one step. A series node nested in a series
node is a sub-chain and is flattened into the sequence, so index `i` addresses the
`i`-th subtree along the whole chain and the length is the number of positions on it;
a `Parallel` child is one position, because its children are unordered alternatives.
Starting from the other endpoint returns the reversed sequence with every step
reversed, so the two walks of a chain are mutual reverses. A `Parallel` tree, a start
node that is not an endpoint of the tree, a leaf that is not an edge of the graph, a
subtree that does not span exactly two nodes, a series node the walk reaches at a node
neither of its outer children touches and a subtree whose spanned pair is not the pair
of chain nodes its position sits between raise `ValueError`. The last of
those is what makes the refusal sound: spanning two nodes leaves a tree whose
subtrees do not meet end to end, and walking one returns a well-formed sequence whose
steps name pairs the graph does not join.

`Series` and `Parallel` compare by identity (every tree node is created once,
by the move that produces it) and hash by cached structure; the folds, `repr` and
`tree_records(tree)` / `tree_from_records(records)` are iterative, so chains deeper than
the recursion limit are fine. `tree_records` is the canonical serialisable form (a
post-order log with parallel children ordered by smallest leaf id, independent of the
hash seed) and the way to compare two trees structurally; `pickle` and `copy.deepcopy`
recurse and are not suitable for deep trees.

`lift(partition, attribute, combine)` combines node attributes per block in
increasing node order, so a non-commutative `combine` still gives reproducible
results. `Partition.compose(finer)` expresses a second quotient level.

#### The reduction loop in C (`engine="c"`)

The three moves run in the C tier and return a flat operation log; Python folds
that log once into the `Leaf`/`Series`/`Parallel` trees and the `Reduced` above,
so the payload algebra stays in Python and nothing about `reduce` changes.
`engine="python"` selects the pure-Python worklist, which is the reference
semantics of the moves and what a caller-supplied `order` selects, since the C
work queue is fixed to increasing node index.

On 20 000 nodes, 25 000 edges and 200 preserved nodes: **0.022 s** for `engine="c"`,
0.087 s for `engine="python"` and 0.146 s for a straightforward networkx
implementation of the same three moves, so 6.7x networkx and 3.9x the Python
engine. A million nodes with 1.25 million edges reduce in 3.3 s, of which
0.29 s is the C loop over 1.75 million operations; the remaining nine tenths
are the fold, which allocates one tree node per operation because the residual
provenance trees are the kernel's output.

`Graph.series_parallel_reduce(preserved_mask, protected_mask, *,
pendant_keep_mask=None, series_blocked_mask=None)` exposes the loop on its own and
returns a `ReductionLog` of twelve int32 `memoryview`s. The four masks hold one byte
per node index and mark membership by a non-zero byte, and are read as buffers, so a
list of node ids is not a mask. A missing `protected_mask`, `pendant_keep_mask` or
`series_blocked_mask` is the empty set, while `None` in place of the preserved node mask
raises `TypeError`, because a reduction without a preserved node deletes every component;
`GraphView` reduces under its own edge and node masks. Only the structural half of
the pendant policy is a mask: whether a removed pendant's material is absorbed or
discarded changes no move, so the log reports the absorbing neighbour either way,
there is no `fold_leaves` here, and `reduce` reads the action off the policy per
removed node while folding the log.
#### The same moves as primitives (`engine="moves"`, `engine="rounds"`)

`Graph.series_parallel_state(preserved_mask, protected_mask)` puts that same
incidence structure behind an opaque `ReductionState` handle and exposes one
short primitive per step over it, so the fixpoint loop can live in Python
without rebuilding the graph between moves. `engine="moves"` asks for the next
applicable move and applies it, one crossing per move; `engine="rounds"` asks
for every currently applicable and mutually independent move of one kind and
applies the whole batch, one crossing per kind per round. Two eligible nodes
conflict exactly when one is a neighbour of the other, so a batch admits
eligible nodes in increasing index while none of their neighbours is admitted
yet; the sparse graphs below take five to six rounds.

```python
state = graph.series_parallel_state(bytes([1, 0, 0, 1]), bytes(4))
move = state.next_move()             # (kind, node, edge_a, edge_b, neighbour_a, neighbour_b)
created, parallel_pending = state.apply_move(move)
batch = state.batch_moves(1)         # every independent series move, as int32
state.apply_batch(1, batch)
log = state.log()                    # the same ReductionLog as the loop above
state.free()                         # the handle goes inert, not dangling
```

The primitives write into raw arrays, so every index handed back is checked
first: an edge must be live and run between the two distinct live nodes named,
a series move must name two distinct edges, the members of a parallel merge
must be live and in increasing slot order, and no move may remove a preserved node.
Anything else raises `ValueError` before that move writes anything.

`apply_batch` checks and applies one record at a time, so the guarantee is per
record and not per batch: a batch whose third record is invalid raises with the
first two applied. Every applied record was checked in full, so the state is
valid rather than half-written, but it is not the state the batch described —
read the log or rebuild the batch rather than assuming the call was atomic. The
engines here only ever feed back a batch that `batch_moves` produced, which is
why this has no caller consequence today.

On 20 000 nodes, 25 000 edges and 200 preserved nodes: 0.022 s for `engine="c"`,
**0.025 s** for `"moves"` and **0.023 s** for `"rounds"`; at a million nodes
2.02 s, 2.19 s and 1.92 s. Only a tenth of a call is the structural work and
four fifths of it is the fold, so 500 000 boundary crossings cost under a
tenth of the total and 18 cost nothing measurable. `"moves"` applies the moves
in the monolith's order and reproduces its `Reduced` exactly; `"rounds"`
changes the order the merges happen in, which leaves the residual, the leaves
of every residual edge and the folded payload of every surviving node alone
but may split `dropped` differently across pendant moves and nest a series
chain differently.

`Graph.series_parallel_reduce(preserved_mask, protected_mask)`
exposes the loop on its own and returns a `ReductionLog` of twelve int32
`memoryview`s. The two masks hold one byte per node index and mark membership by
a non-zero byte, and are read as buffers, so a list of node ids is not a mask;
`None` in place of the preserved node mask raises `TypeError`, because a reduction
without a preserved node deletes every component. `GraphView` reduces under its own
edge and node masks. There is no `fold_leaves` here: the log records which
neighbour absorbs a pendant payload either way and `reduce` decides whether to
apply it.

```python
graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
log = graph.series_parallel_reduce(bytes([1, 0, 0, 1]), bytes(4))

log.op_kind.tolist()         # [0, 0, 0, 1, 1] — three leaves, then two series merges
log.interior_node.tolist()   # [-1, -1, -1, 1, 2] — the nodes the series merges eliminated
log.surviving_nodes.tolist() # [0, 3] — both preserved_nodes, joined by one virtual edge
log.residual_op.tolist()     # [4] — that edge comes from the last operation
```

- Eight arrays hold one entry per operation in the order the moves apply:
  `op_kind` (0 leaf, 1 series, 2 parallel, 3 pendant), `left` and `right`
  (child operation ids, -1 when absent), `endpoint_u` and `endpoint_v` (the
  oriented endpoints of the virtual edge the operation produces, -1 when it
  produces none), `interior_node` (the node a series merge eliminated or a
  pendant deletion removed), `leaf_edge_index` (the input edge index of a leaf)
  and `absorber` (the neighbour a pendant payload moves to).
- A parallel merge of more than two edges is a left-deep chain of binary
  operations of which only the last carries endpoints, so the fold recognises
  the earlier ones as links of one k-ary parallel node.
- `residual_op`, `residual_u` and `residual_v` hold one entry per surviving edge
  and `surviving_nodes` the surviving node indices in increasing order.

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

#### Domain independence check

The reduction kernel was designed while looking at one application, so
`tests/integration_tests/` keeps four cases from unrelated fields as the
standing check that nothing domain-specific leaked into the API. Each models a
real problem and asserts statements that are true about the modelled world, not
numbers that happen to fall out of the code:

| Case | Field | Exercises |
|------|-------|-----------|
| [Seven bridges of Konigsberg](tests/integration_tests/test_koenigsberg_bridges.py) | civil infrastructure, 1736 | `Partition`, `quotient` keeping parallel bridges distinct, `lift` of a per-district headcount, `minimal_toggles` over a parallel provenance node, and the irreducible four-node core where series and parallel moves stop |
| [Caffeine scaffold](tests/integration_tests/test_caffeine_scaffold.py) | cheminformatics | `cycle_basis` and `bridges` to find the ring system, then `reduce` with `fold_leaves` as the side-chain stripping of a Bemis-Murcko decomposition |
| [Perfect maze](tests/integration_tests/test_maze_solving.py) | puzzles, robot path planning | pendant and series moves collapsing a spanning tree to one edge, `paths` over the resulting series chain as the maze solution, `scenario` as a blocked-passage query |
| [Braided maze](tests/integration_tests/test_braided_maze.py) | maze design, game level design | the same maze with three extra passages: the parallel move, `paths` returning all eight routes, `scenario` showing a passage on a loop is no longer a bridge, `minimal_toggles` as the cheapest set of passages to wall up, dead ends pruned into the folded material |

```bash
pytest tests/integration_tests/ -v          # or: pytest tests/ -m integration
```

The pre-commit hooks deselect the marker to stay fast, so CI carries the check:
the `integration` job in `.github/workflows/ci.yml` runs the directory on every
supported Python version.

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

### Workflow profiles

`benchmarks/profile_workflows.py` profiles the end-to-end workflows the library
supports — build a graph, partition it, sweep scenarios, run the full
partition/quotient/lift/reduce pipeline, the weighted and structural families,
path enumeration, DAG structure learning — and writes a per-phase breakdown of
every one of them. Phases, not a call tree: the split that matters here is the C
kernel against the Python objects built around it, and a flat profile hides it.

```bash
python benchmarks/profile_workflows.py               # full sizes, writes profiles/
python benchmarks/profile_workflows.py --size guard   # the sizes the instruction tier counts at
python benchmarks/profile_workflows.py --list         # the registry
```

**The artifacts** land in `profiles/` (gitignored, created on demand):

| File | What it is |
|------|------------|
| `summary.md` | the table to open first: total wall time, every phase with its time and its share, the `tracemalloc` peak and the networkx reference per workflow |
| `summary.json` | the same numbers machine-readably |
| `<workflow>.prof` | raw `cProfile` output — `python -m pstats` or `snakeviz profiles/<workflow>.prof` |
| `<workflow>.txt` | that profile's top 25 entries by cumulative and by total time |

**Two share columns.** The phase that generates a workflow's random inputs is
the harness, not the library, and on most workflows it is a third to nine
tenths of the wall time — 87% of `connected_components` at 5 000 nodes and
still 79% at 200 000, so it is not an artefact of the small sizes. Every table therefore
carries the share of the workflow and the share of the *library* time, the
total without those generation phases. Read the second one.

**How to read them.** Start with the phase shares, not the totals. A workflow
whose C phase is a few percent and whose Python phase is most of the runtime is
not a slow kernel, it is a caller paying for objects — the reduction spends
about a tenth of its `reduce` step in the C loop and the rest folding the
operation log into provenance trees, which is why `framework_pipeline` and
`reduction_log` run the same reduction and differ only in their tails. Times are
best of three runs after a warm-up, with the profiler detached; the `.prof` and
`.txt` files come from a separate run and carry the profiler's overhead, so use
them to find *which* call, never to quote a duration. `release inputs` is the
teardown of the workflow's own inputs, attributed on purpose so the phases add
up to the total.

**The guard.** The instruction tier (see [Performance tests](#performance-tests))
runs every workflow at its guard size under callgrind and counts the
instructions of each phase, named `workflow/<workflow>/<phase>`. CI compares
every phase against the merge base, split into the same three buckets as every
scenario, and fails on a bucket that moves beyond its threshold, while every
smaller change is listed too; nothing is compared against a stored baseline,
so no baseline needs regenerating.

### Comparing the reduction engines

`reduce` has four interchangeable engines and the harness has one workflow per
engine — `reduce_engine_c`, `reduce_engine_moves`, `reduce_engine_rounds`,
`reduce_engine_python` — identical in the seeded graph, the preserved node set, the
sizes and the phase names, differing in the engine and in nothing else.

```bash
python benchmarks/profile_workflows.py --compare-engines      # writes profiles/engine_comparison.md
python benchmarks/profile_workflows.py --rebuild-comparison   # rewrites it, measuring nothing
```

`profiles/engine_comparison.md` holds the phase table for all four engines side
by side at 20,000, 100,000 and 1,000,000 nodes; a `pstats` diff of each
interesting pair, naming the functions whose time differs, the functions only
one profile has, and the number of calls each engine makes across the
Python-to-C boundary; and the answers, with numbers, to what the fold costs,
where `"moves"` loses its margin and why `"rounds"` matches the monolith. The
two smaller sizes are profiled under `cProfile`; the largest is timed only, and
the artifact says which is which. Where an engine has no counterpart to a phase
the phase is recorded as zero rather than omitted, so the four tables line up
row by row. The numbers land in `profiles/engine_comparison.json` next to the
artifact, and `--rebuild-comparison` rewrites the prose from them without
re-measuring.

Two things the artifact contradicts, both worth knowing before quoting the
older numbers: the fold is about nine tenths of a warm `reduce` and about three
quarters of a cold one, so "four fifths" is between the two rather than either;
and the fold-phase noise floor — measured from two engines that fold a
byte-identical log — is several percent, which is the same order as the
differences between the C-backed engines. Read a few percent between them as
noise, not as a ranking.

### Profiling is the practice

Profile whenever an algorithm is exchanged, newly written or updated, and report
the profile with the change — not a wall-clock number on its own. A wall-clock
number says a change got faster; a profile says where the time went, which is
the part a reviewer can argue with.

If runtime or memory can be improved without adding complexity and without
costing maintainability or readability, do it. No ticket, benchmark target or
other justification is needed for that kind of improvement.

Read a bad profile as evidence about the design. Time or memory concentrated
somewhere surprising is usually not a tuning problem but a symptom: the code at
that spot is doing more than the problem asks for. So the first question at a
hot spot is not how to make it faster, it is whether it is more complicated than
it needs to be. Genuine complexity does exist and is fine where the problem
really carries it, but that is the rarer case.

Two measurements from this repository make the case better than the argument
does. Most of a `reduce` call turned out to be building provenance trees, not
the C loop everyone assumed it was — about three quarters of it when the kernel
graph is built on demand and about nine tenths when it is already there, which
is the finding the engine comparison above reports. And a source-to-target
shortest path that inspects a few hundred edges turned out to be settling every
node nearer the source than the target: the search was answering a larger
question than the one asked, and the fix was to stop asking it — two searches
meeting in the middle, not a faster inner loop.

The mechanism is concrete: add or update a workflow in
`benchmarks/profile_workflows.py`, commit the artifact comparison, and state in
the change what moved and why. `profiles/engine_comparison.md` and the
`--compare-engines` workflows above are the worked example.

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pip install networkx pgmpy
python benchmarks/networkx_baselines.py  # networkx baseline per operation, the README tables
python benchmarks/bench_dag_learn.py  # hill-climb K2: pygraphc vs pgmpy
```

## Tests

After the [development setup](#development-setup):

```bash
pytest tests/unit_tests/ -v
```

`tests/unit_tests/test_networkx_differential.py` checks one graph operation per
test against networkx on seeded random graphs; it needs `networkx` installed
and is skipped otherwise.

### Performance tests

`tests/performance_tests/` holds three tiers, all marked `performance`:

- **complexity/**: walltime at doubling sizes, chosen per operation until
  one call takes 0.1 s, asserting the fitted scaling exponent (at most 1.5 for linear and n log n work, which reaches up to 1.42 once the
  working set leaves the CPU caches; quadratic cases measured 1.73 and more). Each
  operation runs on several shapes (path, star, singletons, lattice, random),
  because a superlinear implementation hides on the axis it is quadratic in:
  degree, depth, component count. Superlinear behaviour often shows only at
  large sizes, so this tier stays on walltime.
- **instructions/**: deterministic callgrind instruction counts of small and
  medium scenarios and of every phase of the workflows in
  `benchmarks/profile_workflows.py`. CI builds the merge base and the branch
  with the same packages and with debug info (`CFLAGS=-g`, which leaves the
  generated code unchanged), on x86-64 and arm64, and splits every count by the
  source file of each instruction, inlined lines included:

  | Bucket | Source files | Fails beyond |
  |--------|--------------|-------------:|
  | `algorithm` | `src/pygraphc/_core_src/algorithms/` | 0.1% |
  | `boundary` | the rest of pygraphc's C sources: entry points, `conversion/`, models | 0.5% |
  | `cpython` | CPython, the C library, anything without debug info | 0.5% |

  A count fails in either direction. Within one binary the counts are exact; a
  changed C extension moves unchanged work by up to about 1.1% through the
  allocator, and that noise lands in `cpython`, so a C change often needs
  review even when only the layout moved. pygraphc's own code does not depend
  on addresses, which is why the algorithm bucket can be held to 0.1%. A total
  alone hides kernels: in `strongly_connected_components/directed_acyclic_20k`
  the algorithm is a ninth of the count, so a 10% slower kernel moved the total
  by about 1.1%, inside the noise. The other thresholds stay below the
  noise on purpose, because real code generation changes of 0.5 to 1% matter.
  `algorithms/` only holds code that never reads or creates a Python object
  (raising an exception is allowed); parsing and result building live in
  `conversion/` and the entry points. Every count that moved is in
  one of two aligned tables, more instructions and fewer instructions, each
  largest change first; counts that round to 0.00% are only counted below
  them. A pull request labelled
  `accept-instruction-change` still gets the table but does not fail.

  What the counts cannot see: they measure portable work, compiled with GCC.
  The same instructions can run slower on one machine (splitting `_core.c`
  left topological sort's instructions, simulated cache misses and branch
  mispredictions unchanged, yet 8 to 14% slower on an Apple M3, most likely
  through code alignment), and clang can generate different code than GCC
  (four counts moved by 0.5 to 1.3% under clang only). Machine-specific
  effects show only in walltime, in the experience tier once they are large.
  The design claims ("a mask costs at most 15%") are instruction-count ratios
  in `test_claims.py`. Needs valgrind, so it runs on Linux; on macOS:

  ```bash
  podman run --rm -v "$PWD":/source:ro python:3.13-slim bash -c '
    apt-get -qq update && apt-get -qq install -y gcc valgrind
    cp -r /source /work && CFLAGS=-g pip install -q "/work[dev]" && cd /work/tests
    pytest performance_tests/instructions -v'
  ```
- **experience/**: one realistic million-node input per workflow, end to end,
  against a generous budget. It catches a second turning into a minute, not
  regressions.

```bash
cd tests && pytest performance_tests/complexity -v   # or instructions, experience
```

### Finding where an instruction count moved

The instruction tier in CI says *that* a count moved; `dev_helpers/compare_profiles.py`
says *where*. It builds a base revision and your working tree (uncommitted
changes included) in a Linux container, measures the chosen scenarios with the
same harness as CI, in the same three buckets, and reports, for every scenario
with a bucket beyond its threshold, the source files (named relative to
`_core_src/`) and functions whose own instructions changed most:

```bash
python dev_helpers/compare_profiles.py --only bridges,articulation_points            # against origin/main
python dev_helpers/compare_profiles.py --only bridges --base HEAD~1                   # against any revision
python dev_helpers/compare_profiles.py --only bridges --platform linux/amd64          # emulated x86-64, slow
```

`--only` takes the same comma-separated substrings as `compare.py`. The report
and a per-line annotation of every measurement land in `profiles/comparison/`
(git ignores `profiles/`). It needs podman or docker, nothing else; it runs on
macOS too, natively on the machine's architecture unless `--platform` says
otherwise. Use it as soon as a count or a timing rises, before calling a
change noise. The counts come from the container's compiler, so they can
differ from CI's by layout effects of up to a few percent: CI decides whether
a count moved, this shows why.

## Multi-terminal edge case behavior (up for discussion)

These are the answers the library gives today in situations where a
different answer would also be reasonable. They are decided so that the
library works now, and they may change once the people who use it have said
what they need. Each entry is pinned by a test in
`tests/unit_tests/test_multi_terminal_edge_cases.py` (`test_edge_case_<n>_…`).
Every test that depends on one of these answers carries the marker
`multi_terminal_edge_case`:

```bash
pytest -m multi_terminal_edge_case
```

To change an answer, change its test first, then the code. "Decided by"
says whether the user decided the answer or Claude chose it while
implementing; the D and O numbers refer to `docs/multi_terminal-decisions.md`.

| # | Situation | Current behavior | Alternatives | Decided by |
|---|-----------|------------------|--------------|------------|
| 1 | Two-terminal edge with both nodes masked | Has no active terminal and belongs to no component (D18). | A component without nodes that holds only the edge, as before this pull request. | user |
| 2 | Edge with exactly one active terminal | Connects nothing but belongs to the component of that terminal's node. It is never critical and never makes its node critical (masking case 3). | The edge belongs to no component. | user (membership), Claude (critical elements) |
| 3 | Two-terminal edge with one terminal masked | Connects nothing, like a masked edge, but still belongs to the component of its active end (masking case 5). | Treated exactly like a masked edge, so it belongs to no component. | user |
| 4 | Edge with a single terminal | Allowed on undirected graphs: it connects nothing and belongs to its node's component. A directed edge needs an in-terminal and an out-terminal (D12). | Reject single-terminal edges altogether. | user |
| 5 | Edge with all terminals at one node | Connects nothing and belongs to that node's component. It forms no block and is never critical (masking case 8). | Count it as a loop that puts the node into a block. | Claude |
| 6 | Two terminals of one edge at the same node | The terminals stay distinct: masking one keeps the node attached through the other (masking case 7). The edge `(1, 1, 2)` is critical. | Merge terminals at the same node into one. | user (terminals by index), Claude (critical) |
| 7 | Terminal at a node that is not in the graph | Rejected when the graph is built, as for two-terminal edges; "not connected" is a terminal mask (O1). | Accept the terminal and leave it inactive. | Claude (recommended, not confirmed) |
| 8 | Directed edge with several in-terminals | OR: one reached active in-terminal reaches every active out-terminal (D5). | AND: the out-terminals are reached only once all in-terminals are reached. | user |
| 9 | Directed edge without an active out-terminal (or in-terminal) | Transmits nothing, yet for weakly connected components its active terminals still connect to each other. | Connects nothing at all, even for weak components. | Claude |
| 10 | Weights on a graph with a multi-terminal edge | One weight per terminal for every edge, including the two-terminal edges of that graph. Crossing a two-terminal edge costs its two terminal weights. A graph of only two-terminal edges takes one weight per edge (D11). | One weight per edge everywhere, or per-terminal weights also on graphs of two-terminal edges. | user (per terminal), Claude (two-terminal edges in such graphs) |
| 11 | Path through a multi-terminal edge | Costs the entry terminal's weight plus the exit terminal's weight (masking case 10). | A weight per pair of terminals, or per edge. | user |
| 12 | Critical multi-terminal edge | Critical when removing the whole edge splits the nodes of its terminals. This holds even inside a block with a cycle: in `(1, 2), (2, 3), (1, 2, 3, 4)` the multi-terminal edge is critical, because node 4 hangs on it alone. | Critical only when it lies on no cycle, as a bridge would be. | Claude |
| 13 | Biconnected blocks at a multi-terminal edge | Blocks never split at a multi-terminal edge: a tree-shaped edge with k terminals is one block of k nodes (D14), and a terminal node hanging on an edge of a cycle joins the cycle's block. | Split blocks at the edge, as `nodes_on_simple_paths` does. | user (tree-shaped), Claude (cycle) |
| 14 | Two-edge-connected components | The nodes that no single edge removal separates (as for graphs). A critical multi-terminal edge can lie on a cycle, so it is not removed whole: in `(1, 2, 4), (2, 3, 5), (1, 3)` both multi-terminal edges are critical, yet 1, 2 and 3 stay together. Corrected after the web research: removing every critical edge whole had split them. | Remove every critical edge whole (wrong when a critical edge lies on a cycle). | Claude, corrected |
| 15 | Simple paths through a multi-terminal edge (`nodes_on_simple_paths`) | A path uses each multi-terminal edge at most once and enters and leaves through one terminal each, so in the tree-shaped edge `(1, 2, 3)` the paths from 1 to 2 do not visit 3. | Allow a path to pass the same edge twice through different terminals. | Claude |
| 16 | `edge_indices_at(node_ids)` | Returns the edges whose set of terminal nodes equals the given set. Order, repeats and direction do not matter, and an unknown node matches nothing. | Match with repeats counted (multiset), match edges with a terminal at all given nodes (superset), or respect direction. | Claude |
| 17 | Pair methods on a graph with a multi-terminal edge | `bridges()`, `bridges_with_branch_ids()` and `edge_indices(u, v)` raise for the whole graph, even when asked about a two-terminal edge, and name `critical_edges()` or `edge_indices_at()` instead (D13). | Answer for the two-terminal edges and leave out the multi-terminal ones. | user |
| 18 | Methods not yet defined on terminals | They raise `NotImplementedError` on a graph with a multi-terminal edge and on any view with a terminal mask, rather than answering for part of the graph (O3, D19). | Answer on the two-terminal edges only. | user |
| 19 | Negative terminal weights | Allowed when every pass costs at least 0: every two terminals of an edge (directed: every in-terminal and out-terminal) sum to at least 0, so an undirected edge has at most one negative terminal; otherwise `ValueError`. | Reject every negative terminal weight. | user |
| 20 | Leaving through the entry terminal | Never: a pass goes from one terminal to another. Without negative weights it could not shorten a path anyway. | — | user |
| 21 | Passing an edge again right after leaving it through its negative terminal | Not allowed: in `(1, 2, 3)` with weights `[3, -1, 2]` the path from 1 to 3 costs 5, not `(3 − 1) + (−1 + 2) = 3`. | Allow it: each pass is legal on its own. | user |
| 22 | Coming back to the negative terminal's node another way | Allowed: only the immediate second pass is banned, so a shortest path may visit a node twice (`[1, 2, 4, 2, 3]` in the test). | Ban every second pass through the same edge, which makes shortest paths a constrained, much harder problem. | Claude |
| 23 | Edge with an in- and an out-terminal at the same node (directed) | A cycle, like a self-loop: topological sort raises, the node is its own strongly connected component. | Ignore the pair, so that only cycles through other nodes count. | user (step C proposal) |
| 24 | DAG longest path without weights | A pass through a multi-terminal edge counts 1, as a two-terminal edge does. | Count 2 (entry and exit), or count by terminals. | user (step C proposal) |
| 25 | Directed edge without an active in-terminal in a topological sort | Constrains nothing: its out-terminal nodes do not wait for it. | Keep its out-terminals after its (masked) in-terminals anyway. | user (step C proposal, case 9) |
| 26 | `degree` at a node with two terminals of one edge | Counts terminals: 2, as for a self-loop. Equals XGI and HyperNetX ("edges containing the node") wherever an edge reaches a node once, and NetworkX on two-terminal graphs. | Count edges: 1, as XGI and HyperNetX do on hypergraphs without repeated nodes. | Claude (research: the conventions split here) |
| 27 | A node as its own neighbor | Never, neither through a self-loop nor through two terminals of one edge (XGI, and pygraphc before). | Include it, as NetworkX does for a self-loop. | Claude (research: the conventions split here) |
| 28 | Two terminals of one edge at one node in `cycle_basis` | A cycle `[node]`, like a self-loop; it counts toward the cycle rank. | No cycle: a strict Berge cycle needs distinct nodes. | Claude (research: the conventions split here) |
| 29 | Two edges sharing two nodes in `cycle_basis` | A cycle of length 2, like two parallel two-terminal edges. | No cycle: some authors define Berge cycles from length 3. | Claude (research: the conventions split here) |
| 30 | Which cycle space `cycle_basis` spans | The incidence graph's: Berge cycles, rank terminals − nodes − edges + components. | The hypercycle (oriented matroid) notion of reaction networks, which captures dependencies the incidence graph does not. | Claude (research: the community defaults to the incidence graph) |

### What the literature says about these cases

A web search on 2026-10-31 checked each entry against hypergraph theory,
hypergraph libraries and power system tools. Sources:
- [DPP]: Dewar, Pike, Proos, *Connectivity in Hypergraphs*, arXiv:1611.07087.
- [BS]: Bahmanian, Šajna, *Connection and separation in hypergraphs*, Theory and Applications of Graphs 2(2), 2015.
- [GLNP]: Gallo, Longo, Nguyen, Pallottino, *Directed hypergraphs and applications*, Discrete Applied Mathematics 42, 1993.
- Libraries and tools: HyperNetX, XGI, pandapower, PowerModels.jl, OpenDSS, and CIM (IEC 61970).

**Supported**
- **1–3:** an edge without an active terminal belongs to no component, and one with a single active terminal stays with its node.
  - [BS]: empty edges do not affect connectivity.
  - [DPP]: weak vertex deletion keeps the shrunken edge.
  - CIM `Terminal.connected` and OpenDSS `Open` open a single terminal.
- **8:** OR semantics is the "path" connectivity of [GLNP]. AND semantics would be their B-connectivity.
- **12:** a multi-terminal edge can be critical while lying on a cycle. These are the weak cut edges of [BS], Def. 3.15–3.18.
- **13:** blocks never split at a multi-terminal edge. [BS], Thm. 3.37 and 3.41: the blocks of a hypergraph are not the blocks of its incidence graph.
- **15:** using each edge at most once is a Berge path: distinct nodes and distinct edges.
- **17–18:** raising instead of answering partially matches networkx's `not_implemented_for`.
- **19:** negative star impedances of three-winding transformers are a known artefact; the pair sums are the measured short-circuit impedances, which are positive.
- **20:** no U-turn matches the walk definition of [DPP] and turn-restricted routing (pgRouting TRSP).

**Mixed or without a convention**
- **4:**
  - Single-terminal edges are common: HyperNetX and XGI allow them and offer helpers that remove them.
  - [GLNP] also allows a directed edge with no out-terminal.
- **5:** networkx leaves a self-loop node out of every block. [BS] (Lemma 3.31) makes a single-node edge its own block, so that blocks cover every edge.
- **6:** [DPP] keeps edges as multisets, which matches distinct terminals at one node. [BS], XGI and HyperNetX merge repeated nodes.
- **7:** networkx and XGI create unknown nodes. pygraphc rejects them because its node set is fixed.
- **10:** per-terminal weights are common: HyperNetX incidence weights, CIM transformer ends. Switching the whole graph to per-terminal weights has no precedent.
- **11:** pandapower uses the same entry + exit rule for three windings. A star is exact only up to three windings; CIM requires mesh impedances for more.
- **16:** matching by node set follows HyperNetX and XGI. Ignoring direction departs from networkx's `has_edge(u, v)`.
- **21–22:** banning only the immediate second pass computes a shortest *non-backtracking walk* in the star equivalent, not a shortest simple path. A detour that costs less than −2·w_n still counts the negative weight twice, which the test of entry 22 shows (4 instead of 5). The exact simple-path version is polynomial but needs T-join or matching algorithms (Schrijver, *Combinatorial Optimization*, ch. 29), not Dijkstra.

**Neighborhood queries and cycles (research of 2026-11-01)**
- **Supported:** neighbors as the nodes sharing an edge (XGI `neighbors`, HyperNetX s-adjacency with s = 1); in- and out-degree as the backward and forward stars of [GLNP] and XGI's `DiHypergraph`; Berge cycles as the cycle notion and the incidence (Levi) graph's cyclomatic number as the cycle rank.
- **Split, decided by Claude and listed above:**
  - 26: degree by edges (XGI, HyperNetX on sets) or by incidences with multiplicity (hypergraphs with repeated vertices, NetworkX self-loops).
  - 27: NetworkX lists a self-loop node among its own neighbors, XGI does not.
  - 28–29: whether a Berge cycle may repeat a node or have length 2 depends on the author.
  - 30: reaction-network papers define a hypercycle space that differs from the incidence graph's.

**Contradicted, and corrected**
- **14:** removing every critical edge whole gave wrong two-edge-connected components when critical edges share a cycle. Corrected in this pull request.

**Naming (decided, D22):** pygraphc calls one attachment of an edge to a node a
*terminal*, as CIM and OpenDSS do. Hypergraph papers say *incidence*, XGI
*member*. The draft of this pull request said *port*, but in circuit theory a
port is a pair of terminals, so a "two-port edge" would have been a circuit
one-port.
