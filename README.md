# cgraph

Fast, general-purpose graph algorithm library implemented in C, callable from Python. Zero runtime dependencies.

Edges use **original node IDs** — no manual index mapping needed. A C-side hash map translates IDs to internal indices automatically.

## Benchmarks

### vs networkx (graph algorithms)

End-to-end from domain objects using split lists (gather + algorithm). Sparse random graphs, ~3 edges per node.

| Algorithm | Nodes | cgraph | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Connected Components | 1K | 0.0001s | 0.001s | **17x** |
| Connected Components | 10K | 0.001s | 0.011s | **21x** |
| Connected Components | 100K | 0.011s | 0.313s | **29x** |
| Connected Components | 1M | 0.150s | 6.83s | **46x** |
| Bridges | 1M | 0.401s | 21.42s | **53x** |
| Articulation Points | 1M | 0.350s | 8.97s | **26x** |
| BFS | 1M | 0.165s | 13.62s | **82x** |
| Dijkstra | 1M | 0.390s | 11.11s | **29x** |
| Edge paths (cutoff=5) | 80 | 0.00005s | 0.006s | **~100x** |

### vs pgmpy (DAG structure learning)

Hill-climb with K2 scoring on binary variables. Both produce identical DAGs.

| Scenario | cgraph | pgmpy | Speedup |
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

**17x–900x faster** than the standard Python packages, with identical results.

## Installation

```bash
pip install -e .
```

Or from GitHub:

```bash
pip install git+https://github.com/lejansenGitHub/cgraph.git
```

Requires a C compiler (the extension is compiled at install time with `-O3`).

## API

### Structural algorithms

```python
from cgraph import (
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
from cgraph import connected_components_with_branch_ids

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

# Exclude nodes by ID — removed from output but edges still connect
view = g.without_nodes([2])
list(view.connected_components_with_branch_ids())
# [({1, 3}, {100, 200})]  — node 2 gone, but edges still connect 1 and 3

# Combine both — exclude branches and nodes by their domain IDs
view = g.without_branches([200]).without_nodes([1])
list(view.connected_components_with_branch_ids())
# [({2}, {100}), ({3}, set())]
```

### Weighted algorithms

```python
from cgraph import (
    shortest_path,
    shortest_path_lengths,
    multi_source_shortest_path_lengths,
    eccentricity,
)

node_ids = [0, 1, 2, 3]
edges = [(0, 1), (1, 2), (2, 3)]
weights = [1.0, 2.0, 3.0]

# Shortest path (Dijkstra with C binary heap)
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
from cgraph import Graph

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

### Edge-masked views (exclude edges without rebuilding)

Create lightweight views that exclude edges from the graph without rebuilding the CSR. The base graph is never mutated — views overlay a byte mask on the shared adjacency structure.

```python
from cgraph import Graph, GraphView, for_each_edge_excluded

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
| Dijkstra (single pair) | 6.9ms | 7.3ms | +7% |
| SSSP lengths | 13.6ms | 13.8ms | +2% |

The key win is avoiding O(V + E) graph rebuild per edge modification.

Parallel edges are supported — each edge is tracked by ID, so two edges between the same pair of nodes are handled correctly (e.g. for bridges, Dijkstra weight selection).

### MultiGraph support

cgraph natively supports parallel edges (multigraphs). Each duplicate edge gets a unique index — no deduplication. All algorithms handle them correctly:

```python
from cgraph import Graph

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

Exclude nodes and all their incident edges without rebuilding the CSR. Chainable with edge masks.

```python
g = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])

# Exclude node 2 — edges (1,2) and (2,3) are also excluded
view = g.without_nodes([2])
list(view.connected_components())  # [{0, 1}, {3, 4}]

# Chain with edge masks
view = g.without_edges([0]).without_nodes([3])
view.bfs(1)                        # [1, 2]

# BFS/Dijkstra raise ValueError if source is excluded
view = g.without_nodes([0])
view.bfs(0)                        # ValueError: source node is excluded
```

### DAG structure learning (Bayesian networks)

Learn directed acyclic graph (DAG) structures from discrete data using greedy hill-climb search with K2 Bayesian scoring. Implemented in C — drop-in replacement for pgmpy's `HillClimbSearch` with identical results and orders of magnitude faster.

The algorithm:
1. Start with an empty DAG (no edges)
2. Each iteration, evaluate all legal single-edge operations (add, remove, flip) and pick the one with the highest K2 score improvement
3. A tabu list prevents cycling by forbidding recently reversed operations
4. Stop when no operation improves the score by more than `epsilon`

```python
from cgraph import hill_climb_k2, estimate_cpds, k2_local_score

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

### Edge-path enumeration (edge-disjoint paths)

Find all paths from source to targets where each edge is used at most once. Returns paths as lists of edge indices. By default, nodes may be revisited (critical for multigraphs). Use `node_simple=True` to restrict to node-simple paths where each node is visited at most once.

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

Every cgraph call has three cost phases. Connected components at 1M nodes, sparse random graph (~3 edges per node):

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

- **Gather** is pure Python — extracting `node_a`/`node_b` from your objects. cgraph can't optimize this, but the interface choice affects it (tuples cost 71ms, split lists cost 35ms).
- **Parse** is the C-side input handling — building the node-ID hash map and translating edges to internal indices. Numpy skips per-element Python unpacking via buffer reads.
- **C algorithm** is the actual computation (union-find + building Python `set` results). Fixed cost, identical across all interfaces. The `set` construction via `PySet_Add` accounts for ~50ms of the 62ms — this is the CPython floor for creating hash-based sets of 1M elements.

### Performance across graph topologies

End-to-end from `Branch` objects using split lists. Connected components at 1M nodes:

| Scenario | Components | Gather | cgraph | Total |
|----------|-----------:|-------:|-------:|------:|
| 1 component (all connected) | 1 | 25ms | 30ms | 55ms |
| 10 components (100K each) | 10 | 22ms | 27ms | 48ms |
| 1K components (1K each) | 1,000 | 21ms | 23ms | 44ms |
| 1 dominant + 100K isolated | 100,001 | 19ms | 54ms | 73ms |
| Random sparse (avg degree 3) | 54,266 | 35ms | 74ms | 109ms |
| 1M isolated (0 edges) | 1,000,000 | 0ms | 212ms | 212ms |

The cgraph call splits into union-find algorithm and Python set construction:

| Scenario | Components | Union-find | Set construction | Set % of cgraph |
|----------|-----------:|-----------:|-----------------:|----------------:|
| 1 component (all connected) | 1 | 4ms | 24ms | 86% |
| 10 components (100K each) | 10 | 4ms | 26ms | 87% |
| 1K components (1K each) | 1,000 | 4ms | 20ms | 84% |
| 1 dominant + 100K isolated | 100,001 | 4ms | 41ms | 92% |
| Random sparse (avg degree 3) | 54,266 | 11ms | 50ms | 82% |
| 1M isolated (0 edges) | 1,000,000 | 2ms | 160ms | 99% |

The union-find algorithm is 2-11ms — already near-optimal. **82-99% of the cgraph time is Python set construction** (`PySet_New` + `PySet_Add`), which is the CPython floor for building hash-based sets. Many small components are expensive because each `PySet_New()` allocates a Python set object. Gather cost scales with edge count, not component count.

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
python benchmarks/bench_all.py       # structured cost breakdown + topology scenarios
python benchmarks/bench_dag_learn.py  # hill-climb K2: cgraph vs pgmpy
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit_tests/ -v
```
