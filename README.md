# cgraph

Fast, general-purpose graph algorithm library implemented in C, callable from Python. Zero runtime dependencies.

Edges use **original node IDs** — no manual index mapping needed. A C-side hash map translates IDs to internal indices automatically.

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

## Performance

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

### Benchmarks vs networkx

End-to-end from `Branch` objects using split lists (gather + algorithm). Sparse random graphs, ~3 edges per node.

- **networkx**: `add_nodes_from` + `add_edges_from` + algorithm
- **cgraph**: gather split lists from objects + algorithm

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

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pip install networkx
pytest tests/performance_tests/ -v -s -k speedup
python benchmarks/bench_all.py  # structured cost breakdown + topology scenarios
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit_tests/ -v
```
