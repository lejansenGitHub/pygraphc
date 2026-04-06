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
| 1. Gather (Python loop over objects) | 68ms | **24ms** | 72ms |
| 2. Parse (hash map + edge translation) | 29ms | 25ms | **10ms** |
| 3. C algorithm (union-find + results) | 61ms | 61ms | 61ms |
| **Total** | **158ms** | **110ms** | **143ms** |
| vs tuples | baseline | **1.43x** | 1.11x |

- **Gather** is pure Python — extracting `node_a`/`node_b` from your objects. cgraph can't optimize this, but the interface choice affects it (tuples cost 68ms, flat lists cost 24ms).
- **Parse** is the C-side input handling — building the node-ID hash map and translating edges to internal indices. Numpy skips per-element Python unpacking via buffer reads.
- **C algorithm** is the actual computation. Fixed cost, identical across all interfaces.

### Data gathering strategies

Two calling conventions — both accept Python lists or numpy arrays:

**Strategy A** (tuples) — simplest:
```python
edges = [(b.node_a, b.node_b) for b in branches]
connected_components(node_ids, edges)
```

**Strategy B** (two flat lists) — **1.43x faster** end-to-end:
```python
src = [b.node_a for b in branches]
dst = [b.node_b for b in branches]
connected_components(node_ids, src, dst)
```

Strategy B is faster because building two flat lists avoids creating 1.5M tuple objects. The C parsing cost is similar — `PyLong_AsLong` per element dominates regardless of container shape.

### Benchmarks vs networkx

End-to-end from `Branch` objects (gather + algorithm). Sparse random graphs, ~3 edges per node.

- **networkx**: `add_nodes_from` + `add_edges_from` + algorithm
- **cgraph**: gather edges from objects + algorithm

| Algorithm | Nodes | cgraph | networkx | Speedup |
|-----------|------:|-------:|---------:|--------:|
| Connected Components | 1K | 0.0001s | 0.001s | **13x** |
| Connected Components | 10K | 0.001s | 0.013s | **14x** |
| Connected Components | 100K | 0.015s | 0.349s | **23x** |
| Connected Components | 1M | 0.157s | 6.81s | **43x** |
| Bridges | 1M | 0.401s | 21.42s | **53x** |
| Articulation Points | 1M | 0.350s | 8.97s | **26x** |
| BFS | 1M | 0.165s | 13.62s | **82x** |
| Dijkstra | 1M | 0.390s | 11.11s | **29x** |

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pip install networkx
pytest tests/performance_tests/ -v -s -k speedup
python benchmarks/bench_all.py  # structured cost breakdown
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit_tests/ -v
```
