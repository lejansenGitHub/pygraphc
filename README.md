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

## Benchmarks vs networkx

Realistic end-to-end comparison starting from domain objects. Both sides extract edges from the same `Branch` instances, then build their data structures and run the algorithm.

```python
class Branch:
    def __init__(self, branch_id: int, node_a: int, node_b: int):
        self.branch_id = branch_id
        self.node_a = node_a
        self.node_b = node_b
```

- **networkx**: `add_nodes_from` + `add_edges_from` (bulk insert) + run algorithm
- **cgraph**: gather `(node_a, node_b)` tuples from objects + run algorithm (C handles ID mapping internally)

Sparse random graphs with ~3 edges per node.

### Connected components

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.0009s | **17x** |
| 10K | 0.0005s | 0.013s | **23x** |
| 100K | 0.008s | 0.414s | **51x** |
| 1M | 0.123s | 8.06s | **66x** |

### Bridges

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.005s | **99x** |
| 10K | 0.0007s | 0.058s | **84x** |
| 100K | 0.012s | 1.38s | **113x** |
| 1M | 0.336s | 22.37s | **67x** |

### Articulation points

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.00005s | 0.002s | **35x** |
| 10K | 0.0007s | 0.018s | **25x** |
| 100K | 0.010s | 0.505s | **50x** |
| 1M | 0.328s | 10.09s | **31x** |

### BFS

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.00005s | 0.002s | **43x** |
| 10K | 0.0005s | 0.022s | **41x** |
| 100K | 0.007s | 0.686s | **100x** |
| 1M | 0.143s | 14.14s | **99x** |

### Dijkstra (SSSP)

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.00005s | 0.001s | **22x** |
| 10K | 0.001s | 0.020s | **14x** |
| 100K | 0.027s | 0.676s | **26x** |
| 1M | 0.599s | 11.65s | **20x** |

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pip install networkx
pytest tests/performance_tests/ -v -s -k speedup
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit_tests/ -v
```
