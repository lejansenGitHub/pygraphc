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
| 1K | 0.0001s | 0.0009s | **8x** |
| 10K | 0.001s | 0.011s | **9x** |
| 100K | 0.021s | 0.324s | **16x** |
| 1M | 0.185s | 7.02s | **38x** |

### Bridges

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.005s | **47x** |
| 10K | 0.001s | 0.053s | **38x** |
| 100K | 0.021s | 1.45s | **70x** |
| 1M | 0.410s | 22.90s | **56x** |

### Articulation points

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.002s | **22x** |
| 10K | 0.002s | 0.018s | **9x** |
| 100K | 0.025s | 0.480s | **19x** |
| 1M | 0.421s | 10.26s | **24x** |

### BFS

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.002s | **19x** |
| 10K | 0.001s | 0.019s | **15x** |
| 100K | 0.015s | 0.653s | **45x** |
| 1M | 0.225s | 14.61s | **65x** |

### Dijkstra (SSSP)

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.001s | **10x** |
| 10K | 0.002s | 0.020s | **9x** |
| 100K | 0.039s | 0.678s | **17x** |
| 1M | 0.698s | 12.90s | **18x** |

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
