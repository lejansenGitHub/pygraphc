# cgraph

Fast, general-purpose graph algorithm library implemented in C, callable from Python. Zero runtime dependencies.

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
    connected_components_with_branch_ids,
    bridges,
    articulation_points,
    biconnected_components,
    bfs,
    two_edge_connected_components,
    nodes_on_simple_paths,
)

node_ids = [100, 200, 300, 400]
edges = [(0, 1), (2, 3)]  # indices into node_ids

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

All functions accept edges as either `list[tuple[int, int]]` or a numpy `int32` array of shape `(m, 2)`. Weights accept `list[float]` or numpy `float64` arrays.

## Benchmarks vs networkx

Sparse random graphs with ~3 edges per node. All comparisons against networkx on the same machine.

### Connected components

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.00005s | 0.0002s | **6x** |
| 10K | 0.0004s | 0.0023s | **6x** |
| 100K | 0.007s | 0.110s | **16x** |
| 1M | 0.076s | 1.61s | **21x** |
| 10M | 1.46s | 26.99s | **19x** |

### Bridges

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.004s | **36x** |
| 10K | 0.0008s | 0.046s | **55x** |
| 100K | 0.012s | 0.889s | **73x** |
| 1M | 0.359s | 15.28s | **43x** |

### Articulation points

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.00005s | 0.0008s | **16x** |
| 10K | 0.0007s | 0.008s | **12x** |
| 100K | 0.009s | 0.223s | **25x** |
| 1M | 0.264s | 5.06s | **19x** |

### Biconnected components

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0002s | 0.001s | **7x** |
| 10K | 0.002s | 0.013s | **6x** |
| 100K | 0.028s | 0.327s | **12x** |
| 1M | 1.80s | 7.29s | **4x** |

### BFS

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 1K | 0.0001s | 0.001s | **10x** |
| 10K | 0.001s | 0.014s | **14x** |
| 100K | 0.011s | 0.408s | **37x** |
| 1M | 0.215s | 11.72s | **55x** |

### Dijkstra (SSSP)

| Nodes | cgraph | networkx | Speedup |
|------:|-------:|---------:|--------:|
| 10K | 0.002s | 0.010s | **5x** |
| 100K | 0.029s | 0.313s | **11x** |
| 1M | 0.659s | 6.11s | **9x** |

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
