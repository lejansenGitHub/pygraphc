# ConnectedComponent

Fast connected-components library using a C union-find with path compression and union by rank, callable from Python. Zero runtime dependencies.

## Installation

```bash
pip install -e .
```

Requires a C compiler (the extension is compiled at install time with `-O3`).

## Usage

```python
from connected_component import (
    igp_connected_components,
    igp_connected_components_remapped,
    igp_connected_components_with_branch_ids,
    igp_connected_components_with_branch_ids_remapped,
    igp_connected_component_labels,
)

# Basic: get each component as a set of node indices
for component in igp_connected_components(num_nodes, edges):
    print(component)  # {0, 1, 2}, {3, 4}, ...

# With node ID remapping (no Python loop needed):
# node_ids maps index -> original NodeId
for component in igp_connected_components_remapped(node_ids, edges):
    print(component)  # {NodeId(100), NodeId(200), ...}

# With branch IDs:
for nodes, branches in igp_connected_components_with_branch_ids(num_nodes, edges, branch_ids):
    print(nodes, branches)

# With branch IDs + node ID remapping:
for nodes, branches in igp_connected_components_with_branch_ids_remapped(node_ids, edges, branch_ids):
    print(nodes, branches)

# Raw labels (fastest — no set construction):
labels = igp_connected_component_labels(num_nodes, edges)
# labels[i] = component ID for node i
```

All functions accept edges as either a `list[tuple[int, int]]` or a numpy `int32` array of shape `(m, 2)`.

## Benchmarks

Sparse random graphs with ~3 edges per node (typical grid topology). Compared against `scipy.sparse.csgraph.connected_components` with CSR matrix construction and `defaultdict` grouping.

### vs scipy (end-to-end)

| Nodes | C tuples->sets | C numpy->labels | scipy | Speedup (sets) | Speedup (labels) |
|------:|---------------:|----------------:|------:|---------------:|-----------------:|
| 1K | 0.00005s | 0.00001s | 0.0004s | ~8x | ~33x |
| 10K | 0.0006s | 0.0001s | 0.003s | ~5x | ~28x |
| 100K | 0.006s | 0.002s | 0.032s | ~5x | ~20x |
| 1M | 0.070s | 0.020s | 0.43s | ~6x | ~21x |
| 10M | 1.35s | 0.53s | 5.98s | ~4x | ~11x |

### Remapped variants (real-world pattern)

When node IDs are non-contiguous (e.g. database IDs), the old pattern builds index-based sets in C then remaps in Python. The `_remapped` variants do everything in C:

| Nodes | Old (C sets + Python remap) | New (C remapped) | Speedup |
|------:|----------------------------:|-----------------:|--------:|
| 1M | 0.29s | 0.10s | ~2.8x |
| 10M | 4.56s | 1.62s | ~2.8x |

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pytest src/connected_component/tests/test_performance.py -v -s -k speedup
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
