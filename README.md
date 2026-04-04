# cgraph

Fast connected-components library using a C union-find with path compression and union by rank, callable from Python. Zero runtime dependencies.

## Installation

```bash
pip install -e .
```

Or from GitHub:

```bash
pip install git+https://github.com/lejansenGitHub/cgraph.git
```

Requires a C compiler (the extension is compiled at install time with `-O3`).

## Usage

```python
from cgraph import connected_components, connected_components_with_branch_ids

# Get each component as a set of original node IDs
node_ids = [100, 200, 300, 400]
edges = [(0, 1), (2, 3)]  # indices into node_ids

for component in connected_components(node_ids, edges):
    print(component)  # {100, 200}, {300, 400}

# With branch IDs:
branch_ids = [901, 902]
for nodes, branches in connected_components_with_branch_ids(node_ids, edges, branch_ids):
    print(nodes, branches)  # {100, 200} {901}, {300, 400} {902}
```

All functions accept edges as either a `list[tuple[int, int]]` or a numpy `int32` array of shape `(m, 2)`.

## Benchmarks

Sparse random graphs with ~3 edges per node (typical grid topology). Compared against `scipy.sparse.csgraph.connected_components` with CSR matrix construction and `defaultdict` grouping.

### vs scipy (end-to-end)

| Nodes | C tuples->sets | C numpy->sets | scipy | Speedup (tuples) | Speedup (numpy) |
|------:|---------------:|--------------:|------:|-----------------:|----------------:|
| 1K | 0.00005s | 0.00003s | 0.0004s | ~8x | ~13x |
| 10K | 0.0006s | 0.0003s | 0.003s | ~5x | ~10x |
| 100K | 0.006s | 0.004s | 0.032s | ~5x | ~8x |
| 1M | 0.070s | 0.049s | 0.43s | ~6x | ~9x |
| 10M | 1.35s | 1.17s | 5.98s | ~4x | ~5x |

### Remapped variants (real-world pattern)

When node IDs are non-contiguous (e.g. database IDs), the old pattern builds index-based sets in C then remaps in Python. The `_remapped` variants do everything in C:

| Nodes | Old (C sets + Python remap) | New (C remapped) | Speedup |
|------:|----------------------------:|-----------------:|--------:|
| 1M | 0.29s | 0.10s | ~2.8x |
| 10M | 4.56s | 1.62s | ~2.8x |

### Run benchmarks

```bash
pip install -e ".[benchmark]"
pytest tests/performance_tests/ -v -s -k speedup
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit_tests/ -v
```
