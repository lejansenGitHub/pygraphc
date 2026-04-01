# ConnectedComponent

Fast connected-components library using a C union-find with path compression and union by rank, callable from Python. Zero runtime dependencies.

## Installation

```bash
pip install -e .
```

Requires a C compiler (the extension is compiled at install time with `-O3`).

## Usage

```python
from connected_component import igp_connected_components, igp_connected_components_with_branch_ids

# Basic: get each component as a set of node IDs
for component in igp_connected_components(num_nodes, edges):
    print(component)

# With branch IDs: get (node_set, branch_id_set) per component
for nodes, branches in igp_connected_components_with_branch_ids(num_nodes, edges, branch_ids):
    print(nodes, branches)
```

## Benchmarks

Sparse random graphs with ~3 edges per node (typical grid topology). Compared against `scipy.sparse.csgraph.connected_components` with CSR matrix construction and `defaultdict` grouping.

| Nodes | C union-find | scipy | Speedup |
|------:|-------------:|------:|--------:|
| 1K | 0.00005s | 0.0004s | ~10x |
| 10K | 0.0004s | 0.003s | ~8x |
| 100K | 0.006s | 0.031s | ~5x |
| 1M | 0.063s | 0.449s | ~7x |

Run the benchmarks yourself:

```bash
pip install -e ".[benchmark]"
pytest src/connected_component/tests/test_performance.py -v -s -k speedup
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
