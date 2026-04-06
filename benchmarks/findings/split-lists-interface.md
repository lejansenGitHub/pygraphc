# Split Lists Interface — Findings

## Change
Added `connected_components(node_ids, src, dst)` calling convention where
`src` and `dst` are flat lists (or 1D numpy arrays) of node IDs.

## Implementation
- C: `parse_edges_mapped_split()` reads two separate sequences/buffers
- C: `nid_parse_split()` builds intmap then calls split parser
- C: `py_cc_nid_split()` entry point registered as `cc_nid_split`
- Python: `connected_components()` dispatches on arg count (2 = edges, 3 = src/dst)

## Results (Connected Components, 1M nodes, 1.5M edges)

### End-to-end from Branch objects
| Interface | Gather | Parse | C algo | Total | vs tuples |
|---|---:|---:|---:|---:|---|
| Tuples `[(a,b)...]` | 68ms | 29ms | 61ms | 158ms | baseline |
| Split lists `[a...],[b...]` | 24ms | 25ms | 61ms | 110ms | **1.43x** |
| numpy (m,2) | 72ms | 10ms | 61ms | 143ms | 1.11x |
| Split numpy 1D | 65ms | 12ms | 61ms | 138ms | 1.14x |

### C-only (pre-built data, no gather)
| Interface | Parse+algo time |
|---|---:|
| Tuples | 90ms |
| Split lists | 86ms |
| numpy (m,2) | 71ms |
| Split numpy 1D | 73ms |

Split lists parse time is the same as tuples — `PyLong_AsLong` per element
dominates. The 1.43x total win is entirely from faster gather (no tuple objects).

## Why split lists beats numpy end-to-end
Numpy has the fastest C parse (10ms vs 25ms) but the slowest gather (72ms vs 24ms).
Creating numpy arrays from Python data is expensive — `np.array([...])` iterates
every element. The gather cost outweighs the parse saving.

Split lists has cheap gather (flat list comprehensions) with acceptable parse cost.

## Scaling behavior
| Scale | Split lists speedup |
|---|---|
| 10K | 1.59x |
| 100K | 1.45x |
| 1M | 1.43x |

The relative advantage decreases at larger scales because C algo (fixed overhead
regardless of interface) becomes a larger fraction of total.
