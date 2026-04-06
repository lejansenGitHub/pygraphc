# Optimization 2: Numpy Zero-Copy in parse_edges_mapped — ACCEPTED

## Hypothesis
Adding buffer protocol support to `parse_edges_mapped()` would eliminate millions
of Python/C boundary crossings per call, making numpy input faster than tuples.

## Change
Added numpy buffer protocol fast-path at the top of `parse_edges_mapped()`. When
edges is an `(m,2)` int32 or int64 numpy array, reads values directly from the
buffer and does `intmap_get` to translate — no Python object creation per edge.

Also added `is_int32_fmt` / `is_int64_fmt` buffer format helpers (were missing
from the working copy).

## Results (median of 5 runs, seconds, n=1M)

### Contiguous IDs
| Algorithm | Baseline tuples | Baseline numpy | After numpy | numpy speedup |
|-----------|----------------|----------------|-------------|---------------|
| connected_components | 0.115 | 0.737 | 0.071 | **10.4x faster** |
| bridges | 0.316 | 1.085 | 0.286 | **3.8x faster** |
| articulation_points | 0.364 | 0.970 | 0.277 | **3.5x faster** |
| biconnected_components | 0.620 | 1.185 | 0.542 | **2.2x faster** |
| bfs | 0.116 | 0.787 | 0.092 | **8.6x faster** |
| shortest_path | 0.313 | 0.934 | 0.287 | **3.3x faster** |
| sssp_lengths | 0.562 | 1.203 | 0.544 | **2.2x faster** |

### Non-contiguous IDs (stride-3 from 1M)
| Algorithm | Baseline tuples | Baseline numpy | After numpy | numpy speedup |
|-----------|----------------|----------------|-------------|---------------|
| connected_components | 0.207 | 0.805 | 0.090 | **8.9x faster** |
| bridges | 0.421 | 1.031 | 0.310 | **3.3x faster** |
| articulation_points | 0.411 | 1.033 | 0.276 | **3.7x faster** |
| biconnected_components | 0.668 | 1.292 | 0.517 | **2.5x faster** |
| bfs | 0.228 | 0.815 | 0.097 | **8.4x faster** |
| shortest_path | 0.428 | 1.023 | 0.279 | **3.7x faster** |
| sssp_lengths | 0.696 | 1.377 | 0.563 | **2.4x faster** |

## Key takeaway
- **Numpy input is now faster than tuples** across all algorithms (was 5-10x slower before)
- Biggest wins on algorithms where edge parsing dominates (CC, BFS: ~10x improvement)
- Smaller wins on heavier algorithms (BCC, SSSP: ~2x improvement) where algo time dominates
- No memory impact: still allocates same src/dst arrays, just fills them from buffer instead of Python objects

## Root cause of prior slowness
Numpy arrays iterated via `PySequence_Fast` produce numpy scalar objects (not Python ints),
which are much heavier to unpack per element than Python int objects from regular tuples.
The buffer protocol bypasses this entirely.
