# Optimization 5: 4-ary Heap for Dijkstra — ACCEPTED

## Change
Replaced binary heap (2 children per node) with 4-ary heap in `mh_sift_up` and
`mh_sift_down`. Tree depth is halved (log4 n vs log2 n), reducing cache misses
in sift_down at the cost of more comparisons per level (but in adjacent memory).

## Results (median of 5 runs, seconds, n=1M, contiguous IDs)

### Tuples (list-of-tuples input — main use case)
| Algorithm | Baseline | After | Change |
|-----------|----------|-------|--------|
| shortest_path | 0.304 | 0.277 | **-9%** |
| sssp_lengths | 0.588 | 0.550 | **-6%** |
| CC/bridges/AP/BCC | unchanged | unchanged | ~same (no heap) |

### Numpy input
| Algorithm | Baseline | After | Change |
|-----------|----------|-------|--------|
| shortest_path | 0.290 | 0.239 | **-18%** |
| sssp_lengths | 0.573 | 0.491 | **-14%** |

## Analysis
The 4-ary heap reduces the number of levels traversed during `mh_sift_down` (the
hot path in `mh_pop`). At n=1M, binary heap depth is ~20, 4-ary is ~10. Each level
in the 4-ary heap compares 4 children (contiguous in the array) vs 2 in binary,
but the contiguous access pattern is cache-friendly.

The improvement is larger for numpy input because the overall runtime is lower
(faster edge parsing), so the heap improvement is a larger fraction.

## Conclusion
**Accepted.** Consistent 6-18% improvement for all Dijkstra-based algorithms
with zero memory impact.
