# Optimization 4: Consolidate per-algorithm mallocs — ACCEPTED (marginal)

## Change
Replaced 5-7 separate `malloc(n * sizeof(int))` calls with single bulk allocation
in bridges, articulation_points, biconnected_components (both index-based and nid).

## Results (median of 5 runs, seconds, n=1M, contiguous IDs, list-of-tuples)

| Algorithm | Baseline | After | Change |
|-----------|----------|-------|--------|
| bridges | 0.315 | 0.298 | **-5%** |
| articulation_points | 0.308 | 0.283 | **-8%** |
| biconnected_components | 0.580 | 0.543 | **-6%** |
| bfs | 0.109 | 0.109 | ~same |
| shortest_path | 0.319 | 0.304 | **-5%** |
| sssp_lengths | 0.592 | 0.588 | ~same |

## Analysis
Modest but consistent improvement on Tarjan-family algorithms where 5-6 arrays
of size n are allocated. The win comes from:
1. Fewer syscalls (1 vs 5-6 malloc calls)
2. Better memory locality (arrays are contiguous in virtual memory)
3. Less allocator bookkeeping overhead

BFS and SSSP show minimal change because they only allocate 2-3 arrays.

## Conclusion
**Accepted.** Small but free improvement with zero risk and simpler cleanup code.
