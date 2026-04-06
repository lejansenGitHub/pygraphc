# Optimization 1: IntMap Hash Function — REJECTED

## Hypothesis
Replacing the identity hash `key & mask` with splitmix64 finalizer would reduce
clustering in the hash map and improve lookup performance.

## Change
Replaced `(unsigned long)key & mask` with a 6-operation splitmix64 finalizer
(3 XOR-shifts + 2 multiplies) in `intmap_put` and `intmap_get`.

## Results (median of 5 runs, seconds)

### Contiguous IDs (0..n-1) at n=1M
| Algorithm | Baseline | After | Change |
|-----------|----------|-------|--------|
| connected_components | 0.089 | 0.153 | +72% WORSE |
| bridges | 0.291 | 0.386 | +33% WORSE |
| bfs | 0.106 | 0.189 | +78% WORSE |

### Non-contiguous IDs (stride-3 from 1M) at n=1M
| Algorithm | Baseline | After | Change |
|-----------|----------|-------|--------|
| connected_components | 0.196 | 0.246 | +26% WORSE |
| bridges | 0.402 | 0.484 | +20% WORSE |
| bfs | 0.214 | 0.240 | +12% WORSE |

## Analysis
The identity hash (`key & mask`) works well because:
1. **Contiguous IDs**: Maps perfectly to consecutive slots — zero collisions.
2. **Non-contiguous IDs**: Low bits of typical ID patterns (stride-N, database PKs)
   still vary enough that mask-based hashing distributes well.
3. **Splitmix cost**: 6 arithmetic operations per hash call overwhelms any probing
   reduction. The hash map is at 50% load factor (cap = 2n), so average probe
   length is already ~1.5 with identity hash.

## Conclusion
**Reverted.** The identity hash is optimal for this use case. A better hash would
only help for pathological key distributions (e.g., all keys are multiples of a
power-of-2 matching the table size), which don't occur in practice.

Future direction: if non-contiguous ID performance matters, the bigger win is
eliminating the hash map entirely via the pre-indexed API (see plan.md).
