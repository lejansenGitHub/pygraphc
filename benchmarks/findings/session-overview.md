# Runtime Optimization Session — Overview

## Goal
Improve cgraph runtime without increasing memory consumption.
Find ways to let users pass data more efficiently.

## Key discovery: 3-phase cost model

Every cgraph call has three cost phases (Connected Components at 1M nodes):

| Phase | Tuples | Split lists | numpy (m,2) |
|-------|-------:|------------:|------------:|
| 1. Gather (Python loop over objects) | 68ms | **24ms** | 72ms |
| 2. Parse (hash map + edge translation) | 29ms | 25ms | **10ms** |
| 3. C algorithm (union-find + results) | 61ms | 61ms | 61ms |

- C algorithm is 39-55% of total depending on interface — already tight, nothing to optimize.
- Parse is 16-29ms — dominated by `PyLong_AsLong` per element for lists, near-zero for numpy buffers.
- Gather is 15-45% — pure Python, cgraph can't optimize it, but interface choice affects it heavily.

## What we tried and learned

### C-side optimizations (5 attempted, 3 accepted)

| Optimization | Status | Impact on tuples | Why |
|---|---|---|---|
| Splitmix hash for IntMap | REJECTED | 20-78% slower | Identity hash is optimal at 50% load factor |
| Numpy buffer in parse_edges_mapped | ACCEPTED | No change for tuples | Fixes pathological numpy slowness (was 5-10x worse than tuples) |
| Numpy buffer for node_ids | REJECTED | 5-41% slower | PyLong_FromLong for nid_items output cancels the gain |
| Bulk malloc consolidation | ACCEPTED | ~5-8% on Tarjan | Fewer syscalls, contiguous memory |
| 4-ary heap for Dijkstra | ACCEPTED | 11-13% on Dijkstra | Shallower tree, fewer cache misses |

**Conclusion:** C algorithm is already near-optimal. Tuple parsing overhead (`PyLong_AsLong` per element) is a CPython API bottleneck that can't be optimized further.

### Data gathering (Python side)

| Gathering approach | Time at 1M |
|---|---|
| `[(b.node_a, b.node_b) for b in branches]` (tuples) | 68ms |
| `[b.node_a for b in ...]`, `[b.node_b for b in ...]` (two flat lists) | **24ms** |
| `getattr(b, 'node_a')` per object (dynamic) | 75ms |
| `attrgetter('node_a')` cached | 77ms |
| Bare iteration (no attr access) | 14ms |

**Conclusion:** Direct attribute access (`b.node_a`) is fastest. `getattr`/`attrgetter` is ~2x slower due to missing bytecode inline cache. A "pass objects + property names" API would be slower than letting users do it themselves.

### Interface change: split lists

Added `connected_components(node_ids, src, dst)` — two flat lists instead of list of tuples.

| Interface | Gather | Parse | C algo | Total | vs tuples |
|---|---:|---:|---:|---:|---|
| Tuples | 68ms | 29ms | 61ms | **158ms** | baseline |
| **Split lists** | **24ms** | 25ms | 61ms | **110ms** | **1.43x** |

The win is entirely from gather — avoiding 1.5M tuple object allocations. C parse cost is identical (PyLong_AsLong dominates either way).

## Rejected ideas with evidence

1. **Convert tuples to numpy in Python layer** — `np.array(edges)` costs ~155ms at 1.5M edges, wiping out the C-side gain. Net loss.

2. **Accept objects + attr names in C** — `PyObject_GetAttrString` is ~2x slower than Python's `LOAD_ATTR` bytecode (no inline cache). Users' direct `b.node_a` access is faster than anything cgraph could do.

3. **Numpy node_ids** — Building Python int objects for output (nid_items) cancels out the buffer reading gain on input.

## Future directions (plan.md Phase 4)

The biggest remaining win is the **Graph / MappedGraph class** — pay parsing cost once, reuse across multiple algorithm calls. For users who call CC + bridges + AP on the same graph, this avoids 3x the parse overhead.

Currently split lists is only implemented for `connected_components`. Extending to other algorithms is straightforward (same `nid_parse_split` + `parse_edges_mapped_split` infrastructure).
