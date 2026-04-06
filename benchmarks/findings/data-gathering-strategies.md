# Data Gathering Strategies — Findings

## Context
Users have domain objects (e.g. `Branch` with `.node_a`, `.node_b`) and need
to extract edge data for cgraph. The gathering strategy significantly impacts
end-to-end performance.

## Gathering approaches benchmarked (1M nodes, 1.5M edges)

| Approach | Time | Notes |
|---|---:|---|
| `[(b.node_a, b.node_b) for b in branches]` | 68ms | Creates 1.5M tuple objects |
| `[b.node_a for b in ...]` + `[b.node_b for b in ...]` | **24ms** | Two flat lists, no tuples |
| `[x for b in branches for x in (b.node_a, b.node_b)]` | 66ms | Flat interleaved, still creates tuples internally |
| `np.array([(a,b)...], dtype=np.int32)` | 231ms | Worst: tuple creation + numpy conversion |
| `np.empty((m,2)); edges[i,0]=...` (pre-alloc fill) | 148ms | numpy element assignment is slow |
| `np.column_stack([np.array(src), np.array(dst)])` | 89ms | Two list→array conversions + stack |
| `np.array(src), np.array(dst)` (two 1D arrays) | 81ms | Cheapest numpy path |
| Bare iteration `[b for b in branches]` | 14ms | Floor: loop + object ref, no attr |

## Attribute access cost

| Method | Time for two flat lists |
|---|---:|
| `b.node_a` (direct) | 40ms |
| `getattr(b, 'node_a')` | 75ms (+87%) |
| `attrgetter('node_a')` | 77ms (+91%) |

Direct attribute access uses Python's `LOAD_ATTR` bytecode with inline cache
and `__slots__` fast-path. `getattr()` goes through full attribute resolution
every time. This means a C function using `PyObject_GetAttrString` would be
~2x slower than users doing `b.node_a` themselves.

## Conclusion
- **Best without numpy:** two flat list comprehensions → split lists interface (24ms gather)
- **Best with numpy:** two `np.array()` calls → split numpy 1D interface (81ms gather)
- **Don't:** create numpy arrays from tuples, use `getattr`, or build a "pass objects" API
- The 14ms bare iteration floor means 10ms is attribute access overhead per list — unavoidable
