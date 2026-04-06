# Set Construction — CPython Floor

## Problem
Building `list[set[NodeId]]` from component labels is ~50ms at 1M nodes,
dominating the C algorithm phase.

## What we tried

### 1. Direct PySetObject table manipulation
Bypassed `PySet_Add` to write directly into the internal hash table with
pre-allocated size (no rehashing). Result: **worse** across all scenarios
(+6% to +15%). The bucketing overhead to group items by component before
insertion outweighed the rehash savings.

### 2. Hybrid PySet_New(list) for large components
Build temporary PyList per large component, call `PySet_New(list)` (pre-sizes
hash table). Use direct `PySet_Add` for small components. Result: **no improvement**.
The `Py_INCREF` + list allocation/deallocation cost equals the rehash savings.

### 3. PySet_New(list) for all components
Result: **much worse** for many-component graphs. Creating 54K temporary
PyList objects (mostly single-element) adds massive overhead.

## Why nothing works
`PySet_Add` internally does: `PyObject_Hash(key)` + probe table + store.
Our manual code does the same operations. The function call overhead of
`PySet_Add` is ~1-2ns — negligible at 50ns per insert. The rehash savings
(~20 resizes for 1M elements, total ~2ms) are eaten by any bookkeeping
we add.

## Breakdown of the 50ms (1M nodes, few large components)
- `PyObject_Hash(key)`: ~15ms (hash is identity for ints, but still a function call + type check)
- Hash table probing + store: ~15ms (cache-unfriendly random access)
- `Py_INCREF(key)`: ~5ms
- `PySet_Add` function call overhead: ~5ms
- Set resize (amortized): ~2ms
- `PySet_New` object creation: <1ms (few components)

## Topology impact at 1M nodes
| Scenario | Components | Time |
|---|---:|---:|
| 1 big component | 1 | 28ms |
| 1K equal (1K each) | 1,000 | 22ms |
| 10 equal (100K each) | 10 | 28ms |
| 1 dominant + 100K isolated | 100,001 | 48ms |
| Random sparse | 54,266 | 57ms |
| All isolated | 1,000,000 | 168ms |

Many small components are expensive because `PySet_New()` allocates a full
Python object per component (~170ns each).

## Conclusion
The 50ms is the CPython floor for building sets of 1M Python int objects.
The only way past it is not building Python sets (return labels array or
a custom C type), which changes the API contract.
