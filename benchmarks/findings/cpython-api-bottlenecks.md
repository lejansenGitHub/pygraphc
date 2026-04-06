# CPython API Bottlenecks — Findings

## Context
When cgraph receives Python lists, the C code must extract integer values using
the CPython API. These API calls are the dominant cost in the parse phase.

## Key bottlenecks measured

### PyLong_AsLong — ~20ns per call
Every integer extracted from a Python list goes through `PyLong_AsLong`.
At 1.5M edges × 2 endpoints = 3M calls, this adds ~60ms. Cannot be avoided
for Python list input — it's the CPython API's only way to read int values.

Split lists vs tuples: same `PyLong_AsLong` count, so same parse cost (~25-29ms).

### PySequence_Fast + tuple unpacking — ~7ms overhead at 1.5M edges
Tuples require `PySequence_Fast` per edge to unpack the (src, dst) pair,
plus `PySequence_Fast_ITEMS` to get the element pointers. This adds ~4ms
on top of the `PyLong_AsLong` cost.

Split lists avoid this entirely — direct `PySequence_Fast_ITEMS` on each
flat list, then linear iteration.

### PyObject_GetBuffer — near-zero for numpy
When input is a numpy array, `PyObject_GetBuffer` gives a raw C pointer.
Reading `int *data[i]` is orders of magnitude faster than `PyLong_AsLong`.
This is why numpy parse is 10ms vs 25-29ms for lists.

### PyObject_GetAttrString — ~50ns per call
Proposed "pass objects + property names" API would use this. At 1.5M objects × 2
attrs = 3M calls at 50ns = 150ms. Versus Python's `LOAD_ATTR` which uses
inline caching and resolves in ~13ns for `__slots__` objects.

**Conclusion:** Cannot beat Python's own attribute access from C.

## Implications for future work
- List input parse cost is fundamentally bounded by `PyLong_AsLong` throughput
- Only way to eliminate parse cost: accept pre-indexed data (skip hash map + `PyLong_AsLong`)
- The Graph/MappedGraph class in plan.md is the right approach — parse once, reuse
- Numpy is the fastest input format but numpy array creation from Python data is expensive
