# Optimization 3: Numpy Fast-Path for node_ids — REJECTED

## Hypothesis
Reading node_ids from numpy buffer would avoid PyLong_AsLong per node when
building the intmap, speeding up nid_parse.

## Change
Added buffer protocol path in `nid_parse()`: when node_ids is a numpy int32/int64
array, reads values directly from buffer for intmap_put, and builds a Python list
of proper ints (via PyLong_FromLong) for nid_items output remapping.

## Results (median of 5 runs, seconds, n=1M, contiguous IDs)

| Algorithm | list nids + np edges | np nids + np edges | Change |
|-----------|---------------------|--------------------|--------|
| connected_components | 0.077 | 0.085 | +10% WORSE |
| bridges | 0.299 | 0.314 | +5% WORSE |
| bfs | 0.093 | 0.131 | +41% WORSE |
| sssp_lengths | 0.586 | 0.655 | +12% WORSE |

## Analysis
The buffer path avoids `PyLong_AsLong(nid_items[i])` per node for intmap building,
but must call `PyLong_FromLong(data[i])` per node to create the nid_items list
needed by every algorithm's output path. This is a zero-sum trade: we avoid
unpacking Python objects on input, but create the same number on a different path.

When node_ids is a Python list, those int objects already exist — no creation needed.
The numpy path adds memory allocation overhead (new PyList + n PyLong objects).

## Conclusion
**Reverted.** Cannot optimize node_ids parsing without also eliminating the need
for Python int objects in output remapping. The pre-indexed API (plan.md) is the
right solution: users pass int indices directly, skipping both intmap and remapping.
