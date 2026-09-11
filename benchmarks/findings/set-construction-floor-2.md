# Set Construction — CPython Floor, Second Attempt

Follow-up to `set-construction-floor.md`, testing one specific hypothesis that
the first round did not settle from the source.

## Hypothesis

Growing a set with repeated `PySet_Add` forces several table resizes, each
reinserting every element. CPython's set update path contains a presize step
when it can get a length hint from the iterable it is given. So filling a
`PyListObject` in bulk with `PyList_SET_ITEM` and then calling `PySet_New(list)`
once should skip those resizes and win.

**The hypothesis is false for this interpreter, and the source settles it before
any measurement.**

## Source evidence

Interpreter: `3.11.11 (main, Jul 16 2025, 22:29:00) [Clang 17.0.0]`.
Source read from the matching tarball in the pyenv cache,
`Python-3.11.11/Objects/setobject.c`.

`PySet_New` forwards straight to `make_new_set` (line 2286):

```c
2285: PyObject *
2286: PySet_New(PyObject *iterable)
2287: {
2288:     return make_new_set(&PySet_Type, iterable);
2289: }
```

`make_new_set` always starts the table at the minimum size, regardless of the
iterable it was handed, then defers to `set_update_internal` (lines 963, 969-970):

```c
963:     so->mask = PySet_MINSIZE - 1;
...
969:     if (iterable != NULL) {
970:         if (set_update_internal(so, iterable)) {
```

`set_update_internal` presizes in exactly two cases, and a list is neither of
them (lines 885-922):

```c
885:     if (PyAnySet_Check(other))
886:         return set_merge(so, other);
887:
888:     if (PyDict_CheckExact(other)) {
...
892:         Py_ssize_t dictsize = PyDict_GET_SIZE(other);
893:
894:         /* Do one big resize at the start, rather than
895:         * incrementally resizing as we insert new keys.  Expect
896:         * that there will be no (or few) overlapping keys.
897:         */
898:         if (dictsize < 0)
899:             return -1;
900:         if ((so->fill + dictsize)*5 >= so->mask*3) {
901:             if (set_table_resize(so, (so->used + dictsize)*2) != 0)
902:                 return -1;
903:         }
...
909:     }
910:
911:     it = PyObject_GetIter(other);
912:     if (it == NULL)
913:         return -1;
914:
915:     while ((key = PyIter_Next(it)) != NULL) {
916:         if (set_add_key(so, key)) {
```

The presize is reached only for a set argument (line 886, via `set_merge`, which
presizes at line 577) and for an exact dict (lines 900-903). A list, a tuple, and
every other iterable fall through to the generic loop at lines 911-922: no presize
at all. `grep` for `LengthHint`, `PyObject_Size` and `PyObject_Length` across
`setobject.c` returns nothing — there is no length-hint path in this file.

So `PySet_New(list)` performs the *same* `set_add_key` calls and therefore the
*same* resizes as the incremental loop, and adds on top of them: a list
allocation, N `Py_INCREF` into the list, an iterator object, N `PyIter_Next`
calls each returning a new strong reference, and N matching `Py_DECREF`.
It is strictly more work for identical table behaviour.

## What was measured anyway

A standalone extension with five variants, each building `list[set[node_id]]`
from a node-id sequence plus an int32 label buffer, mirroring the library's own
boundary exactly. Best of five after a warm-up; each measurement in a fresh
subprocess so peak RSS is attributable. Two independent full runs agreed within
noise.

- `incremental` — the current path: `PySet_New(NULL)` then `PySet_Add` per node
- `bulk_list` — count sizes, fill `PyList_New(size)` with `PyList_SET_ITEM`, one `PySet_New` per component
- `bulk_tuple` — same, staged in a `PyTupleObject`
- `frozen_list` — same, but `PyFrozenSet_New`; changes the returned type
- `via_dict` — stage in a `PyDict`, then `PySet_New(dict)`; the one non-set
  iterable this version does presize for

### `bulk_list` against the current path, wall time

| shape | 100K | 1M | 10M |
|---|---:|---:|---:|
| one giant component | +50.0% | +70.8% | +70.4% |
| a thousand equal components | +37.5% | **-14.8%** | -7.8% |
| a million singletons | +34.3% | **-19.4%** | not run |
| realistic sparse mixture | +250.0% | +133.1% | +54.6% |

### Full matrix, 1M elements

| shape | variant | ms | peak MB | vs current |
|---|---|---:|---:|---:|
| one giant | incremental | 6.5 | 136.8 | base |
| one giant | bulk_list | 11.1 | 144.8 | +70.8% |
| one giant | bulk_tuple | 11.0 | 144.7 | +69.2% |
| one giant | frozen_list | 12.2 | 144.8 | +87.7% |
| one giant | via_dict | 21.9 | 188.4 | +236.9% |
| thousand equal | incremental | 25.0 | 127.2 | base |
| thousand equal | bulk_list | 21.3 | 124.5 | -14.8% |
| thousand equal | bulk_tuple | 23.1 | 124.6 | -7.6% |
| thousand equal | frozen_list | 21.1 | 124.7 | -15.6% |
| thousand equal | via_dict | 58.8 | 208.4 | +135.2% |
| singletons | incremental | 748.5 | 306.7 | base |
| singletons | bulk_list | 603.1 | 411.1 | -19.4% |
| singletons | bulk_tuple | 768.9 | 379.1 | +2.7% |
| singletons | frozen_list | 594.0 | 411.2 | -20.6% |
| singletons | via_dict | 853.6 | 539.6 | +14.0% |
| sparse mixture | incremental | 55.9 | 163.5 | base |
| sparse mixture | bulk_list | 130.3 | 182.7 | +133.1% |
| sparse mixture | bulk_tuple | 136.2 | 179.6 | +143.6% |
| sparse mixture | frozen_list | 129.8 | 182.7 | +132.2% |
| sparse mixture | via_dict | 185.5 | 213.2 | +231.8% |

`via_dict` is the direct measurement of what the presize is worth when it is
actually taken: it is the slowest variant nearly everywhere, +135% at 1M on the
uniform shape and +232% on the realistic one. Obtaining the presize costs far
more than the resizes it avoids. Its one win, -35% at 100K singletons, is a
freelist artifact of one-element containers — a one-element set never resizes,
so no presize is involved.

`frozen_list` tracks `bulk_list` to within a couple of percent everywhere, so
giving up `set` for `frozenset` buys nothing. The type change is unnecessary
as well as unhelpful.

## Where the two wins come from, and why they do not count

`bulk_list` wins only on the two artificially uniform shapes, and only at 1M.
The cause is cache locality, not presizing: with a thousand live component
tables the incremental loop scatters writes across roughly 8 MB of hash tables,
past L2, while the bulk path writes sequentially into lists and then builds one
table at a time with only that table hot. This is why the effect appears at 1M,
weakens by 10M, and is absent at 100K where everything fits in cache anyway.

It is a narrow band in both dimensions, it reverses sign on either side of that
band, and it costs up to +34% peak memory on the singleton shape. Neither of
the two shapes a real graph actually produces is in the band.

## End-to-end

On a realistic sparse graph — 1M nodes, 1M random edges, 162,063 components,
largest 796,603 — through the real `pygraphc.connected_components`:

| | ms | share of call |
|---|---:|---:|
| full `connected_components` call | 83.8 | 100% |
| set construction, current path | 49.3 | 58.8% |
| set construction, `bulk_list` | 134.3 | **+101.6% of the whole call** |

Adopting `bulk_list` would take the realistic 1M-node call from 83.8 ms to
about 169 ms. It more than doubles it.

## Verified along the way

The node ids handed back are the caller's own `PyObject` pointers. `nid_parse`
holds the caller's sequence with `PySequence_Fast` and takes
`PySequence_Fast_ITEMS` (`_core.c:465-468`, `:506-509`); every set-returning
kernel then passes those borrowed pointers straight to `PySet_Add`, which only
increments. No `PyLong` is allocated per node on any path touched here. The
benchmark reproduces this exactly, so it is measuring the same work the library
does and not a more expensive stand-in.

A sibling branch found that scanning an int32 label buffer with `bytes.find` beat
a per-node Python comparison loop by 17x. That result is about moving a loop out
of Python into C. It does not transfer here: the baseline measured above is
already a tight C loop over the label buffer, so there is no interpreter overhead
left for that technique to remove.

## Conclusion

No variant wins. The presize the hypothesis rests on does not exist for a list
in CPython 3.11.11, and the one argument type that does get presized is the
slowest option measured. The only wins are a narrow cache-locality band on two
synthetic shapes that reverses outside it, and both shapes a real graph produces
lose — the realistic one by 55% to 250% in isolation and by more than the whole
call end to end.

Recommendation: **do not change the set-returning kernels.** The first findings
note's conclusion stands, and is now settled from the source rather than
inferred from timings. The incremental `PySet_Add` loop is the right code.
Getting past this cost means not building Python sets, which changes the API
contract the callers depend on.
