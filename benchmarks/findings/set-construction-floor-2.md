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

## What CPython is doing about it (searched 2026-09-11)

The mechanism this experiment hypothesised does not exist today, and that is
why every variant measured worse. It is being added.

[PEP 839](https://peps.python.org/pep-0839/) proposes `PyFrozenSetWriter`:

```c
PyFrozenSetWriter *PyFrozenSetWriter_Create(Py_ssize_t size_hint);
int  PyFrozenSetWriter_Add(PyFrozenSetWriter *writer, PyObject *item);
int  PyFrozenSetWriter_Update(PyFrozenSetWriter *writer, PyObject *iterable);
PyObject *PyFrozenSetWriter_Finish(PyFrozenSetWriter *writer);
void PyFrozenSetWriter_Discard(PyFrozenSetWriter *writer);
```

That is exactly the missing piece: a size hint taken up front, described as a
hint rather than a limit, with construction in a single pass, no intermediate
container and no copy at `Finish`. Three things make it unusable here:

- it is **Draft** status targeting **Python 3.16**, and this package supports
  3.11 to 3.13;
- it is **frozenset only**, so adopting it changes the return type of every
  set-returning call, and read-only set algebra would still work while
  mutation would not;
- the PEP reports **no benchmarks**, so the size of the win is unmeasured even
  by its authors.

Two adjacent facts from the same search, worth recording for later:

- `PySet_Add` is **soft deprecated on frozensets as of Python 3.14**, with the
  writer named as its replacement. A frozenset-returning API would eventually
  want the writer rather than the add loop.
- There is an open [performance regression in `PySet_Add` on free-threaded
  3.14](https://github.com/python/cpython/issues/140476): a critical section is
  taken even when the set is uniquely referenced during construction. Relevant
  if this package ever supports free-threaded builds, where set construction
  is already the dominant cost.

**Revisit when 3.16 is a supported target**, and only together with a decision
about whether any call may return a frozenset. Until then the conclusion of
this note stands unchanged: there is no faster path, and the two negative cells
in the matrix above are cache locality on synthetic inputs, not presizing.

---

# Addendum: what PEP 839's writer could actually buy, measured (2026-09-11)

The note above closes with "revisit when 3.16 is a supported target". This
section puts a number on that decision now, so that it and the feedback sent to
the PEP's author rest on measurement rather than on hope.

Measurement script: `benchmarks/pep839/bench_presize_estimate.py` with
`benchmarks/pep839/_presize_bench.c`. Interpreter as above,
`3.11.11 (main, Jul 16 2025, 22:29:00) [Clang 17.0.0]`, arm64 macOS. Five
independent full runs — three covering both insertion orders, two covering the
scattered order only; best of seven trials per cell, median alongside. The
tables below are one run, the last; the ranges span all of them.

## What is measurable today, and why the answer is an upper bound

`PyFrozenSetWriter` is `Create(size_hint)` + `Add` × N + `Finish`. None of it is
callable on 3.11. The *insertion half* of it is, because the source above
settles that a **set** argument and an **exact dict** argument both reach the
presize path: `PySet_New(source)` fills a table that was sized once, up front.

So the measurement is: build the source container **before the clock starts**,
time only the `PySet_New(source)` call, and compare against the library's
current path, `PySet_New(NULL)` followed by N `PySet_Add`. The excluded source
construction is precisely the work the writer removes, which makes the gap
**an upper bound on what the writer can save**. Three separate reasons the true
win must come out smaller:

1. A real writer pays its own `Create` and `Finish`. Not included here.
2. Both presized paths **reuse a hash somebody else already computed**. The dict
   path calls `set_add_entry(so, key, hash)` with the hash `_PyDict_Next` hands
   back (`setobject.c:904-905`); a writer's `Add` receives a bare `PyObject *`
   and must call `PyObject_Hash` itself.
3. With a **set** argument the path degenerates entirely. `set_merge` copies
   pointers slot by slot when the masks match and the source has no dummies
   (`setobject.c:584-599`), and uses `set_insert_clean` — no duplicate check —
   whenever the destination is merely empty (`:601-615`):

```c
584:     /* If our table is empty, and both tables have the same size, and
585:        there are no dummies to eliminate, then just copy the pointers. */
586:     if (so->fill == 0 && so->mask == other->mask && other->fill == other->used) {
...
601:     /* If our table is empty, we can use set_insert_clean() */
602:     if (so->fill == 0) {
```

   At 800,000 members that path costs **2.7 ns per element**, which is a table
   clone, not insertion. A writer is handed items one at a time and can never
   do this.

Two models are therefore reported:

- **writer model** — the dict path plus a separately measured `PyObject_Hash`
  per item, added back. Closest available analogue of a writer's `Add` loop:
  presized table, one full duplicate-checking probe per item, hashing paid.
- **loose ceiling** — the fastest presized path, as the question was posed. It
  is a true upper bound and a useless target: it is the clone above.

A control pins the third point down. Poisoning the source set with dummies
(`fill != used`) disqualifies the clone and forces `set_insert_clean`; the
`set_from_poisoned_set` column is that loop, and it is the floor of a writer's
`Add` — what insertion into a presized table costs with no duplicate check and
no hashing.

## Insertion order dominates everything else

For integer node ids the library hands ids to the set in **ascending** order:
the kernel walks nodes in index order, `hash(int) == int`, so the slot index
`hash & mask` ascends too and the writes are sequential. Resizing a table whose
contents are laid out in ascending slot order is also a sequential walk. Both
orders were measured; the difference is a factor of five on the largest
component, and it decides the answer.

### Per-element nanoseconds, ascending ids (faithful to this workload)

| members | incremental `PySet_Add` | from set (clone) | from poisoned set (clean insert) | from exact dict (presized + probe) | `PyObject_Hash` alone |
|---:|---:|---:|---:|---:|---:|
| 1 | 186.6 | 196.1 | 193.4 | 195.9 | 2.2 |
| 2 | 100.2 | 104.7 | 102.0 | 108.8 | 2.2 |
| 3 | 76.5 | 72.1 | 68.6 | 79.2 | 2.0 |
| 5 | 70.4 | 42.1 | 42.5 | 51.4 | 1.9 |
| 10 | 33.4 | 23.0 | 24.7 | 26.2 | 1.8 |
| 100 | 21.5 | 11.7 | 11.8 | 15.5 | 1.7 |
| 1,000 | 11.3 | 4.7 | 7.0 | 9.5 | 1.6 |
| 100,000 | 15.5 | 6.7 | 8.0 | 7.0 | 2.2 |
| 800,000 | 8.9 | 2.7 | 3.7 | 4.6 | 1.6 |

### Per-element nanoseconds, scattered ids (node ids whose hashes are not ordered)

| members | incremental `PySet_Add` | from set (clone) | from poisoned set (clean insert) | from exact dict (presized + probe) | `PyObject_Hash` alone |
|---:|---:|---:|---:|---:|---:|
| 1 | 219.2 | 194.0 | 202.4 | 206.4 | 2.2 |
| 2 | 115.8 | 105.1 | 104.1 | 114.9 | 2.2 |
| 3 | 73.5 | 68.5 | 73.7 | 79.1 | 2.0 |
| 5 | 60.7 | 44.7 | 44.0 | 47.6 | 1.9 |
| 10 | 26.8 | 23.5 | 24.8 | 28.0 | 1.7 |
| 100 | 20.8 | 11.9 | 11.9 | 15.9 | 1.7 |
| 1,000 | 11.6 | 4.7 | 5.7 | 9.8 | 1.6 |
| 100,000 | 28.1 | 6.7 | 8.7 | 25.6 | 2.7 |
| 800,000 | 44.1 | 2.7 | 3.7 | 24.6 | 9.1 |

### The saving per set, in nanoseconds

Small sizes are reported per set over a large batch (200,000 sets at size 1
down to 3 at size 800,000), because one tiny set is unmeasurable.

| members | ascending: writer model | ascending: loose ceiling | scattered: writer model | scattered: loose ceiling |
|---:|---:|---:|---:|---:|
| 1 | -11 | -5 | 11 | 25 |
| 2 | -21 | -9 | -3 | 21 |
| 3 | -14 | 13 | -23 | 18 |
| 5 | 85 | 142 | 56 | 80 |
| 10 | 54 | 104 | -28 | 34 |
| 100 | 428 | 982 | 316 | 892 |
| 1,000 | 143 | 6,605 | 236 | 7,108 |
| 100,000 | 628,450 | 881,350 | -23,000 | 2,136,550 |
| 800,000 | 2,131,000 | 4,937,000 | 8,305,700 | 33,091,700 |

Below about 5 members there is nothing to save and the numbers say so: the
`smalltable` of 8 slots holds up to 4 entries without a resize, so the ~200 ns
is object allocation and the sign of the difference is noise. Between 10 and
1,000 members the writer model is worth 1-3% of the per-set cost. Only above
100,000 members does it become a real fraction, and only in scattered order
does it become a large one.

## Peak memory

Building one 800,000-member container in a fresh process, RSS in MB, measured
after the member list (65.0), after the source container, and after the build:

| path | + source | + built container |
|---|---:|---:|
| incremental `PySet_Add` | — | 126.5 |
| from set | 126.5 | 160.1 |
| from exact dict | 143.5 | 177.1 |
| from poisoned set | 294.5 | 294.5 |

The incremental build costs **60 to 62 MB** for a set whose final table is 33.6
MB (2²¹ slots × 16 bytes): the extra is the doubling transient, the 2²⁰-slot
table still alive while the 2²¹ one is filled. Every presized build costs
**33.6 MB**, the final table and nothing else. This is the one place where the
writer wins clearly and for a structural reason rather than by a few
nanoseconds: roughly **28 MB less peak** on the largest component, against the
136-163 MB peak this note records for the whole call. The presize also picks a
smaller table than incremental growth does at middling sizes — `(used + hint) *
2` against `used * 4` — so a 1,000-member set lands in 2,048 slots instead of
4,096.

The whole-sweep RSS figures (440 MB incremental, 416 MB from set, 433 MB from
dict, 480 MB poisoned, 67 MB hash-only) are high-water marks over all nine
sizes and compare the paths only loosely; the single-container table above is
the attributable one.

## The combined estimate for the real workload

Same graph as above: 1,000,000 nodes, 1,000,000 random edges, reproduced here as
**161,844 components, largest 796,524** (the note's run: 162,063 and 796,603).
The distribution is 135,016 singletons, 18,515 pairs, 5,012 triples, a tail
reaching 17 members — and one component holding 79.7% of all nodes.

Per-size savings were interpolated log-linearly in size and summed over the real
distribution. Because the largest component is 796,524 members and the largest
measured size is 800,000, the estimate is dominated by a cell that was measured
almost exactly, not by interpolation. The 161,843 small components contribute at
or below noise, sometimes negative, within about ±1 ms of zero in total.

The microbenchmark's own incremental path predicts 38-40 ms of set construction
for this distribution in ascending order against the 49.3 ms the real call
spends, so absolute savings are scaled by that ratio (×1.22 to ×1.30) to land on
the real call. In scattered order it predicts 66-71 ms, scaling by ×0.69 to
×0.74.

| | upper bound on the saving | of the 49.3 ms of set construction | of the 83.8 ms call |
|---|---:|---:|---:|
| **writer model, ascending ids** (this workload) | **0.4 to 3.1 ms** | **1-6%** | **0.5-4%** |
| writer model, scattered ids | 4.1 to 7.9 ms | 8-16% | 5-9% |
| loose ceiling, ascending ids | 5.8 to 9.3 ms | 12-19% | 7-11% |
| loose ceiling, scattered ids | 24.4 to 26.5 ms | 50-54% | 29-32% |

Ranges span the independent runs — three in ascending order, five in scattered
order — and, within each run, best-of-seven against median trial. In ascending
order the estimate sits at the edge of measurability: the largest component
alone accounts for 2.1 ms of saving, and the 161,843 small components cancel
most of it with measured savings that are slightly negative.

Every figure in the table is an upper bound: none of them charges the writer for
its own `Create` and `Finish`, and none of them charges it for the table-sizing
policy it will actually use, which has to keep a load factor below about 60% and
so may size differently from the `(hint) * 2` the presize picks.

**The answer for this library is under 1 to 3 ms of an 83.8 ms call**, because
its node ids are integers in a contiguous range and therefore arrive in
ascending hash order, which already makes the resizes nearly free. Node ids
with scattered hashes — strings, UUIDs — would see 4 to 8 ms.

One more cost belongs in the decision. The writer is frozenset-only, and the
`frozen_incremental` column measures what that type change costs on today's add
loop: **+12% to +16% at 800,000 members** and +18% to +29% at 1,000 members
(`PySet_Add` on a frozenset takes the refcount-checking branch). A non-trivial
slice of the writer's win would go to paying back the switch it forces.

## Recommendation

**No — do not plan to adopt the writer, and do not require Python 3.16 for
it.** At most three milliseconds, possibly well under one, of an 83.8 ms call —
for the price of dropping 3.11 through 3.15 and changing the return type of
every set-returning call from `set` to `frozenset` — is not a trade worth
making. The estimate is an upper bound and the real figure is smaller.

Revisit only when one of these changes:

- **3.16 becomes the support floor for unrelated reasons.** Then the writer costs
  nothing to adopt and should be taken: the win is small but real, and the
  ~28 MB lower peak on large components is the better half of it.
- **Node ids with unordered hashes become the common case.** 4 to 8 ms, 5-9% of
  the call, is still not large, but it is four times the ordered-id figure.
- **Free-threaded builds become a target.** Set construction is already the
  dominant cost there, and the open `PySet_Add` critical-section regression
  (cpython#140476) makes the add loop worse while leaving the writer untouched.

## Feedback worth sending to the PEP

1. **Publish benchmarks, and vary insertion order in them.** The size of the win
   here moves by 5× depending on whether items arrive in ascending hash order.
   A benchmark that only inserts `range(n)` will overstate the resize cost it
   removes for scattered inputs and understate the win for ordered ones.
2. **The clearest benefit measured is peak memory, not time**: 33.6 MB instead
   of about 61 MB for an 800,000-member container, because the doubling
   transient never exists. That is worth stating in the PEP; it is a stronger
   argument than the time saved.
3. **Let `Add` take a precomputed hash.** A C producer often already has it, and
   the measurements above show that reusing a stored hash is where the presized
   paths keep their remaining advantage — 1.6 to 9.1 ns per item, which is 35%
   to 37% of the presized insertion cost itself.
4. **A `PySetWriter` for mutable sets would widen the audience.** A
   frozenset-only API forces a return-type change on any library that hands back
   mutable sets, and that change alone costs 12-29% on the current add loop
   before the writer recovers anything.
5. Note that `PySet_Add`'s soft deprecation on frozensets from 3.14 leaves
   frozenset producers with no non-deprecated incremental path until 3.16.

## The struct-level prototype was not attempted

Part two — allocating a frozenset, sizing its table by hand through
`PySetObject` and filling it directly — was judged not worth the risk, because
the two paths already reached through the public API **bracket a writer's `Add`
loop from both sides on the real interpreter**:

- `set_insert_clean` via the poisoned set — presized table, no duplicate check:
  3.7 ns per element at 800,000.
- `set_add_entry` via the exact dict — presized table, full duplicate check:
  4.6 ns ascending, 24.6 ns scattered.

A hand-written prototype would execute one of those same two loops. It would
not produce a number the public API cannot already produce, and it would depend
on non-stable struct layout with a subtly-broken-object failure mode that only
exhaustive validation can catch. The first findings note also records that
direct `PySetObject` table manipulation was already tried on this problem and
measured 6-15% *worse*. The upper bound from part one is unambiguous enough to
decide on: no prototype was written, and none is needed.
