"""Estimate what PEP 839's PyFrozenSetWriter could buy this library.

The writer takes a size hint, inserts without resizing, and finishes without
copying.  None of that is callable today, but the *insertion half* of it is
measurable: a set argument and an exact dict argument both reach CPython's
presize path, so `PySet_New(source)` fills an already-sized table.  Building the
source container is done before the clock starts, which is exactly the cost the
writer removes, so the gap between the incremental `PySet_Add` loop and the
presized path is an upper bound on the writer's win.

Run with no arguments for the full sweep plus the combined workload estimate.
Each variant runs in a fresh subprocess so peak RSS is attributable.

    python benchmarks/pep839/bench_presize_estimate.py
"""

# A reporting script prints its tables (T201) and raises with the offending
# sizes inline (TRY003); numpy and pygraphc are imported inside the workload
# path so the timing subprocesses never load them (PLC0415).
# ruff: noqa: T201, TRY003, PLC0415

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import resource
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_presize_bench.c"
BUILD = HERE / "build"

SIZES = [1, 2, 3, 5, 10, 100, 1_000, 100_000, 800_000]
REPEATS = {
    1: 200_000,
    2: 200_000,
    3: 200_000,
    5: 150_000,
    10: 80_000,
    100: 15_000,
    1_000: 1_200,
    100_000: 20,
    800_000: 3,
}
TRIALS = 7
ID_SPACE = 1_000_000
VARIANTS = [
    "set_incremental",
    "frozen_incremental",
    "set_from_set",
    "set_from_poisoned_set",
    "set_from_dict",
    "frozen_from_set",
    "hash_only",
]
PRESIZED_VARIANTS = ["set_from_set", "set_from_dict", "frozen_from_set"]
ORDERS = ["sorted", "random"]

WORKLOAD_NODES = 1_000_000
WORKLOAD_EDGES = 1_000_000
WORKLOAD_SEED = 20260911
SET_CONSTRUCTION_MS = 49.3
FULL_CALL_MS = 83.8


def build_extension():
    """Compile and import the measurement extension."""
    BUILD.mkdir(exist_ok=True)
    target = BUILD / "_presize_bench.so"
    if not target.exists() or target.stat().st_mtime < SOURCE.stat().st_mtime:
        compiler = sysconfig.get_config_var("CC") or "cc"
        command = [
            *compiler.split(),
            "-O3",
            "-shared",
            "-undefined",
            "dynamic_lookup",
            "-I" + sysconfig.get_paths()["include"],
            str(SOURCE),
            "-o",
            str(target),
        ]
        subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location("_presize_bench", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def member_ids(size, rng, order):
    """Distinct node ids drawn from the workload's id space, as the caller's own ints.

    `sorted` matches the library's own boundary for integer node ids: the kernel
    walks nodes in index order, so ids — and therefore hashes, and therefore
    table slots — arrive ascending.  `random` is the faithful order for node ids
    whose hashes are not ordered, a string or a UUID.
    """
    members = rng.sample(range(ID_SPACE), size)
    return sorted(members) if order == "sorted" else members


def build_source(variant, members):
    """The container handed to the timed region, built before the clock starts.

    Drawing the spare ids from a private generator keeps every variant's member
    ids identical at every size.
    """
    if variant.endswith("incremental") or variant == "hash_only":
        return members
    if variant == "set_from_poisoned_set":
        # Dummies make other->fill != other->used, which disqualifies the
        # pointer-copy clone path in set_merge (setobject.c:586) and forces the
        # set_insert_clean loop at :602 instead.
        spare = random.Random(99).sample(range(ID_SPACE, 2 * ID_SPACE), len(members))
        poisoned = set(members) | set(spare)
        for identifier in spare:
            poisoned.discard(identifier)
        return poisoned
    if variant.endswith("from_set"):
        return set(members)
    return dict.fromkeys(members)


def run_variant(variant, order):
    """Time one variant across every size; return per-size seconds and peak RSS."""
    bench = build_extension()
    rng = random.Random(4242)
    results = {}
    for size in SIZES:
        members = member_ids(size, rng, order)
        repeats = REPEATS[size]
        source = build_source(variant, members)
        bench.time_build(variant, source, min(repeats, 8))
        trials = []
        for _ in range(TRIALS):
            elapsed, containers, total = bench.time_build(variant, source, repeats)
            if containers != repeats or total != repeats * size:
                raise AssertionError(f"{variant} at size {size} built the wrong thing")
            trials.append(elapsed)
        trials.sort()
        results[size] = {
            "repeats": repeats,
            "min_per_set_ns": trials[0] / repeats * 1e9,
            "median_per_set_ns": trials[len(trials) // 2] / repeats * 1e9,
        }
        del source, members
    return {"variant": variant, "sizes": results, "peak_rss_mb": peak_rss_mb()}


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_peak(variant, order):
    """Peak RSS of building a single largest-size container on one path."""
    bench = build_extension()
    rng = random.Random(4242)
    size = SIZES[-1]
    members = member_ids(size, rng, order)
    baseline = peak_rss_mb()
    source = build_source(variant, members)
    after_source = peak_rss_mb()
    bench.time_build(variant, source, 1)
    return {
        "variant": variant,
        "size": size,
        "members_mb": baseline,
        "after_source_mb": after_source,
        "peak_mb": peak_rss_mb(),
    }


def component_sizes():
    """Component sizes of the real 1M-node workload, largest first."""
    import numpy

    rng = numpy.random.default_rng(WORKLOAD_SEED)
    node_ids = list(range(WORKLOAD_NODES))
    source = rng.integers(0, WORKLOAD_NODES, WORKLOAD_EDGES).tolist()
    target = rng.integers(0, WORKLOAD_NODES, WORKLOAD_EDGES).tolist()

    from pygraphc import connected_components

    timings = []
    for _ in range(5):
        started = time.perf_counter()
        components = list(connected_components(node_ids, source, target))
        timings.append(time.perf_counter() - started)
        sizes = sorted((len(component) for component in components), reverse=True)
        del components
    return sizes, min(timings)


def timing(measured, variant, size, key):
    return measured[variant]["sizes"][str(size)][key]


def loose_ceiling_ns(measured, size, key):
    """Fastest presized path, whatever shortcut it takes internally."""
    return min(timing(measured, variant, size, key) for variant in PRESIZED_VARIANTS)


def writer_model_ns(measured, size, key):
    """Presized insertion that still hashes every item, as a writer must.

    The dict path calls `set_add_entry` per key with a full duplicate-checking
    probe but reuses the hash the dict already stored (setobject.c:904-905), so
    the per-item `PyObject_Hash` a writer cannot avoid is added back.
    """
    return timing(measured, "set_from_dict", size, key) + timing(measured, "hash_only", size, key)


def saving_points(measured, key, presized):
    """Per-element saving in ns at each measured size, as (size, ns) pairs."""
    return [
        (
            size,
            (timing(measured, "set_incremental", size, key) - presized(measured, size, key)) / size,
        )
        for size in SIZES
    ]


def interpolate(points, size):
    """Per-element saving at an arbitrary size, linear in log(size)."""
    if size <= points[0][0]:
        return points[0][1]
    if size >= points[-1][0]:
        return points[-1][1]
    for (low_size, low), (high_size, high) in zip(points, points[1:], strict=False):
        if low_size <= size <= high_size:
            weight = (math.log(size) - math.log(low_size)) / (math.log(high_size) - math.log(low_size))
            return low + weight * (high - low)
    raise AssertionError("size outside the measured range")


def combined_saving_ms(measured, sizes, key, presized):
    points = saving_points(measured, key, presized)
    return sum(size * interpolate(points, size) for size in sizes) / 1e6


def print_table(measured):
    header = f"{'size':>8} " + " ".join(f"{variant:>21}" for variant in VARIANTS)
    for label, per_set in (("Per-set", True), ("Per-element", False)):
        print(f"\n{label} nanoseconds, best of {TRIALS} (median in parentheses)")
        print(header)
        for size in SIZES:
            divisor = 1 if per_set else size
            digits = 1 if per_set else 2
            cells = [
                f"{timing(measured, variant, size, 'min_per_set_ns') / divisor:.{digits}f} "
                f"({timing(measured, variant, size, 'median_per_set_ns') / divisor:.{digits}f})"
                for variant in VARIANTS
            ]
            print(f"{size:>8} " + " ".join(f"{cell:>21}" for cell in cells))

    print(f"\nSaving per set (ns), best of {TRIALS}")
    print(
        f"{'size':>8} {'incremental':>13} {'writer model':>13} {'saving':>12} {'%':>7}"
        f"   {'loose ceiling':>13} {'saving':>12} {'%':>7}"
    )
    for size in SIZES:
        incremental = timing(measured, "set_incremental", size, "min_per_set_ns")
        model = writer_model_ns(measured, size, "min_per_set_ns")
        ceiling = loose_ceiling_ns(measured, size, "min_per_set_ns")
        print(
            f"{size:>8} {incremental:>13.1f} {model:>13.1f} {incremental - model:>12.1f} "
            f"{(incremental - model) / incremental * 100:>6.1f}%   {ceiling:>13.1f} "
            f"{incremental - ceiling:>12.1f} "
            f"{(incremental - ceiling) / incremental * 100:>6.1f}%"
        )

    print("\nPeak RSS per variant (MB), whole sweep in a fresh subprocess")
    for variant in VARIANTS:
        print(f"  {variant:<24} {measured[variant]['peak_rss_mb']:>8.1f}")


def predicted_construction_ms(measured, sizes, key):
    """What the microbenchmark's incremental path says the whole workload costs."""
    points = [(size, timing(measured, "set_incremental", size, key) / size) for size in SIZES]
    return sum(size * interpolate(points, size) for size in sizes) / 1e6


def print_estimate(measured, workload):
    sizes = workload["sizes"]
    print(
        f"\nWorkload: {WORKLOAD_NODES} nodes, {WORKLOAD_EDGES} edges, "
        f"{len(sizes)} components, largest {sizes[0]}, "
        f"best-of-5 call {workload['call_seconds'] * 1e3:.1f} ms"
    )
    predicted = predicted_construction_ms(measured, sizes, "min_per_set_ns")
    calibration = SET_CONSTRUCTION_MS / predicted
    print(
        f"Microbenchmark predicts {predicted:.1f} ms of set construction for this "
        f"distribution against the {SET_CONSTRUCTION_MS} ms the real call spends, "
        f"so absolute savings scale by {calibration:.2f}"
    )
    models = (("writer model", writer_model_ns), ("loose ceiling", loose_ceiling_ns))
    for label, presized in models:
        best = combined_saving_ms(measured, sizes, "min_per_set_ns", presized)
        typical = combined_saving_ms(measured, sizes, "median_per_set_ns", presized)
        low, high = sorted((best, typical))
        largest = sizes[0] * interpolate(saving_points(measured, "min_per_set_ns", presized), sizes[0]) / 1e6
        share_low = low / predicted * 100
        share_high = high / predicted * 100
        print(
            f"{label:<14} raw saving {low:.1f} to {high:.1f} ms = "
            f"{share_low:.0f}-{share_high:.0f}% of construction; calibrated onto the "
            f"real call: {low * calibration:.1f} to {high * calibration:.1f} ms, "
            f"{low * calibration / FULL_CALL_MS * 100:.0f}-"
            f"{high * calibration / FULL_CALL_MS * 100:.0f}% of the {FULL_CALL_MS} ms "
            f"call; largest component alone {largest:.1f} ms raw"
        )

    histogram = {}
    for size in sizes:
        bucket = 10 ** int(math.log10(size))
        histogram[bucket] = histogram.get(bucket, 0) + 1
    print(
        "\nComponent size histogram (decade buckets): "
        + ", ".join(f"{bucket}+: {count}" for bucket, count in sorted(histogram.items()))
    )


def in_subprocess(*arguments):
    finished = subprocess.run(
        [sys.executable, __file__, *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return finished.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--order", choices=ORDERS, default=ORDERS[0])
    parser.add_argument("--peak", action="store_true")
    parser.add_argument("--workload", action="store_true")
    arguments = parser.parse_args()

    if arguments.variant:
        runner = run_peak if arguments.peak else run_variant
        print(json.dumps(runner(arguments.variant, arguments.order)))
        return
    if arguments.workload:
        sizes, call_seconds = component_sizes()
        print(json.dumps({"sizes": sizes, "call_seconds": call_seconds}))
        return

    build_extension()
    workload = json.loads(in_subprocess("--workload"))
    for order in ORDERS:
        print(f"\n{'=' * 30} insertion order: {order} {'=' * 30}")
        measured = {variant: json.loads(in_subprocess("--variant", variant, "--order", order)) for variant in VARIANTS}
        print_table(measured)

        print(f"\nPeak RSS (MB) building one container of {SIZES[-1]} members, fresh process")
        print(f"  {'variant':<24} {'members':>9} {'+source':>9} {'+built':>9}")
        for variant in VARIANTS:
            peak = json.loads(in_subprocess("--variant", variant, "--order", order, "--peak"))
            print(f"  {variant:<24} {peak['members_mb']:>9.1f} {peak['after_source_mb']:>9.1f} {peak['peak_mb']:>9.1f}")

        print_estimate(measured, workload)


if __name__ == "__main__":
    main()
