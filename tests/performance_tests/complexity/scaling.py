"""Walltime at doubling input sizes and the fitted scaling exponent.

Sizes adapt to each operation instead of being fixed: a case starts small and
doubles until one call takes ``STOP_SECONDS`` or the size reaches
``MAX_SIZE``. Each size repeats the call until ``SAMPLE_SECONDS`` have
accumulated, so even a fast operation yields a clean per-call time at a
moderate size, and the exponent comes from the last doublings, where the
linear term no longer hides a superlinear one. A quadratic case
reaches the stop after a few doublings, long before it costs minutes or
exhausts memory; the cases this tier found were obvious from 25,000 on.

Noise does not blur the answer: a 20% timing error moves one doubling by
about 0.26, and averaging two of them halves that, while linear and
quadratic implementations lie 0.4 apart.
"""

import gc
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from performance_tests.complexity.known_superlinear import KNOWN_SUPERLINEAR

START_SIZE = 4_000
MAX_SIZE = 512_000
STOP_SECONDS = 0.1
SAMPLE_SECONDS = 0.02
SAMPLES = 2
FIT_POINTS = 4
"""Sizes measured before a slow call may end the series: three doublings to judge."""
BLOW_UP_SECONDS = 1.0
"""A call this slow ends the series after three sizes, enough to fit a blow-up."""
LINEAR_LIMIT = 1.5
"""Upper bound for linear and n log n operations.

In theory they measure 1.0 to 1.1; with the cache step left out, linear
operations measured up to 1.29 and every quadratic implementation 1.68 or more.
"""


@dataclass(frozen=True)
class Scaling:
    """Per-call walltime at every measured size and the least-squares exponent over the largest sizes."""

    sizes: tuple[int, ...]
    seconds: tuple[float, ...]

    @property
    def exponent(self) -> float:
        """Doubling exponent over the last three doublings, the largest of them left out.

        Each doubling contributes log2 of its time ratio. Leaving out the
        largest of the last three ignores one step, the jump where the working
        set leaves a CPU cache, which lands at a different size on every
        machine; a superlinear case raises every ratio and is not hidden by it.
        """
        doublings = [
            math.log2(max(later, 1e-12) / max(earlier, 1e-12))
            for earlier, later in zip(self.seconds, self.seconds[1:], strict=False)
        ][-3:]
        if len(doublings) >= 3:
            return sum(sorted(doublings)[:-1]) / (len(doublings) - 1)
        return min(doublings)

    def __str__(self) -> str:
        rows = ", ".join(
            f"{size:,}: {seconds * 1000:.3g} ms" for size, seconds in zip(self.sizes, self.seconds, strict=True)
        )
        return f"exponent {self.exponent:.2f} ({rows})"


def _per_call_seconds(call: Callable[[], object]) -> float:
    """Fastest of a few samples, each repeating the call until it has run ``SAMPLE_SECONDS``."""
    fastest = math.inf
    for _ in range(SAMPLES):
        gc.collect()
        calls = 0
        start = time.perf_counter()
        while True:
            call()
            calls += 1
            elapsed = time.perf_counter() - start
            if elapsed >= SAMPLE_SECONDS:
                break
        fastest = min(fastest, elapsed / calls)
    return fastest


def _single_call_seconds(call: Callable[[], object]) -> float:
    """One timed call after a collection, for sizes where a call is already slow."""
    gc.collect()
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def measure_scaling(
    prepare: Callable[[int], Callable[[], object]],
    start_size: int = START_SIZE,
    max_size: int = MAX_SIZE,
) -> Scaling:
    """Double the size from ``start_size`` until a call takes ``STOP_SECONDS`` or ``max_size`` is passed.

    ``prepare`` builds the input for a size outside the timer and returns the
    call to time. The first call of every size is a warm-up and carries
    one-time work such as lazily built caches, so it is never the measurement
    unless it blows up: a slow warm-up is followed by one timed call.
    """
    sizes: list[int] = []
    seconds: list[float] = []
    size = start_size
    while size <= max_size:
        call = prepare(size)
        gc.collect()
        start = time.perf_counter()
        call()
        warm_up_seconds = time.perf_counter() - start
        if warm_up_seconds >= BLOW_UP_SECONDS:
            per_call = warm_up_seconds
        elif warm_up_seconds >= STOP_SECONDS:
            per_call = min(warm_up_seconds, _single_call_seconds(call))
        else:
            per_call = _per_call_seconds(call)
        sizes.append(size)
        seconds.append(per_call)
        blown_up = per_call >= BLOW_UP_SECONDS and len(sizes) >= 3
        if blown_up or (per_call >= STOP_SECONDS and len(sizes) >= FIT_POINTS):
            break
        size *= 2
    return Scaling(tuple(sizes), tuple(seconds))


def assert_scales_linearly(case: str, scaling: Scaling) -> None:
    """Fail when the exponent exceeds the linear limit, or the case's ratchet limit if it is a known case.

    Prints every measurement, so a CI log is the calibration record of its runner.
    """
    exponent_limit, cause = KNOWN_SUPERLINEAR.get(case, (LINEAR_LIMIT, ""))
    print(f"\n  {case}: {scaling} [limit {exponent_limit}]")  # noqa: T201 — the calibration record
    assert scaling.exponent <= exponent_limit, f"{case} scales worse than n^{exponent_limit}: {scaling} {cause}"
