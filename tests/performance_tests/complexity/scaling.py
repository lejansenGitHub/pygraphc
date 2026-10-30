"""Walltime at doubling input sizes and the fitted scaling exponent.

Sizes adapt to each operation instead of being fixed: a case starts small and
doubles until one call takes ``STOP_SECONDS`` or the size reaches
``MAX_SIZE``. Each size repeats the call until ``SAMPLE_SECONDS`` have
accumulated, so even a fast operation yields a clean per-call time at a
moderate size, and the exponent is fitted over the last ``FIT_POINTS`` sizes,
where the linear term no longer hides a superlinear one. A quadratic case
reaches the stop after a few doublings, long before it costs minutes or
exhausts memory; the cases this tier found were obvious from 25,000 on.

Noise does not blur the answer: a 20% timing error moves the fit by about
0.1, while linear and quadratic cases lie 0.3 apart even after cache effects.
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
BLOW_UP_SECONDS = 1.0
"""A call this slow ends the series after three sizes, enough to fit a blow-up."""
LINEAR_LIMIT = 1.5
"""Upper bound for linear and n log n operations.

In theory they fit 1.0 to 1.1, but between 64,000 and 512,000 the working set
moves through the CPU caches and the cost per element rises: linear
operations measured up to 1.42. Every quadratic case measured 1.73 or more.
"""


@dataclass(frozen=True)
class Scaling:
    """Per-call walltime at every measured size and the least-squares exponent over the largest sizes."""

    sizes: tuple[int, ...]
    seconds: tuple[float, ...]

    @property
    def exponent(self) -> float:
        """Slope of log time over log size across the last ``FIT_POINTS`` sizes."""
        log_sizes = [math.log(size) for size in self.sizes[-FIT_POINTS:]]
        log_seconds = [math.log(max(seconds, 1e-9)) for seconds in self.seconds[-FIT_POINTS:]]
        mean_size = sum(log_sizes) / len(log_sizes)
        mean_seconds = sum(log_seconds) / len(log_seconds)
        covariance = sum(
            (log_size - mean_size) * (log_second - mean_seconds)
            for log_size, log_second in zip(log_sizes, log_seconds, strict=True)
        )
        variance = sum((log_size - mean_size) ** 2 for log_size in log_sizes)
        return covariance / variance

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
