"""Guard on the workflow phase profiles in ``benchmarks/profile_workflows.py``.

Deliberately coarse. This is not a benchmark: it never asserts that anything is
fast, only that the *shape* of a workflow has not changed — that no phase has
taken over its workflow and that the instrumentation still accounts for the
time it measures. A guard that cries wolf gets deleted by the next person, so
every margin below is wide enough that machine-to-machine variation and CI
noise cannot reach it, and each one says why it is what it is.

Run the harness yourself to see the numbers::

    python benchmarks/profile_workflows.py --size guard
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPOSITORY_ROOT / "benchmarks" / "profile_workflows.py"
BASELINE_PATH = REPOSITORY_ROOT / "benchmarks" / "baseline.json"

# A phase's share of its workflow may move by this many percentage points. Shares
# are ratios, so a uniformly slower machine does not move them at all; what does
# move them is a phase that scales differently from the rest (memory bandwidth
# against branch prediction, a different allocator, a GC pause landing in one
# phase). Measured drift between this laptop and a CI runner is a few points.
# 20 points still catches a phase that doubles out of half the runtime, or one
# that appears where there was almost nothing, which is the leak this guard is for.
SHARE_MARGIN = 0.20

# A phase is compared only when it is at least this share of its workflow either
# at baseline or now — the "or now" matters, otherwise a phase growing out of
# nothing, which is what a new leak looks like, would be exempt. Below a tenth
# of a workflow that runs in milliseconds at the guard sizes, a phase is a few
# hundred microseconds and its share is dominated by timer granularity and
# allocator luck, not by the code.
SHARE_FLOOR = 0.10

# A workflow's total may be this multiple of its baseline total. The baseline
# comes from one developer machine; a shared-vCPU CI runner without turbo is
# routinely three to four times slower on single-threaded Python, and a noisy
# neighbour adds more. Eight times leaves room for all of that and still fails
# on an order-of-magnitude regression, which is the only total-time claim worth
# making from a machine nobody controls.
TOTAL_MULTIPLE = 8.0

# The phases of a workflow must account for at least this share of the wall time
# measured around the whole body. The harness attributes even the teardown of a
# workflow's own inputs to a phase, so it reports 99.5% or better here; the
# remainder is the result string and the frame teardown of a few small locals.
# 95% leaves room for one GC pause landing between two phases and still fails
# loudly if a whole step of a workflow is left uninstrumented, which would make
# the whole table a lie about where the time goes.
MINIMUM_ACCOUNTED_SHARE = 0.95


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    """The benchmark harness imported by path: ``benchmarks/`` is not a package."""
    specification = importlib.util.spec_from_file_location("profile_workflows", HARNESS_PATH)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    """The committed numbers, measured at the guard sizes."""
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def measurements(harness: ModuleType) -> list[Any]:
    """One harness run at the guard sizes, without artifacts, peaks or networkx."""
    return [
        harness.measure(workflow, workflow.guard_size, with_peak=False, with_networkx=False)
        for workflow in harness.WORKFLOWS
    ]


def report(harness: ModuleType, measurements: list[Any], problems: list[str]) -> str:
    """The failure message: what moved, then the whole table it moved in."""
    return "\n".join([*problems, "", harness.console_table(measurements)])


def test_baseline_covers_every_workflow_and_phase(
    harness: ModuleType,
    baseline: dict[str, Any],
    measurements: list[Any],
) -> None:
    """A new workflow or a renamed phase needs a regenerated baseline, not a silent pass."""
    baselined = baseline["workflows"]
    problems = [
        f"{measurement.name}: no baseline, regenerate with --size guard --write-baseline"
        for measurement in measurements
        if measurement.name not in baselined
    ]
    for measurement in measurements:
        entry = baselined.get(measurement.name)
        if entry is None:
            continue
        new_phases = [name for name in measurement.phase_seconds if name not in entry["phase_seconds"]]
        gone_phases = [name for name in entry["phase_seconds"] if name not in measurement.phase_seconds]
        if new_phases or gone_phases:
            problems.append(f"{measurement.name}: phases added {new_phases}, phases gone {gone_phases}")
    stale = [name for name in baselined if name not in {measurement.name for measurement in measurements}]
    if stale:
        problems.append(f"baseline holds workflows the registry no longer has: {stale}")
    # --- Assert ---
    assert not problems, report(harness, measurements, problems)


def test_phases_account_for_the_measured_total(harness: ModuleType, measurements: list[Any]) -> None:
    """Self-consistency: the phases must add up to the wall time of the workflow.

    Without this the table could attribute a tenth of a workflow and stay
    silent about the rest, which is exactly the failure it exists to prevent.
    """
    problems = [
        f"{measurement.name}: phases account for {measurement.accounted_share * 100:.1f}% of "
        f"{measurement.total_seconds:.4f}s, below {MINIMUM_ACCOUNTED_SHARE * 100:.0f}%"
        for measurement in measurements
        if measurement.accounted_share < MINIMUM_ACCOUNTED_SHARE
    ]
    problems.extend(
        f"{measurement.name}: phases sum to more than the total ({measurement.accounted_share * 100:.1f}%)"
        for measurement in measurements
        if measurement.accounted_share > 1.0
    )
    # --- Assert ---
    assert not problems, report(harness, measurements, problems)


def test_phase_shares_stay_near_the_baseline(
    harness: ModuleType,
    baseline: dict[str, Any],
    measurements: list[Any],
) -> None:
    """A phase taking over its workflow is the leak this guard is for."""
    problems = []
    for measurement in measurements:
        entry = baseline["workflows"].get(measurement.name)
        if entry is None:
            continue
        shares = measurement.phase_shares
        for phase_name, baseline_share in entry["phase_shares"].items():
            if phase_name not in shares:
                continue
            if max(baseline_share, shares[phase_name]) < SHARE_FLOOR:
                continue
            drift = shares[phase_name] - baseline_share
            if abs(drift) > SHARE_MARGIN:
                problems.append(
                    f"{measurement.name} / {phase_name}: {shares[phase_name] * 100:.1f}% of the workflow, "
                    f"baseline {baseline_share * 100:.1f}%, drift {drift * 100:+.1f} points"
                )
    # --- Assert ---
    assert not problems, report(harness, measurements, problems)


def test_totals_stay_within_a_generous_multiple_of_the_baseline(
    harness: ModuleType,
    baseline: dict[str, Any],
    measurements: list[Any],
) -> None:
    """Only an order-of-magnitude blow-up should fail here; slower machines must not."""
    problems = []
    for measurement in measurements:
        entry = baseline["workflows"].get(measurement.name)
        if entry is None or entry["total_seconds"] <= 0.0:
            continue
        multiple = measurement.total_seconds / entry["total_seconds"]
        if multiple > TOTAL_MULTIPLE:
            problems.append(
                f"{measurement.name}: {measurement.total_seconds:.4f}s is {multiple:.1f}x the baseline "
                f"{entry['total_seconds']:.4f}s, above {TOTAL_MULTIPLE:.0f}x"
            )
    # --- Assert ---
    assert not problems, report(harness, measurements, problems)


def test_baseline_records_the_machine_it_came_from(baseline: dict[str, Any]) -> None:
    """Numbers without a machine cannot be argued with, so the baseline carries one."""
    assert baseline["size_profile"] == "guard"
    for key in ("platform", "machine", "python", "implementation"):
        assert baseline["machine"][key]
