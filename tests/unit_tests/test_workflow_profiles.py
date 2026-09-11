"""Guard on the workflow phase profiles in ``benchmarks/profile_workflows.py``.

Deliberately coarse. This is not a benchmark: it never asserts that anything is
fast, only that the *shape* of a workflow has not changed — that no phase has
taken over its workflow, that no phase has become an order of magnitude slower
than it was, and that the instrumentation still accounts for the time it
measures. A guard that cries wolf gets deleted by the next person, so every
margin below is wide enough that machine-to-machine variation and CI noise
cannot reach it, and each one says what it can and cannot catch.

**Shares are taken over the library time, not the workflow total.** Each
workflow generates its own random inputs, and at the guard sizes that
generation is 14% of one workflow and between a third and nine tenths of the
other eight — 87% of ``connected_components`` at 5,000 nodes and still 79% at
200,000, so it does not scale away and raising the guard sizes would not change
a single share. Shares of the total would therefore be mostly statements about
``random``; the harness reports shares of the total without the generation
phases, and this file compares those.

**What the share test can catch.** A phase that grows takes share from every
other phase, so the multiple needed to move it by ``SHARE_MARGIN`` depends on
where it started. Solving for it, a phase at library share ``s`` fires when it
becomes ``k`` times slower:

    s      0.50   0.30   0.10   0.05   0.02   0.01
    k       2.4    2.4    3.9    6.4   14.0   26.4

(the exact thresholds are 2.34, 2.34, 3.86, 6.34, 13.8 and 26.3; the row above
is rounded up to a multiple that fires)

and above a share of 0.80 it can never fire by growing, because there are not
20 points left to gain. Sharpest in the middle, blunt at both ends, which is
why the absolute per-phase test below exists. Three of the four C kernels
measured here are a single-digit percentage of their workflow's library time
whatever the graph size — the reduction loop 1.5%, the degree kernel 3.6%, the
bridge kernel 4.9% — so the share test would need a sixfold to an eighteenfold
regression in them before saying anything.

**What the absolute test can catch.** ``PHASE_MULTIPLE`` on the phase itself,
in any library phase above ``PHASE_FLOOR_SECONDS`` whatever its share, and that
is the gate that covers the kernels. An injected tenfold regression in the
label, degree and bridge kernels fails it every time; in the C reduction loop,
a phase of about 100 microseconds, ten times fails about two runs in three and
twelve times fails every run, because both sides of the comparison are minima
and a minimum of a slowed phase drifts down towards the gate. The guard sizes
of ``graph_build``, ``framework_pipeline``, ``reduction_log`` and
``structural_queries`` are chosen to keep their kernels above the floor; the
phases still below it are a handful of microseconds of Python bookkeeping and
stay unguarded, namely ``path_enumeration``'s graph
construction and release, ``dag_structure_learning``'s CPD estimate and
release, and ``reduction_log``'s reduction-graph build and log scan.

**No gate fails on a single run.** A workflow that trips is measured again and
only what survives the fastest of ``CONFIRMATION_ROUNDS`` fresh runs is
reported. Absolute times are the obvious reason — one phase of one round was
measured at eleven times its baseline on a machine oversubscribed by a third —
but shares need it just as much, for a subtler reason: the harness reports the
phases of the round with the best *total*, so one stall inside one phase of
that round skews every share in that workflow. Without confirmation the share
test failed ten runs in twenty under that load; with it, none in twenty.

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

# A phase's share of its workflow's library time may move by this many percentage
# points. Shares are ratios, so a machine that is uniformly slower does not move
# them; what moves them is a phase that scales differently from the rest (memory
# bandwidth against branch prediction, a different allocator, a GC pause landing
# in one phase) and, measured rather than assumed, a single preempted round —
# see the confirmation pass on the share test. 20 points still catches a phase
# that doubles out of half the library time, which is the leak this guard is
# for; the table in the module docstring says what it costs at other shares.
SHARE_MARGIN = 0.20

# A phase is compared only when it is at least this share of its workflow's
# library time either at baseline or now — the "or now" matters, otherwise a
# phase growing out of nothing, which is what a new leak looks like, would be
# exempt.
SHARE_FLOOR = 0.10

# A workflow's library time, and each phase of it, may be this multiple of the
# baseline. The baseline comes from one developer machine; a shared-vCPU CI
# runner without turbo is routinely three to four times slower on single-threaded
# Python, and the per-phase spread measured over seven runs of this harness on
# one quiet machine reaches 1.4x on the smallest phases. Eight times leaves room
# for both and still fails on an order-of-magnitude regression, which is the only
# absolute-time claim worth making from a machine nobody controls.
LIBRARY_MULTIPLE = 8.0
PHASE_MULTIPLE = 8.0

# How many fresh runs a workflow gets before an absolute gate is allowed to
# fail, with the per-phase fastest of them deciding. Only ever paid on the
# failure path, where it costs a few hundred milliseconds.
CONFIRMATION_ROUNDS = 3

# A phase is compared in absolute time only above this duration. Below it, timer
# granularity, allocator luck and a single GC pause are all larger than the
# signal: the smallest phases here spread by 1.4x between repeated runs on a
# quiet machine, against 1.03x for the largest.
PHASE_FLOOR_SECONDS = 50e-6

# The phases of a workflow must account for at least this share of the wall time
# measured around the whole body. The harness attributes even the teardown of a
# workflow's own inputs to a phase, so the baseline reports 98.44% at worst
# (`structural_queries`) and 99.4% or better on six of the nine; the remainder
# is the result string and the frame teardown of a few small locals. Observed
# over fifteen runs the worst goes to 97.9%, so this floor has about three
# points of room and only notices unphased work above roughly that — enough to
# catch a whole step left uninstrumented, which is the failure that would make
# the table a lie about where the time goes, and not enough to notice a stray
# line or two.
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


def share_guard_fires(baseline_share: float, multiple: float) -> bool:
    """Would the share test fire if a phase at ``baseline_share`` became ``multiple`` slower?

    The arithmetic of the sensitivity table in the module docstring, so that the
    table is a checked claim rather than a comment.
    """
    grown = baseline_share * multiple
    new_share = grown / (1.0 - baseline_share + grown)
    if max(baseline_share, new_share) < SHARE_FLOOR:
        return False
    return abs(new_share - baseline_share) > SHARE_MARGIN


def library_shares_of(setup_phases: set[str], phase_seconds: dict[str, float]) -> dict[str, float]:
    """Each library phase's share of the library time, from one set of phase timings."""
    library = {name: seconds for name, seconds in phase_seconds.items() if name not in setup_phases}
    total = sum(library.values())
    if total <= 0.0:
        return dict.fromkeys(library, 0.0)
    return {name: seconds / total for name, seconds in library.items()}


def share_problems(
    baseline: dict[str, Any],
    seconds_by_workflow: dict[str, dict[str, float]],
) -> list[tuple[str, str]]:
    """Phases whose share of the library time has drifted past ``SHARE_MARGIN``."""
    problems = []
    for name, phase_seconds in seconds_by_workflow.items():
        entry = baseline["workflows"].get(name)
        if entry is None:
            continue
        shares = library_shares_of(set(entry["setup_phases"]), phase_seconds)
        for phase_name, baseline_share in entry["library_shares"].items():
            if phase_name not in shares:
                continue
            if max(baseline_share, shares[phase_name]) < SHARE_FLOOR:
                continue
            drift = shares[phase_name] - baseline_share
            if abs(drift) > SHARE_MARGIN:
                problems.append((
                    name,
                    (
                        f"{name} / {phase_name}: {shares[phase_name] * 100:.1f}% of the library time, "
                        f"baseline {baseline_share * 100:.1f}%, drift {drift * 100:+.1f} points"
                    ),
                ))
    return problems


def library_time_problems(
    baseline: dict[str, Any],
    seconds_by_workflow: dict[str, dict[str, float]],
) -> list[tuple[str, str]]:
    """Workflows whose library time is more than ``LIBRARY_MULTIPLE`` of the baseline."""
    problems = []
    for name, phase_seconds in seconds_by_workflow.items():
        entry = baseline["workflows"].get(name)
        if entry is None or entry["library_seconds"] <= 0.0:
            continue
        setup = set(entry["setup_phases"])
        library_seconds = sum(seconds for phase, seconds in phase_seconds.items() if phase not in setup)
        multiple = library_seconds / entry["library_seconds"]
        if multiple > LIBRARY_MULTIPLE:
            problems.append((
                name,
                (
                    f"{name}: {library_seconds:.4f}s of library time is {multiple:.1f}x "
                    f"the baseline {entry['library_seconds']:.4f}s, above {LIBRARY_MULTIPLE:.0f}x"
                ),
            ))
    return problems


def phase_time_problems(
    baseline: dict[str, Any],
    seconds_by_workflow: dict[str, dict[str, float]],
) -> list[tuple[str, str]]:
    """Phases above the floor that are more than ``PHASE_MULTIPLE`` of the baseline."""
    problems = []
    for name, phase_seconds in seconds_by_workflow.items():
        entry = baseline["workflows"].get(name)
        if entry is None:
            continue
        setup = set(entry["setup_phases"])
        for phase_name, seconds in phase_seconds.items():
            baseline_seconds = entry["phase_seconds"].get(phase_name, 0.0)
            if phase_name in setup or baseline_seconds < PHASE_FLOOR_SECONDS:
                continue
            multiple = seconds / baseline_seconds
            if multiple > PHASE_MULTIPLE:
                problems.append((
                    name,
                    (
                        f"{name} / {phase_name}: {seconds * 1e6:.0f}us is {multiple:.1f}x the baseline "
                        f"{baseline_seconds * 1e6:.0f}us, above {PHASE_MULTIPLE:.0f}x"
                    ),
                ))
    return problems


def phase_seconds_of(measurements: list[Any]) -> dict[str, dict[str, float]]:
    """The phases of one harness run, by workflow."""
    return {measurement.name: dict(measurement.phase_seconds) for measurement in measurements}


def fastest_phase_seconds(harness: ModuleType, names: set[str]) -> dict[str, dict[str, float]]:
    """The per-phase fastest of ``CONFIRMATION_ROUNDS`` fresh runs of the named workflows.

    The confirmation pass for the two absolute gates. One round of one phase
    can lose a scheduler slice to a noisy neighbour — measured at eleven times
    its baseline on a machine deliberately oversubscribed by a third — and that
    is noise, not a regression. A regression is in every round, so it survives
    the fastest of several and noise does not.
    """
    fastest: dict[str, dict[str, float]] = {}
    for workflow in harness.WORKFLOWS:
        if workflow.name not in names:
            continue
        for _round in range(CONFIRMATION_ROUNDS):
            measurement = harness.measure(workflow, workflow.guard_size, with_peak=False, with_networkx=False)
            phases = fastest.setdefault(workflow.name, dict(measurement.phase_seconds))
            for phase_name, seconds in measurement.phase_seconds.items():
                phases[phase_name] = min(phases[phase_name], seconds)
    return fastest


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
        if sorted(measurement.setup_phases) != entry["setup_phases"]:
            problems.append(
                f"{measurement.name}: setup phases are {sorted(measurement.setup_phases)}, "
                f"baseline has {entry['setup_phases']} — every share below is relative to the rest"
            )
        if measurement.size != entry["size"]:
            problems.append(f"{measurement.name}: guard size {measurement.size}, baseline {entry['size']}")
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
    """A phase taking over its workflow is the leak this guard is for.

    Confirmed before it fails, like the absolute gates and for a reason that
    took a loaded machine to find: the harness reports the phases of the round
    with the best *total*, so one stall inside one phase of that round skews
    every share in the workflow. Under a third of oversubscription that failed
    ten runs in twenty; re-measuring and taking each phase's own fastest round
    leaves one.
    """
    suspected = share_problems(baseline, phase_seconds_of(measurements))
    if not suspected:
        return
    confirmed = share_problems(baseline, fastest_phase_seconds(harness, {name for name, _ in suspected}))
    # --- Assert ---
    assert not confirmed, report(harness, measurements, [message for _, message in confirmed])


def test_library_time_stays_within_a_generous_multiple_of_the_baseline(
    harness: ModuleType,
    baseline: dict[str, Any],
    measurements: list[Any],
) -> None:
    """Only an order-of-magnitude blow-up should fail here; slower machines must not.

    On the library time rather than the workflow total, because input generation
    is most of the total on half these workflows and would absorb the regression.
    """
    suspected = library_time_problems(baseline, phase_seconds_of(measurements))
    if not suspected:
        return
    confirmed = library_time_problems(baseline, fastest_phase_seconds(harness, {name for name, _ in suspected}))
    # --- Assert ---
    assert not confirmed, report(harness, measurements, [message for _, message in confirmed])


def test_no_phase_above_the_floor_is_an_order_of_magnitude_slower(
    harness: ModuleType,
    baseline: dict[str, Any],
    measurements: list[Any],
) -> None:
    """The test that covers the kernels, which are too small a share for the share test."""
    suspected = phase_time_problems(baseline, phase_seconds_of(measurements))
    if not suspected:
        return
    confirmed = phase_time_problems(baseline, fastest_phase_seconds(harness, {name for name, _ in suspected}))
    # --- Assert ---
    assert not confirmed, report(harness, measurements, [message for _, message in confirmed])


def test_the_share_guard_fires_at_the_multiples_the_docstring_claims() -> None:
    """The sensitivity table is arithmetic, so it is asserted instead of trusted."""
    fires_at = {0.50: 2.4, 0.30: 2.4, 0.10: 3.9, 0.05: 6.4, 0.02: 14.0, 0.01: 26.4}
    problems = []
    for baseline_share, multiple in fires_at.items():
        if not share_guard_fires(baseline_share, multiple):
            problems.append(f"share {baseline_share}: {multiple}x should fire and does not")
        if share_guard_fires(baseline_share, multiple * 0.8):
            problems.append(f"share {baseline_share}: {multiple * 0.8:.1f}x should not fire and does")
    if share_guard_fires(0.85, 1000.0):
        problems.append("a phase at 0.85 cannot gain 20 points, so growth must never fire")
    # --- Assert ---
    assert not problems, problems


def test_baseline_records_where_and_when_it_came_from(baseline: dict[str, Any]) -> None:
    """Numbers without a machine and a date cannot be argued with, so the baseline carries both."""
    assert baseline["size_profile"] == "guard"
    assert baseline["taken_at"]
    assert baseline["revision"]
    for key in ("platform", "machine", "python", "implementation"):
        assert baseline["machine"][key]
