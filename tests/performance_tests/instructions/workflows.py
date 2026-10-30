"""The workflows of ``benchmarks/profile_workflows.py`` as instruction-count scenarios, one count per phase.

The harness hands every workflow a ``PhaseTimer`` and the workflow marks its
phases with ``timer.phase(name)``. ``InstructionTimer`` keeps that interface
and switches callgrind's collection on and off around each phase instead of
reading a clock, so every phase of every workflow is compared against the
merge base like any other scenario, named ``workflow/<workflow>/<phase>``.
"""

import ctypes
import importlib.util
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

HARNESS_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "profile_workflows.py"
PREFIX = "workflow/"


def harness() -> ModuleType:
    """The workflow harness, loaded from the repository's ``benchmarks`` directory."""
    specification = importlib.util.spec_from_file_location("profile_workflows", HARNESS_PATH)
    if specification is None or specification.loader is None:
        message = f"cannot load the workflow harness from {HARNESS_PATH}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def workflow_scenario_names() -> list[str]:
    """One ``workflow/<name>`` entry per workflow of the harness."""
    return [f"{PREFIX}{workflow.name}" for workflow in harness().WORKFLOWS]


def run_workflow(name: str, marker: ctypes.CDLL, record: Callable[[str], None]) -> None:
    """Run one workflow at its guard size with every phase measured; ``record`` gets each dump's name in order."""
    module = harness()
    workflow = module.WORKFLOWS_BY_NAME[name.removeprefix(PREFIX)]

    class InstructionTimer(module.PhaseTimer):  # type: ignore[misc,name-defined]  # loaded at runtime
        """A phase timer that counts instructions instead of seconds."""

        @contextmanager
        def phase(self, phase_name: str, *, setup: bool = False) -> Generator[None, None, None]:
            if setup:
                self.setup_phases.add(phase_name)
            marker.start_measure()
            try:
                yield
            finally:
                marker.stop_measure()
                record(f"{name}/{phase_name}")

    size = workflow.guard_size
    workflow.body(module.PhaseTimer(), size)
    workflow.body(InstructionTimer(), size)
