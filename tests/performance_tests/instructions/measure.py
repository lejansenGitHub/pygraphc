"""Entry point run inside callgrind: prepare each scenario, then call it through the marker library.

Writes the name of every dump, in order, to ``PERFORMANCE_MANIFEST``. A
scenario the installed pygraphc cannot prepare (an API the build does not
have yet) is reported on stderr and left out, so a branch can add scenarios
that its merge base cannot run. A ``workflow/<name>`` entry runs that workflow
of the harness and records one dump per phase.
"""

import ctypes
import gc
import os
import sys
from pathlib import Path

from performance_tests.instructions.scenarios import SCENARIOS
from performance_tests.instructions.workflows import PREFIX, run_workflow

CALLBACK = ctypes.CFUNCTYPE(None)


def _measure_scenario(name: str, marker: ctypes.CDLL, record: list[str]) -> None:
    """Prepare one scenario outside the measurement and run it inside."""
    try:
        measured_call = SCENARIOS[name]()
    except (AttributeError, ImportError, TypeError) as error:
        print(f"not measured: {name}: {error!r}", file=sys.stderr)  # noqa: T201 — reported, not raised
        return
    failures: list[BaseException] = []

    def run() -> None:
        try:
            measured_call()
        except BaseException as error:
            failures.append(error)

    gc.collect()
    marker.run_measured(CALLBACK(run))
    if failures:
        raise failures[0]
    record.append(name)


def main(scenario_names: list[str]) -> None:
    """Measure the named scenarios and workflows in order."""
    marker = ctypes.CDLL(os.environ["PERFORMANCE_MARKER_LIBRARY"])
    record: list[str] = []
    for name in scenario_names:
        if name.startswith(PREFIX):
            gc.collect()
            try:
                run_workflow(name, marker, record.append)
            except (AttributeError, ImportError, KeyError, TypeError) as error:
                print(f"not measured: {name}: {error!r}", file=sys.stderr)  # noqa: T201 — reported, not raised
        else:
            _measure_scenario(name, marker, record)
    Path(os.environ["PERFORMANCE_MANIFEST"]).write_text("\n".join(record), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS))
