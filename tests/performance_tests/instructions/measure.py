"""Entry point run inside callgrind: prepare each scenario, then call it through the marker library.

Writes the names it measured, in order, to ``PERFORMANCE_MANIFEST``. A
scenario the installed pygraphc cannot prepare (an API the build does not
have yet) is reported on stderr and left out, so a branch can add scenarios
that its merge base cannot run.
"""

import ctypes
import gc
import os
import sys
from pathlib import Path

from performance_tests.instructions.scenarios import SCENARIOS

CALLBACK = ctypes.CFUNCTYPE(None)


def main(scenario_names: list[str]) -> None:
    """Measure the named scenarios in order."""
    marker = ctypes.CDLL(os.environ["PERFORMANCE_MARKER_LIBRARY"])
    measured_names: list[str] = []
    for name in scenario_names:
        try:
            measured_call = SCENARIOS[name]()
        except (AttributeError, ImportError, TypeError) as error:
            print(f"not measured: {name}: {error!r}", file=sys.stderr)  # noqa: T201 — reported, not raised
            continue
        failures: list[BaseException] = []

        def run(measured_call: object = measured_call, failures: list[BaseException] = failures) -> None:
            try:
                measured_call()  # type: ignore[operator]  # every scenario returns a zero-argument callable
            except BaseException as error:
                failures.append(error)

        gc.collect()
        marker.run_measured(CALLBACK(run))
        if failures:
            raise failures[0]
        measured_names.append(name)
    Path(os.environ["PERFORMANCE_MANIFEST"]).write_text("\n".join(measured_names), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENARIOS))
