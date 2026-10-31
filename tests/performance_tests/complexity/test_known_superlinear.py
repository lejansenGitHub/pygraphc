"""The ratchet registry may only name cases that exist, or a fixed case would keep a loose limit forever."""

import pytest

from performance_tests.complexity import test_directed_scaling, test_reduction_scaling, test_undirected_scaling
from performance_tests.complexity.known_superlinear import KNOWN_SUPERLINEAR

pytestmark = pytest.mark.performance


def _all_cases() -> set[str]:
    undirected = {
        f"{operation}/{shape}"
        for operation in test_undirected_scaling.OPERATIONS
        for shape in test_undirected_scaling.SHAPES
    }
    parsing = {f"parse/{shape}" for shape in test_undirected_scaling.SHAPES}
    directed = {
        f"directed/{operation}/{shape}"
        for operations, shapes in (
            (test_directed_scaling.OPERATIONS, test_directed_scaling.SHAPES),
            (test_directed_scaling.ACYCLIC_OPERATIONS, test_directed_scaling.ACYCLIC_SHAPES),
        )
        for operation in operations
        for shape in shapes
    }
    reduction = {
        f"reduce_{engine}/{shape}"
        for engine in ("c", "moves", "rounds", "python")
        for shape in test_reduction_scaling.SHAPES
    }
    folds = {f"tree_folds/{shape}" for shape in ("path", "two_hubs")}
    partitions = {f"scenario_and_partition/{shape}" for shape in ("meshed_feeders", "lattice")}
    return undirected | parsing | directed | reduction | folds | partitions | {"all_edge_paths/path"}


def test_every_known_case_exists() -> None:
    """A registry entry for a renamed or removed case would silently stop applying."""
    assert set(KNOWN_SUPERLINEAR) <= _all_cases(), sorted(set(KNOWN_SUPERLINEAR) - _all_cases())
