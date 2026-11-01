"""Type stub of the ``_dag_learn`` C extension, read from its ``PyArg_ParseTuple`` formats."""

from collections.abc import Sequence

def hill_climb_k2(
    data: Sequence[Sequence[int]],
    cardinalities: Sequence[int],
    max_indegree: int = 1,
    tabu_length: int = 100,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
    /,
) -> list[tuple[int, int]]: ...
def k2_local_score(
    data: Sequence[Sequence[int]], cardinalities: Sequence[int], child: int, parents: Sequence[int], /
) -> float: ...
def estimate_cpds(
    data: Sequence[Sequence[int]], cardinalities: Sequence[int], edges: Sequence[tuple[int, int]], /
) -> dict[int, list[list[float]]]: ...
