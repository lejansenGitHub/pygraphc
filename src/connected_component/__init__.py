"""Fast connected-components via C union-find with path compression + union by rank."""

from collections.abc import Generator

from connected_component._core import (
    connected_components as _cc,
    connected_components_with_branches as _cc_branches,
)

type NodeId = int
type BranchId = int


def igp_connected_components(
    number_of_nodes: int,
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """Yield each connected component as a set of node IDs."""
    yield from _cc(number_of_nodes, edges)


def igp_connected_components_with_branch_ids(
    number_of_nodes: int,
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """Yield (node_set, branch_id_set) for each connected component."""
    yield from _cc_branches(number_of_nodes, edges, branch_ids)
