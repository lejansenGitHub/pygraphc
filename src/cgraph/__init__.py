"""Fast connected-components via C union-find with path compression + union by rank."""

from collections.abc import Generator

from cgraph._core import connected_components_remapped as _cc_remapped
from cgraph._core import connected_components_with_branches_remapped as _cc_branches_remapped

__all__ = [
    "BranchId",
    "NodeId",
    "connected_components",
    "connected_components_with_branch_ids",
]

NodeId = int
BranchId = int


def connected_components(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """
    Yield each connected component as a set of original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_remapped(node_ids, edges)


def connected_components_with_branch_ids(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """
    Yield (node_id_set, branch_id_set) with original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_branches_remapped(node_ids, edges, branch_ids)
