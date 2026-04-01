"""Fast connected-components via C union-find with path compression + union by rank."""

from collections.abc import Generator

from connected_component._core import (
    connected_components as _cc,
    connected_components_remapped as _cc_remapped,
    connected_components_with_branches as _cc_branches,
    connected_components_with_branches_remapped as _cc_branches_remapped,
    labels as _labels,
)

NodeId = int
BranchId = int


def igp_connected_components(
    number_of_nodes: int,
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """Yield each connected component as a set of node indices."""
    yield from _cc(number_of_nodes, edges)


def igp_connected_components_with_branch_ids(
    number_of_nodes: int,
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """Yield (node_index_set, branch_id_set) for each connected component."""
    yield from _cc_branches(number_of_nodes, edges, branch_ids)


def igp_connected_component_labels(
    number_of_nodes: int,
    edges: list[tuple[int, int]],
) -> list[int]:
    """Return component label for each node. Fastest output — no set construction."""
    return _labels(number_of_nodes, edges)


def igp_connected_components_remapped(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """Yield each connected component as a set of original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_remapped(node_ids, edges)


def igp_connected_components_with_branch_ids_remapped(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """Yield (node_id_set, branch_id_set) with original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_branches_remapped(node_ids, edges, branch_ids)
