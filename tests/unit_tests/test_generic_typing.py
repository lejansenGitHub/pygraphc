"""Branded integer ids round-trip through the kernel unchanged, at runtime and under mypy."""

import re
import subprocess
import sys
from pathlib import Path
from typing import NewType

from pygraphc import EdgeIndex, Graph, for_each_edge_excluded

NodeId = NewType("NodeId", int)
BranchId = NewType("BranchId", int)

# Values above the small-int cache, so identity checks are meaningful.
LARGE = 10**12

CHECK_FILE = Path(__file__).resolve().parent.parent / "typing" / "graph_generic_check.py"
REPO_ROOT = CHECK_FILE.parent.parent.parent


def _branded_graph() -> tuple[Graph[NodeId, BranchId], list[NodeId], list[BranchId]]:
    """Chain 1 -- 2 -- 3 plus an isolated node 4, every id a distinct large object."""
    node_ids = [NodeId(LARGE + offset) for offset in (1, 2, 3, 4)]
    edges = [(node_ids[0], node_ids[1]), (node_ids[1], node_ids[2])]
    branch_ids = [BranchId(LARGE + 10), BranchId(LARGE + 20)]
    return Graph[NodeId, BranchId](node_ids, edges, branch_ids=branch_ids), node_ids, branch_ids


def _same_objects(returned: set[int] | list[int], originals: list[int]) -> bool:
    """True when every returned id is one of the original id objects, not a copy."""
    original_ids = {id(original) for original in originals}
    return all(id(value) in original_ids for value in returned)


def test_connected_components_return_the_original_node_objects() -> None:
    """The C layer incref's the id objects it was handed, so a branded id comes back by identity."""
    graph, node_ids, _branch_ids = _branded_graph()

    components = sorted(graph.connected_components(), key=min)

    assert components == [{node_ids[0], node_ids[1], node_ids[2]}, {node_ids[3]}]
    assert all(_same_objects(component, node_ids) for component in components)


def test_bridges_with_branch_ids_return_the_original_objects() -> None:
    """Both endpoints of a bridge and its branch id are the caller's objects: every chain edge is a bridge."""
    graph, node_ids, branch_ids = _branded_graph()

    triples = graph.bridges_with_branch_ids()

    assert sorted(triples) == [
        (node_ids[0], node_ids[1], branch_ids[0]),
        (node_ids[1], node_ids[2], branch_ids[1]),
    ]
    assert all(_same_objects([u, v], node_ids) for u, v, _branch_id in triples)
    assert _same_objects([branch_id for _u, _v, branch_id in triples], branch_ids)


def test_bfs_returns_the_original_node_objects() -> None:
    """A traversal order is built from the input objects, so the chain comes back branded and node 4 stays out."""
    graph, node_ids, _branch_ids = _branded_graph()

    order = graph.bfs(node_ids[0])

    assert order == node_ids[:3]
    assert _same_objects(order, node_ids)


def test_without_branches_keeps_the_original_objects() -> None:
    """Excluding a branch by its branded id splits the chain; the surviving ids are still the caller's objects."""
    graph, node_ids, branch_ids = _branded_graph()

    view = graph.without_branches([branch_ids[1]])
    components = sorted(view.connected_components_with_branch_ids(), key=lambda pair: min(pair[0]))

    assert components == [
        ({node_ids[0], node_ids[1]}, {branch_ids[0]}),
        ({node_ids[2]}, set()),
        ({node_ids[3]}, set()),
    ]
    assert all(_same_objects(nodes, node_ids) for nodes, _branches in components)
    assert all(_same_objects(branches, branch_ids) for _nodes, branches in components)


def test_split_node_returns_the_original_and_the_new_node_object() -> None:
    """A rebuild reroutes edge 1 to the new branded node and gives it the branch id of the edge it replaces."""
    graph, node_ids, branch_ids = _branded_graph()
    new_node_id = NodeId(LARGE + 5)

    view = graph.split_node(node_ids[1], new_node_id, graph.edge_indices(node_ids[1], node_ids[2]))
    components = sorted(view.connected_components(), key=min)

    assert components == [{node_ids[0], node_ids[1]}, {node_ids[2], new_node_id}, {node_ids[3]}]
    assert all(_same_objects(component, [*node_ids, new_node_id]) for component in components)
    assert view.bridges_with_branch_ids() == [
        (node_ids[0], node_ids[1], branch_ids[0]),
        (new_node_id, node_ids[2], branch_ids[1]),
    ]


def test_edge_indices_are_positions_in_the_edge_list() -> None:
    """Edge indices are freshly built positions, not ids: edge 1 is the second pair, and node 2 carries both."""
    graph, node_ids, _branch_ids = _branded_graph()

    assert graph.edge_indices(node_ids[1], node_ids[2]) == [EdgeIndex(1)]
    assert graph.incident_edge_indices(node_ids[1]) == [EdgeIndex(0), EdgeIndex(1)]
    assert [edge_index for edge_index, _result in for_each_edge_excluded(graph, "bridges")] == [0, 1]


def test_free_functions_accept_tuples() -> None:
    """Id parameters are sequences, so an immutable tuple of branded ids builds the same graph as a list."""
    _graph, node_ids, branch_ids = _branded_graph()
    edges = ((node_ids[0], node_ids[1]), (node_ids[1], node_ids[2]))

    graph = Graph(tuple(node_ids), edges, branch_ids=tuple(branch_ids))

    assert sorted(graph.connected_components(), key=min) == [{*node_ids[:3]}, {node_ids[3]}]


# --- Static check ---


def _run_mypy(target: Path) -> subprocess.CompletedProcess[str]:
    """Type-check ``target`` the way a strict downstream consumer would.

    The repository config is bypassed on purpose: its ``tests.*`` override
    disables ``arg-type`` and ``unused-ignore``, which would silence exactly the
    negative cases the check file relies on.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=",
            "--strict",
            "--disallow-any-explicit",
            "--disallow-any-unimported",
            "--python-version=3.11",
            str(target),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_check_file_passes_strict_mypy() -> None:
    """Every branded annotation in the check file is satisfied without a cast, so a strict caller needs none."""
    result = _run_mypy(CHECK_FILE)

    assert result.returncode == 0, result.stdout + result.stderr


def test_check_file_negative_cases_are_errors(tmp_path: Path) -> None:
    """Stripping the ignore comments must surface exactly one error per negative case."""
    source = CHECK_FILE.read_text()
    ignore_comment = re.compile(r"  # type: ignore\[[a-z-]+\]  # [^\n]*")
    expected_error_count = len(ignore_comment.findall(source))
    poisoned = tmp_path / "graph_generic_poisoned.py"
    poisoned.write_text(ignore_comment.sub("", source))

    result = _run_mypy(poisoned)

    assert result.returncode != 0
    assert "[assignment]" in result.stdout
    assert "[arg-type]" in result.stdout
    assert result.stdout.count(" error: ") == expected_error_count, result.stdout
