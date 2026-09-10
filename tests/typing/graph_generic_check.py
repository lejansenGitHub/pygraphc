"""Static check that ``Graph`` and ``GraphView`` preserve the caller's branded id types.

Type-checked by ``tests/unit_tests/test_generic_typing.py`` under mypy strict with
``disallow_any_explicit``; never executed. Every assignment target is annotated
with the branded type the result must have. The lines carrying
``type: ignore[...]`` are the negative cases: ``warn_unused_ignores`` fails the
check as soon as one of them stops being an error.
"""

from typing import NewType

from pygraphc import (
    EdgeIndex,
    Graph,
    GraphView,
    bfs,
    bridges,
    connected_components,
    connected_components_with_branch_ids,
    for_each_edge_excluded,
)

NodeId = NewType("NodeId", int)
BranchId = NewType("BranchId", int)

node_ids: list[NodeId] = [NodeId(1), NodeId(2), NodeId(3)]
edges: list[tuple[NodeId, NodeId]] = [(NodeId(1), NodeId(2)), (NodeId(2), NodeId(3))]
branch_ids: list[BranchId] = [BranchId(10), BranchId(20)]

# --- Construction: explicit, inferred, and from tuples ---
graph: Graph[NodeId, BranchId] = Graph[NodeId, BranchId](node_ids, edges, branch_ids=branch_ids)
inferred: Graph[NodeId, BranchId] = Graph(node_ids, edges, branch_ids=branch_ids)
from_tuples: Graph[NodeId, BranchId] = Graph(tuple(node_ids), tuple(edges), branch_ids=tuple(branch_ids))
split_lists: Graph[NodeId, int] = Graph(node_ids, [NodeId(1), NodeId(2)], [NodeId(2), NodeId(3)])
without_branch_ids: Graph[NodeId, int] = Graph(node_ids, edges)
plain: Graph[int, int] = Graph([1, 2, 3], [(1, 2), (2, 3)])

# --- Graph results carry the brands ---
components: list[set[NodeId]] = list(graph.connected_components())
components_with_branches: list[tuple[set[NodeId], set[BranchId]]] = list(graph.connected_components_with_branch_ids())
bridge_pairs: list[tuple[NodeId, NodeId]] = graph.bridges()
bridge_triples: list[tuple[NodeId, NodeId, BranchId]] = graph.bridges_with_branch_ids()
articulation: set[NodeId] = graph.articulation_points()
blocks: list[set[NodeId]] = list(graph.biconnected_components())
cycles: list[list[NodeId]] = graph.cycle_basis()
order: list[NodeId] = graph.bfs(NodeId(1))
neighbors: set[NodeId] = graph.neighbors(NodeId(2))
path: list[NodeId] = graph.shortest_path([1.0, 1.0], NodeId(1), NodeId(3))
distances: dict[NodeId, float] = graph.shortest_path_lengths([1.0, 1.0], NodeId(1))
multi_distances: dict[NodeId, float] = graph.multi_source_shortest_path_lengths([1.0, 1.0], [NodeId(1)])
two_edge: list[set[NodeId]] = list(graph.two_edge_connected_components())
on_paths: set[NodeId] = graph.nodes_on_simple_paths(NodeId(1), [NodeId(3)])
edge_indices: list[EdgeIndex] = graph.edge_indices(NodeId(1), NodeId(2))
incident: list[EdgeIndex] = graph.incident_edge_indices(NodeId(2))
edge_paths: list[list[EdgeIndex]] = graph.all_edge_paths(NodeId(1), [NodeId(3)])
plain_components: list[set[int]] = list(plain.connected_components())

# --- Views keep both type parameters through every derivation ---
view: GraphView[NodeId, BranchId] = graph.without_branches([BranchId(20)])
view_components: list[tuple[set[NodeId], set[BranchId]]] = list(view.connected_components_with_branch_ids())
view_triples: list[tuple[NodeId, NodeId, BranchId]] = view.bridges_with_branch_ids()
view_indices: list[EdgeIndex] = view.incident_edge_indices(NodeId(2))
view_paths: list[list[EdgeIndex]] = view.all_edge_paths(NodeId(1), NodeId(3))
masked: GraphView[NodeId, BranchId] = graph.without_edges(edge_indices).without_nodes([NodeId(3)])
rebuilt: GraphView[NodeId, BranchId] = view.with_edges([(NodeId(3), NodeId(1))], [BranchId(30)])
split: GraphView[NodeId, BranchId] = graph.split_node(NodeId(2), NodeId(4), edge_indices)
split_again: GraphView[NodeId, BranchId] = split.split_node(
    NodeId(3), NodeId(5), split.incident_edge_indices(NodeId(3))
)
split_components: list[set[NodeId]] = list(split.connected_components())

for edge_index, _result in for_each_edge_excluded(graph, "connected_components"):
    excluded: EdgeIndex = edge_index

# --- Free functions are generic too ---
free_components: list[set[NodeId]] = list(connected_components(node_ids, edges))
free_split: list[set[NodeId]] = list(connected_components(node_ids, [NodeId(1)], [NodeId(2)]))
free_with_branches: list[tuple[set[NodeId], set[BranchId]]] = list(
    connected_components_with_branch_ids(node_ids, edges, branch_ids)
)
free_bridges: list[tuple[NodeId, NodeId]] = bridges(node_ids, edges)
free_order: list[NodeId] = bfs(node_ids, edges, NodeId(1))

# --- Negative cases: each line must stay an error ---
wrong_nodes: set[NodeId] = next(graph.connected_components_with_branch_ids())[1]  # type: ignore[assignment]  # set[BranchId] is not set[NodeId]
wrong_branches: list[BranchId] = graph.bfs(NodeId(1))  # type: ignore[assignment]  # list[NodeId] is not list[BranchId]
wrong_indices: list[BranchId] = graph.edge_indices(NodeId(1), NodeId(2))  # type: ignore[assignment]  # EdgeIndex is not BranchId
wrong_plain: list[NodeId] = plain.bfs(1)  # type: ignore[assignment]  # int results do not become NodeId
wrong_view: GraphView[int, int] = graph.without_edges(edge_indices)  # type: ignore[assignment]  # views keep the brands
graph.bfs(BranchId(10))  # type: ignore[arg-type]  # a branch id is not a node id
graph.without_branches([NodeId(1)])  # type: ignore[list-item]  # a node id is not a branch id
