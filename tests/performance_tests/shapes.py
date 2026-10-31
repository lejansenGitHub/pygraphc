"""Deterministic graph shapes for the performance tiers.

Each shape stresses one axis along which a superlinear implementation shows
up: a star the degree of one node, a path the depth of a traversal, singletons
the number of components, a lattice the number of cycles. Every generator
returns ``(node_ids, edges)`` and is seeded, so equal arguments give equal graphs.
"""

import random

Edges = list[tuple[int, int]]


def path(node_count: int) -> tuple[list[int], Edges]:
    """One chain ``0 - 1 - ... - n-1``: maximal depth, every edge a bridge."""
    return list(range(node_count)), [(node_id, node_id + 1) for node_id in range(node_count - 1)]


def star(node_count: int) -> tuple[list[int], Edges]:
    """Node 0 joined to every other node: one node of degree n-1."""
    return list(range(node_count)), [(0, node_id) for node_id in range(1, node_count)]


def singletons(node_count: int) -> tuple[list[int], Edges]:
    """n isolated nodes with one self-loop each, so the graph has edges: n components."""
    return list(range(node_count)), [(node_id, node_id) for node_id in range(node_count)]


def lattice(node_count: int) -> tuple[list[int], Edges]:
    """Square grid of about n nodes: many short cycles, no bridges inside."""
    width = max(2, round(node_count**0.5))
    edges: Edges = []
    for row in range(width):
        for column in range(width):
            node_id = row * width + column
            if column + 1 < width:
                edges.append((node_id, node_id + 1))
            if row + 1 < width:
                edges.append((node_id, node_id + width))
    return list(range(width * width)), edges


def sparse_random(node_count: int, edges_per_node: float = 1.5, seed: int = 0) -> tuple[list[int], Edges]:
    """A spanning path plus random chords: one giant component, low average degree."""
    rng = random.Random(seed)
    node_ids, edges = path(node_count)
    chord_count = int(node_count * (edges_per_node - 1))
    edges += [(rng.randrange(node_count), rng.randrange(node_count)) for _ in range(chord_count)]
    return node_ids, edges


def meshed_feeders(node_count: int, seed: int = 0) -> tuple[list[int], Edges]:
    """Radial feeders from node 0 with one meshing line per 20 nodes: the shape of a distribution grid."""
    rng = random.Random(seed)
    edges = [(rng.randrange(max(0, node_id - 50), node_id), node_id) for node_id in range(1, node_count)]
    edges += [(rng.randrange(node_count), rng.randrange(node_count)) for _ in range(node_count // 20)]
    return list(range(node_count)), edges


def directed_acyclic(node_count: int, edges_per_node: int = 3, seed: int = 0) -> tuple[list[int], Edges]:
    """Edges only from lower to higher id: a DAG with a spanning path."""
    rng = random.Random(seed)
    edges = [(node_id, node_id + 1) for node_id in range(node_count - 1)]
    for _ in range(node_count * (edges_per_node - 1)):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second:
            edges.append((min(first, second), max(first, second)))
    return list(range(node_count)), edges


PortEdges = list[tuple[int, ...]]


def wide_edge(node_count: int) -> tuple[list[int], PortEdges]:
    """One edge with a port at every node: the number of ports of a single edge grows."""
    return list(range(node_count)), [tuple(range(node_count))]


def port_chain(node_count: int) -> tuple[list[int], PortEdges]:
    """Three-port edges (2k, 2k + 1, 2k + 2) chained through their last port: depth in multiport edges."""
    return list(range(node_count)), [(node_id, node_id + 1, node_id + 2) for node_id in range(0, node_count - 2, 2)]


def random_ports(node_count: int, seed: int = 0) -> tuple[list[int], PortEdges]:
    """A spanning path of lines plus random edges with two to four ports."""
    rng = random.Random(seed)
    node_ids, edges = path(node_count)
    port_edges: PortEdges = list(edges)
    port_edges += [tuple(rng.randrange(node_count) for _ in range(rng.randint(2, 4))) for _ in range(node_count // 2)]
    return node_ids, port_edges
