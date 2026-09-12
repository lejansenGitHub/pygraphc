"""Performance tests for the terminal-preserving reduction.

A sparse random graph against a straightforward networkx implementation of
the same three moves, the hub-heavy case that a per-move scan of the smaller
incidence set made quadratic, and the partition through the C label kernel
against the earlier path that materialised one Python set per block.
"""

import random
import time
import timeit
import tracemalloc
from bisect import bisect_left
from collections.abc import Callable

import networkx
import pytest

from pygraphc import ReductionState
from pygraphc.reduction import MultiGraph, Partition, leaves, reduce

pytestmark = pytest.mark.performance


def sparse_multigraph(node_count: int, edge_count: int, seed: int) -> MultiGraph[int]:
    rng = random.Random(seed)
    nodes = list(range(node_count))
    endpoints = {edge_id: (rng.randrange(node_count), rng.randrange(node_count)) for edge_id in range(edge_count)}
    return MultiGraph(nodes, endpoints)


def set_based_partition(graph: MultiGraph[int], active: set[int]) -> Partition:
    """The partition path before the label kernel: one Python set per block from ``connected_components``."""
    kernel = graph._kernel
    excluded = [index for index, edge_id in enumerate(kernel.edge_ids) if edge_id not in active]
    block_of: dict[int, int] = {}
    for group in kernel.graph.without_edges(excluded).connected_components():
        representative = min(group)
        for node_id in group:
            block_of[node_id] = representative
    return Partition(block_of)


def time_and_peak(function: Callable[[], Partition]) -> tuple[float, int]:
    """Wall time of one call and the tracemalloc peak of a second call, in bytes."""
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    tracemalloc.start()
    function()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


@pytest.mark.parametrize("node_count", [100_000, 1_000_000])
def test_label_partition_uses_less_memory_than_the_set_partition(node_count: int) -> None:
    """The label kernel returns 4 bytes per node; the set path allocated one
    Python int reference per node inside per-block sets on top of the dict."""
    graph = sparse_multigraph(node_count, edge_count=node_count * 5 // 4, seed=42)
    active = set(graph.endpoints)
    graph._kernel  # noqa: B018 — build the cached C graph outside the measurement

    set_elapsed, set_peak = time_and_peak(lambda: set_based_partition(graph, active))
    label_elapsed, label_peak = time_and_peak(lambda: Partition.from_components(graph, active))

    print(  # noqa: T201 — benchmark output is intentional
        f"\n  partition of {node_count:,} nodes / {len(graph.endpoints):,} edges: "
        f"sets {set_elapsed:.3f}s peak {set_peak / 2**20:.1f} MiB, "
        f"labels {label_elapsed:.3f}s peak {label_peak / 2**20:.1f} MiB"
    )
    assert Partition.from_components(graph, active) == set_based_partition(graph, active)
    assert label_peak < set_peak, f"labels peak {label_peak} not below sets peak {set_peak}"


def networkx_reduce(graph: MultiGraph[int], terminals: set[int]) -> set[int]:
    """The three moves as a networkx caller writes them: full sweeps to a fixpoint.

    Self-loops take part in no move and are dropped first, terminal-free
    components next. Degree-1 non-terminals are removed, degree-2
    non-terminals with two distinct neighbours are contracted into one of
    them, and parallel edges collapse to one; the sweep repeats until nothing
    changes. Returns the surviving node set.
    """
    nx_graph = networkx.MultiGraph()
    nx_graph.add_nodes_from(graph.nodes)
    for edge_id, (from_node, to_node) in graph.endpoints.items():
        if from_node != to_node:
            nx_graph.add_edge(from_node, to_node, key=edge_id)
    for component in list(networkx.connected_components(nx_graph)):
        if not component & terminals:
            nx_graph.remove_nodes_from(component)
    changed = True
    while changed:
        changed = False
        for node_id in list(nx_graph.nodes):
            if node_id in terminals or node_id not in nx_graph:
                continue
            degree = nx_graph.degree(node_id)
            if degree == 1:
                nx_graph.remove_node(node_id)
                changed = True
            elif degree == 2:
                neighbours = list(nx_graph.neighbors(node_id))
                if len(neighbours) == 2:
                    networkx.contracted_nodes(nx_graph, neighbours[0], node_id, self_loops=False, copy=False)
                    changed = True
        for from_node, to_node in {(min(pair), max(pair)) for pair in nx_graph.edges(keys=False)}:
            keys = list(nx_graph[from_node][to_node])
            for key in keys[1:]:
                nx_graph.remove_edge(from_node, to_node, key=key)
                changed = True
    return set(nx_graph.nodes)


def test_the_c_engine_beats_networkx_by_four_and_the_python_engine_by_two() -> None:
    """Both engines against the networkx sweeps on the same graph.

    The C engine runs the structural loop in the C tier and folds the
    operation log in Python; the Python engine runs the worklist. The margin
    over the Python engine is bounded by the fold, which builds the same
    provenance trees either way and is over nine tenths of the C engine's
    time, so the two engines share most of their work.

    The floors are half the measured margins (6.7x networkx, 3.9x the Python
    engine) so that a loaded runner cannot trip them; the printed ratios, not
    the floors, are the numbers to read.
    """
    node_count = 20_000
    graph = sparse_multigraph(node_count, edge_count=25_000, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, node_count // 100))
    graph._kernel  # noqa: B018 — build the cached C graph outside the measurement

    c_elapsed = min(timeit.repeat(lambda: reduce(graph, terminals, engine="c"), number=1, repeat=9))
    python_elapsed = min(timeit.repeat(lambda: reduce(graph, terminals, engine="python"), number=1, repeat=9))
    networkx_elapsed = min(timeit.repeat(lambda: networkx_reduce(graph, terminals), number=1, repeat=9))
    reduced = reduce(graph, terminals)
    networkx_nodes = networkx_reduce(graph, terminals)

    print(  # noqa: T201 — benchmark output is intentional
        f"\n  reduce 20k nodes / 25k edges / 200 terminals (best of 9): c {c_elapsed:.3f}s, "
        f"python {python_elapsed:.3f}s, networkx sweeps {networkx_elapsed:.3f}s "
        f"-> {len(reduced.graph.nodes)} nodes, {len(reduced.graph.endpoints)} edges; "
        f"networkx/c {networkx_elapsed / c_elapsed:.2f}, python/c {python_elapsed / c_elapsed:.2f}"
    )
    assert terminals <= set(reduced.graph.nodes)
    assert set(reduced.graph.nodes) == networkx_nodes
    assert c_elapsed * 4 <= networkx_elapsed, (
        f"c {c_elapsed:.3f}s only {networkx_elapsed / c_elapsed:.2f}x networkx {networkx_elapsed:.3f}s"
    )
    assert c_elapsed * 2 <= python_elapsed, (
        f"c {c_elapsed:.3f}s only {python_elapsed / c_elapsed:.2f}x python {python_elapsed:.3f}s"
    )


def node_masks(graph: MultiGraph[int], terminals: set[int]) -> tuple[bytes, bytes]:
    """The terminal set as one byte per node index, and an empty protected mask."""
    nodes = graph._kernel.nodes
    terminal_mask = bytearray(len(nodes))
    for node_id in terminals:
        terminal_mask[bisect_left(nodes, node_id)] = 1
    return bytes(terminal_mask), bytes(len(nodes))


def _structural_and_rounds(graph: MultiGraph[int], terminals: set[int], *, batched: bool) -> tuple[float, int]:
    """Wall time of one Python-orchestrated loop without the fold, and the rounds a batched loop takes.

    The very loop ``reduce`` runs for the ``"moves"`` and ``"rounds"`` engines,
    stopped at the operation log, so the difference to the whole call is the
    tree building and the difference to the monolith's C loop is what the
    Python boundary adds.
    """
    kernel = graph._kernel
    terminal_mask, protected_mask = node_masks(graph, terminals)
    rounds = 0
    start = time.perf_counter()
    with kernel.graph.series_parallel_state(terminal_mask, protected_mask) as state:
        if batched:
            progressed = True
            while progressed:
                progressed = False
                rounds += 1
                for kind in (ReductionState.PENDANT, ReductionState.SERIES, ReductionState.PARALLEL):
                    batch = state.batch_moves(kind)
                    if batch is not None:
                        state.apply_batch(kind, batch)
                        progressed = True
        else:
            first = state.batch_moves(ReductionState.PARALLEL)
            if first is not None:
                state.apply_batch(ReductionState.PARALLEL, first)
            move = state.next_move()
            while move is not None:
                _created, parallel_pending = state.apply_move(move)
                if parallel_pending:
                    members = state.pair_edges(move[4], move[5])
                    if members is not None:
                        state.apply_parallel(move[4], move[5], members)
                move = state.next_move()
        state.log()
    return time.perf_counter() - start, rounds


def _monolith_loop(graph: MultiGraph[int], terminals: set[int]) -> float:
    """Wall time of the whole structural loop in C, the reference for a boundary-free run."""
    kernel = graph._kernel
    terminal_mask, protected_mask = node_masks(graph, terminals)
    start = time.perf_counter()
    kernel.graph.series_parallel_reduce(terminal_mask, protected_mask)
    return time.perf_counter() - start


@pytest.mark.parametrize(("node_count", "edge_count"), [(20_000, 25_000), (100_000, 125_000), (1_000_000, 1_250_000)])
def test_all_four_engines_reduce_the_sparse_graph(node_count: int, edge_count: int) -> None:
    """The table a human compares the engines by: no assertion here rates one against another.

    ``"c"`` runs the whole fixpoint in C, ``"moves"`` and ``"rounds"`` run it
    in Python over C primitives on one reduction state, ``"python"`` runs the
    worklist. The printed shares split each Python-orchestrated engine into the
    structural work the monolith also does, what the boundary adds on top and
    the tree building that follows.
    """
    graph = sparse_multigraph(node_count, edge_count, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, node_count // 100))
    graph._kernel  # noqa: B018 — build the cached C graph outside the measurement
    engines = ("c", "moves", "rounds", "python")
    for engine in engines:
        reduce(graph, terminals, engine=engine)

    elapsed = {
        engine: min(timeit.repeat(lambda engine=engine: reduce(graph, terminals, engine=engine), number=1, repeat=5))
        for engine in engines
    }
    loop_in_c = min(_monolith_loop(graph, terminals) for _ in range(5))
    structural = {
        "moves": min(_structural_and_rounds(graph, terminals, batched=False) for _ in range(5)),
        "rounds": min(_structural_and_rounds(graph, terminals, batched=True) for _ in range(5)),
    }
    log = graph._kernel.graph.series_parallel_reduce(*node_masks(graph, terminals))
    moves = len(log.op_kind) - log.op_kind.tolist().count(0)
    residual = reduce(graph, terminals, engine="c")

    report = ", ".join(f"{engine} {elapsed[engine]:.4f}s" for engine in engines)
    shares = "; ".join(
        f"{name} C {loop_in_c / elapsed[name]:.0%} / boundary {max(seconds - loop_in_c, 0.0) / elapsed[name]:.0%}"
        f" / trees {max(elapsed[name] - seconds, 0.0) / elapsed[name]:.0%}"
        for name, (seconds, _rounds) in structural.items()
    )
    print(  # noqa: T201 — benchmark output is intentional
        f"\n  reduce {node_count:,} nodes / {edge_count:,} edges / {len(terminals):,} terminals (best of 5): "
        f"{report}\n    {moves:,} moves, {structural['rounds'][1]} rounds, "
        f"monolith C loop {loop_in_c:.4f}s -> {len(residual.graph.nodes):,} nodes, "
        f"{len(residual.graph.endpoints):,} edges\n    {shares}"
    )
    assert terminals <= set(residual.graph.nodes)
    for engine in engines:
        assert elapsed[engine] < 60.0, f"engine {engine} took {elapsed[engine]:.3f}s, limit 60s"


@pytest.mark.parametrize("node_count", [100_000, 1_000_000])
def test_the_c_engine_scales_to_a_million_nodes(node_count: int) -> None:
    """Scaling row for the C engine, with the share the structural loop takes.

    The loop itself is linear in the operation count; the rest is the Python
    fold, which allocates one tree node per operation because the residual
    trees are the kernel's output.
    """
    graph = sparse_multigraph(node_count, edge_count=node_count * 5 // 4, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, node_count // 100))
    kernel = graph._kernel
    terminal_mask = bytearray(len(kernel.nodes))
    for node_id in terminals:
        terminal_mask[bisect_left(kernel.nodes, node_id)] = 1

    start = time.perf_counter()
    log = kernel.graph.series_parallel_reduce(terminal_mask, bytes(len(kernel.nodes)))
    loop_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    reduced = reduce(graph, terminals, engine="c")
    total_elapsed = time.perf_counter() - start

    print(  # noqa: T201 — benchmark output is intentional
        f"\n  reduce {node_count:,} nodes / {len(graph.endpoints):,} edges / {len(terminals):,} terminals: "
        f"{total_elapsed:.3f}s total, {loop_elapsed:.3f}s in the C loop for {len(log.op_kind):,} operations "
        f"-> {len(reduced.graph.nodes):,} nodes, {len(reduced.graph.endpoints):,} edges"
    )
    assert terminals <= set(reduced.graph.nodes)
    assert loop_elapsed < total_elapsed
    assert total_elapsed < 30.0, f"took {total_elapsed:.3f}s, limit 30s"


def test_two_hubs_joined_by_8000_two_paths_reduce_in_linear_time() -> None:
    """Every series move at a hub pair used to scan the smaller hub's incidence
    set for the parallel check, O(d) per move and quadratic overall."""
    path_count = 8_000
    nodes = list(range(path_count + 2))
    endpoints: dict[int, tuple[int, int]] = {}
    for middle in range(2, path_count + 2):
        endpoints[2 * middle] = (0, middle)
        endpoints[2 * middle + 1] = (middle, 1)
    graph = MultiGraph(nodes, endpoints)

    start = time.perf_counter()
    reduced = reduce(graph, terminals={0, 1})
    elapsed = time.perf_counter() - start

    print(f"\n  reduce two hubs with {path_count} two-paths: {elapsed:.3f}s")  # noqa: T201 — benchmark output is intentional
    (tree,) = reduced.provenance.values()
    assert reduced.graph.nodes == [0, 1]
    assert len(leaves(tree)) == 2 * path_count
    assert elapsed < 1.0, f"took {elapsed:.3f}s, limit 1s"
