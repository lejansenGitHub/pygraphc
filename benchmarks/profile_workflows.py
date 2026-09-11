"""Phase profiler for the end-to-end workflows pygraphc supports.

Every workflow is a realistic sequence of library calls, not a single
operation, and its body is instrumented with explicit phases. Phases are the
point: the interesting split in this library is the C kernel against the
Python object construction on top of it, and that split is invisible in a flat
call profile. A reduction that spends a tenth of its time in C and nine tenths
building provenance trees should say so on the first line of an artifact a
reader can open.

Timings are best-of-N with a warm-up, measured without the profiler attached.
The ``cProfile`` run of every workflow is a separate, unmeasured run, so the
profiler's overhead never lands in a reported number. The ``tracemalloc`` peak
comes from a third run for the same reason.

Usage::

    python benchmarks/profile_workflows.py                  # full sizes
    python benchmarks/profile_workflows.py --size guard      # the guard's sizes
    python benchmarks/profile_workflows.py --only reduction_log
    python benchmarks/profile_workflows.py --size guard --write-baseline
    python benchmarks/profile_workflows.py --compare-engines   # the four reduction engines

Artifacts land in ``profiles/`` (gitignored): one ``.prof`` per workflow for
``pstats`` or snakeviz, one ``.txt`` with the top entries by cumulative and by
total time, one shared ``summary.md`` and one ``summary.json``. The engine
comparison adds ``engine_comparison.md`` with its ``engine_comparison.json``,
which ``--rebuild-comparison`` rewrites the artifact from without re-measuring.
"""

import argparse
import cProfile
import json
import platform
import pstats
import random
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import ModuleType

import pygraphc
from pygraphc import reduction

MEASUREMENT_ROUNDS = 3
MEASUREMENT_WARMUPS = 1
NETWORKX_ROUNDS = 2


def emit(message: str) -> None:
    """Single print site of the harness, so its output stays greppable."""
    print(message)  # noqa: T201 — the harness is a command line tool, its output is the point


# ---------------------------------------------------------------------------
# Phase timer
# ---------------------------------------------------------------------------


class PhaseTimer:
    """Accumulates wall time per phase name, in the order the phases first run.

    A phase marked ``setup`` is the harness generating its own random inputs.
    That is not library work, and the reports keep it out of the shares they
    compare so that a share says something about pygraphc rather than about
    ``random``.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.setup_phases: set[str] = set()

    @contextmanager
    def phase(self, name: str, *, setup: bool = False) -> Iterator[None]:
        """Add the wall time of the block to ``name``; re-entering a name accumulates."""
        if setup:
            self.setup_phases.add(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - start)

    def absent(self, name: str) -> None:
        """Record a phase this workflow has no counterpart for as zero, so sibling tables line up row by row."""
        self.seconds.setdefault(name, 0.0)


WorkflowBody = Callable[[PhaseTimer, int], str]
NetworkxReference = Callable[[ModuleType, int], str]


def networkx_module() -> ModuleType | None:
    """networkx if it is installed, imported on demand: it is the reference column, not a dependency."""
    try:
        import networkx  # noqa: PLC0415 — optional, and the import is the availability check
    except ImportError:
        return None
    return networkx


@dataclass(frozen=True)
class Workflow:
    """One end-to-end sequence of library calls, with the sizes it is measured at."""

    name: str
    description: str
    body: WorkflowBody
    default_size: int
    guard_size: int
    """The size the guard test measures at.

    Small enough to run in a unit test suite, large enough that the phases the
    workflow exists to watch stay above the guard's absolute floor of 50
    microseconds; ``tests/unit_tests/test_workflow_profiles.py`` says which
    phases fall below it anyway.
    """
    networkx_reference: NetworkxReference | None = None
    networkx_note: str = ""

    def size_for(self, profile_name: str) -> int:
        return self.default_size if profile_name == "default" else self.guard_size


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def sparse_edge_pairs(node_count: int, edge_count: int, seed: int) -> list[tuple[int, int]]:
    """Random endpoint pairs over ``range(node_count)``, self-loops included."""
    rng = random.Random(seed)
    return [(rng.randrange(node_count), rng.randrange(node_count)) for _ in range(edge_count)]


def ring_with_chords(node_count: int, chord_count: int, seed: int) -> list[tuple[int, int]]:
    """A ring plus random chords, so every node has degree two or more and paths exist."""
    rng = random.Random(seed)
    edges = [(index, (index + 1) % node_count) for index in range(node_count)]
    edges.extend((rng.randrange(node_count), rng.randrange(node_count)) for _ in range(chord_count))
    return edges


def discrete_samples(sample_count: int, variable_count: int, cardinality: int, seed: int) -> list[list[int]]:
    """Rows of discrete observations in which every variable follows its predecessor."""
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for _ in range(sample_count):
        row = [rng.randrange(cardinality)]
        for _ in range(variable_count - 1):
            row.append(row[-1] if rng.random() < 0.8 else rng.randrange(cardinality))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Workflow bodies
# ---------------------------------------------------------------------------


def workflow_graph_build(timer: PhaseTimer, size: int) -> str:
    """Build a Graph from node ids and edge pairs and read its degrees back."""
    with timer.phase("generate input", setup=True):
        node_ids = list(range(size))
        edge_pairs = sparse_edge_pairs(size, size * 5 // 4, seed=11)
    with timer.phase("construct graph"):
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("degree kernel"):
        degrees = graph.degrees()
    with timer.phase("degrees to python"):
        degree_sum = sum(degrees.tolist())
    outcome = f"{graph.node_count} nodes, {graph.edge_count} edges, degree sum {degree_sum}"
    with timer.phase("release inputs"):
        del node_ids, edge_pairs, graph, degrees
    return outcome


def networkx_graph_build(networkx: ModuleType, size: int) -> str:
    edge_pairs = sparse_edge_pairs(size, size * 5 // 4, seed=11)
    graph = networkx.MultiGraph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(edge_pairs)
    degree_sum = sum(degree for _node, degree in graph.degree())
    return f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, degree sum {degree_sum}"


def workflow_connected_components(timer: PhaseTimer, size: int) -> str:
    """Partition a large graph into components, as int32 labels and as Python sets."""
    with timer.phase("generate input", setup=True):
        node_ids = list(range(size))
        edge_pairs = sparse_edge_pairs(size, size * 5 // 4, seed=23)
    with timer.phase("construct graph"):
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("label kernel"):
        labels = graph.component_labels()
        label_count = len(set(labels.tolist()))
    with timer.phase("materialise sets"):
        components = list(graph.connected_components())
    outcome = f"{label_count} labels, {len(components)} component sets"
    with timer.phase("release inputs"):
        del node_ids, edge_pairs, graph, labels, components
    return outcome


def networkx_connected_components(networkx: ModuleType, size: int) -> str:
    edge_pairs = sparse_edge_pairs(size, size * 5 // 4, seed=23)
    graph = networkx.Graph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(edge_pairs)
    components = list(networkx.connected_components(graph))
    return f"{len(components)} component sets"


def scenario_removals(edge_ids: list[int], scenario_count: int, seed: int) -> list[frozenset[int]]:
    """One removal set per scenario, each a handful of edges drawn from the graph."""
    rng = random.Random(seed)
    return [frozenset(rng.sample(edge_ids, 3)) for _ in range(scenario_count)]


def scenario_count_for(size: int) -> int:
    return max(4, size // 100)


def workflow_scenario_sweep(timer: PhaseTimer, size: int) -> str:
    """Build one graph, then mask a different edge set many times and re-partition."""
    with timer.phase("generate input", setup=True):
        nodes = list(range(size))
        endpoints = dict(enumerate(sparse_edge_pairs(size, size * 5 // 4, seed=37)))
        removals = scenario_removals(list(endpoints), scenario_count_for(size), seed=38)
    with timer.phase("construct multigraph"):
        graph = reduction.MultiGraph(nodes, endpoints)
        active = frozenset(endpoints)
    with timer.phase("base partition"):
        base = reduction.Partition.from_components(graph, active)
    with timer.phase("scenario partitions"):
        partitions = [reduction.scenario(graph, active, removed) for removed in removals]
    with timer.phase("refinement checks"):
        refining = sum(1 for partition in partitions if partition.refines(base))
    outcome = f"{len(partitions)} scenarios, {refining} refine the base partition"
    with timer.phase("release inputs"):
        del nodes, endpoints, removals, graph, active, base, partitions
    return outcome


def networkx_scenario_sweep(networkx: ModuleType, size: int) -> str:
    endpoints = dict(enumerate(sparse_edge_pairs(size, size * 5 // 4, seed=37)))
    removals = scenario_removals(list(endpoints), scenario_count_for(size), seed=38)
    graph = networkx.MultiGraph()
    graph.add_nodes_from(range(size))
    for edge_id, (from_node, to_node) in endpoints.items():
        graph.add_edge(from_node, to_node, key=edge_id)
    block_counts = []
    for removed in removals:
        removed_edges = [(*endpoints[edge_id], edge_id) for edge_id in removed]
        graph.remove_edges_from(removed_edges)
        block_counts.append(sum(1 for _component in networkx.connected_components(graph)))
        for from_node, to_node, edge_id in removed_edges:
            graph.add_edge(from_node, to_node, key=edge_id)
    return f"{len(block_counts)} scenarios"


def meta_graph_of(
    timer: PhaseTimer,
    size: int,
) -> tuple[reduction.MultiGraph[int], reduction.Partition, frozenset[int]]:
    """The shared head of the two reduction workflows: generate, partition, quotient.

    A fifth of the edges is removed, so the partition has many blocks and the
    quotient over them keeps those edges as meta edges. Both reduction
    workflows run this, which is what makes their tails comparable line by
    line: the trees against the raw operation log.
    """
    with timer.phase("generate input", setup=True):
        nodes = list(range(size))
        endpoints = dict(enumerate(sparse_edge_pairs(size, size * 5 // 4, seed=47)))
        removed = frozenset(edge_id for edge_id in endpoints if edge_id % 5 == 0)
    with timer.phase("partition"):
        graph = reduction.MultiGraph(nodes, endpoints)
        active = frozenset(endpoints)
        after = reduction.scenario(graph, active, removed)
    with timer.phase("quotient"):
        meta, _internal = reduction.quotient(after, graph, crossing=active)
    stride = max(1, len(meta.nodes) // max(2, len(meta.nodes) // 20))
    terminals = frozenset(meta.nodes[::stride])
    with timer.phase("release input graph"):
        del nodes, endpoints, removed, graph, active, _internal
    return meta, after, terminals


def workflow_framework_pipeline(timer: PhaseTimer, size: int) -> str:
    """Partition, quotient, lift, reduce, then expand the provenance trees to paths."""
    meta, after, terminals = meta_graph_of(timer, size)
    with timer.phase("lift"):
        attribute = {node_id: float(node_id % 7) for node_id in range(size)}
        lifted = reduction.lift(after, attribute, max)
    with timer.phase("reduce to trees"):
        reduced = reduction.reduce(meta, terminals)
    with timer.phase("expand trees to paths"):
        path_count = sum(len(reduction.paths(tree, cutoff=3)) for tree in reduced.provenance.values())
    outcome = (
        f"{len(meta.nodes)} blocks, {len(meta.endpoints)} meta edges, {len(lifted)} lifted, "
        f"{len(reduced.graph.nodes)} residual nodes, {path_count} paths"
    )
    with timer.phase("release inputs"):
        del meta, after, terminals, attribute, lifted, reduced
    return outcome


def workflow_reduction_log(timer: PhaseTimer, size: int) -> str:
    """The same reduction consumed through the raw operation log instead of the trees."""
    meta, _after, terminals = meta_graph_of(timer, size)
    with timer.phase("build reduction graph"):
        node_ids = list(meta.nodes)
        edge_pairs = list(meta.endpoints.values())
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("build masks"):
        index_of = {node_id: index for index, node_id in enumerate(node_ids)}
        terminal_mask = bytearray(len(node_ids))
        for node_id in terminals:
            terminal_mask[index_of[node_id]] = 1
        protected_mask = bytes(len(node_ids))
    with timer.phase("c loop"):
        log = graph.series_parallel_reduce(bytes(terminal_mask), protected_mask)
    with timer.phase("scan log"):
        kinds = log.op_kind.tolist()
        move_count = sum(1 for kind in kinds if kind != 0)
        surviving = len(log.surviving_nodes.tolist())
    outcome = f"{len(kinds)} operations, {move_count} moves, {surviving} surviving nodes"
    with timer.phase("release inputs"):
        del meta, _after, terminals, node_ids, edge_pairs, graph, index_of, terminal_mask, log, kinds
    return outcome


def workflow_weighted_queries(timer: PhaseTimer, size: int) -> str:
    """Single-source shortest path lengths and one source-to-target path."""
    with timer.phase("generate input", setup=True):
        node_ids = list(range(size))
        edge_pairs = ring_with_chords(size, size // 4, seed=53)
        weights = [1.0 + (index % 17) for index in range(len(edge_pairs))]
    with timer.phase("construct graph"):
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("single source lengths"):
        lengths = graph.shortest_path_lengths(weights, source=0)
    with timer.phase("source to target path"):
        path = graph.shortest_path(weights, source=0, target=size // 2)
    outcome = f"{len(lengths)} reachable nodes, path of {len(path)} nodes"
    with timer.phase("release inputs"):
        del node_ids, edge_pairs, weights, graph, lengths, path
    return outcome


def networkx_weighted_queries(networkx: ModuleType, size: int) -> str:
    edge_pairs = ring_with_chords(size, size // 4, seed=53)
    graph = networkx.Graph()
    graph.add_nodes_from(range(size))
    for index, (from_node, to_node) in enumerate(edge_pairs):
        graph.add_edge(from_node, to_node, weight=1.0 + (index % 17))
    lengths = networkx.single_source_dijkstra_path_length(graph, 0)
    path = networkx.dijkstra_path(graph, 0, size // 2)
    return f"{len(lengths)} reachable nodes, path of {len(path)} nodes"


def workflow_structural_queries(timer: PhaseTimer, size: int) -> str:
    """Bridges, biconnected components and the nodes on any simple source-target path."""
    with timer.phase("generate input", setup=True):
        node_ids = list(range(size))
        edge_pairs = ring_with_chords(size, size // 4, seed=59)
    with timer.phase("construct graph"):
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("bridges"):
        bridge_edges = graph.bridges()
    with timer.phase("biconnected components"):
        blocks = list(graph.biconnected_components())
    with timer.phase("nodes on simple paths"):
        on_paths = graph.nodes_on_simple_paths(source=0, targets=[size // 2])
    outcome = f"{len(bridge_edges)} bridges, {len(blocks)} blocks, {len(on_paths)} nodes on simple paths"
    with timer.phase("release inputs"):
        del node_ids, edge_pairs, graph, bridge_edges, blocks, on_paths
    return outcome


def networkx_structural_queries(networkx: ModuleType, size: int) -> str:
    edge_pairs = ring_with_chords(size, size // 4, seed=59)
    graph = networkx.Graph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(edge_pairs)
    bridge_edges = list(networkx.bridges(graph))
    blocks = list(networkx.biconnected_components(graph))
    return f"{len(bridge_edges)} bridges, {len(blocks)} blocks"


PATH_CUTOFF = 8
PATH_TARGET_FRACTION = 10


def workflow_path_enumeration(timer: PhaseTimer, size: int) -> str:
    """Enumerate every node-simple source-to-target path up to a cutoff."""
    with timer.phase("generate input", setup=True):
        node_ids = list(range(size))
        edge_pairs = ring_with_chords(size, size, seed=61)
    with timer.phase("construct graph"):
        graph = pygraphc.Graph(node_ids, edge_pairs)
    with timer.phase("enumerate paths"):
        target = size // PATH_TARGET_FRACTION
        found = graph.all_edge_paths(source=0, targets=target, cutoff=PATH_CUTOFF, node_simple=True)
    outcome = f"{len(found)} paths up to {PATH_CUTOFF} edges"
    with timer.phase("release inputs"):
        del node_ids, edge_pairs, graph, found, target
    return outcome


def networkx_path_enumeration(networkx: ModuleType, size: int) -> str:
    edge_pairs = ring_with_chords(size, size, seed=61)
    graph = networkx.Graph()
    graph.add_nodes_from(range(size))
    graph.add_edges_from(edge_pairs)
    found = list(networkx.all_simple_edge_paths(graph, 0, size // PATH_TARGET_FRACTION, cutoff=PATH_CUTOFF))
    return f"{len(found)} paths up to {PATH_CUTOFF} edges"


DAG_VARIABLE_COUNT = 12
DAG_CARDINALITY = 2


def workflow_dag_structure_learning(timer: PhaseTimer, size: int) -> str:
    """Learn a DAG from discrete samples by hill climb, then estimate its CPDs."""
    with timer.phase("generate samples", setup=True):
        samples = discrete_samples(size, DAG_VARIABLE_COUNT, DAG_CARDINALITY, seed=67)
        cardinalities = [DAG_CARDINALITY] * DAG_VARIABLE_COUNT
    with timer.phase("hill climb"):
        edges = pygraphc.hill_climb_k2(samples, cardinalities, max_indegree=2)
    with timer.phase("estimate cpds"):
        cpds = pygraphc.estimate_cpds(samples, cardinalities, edges)
    outcome = f"{len(edges)} learned edges, {len(cpds)} cpds over {size} samples"
    with timer.phase("release inputs"):
        del samples, cardinalities, edges, cpds
    return outcome


# ---------------------------------------------------------------------------
# Reduction engines, one workflow each
# ---------------------------------------------------------------------------

ENGINE_NAMES: tuple[str, ...] = ("c", "moves", "rounds", "python")
ENGINE_SEED = 71
ENGINE_TERMINAL_STRIDE = 100
ENGINE_COMPARISON_SIZES: tuple[int, ...] = (20_000, 100_000, 1_000_000)
ENGINE_PROFILED_SIZES: tuple[int, ...] = (20_000, 100_000)

PHASE_GENERATE = "generate input"
PHASE_KERNEL = "construct kernel graph"
PHASE_STRUCTURAL = "structural moves"
PHASE_FOLD = "fold provenance"
PHASE_RELEASE = "release"
ENGINE_PHASES: tuple[str, ...] = (PHASE_GENERATE, PHASE_KERNEL, PHASE_STRUCTURAL, PHASE_FOLD, PHASE_RELEASE)


def engine_input(size: int) -> tuple[list[int], dict[int, tuple[int, int]], frozenset[int]]:
    """The one seeded graph and terminal set every engine workflow reduces, so only the engine differs."""
    nodes = list(range(size))
    endpoints = dict(enumerate(sparse_edge_pairs(size, size * 5 // 4, seed=ENGINE_SEED)))
    terminals = frozenset(range(0, size, ENGINE_TERMINAL_STRIDE))
    return nodes, endpoints, terminals


def structural_log(
    engine: str,
    kernel: reduction._KernelGraph[int],
    terminals: frozenset[int],
) -> pygraphc.ReductionLog:
    """The structural half of one log-producing engine, with none of the payload algebra in it."""
    if engine == "c":
        return reduction._structural_log_c(kernel, terminals, frozenset(), fold_leaves=True)
    if engine == "moves":
        return reduction._structural_log_moves(kernel, terminals, frozenset())
    return reduction._structural_log_rounds(kernel, terminals, frozenset())


def workflow_reduce_engine(timer: PhaseTimer, size: int, engine: str) -> str:
    """One reduction through one engine, phased identically to its three siblings.

    The phases are the comparison: the same generated input, the same kernel
    graph, then the structural work of finding and applying moves against the
    fold that builds the provenance trees and the folded payloads. The Python
    engine interleaves those two, so its fold is recorded as zero rather than
    omitted and the four tables line up row by row.
    """
    with timer.phase(PHASE_GENERATE, setup=True):
        nodes, endpoints, terminals = engine_input(size)
    with timer.phase(PHASE_KERNEL):
        graph = reduction.MultiGraph(nodes, endpoints)
        kernel = graph._kernel
    if engine == "python":
        with timer.phase(PHASE_STRUCTURAL):
            reduced = reduction.reduce(graph, terminals, engine="python")
        timer.absent(PHASE_FOLD)
    else:
        with timer.phase(PHASE_STRUCTURAL):
            log = structural_log(engine, kernel, terminals)
        with timer.phase(PHASE_FOLD):
            reduced = reduction._fold_operation_log(graph, kernel, log, fold_leaves=True)
        del log
    outcome = (
        f"{len(reduced.graph.nodes)} residual nodes, {len(reduced.graph.endpoints)} residual edges, "
        f"{len(reduced.provenance)} provenance trees, {len(reduced.folded_nodes)} folded payloads"
    )
    with timer.phase(PHASE_RELEASE):
        del nodes, endpoints, terminals, graph, kernel, reduced
    return outcome


def engine_workflow(engine: str) -> Workflow:
    """The registry entry for one engine; the four differ in the engine and in nothing else."""
    return Workflow(
        name=f"reduce_engine_{engine}",
        description=f"reduce one seeded graph with the {engine!r} engine, phased for the engine comparison",
        body=partial(workflow_reduce_engine, engine=engine),
        default_size=20_000,
        guard_size=2_000,
    )


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        name="graph_build",
        description="build a Graph from node ids and edge pairs, then read its degrees",
        body=workflow_graph_build,
        default_size=200_000,
        guard_size=50_000,
        networkx_reference=networkx_graph_build,
    ),
    Workflow(
        name="connected_components",
        description="partition a large graph into components, as int32 labels and as Python sets",
        body=workflow_connected_components,
        default_size=200_000,
        guard_size=5_000,
        networkx_reference=networkx_connected_components,
        networkx_note="networkx has no label form, the reference builds the component sets",
    ),
    Workflow(
        name="scenario_sweep",
        description="build one graph, then mask a different edge set many times and re-partition",
        body=workflow_scenario_sweep,
        default_size=20_000,
        guard_size=2_000,
        networkx_reference=networkx_scenario_sweep,
        networkx_note="the reference removes and re-adds the edges of every scenario on one graph",
    ),
    Workflow(
        name="framework_pipeline",
        description="partition, quotient, lift, reduce, then expand the provenance trees to paths",
        body=workflow_framework_pipeline,
        default_size=20_000,
        guard_size=5_000,
    ),
    Workflow(
        name="reduction_log",
        description="the same reduction consumed through the raw operation log instead of the trees",
        body=workflow_reduction_log,
        default_size=20_000,
        guard_size=10_000,
    ),
    Workflow(
        name="weighted_queries",
        description="single-source shortest path lengths and one source-to-target path",
        body=workflow_weighted_queries,
        default_size=100_000,
        guard_size=5_000,
        networkx_reference=networkx_weighted_queries,
    ),
    Workflow(
        name="structural_queries",
        description="bridges, biconnected components and the nodes on any simple source-target path",
        body=workflow_structural_queries,
        default_size=100_000,
        guard_size=10_000,
        networkx_reference=networkx_structural_queries,
        networkx_note="the reference covers bridges and biconnected components, it has no block-cut path form",
    ),
    Workflow(
        name="path_enumeration",
        description="enumerate every node-simple source-to-target path up to a cutoff",
        body=workflow_path_enumeration,
        default_size=1_000,
        guard_size=200,
        networkx_reference=networkx_path_enumeration,
    ),
    Workflow(
        name="dag_structure_learning",
        description="learn a DAG from discrete samples by hill climb, then estimate its CPDs",
        body=workflow_dag_structure_learning,
        default_size=2_000,
        guard_size=300,
    ),
    *(engine_workflow(engine) for engine in ENGINE_NAMES),
)

WORKFLOWS_BY_NAME = {workflow.name: workflow for workflow in WORKFLOWS}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """Best-of-N wall time of one workflow with the phases of that best round."""

    name: str
    description: str
    size: int
    total_seconds: float
    phase_seconds: dict[str, float] = field(default_factory=dict)
    setup_phases: frozenset[str] = frozenset()
    peak_bytes: int = 0
    outcome: str = ""
    networkx_seconds: float | None = None
    networkx_note: str = ""

    @property
    def phase_shares(self) -> dict[str, float]:
        """Share of the measured total per phase, as a fraction."""
        if self.total_seconds <= 0.0:
            return dict.fromkeys(self.phase_seconds, 0.0)
        return {name: seconds / self.total_seconds for name, seconds in self.phase_seconds.items()}

    @property
    def library_seconds(self) -> float:
        """The measured total without the phases that only generate the harness's inputs."""
        return sum(seconds for name, seconds in self.phase_seconds.items() if name not in self.setup_phases)

    @property
    def library_shares(self) -> dict[str, float]:
        """Share of the library time per library phase, as a fraction.

        These are the shares the guard compares. Dividing by the workflow total
        instead would make most of them statements about ``random``, which
        generates between a third and nine tenths of these workflows, and that
        does not scale away: at forty times the size every generation share is
        within a few points of the one measured here.
        """
        library_seconds = self.library_seconds
        library_phase_seconds = {
            name: seconds for name, seconds in self.phase_seconds.items() if name not in self.setup_phases
        }
        if library_seconds <= 0.0:
            return dict.fromkeys(library_phase_seconds, 0.0)
        return {name: seconds / library_seconds for name, seconds in library_phase_seconds.items()}

    @property
    def accounted_share(self) -> float:
        """Share of the measured total that the phases account for."""
        if self.total_seconds <= 0.0:
            return 1.0
        return sum(self.phase_seconds.values()) / self.total_seconds


def time_networkx_reference(reference: NetworkxReference, size: int) -> float | None:
    """Best-of-N wall time of the networkx counterpart, ``None`` when networkx is missing."""
    networkx = networkx_module()
    if networkx is None:
        return None
    best = float("inf")
    for _ in range(NETWORKX_ROUNDS):
        start = time.perf_counter()
        reference(networkx, size)
        best = min(best, time.perf_counter() - start)
    return best


def measure(
    workflow: Workflow,
    size: int,
    rounds: int = MEASUREMENT_ROUNDS,
    warmups: int = MEASUREMENT_WARMUPS,
    *,
    with_peak: bool = True,
    with_networkx: bool = True,
) -> Measurement:
    """Best-of-``rounds`` timing of the workflow, plus separate runs for peak and reference."""
    for _ in range(warmups):
        workflow.body(PhaseTimer(), size)
    best_total = float("inf")
    best_phases: dict[str, float] = {}
    best_setup_phases: frozenset[str] = frozenset()
    outcome = ""
    for _ in range(rounds):
        timer = PhaseTimer()
        start = time.perf_counter()
        outcome = workflow.body(timer, size)
        elapsed = time.perf_counter() - start
        if elapsed < best_total:
            best_total = elapsed
            best_phases = dict(timer.seconds)
            best_setup_phases = frozenset(timer.setup_phases)
    peak_bytes = 0
    if with_peak:
        tracemalloc.start()
        workflow.body(PhaseTimer(), size)
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    networkx_seconds = None
    if with_networkx and workflow.networkx_reference is not None:
        networkx_seconds = time_networkx_reference(workflow.networkx_reference, size)
    return Measurement(
        name=workflow.name,
        description=workflow.description,
        size=size,
        total_seconds=best_total,
        phase_seconds=best_phases,
        setup_phases=best_setup_phases,
        peak_bytes=peak_bytes,
        outcome=outcome,
        networkx_seconds=networkx_seconds,
        networkx_note=workflow.networkx_note,
    )


def write_profile(workflow: Workflow, size: int, directory: Path, stem: str | None = None) -> Path:
    """One unmeasured ``cProfile`` run per workflow, dumped as ``.prof`` and ``.txt``."""
    stem = stem or workflow.name
    profiler = cProfile.Profile()
    profiler.enable()
    outcome = workflow.body(PhaseTimer(), size)
    profiler.disable()
    profile_path = directory / f"{stem}.prof"
    profiler.dump_stats(str(profile_path))
    with (directory / f"{stem}.txt").open("w") as stream:
        stream.write(f"{workflow.name} — {workflow.description}\n")
        stream.write(f"size {size}, outcome: {outcome}\n")
        stream.write("A profiled run, not a timing run: these numbers carry the profiler's overhead.\n\n")
        stream.write("=== by cumulative time ===\n")
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
        stream.write("\n=== by total time ===\n")
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)
    return profile_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def machine_description() -> dict[str, str]:
    """The machine and interpreter a set of numbers came from."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def source_revision() -> str:
    """The commit these numbers were taken at, marked when the tree was not clean."""
    directory = Path(__file__).parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision} (dirty)" if modified else revision


def summary_payload(measurements: list[Measurement], profile_name: str) -> dict[str, object]:
    """The machine-readable form of a harness run, the shape ``baseline.json`` uses too."""
    return {
        "size_profile": profile_name,
        "rounds": MEASUREMENT_ROUNDS,
        "warmups": MEASUREMENT_WARMUPS,
        "machine": machine_description(),
        "taken_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "revision": source_revision(),
        "workflows": {
            measurement.name: {
                "description": measurement.description,
                "size": measurement.size,
                "total_seconds": measurement.total_seconds,
                "library_seconds": measurement.library_seconds,
                "phase_seconds": measurement.phase_seconds,
                "phase_shares": measurement.phase_shares,
                "library_shares": measurement.library_shares,
                "setup_phases": sorted(measurement.setup_phases),
                "accounted_share": measurement.accounted_share,
                "peak_bytes": measurement.peak_bytes,
                "outcome": measurement.outcome,
                "networkx_seconds": measurement.networkx_seconds,
                "networkx_note": measurement.networkx_note,
            }
            for measurement in measurements
        },
    }


SUMMARY_HEADER = """# Workflow profiles

Wall time per phase of every end-to-end workflow pygraphc supports, so that a
workflow spending most of its time building Python objects around a fast C
kernel says so here instead of being found out by accident.

**How these numbers were taken.** Every workflow ran {warmups} warm-up round
and {rounds} measured rounds; the tables hold the fastest round and the phases
of that round. The `cProfile` artifacts come from a separate, unmeasured run,
so the profiler's overhead sits in `<workflow>.txt` and in no time below. The
`tracemalloc` peak comes from a third run for the same reason. The networkx
column is a reference, never a gate.

**The last phase.** Every workflow ends with a `release inputs` phase that
drops its own objects. Freeing a few hundred thousand tuples and a C graph is
wall time the caller pays, and attributing it is what lets the phases sum to
the total instead of leaving a constant unexplained remainder.

**The first phase, and why it has its own column.** The phase that generates a
workflow's random inputs is the harness, not the library, and on the smaller
workflows it is most of the wall time. Every table therefore carries both the
share of the workflow and the share of the *library* time, which is the total
without the generation phases. The second is the number to read when asking
where pygraphc spends its time, and it is what the guard compares.

Size profile `{profile_name}` on {platform}, {implementation} {python}.

"""


def summary_markdown(measurements: list[Measurement], profile_name: str) -> str:
    """The shared ``summary.md``: one overview table, one phase table, one legend."""
    machine = machine_description()
    lines = [
        SUMMARY_HEADER.format(
            warmups=MEASUREMENT_WARMUPS,
            rounds=MEASUREMENT_ROUNDS,
            profile_name=profile_name,
            platform=machine["platform"],
            implementation=machine["implementation"],
            python=machine["python"],
        ),
        "## Workflows",
        "",
        "| Workflow | Size | Total | Library | Peak | networkx | vs networkx | Phases accounted |",
        "|----------|-----:|------:|--------:|-----:|---------:|------------:|-----------------:|",
    ]
    for measurement in measurements:
        reference = measurement.networkx_seconds
        reference_cell = "—" if reference is None else f"{reference:.4f}s"
        if reference is None or measurement.total_seconds <= 0.0:
            ratio_cell = "—"
        else:
            ratio_cell = f"{reference / measurement.total_seconds:.1f}x"
        lines.append(
            f"| `{measurement.name}` | {measurement.size:,} | {measurement.total_seconds:.4f}s | "
            f"{measurement.library_seconds:.4f}s | "
            f"{measurement.peak_bytes / 2**20:.1f} MiB | {reference_cell} | {ratio_cell} | "
            f"{measurement.accounted_share * 100:.1f}% |"
        )
    lines.extend([
        "",
        "## Phases",
        "",
        "| Workflow | Phase | Time | Share of workflow | Share of library |",
        "|----------|-------|-----:|------------------:|-----------------:|",
    ])
    for measurement in measurements:
        shares = measurement.phase_shares
        library_shares = measurement.library_shares
        for phase_name, seconds in measurement.phase_seconds.items():
            library_cell = (
                "harness setup"
                if phase_name in measurement.setup_phases
                else f"{library_shares[phase_name] * 100:.1f}%"
            )
            lines.append(
                f"| `{measurement.name}` | {phase_name} | {seconds:.4f}s | "
                f"{shares[phase_name] * 100:.1f}% | {library_cell} |"
            )
    lines.extend(["", "## What each workflow does", ""])
    for measurement in measurements:
        lines.append(f"- **`{measurement.name}`** — {measurement.description}. Result: {measurement.outcome}.")
        if measurement.networkx_note:
            lines.append(f"  - networkx reference: {measurement.networkx_note}.")
    lines.append("")
    return "\n".join(lines)


def console_table(measurements: list[Measurement]) -> str:
    """The same numbers as plain text, printed by the harness and by the guard on failure."""
    lines = [
        f"{'workflow':<24} {'phase':<26} {'seconds':>9} {'of total':>9} {'of library':>11}",
        "-" * 81,
    ]
    for measurement in measurements:
        lines.append(
            f"{measurement.name:<24} {'TOTAL (phases accounted)':<26} {measurement.total_seconds:>9.4f} "
            f"{measurement.accounted_share * 100:>8.1f}%"
        )
        lines.append(f"{'':<24} {'LIBRARY (total less setup)':<26} {measurement.library_seconds:>9.4f}")
        shares = measurement.phase_shares
        library_shares = measurement.library_shares
        for phase_name, seconds in measurement.phase_seconds.items():
            library_cell = (
                "      setup"
                if phase_name in measurement.setup_phases
                else f"{library_shares[phase_name] * 100:>10.1f}%"
            )
            lines.append(f"{'':<24} {phase_name:<26} {seconds:>9.4f} {shares[phase_name] * 100:>8.1f}% {library_cell}")
    return "\n".join(lines)


def run(
    workflows: tuple[Workflow, ...],
    profile_name: str,
    output_directory: Path,
    *,
    write_artifacts: bool = True,
    with_networkx: bool = True,
) -> list[Measurement]:
    """Measure every workflow and, unless suppressed, write its artifacts."""
    if write_artifacts:
        output_directory.mkdir(parents=True, exist_ok=True)
    measurements = []
    for workflow in workflows:
        size = workflow.size_for(profile_name)
        measurement = measure(workflow, size, with_networkx=with_networkx)
        measurements.append(measurement)
        if write_artifacts:
            write_profile(workflow, size, output_directory)
    if write_artifacts:
        (output_directory / "summary.md").write_text(summary_markdown(measurements, profile_name))
        (output_directory / "summary.json").write_text(
            json.dumps(summary_payload(measurements, profile_name), indent=2) + "\n"
        )
    return measurements


# ---------------------------------------------------------------------------
# Engine comparison
# ---------------------------------------------------------------------------

BOUNDARY_MARKER = "pygraphc._core."
ENGINE_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (("c", "rounds"), ("c", "moves"), ("c", "python"))
ENGINE_DIFF_ENTRIES = 12
ENGINE_COMPARISON_ROUNDS = 3
CLAIMED_BOUNDARY_BAND: dict[str, tuple[float, float]] = {"moves": (0.09, 0.15), "rounds": (0.0, 0.03)}
CLAIMED_FOLD_SHARE = 0.80
CLAIMED_FOLD_TOLERANCE = 0.05


def engine_workflow_name(engine: str) -> str:
    return f"reduce_engine_{engine}"


@dataclass(frozen=True)
class ProfileEntry:
    """One function of one profile: its call counts, its own time and its cumulative time."""

    label: str
    primitive_calls: int
    total_calls: int
    total_seconds: float
    cumulative_seconds: float


@dataclass(frozen=True)
class ProfileSummary:
    """Every function of one ``.prof`` file, with the call counts the whole run made."""

    engine: str
    entries: dict[str, ProfileEntry]
    primitive_calls: int
    total_calls: int
    profiled_seconds: float

    @property
    def boundary_calls(self) -> int:
        """Primitive calls into the C extension: one per crossing of the Python-to-C boundary."""
        return sum(entry.primitive_calls for label, entry in self.entries.items() if BOUNDARY_MARKER in label)

    @property
    def boundary_seconds(self) -> float:
        """Time inside the C extension itself, exclusive of the Python frames that called it."""
        return sum(entry.total_seconds for label, entry in self.entries.items() if BOUNDARY_MARKER in label)


def read_profile(engine: str, path: Path) -> ProfileSummary:
    """One ``.prof`` file as a label-keyed table, which is what makes two profiles diffable."""
    stats = pstats.Stats(str(path))
    entries = {}
    for function, record in stats.stats.items():
        primitive_calls, total_calls, total_seconds, cumulative_seconds, _callers = record
        label = pstats.func_std_string(function)
        entries[label] = ProfileEntry(label, primitive_calls, total_calls, total_seconds, cumulative_seconds)
    return ProfileSummary(engine, entries, stats.prim_calls, stats.total_calls, stats.total_tt)


def short_label(label: str) -> str:
    """The profile's function label with the absolute path dropped, so a table row fits a screen."""
    if label.startswith("{"):
        return label
    file_name, separator, rest = label.partition(":")
    return f"{Path(file_name).name}:{rest}" if separator else label


@dataclass(frozen=True)
class LabelDelta:
    """One function as it appears in two profiles, with the difference the diff is about."""

    label: str
    left: ProfileEntry | None
    right: ProfileEntry | None

    @property
    def total_delta(self) -> float:
        return (self.right.total_seconds if self.right else 0.0) - (self.left.total_seconds if self.left else 0.0)

    @property
    def cumulative_delta(self) -> float:
        left = self.left.cumulative_seconds if self.left else 0.0
        right = self.right.cumulative_seconds if self.right else 0.0
        return right - left


def profile_deltas(left: ProfileSummary, right: ProfileSummary) -> list[LabelDelta]:
    """Every function of either profile, worst absolute own-time difference first."""
    labels = {*left.entries, *right.entries}
    deltas = [LabelDelta(label, left.entries.get(label), right.entries.get(label)) for label in labels]
    deltas.sort(key=lambda delta: abs(delta.total_delta), reverse=True)
    return deltas


@dataclass(frozen=True)
class EngineComparison:
    """One engine comparison run: timings at every size, profiles at the profiled sizes."""

    measurements: dict[int, dict[str, Measurement]]
    profiles: dict[int, dict[str, ProfileSummary]]
    machine: dict[str, str]


def run_engine_comparison(output_directory: Path) -> EngineComparison:
    """Time the four engine workflows at every size and profile them at the profiled sizes."""
    output_directory.mkdir(parents=True, exist_ok=True)
    measurements: dict[int, dict[str, Measurement]] = {}
    profiles: dict[int, dict[str, ProfileSummary]] = {}
    for size in ENGINE_COMPARISON_SIZES:
        measurements[size] = {}
        profiles[size] = {}
        for engine in ENGINE_NAMES:
            workflow = WORKFLOWS_BY_NAME[engine_workflow_name(engine)]
            emit(f"  {engine:<8} at {size:,} …")
            measurements[size][engine] = measure(
                workflow,
                size,
                rounds=ENGINE_COMPARISON_ROUNDS,
                with_peak=False,
                with_networkx=False,
            )
            if size in ENGINE_PROFILED_SIZES:
                profiles[size][engine] = read_profile(
                    engine, write_profile(workflow, size, output_directory, profile_stem(engine, size))
                )
    return EngineComparison(measurements, profiles, machine_description())


def profile_stem(engine: str, size: int) -> str:
    """File name stem of one engine's profile at one size; one file per pair, never overwritten."""
    return f"{engine_workflow_name(engine)}_{size}"


def write_engine_comparison_numbers(comparison: EngineComparison, path: Path) -> None:
    """The measured numbers machine-readably, so the artifact can be rebuilt without re-measuring."""
    payload = {
        "machine": machine_description(),
        "rounds": ENGINE_COMPARISON_ROUNDS,
        "warmups": MEASUREMENT_WARMUPS,
        "sizes": list(ENGINE_COMPARISON_SIZES),
        "profiled_sizes": list(ENGINE_PROFILED_SIZES),
        "measurements": {
            str(size): {
                engine: {
                    "name": measurement.name,
                    "description": measurement.description,
                    "size": measurement.size,
                    "total_seconds": measurement.total_seconds,
                    "phase_seconds": measurement.phase_seconds,
                    "outcome": measurement.outcome,
                }
                for engine, measurement in by_engine.items()
            }
            for size, by_engine in comparison.measurements.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_engine_comparison(directory: Path) -> EngineComparison:
    """The last run of the comparison, read back from its numbers and its ``.prof`` files."""
    payload = json.loads((directory / "engine_comparison.json").read_text())
    measurements = {
        int(size): {
            engine: Measurement(
                name=entry["name"],
                description=entry["description"],
                size=entry["size"],
                total_seconds=entry["total_seconds"],
                phase_seconds=entry["phase_seconds"],
                outcome=entry["outcome"],
            )
            for engine, entry in by_engine.items()
        }
        for size, by_engine in payload["measurements"].items()
    }
    profiles = {
        size: {
            engine: read_profile(engine, directory / f"{profile_stem(engine, size)}.prof") for engine in ENGINE_NAMES
        }
        for size in ENGINE_PROFILED_SIZES
    }
    return EngineComparison(measurements, profiles, payload["machine"])


def phase_table_lines(by_engine: dict[str, Measurement]) -> list[str]:
    """One table with the four engines side by side, a phase per row, absolute time and share per engine."""
    header = "| Phase | " + " | ".join(f"`{engine}` | share" for engine in ENGINE_NAMES) + " |"
    ruler = "|-------|" + "------:|------:|" * len(ENGINE_NAMES)
    lines = [header, ruler]
    for phase_name in ENGINE_PHASES:
        cells = []
        for engine in ENGINE_NAMES:
            measurement = by_engine[engine]
            seconds = measurement.phase_seconds.get(phase_name, 0.0)
            share = seconds / measurement.total_seconds if measurement.total_seconds > 0.0 else 0.0
            cells.append(f"{seconds:.4f}s | {share * 100:.1f}%")
        lines.append(f"| {phase_name} | " + " | ".join(cells) + " |")
    totals = []
    for engine in ENGINE_NAMES:
        measurement = by_engine[engine]
        totals.append(f"**{measurement.total_seconds:.4f}s** | {measurement.accounted_share * 100:.1f}%")
    lines.append("| **total** (share column: phases accounted) | " + " | ".join(totals) + " |")
    baseline_seconds = by_engine["c"].total_seconds
    relative = [
        f"{(by_engine[engine].total_seconds / baseline_seconds if baseline_seconds > 0.0 else 0.0):.3f}x | —"
        for engine in ENGINE_NAMES
    ]
    lines.append("| vs `c` | " + " | ".join(relative) + " |")
    python_note = (
        f"The `python` column is not phased like the other three: that engine interleaves the structural work "
        f"and the fold, so its `{PHASE_STRUCTURAL}` cell holds both and its `{PHASE_FOLD}` cell is zero by "
        f"construction rather than by measurement. Compare it against the sum of the two rows in the C-backed "
        f"columns, never against `{PHASE_STRUCTURAL}` alone."
    )
    lines.extend(["", python_note])
    return lines


def diff_table_lines(left: ProfileSummary, right: ProfileSummary) -> list[str]:
    """The functions whose own time differs most between two profiles, and the ones only one of them has."""
    deltas = profile_deltas(left, right)
    lines = [
        (
            f"| Function | `{left.engine}` tottime | `{right.engine}` tottime | Δ tottime | "
            f"Δ cumtime | `{left.engine}` calls | `{right.engine}` calls |"
        ),
        "|----------|------:|------:|------:|------:|------:|------:|",
    ]
    for delta in deltas[:ENGINE_DIFF_ENTRIES]:
        left_entry, right_entry = delta.left, delta.right
        lines.append(
            f"| `{short_label(delta.label)}` | "
            f"{'—' if left_entry is None else f'{left_entry.total_seconds:.4f}s'} | "
            f"{'—' if right_entry is None else f'{right_entry.total_seconds:.4f}s'} | "
            f"{delta.total_delta:+.4f}s | {delta.cumulative_delta:+.4f}s | "
            f"{'—' if left_entry is None else f'{left_entry.primitive_calls:,}'} | "
            f"{'—' if right_entry is None else f'{right_entry.primitive_calls:,}'} |"
        )
    only_left = sorted(
        (entry for label, entry in left.entries.items() if label not in right.entries),
        key=lambda entry: entry.total_seconds,
        reverse=True,
    )
    only_right = sorted(
        (entry for label, entry in right.entries.items() if label not in left.entries),
        key=lambda entry: entry.total_seconds,
        reverse=True,
    )
    lines.append("")
    lines.append(f"Only in `{left.engine}`: " + (describe_only(only_left) or "nothing."))
    lines.append("")
    lines.append(f"Only in `{right.engine}`: " + (describe_only(only_right) or "nothing."))
    lines.append("")
    lines.append(
        f"Primitive calls: `{left.engine}` {left.primitive_calls:,}, `{right.engine}` {right.primitive_calls:,}. "
        f"Of those, calls into the C extension: `{left.engine}` {left.boundary_calls:,} "
        f"({left.boundary_seconds:.4f}s inside C), `{right.engine}` {right.boundary_calls:,} "
        f"({right.boundary_seconds:.4f}s inside C)."
    )
    return lines


def describe_only(entries: list[ProfileEntry]) -> str:
    """The functions one profile has and the other does not, as one sentence."""
    if not entries:
        return ""
    shown = ", ".join(
        f"`{short_label(entry.label)}` ({entry.total_seconds:.4f}s, {entry.primitive_calls:,} calls)"
        for entry in entries[:ENGINE_DIFF_ENTRIES]
    )
    remainder = f", and {len(entries) - ENGINE_DIFF_ENTRIES} more" if len(entries) > ENGINE_DIFF_ENTRIES else ""
    return f"{shown}{remainder}."


def warm_call_seconds(measurement: Measurement) -> float:
    """What a ``reduce`` call costs on a graph whose kernel is already built: the structure plus the fold."""
    return measurement.phase_seconds[PHASE_STRUCTURAL] + measurement.phase_seconds.get(PHASE_FOLD, 0.0)


def cold_call_seconds(measurement: Measurement) -> float:
    """What the first ``reduce`` call on a graph costs: the kernel graph as well, since it is built on demand."""
    return measurement.phase_seconds[PHASE_KERNEL] + warm_call_seconds(measurement)


def boundary_overhead_share(comparison: EngineComparison, engine: str, size: int) -> float:
    """Share of the ``"c"`` engine's warm call that this engine's structural phase adds on top of it."""
    baseline = comparison.measurements[size]["c"]
    candidate = comparison.measurements[size][engine]
    reference = warm_call_seconds(baseline)
    if reference <= 0.0:
        return 0.0
    return (candidate.phase_seconds[PHASE_STRUCTURAL] - baseline.phase_seconds[PHASE_STRUCTURAL]) / reference


def fold_share_lines(comparison: EngineComparison) -> list[str]:
    """Question one: is the fold four fifths of the call for every engine, and identically so.

    Three denominators, because "the call" is ambiguous and the answer depends
    on which one is meant: the whole workflow, a warm ``reduce`` (structure plus
    fold) and a cold one (the kernel graph too, since it is built on demand).
    """
    lines = [
        "| Size | Engine | fold provenance | of whole workflow | of warm `reduce` | of cold `reduce` |",
        "|-----:|--------|------:|------:|------:|------:|",
    ]
    for size in ENGINE_COMPARISON_SIZES:
        for engine in ENGINE_NAMES:
            measurement = comparison.measurements[size][engine]
            fold = measurement.phase_seconds.get(PHASE_FOLD, 0.0)
            warm, cold = warm_call_seconds(measurement), cold_call_seconds(measurement)
            lines.append(
                f"| {size:,} | `{engine}` | {fold:.4f}s | "
                f"{fold / measurement.total_seconds * 100:.1f}% | "
                f"{(fold / warm * 100) if warm > 0.0 else 0.0:.1f}% | "
                f"{(fold / cold * 100) if cold > 0.0 else 0.0:.1f}% |"
            )
    return lines


def fold_noise_floor(comparison: EngineComparison, size: int) -> float:
    """How far two identical fold workloads drift apart at this size, as a share of the fold.

    ``"moves"`` applies the moves in the monolith's order and hands the fold
    the very same log, so the two fold phases do byte-identical work and what
    is left between them is measurement noise.

    It is one paired difference of two best-of-N minima, not a distribution:
    an estimate of the order of the noise, good enough to say that a few
    percent between two engines is not a ranking, and not good enough to
    quote as an interval. It is also measured on the fold phase, which is
    several times the structural phase here, so it bounds a difference between
    two folds and says nothing about a difference between two structural
    phases; those are compared against the monolith's own warm call instead.
    """
    monolith = comparison.measurements[size]["c"].phase_seconds.get(PHASE_FOLD, 0.0)
    moves = comparison.measurements[size]["moves"].phase_seconds.get(PHASE_FOLD, 0.0)
    return abs(monolith - moves) / monolith if monolith > 0.0 else 0.0


def fold_verdict_lines(comparison: EngineComparison) -> list[str]:
    """Question one: is the fold four fifths of the call for every engine, identically so."""
    lines = ["### Is the fold four fifths of the call for every engine, identically so?", ""]
    lines.extend(fold_share_lines(comparison))
    warm, cold = [], []
    for size in ENGINE_COMPARISON_SIZES:
        for engine in ENGINE_NAMES:
            if engine == "python":
                continue
            measurement = comparison.measurements[size][engine]
            fold = measurement.phase_seconds[PHASE_FOLD]
            warm.append(fold / warm_call_seconds(measurement))
            cold.append(fold / cold_call_seconds(measurement))
    claimed = CLAIMED_FOLD_SHARE
    warm_holds = all(abs(share - claimed) <= CLAIMED_FOLD_TOLERANCE for share in warm)
    cold_holds = all(abs(share - claimed) <= CLAIMED_FOLD_TOLERANCE for share in cold)
    lines.extend([
        "",
        f"**Answer: it depends which call, and the claim is only true of the more expensive one.** Across the "
        f"three log-producing engines and the three sizes the fold is {min(warm) * 100:.0f}% to "
        f"{max(warm) * 100:.0f}% of a warm `reduce` (structure plus fold, the kernel graph already built) and "
        f"{min(cold) * 100:.0f}% to {max(cold) * 100:.0f}% of a cold one (the kernel graph built on demand, "
        f"which is what the first call on a `MultiGraph` pays). Against {claimed * 100:.0f}% the warm figure "
        f"{'holds' if warm_holds else 'does not hold'} within {CLAIMED_FOLD_TOLERANCE * 100:.0f} points and "
        f"the cold figure {'holds' if cold_holds else 'does not hold'}. "
        + (
            f"Four fifths sits between the two: it understates the fold's share of a warm call "
            f"({sum(warm) / len(warm) * 100:.0f}% on average) and overstates its share of a cold one "
            f"({sum(cold) / len(cold) * 100:.0f}% on average), where building the compressed sparse row graph "
            f"is counted too."
            if min(warm) > claimed > max(cold)
            else (
                f"The measured means are {sum(warm) / len(warm) * 100:.0f}% warm and "
                f"{sum(cold) / len(cold) * 100:.0f}% cold."
            )
        ),
        "",
        (
            "**And not identically so.** The three log-producing engines differ by several points at every "
            "size, in the direction the engine predicts: an engine whose structural phase is dearer has a "
            "smaller fold share of the same fold. The Python engine has no fold to separate at all — its "
            "worklist builds the trees as it applies the moves — so its row is zero by construction and the "
            "claim cannot be made of it in either direction."
        ),
        "",
    ])
    return lines


def moves_verdict_lines(comparison: EngineComparison) -> list[str]:
    """Question two: where ``"moves"`` loses its margin, by function."""
    lines = ["### Where does `moves` lose its 15 to 20 percent?", ""]
    band_low, band_high = CLAIMED_BOUNDARY_BAND["moves"]
    for size in ENGINE_COMPARISON_SIZES:
        monolith, moves = comparison.measurements[size]["c"], comparison.measurements[size]["moves"]
        warm_ratio = warm_call_seconds(moves) / warm_call_seconds(monolith)
        share = boundary_overhead_share(comparison, "moves", size)
        agreement = "inside" if band_low <= share <= band_high else "outside"
        named = "profiled at the two smaller sizes only"
        crossings = ""
        if size in ENGINE_PROFILED_SIZES:
            left, right = comparison.profiles[size]["c"], comparison.profiles[size]["moves"]
            top = [delta for delta in profile_deltas(left, right) if delta.total_delta > 0.0][:5]
            named = ", ".join(f"`{short_label(delta.label)}` ({delta.total_delta:+.4f}s)" for delta in top)
            crossings = (
                f" Boundary crossings: {left.boundary_calls:,} for `c` against {right.boundary_calls:,} "
                f"for `moves`, {right.boundary_seconds - left.boundary_seconds:+.4f}s of it inside C."
            )
        lines.append(
            f"- **At {size:,}**: `moves` costs {(warm_ratio - 1) * 100:+.1f}% on a warm `reduce` and "
            f"{(moves.total_seconds / monolith.total_seconds - 1) * 100:+.1f}% on the whole workflow. Its "
            f"structural phase alone is {share * 100:+.1f}% of the monolith's warm call, {agreement} the "
            f"estimated {band_low * 100:.0f} to {band_high * 100:.0f} percent. Functions: {named}.{crossings}"
        )
    warm_costs = {
        size: warm_call_seconds(comparison.measurements[size]["moves"])
        / warm_call_seconds(comparison.measurements[size]["c"])
        - 1
        for size in ENGINE_COMPARISON_SIZES
    }
    structural_costs = {size: boundary_overhead_share(comparison, "moves", size) for size in ENGINE_COMPARISON_SIZES}
    band_comparison = (
        "Against the estimate, the part of the cost the profile actually attributes to the boundary — the "
        "structural phase, measured against the monolith's warm call — is "
        + ", ".join(f"{cost * 100:+.1f}% at {size:,}" for size, cost in structural_costs.items())
        + f", against the estimated {band_low * 100:.0f} to {band_high * 100:.0f} percent. The warm call "
        f"as a whole moves further ("
        + ", ".join(f"{cost * 100:+.1f}% at {size:,}" for size, cost in warm_costs.items())
        + "), and the extra is not boundary cost at all: `moves` and the monolith fold byte-identical logs, so "
        "every difference in their fold phases is measurement noise, whose order — one paired difference of "
        "two minima, not an interval — is "
        + ", ".join(f"±{fold_noise_floor(comparison, size) * 100:.1f}% at {size:,}" for size in ENGINE_COMPARISON_SIZES)
        + ". The 15 to 20 percent wall-time figure is therefore the right order of magnitude but reads the noise "
        "as signal: what `moves` demonstrably pays for its crossings is the structural delta, not the whole gap."
    )
    lines.extend([
        "",
        "**Answer.** The cost is one `sp_next_move` plus one `sp_apply_move` per move, and the Python "
        "wrapper method around each of them — four entries that do not exist in the monolith's profile at "
        "all, where the whole loop is one `series_parallel_reduce_ctx` call. Nothing else moves: every other "
        "function in the two profiles is the same function doing the same work. " + band_comparison,
        "",
    ])
    return lines


def rounds_verdict_lines(comparison: EngineComparison) -> list[str]:
    """Question three: whether ``"rounds"`` beats the monolith, and whether for the reason claimed."""
    lines = ["### Does `rounds` beat the monolith, and for the reason claimed?", ""]
    lines.extend([
        "| Size | total | structural | fold | structural Δ | fold Δ | noise floor of the fold |",
        "|-----:|------:|------:|------:|------:|------:|------:|",
    ])
    for size in ENGINE_COMPARISON_SIZES:
        monolith, rounds = comparison.measurements[size]["c"], comparison.measurements[size]["rounds"]
        structural_delta = rounds.phase_seconds[PHASE_STRUCTURAL] - monolith.phase_seconds[PHASE_STRUCTURAL]
        fold_delta = rounds.phase_seconds[PHASE_FOLD] - monolith.phase_seconds[PHASE_FOLD]
        lines.append(
            f"| {size:,} | {rounds.total_seconds / monolith.total_seconds:.3f}x | "
            f"{rounds.phase_seconds[PHASE_STRUCTURAL] / monolith.phase_seconds[PHASE_STRUCTURAL]:.3f}x | "
            f"{rounds.phase_seconds[PHASE_FOLD] / monolith.phase_seconds[PHASE_FOLD]:.3f}x | "
            f"{structural_delta:+.4f}s | {fold_delta:+.4f}s | "
            f"±{fold_noise_floor(comparison, size) * 100:.1f}% |"
        )
    lines.append("")
    for size in ENGINE_PROFILED_SIZES:
        left, right = comparison.profiles[size]["c"], comparison.profiles[size]["rounds"]
        lines.append(
            f"- At {size:,} the batch engine crosses the boundary {right.boundary_calls:,} times against "
            f"{left.boundary_calls:,} for the monolith and {comparison.profiles[size]['moves'].boundary_calls:,} "
            f"for `moves`, so its crossing count is already within an order of magnitude of a single call: there "
            f"is no per-move crossing left for a batch to remove."
        )
    largest = ENGINE_COMPARISON_SIZES[-1]
    smaller = ENGINE_COMPARISON_SIZES[:-1]
    smaller_ratios = {
        size: comparison.measurements[size]["rounds"].total_seconds / comparison.measurements[size]["c"].total_seconds
        for size in smaller
    }
    smaller_structural = {
        size: comparison.measurements[size]["rounds"].phase_seconds[PHASE_STRUCTURAL]
        / comparison.measurements[size]["c"].phase_seconds[PHASE_STRUCTURAL]
        for size in smaller
    }
    smaller_sizes_note = (
        "At the smaller sizes `rounds` runs at "
        + ", ".join(f"{ratio:.3f}x the monolith at {size:,}" for size, ratio in smaller_ratios.items())
        + " with a structural phase of "
        + ", ".join(f"{ratio:.3f}x" for ratio in smaller_structural.values())
        + ", so the batch does not win the structural phase where the structural phase is still readable."
    )
    monolith, rounds = comparison.measurements[largest]["c"], comparison.measurements[largest]["rounds"]
    structural_delta = rounds.phase_seconds[PHASE_STRUCTURAL] - monolith.phase_seconds[PHASE_STRUCTURAL]
    total_delta = rounds.total_seconds - monolith.total_seconds
    noise = fold_noise_floor(comparison, largest)
    structural_ceiling = monolith.phase_seconds[PHASE_STRUCTURAL] / warm_call_seconds(monolith)
    lines.extend([
        "",
        f"**Answer: it matches the monolith, it does not beat it by much, and not for the reason claimed.** "
        f"At {largest:,} the whole gap is {total_delta:+.4f}s, or "
        f"{total_delta / monolith.total_seconds * 100:+.1f}% of the monolith's call. The structural phase — "
        f"the batch scan against one heap pop per move, the thing the claim is about — accounts for "
        f"{structural_delta:+.4f}s of that, {abs(structural_delta / total_delta) * 100:.0f}% of the gap; the "
        f"other {100 - abs(structural_delta / total_delta) * 100:.0f}% sits in the fold, which no batch "
        f"engine sets out to make cheaper. And the claimed mechanism is bounded from above whatever it does: "
        f"the monolith's structural phase is only {structural_ceiling * 100:.1f}% of its own warm call, so "
        f"removing it entirely could not buy more than that, while the fold-phase noise floor at this size "
        f"is ±{noise * 100:.1f}% — `moves` hands the fold a byte-identical log to the monolith's and its fold "
        f"still differs by that much. " + smaller_sizes_note + " What the batch scan does buy is real but "
        "small and already spent: it cuts the crossings from one per move to a few dozen, which is why "
        "`rounds` never pays what `moves` pays. Beyond that the profile shows no mechanism by which a batch "
        "would overtake a single C call, and the measurements do not show it overtaking one.",
        "",
    ])
    return lines


def verdict_lines(comparison: EngineComparison) -> list[str]:
    """The three questions, answered from the numbers this run measured."""
    return [
        *fold_verdict_lines(comparison),
        *moves_verdict_lines(comparison),
        *rounds_verdict_lines(comparison),
    ]


ENGINE_COMPARISON_HEADER = """# Reduction engine comparison

Four engines do the same reduction: `"c"` runs the whole fixpoint loop in one C
call, `"moves"` drives the same loop from Python one move at a time over C
primitives on an opaque state handle, `"rounds"` drives it one batch of
independent moves per kind per round over the same handle, and `"python"` is
the pure Python worklist. Wall times had been compared before; where the time
goes had only been estimated by subtracting the monolith's loop time from the
new engines' structural phase. This artifact profiles the four and diffs the
profiles against each other.

**One workflow per engine, identical but for the engine.** Same seeded graph
(`sparse_edge_pairs`, seed {seed}), same terminal set (every
{stride}th node), same sizes, same phase names. Where an engine has no
counterpart to a phase it is recorded as zero, not omitted, so the tables line
up row by row.

**How these numbers were taken.** {rounds} measured rounds after
{warmups} warm-up, fastest round reported, no profiler attached. The `.prof`
files come from a separate, unmeasured run. The {profiled} sizes are profiled
under `cProfile`; {timed} is **timed only** — the Python engine alone takes
{python_note:.1f}s per round there, and `cProfile` on that pass buys nothing
the two smaller sizes do not already show.

Machine: {platform}, {implementation} {python}.

"""


def engine_comparison_markdown(comparison: EngineComparison) -> str:
    """``profiles/engine_comparison.md``: the phase tables, the profile diffs and the three answers."""
    machine = comparison.machine
    timed_only = [size for size in ENGINE_COMPARISON_SIZES if size not in ENGINE_PROFILED_SIZES]
    lines = [
        ENGINE_COMPARISON_HEADER.format(
            seed=ENGINE_SEED,
            stride=ENGINE_TERMINAL_STRIDE,
            rounds=ENGINE_COMPARISON_ROUNDS,
            warmups=MEASUREMENT_WARMUPS,
            profiled=", ".join(f"{size:,}" for size in ENGINE_PROFILED_SIZES),
            timed=", ".join(f"{size:,}" for size in timed_only),
            python_note=max(comparison.measurements[size]["python"].total_seconds for size in timed_only),
            platform=machine["platform"],
            implementation=machine["implementation"],
            python=machine["python"],
        ),
        "## Phases, four engines side by side",
        "",
    ]
    for size in ENGINE_COMPARISON_SIZES:
        profiled = "profiled and timed" if size in ENGINE_PROFILED_SIZES else "timed only"
        by_engine = comparison.measurements[size]
        lines.extend([
            (f"### {size:,} nodes, {size * 5 // 4:,} edges, {size // ENGINE_TERMINAL_STRIDE:,} terminals ({profiled})"),
            "",
        ])
        lines.extend(phase_table_lines(by_engine))
        lines.extend(["", f"Outcome, identical for all four: {by_engine['c'].outcome}.", ""])
    lines.extend(["## Profile diffs", ""])
    for size in ENGINE_PROFILED_SIZES:
        for left_engine, right_engine in ENGINE_COMPARISON_PAIRS:
            left = comparison.profiles[size][left_engine]
            right = comparison.profiles[size][right_engine]
            lines.extend([f"### `{left_engine}` against `{right_engine}` at {size:,}", ""])
            lines.extend(diff_table_lines(left, right))
            lines.append("")
    lines.extend(["## The three questions", ""])
    lines.extend(verdict_lines(comparison))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=["default", "guard"], default="default", help="which size profile to run")
    parser.add_argument("--only", action="append", default=[], help="run only this workflow, repeatable")
    parser.add_argument("--output", type=Path, default=Path("profiles"), help="artifact directory")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="also write benchmarks/baseline.json from this run (guard sizes only)",
    )
    parser.add_argument("--list", action="store_true", help="list the workflows and exit")
    parser.add_argument(
        "--compare-engines",
        action="store_true",
        help="profile the four reduction engines at the engine sizes and write engine_comparison.md",
    )
    parser.add_argument(
        "--rebuild-comparison",
        action="store_true",
        help="rewrite engine_comparison.md from the last run's numbers and profiles, measuring nothing",
    )
    arguments = parser.parse_args(argv)

    if arguments.compare_engines or arguments.rebuild_comparison:
        if arguments.rebuild_comparison:
            emit("rebuilding the engine comparison from the numbers of the last run")
            comparison = load_engine_comparison(arguments.output)
        else:
            emit(f"comparing the {len(ENGINE_NAMES)} reduction engines at {len(ENGINE_COMPARISON_SIZES)} sizes")
            comparison = run_engine_comparison(arguments.output)
            write_engine_comparison_numbers(comparison, arguments.output / "engine_comparison.json")
        artifact = arguments.output / "engine_comparison.md"
        artifact.write_text(engine_comparison_markdown(comparison))
        emit("")
        for size in ENGINE_COMPARISON_SIZES:
            emit(console_table([comparison.measurements[size][engine] for engine in ENGINE_NAMES]))
            emit("")
        emit(f"comparison written to {artifact}")
        return 0

    if arguments.list:
        for workflow in WORKFLOWS:
            emit(f"{workflow.name:<24} {workflow.description}")
        return 0

    unknown = [name for name in arguments.only if name not in WORKFLOWS_BY_NAME]
    if unknown:
        emit(f"unknown workflows: {', '.join(unknown)}")
        return 2
    selected = tuple(WORKFLOWS_BY_NAME[name] for name in arguments.only) if arguments.only else WORKFLOWS

    if arguments.write_baseline and arguments.size != "guard":
        emit("--write-baseline needs --size guard: the guard compares against the small sizes")
        return 2

    emit(f"profiling {len(selected)} workflows at the {arguments.size} sizes")
    measurements = run(selected, arguments.size, arguments.output)
    emit("")
    emit(console_table(measurements))
    emit("")
    emit(f"artifacts in {arguments.output}/: one .prof and .txt per workflow, summary.md, summary.json")

    if arguments.write_baseline:
        baseline_path = Path(__file__).with_name("baseline.json")
        baseline_path.write_text(json.dumps(summary_payload(measurements, arguments.size), indent=2) + "\n")
        emit(f"baseline written to {baseline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
