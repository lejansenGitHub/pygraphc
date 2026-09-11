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

Artifacts land in ``profiles/`` (gitignored): one ``.prof`` per workflow for
``pstats`` or snakeviz, one ``.txt`` with the top entries by cumulative and by
total time, one shared ``summary.md`` and one ``summary.json``.
"""

import argparse
import cProfile
import json
import platform
import pstats
import random
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    """Accumulates wall time per phase name, in the order the phases first run."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Add the wall time of the block to ``name``; re-entering a name accumulates."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - start)


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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate input"):
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
    with timer.phase("generate samples"):
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


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        name="graph_build",
        description="build a Graph from node ids and edge pairs, then read its degrees",
        body=workflow_graph_build,
        default_size=200_000,
        guard_size=5_000,
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
        guard_size=2_000,
    ),
    Workflow(
        name="reduction_log",
        description="the same reduction consumed through the raw operation log instead of the trees",
        body=workflow_reduction_log,
        default_size=20_000,
        guard_size=2_000,
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
        guard_size=5_000,
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
    outcome = ""
    for _ in range(rounds):
        timer = PhaseTimer()
        start = time.perf_counter()
        outcome = workflow.body(timer, size)
        elapsed = time.perf_counter() - start
        if elapsed < best_total:
            best_total = elapsed
            best_phases = dict(timer.seconds)
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
        peak_bytes=peak_bytes,
        outcome=outcome,
        networkx_seconds=networkx_seconds,
        networkx_note=workflow.networkx_note,
    )


def write_profile(workflow: Workflow, size: int, directory: Path) -> None:
    """One unmeasured ``cProfile`` run per workflow, dumped as ``.prof`` and ``.txt``."""
    profiler = cProfile.Profile()
    profiler.enable()
    outcome = workflow.body(PhaseTimer(), size)
    profiler.disable()
    profiler.dump_stats(str(directory / f"{workflow.name}.prof"))
    with (directory / f"{workflow.name}.txt").open("w") as stream:
        stream.write(f"{workflow.name} — {workflow.description}\n")
        stream.write(f"size {size}, outcome: {outcome}\n")
        stream.write("A profiled run, not a timing run: these numbers carry the profiler's overhead.\n\n")
        stream.write("=== by cumulative time ===\n")
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
        stream.write("\n=== by total time ===\n")
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)


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


def summary_payload(measurements: list[Measurement], profile_name: str) -> dict[str, object]:
    """The machine-readable form of a harness run, the shape ``baseline.json`` uses too."""
    return {
        "size_profile": profile_name,
        "rounds": MEASUREMENT_ROUNDS,
        "warmups": MEASUREMENT_WARMUPS,
        "machine": machine_description(),
        "workflows": {
            measurement.name: {
                "description": measurement.description,
                "size": measurement.size,
                "total_seconds": measurement.total_seconds,
                "phase_seconds": measurement.phase_seconds,
                "phase_shares": measurement.phase_shares,
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
        "| Workflow | Size | Total | Peak | networkx | vs networkx | Phases accounted |",
        "|----------|-----:|------:|-----:|---------:|------------:|-----------------:|",
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
            f"{measurement.peak_bytes / 2**20:.1f} MiB | {reference_cell} | {ratio_cell} | "
            f"{measurement.accounted_share * 100:.1f}% |"
        )
    lines.extend(["", "## Phases", "", "| Workflow | Phase | Time | Share |", "|----------|-------|-----:|------:|"])
    for measurement in measurements:
        shares = measurement.phase_shares
        for phase_name, seconds in measurement.phase_seconds.items():
            lines.append(f"| `{measurement.name}` | {phase_name} | {seconds:.4f}s | {shares[phase_name] * 100:.1f}% |")
    lines.extend(["", "## What each workflow does", ""])
    for measurement in measurements:
        lines.append(f"- **`{measurement.name}`** — {measurement.description}. Result: {measurement.outcome}.")
        if measurement.networkx_note:
            lines.append(f"  - networkx reference: {measurement.networkx_note}.")
    lines.append("")
    return "\n".join(lines)


def console_table(measurements: list[Measurement]) -> str:
    """The same numbers as plain text, printed by the harness and by the guard on failure."""
    lines = [f"{'workflow':<24} {'phase':<26} {'seconds':>9} {'share':>7}", "-" * 69]
    for measurement in measurements:
        lines.append(
            f"{measurement.name:<24} {'TOTAL (phases accounted)':<26} {measurement.total_seconds:>9.4f} "
            f"{measurement.accounted_share * 100:>6.1f}%"
        )
        shares = measurement.phase_shares
        for phase_name, seconds in measurement.phase_seconds.items():
            lines.append(f"{'':<24} {phase_name:<26} {seconds:>9.4f} {shares[phase_name] * 100:>6.1f}%")
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
    arguments = parser.parse_args(argv)

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
