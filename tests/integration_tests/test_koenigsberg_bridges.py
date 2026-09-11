"""Integration case: the seven bridges of Konigsberg, 1736 (civil infrastructure, history of graph theory).

Field: civil infrastructure and the origin of graph theory. The facts about the
city are Euler's own: the four land masses of Konigsberg on the Pregel, the
seven named bridges joining them, and the impossibility of a walk that crosses
each bridge exactly once. Source: L. Euler, "Solutio problematis ad geometriam
situs pertinentis", Commentarii academiae scientiarum Petropolitanae 8 (1741),
128-140, presented to the Petersburg academy in 1736.

What this case exercises: the two-layer part of the framework. The city is
modelled at district granularity, one layer below Euler's own abstraction.
``Partition`` recovers the land masses from the land connections, ``quotient``
turns the district graph into Euler's four-node multigraph without losing the
identity of any single bridge, ``lift`` carries a per-district quantity up to
the land masses, and ``reduce`` with the two river banks as terminals shows
both what series-parallel provenance preserves (a pair of twin bridges stays
two distinguishable bridges) and where the reduction has to stop.
"""

import pytest

from pygraphc.reduction import (
    MultiGraph,
    Parallel,
    Partition,
    leaves,
    lift,
    minimal_toggles,
    quotient,
    reduce,
)

pytestmark = pytest.mark.integration

# The five districts of the city, at the granularity below Euler's land masses.
ALTSTADT = 0
LOEBENICHT = 1
VORSTADT = 2
KNEIPHOF = 3
LOMSE = 4

DISTRICT_NAMES = {
    ALTSTADT: "Altstadt",
    LOEBENICHT: "Loebenicht",
    VORSTADT: "Vorstadt",
    KNEIPHOF: "Kneiphof",
    LOMSE: "Lomse",
}

# The one connection that is not a bridge: the Altstadt and the Loebenicht lie
# side by side on the north bank of the Pregel and are reachable from each
# other on foot without crossing water.
NORTH_BANK_LAND_LINK = "land: Altstadt-Loebenicht"

# The seven bridges, with the district each end stood on. The Kraemerbruecke
# and the Schmiedebruecke reached the Kneiphof from the Altstadt; the Gruene
# Bruecke and the Koettelbruecke reached it from the Vorstadt on the south
# bank; the Holzbruecke reached the Lomse from the north bank on the Loebenicht
# side, the Hohe Bruecke reached it from the Vorstadt, and the Honigbruecke
# joined the two islands directly. Only the side of the river matters for
# Euler's argument; the district each north-bank bridge landed in matters only
# for the layer below it, and the quotient is the same either way.
BRIDGES: dict[str, tuple[int, int]] = {
    "Kraemerbruecke": (ALTSTADT, KNEIPHOF),
    "Schmiedebruecke": (ALTSTADT, KNEIPHOF),
    "Gruene Bruecke": (VORSTADT, KNEIPHOF),
    "Koettelbruecke": (VORSTADT, KNEIPHOF),
    "Holzbruecke": (LOEBENICHT, LOMSE),
    "Hohe Bruecke": (VORSTADT, LOMSE),
    "Honigbruecke": (KNEIPHOF, LOMSE),
}

# Illustrative district populations. The numbers are invented, not historical;
# what the test claims about them is only that combining them over a land mass
# is additive, which is a property of a headcount and not of these values.
DISTRICT_INHABITANTS = {
    ALTSTADT: 4200,
    LOEBENICHT: 1800,
    VORSTADT: 3100,
    KNEIPHOF: 2500,
    LOMSE: 400,
}


def city() -> MultiGraph[str]:
    """The district graph: five districts, the north-bank land link and the seven bridges."""
    endpoints: dict[str, tuple[int, int]] = {NORTH_BANK_LAND_LINK: (ALTSTADT, LOEBENICHT), **BRIDGES}
    return MultiGraph(list(DISTRICT_NAMES), endpoints)


def land_masses() -> Partition:
    """The land masses: the districts one can walk between without crossing the river."""
    return Partition.from_components(city(), active=frozenset({NORTH_BANK_LAND_LINK}))


def bridge_counts(euler_graph: MultiGraph[str]) -> dict[int, int]:
    """Number of bridge ends on each land mass."""
    counts = dict.fromkeys(euler_graph.nodes, 0)
    for near_side, far_side in euler_graph.endpoints.values():
        counts[near_side] += 1
        counts[far_side] += 1
    return counts


def test_walking_without_crossing_water_yields_the_four_land_masses() -> None:
    """A visitor to Konigsberg in 1736 stands in one of five named districts and
    asks which other districts she can reach without getting her feet wet. Only
    the Altstadt and the Loebenicht are joined by land; every other pair is
    separated by an arm of the Pregel. ``Partition.from_components`` over the
    land connections alone is exactly that question, and it is the step Euler
    performed silently in his first paragraph when he replaced the city by four
    regions; without it a caller would have to hand-group the districts and hope
    the grouping matched the map. The test establishes that the city really does
    consist of four land masses and that the north bank is the one made of two
    districts, which a reader can check against any map of the city: two towns
    on the north bank, one on the south bank, two islands.
    """
    # --- Input ---
    partition = land_masses()

    # --- Assert ---
    blocks = partition.blocks()
    assert len(blocks) == 4  # Euler's four regions, recovered from the land links
    # The Altstadt and the Loebenicht form the north bank; the other three stand alone.
    assert sorted(blocks.values()) == [[ALTSTADT, LOEBENICHT], [VORSTADT], [KNEIPHOF], [LOMSE]]
    # Walking from the Altstadt to the Loebenicht crosses no water.
    assert partition.block_of[ALTSTADT] == partition.block_of[LOEBENICHT]
    # Reaching either island, or the south bank, needs a bridge.
    assert partition.block_of[KNEIPHOF] != partition.block_of[ALTSTADT]
    assert partition.block_of[LOMSE] != partition.block_of[KNEIPHOF]
    assert partition.block_of[VORSTADT] != partition.block_of[ALTSTADT]


def test_quotient_by_the_bridges_reproduces_eulers_multigraph() -> None:
    """Euler's argument is about land masses and bridges, not districts, so the
    city has to be coarsened before it can be argued about: the question "can I
    cross all seven bridges once" is asked of the four regions. ``quotient``
    performs that coarsening while keeping every bridge a separate, named edge,
    which matters here because the two bridges from the north bank to the
    Kneiphof would be indistinguishable in a simple graph and the whole problem
    would change. By hand this is bookkeeping over parallel edges that a plain
    adjacency structure cannot hold at all. The test establishes Euler's own
    numbers: four land masses, seven bridges, and a bridge count of five on the
    Kneiphof and three on each of the other three land masses -- so all four are
    odd, and by Euler's criterion (a walk using every bridge once needs zero or
    two odd regions) no such walk exists. A reader can count the odd regions off
    the map instead of trusting the code.
    """
    # --- Input ---
    partition = land_masses()
    euler_graph, internal = quotient(partition, city(), crossing=frozenset(BRIDGES))

    # --- Assert ---
    assert len(euler_graph.nodes) == 4  # four land masses
    assert len(euler_graph.endpoints) == 7  # seven bridges, none merged away
    assert set(euler_graph.endpoints) == set(BRIDGES)  # each bridge kept its name
    assert internal == {}  # no bridge has both ends on one land mass

    north_bank = partition.block_of[ALTSTADT]
    south_bank = partition.block_of[VORSTADT]
    counts = bridge_counts(euler_graph)
    # Five bridges touched the Kneiphof; the other three land masses had three each.
    assert counts[KNEIPHOF] == 5
    assert counts[north_bank] == 3
    assert counts[south_bank] == 3
    assert counts[LOMSE] == 3
    # Every bridge has two ends, so the counts sum to twice the number of bridges.
    assert sum(counts.values()) == 2 * 7
    # All four land masses have an odd bridge count.
    odd_land_masses = [land_mass for land_mass, count in counts.items() if count % 2 == 1]
    assert len(odd_land_masses) == 4
    # Euler 1736: a walk crossing every bridge exactly once needs zero or two
    # odd regions, so four of them means no such walk exists.
    assert len(odd_land_masses) not in {0, 2}


def test_lift_adds_the_two_north_bank_districts_into_one_land_mass() -> None:
    """A city official who keeps headcounts per district but has to report per
    land mass -- how many people live on the north bank, how many on each island
    -- needs the district figures summed along the same grouping that the land
    links define. ``lift`` takes the partition that the land links produced and
    a combining rule, and pushes the per-district quantity up to the land
    masses, in a fixed order so the result does not depend on dictionary order.
    The framework supplies the grouping and the fold; the caller still has to
    supply a rule that is associative and commutative, because ``lift`` does not
    check that and would silently give an order-dependent answer for a rule that
    is not. The test establishes that the north bank's figure is the sum of its
    two districts and that the three single-district land masses carry their own
    figure unchanged, which is what "population of a land mass" means to anyone
    who would read the report.
    """
    # --- Input ---
    partition = land_masses()
    inhabitants_per_land_mass = lift(partition, DISTRICT_INHABITANTS, combine=lambda left, right: left + right)

    # --- Assert ---
    north_bank = partition.block_of[ALTSTADT]
    # The north bank holds everyone who lives in the Altstadt or the Loebenicht.
    assert inhabitants_per_land_mass[north_bank] == DISTRICT_INHABITANTS[ALTSTADT] + DISTRICT_INHABITANTS[LOEBENICHT]
    # A land mass that is a single district keeps that district's figure.
    assert inhabitants_per_land_mass[partition.block_of[VORSTADT]] == DISTRICT_INHABITANTS[VORSTADT]
    assert inhabitants_per_land_mass[partition.block_of[KNEIPHOF]] == DISTRICT_INHABITANTS[KNEIPHOF]
    assert inhabitants_per_land_mass[partition.block_of[LOMSE]] == DISTRICT_INHABITANTS[LOMSE]
    # Nobody is counted twice and nobody is lost.
    assert sum(inhabitants_per_land_mass.values()) == sum(DISTRICT_INHABITANTS.values())


def test_both_twin_bridges_must_go_to_cut_a_bank_off_the_kneiphof() -> None:
    """A city engineer asks what it would take to cut the Kneiphof off from the
    north bank -- a question about demolition, or about two bridges being shut
    at once. Two bridges, the Kraemerbruecke and the Schmiedebruecke, ran side
    by side between those two land masses, so the answer is "both", and the
    point of the reduction is that this answer survives the coarsening:
    ``reduce`` replaces the twin bridges by one residual route, but its
    provenance is a ``Parallel`` node that still names both, and
    ``minimal_toggles`` over that node is precisely the question "which bridges
    must change state for the route to be gone". Done by hand this means
    enumerating subsets of bridges and re-testing reachability for each. The
    test establishes that the pair is not silently collapsed into a single
    bridge and that cutting the north bank's access to the Kneiphof costs
    exactly the two named bridges -- which a reader can verify from the bridge
    list, since those are the only two with one end on the north bank and the
    other on the Kneiphof.
    """
    # --- Input ---
    partition = land_masses()
    euler_graph, _internal = quotient(partition, city(), crossing=frozenset(BRIDGES))
    north_bank = partition.block_of[ALTSTADT]
    south_bank = partition.block_of[VORSTADT]
    reduced = reduce(euler_graph, terminals=frozenset({north_bank, south_bank}))

    # --- Assert ---
    north_bank_twins = frozenset({"Kraemerbruecke", "Schmiedebruecke"})
    twin_bridge_routes = [
        (route, tree) for route, tree in reduced.provenance.items() if leaves(tree) == north_bank_twins
    ]
    assert len(twin_bridge_routes) == 1  # the two twin bridges became one residual route
    route, tree = twin_bridge_routes[0]
    # That route still runs between the north bank and the Kneiphof.
    assert set(reduced.graph.endpoints[route]) == {north_bank, KNEIPHOF}
    # It remembers that it is two bridges side by side, not one bridge.
    assert isinstance(tree, Parallel)
    assert len(tree.children) == 2

    all_bridges_standing = dict.fromkeys(BRIDGES, True)
    must_go = minimal_toggles(tree, all_bridges_standing, target_closed=False)
    # Demolishing only one of the twins leaves the other one carrying the route,
    # so both have to go -- and nothing else does.
    assert must_go == north_bank_twins

    south_bank_twins = frozenset({"Gruene Bruecke", "Koettelbruecke"})
    south_route = next(tree for tree in reduced.provenance.values() if leaves(tree) == south_bank_twins)
    # The same holds for the south bank's two bridges to the Kneiphof.
    assert minimal_toggles(south_route, all_bridges_standing, target_closed=False) == south_bank_twins
    # The two answers name disjoint bridges: demolishing one bank's pair says
    # nothing about the other bank's access to the Kneiphof.
    assert not (must_go & south_bank_twins)


def test_reduction_stops_at_four_land_masses_and_five_routes() -> None:
    """With the two river banks fixed as the endpoints of interest, a planner
    asks how much of the bridge network can be simplified away without changing
    any question about getting from bank to bank. ``reduce`` applies the only
    three simplifications that are always safe -- drop a dead end, merge a
    through-route, merge side-by-side routes -- and stops when none applies;
    that fixpoint is the honest answer, not a failure. Here it merges each pair
    of twin bridges and then halts, because after the merge both islands have
    three routes each and neither is a dead end nor a through-route. The test
    establishes the limit that makes this case worth keeping: a network of this
    shape -- four land masses, the two banks as the endpoints of interest, the
    two islands each joined to both banks and to each other -- is the smallest
    two-endpoint graph that series and parallel moves cannot reduce further (K4
    minus the edge between the two terminals), so five routes is the floor and
    not an artefact. A reader can confirm it by looking for a dead end or a
    land mass with exactly two routes in the bridge list; there is none.
    """
    # --- Input ---
    partition = land_masses()
    euler_graph, _internal = quotient(partition, city(), crossing=frozenset(BRIDGES))
    north_bank = partition.block_of[ALTSTADT]
    south_bank = partition.block_of[VORSTADT]
    reduced = reduce(euler_graph, terminals=frozenset({north_bank, south_bank}))

    # --- Assert ---
    # No land mass could be removed: the two banks are the endpoints, and both
    # islands keep three routes each.
    assert sorted(reduced.graph.nodes) == sorted(euler_graph.nodes)
    assert len(reduced.graph.nodes) == 4
    # Seven bridges, two twin pairs merged, five routes left.
    assert len(reduced.graph.endpoints) == 5
    # Every one of the seven bridges is still accounted for by some route.
    all_leaves: frozenset[str] = frozenset().union(*(leaves(tree) for tree in reduced.provenance.values()))
    assert all_leaves == frozenset(BRIDGES)
    # Nothing was folded into a land mass: no bridge ends in a dead end.
    assert all(not folded for folded in reduced.folded_nodes.values())
    assert reduced.dropped == []
    # Each island still has three routes, which is why no further move applies.
    counts = bridge_counts(reduced.graph)
    assert counts[KNEIPHOF] == 3
    assert counts[LOMSE] == 3
    # Reducing the residual again changes nothing -- the reduction really stopped.
    again = reduce(reduced.graph, terminals=frozenset({north_bank, south_bank}))
    assert sorted(again.graph.nodes) == sorted(reduced.graph.nodes)
    assert len(again.graph.endpoints) == len(reduced.graph.endpoints)
