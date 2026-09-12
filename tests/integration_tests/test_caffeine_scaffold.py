"""Integration case: the molecular scaffold of caffeine (cheminformatics).

Field: cheminformatics, specifically scaffold analysis of small molecules.
The facts about the molecule are the standard structure of caffeine,
1,3,7-trimethylxanthine, C8H10N4O2: a purine bicyclic (a six-membered
pyrimidinedione ring fused to a five-membered imidazole ring across one bond)
carrying three N-methyl groups and two carbonyl oxygens. Hydrogens are not
modelled, which is the normal convention for scaffold work. The definition of
the scaffold is the one from G. W. Bemis and M. A. Murcko, "The Properties of
Known Drugs. 1. Molecular Frameworks", J. Med. Chem. 39 (1996), 2887-2893:
remove the side chains, keep the ring systems and the linkers between them.

What this case exercises: the pendant move of ``reduce`` with ``fold_leaves``
and the terminal set as a chemical constraint. Side-chain stripping is the
pendant move, and nothing else; the accounting of which side-chain atom sat on
which ring atom is ``Reduced.folded_nodes``. The ring atoms that become
terminals are found with the library's own cycle machinery rather than written
down, so the test does not smuggle in the answer it checks.
"""

import pytest

import pygraphc
from pygraphc.reduction import Leaf, MultiGraph, reduce

pytestmark = pytest.mark.integration

# Heavy atoms of caffeine. The nine ring atoms carry their purine locants; the
# five exocyclic atoms are named for the ring atom they hang on.
NITROGEN_1 = 1
CARBON_2 = 2
NITROGEN_3 = 3
CARBON_4 = 4
CARBON_5 = 5
CARBON_6 = 6
NITROGEN_7 = 7
CARBON_8 = 8
NITROGEN_9 = 9
METHYL_CARBON_ON_NITROGEN_1 = 11
METHYL_CARBON_ON_NITROGEN_3 = 13
METHYL_CARBON_ON_NITROGEN_7 = 17
CARBONYL_OXYGEN_ON_CARBON_2 = 22
CARBONYL_OXYGEN_ON_CARBON_6 = 26

ELEMENT_OF_ATOM = {
    NITROGEN_1: "N",
    CARBON_2: "C",
    NITROGEN_3: "N",
    CARBON_4: "C",
    CARBON_5: "C",
    CARBON_6: "C",
    NITROGEN_7: "N",
    CARBON_8: "C",
    NITROGEN_9: "N",
    METHYL_CARBON_ON_NITROGEN_1: "C",
    METHYL_CARBON_ON_NITROGEN_3: "C",
    METHYL_CARBON_ON_NITROGEN_7: "C",
    CARBONYL_OXYGEN_ON_CARBON_2: "O",
    CARBONYL_OXYGEN_ON_CARBON_6: "O",
}

# The six-membered ring of the purine: N1-C2-N3-C4-C5-C6 and back to N1.
SIX_MEMBERED_RING_BONDS: dict[str, tuple[int, int]] = {
    "N1-C2": (NITROGEN_1, CARBON_2),
    "C2-N3": (CARBON_2, NITROGEN_3),
    "N3-C4": (NITROGEN_3, CARBON_4),
    "C4-C5": (CARBON_4, CARBON_5),
    "C5-C6": (CARBON_5, CARBON_6),
    "C6-N1": (CARBON_6, NITROGEN_1),
}

# The five-membered ring: C4-C5-N7-C8-N9 and back to C4. It shares the C4-C5
# bond with the six-membered ring, which is what "fused bicyclic" means.
FIVE_MEMBERED_RING_BONDS: dict[str, tuple[int, int]] = {
    "C5-N7": (CARBON_5, NITROGEN_7),
    "N7-C8": (NITROGEN_7, CARBON_8),
    "C8-N9": (CARBON_8, NITROGEN_9),
    "N9-C4": (NITROGEN_9, CARBON_4),
}

# The side chains: three methyls on N1, N3 and N7 (caffeine is the
# 1,3,7-trimethyl derivative of xanthine) and the two carbonyl oxygens on
# C2 and C6 (xanthine is the 2,6-dioxo purine).
SIDE_CHAIN_BONDS: dict[str, tuple[int, int]] = {
    "N1-CH3": (NITROGEN_1, METHYL_CARBON_ON_NITROGEN_1),
    "N3-CH3": (NITROGEN_3, METHYL_CARBON_ON_NITROGEN_3),
    "N7-CH3": (NITROGEN_7, METHYL_CARBON_ON_NITROGEN_7),
    "C2=O": (CARBON_2, CARBONYL_OXYGEN_ON_CARBON_2),
    "C6=O": (CARBON_6, CARBONYL_OXYGEN_ON_CARBON_6),
}

RING_BONDS: dict[str, tuple[int, int]] = {**SIX_MEMBERED_RING_BONDS, **FIVE_MEMBERED_RING_BONDS}
ALL_BONDS: dict[str, tuple[int, int]] = {**RING_BONDS, **SIDE_CHAIN_BONDS}


def caffeine() -> MultiGraph[str]:
    """The heavy-atom graph of caffeine: fourteen atoms, fifteen bonds, no hydrogens."""
    return MultiGraph(list(ELEMENT_OF_ATOM), ALL_BONDS)


def ring_atoms_found_by_the_library() -> frozenset[int]:
    """Atoms that lie on a ring, taken from the library's fundamental cycle basis.

    Every bond that lies on some cycle appears in some fundamental cycle, so the
    atoms of the basis cycles are exactly the ring atoms. Nothing here is
    hardcoded from knowledge of the molecule.
    """
    molecule = caffeine()
    cycles = pygraphc.cycle_basis(list(molecule.nodes), list(molecule.endpoints.values()))
    return frozenset(atom for cycle in cycles for atom in cycle)


def test_the_heavy_atom_graph_of_caffeine_has_two_rings() -> None:
    """A chemist who has drawn caffeine by hand -- a purine core with three
    N-methyls and two carbonyls -- wants to know that the structure she typed in
    is really caffeine before she reasons about it, and the cheapest check is
    the atom tally against the molecular formula C8H10N4O2 plus the ring count.
    The framework does no chemistry here; ``MultiGraph`` only refuses a bond
    whose end is not an atom, and the ring count comes from the library's
    ``cycle_basis``, which is the graph-theoretic form of "how many rings does
    this molecule have" (the cycle rank, bonds minus atoms plus fragments). What
    a chemist would otherwise do is count on the drawing. The test establishes
    that the graph built below is caffeine's heavy-atom skeleton -- eight
    carbons, four nitrogens, two oxygens, fifteen bonds, one connected molecule
    -- and that it has a cycle rank of two, which is what makes it a purine and
    not, say, a single ring with a tail. A reader can check every count against
    a structure drawing of caffeine.
    """
    # --- Input ---
    molecule = caffeine()
    atom_count = len(molecule.nodes)
    bond_count = len(molecule.endpoints)
    fragments = list(pygraphc.connected_components(list(molecule.nodes), list(molecule.endpoints.values())))

    # --- Assert ---
    # C8H10N4O2, heavy atoms only: eight carbons, four nitrogens, two oxygens.
    elements = [ELEMENT_OF_ATOM[atom] for atom in molecule.nodes]
    assert elements.count("C") == 8
    assert elements.count("N") == 4
    assert elements.count("O") == 2
    assert atom_count == 14
    assert bond_count == 15
    # One molecule, not a mixture: every atom is bonded into the same fragment.
    assert len(fragments) == 1

    cycles = pygraphc.cycle_basis(list(molecule.nodes), list(molecule.endpoints.values()))
    # A purine is bicyclic: two independent rings.
    assert len(cycles) == 2
    # The ring count is the cycle rank, bonds minus atoms plus fragments.
    assert bond_count - atom_count + len(fragments) == 2
    # The two rings share a bond, so the smaller and larger ring together span
    # nine atoms rather than eleven.
    assert len(set(SIX_MEMBERED_RING_BONDS) | set(FIVE_MEMBERED_RING_BONDS)) == 10
    assert len({atom for bond in RING_BONDS.values() for atom in bond}) == 9


def test_the_ring_system_is_nine_atoms_and_the_side_chain_bonds_are_the_only_bridges() -> None:
    """Before a scaffold can be extracted, someone has to decide which atoms are
    ring atoms and which are side chain. A chemist reads that off the drawing; a
    pipeline has to compute it, and getting it wrong is how scaffold
    decompositions go wrong. ``cycle_basis`` gives the ring atoms and ``bridges``
    gives the bonds whose removal breaks the molecule in two -- which for a
    molecule with one ring system are exactly the side-chain bonds, because a
    bond inside a ring always has a way round. The test establishes the two
    chemical facts the next test depends on, from the structure rather than from
    a written-down list: caffeine's purine ring system is the nine atoms
    N1-C2-N3-C4-C5-C6-N7-C8-N9, and its five acyclic bonds are precisely the
    three N-methyl bonds and the two carbonyl bonds. A reader can confirm both
    on a drawing: the ring atoms are the ones on the two fused rings, and the
    five bonds that stick out are the ones that can be cut without opening a
    ring.
    """
    # --- Input ---
    molecule = caffeine()
    ring_atoms = ring_atoms_found_by_the_library()
    acyclic_bonds = pygraphc.bridges(list(molecule.nodes), list(molecule.endpoints.values()))

    # --- Assert ---
    # The purine ring system is nine atoms, and they are the atoms of the ring bonds.
    assert len(ring_atoms) == 9
    assert ring_atoms == frozenset(atom for bond in RING_BONDS.values() for atom in bond)
    # No side-chain atom is a ring atom.
    side_chain_atoms = {side_atom for _ring_atom, side_atom in SIDE_CHAIN_BONDS.values()}
    assert len(side_chain_atoms) == 5
    assert not (ring_atoms & side_chain_atoms)
    # The only bonds that can be cut without opening a ring are the five
    # side-chain bonds: the three N-methyls and the two carbonyls.
    cut_pairs = {frozenset(pair) for pair in acyclic_bonds}
    assert cut_pairs == {frozenset(pair) for pair in SIDE_CHAIN_BONDS.values()}
    assert len(cut_pairs) == 5


def test_stripping_the_side_chains_leaves_the_purine_scaffold() -> None:
    """The Bemis-Murcko decomposition of a drug molecule asks: what is left of
    caffeine when the decoration is removed and the ring system kept? The answer
    for caffeine is the bare purine, because the three methyls and the two
    carbonyl oxygens are side chains and there is only one ring system, so there
    are no linkers either. Removing a side chain is exactly the pendant move of
    ``reduce``: an atom with a single bond that is not one of the atoms we
    insisted on keeping is deleted together with its bond, repeatedly, so a
    longer side chain would peel off one atom at a time without any extra code;
    marking the ring atoms as terminals is what stops the peeling at the ring,
    and ``fold_leaves`` keeps the record of which ring atom each removed atom
    sat on, which is the substitution pattern a chemist would want back. What
    the framework does not do is decide what counts as a ring atom -- that came
    from the previous test's ``cycle_basis`` call -- nor infer that no linker
    exists. The test establishes that the residual graph is the purine ring
    system, nine atoms and ten bonds with the ring bonds unchanged, that the
    five side-chain atoms are gone, and that each is recorded against the ring
    atom it was bonded to. A reader can check the result against any picture of
    the purine scaffold and the folding against the five side-chain bonds above.
    """
    # --- Input ---
    molecule = caffeine()
    ring_atoms = ring_atoms_found_by_the_library()
    scaffold = reduce(molecule, terminals=ring_atoms, fold_leaves=True)

    # --- Assert ---
    # What remains is the purine ring system: nine atoms, ten bonds.
    assert frozenset(scaffold.graph.nodes) == ring_atoms
    assert len(scaffold.graph.nodes) == 9
    assert len(scaffold.graph.endpoints) == 10
    # The scaffold bonds are the original ring bonds, unmerged and still named.
    assert set(scaffold.graph.endpoints) == set(RING_BONDS)
    for bond_id, pair in RING_BONDS.items():
        assert scaffold.graph.endpoints[bond_id] == pair
        # No ring bond was merged with another: each stands for itself.
        assert scaffold.provenance[bond_id] == Leaf(bond_id)

    # Every side-chain atom left the graph, and no side-chain bond survived.
    side_chain_atoms = {side_atom for _ring_atom, side_atom in SIDE_CHAIN_BONDS.values()}
    assert not (side_chain_atoms & set(scaffold.graph.nodes))
    assert not (set(SIDE_CHAIN_BONDS) & set(scaffold.graph.endpoints))

    # Each removed atom is recorded on the ring atom it was bonded to: the three
    # methyls on N1, N3 and N7, the two oxygens on C2 and C6.
    expected_folding = {ring_atom: [] for ring_atom in sorted(ring_atoms)}
    for ring_atom, side_atom in SIDE_CHAIN_BONDS.values():
        expected_folding[ring_atom].append(side_atom)
    assert {atom: sorted(folded) for atom, folded in scaffold.folded_nodes.items()} == expected_folding
    # Exactly five atoms were stripped, one per side-chain bond.
    assert sum(len(folded) for folded in scaffold.folded_nodes.values()) == 5
    # Stripping the scaffold again removes nothing: the decomposition is done.
    again = reduce(scaffold.graph, terminals=ring_atoms, fold_leaves=True)
    assert sorted(again.graph.nodes) == sorted(scaffold.graph.nodes)
    assert set(again.graph.endpoints) == set(scaffold.graph.endpoints)
