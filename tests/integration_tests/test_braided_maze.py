"""Integration case: a braided maze (recreational puzzles, game level design, robot path planning).

Field: maze design. Braiding is the standard technique for taking a maze that
has exactly one route and knocking a few extra passages through its walls, so
that some cells can be reached in more than one way. Designers braid to remove
dead ends, so a walker never has to back out of a cul-de-sac, and to give a
solver or a game character a choice of routes and therefore a detour when one
passage is shut. The maze here is the perfect maze of ``test_maze_solving``
with three walls knocked out -- the same layout, braided -- so the two cases
can be read side by side; the drawing is given in full in
``BRAIDED_MAZE_DRAWING`` and the cell graph comes from that drawing through
``parse_maze`` of ``maze_geometry``, the same parser the perfect maze uses, so
picture and graph cannot drift apart.

What this case adds over the perfect maze: everything that only exists once a
maze has loops. The provenance of the reduced maze is no longer a plain chain
but carries parallel nodes, ``paths`` returns eight routes instead of one, a
blocked passage no longer necessarily strands the walker, and the question
"which passages have to be walled up to make this maze unsolvable" becomes a
real minimal-cut question answered by ``minimal_toggles``. The pendant move
still does the dead-end pruning of the perfect-maze case, in the same pass, so
loops and dead ends are handled together.
"""

import pytest
from maze_geometry import COLUMNS, ROWS, cell_id, parse_maze, passage_id

from pygraphc.reduction import (
    MultiGraph,
    Parallel,
    Partition,
    SPTree,
    leaves,
    minimal_toggles,
    paths,
    reduce,
    scenario,
    tree_records,
)

pytestmark = pytest.mark.integration

# The perfect maze of test_maze_solving with three wall squares knocked out.
# '#' is wall, a space is open. A cell sits at every odd row and odd column;
# two neighbouring cells have a passage between them when the wall square
# between their two centres is open. 'S' is the entrance, cell (row 0,
# column 0); 'E' is the exit, cell (row 9, column 9).
#
# The copy below marks the three knocked-out squares with '+' so they can be
# found in the picture; in the constant itself they are ordinary open squares.
#
# #####################
# #S  #       #       #
# ### # # ### # ##### #
# #   # + # # #     # #
# # ### # # # # ##### #
# # #   # # # # #     #
# # # ### # # ### ### #
# # # #   # #   # # # #
# # ### ### ### # # # #
# #     #     # + # # #
# ####### ### # # # # #
# #       #   #   # # #
# # ### ########### # #
# # #   #           # #
# # # ### ####### ### #
# # # #   #   #   #   #
# ### # ##### # # # ###
# #   #     # # # # + #
# # ####### # # ### # #
# #           #      E#
# #####################
BRAIDED_MAZE_DRAWING = """\
#####################
#S  #       #       #
### # # ### # ##### #
#   #   # # #     # #
# ### # # # # ##### #
# #   # # # # #     #
# # ### # # ### ### #
# # #   # #   # # # #
# ### ### ### # # # #
#     #     #   # # #
####### ### # # # # #
#       #   #   # # #
# ### ########### # #
# #   #           # #
# # ### ####### ### #
# # #   #   #   #   #
### # ##### # # # ###
#   #     # # # #   #
# ####### # # ### # #
#           #      E#
#####################"""

ENTRANCE_CELL = (0, 0)
EXIT_CELL = (9, 9)
ENTRANCE_DOOR_CELLS = ((0, 0), (0, 1))

# The three passages knocked through, each closing a ring of four cells:
#   (1, 2)-(1, 3) closes the ring (0, 2), (0, 3), (1, 3), (1, 2) in the top left,
#   (4, 6)-(4, 7) closes the ring (4, 6), (5, 6), (5, 7), (4, 7) in the middle,
#   (8, 8)-(8, 9) closes the ring (8, 8), (9, 8), (9, 9), (8, 9) at the exit and
#   in doing so turns the former dead end (8, 9) into a through cell.
BRAID_PASSAGE_CELLS = (((1, 2), (1, 3)), ((4, 6), (4, 7)), ((8, 8), (8, 9)))

# A passage on the middle ring, and one of the two the perfect maze's single
# route used there: (4, 6) down to (5, 6). The ring's other side is still open
# once it is blocked.
LOOP_PASSAGE_CELLS = ((4, 6), (5, 6))

# A passage in the long corridor between the top-left ring and the middle ring,
# well away from all three rings: (3, 5) across to (3, 6). Every route has to
# use it, so walling it up leaves no route at all.
CHOKE_PASSAGE_CELLS = ((3, 5), (3, 6))


def walk_cells(maze: MultiGraph[str], route_passages: frozenset[str]) -> list[int] | None:
    """Cells of a route in walking order from entrance to exit, or None if the passages are not such a walk."""
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    steps_at: dict[int, list[tuple[int, str]]] = {}
    for passage in route_passages:
        first_cell, second_cell = maze.endpoints[passage]
        steps_at.setdefault(first_cell, []).append((second_cell, passage))
        steps_at.setdefault(second_cell, []).append((first_cell, passage))
    if len(steps_at.get(entrance, [])) != 1 or len(steps_at.get(exit_cell, [])) != 1:
        return None
    visited_cells = [entrance]
    used_passages: set[str] = set()
    while visited_cells[-1] != exit_cell:
        onward = [step for step in steps_at[visited_cells[-1]] if step[1] not in used_passages]
        if len(onward) != 1:
            return None
        next_cell, passage = onward[0]
        used_passages.add(passage)
        visited_cells.append(next_cell)
    return visited_cells if len(used_passages) == len(route_passages) else None


def dead_end_cells(maze: MultiGraph[str]) -> frozenset[int]:
    """Cells with exactly one passage that are neither the entrance nor the exit."""
    passage_count: dict[int, int] = dict.fromkeys(maze.nodes, 0)
    for first_cell, second_cell in maze.endpoints.values():
        passage_count[first_cell] += 1
        passage_count[second_cell] += 1
    terminals = {cell_id(ENTRANCE_CELL), cell_id(EXIT_CELL)}
    return frozenset(cell for cell, count in passage_count.items() if count == 1 and cell not in terminals)


def route_tree(
    maze: MultiGraph[str],
    from_cell: tuple[int, int] = ENTRANCE_CELL,
    to_cell: tuple[int, int] = EXIT_CELL,
) -> SPTree[str]:
    """Provenance tree of the single edge the braided maze reduces to between two cells."""
    solved = reduce(maze, terminals=frozenset({cell_id(from_cell), cell_id(to_cell)}))
    (residual_passage,) = solved.graph.endpoints
    return solved.provenance[residual_passage]


def single_passage_cuts(maze: MultiGraph[str], from_cell: tuple[int, int], to_cell: tuple[int, int]) -> list[str]:
    """Every passage that on its own separates the two cells, found by blocking each one in turn."""
    from_id, to_id = cell_id(from_cell), cell_id(to_cell)
    all_passages = frozenset(maze.endpoints)
    separating = []
    for passage in sorted(all_passages):
        blocked = scenario(maze, active=all_passages, removed=frozenset({passage}))
        if blocked.block_of[from_id] != blocked.block_of[to_id]:
            separating.append(passage)
    return separating


def test_the_drawing_is_a_braided_maze_and_not_a_perfect_one() -> None:
    """A designer who has braided a maze wants to know that the braiding took
    and that nothing else broke: every cell still reachable, and now genuinely
    more ways round than before. Perfect means a spanning tree -- connected,
    with one fewer passage than cells -- so braiding shows up as passages above
    that count, and the number of independent loops in a connected maze is
    exactly passages minus cells plus one, its cycle rank.
    ``Partition.from_components`` over every passage answers reachability in one
    call and the two counts answer the rest, so the designer never has to
    enumerate loops in order to count them. The test establishes that this is a
    hundred-cell maze with a hundred and two passages and therefore three loops,
    one per knocked-out wall, and a reader can check that against the picture by
    finding the three squares marked '+' in the comment copy above and the ring
    of four open cells each one closes.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    reachable = Partition.from_components(maze, active=frozenset(maze.endpoints))
    braid_passages = {passage_id(first_cell, second_cell) for first_cell, second_cell in BRAID_PASSAGE_CELLS}

    # --- Assert ---
    assert len(maze.nodes) == ROWS * COLUMNS == 100  # ten by ten cells, as in the perfect maze
    # No walled-off room: one region, and the walker can get from S to E at all.
    assert len(reachable.blocks()) == 1
    assert reachable.block_of[cell_id(ENTRANCE_CELL)] == reachable.block_of[cell_id(EXIT_CELL)]
    # More passages than a perfect maze of this size can have, which is what
    # braiding means: the drawing is not a spanning tree of its cells.
    assert len(maze.endpoints) == 102
    assert len(maze.endpoints) > len(maze.nodes) - 1
    # Three independent loops, one per wall knocked out.
    assert len(maze.endpoints) - len(maze.nodes) + 1 == len(BRAID_PASSAGE_CELLS) == 3
    # The three knocked-through passages are all in the drawing.
    assert len(braid_passages) == 3
    assert braid_passages <= set(maze.endpoints)


def test_reducing_the_braided_maze_gives_eight_routes() -> None:
    """The walker still only wants to get from S to E, but now there is more
    than one answer and a designer wants to know how many, because that is what
    the braiding was for. The reduction does the same two things it does on a
    perfect maze -- fill in the dead ends (the pendant move) and treat a
    choiceless stretch of corridor as one step (the series move) -- and adds the
    third: where two stretches of corridor run between the same two cells, it
    records them as one step the walker may take either way round (the parallel
    move). So the maze again collapses to a single edge, but its provenance is
    no longer a plain chain: it carries one parallel node per ring, and ``paths``
    multiplies the choices out. The eight routes are the three independent
    choices in the picture: at the top-left ring, step (1, 3) to (0, 3) or go
    round through (1, 2) and (0, 2); at the middle ring, step (4, 6) to (4, 7)
    or go round through (5, 6) and (5, 7); at the exit ring, reach E through
    (9, 8) or through (8, 9). Two times two times two is eight, and because the
    first two choices cost one passage or three while the last costs two either
    way, the route lengths come out as 34, 36, 36 and 38 steps, each of them
    twice. What the framework does not hand over is the order of a route:
    ``paths`` returns each route as a set of passages, so ``walk_cells`` above
    has to walk the set back into a sequence before anyone can follow it --
    the same gap the perfect-maze case recorded. The test establishes the count,
    the lengths and that each of the eight is a real walk through the drawing,
    and a reader can check it by taking the three choices above in all their
    combinations.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    tree = route_tree(maze)
    kinds = [kind for _index, kind, _children, _interior, _leaf in tree_records(tree)]
    routes = paths(tree)

    # --- Assert ---
    # The maze still collapses to one step from S to E, but that step now holds
    # a fork for every ring: the walker may go either way round each of them.
    assert set(kinds) == {"leaf", "series", "parallel"}
    assert kinds.count("parallel") == len(BRAID_PASSAGE_CELLS) == 3
    # Eight routes: two ways round each of the three rings.
    assert len(routes) == 8
    assert sorted(len(route) for route in routes) == [34, 34, 36, 36, 36, 36, 38, 38]
    # Thirty passages are common to all eight; the rest are the three rings,
    # four passages each, of which a route walks one side or the other.
    common_passages = frozenset.intersection(*routes)
    assert len(common_passages) == 30
    assert len(frozenset().union(*routes) - common_passages) == 12
    # Every route is a walk that starts at S, ends at E and repeats no cell.
    for route in routes:
        cells_walked = walk_cells(maze, route)
        assert cells_walked is not None
        assert cells_walked[0] == entrance
        assert cells_walked[-1] == exit_cell
        assert len(cells_walked) == len(set(cells_walked)) == len(route) + 1


def test_blocking_a_passage_on_a_loop_leaves_the_maze_solvable() -> None:
    """This is the case the perfect maze could not make. There, every passage
    was a bridge, so any dropped box or shut door split the maze in two and the
    only question left was which side the walker was on. Braiding is what buys a
    detour: block a passage that lies on one of the rings and the ring's other
    side carries the walker, so the maze stays whole and stays solvable; block a
    passage that every route needs and the maze falls apart exactly as before.
    ``scenario`` is the query in both cases -- mask the blocked passage and
    re-derive which cells can still reach each other -- and it answers on the
    full maze rather than on the reduced one, so a designer can ask it about any
    passage without having reduced anything first. The two passages here are
    picked to be checkable by eye: (4, 6)-(5, 6) is on the middle ring and was
    one of the two passages the perfect maze's single route used there, so it is
    precisely a passage whose blocking used to be fatal and no longer is;
    (3, 5)-(3, 6) lies in the long corridor between the top-left ring and the
    middle ring, which no ring touches, so nothing can get round it. The test
    establishes both halves, and a reader can confirm them on the drawing by
    filling in one square at a time and trying to trace a route.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    all_passages = frozenset(maze.endpoints)
    on_a_loop = passage_id(*LOOP_PASSAGE_CELLS)
    on_every_route = passage_id(*CHOKE_PASSAGE_CELLS)
    routes = paths(route_tree(maze))

    # --- Assert ---
    # The one lies on some routes only, the other on all of them.
    assert any(on_a_loop in route for route in routes)
    assert not all(on_a_loop in route for route in routes)
    assert all(on_every_route in route for route in routes)

    with_loop_passage_blocked = scenario(maze, active=all_passages, removed=frozenset({on_a_loop}))
    # The ring's other side still carries the walker: the maze does not even
    # fall into two parts, let alone separate S from E.
    assert len(with_loop_passage_blocked.blocks()) == 1
    assert with_loop_passage_blocked.block_of[entrance] == with_loop_passage_blocked.block_of[exit_cell]

    with_choke_passage_blocked = scenario(maze, active=all_passages, removed=frozenset({on_every_route}))
    # No ring spans that corridor, so blocking it splits the maze and leaves S
    # and E on different sides: the maze is unsolvable.
    assert len(with_choke_passage_blocked.blocks()) == 2
    assert with_choke_passage_blocked.block_of[entrance] != with_choke_passage_blocked.block_of[exit_cell]

    # Nothing blocked: the whole maze is one region again.
    assert len(scenario(maze, active=all_passages, removed=frozenset()).blocks()) == 1


def test_one_passage_is_enough_when_the_entrance_has_a_single_door() -> None:
    """The question a designer actually asks about a braided maze: how much
    walling up does it take to make it unsolvable -- one passage, or does the
    braiding force several? That is a minimal cut between entrance and exit, and
    on the reduced maze it is a single call: ``minimal_toggles`` over the
    provenance tree of the one surviving edge, with every passage starting out
    walkable and the target state the one where nothing gets through any more
    (the fold's flag is True when a piece of the maze lets a walker through, so
    a walkable passage goes in as True and the target goes in as False). Here
    the answer is one passage, (0, 0)-(0, 1): the entrance cell S has exactly
    one door in the drawing, so walling that door up traps the walker in S, and
    no set smaller than a single passage exists. That is the honest lesson of
    this drawing -- three loops bought eight routes and not one bit of
    robustness, because none of the loops is anywhere near the entrance. It is
    also not the only cheapest answer: each of the thirty passages that all
    eight routes share would do just as well, and ``minimal_toggles`` returns
    one of them, the lowest-numbered, rather than all thirty, so a designer who
    wants the full list of choke points has to find it another way; the
    framework does not offer that.

    What this test does *not* establish is the parallel half of the fold. With
    a one-door entrance a single passage is always the cheapest cut, so the
    answer would be a one-element set whatever a ring cost, and the strict-subset
    check below reduces to "the empty set does not separate". The claim that a
    ring costs two is tested by
    ``test_severing_a_route_through_a_ring_costs_two_passages`` instead, on a
    terminal pair whose minimum cut really does have two members.

    The test establishes the answer and then checks it without using the
    reduction at all: blocking exactly that set with ``scenario`` separates S
    from E. A reader can check it on the drawing by looking at the cell marked S
    and counting its open sides.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    all_passages = frozenset(maze.endpoints)
    tree = route_tree(maze)
    every_passage_walkable = dict.fromkeys(leaves(tree), True)
    must_be_walled_up = minimal_toggles(tree, every_passage_walkable, target_closed=False)

    # --- Assert ---
    # Wall up the entrance's one door and no route survives.
    assert must_be_walled_up == frozenset({passage_id(*ENTRANCE_DOOR_CELLS)})
    assert len(must_be_walled_up) == 1

    # Checked independently of the reduction: that set does separate S from E.
    with_cut_blocked = scenario(maze, active=all_passages, removed=must_be_walled_up)
    assert with_cut_blocked.block_of[entrance] != with_cut_blocked.block_of[exit_cell]
    # Blocking nothing leaves them connected, which is the only strict subset a
    # one-passage cut has -- so minimality is trivial here, not demonstrated.
    with_nothing_blocked = scenario(maze, active=all_passages, removed=frozenset())
    assert with_nothing_blocked.block_of[entrance] == with_nothing_blocked.block_of[exit_cell]


# The top-left ring, the one closed by the braid passage (1, 2)-(1, 3). Taking
# two diagonally opposite cells of it as the terminals puts one arc of the ring
# on either side of every route between them, which is the situation the one-door
# entrance never produces.
RING_TERMINAL_CELLS = ((0, 3), (1, 2))
RING_ARC_VIA_TOP_LEFT = (((0, 2), (0, 3)), ((0, 2), (1, 2)))
RING_ARC_VIA_BOTTOM_RIGHT = (((0, 3), (1, 3)), ((1, 2), (1, 3)))


def test_severing_a_route_through_a_ring_costs_two_passages() -> None:
    """The half of the fold the entrance-to-exit cut cannot reach: what a ring
    costs. A series step needs all of its passages, so the cheapest way to break
    one is to break its cheapest single passage, and a one-passage answer is the
    right answer whenever any single passage separates the two cells. A ring is
    the other case -- a walker who loses one side of it simply goes round the
    other -- so both sides have to be broken and the cheapest cut has two
    members. The terminals here are (0, 3) and (1, 2), the two diagonally
    opposite cells of the ring that the braid passage (1, 2)-(1, 3) closed in the
    top left. Every way from one to the other runs round one arc of that ring or
    the other, so no single passage anywhere in the maze separates them, and the
    provenance ``reduce`` produces between them is a ``Parallel`` of the two
    arcs. The test establishes that ``minimal_toggles`` answers with two
    passages, one drawn from each arc, and it fails if the library ever answers
    with one: the size is asserted directly, and independently of the reduction
    every one of the hundred and two passages is blocked in turn with
    ``scenario`` and none of them on its own separates the pair. Then the answer
    itself is checked the same way -- blocking both does separate them, and
    blocking either one alone does not. A reader can check it on the drawing by
    finding the four open cells of the top-left ring and tracing the two ways
    round it.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    from_cell, to_cell = RING_TERMINAL_CELLS
    from_id, to_id = cell_id(from_cell), cell_id(to_cell)
    all_passages = frozenset(maze.endpoints)
    tree = route_tree(maze, from_cell, to_cell)
    every_passage_walkable = dict.fromkeys(leaves(tree), True)
    must_be_walled_up = minimal_toggles(tree, every_passage_walkable, target_closed=False)

    # --- Assert ---
    # The reduced route between the two cells is the ring: two arcs side by side.
    assert isinstance(tree, Parallel)
    assert len(tree.children) == 2
    # A ring costs two, and the library does not get away with saying one.
    assert len(must_be_walled_up) == 2
    # One passage from each arc -- breaking two of the same arc leaves the other open.
    arc_via_top_left = {passage_id(*cells) for cells in RING_ARC_VIA_TOP_LEFT}
    arc_via_bottom_right = {passage_id(*cells) for cells in RING_ARC_VIA_BOTTOM_RIGHT}
    assert len(must_be_walled_up & arc_via_top_left) == 1
    assert len(must_be_walled_up & arc_via_bottom_right) == 1
    # The exact answer, ties broken by edge id order as the fold documents.
    assert must_be_walled_up == frozenset({passage_id((0, 2), (1, 2)), passage_id((1, 2), (1, 3))})

    # Checked without the reduction: no single passage in the whole maze separates
    # the pair, so two really is the minimum and not an over-count.
    assert single_passage_cuts(maze, from_cell, to_cell) == []
    # And the pair the fold named does separate them.
    with_cut_blocked = scenario(maze, active=all_passages, removed=must_be_walled_up)
    assert with_cut_blocked.block_of[from_id] != with_cut_blocked.block_of[to_id]
    # While each strict subset -- either passage on its own -- leaves them joined.
    for left_out in must_be_walled_up:
        with_subset_blocked = scenario(maze, active=all_passages, removed=must_be_walled_up - {left_out})
        assert with_subset_blocked.block_of[from_id] == with_subset_blocked.block_of[to_id]


def test_dead_ends_are_pruned_and_kept_in_the_folded_material() -> None:
    """Braiding is usually done to get rid of dead ends, and three walls do not
    get rid of nine of them: the braided drawing still has nine cul-de-sacs, one
    fewer than the perfect maze had, because the ring at the exit swallowed the
    dead end (8, 9). A designer wants two things from a solver here: that it
    offers no route through a cul-de-sac, and that the cells it threw away are
    still accounted for, so the solver can say where the walker would have
    wandered. ``reduce`` does both in the pass that also handles the loops: the
    pendant move deletes a one-door cell that is neither entrance nor exit and,
    with ``fold_leaves`` on, records the deleted material on the cell it hung
    off, so nothing is lost -- every eliminated cell turns up exactly once,
    either folded onto a surviving cell, folded onto a cell that a later series
    move absorbed, or named as an interior cell of a series step. The test
    establishes that the nine dead ends are gone from the residual maze, that
    all nine are in the folded material, and that the folded material together
    with the entrance and the exit accounts for all hundred cells exactly once.
    A reader can check the dead-end count on the drawing by looking for open
    cells with walls on three sides.
    """
    # --- Input ---
    maze = parse_maze(BRAIDED_MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    solved = reduce(maze, terminals=frozenset({entrance, exit_cell}))
    dead_ends = dead_end_cells(maze)
    (residual_passage,) = solved.graph.endpoints
    folded_material = [
        cell for material in (*solved.folded_nodes.values(), *solved.folded_interior.values()) for cell in material
    ]
    series_interior = [
        cell
        for _index, kind, _children, interior_cells, _leaf in tree_records(solved.provenance[residual_passage])
        if kind == "series"
        for cell in interior_cells
    ]

    # --- Assert ---
    # Nine cul-de-sacs left after braiding; the ring at the exit removed one.
    assert len(dead_ends) == 9
    # None of them is offered as part of the maze any more: the residual is the
    # entrance, the exit and the one step between them.
    assert sorted(solved.graph.nodes) == sorted([entrance, exit_cell])
    assert dead_ends.isdisjoint(solved.graph.nodes)
    # Every dead end is still accounted for in the folded material.
    assert dead_ends <= frozenset(folded_material)
    # And the bookkeeping loses nothing: every cell of the maze is a terminal,
    # folded material or an interior cell of a series step, exactly once.
    accounted_for = [entrance, exit_cell, *folded_material, *series_interior]
    assert sorted(accounted_for) == sorted(maze.nodes)
    assert len(accounted_for) == len(set(accounted_for)) == 100
