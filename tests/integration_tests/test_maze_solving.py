"""Integration case: solving a perfect maze (recreational puzzles, robot path planning).

Field: maze theory and the cell-by-cell path planning a floor robot does. The facts are
the standard ones about perfect mazes -- a maze with no loops and no closed-off
cells, which is to say a spanning tree of its cells, so any two of them are
joined by exactly one corridor. The maze used here is drawn in full in
``MAZE_DRAWING`` below and the cell graph is derived from that drawing by
``parse_maze`` of ``maze_grid``, the grid geometry this case shares with the
braided maze of ``test_braided_maze``, so the picture and the graph cannot
drift apart: a reader checks the picture, and the code cannot disagree with it.

What this case exercises: the pendant and series moves of ``reduce`` down to a
single residual edge, the ``paths`` fold over the resulting provenance chain,
and ``scenario`` as a blocked-passage query. On a tree with two terminals the
reduction is a solver: dead ends fall away as pendants, corridors merge in
series, and the one edge that survives carries the unique route.
"""

import pytest
from maze_grid import COLUMNS, ROWS, cell_id, parse_maze, passage_id

from pygraphc.reduction import Partition, leaves, paths, reduce, scenario, tree_records

pytestmark = pytest.mark.integration

# A ten by ten perfect maze. '#' is wall, a space is open. A cell sits at every
# odd row and odd column; two neighbouring cells have a passage between them
# when the wall square between their two centres is open. 'S' is the entrance,
# cell (row 0, column 0); 'E' is the exit, cell (row 9, column 9).
#
# #####################
# #S  #       #       #
# ### # # ### # ##### #
# #   # # # # #     # #
# # ### # # # # ##### #
# # #   # # # # #     #
# # # ### # # ### ### #
# # # #   # #   # # # #
# # ### ### ### # # # #
# #     #     # # # # #
# ####### ### # # # # #
# #       #   #   # # #
# # ### ########### # #
# # #   #           # #
# # # ### ####### ### #
# # # #   #   #   #   #
# ### # ##### # # # ###
# #   #     # # # # # #
# # ####### # # ### # #
# #           #      E#
# #####################
MAZE_DRAWING = """\
#####################
#S  #       #       #
### # # ### # ##### #
#   # # # # #     # #
# ### # # # # ##### #
# #   # # # # #     #
# # ### # # ### ### #
# # #   # #   # # # #
# ### ### ### # # # #
#     #     # # # # #
####### ### # # # # #
#       #   #   # # #
# ### ########### # #
# #   #           # #
# # ### ####### ### #
# # #   #   #   #   #
### # ##### # # # ###
#   #     # # # # # #
# ####### # # ### # #
#           #      E#
#####################"""

ENTRANCE_CELL = (0, 0)
EXIT_CELL = (9, 9)

# The one route from S to E, read off the drawing square by square. Thirty-seven
# cells means thirty-six steps.
SOLUTION_CELLS = [
    (0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2),
    (3, 2), (3, 3), (2, 3), (1, 3), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5),
    (3, 5), (3, 6), (4, 6), (5, 6), (5, 7), (4, 7), (3, 7), (2, 7), (2, 8),
    (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (7, 8), (8, 8), (9, 8),
    (9, 9),
]  # fmt: skip

# A passage well away from the route: the bottom-left corner cell and its
# neighbour to the right. It lies inside a side region of twenty cells that a
# walker can only enter and leave again -- none of its cells is on the route,
# and it holds three of the maze's dead ends.
DEAD_END_PASSAGE_CELLS = ((9, 0), (9, 1))


def solution_passages() -> list[str]:
    """The passages of the route read off the drawing, step by step."""
    return [passage_id(here, there) for here, there in zip(SOLUTION_CELLS, SOLUTION_CELLS[1:], strict=False)]


def test_the_drawing_is_a_perfect_maze() -> None:
    """Someone who has drawn a maze on paper, or a program that generated one,
    wants to know it is a proper maze before handing it to a solver: every cell
    reachable, and no loops, so there is never more than one way round and never
    a walled-off room. That is the definition of a perfect maze, and in graph
    terms it is the definition of a spanning tree -- connected, and with exactly
    one fewer passage than there are cells. ``Partition.from_components`` over
    all passages answers the reachability half in one call, and the passage
    count answers the loop half, because a connected graph with cells-minus-one
    edges cannot contain a cycle. The test establishes that the drawing above is
    a perfect maze on a hundred cells with ninety-nine passages, and a reader
    can check the count by walking the picture, or check the weaker and easier
    claim that no square of four open cells appears anywhere in it.
    """
    # --- Input ---
    maze = parse_maze(MAZE_DRAWING)
    reachable = Partition.from_components(maze, active=frozenset(maze.endpoints))

    # --- Assert ---
    assert len(maze.nodes) == ROWS * COLUMNS == 100  # ten by ten cells
    # Every cell can be reached from every other: one region, no walled-off room.
    assert len(reachable.blocks()) == 1
    assert reachable.block_of[cell_id(ENTRANCE_CELL)] == reachable.block_of[cell_id(EXIT_CELL)]
    # A connected maze with cells-minus-one passages has no loop anywhere, so
    # every pair of cells is joined by exactly one corridor.
    assert len(maze.endpoints) == len(maze.nodes) - 1 == 99
    # The drawing and the read-off route agree on every step of the route.
    assert set(solution_passages()) <= set(maze.endpoints)
    assert len(solution_passages()) == 36


def test_reducing_the_maze_to_one_edge_solves_it() -> None:
    """A robot placed at the entrance has to get to the exit and would rather
    not wander into dead ends. The two classic hand methods are exactly the two
    moves the reduction makes: filling in every dead end until none is left
    (the pendant move -- a cell with one passage that is neither entrance nor
    exit cannot be on the route), and treating a stretch of corridor with no
    choices as a single step (the series move). Because a perfect maze is a
    tree, running both to exhaustion with the entrance and the exit as the only
    cells worth keeping leaves precisely one edge, and its provenance chain is
    the route in order; ``paths`` over that chain returns the one and only route
    because a series chain with no parallel branch has exactly one path through
    it. A caller writing this by hand would implement dead-end filling and
    corridor collapsing themselves and then still need somewhere to keep the
    original passages a merged step stands for. The test establishes that the
    route the reduction finds is the route in the picture: thirty-six steps
    through the thirty-seven cells listed above, in that order. A reader can
    trace it on the drawing with a finger.
    """
    # --- Input ---
    maze = parse_maze(MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    solved = reduce(maze, terminals=frozenset({entrance, exit_cell}))

    # --- Assert ---
    # Every dead end has been filled and every corridor collapsed: only the
    # entrance, the exit and the one route between them are left.
    assert sorted(solved.graph.nodes) == sorted([entrance, exit_cell])
    assert len(solved.graph.endpoints) == 1
    (route,) = solved.graph.endpoints
    assert set(solved.graph.endpoints[route]) == {entrance, exit_cell}

    route_tree = solved.provenance[route]
    kinds = {kind for _index, kind, _children, _interior, _leaf in tree_records(route_tree)}
    # The route is a chain of steps one after another, with no fork anywhere:
    # in a maze with no loops there is never an alternative way round.
    assert kinds == {"leaf", "series"}
    assert tree_records(route_tree)[-1][1] == "series"

    # The route is the corridor from the drawing, passage for passage.
    assert leaves(route_tree) == frozenset(solution_passages())
    # Thirty-six passages, which is thirty-six steps from S to E.
    assert len(leaves(route_tree)) == 36
    # There is exactly one way through, and it is that corridor.
    assert paths(route_tree) == {frozenset(solution_passages())}
    # The cells the route runs through are the cells listed off the picture.
    cells_on_route = {cell for passage in leaves(route_tree) for cell in maze.endpoints[passage]}
    assert cells_on_route == {cell_id(cell) for cell in SOLUTION_CELLS}
    assert len(cells_on_route) == 37


def test_blocking_a_passage_on_the_route_strands_the_robot() -> None:
    """A floor robot finds a passage blocked -- a closed door, a dropped box --
    and has to decide whether to re-plan or give up. In a maze with no loops
    there is no way round anything, so the answer turns entirely on whether the
    blocked passage was on the route: block one of the thirty-six route passages
    and the entrance and the exit end up in different parts of the maze, with no
    route at all; block a passage inside a dead-end branch and the route is
    untouched. ``scenario`` is that query -- it masks the blocked passage and
    re-derives which cells are still mutually reachable -- and the thing worth
    noting is that it answers the reachability question directly on the full
    maze, so the collapsed route from the previous test is not what is consulted
    here; a caller who wanted to answer the question from the provenance chain
    instead would have to check membership in the chain's leaves themselves. The
    test establishes both halves of the real decision, and a reader can confirm
    each on the drawing: cutting the corridor between S and E separates them,
    and cutting a passage in the bottom-left dead end separates only that dead
    end from everything else.
    """
    # --- Input ---
    maze = parse_maze(MAZE_DRAWING)
    entrance = cell_id(ENTRANCE_CELL)
    exit_cell = cell_id(EXIT_CELL)
    all_passages = frozenset(maze.endpoints)
    blocked_on_route = solution_passages()[17]
    blocked_in_dead_end = passage_id(*DEAD_END_PASSAGE_CELLS)

    # --- Assert ---
    assert blocked_in_dead_end not in set(solution_passages())  # not on the route

    on_route = scenario(maze, active=all_passages, removed=frozenset({blocked_on_route}))
    # With a route passage blocked the maze falls into two parts and the
    # entrance and the exit are in different ones: the robot cannot get there.
    assert len(on_route.blocks()) == 2
    assert on_route.block_of[entrance] != on_route.block_of[exit_cell]

    in_dead_end = scenario(maze, active=all_passages, removed=frozenset({blocked_in_dead_end}))
    # Blocking a passage inside a dead end also splits the maze -- in a maze
    # with no loops every passage does -- but the part that is cut off is the
    # dead end, so the entrance and the exit stay connected and the route holds.
    assert len(in_dead_end.blocks()) == 2
    assert in_dead_end.block_of[entrance] == in_dead_end.block_of[exit_cell]
    assert in_dead_end.block_of[cell_id(DEAD_END_PASSAGE_CELLS[0])] != in_dead_end.block_of[entrance]
    # Not one cell of the route ends up on the cut-off side.
    severed_side = {cell for cell, block in in_dead_end.block_of.items() if block != in_dead_end.block_of[entrance]}
    assert not (severed_side & {cell_id(cell) for cell in SOLUTION_CELLS})

    # Nothing blocked: the whole maze is one region again.
    assert len(scenario(maze, active=all_passages, removed=frozenset()).blocks()) == 1
