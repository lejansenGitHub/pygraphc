"""Grid geometry shared by the maze integration cases: cell ids, passage ids, and the drawing parser.

Both maze cases use the same ten by ten grid of cells and the same ASCII
convention for drawing it, so the parser lives here once instead of twice: the
perfect maze of ``test_maze_solving`` and the braided maze of
``test_braided_maze`` are then guaranteed to be read the same way.
"""

from pygraphc.reduction import MultiGraph

ROWS = 10
COLUMNS = 10


def cell_id(cell: tuple[int, int]) -> int:
    """Node id of a cell, row-major, so cell (row, column) is row * 10 + column."""
    row, column = cell
    return row * COLUMNS + column


def passage_id(first_cell: tuple[int, int], second_cell: tuple[int, int]) -> str:
    """Edge id of the passage between two neighbouring cells, independent of direction."""
    low, high = sorted([cell_id(first_cell), cell_id(second_cell)])
    return f"{low}-{high}"


def parse_maze(drawing: str) -> MultiGraph[str]:
    """Cell graph of the drawing: one node per cell, one edge per open wall square between two cells."""
    squares = drawing.splitlines()
    endpoints: dict[str, tuple[int, int]] = {}
    for row in range(ROWS):
        for column in range(COLUMNS):
            if column + 1 < COLUMNS and squares[2 * row + 1][2 * column + 2] != "#":
                neighbour = (row, column + 1)
                endpoints[passage_id((row, column), neighbour)] = (cell_id((row, column)), cell_id(neighbour))
            if row + 1 < ROWS and squares[2 * row + 2][2 * column + 1] != "#":
                neighbour = (row + 1, column)
                endpoints[passage_id((row, column), neighbour)] = (cell_id((row, column)), cell_id(neighbour))
    return MultiGraph([cell_id((row, column)) for row in range(ROWS) for column in range(COLUMNS)], endpoints)
