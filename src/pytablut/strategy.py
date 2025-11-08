import random
from enum import Enum, IntEnum
from typing import overload

import numpy as np
import sharklog

from pytablut.rules import AshtonServerGameState, Role

_logger = sharklog.getLogger()

class Strategy(Enum):

    HUMAN = "HUMAN"
    RANDOM = "RANDOM"
    MINIMAX = "MINIMAX"


class Checker(IntEnum):
    EMPTY = 0
    WHITE_SOLDIER = 1
    BLACK_SOLDIER = 2
    WHITE_KING = 4


class Cell(IntEnum):
    NORMAL = 0
    CASTLE = 1
    CAMP = 2
    ESCAPE = 4


ASHTON_MAP = np.array([
    [Cell.NORMAL, Cell.ESCAPE, Cell.ESCAPE, Cell.CAMP,   Cell.CAMP,   Cell.CAMP,   Cell.ESCAPE, Cell.ESCAPE, Cell.NORMAL],
    [Cell.ESCAPE, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.CAMP,   Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.ESCAPE],
    [Cell.ESCAPE, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.ESCAPE],
    [Cell.CAMP,   Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.CAMP],
    [Cell.CAMP,   Cell.CAMP,   Cell.NORMAL, Cell.NORMAL, Cell.CASTLE, Cell.NORMAL, Cell.NORMAL, Cell.CAMP,   Cell.CAMP],
    [Cell.CAMP,   Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.CAMP],
    [Cell.ESCAPE, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.ESCAPE],
    [Cell.ESCAPE, Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.CAMP,   Cell.NORMAL, Cell.NORMAL, Cell.NORMAL, Cell.ESCAPE],
    [Cell.NORMAL, Cell.ESCAPE, Cell.ESCAPE, Cell.CAMP,   Cell.CAMP,   Cell.CAMP,   Cell.ESCAPE, Cell.ESCAPE, Cell.NORMAL]
]) # fmt: skip


ASHTON_MAP_SHAPE = ASHTON_MAP.shape


def position2index(position: str) -> tuple[int, int]:
    """Convert a board position in Ashton format (e.g., 'e5') to row and column indices.

    Args:
        position (str): The board position in Ashton format.

    Returns:
        tuple[int, int]: A tuple containing the row and column indices.
    """
    column_char = position[0].lower()
    row_char = position[1]

    column_index = ord(column_char) - ord("a")
    row_index = int(row_char) - 1

    return row_index, column_index

@overload
def index2position(index: tuple[int, int]) -> str:
    """Convert a (row, column) index to a board position in Ashton format (e.g., 'e5').

    Args:
        index (tuple[int, int]): The (row, column) index.

    Returns:
        str: The board position in Ashton format.
    """
    ...


@overload
def index2position(row: int, column: int) -> str:
    """Convert row and column indices to a board position in Ashton format (e.g., 'e5').

    Args:
        row (int): The row index (0-8).
        column (int): The column index (0-8).

    Returns:
        str: The board position in Ashton format.
    """
    ...


def index2position(*args):
    """Convert row and column indices to a board position in Ashton format (e.g., 'e5').

    Can be called with either:
    - A single tuple: index2position((row, col))
    - Two integers: index2position(row, col)

    Args:
        *args: Either a single tuple[int, int] or two integers (row, column)

    Returns:
        str: The board position in Ashton format.
    """
    if len(args) == 1 and isinstance(args[0], tuple):
        row, column = args[0]
    elif len(args) == 2:
        row, column = args
    else:
        raise TypeError(f"index2position expects either a tuple or two integers, got {args}")

    column_char = chr(column + ord("a"))
    row_char = str(row + 1)

    return f"{column_char}{row_char}"


def AshtonServerGameState2numpy(game_state: AshtonServerGameState) -> np.ndarray:
    """Convert AshtonServerGameState to a numpy array representation.

    Args:
        game_state (AshtonServerGameState): The game state to convert.

    Returns:
        np.ndarray: A 9x9 numpy array representing the board.
    """
    board_array = np.zeros((9, 9), dtype=np.int8)

    for r in range(9):
        for c in range(9):
            cell_value = game_state.board[r][c]
            if cell_value == "EMPTY" or cell_value == "THRONE":
                board_array[r, c] = Checker.EMPTY
            elif cell_value == "WHITE":
                board_array[r, c] = Checker.WHITE_SOLDIER
            elif cell_value == "BLACK":
                board_array[r, c] = Checker.BLACK_SOLDIER
            elif cell_value == "KING":
                board_array[r, c] = Checker.WHITE_KING
            else:
                raise ValueError(f"Unknown cell value: {cell_value}")

    return board_array


def numpy2AshtonServerGameState(
    board_array: np.ndarray, turn: str
) -> AshtonServerGameState:
    """Convert a numpy array representation of the board to AshtonServerGameState.

    Args:
        board_array (np.ndarray): A 9x9 numpy array representing the board.
        turn (str): The current turn ("WHITE" or "BLACK").

    Returns:
        AshtonServerGameState: The converted game state.
    """
    board_list = []

    for r in range(9):
        row_list = []
        for c in range(9):
            cell_value = board_array[r, c]
            if cell_value == Checker.EMPTY:
                row_list.append("EMPTY")
            elif cell_value == Checker.WHITE_SOLDIER:
                row_list.append("WHITE")
            elif cell_value == Checker.BLACK_SOLDIER:
                row_list.append("BLACK")
            elif cell_value == Checker.WHITE_KING:
                row_list.append("KING")
            else:
                raise ValueError(f"Unknown cell value: {cell_value}")
        board_list.append(row_list)

    if board_array[4, 4] == Checker.EMPTY:
        board_list[4][4] = "THRONE"

    return AshtonServerGameState(board=board_list, turn=turn)


def get_available_moves(
    board: np.ndarray, checker_index: tuple[int, int]
) -> list[tuple[int, int]]:
    """Get all available moves for a checker at the given position.

    Args:
        board (np.ndarray): The current board state as a numpy array.
        checker_index (tuple[int, int]): The (row, column) index of the checker.

    Returns:
        list[tuple[int, int]]: A list of (row, column) indices representing available moves.
    """
    available_moves = []
    row, col = checker_index
    checker_type = board[row, col]
    checker_in_camp = ASHTON_MAP[row, col] == Cell.CAMP

    if checker_type == Checker.EMPTY:
        return available_moves  # No moves for an empty cell

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    if checker_type == Checker.WHITE_SOLDIER:
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while (
                0 <= r < 9
                and 0 <= c < 9
                and board[r, c] == Checker.EMPTY
                and (ASHTON_MAP[r, c] == Cell.NORMAL or ASHTON_MAP[r, c] == Cell.ESCAPE)
            ):
                available_moves.append((r, c))
                r += dr
                c += dc
    elif checker_type == Checker.WHITE_KING:
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while (
                0 <= r < 9
                and 0 <= c < 9
                and board[r, c] == Checker.EMPTY
                and ASHTON_MAP[r, c] != Cell.CAMP
                and ASHTON_MAP[r, c] != Cell.CASTLE
            ):
                available_moves.append((r, c))
                r += dr
                c += dc
    elif checker_type == Checker.BLACK_SOLDIER:
        if checker_in_camp:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while (
                    0 <= r < 9
                    and 0 <= c < 9
                    and board[r, c] == Checker.EMPTY
                    and (
                        (
                            ASHTON_MAP[r, c] != Cell.CAMP
                            and ASHTON_MAP[r, c] != Cell.CASTLE
                        )
                        or (
                            ASHTON_MAP[r, c] == Cell.CAMP
                            and abs(r - row) < 3
                            and abs(c - col) < 3
                        )
                    )
                ):
                    available_moves.append((r, c))
                    r += dr
                    c += dc
        else:
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while (
                    0 <= r < 9
                    and 0 <= c < 9
                    and board[r, c] == Checker.EMPTY
                    and (
                        ASHTON_MAP[r, c] == Cell.NORMAL
                        or ASHTON_MAP[r, c] == Cell.ESCAPE
                    )
                ):
                    available_moves.append((r, c))
                    r += dr
                    c += dc

    return available_moves


def evaluate_random_move(
    board: np.ndarray, role: Role
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Evaluate a random move for a checker of the given type.

    Args:
        board (np.ndarray): The current board state as a numpy array.
        role (Role): The role of the player (e.g., WHITE, BLACK).

    Returns:
        tuple[tuple[int, int], tuple[int, int]] | None: A tuple containing the from and to indices of the move,
        or None if no moves are available.
    """

    if role == Role.WHITE:
        checker_positions = np.argwhere((board == Checker.WHITE_SOLDIER) | (board == Checker.WHITE_KING))
    else:
        checker_positions = np.argwhere(board == Checker.BLACK_SOLDIER)

    _logger.debug(f"Available checker positions for role {role.name}: {checker_positions}")
    while checker_positions.size > 0:
        select_checker_position = tuple(random.choice(checker_positions.tolist()))
        _logger.debug(f"Selected checker at position: {select_checker_position}")
        available_moves = get_available_moves(board, select_checker_position)
        _logger.debug(f"Available moves for selected checker: {available_moves}")
        if not available_moves:
            mask = (checker_positions == select_checker_position).all(axis=1)
            checker_positions = checker_positions[~mask]
            _logger.debug(f"Current available checker positions: {checker_positions}")
            continue
        select_move = random.choice(available_moves)
        _logger.debug(f"Selected move: {select_move}")
        return (select_checker_position, select_move)

    return None
