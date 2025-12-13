import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum, IntEnum, IntFlag
from typing import overload

import numpy as np
import sharklog
from timevery import Timer

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


class GameStatus(IntFlag):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 4
    TERMINATED = WHITE_WIN | BLACK_WIN | DRAW


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

SPECIAL_KING_AREA = (
    (4, 4), (3, 4), (4, 3), (4, 5), (5, 4)
)

INF = float('inf')
NEG_INF = float('-inf')

BIG_NUMBER = 10000

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


def apply_move(board: np.ndarray, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> np.ndarray:
    """Apply a move to the board and return a new board state.

    Args:
        board (np.ndarray): The current board state.
        from_pos (tuple[int, int]): The starting position (row, col).
        to_pos (tuple[int, int]): The destination position (row, col).

    Returns:
        np.ndarray: A new board state with the move applied.
    """
    new_board = board.copy()
    from_r, from_c = from_pos
    to_r, to_c = to_pos

    # Move the piece
    new_board[to_r, to_c] = new_board[from_r, from_c]
    new_board[from_r, from_c] = Checker.EMPTY

    # Check for captures
    new_board = check_captures(new_board, to_pos)

    return new_board


def check_captures(board: np.ndarray, moved_to: tuple[int, int]) -> np.ndarray:
    """Check and apply captures after a move.

    Args:
        board (np.ndarray): The current board state.
        moved_to (tuple[int, int]): The position where a piece just moved.

    Returns:
        np.ndarray: The board with captures applied.
    """
    r, c = moved_to
    mover = board[r, c]

    # Determine the mover's team
    if mover == Checker.EMPTY:
        return board

    is_white_team = mover in (Checker.WHITE_SOLDIER, Checker.WHITE_KING)

    # Check all four directions for captures
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dr, dc in directions:
        target_r, target_c = r + dr, c + dc
        beyond_r, beyond_c = r + 2 * dr, c + 2 * dc

        # Check bounds
        if not (0 <= target_r < 9 and 0 <= target_c < 9):
            continue
        if not (0 <= beyond_r < 9 and 0 <= beyond_c < 9):
            continue

        target = board[target_r, target_c]
        beyond = board[beyond_r, beyond_c]

        # Skip if target is empty or on same team
        if target == Checker.EMPTY:
            continue

        target_is_white = target in (Checker.WHITE_SOLDIER, Checker.WHITE_KING)
        if is_white_team == target_is_white:
            continue

        # Special handling for king
        if target == Checker.WHITE_KING and (target_r, target_c) in SPECIAL_KING_AREA:
            # King requires capture from all 4 sides (or 3 sides + castle)
            # For simplicity, we'll only capture king if surrounded on all 4 sides
            adjacent_enemies = 0
            for kr, kc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                kr_pos, kc_pos = target_r + kr, target_c + kc
                if 0 <= kr_pos < 9 and 0 <= kc_pos < 9:
                    cell = board[kr_pos, kc_pos]
                    cell_type = ASHTON_MAP[kr_pos, kc_pos]
                    if cell == Checker.BLACK_SOLDIER or cell_type == Cell.CASTLE:
                        adjacent_enemies += 1
            if adjacent_enemies >= 4:
                board[target_r, target_c] = Checker.EMPTY
            continue

        # Check if beyond position is occupied by ally or special cell
        is_valid_capture = False

        # Ally piece
        if beyond in (Checker.WHITE_SOLDIER, Checker.WHITE_KING) and is_white_team:
            is_valid_capture = True
        elif beyond == Checker.BLACK_SOLDIER and not is_white_team:
            is_valid_capture = True

        # Special cells (castle, camp) can act as allies
        beyond_cell_type = ASHTON_MAP[beyond_r, beyond_c]
        if beyond_cell_type in (Cell.CASTLE, Cell.CAMP) and ASHTON_MAP[target_r, target_c] != Cell.CAMP:
            is_valid_capture = True

        # Apply capture
        if is_valid_capture:
            board[target_r, target_c] = Checker.EMPTY

    return board


def is_terminal_state(board: np.ndarray) -> GameStatus:
    """Check if the board is in a terminal state (game over).

    Args:
        board (np.ndarray): The current board state.

    Returns:
        GameStatus: The current game status.
    """
    # Find king position
    king_positions = np.argwhere(board == Checker.WHITE_KING)

    # 1. King captured = Black wins
    if len(king_positions) == 0:
        return GameStatus.BLACK_WIN

    king_r, king_c = king_positions[0]

    # 2. King on escape cell = White wins
    if ASHTON_MAP[king_r, king_c] == Cell.ESCAPE:
        return GameStatus.WHITE_WIN

    # Check if any white piece can move
    white_positions = np.argwhere((board == Checker.WHITE_SOLDIER) | (board == Checker.WHITE_KING))
    has_white_moves = False
    for pos in white_positions:
        if len(get_available_moves(board, tuple(pos))) > 0:
            has_white_moves = True
            break

    # Check if any black piece can move
    black_positions = np.argwhere(board == Checker.BLACK_SOLDIER)
    has_black_moves = False
    for pos in black_positions:
        if len(get_available_moves(board, tuple(pos))) > 0:
            has_black_moves = True
            break

    # 3. If a side has no moves, they lose
    if not has_white_moves and len(white_positions) > 0:
        return GameStatus.BLACK_WIN
    if not has_black_moves and len(black_positions) > 0:
        return GameStatus.WHITE_WIN

    # 4. Draw, the same state repeated twice. (Handled in Server)

    return GameStatus.ONGOING


def evaluate_board(board: np.ndarray, role: Role) -> float:
    """Evaluate the board state from the perspective of the given role.

    Args:
        board (np.ndarray): The current board state.
        role (Role): The role to evaluate for (WHITE or BLACK).

    Returns:
        float: The evaluation score. Higher is better for the given role.
               Returns +inf for winning positions, -inf for losing positions.
    """
    # Check terminal states first - use infinity for terminal states
    # This ensures winning moves are ALWAYS chosen and losing moves are NEVER chosen
    game_status = is_terminal_state(board)
    if game_status & GameStatus.TERMINATED:
        if role == Role.WHITE:
            return INF if game_status == GameStatus.WHITE_WIN else NEG_INF
        else:
            return INF if game_status == GameStatus.BLACK_WIN else NEG_INF

    score = 0.0

    WHITE_SOLDER_NUM_FACTOR = 160 / 8
    BLACK_SOLDER_NUM_FACTOR = 160 / 16

    DIRECT_ESCAPE_PATH_NUM_FACTOR = 20

    WHITE_POSITION_SCORE_FACTOR = 15 / 8
    BLACK_POSITION_SCORE_FACTOR = 15 / 8

    WHITE_MAP_WEIGHT = np.array([
        [0.0, 0.4, 0.3, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0],
        [0.4, 0.4, 0.5, 0.2, 0.0, 0.2, 0.5, 0.4, 0.4],
        [0.3, 0.5, 0.4, 0.5, 0.3, 0.5, 0.4, 0.5, 0.3],
        [0.0, 0.2, 0.5, 0.4, 0.3, 0.4, 0.5, 0.2, 0.0],
        [0.0, 0.0, 0.3, 0.3, 0.0, 0.3, 0.3, 0.0, 0.0],
        [0.0, 0.2, 0.5, 0.4, 0.3, 0.4, 0.5, 0.2, 0.0],
        [0.3, 0.5, 0.4, 0.5, 0.3, 0.5, 0.4, 0.5, 0.3],
        [0.4, 0.4, 0.5, 0.2, 0.0, 0.2, 0.5, 0.4, 0.4],
        [0.0, 0.4, 0.3, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0]
    ])

    BLACK_MAP_WEIGHT = np.array([
        [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        [0.1, 0.2, 0.4, 0.2, 0.0, 0.2, 0.4, 0.2, 0.1],
        [0.1, 0.4, 0.3, 0.5, 0.2, 0.5, 0.3, 0.4, 0.1],
        [0.0, 0.2, 0.5, 0.4, 0.2, 0.4, 0.5, 0.2, 0.0],
        [0.0, 0.0, 0.2, 0.2, 0.0, 0.2, 0.2, 0.0, 0.0],
        [0.0, 0.2, 0.5, 0.4, 0.2, 0.4, 0.5, 0.2, 0.0],
        [0.1, 0.4, 0.3, 0.5, 0.2, 0.5, 0.3, 0.4, 0.1],
        [0.1, 0.2, 0.4, 0.2, 0.0, 0.2, 0.4, 0.2, 0.1],
        [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0]
    ])

    # Material count
    white_soldiers = np.sum(board == Checker.WHITE_SOLDIER)
    black_soldiers = np.sum(board == Checker.BLACK_SOLDIER)
    king_positions = np.argwhere(board == Checker.WHITE_KING)

    def direct_paths_to_escape(board: np.ndarray, king_r: int, king_c: int) -> int:
        num_paths = 0
        if np.all(board[0:king_r, king_c] == Checker.EMPTY) and ASHTON_MAP[0, king_c] == Cell.ESCAPE:
            num_paths += 1
        if np.all(board[king_r + 1:9, king_c] == Checker.EMPTY) and ASHTON_MAP[8, king_c] == Cell.ESCAPE:
            num_paths += 1
        if np.all(board[king_r, 0:king_c] == Checker.EMPTY) and ASHTON_MAP[king_r, 0] == Cell.ESCAPE:
            num_paths += 1
        if np.all(board[king_r, king_c + 1:9] == Checker.EMPTY) and ASHTON_MAP[king_r, 8] == Cell.ESCAPE:
            num_paths += 1
        return num_paths

    def king_safety_score(board: np.ndarray, king_r: int, king_c: int) -> float:

        safety_score = 0.0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if ASHTON_MAP[king_r, king_c] == Cell.CASTLE:
            for dr, dc in directions:
                r, c = king_r + dr, king_c + dc
                if 0 <= r < 9 and 0 <= c < 9:
                    if board[r, c] == Checker.BLACK_SOLDIER:
                        safety_score -= 20.0
        else:
            adjacent_to_castle = False
            for dr, dc in directions:
                r, c = king_r + dr, king_c + dc
                if 0 <= r < 9 and 0 <= c < 9:
                    if ASHTON_MAP[r, c] == Cell.CASTLE:
                        adjacent_to_castle = True
                        break
            if adjacent_to_castle:
                for dr, dc in directions:
                    r, c = king_r + dr, king_c + dc
                    if 0 <= r < 9 and 0 <= c < 9:
                        if board[r, c] == Checker.BLACK_SOLDIER:
                            safety_score -= 30.0
            else:
                for dr, dc in directions:
                    r, c = king_r + dr, king_c + dc
                    if 0 <= r < 9 and 0 <= c < 9:
                        if board[r, c] == Checker.BLACK_SOLDIER or ASHTON_MAP[r, c] == Cell.CAMP:
                            safety_score -= 50.0

        up_left_area = ((0,0), (3,3))
        up_right_area = ((0,5), (3,8))
        down_left_area = ((5,0), (8,3))
        down_right_area = ((5,5), (8,8))

        area_safety_table = {
            0: 50.0,
            1: 30.0,
            2: 0.0,
            3: -50.0,
            4: -100.0
        }

        # Check which area the king is in
        if up_left_area[0][0] <= king_r <= up_left_area[1][0] and up_left_area[0][1] <= king_c <= up_left_area[1][1]:
            area = up_left_area
        elif up_right_area[0][0] <= king_r <= up_right_area[1][0] and up_right_area[0][1] <= king_c <= up_right_area[1][1]:
            area = up_right_area
        elif down_left_area[0][0] <= king_r <= down_left_area[1][0] and down_left_area[0][1] <= king_c <= down_left_area[1][1]:
            area = down_left_area
        elif down_right_area[0][0] <= king_r <= down_right_area[1][0] and down_right_area[0][1] <= king_c <= down_right_area[1][1]:
            area = down_right_area
        else:
            area = None

        if area is not None:
            black_count = 0
            for r in range(area[0][0], area[1][0] + 1):
                for c in range(area[0][1], area[1][1] + 1):
                    if board[r, c] == Checker.BLACK_SOLDIER:
                        black_count += 1
            safety_score += area_safety_table.get(black_count, -100.0)

        return safety_score

    def position_score(board: np.ndarray, role: Role) -> float:
        pos_score = 0.0
        if role == Role.WHITE:
            for r in range(9):
                for c in range(9):
                    if board[r, c] == Checker.WHITE_SOLDIER or board[r, c] == Checker.WHITE_KING:
                        pos_score += WHITE_MAP_WEIGHT[r, c] * WHITE_POSITION_SCORE_FACTOR
        else:
            for r in range(9):
                for c in range(9):
                    if board[r, c] == Checker.BLACK_SOLDIER:
                        pos_score += BLACK_MAP_WEIGHT[r, c] * BLACK_POSITION_SCORE_FACTOR
        return pos_score

    if role == Role.WHITE:
        score += white_soldiers * WHITE_SOLDER_NUM_FACTOR
        score -= black_soldiers * BLACK_SOLDER_NUM_FACTOR

        king_r, king_c = king_positions[0]

        # Check the number of direct paths to escape
        num_direct_escape_paths = direct_paths_to_escape(board, king_r, king_c)
        if num_direct_escape_paths < 2:
            score += num_direct_escape_paths * DIRECT_ESCAPE_PATH_NUM_FACTOR
        else:
            score += BIG_NUMBER / 2

        score += king_safety_score(board, king_r, king_c)
        score += position_score(board, Role.WHITE)

    else:
        score += black_soldiers * BLACK_SOLDER_NUM_FACTOR
        score -= white_soldiers * WHITE_SOLDER_NUM_FACTOR

        king_r, king_c = king_positions[0]

        # Check the number of direct paths to escape
        num_direct_escape_paths = direct_paths_to_escape(board, king_r, king_c)
        if num_direct_escape_paths > 1:
            score -= BIG_NUMBER / 2
        elif num_direct_escape_paths == 1:
            score -= BIG_NUMBER / 4
        else:
            pass

        score -= king_safety_score(board, king_r, king_c)
        score += position_score(board, Role.BLACK)

    return score


def minimax(
    board: np.ndarray,
    role: Role,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool
) -> float:
    """Minimax algorithm with alpha-beta pruning.

    Args:
        board (np.ndarray): The current board state.
        role (Role): The role of the player making the move.
        depth (int): Remaining search depth.
        alpha (float): Alpha value for pruning.
        beta (float): Beta value for pruning.
        maximizing (bool): True if maximizing player, False if minimizing.

    Returns:
        float: The evaluation score of the best move.
    """
    # Base case: terminal state or depth 0
    game_status = is_terminal_state(board)
    if game_status & GameStatus.TERMINATED or depth == 0:
        eval_score = evaluate_board(board, role)
        return eval_score

    if maximizing:
        max_eval = -BIG_NUMBER

        # Get all pieces for current role
        if role == Role.WHITE:
            piece_positions = np.argwhere((board == Checker.WHITE_SOLDIER) | (board == Checker.WHITE_KING))
        else:
            piece_positions = np.argwhere(board == Checker.BLACK_SOLDIER)

        # Try all possible moves
        for pos in piece_positions:
            pos_tuple = tuple(pos)
            moves = get_available_moves(board, pos_tuple)

            for move in moves:
                new_board = apply_move(board, pos_tuple, move)
                eval_score = minimax(new_board, role, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Beta cutoff

        return max_eval

    else:
        min_eval = BIG_NUMBER

        # Get all pieces for opponent
        opponent_role = Role.BLACK if role == Role.WHITE else Role.WHITE
        if opponent_role == Role.WHITE:
            piece_positions = np.argwhere((board == Checker.WHITE_SOLDIER) | (board == Checker.WHITE_KING))
        else:
            piece_positions = np.argwhere(board == Checker.BLACK_SOLDIER)

        # Try all possible moves
        for pos in piece_positions:
            pos_tuple = tuple(pos)
            moves = get_available_moves(board, pos_tuple)

            for move in moves:
                new_board = apply_move(board, pos_tuple, move)
                eval_score = minimax(new_board, role, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha cutoff

        return min_eval


def _evaluate_piece_moves(args: tuple) -> tuple[tuple[int, int], tuple[int, int], float] | None:
    """Helper function to evaluate all moves for a single piece (for parallel processing).

    Args:
        args: Tuple of (board, piece_position, role, depth)

    Returns:
        tuple[from_pos, to_pos, score] | None: Best move for this piece and its score, or None.
    """
    board, pos_tuple, role, depth = args

    moves = get_available_moves(board, pos_tuple)
    if not moves:
        return None

    best_move_for_piece = None
    best_score_for_piece = NEG_INF

    for move in moves:
        new_board = apply_move(board, pos_tuple, move)
        score = minimax(new_board, role, depth - 1, NEG_INF, INF, False)

        if score > best_score_for_piece:
            best_score_for_piece = score
            best_move_for_piece = (pos_tuple, move)

    if best_move_for_piece:
        result = (best_move_for_piece[0], best_move_for_piece[1], best_score_for_piece)
        if best_score_for_piece == INF or best_score_for_piece > BIG_NUMBER:
            _logger.info(
                f"[PIECE RESULT] {index2position(pos_tuple)}: best_move={index2position(best_move_for_piece[1])}, "
                f"score={best_score_for_piece}"
            )
        return result
    return None


def evaluate_minimax_move(
    board: np.ndarray, role: Role, depth: int = 3, timeout: float = 57.0, max_workers: int = 4
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Evaluate the best move using the Minimax algorithm with parallel processing.

    Args:
        board (np.ndarray): The current board state as a numpy array.
        role (Role): The role of the player.
        depth (int): The depth of the Minimax search (default: 3).
        timeout (float): The maximum time allowed for computation in seconds (default: 57.0).
        max_workers (int): Maximum number of parallel processes (default: 4).

    Returns:
        tuple[tuple[int, int], tuple[int, int]] | None: A tuple containing the from and to indices of the move,
        or None if no moves are available.
    """
    best_move = None
    best_score = NEG_INF
    search_timer = Timer(logger=_logger.debug)

    # Get all pieces for current role
    if role == Role.WHITE:
        piece_positions = np.argwhere((board == Checker.WHITE_SOLDIER) | (board == Checker.WHITE_KING))
    else:
        piece_positions = np.argwhere(board == Checker.BLACK_SOLDIER)

    _logger.info(f"Evaluating minimax for role {role.name} at depth {depth} with {max_workers} workers")
    _logger.debug(f"Found {len(piece_positions)} pieces to evaluate")

    search_timer.start()

    # Prepare tasks for parallel processing
    tasks = []
    for pos in piece_positions:
        pos_tuple = tuple(pos)
        tasks.append((board.copy(), pos_tuple, role, depth))

    # Use ProcessPoolExecutor for parallel evaluation
    executor = ProcessPoolExecutor(max_workers=max_workers)
    timed_out = False

    try:
        # Submit all tasks
        future_to_piece = {executor.submit(_evaluate_piece_moves, task): task[1] for task in tasks}

        try:
            # Process results as they complete
            for future in as_completed(future_to_piece, timeout=timeout):
                piece_pos = future_to_piece[future]
                try:
                    result = future.result(timeout=1.0)  # Short timeout for individual result
                    if result:
                        from_pos, to_pos, score = result

                        if score > best_score:
                            _logger.info(
                                f"[NEW BEST] {index2position(from_pos)} -> {index2position(to_pos)}: "
                                f"score={score} (prev best={best_score})"
                            )
                            best_score = score
                            best_move = (from_pos, to_pos)

                    search_timer.lap("one piece evaluation (parallel)")
                except Exception as e:
                    _logger.error(f"Error evaluating piece at {index2position(piece_pos)}: {e}")

        except TimeoutError:
            _logger.warning(f"Minimax evaluation timed out after {timeout}s, using best move found so far")
            timed_out = True
            # Cancel remaining futures
            for future in future_to_piece:
                future.cancel()
        except KeyboardInterrupt:
            _logger.warning("Minimax evaluation interrupted by user")
            # Cancel all futures on interrupt
            for future in future_to_piece:
                future.cancel()
            raise KeyboardInterrupt
    finally:
        # Shutdown executor immediately without waiting if timed out
        executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    search_timer.stop()

    if best_move:
        _logger.info(
            f"Best move: {index2position(best_move[0])} -> {index2position(best_move[1])} (score: {best_score})"
        )
    else:
        # TODO: Handle no valid moves found
        _logger.warning("No valid moves found, falling back to random")

    return best_move
