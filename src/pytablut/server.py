import socket
from collections import defaultdict

import numpy as np
import sharklog
from pydantic.dataclasses import dataclass

from pytablut.rules import (
    ASHTON_SERVER_ENCODECODING,
    AshtonServerGameState,
    AshtonServerMove,
)
from pytablut.strategy import (
    ASHTON_MAP,
    AshtonServerGameState2numpy,
    Cell,
    Checker,
    GameStatus,
    apply_move,
    get_available_moves,
    is_terminal_state,
    numpy2AshtonServerGameState,
    position2index,
)
from pytablut.utils import receive_exact_bytes

_logger = sharklog.getLogger()


@dataclass
class AshtonServerConfig:
    host: str = "localhost"
    port_white: int = 5800
    port_black: int = 5801
    max_turns: int = 300  # Maximum number of turns before declaring a draw
    enable_draw_by_repetition: bool = True  # Enable draw by state repetition


class AshtonServer:
    """Tablut game server implementing the Ashton protocol.

    This server manages a Tablut game between two players (WHITE and BLACK),
    handling connections, move validation, game state updates, and win condition checks.
    """

    def __init__(self, server_config: AshtonServerConfig = None):
        self.config(server_config)
        self.server_socket = None
        self.client_sockets = []
        self.game_state = None
        self.board_array = None
        self.state_history = defaultdict(int)  # Track state repetitions
        self.turn_count = 0

    def config(self, server_config: AshtonServerConfig = None):
        if server_config is None:
            server_config = AshtonServerConfig()
        self.server_config = server_config

    def send_message(self, conn: socket.socket, data: bytes | str | dict | AshtonServerGameState):
        """Send a message to a client using length-prefixed protocol.

        Args:
            conn: The socket connection to send to.
            data: The data to send (bytes, string, dict, or AshtonServerGameState).
        """
        if isinstance(data, (str, dict)):
            import json
            data = json.dumps(data).encode(ASHTON_SERVER_ENCODECODING)
        elif isinstance(data, AshtonServerGameState):
            data = data.to_bytes()
        elif not isinstance(data, bytes):
            raise TypeError(f"Unsupported data type: {type(data)}")

        length = len(data)
        conn.sendall(length.to_bytes(4, byteorder="big"))
        conn.sendall(data)

    def receive_message(self, conn: socket.socket) -> bytes:
        """Receive a length-prefixed message from a client.

        Args:
            conn: The socket connection to receive from.

        Returns:
            The received message as bytes.
        """
        length_bytes = receive_exact_bytes(conn, 4)
        expected_length = int.from_bytes(length_bytes, byteorder="big")
        return receive_exact_bytes(conn, expected_length)

    def receive_player_name(self, conn: socket.socket) -> str:
        """Receive and decode the player name from a client.

        Args:
            conn: The socket connection to receive from.

        Returns:
            The player name as a string.
        """
        name_bytes = self.receive_message(conn)
        return name_bytes.decode(ASHTON_SERVER_ENCODECODING)

    def initialize_game(self):
        """Initialize the game state with the starting position."""
        self.game_state = AshtonServerGameState()
        self.game_state.turn = "WHITE"
        self.board_array = AshtonServerGameState2numpy(self.game_state)
        self.state_history.clear()
        self.turn_count = 0
        self._record_state()
        _logger.info("Game initialized with starting position")

    def _record_state(self):
        """Record the current board state for draw detection."""
        # TODO: Keep record for replay, training, retract moves, etc.
        if self.server_config.enable_draw_by_repetition:
            # Create a hashable representation of the board
            state_tuple = tuple(map(tuple, self.board_array.tolist()))
            self.state_history[state_tuple] += 1

    def _check_draw_by_repetition(self) -> bool:
        """Check if the current state has occurred enough times to declare a draw.

        Returns:
            True if a draw should be declared, False otherwise.
        """
        if not self.server_config.enable_draw_by_repetition:
            return False

        state_tuple = tuple(map(tuple, self.board_array.tolist()))
        return self.state_history[state_tuple] >= 3

    def validate_move(self, move: AshtonServerMove) -> tuple[bool, str]:
        """Validate a move according to Tablut rules.

        Args:
            move: The move to validate.

        Returns:
            A tuple of (is_valid, error_message). error_message is empty if valid.
        """
        # Check if it's the correct player's turn
        if move.turn != self.game_state.turn:
            return False, f"Wrong turn: expected {self.game_state.turn}, got {move.turn}"

        # Parse positions
        try:
            from_r, from_c = position2index(move.from_)
            to_r, to_c = position2index(move.to)
        except (ValueError, IndexError) as e:
            return False, f"Invalid position format: {e}"

        # Check bounds
        if not (0 <= from_r < 9 and 0 <= from_c < 9 and 0 <= to_r < 9 and 0 <= to_c < 9):
            return False, "Position out of bounds"

        # Check if there's a piece to move
        piece = self.board_array[from_r, from_c]
        if piece == Checker.EMPTY:
            return False, f"No piece at {move.from_}"

        # Check if the piece belongs to the current player
        if move.turn == "WHITE":
            if piece not in (Checker.WHITE_SOLDIER, Checker.WHITE_KING):
                return False, f"Piece at {move.from_} does not belong to WHITE"
        else:  # BLACK
            if piece != Checker.BLACK_SOLDIER:
                return False, f"Piece at {move.from_} does not belong to BLACK"

        # Check if the move is in the available moves for this piece
        available_moves = get_available_moves(self.board_array, (from_r, from_c))
        if (to_r, to_c) not in available_moves:
            return False, f"Move from {move.from_} to {move.to} is not allowed"

        return True, ""

    def apply_move_to_game(self, move: AshtonServerMove):
        """Apply a validated move to the game state.

        Args:
            move: The move to apply.
        """
        from_pos = position2index(move.from_)
        to_pos = position2index(move.to)

        # Apply the move using the strategy module's function
        self.board_array = apply_move(self.board_array, from_pos, to_pos)

        # Update the game state
        self.game_state = numpy2AshtonServerGameState(self.board_array, self.game_state.turn)

        # Record state for draw detection
        self._record_state()

        _logger.info(f"Move applied: {move.from_} -> {move.to}")

    def check_game_over(self) -> GameStatus:
        """Check if the game is over and update game state accordingly.

        Returns:
            The current game status.
        """
        # Check terminal state (win/loss)
        status = is_terminal_state(self.board_array)

        if status == GameStatus.WHITE_WIN:
            self.game_state.turn = "WHITEWIN"
            _logger.info("Game over: WHITE wins!")
            return status
        elif status == GameStatus.BLACK_WIN:
            self.game_state.turn = "BLACKWIN"
            _logger.info("Game over: BLACK wins!")
            return status

        # Check for draw by repetition
        if self._check_draw_by_repetition():
            self.game_state.turn = "DRAW"
            _logger.info("Game over: DRAW by repetition!")
            return GameStatus.DRAW

        # Check for draw by turn limit
        if self.turn_count >= self.server_config.max_turns:
            self.game_state.turn = "DRAW"
            _logger.info(f"Game over: DRAW by turn limit ({self.server_config.max_turns} turns)")
            return GameStatus.DRAW

        return GameStatus.ONGOING

    def switch_turn(self):
        """Switch the turn to the other player."""
        self.turn_count += 1
        if self.game_state.turn == "WHITE":
            self.game_state.turn = "BLACK"
        else:
            self.game_state.turn = "WHITE"
        _logger.debug(f"Turn switched to {self.game_state.turn} (turn {self.turn_count})")

    def start(self):
        """Start the Tablut game server and manage the game loop."""
        _logger.info("Starting Tablut server...")
        _logger.info(f"Waiting for WHITE player on {self.server_config.host}:{self.server_config.port_white}")
        _logger.info(f"Waiting for BLACK player on {self.server_config.host}:{self.server_config.port_black}")

        # Create sockets for both players
        self.white_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.white_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.white_socket.bind((self.server_config.host, self.server_config.port_white))
        self.white_socket.listen(1)

        self.black_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.black_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.black_socket.bind((self.server_config.host, self.server_config.port_black))
        self.black_socket.listen(1)

        try:
            # Accept connections from both players
            _logger.info("Waiting for WHITE player to connect...")
            self.white_conn, self.white_addr = self.white_socket.accept()
            white_name = self.receive_player_name(self.white_conn)
            _logger.info(f"WHITE player '{white_name}' connected from {self.white_addr}")

            _logger.info("Waiting for BLACK player to connect...")
            self.black_conn, self.black_addr = self.black_socket.accept()
            black_name = self.receive_player_name(self.black_conn)
            _logger.info(f"BLACK player '{black_name}' connected from {self.black_addr}")

            # Initialize the game
            self.initialize_game()

            # Send initial game state to both players
            _logger.info("Sending initial game state to both players...")
            self.send_message(self.white_conn, self.game_state)
            self.send_message(self.black_conn, self.game_state)

            # Main game loop
            while True:
                # Determine whose turn it is
                if self.game_state.turn == "WHITE":
                    current_conn = self.white_conn
                    current_name = white_name
                    waiting_conn = self.black_conn
                elif self.game_state.turn == "BLACK":
                    current_conn = self.black_conn
                    current_name = black_name
                    waiting_conn = self.white_conn
                else:
                    # Game is over
                    _logger.info(f"Game ended with result: {self.game_state.turn}")
                    break

                # Receive move from current player
                _logger.info(f"Waiting for move from {current_name} ({self.game_state.turn})...")
                try:
                    move_bytes = self.receive_message(current_conn)
                    move = AshtonServerMove.from_bytes(move_bytes)
                    _logger.info(f"Received move: {move.from_} -> {move.to}")
                except Exception as e:
                    _logger.error(f"Error receiving move from {current_name}: {e}")
                    # Declare opponent as winner
                    if self.game_state.turn == "WHITE":
                        self.game_state.turn = "BLACKWIN"
                    else:
                        self.game_state.turn = "WHITEWIN"
                    break

                # Validate the move
                is_valid, error_msg = self.validate_move(move)
                if not is_valid:
                    _logger.warning(f"Invalid move from {current_name}: {error_msg}")
                    # Declare opponent as winner due to illegal move
                    if self.game_state.turn == "WHITE":
                        self.game_state.turn = "BLACKWIN"
                    else:
                        self.game_state.turn = "WHITEWIN"
                    _logger.info(f"Game over: {current_name} made an illegal move")
                    break

                # Apply the move
                self.apply_move_to_game(move)

                # Check if the game is over after this move
                game_status = self.check_game_over()

                if game_status & GameStatus.TERMINATED:
                    # Game is over, send final state to both players
                    _logger.info("Sending final game state to both players...")
                    self.send_message(self.white_conn, self.game_state)
                    self.send_message(self.black_conn, self.game_state)
                    break

                # Switch turn
                self.switch_turn()

                # Send updated game state to both players
                self.send_message(current_conn, self.game_state)
                self.send_message(waiting_conn, self.game_state)

            _logger.info(f"Game finished: {self.game_state.turn}")

        except KeyboardInterrupt:
            _logger.warning("Server interrupted by user")
        except Exception as e:
            _logger.error(f"Server error: {e}", exc_info=True)
        finally:
            # Clean up connections
            _logger.info("Closing connections...")
            if hasattr(self, 'white_conn'):
                self.white_conn.close()
            if hasattr(self, 'black_conn'):
                self.black_conn.close()
            self.white_socket.close()
            self.black_socket.close()
            _logger.info("Server stopped")
