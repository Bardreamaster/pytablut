import json
import socket

import sharklog
from pydantic.dataclasses import dataclass

from pytablut.rules import (
    ASHTON_SERVER_ENCODECODING,
    AshtonServerGameState,
    AshtonServerMove,
    Role,
)
from pytablut.strategy import (
    AshtonServerGameState2numpy,
    Strategy,
    evaluate_random_move,
    index2position,
)
from pytablut.utils import receive_exact_bytes

_logger = sharklog.getLogger()

@dataclass
class PlayerClientConfig:
    role: Role
    name: str = "Player"
    server_ip: str = "localhost"
    server_port: int = 5800
    strategy: Strategy = Strategy.HUMAN

    def __post_init__(self):
        if self.name == "":
            object.__setattr__(self, 'name', "WP" if self.role == Role.WHITE else "BP")


class PlayerClient:

    def __init__(self, player_client_config: PlayerClientConfig = None):
        self.player_client_config = player_client_config
        self.socket = None

    def config(self, player_client_config: PlayerClientConfig):
        self.player_client_config = player_client_config

    def connect_game_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(
            (self.player_client_config.server_ip, self.player_client_config.server_port)
        )
        _logger.debug(
            f"Connected to server at "
            f"{self.player_client_config.server_ip}:"
            f"{self.player_client_config.server_port}"
        )

    def send_message(self, data: bytes | str | dict | AshtonServerMove):
        if self.socket is None:
            raise Exception("socket is None")

        if isinstance(data, (str, dict)):
            data = json.dumps(data).encode(ASHTON_SERVER_ENCODECODING)
        elif isinstance(data, AshtonServerMove):
            data = data.to_bytes()
        elif not isinstance(data, bytes):
            raise Exception(f"data type {type(data)} is not supported")

        length = len(data)
        self.socket.sendall(length.to_bytes(4, byteorder="big"))
        self.socket.sendall(data)

    def receive_game_state(self) -> AshtonServerGameState:
        if self.socket is None:
            raise Exception("socket is None")
        try:
            length_bytes = receive_exact_bytes(self.socket, 4)
            expected_length = int.from_bytes(length_bytes, byteorder="big")

            game_state_bytes = receive_exact_bytes(self.socket, expected_length)
        except Exception as e:
            _logger.error(f"Failed to receive game state: {e}")
            raise Exception("Connection lost") from e

        return AshtonServerGameState.from_bytes(game_state_bytes)

    def get_move_from_user(self) -> AshtonServerMove:
        from_pos = input("Enter move FROM position (e.g., e3): ")
        to_pos = input("Enter move TO position (e.g., e4): ")
        return AshtonServerMove(
            from_=from_pos, to=to_pos, turn=self.player_client_config.role.name
        )

    def get_random_move(self, game_state: AshtonServerGameState) -> AshtonServerMove:
        board_array = AshtonServerGameState2numpy(game_state)
        move_indices = evaluate_random_move(
            board_array, self.player_client_config.role
        )
        if move_indices is None:
            raise Exception(
                "No available moves"
            )  # which should not happen because the game should be over.
        from_pos = index2position(move_indices[0])
        to_pos = index2position(move_indices[1])
        return AshtonServerMove(
            from_=from_pos, to=to_pos, turn=self.player_client_config.role.name
        )

    def evaluate_move(self, game_state: AshtonServerGameState) -> AshtonServerMove:
        if self.player_client_config.strategy == Strategy.HUMAN:
            return self.get_move_from_user()
        elif self.player_client_config.strategy == Strategy.RANDOM:
            return self.get_random_move(game_state)
        elif self.player_client_config.strategy == Strategy.MINIMAX:
            return self.get_minimax_move(game_state)

    def start_game(self):
        # 0. Connect to server
        self.connect_game_server()

        # 1. Send player name to server
        name = self.player_client_config.name
        self.send_message(name)
        _logger.info(f"Sent player name: {name}")

        # 2. Wait for initial game state from server
        game_state = self.receive_game_state()
        _logger.debug(f"Received initial game state: {game_state}")

        while True:

            # 3. Send moves to server if it's this player's turn
            if game_state.turn == self.player_client_config.role.name:
                move = self.evaluate_move(game_state)
                self.send_message(move)
                _logger.info(f"Sent move: {move}")
            elif game_state.turn in ["WHITEWIN", "BLACKWIN", "DRAW"]:
                _logger.info(f"Game over with result: {game_state.turn}")
                break
            else:
                _logger.info("Waiting for opponent's move...")

            # 4. Receive updated game state from server
            game_state = self.receive_game_state()
            _logger.debug(f"Received updated game state: {game_state}")

            pass
