from enum import IntEnum
from typing import Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


class Role(IntEnum):
    WHITE = 1
    BLACK = 2


class GameResult(IntEnum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 4


ASHTON_SERVER_ENCODECODING = "utf-8"


@dataclass
class AshtonServerGameState:
    board: list[list[Literal["BLACK", "WHITE", "KING", "EMPTY", "THRONE"]]] = Field(
        default_factory=lambda: [
                ["EMPTY","EMPTY","EMPTY","BLACK","BLACK","BLACK","EMPTY","EMPTY","EMPTY"],
                ["EMPTY","EMPTY","EMPTY","EMPTY","BLACK","EMPTY","EMPTY","EMPTY","EMPTY"],
                ["EMPTY","EMPTY","EMPTY","EMPTY","EMPTY","WHITE","EMPTY","EMPTY","EMPTY"],
                ["BLACK","EMPTY","EMPTY","EMPTY","WHITE","EMPTY","EMPTY","EMPTY","BLACK"],
                ["BLACK","BLACK","WHITE","WHITE","KING","WHITE","WHITE","BLACK","BLACK"],
                ["BLACK","EMPTY","EMPTY","EMPTY","WHITE","EMPTY","EMPTY","EMPTY","BLACK"],
                ["EMPTY","EMPTY","EMPTY","EMPTY","WHITE","EMPTY","EMPTY","EMPTY","EMPTY"],
                ["EMPTY","EMPTY","EMPTY","EMPTY","BLACK","EMPTY","EMPTY","EMPTY","EMPTY"],
                ["EMPTY","EMPTY","EMPTY","BLACK","BLACK","BLACK","EMPTY","EMPTY","EMPTY"]
            ]   # fmt: skip
    )
    turn: Literal["WHITE", "BLACK", "WHITEWIN", "BLACKWIN", "DRAW"] = ""

    @classmethod
    def from_bytes(cls, data: bytes) -> "AshtonServerGameState":
        import json

        state_json = data.decode(ASHTON_SERVER_ENCODECODING)
        state_dict = json.loads(state_json)
        return cls(**state_dict)

    def to_bytes(self) -> bytes:
        import json

        state_dict = {"board": self.board, "turn": self.turn}
        state_json = json.dumps(state_dict)
        return state_json.encode(ASHTON_SERVER_ENCODECODING)


@dataclass(config=ConfigDict(populate_by_name=True))
class AshtonServerMove:
    from_: str = Field(alias="from")
    to: str = ""
    turn: Literal["WHITE", "BLACK"] = ""

    @classmethod
    def from_bytes(cls, data: bytes) -> "AshtonServerMove":
        import json

        move_json = data.decode(ASHTON_SERVER_ENCODECODING)
        move_dict = json.loads(move_json)
        return cls(**move_dict)

    def to_bytes(self) -> bytes:
        import json

        move_dict = {"from": self.from_, "to": self.to, "turn": self.turn}
        move_json = json.dumps(move_dict)
        return move_json.encode(ASHTON_SERVER_ENCODECODING)
