
import sharklog

_logger = sharklog.getLogger()
_logger.addHandler(sharklog.NullHandler())

from .client import PlayerClient, PlayerClientConfig
from .rules import Role
from .strategy import Strategy

__all__ = [
    "Role",
    "Strategy",
    "PlayerClient",
    "PlayerClientConfig",
]
