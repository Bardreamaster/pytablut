"""
Command-line interface for pytablut.

Usage:
    python -m pytablut run client --role white
    python -m pytablut run client --role black --strategy random
"""

import sys

import click
import sharklog

from .client import PlayerClient, PlayerClientConfig
from .rules import Role
from .strategy import Strategy


@click.group()
def cli():
    """Tablut game tools."""
    pass


@cli.group()
def run():
    """Run a component."""
    pass


@run.command()
@click.option(
    '--role',
    type=click.Choice(['white', 'black'], case_sensitive=False),
    required=True,
    help='Player role (white or black)'
)
@click.option(
    '--strategy',
    type=click.Choice(['human', 'random', 'minimax'], case_sensitive=False),
    default='random',
    help='Player strategy (default: random)'
)
@click.option(
    '--name',
    default='Player',
    help='Player name (default: Player)'
)
@click.option(
    '--host',
    default='localhost',
    help='Server host (default: localhost)'
)
@click.option(
    '--port',
    type=int,
    default=0,
    help='Server port (default: auto-select based on role)'
)
@click.option(
    '--log',
    type=click.Choice(['error', 'warning', 'info', 'debug'], case_sensitive=False),
    default='warning',
    help='Logging level (default: warning)'
)
@click.option(
    "--debug",
    is_flag=True,
    help="Set logging level to debug"
)
def client(role, strategy, name, host, port, log, debug):
    """Run a Tablut client."""
    log_level = sharklog.getLevelName(log.upper())

    # Initialize logging
    sharklog.init(name="pytablut", level=log_level, debug=debug)
    _logger = sharklog.getLogger(name="pytablut.__main__")

    # Parse role
    role_enum = Role.WHITE if role.lower() == 'white' else Role.BLACK

    # Parse strategy
    strategy_enum = Strategy[strategy.upper()]

    # Determine port if not specified
    if not port:
        if role_enum == Role.WHITE:
            port = 5800
        elif role_enum == Role.BLACK:
            port = 5801

    # Create client configuration
    config = PlayerClientConfig(
        role=role_enum,
        name=name,
        server_ip=host,
        server_port=port,
        strategy=strategy_enum
    )

    # Create and start client
    client_instance = PlayerClient(config)

    _logger.info(f"Starting {role_enum.name} player with {strategy_enum.name} strategy...")
    _logger.info(f"Connecting to {host}:{config.server_port}")

    try:
        client_instance.start_game()
    except KeyboardInterrupt:
        _logger.warning("Game interrupted by user.")
        sys.exit(0)
    except Exception as e:
        _logger.error(f"Error: {e}")
        if log.lower() == 'debug':
            raise
        sys.exit(1)


if __name__ == "__main__":
    cli()
