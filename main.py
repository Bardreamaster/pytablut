import sharklog

from pytablut import PlayerClient, PlayerClientConfig, Role, Strategy

if __name__ == "__main__":
    sharklog.init(debug=True)
    sharklog.init(name="pytablut", debug=True)

    white_client_config = PlayerClientConfig(role=Role.WHITE,
                                             strategy=Strategy.RANDOM)

    black_client_config = PlayerClientConfig(role=Role.BLACK,
                                             strategy=Strategy.RANDOM)

    client = PlayerClient(black_client_config)

    sharklog.info("Starting player client...")

    client.start_game()
