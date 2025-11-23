# pytablut

## Installation

```bash
python3 -m pip install pytablut
```

## Usage

In a terminal, start the [Java Tablut server](https://github.com/AGalassi/TablutCompetition) (assuming you have Java installed):

```bash
java -jar Executables/Server.jar -g # Start server with GUI
```

In a new terminal, start one player client:

```bash
pytablut run client -r white -t 60.0 --host localhost
```

In another terminal, start the second player client:

```bash
pytablut run client -r black -t 60.0 --host localhost
```

To see all available options for the client:

```bash
pytablut run client --help
```

## Development

Clone the repository:

```bash
git clone https://github.com/Bardreamaster/pytablut.git
cd pytablut
```

Install development dependencies with uv:

```bash
uv sync
```

Run with uv: `uv run pytablut run client` or activate the virtual environment and run directly like normal user.


## License

[MIT License](LICENSE)
