# pytablut

## Installation

```bash
python3 -m pip install pytablut
```

## Usage

In a terminal, start the Python server:

```bash
pytablut run server
```

In a new terminal, start one player client:

```bash
pytablut run client -r white -t 60.0 --host localhost
```

In another terminal, start the second player client:

```bash
pytablut run client -r black -s random
```

Alternatively, you can use the Java server given in [this repository](https://github.com/AGalassi/TablutCompetition):

```bash
java -jar Executables/Server.jar -g # Start server with GUI
```

The java server should be used if you are participating in FUNDAMENTALS OF ARTIFICIAL INTELLIGENCE AND KNOWLEDGE REPRESENTATION Course Project competitions.

### Command Options

To see all available options:

```bash
pytablut run server --help  # Server options
pytablut run client --help  # Client options
```

Available strategies: `human`, `random`, `minimax`

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
