# pytablut

## Installation

```bash
python3 -m pip install pytablut
```

## Usage

In a terminal, start the [Java Tablut server](https://github.com/AGalassi/TablutCompetition) (assuming you have Java installed):

```bash
java -jar Executables/Server.jar
```

In a new terminal, start one player client:

```bash
python3 -m pytablut run client --role white --log info --strategy minimax
```

In another terminal, start the second player client:

```bash
python3 -m pytablut run client --role black --log info --strategy random
```

## License

MIT License
