# Quick Start

Install the package:

```bash
python3 -m pip install pytablut
```

Start the server in one terminal:

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
