# Development

## Introduction

Currently, `pytablut` includes the server and client implementations. To be compatible with the [competition framework](https://github.com/AGalassi/TablutCompetition) for course project of Fundamentals of Artificial Intelligence and Knowledge Representation in the university of Bologna, the server and client follow specific interfaces defined in the competition documentation.

So inside the given interface, there I convert all game info/status to numbers so that we can use features from numpy to process data efficently and prepare for making a self-training agent easier.

## Environment

Clone this repo:

```bash
git clone https://github.com/Bardreamaster/pytablut.git
cd pytablut
```

Install development dependencies with uv:

```bash
uv sync
```
