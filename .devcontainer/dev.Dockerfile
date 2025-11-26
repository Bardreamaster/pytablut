FROM mcr.microsoft.com/devcontainers/python:3

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    black \
    ruff \
    pre-commit
