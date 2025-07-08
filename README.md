# Sparse RTRL

## Setting up

1. Install `uv`
2. Run `uv sync`

## Running

Basic commandline is the follows

```aiignore
uv run python main.py [--prune] --method [bptt|rtrl]
```

This allows you to test both BPTT and RTRL with and without pruning on a simple toy task.