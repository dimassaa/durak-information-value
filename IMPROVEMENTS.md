# Improvement Backlog

Non-critical findings raised during development, appended as they are found. Each
line is prefixed with the stage that produced it and resolved in a later stage.

Resolution priority is the project owner's call, not the implementer's.

- **Stage 01** — `pyproject.toml` declares `durak.strategies`, `analysis`, and
  `reports` as packages before those directories exist (Tasks 5, 17, 19). Harmless
  for test runs from the source tree; will matter if the package is ever built
  with an isolated build backend before those stages land.
- **Stage 02** — `durak.rng` installs a NumPy overflow-warning filter scoped by message.
  A `pytest -W error::RuntimeWarning` run therefore fails under `NUMBA_DISABLE_JIT=1` by
  design (command-line filters take precedence). If a future contributor runs the suite
  that way and reads the failure as a bug, this line explains it.
