# Improvement Backlog

Non-critical findings raised during development, appended as they are found. Each
line is prefixed with the stage that produced it and resolved in a later stage.

Resolution priority is the project owner's call, not the implementer's.

- **Stage 01** — `pyproject.toml` declares `durak.strategies`, `analysis`, and
  `reports` as packages before those directories exist (Tasks 5, 17, 19). Harmless
  for test runs from the source tree; will matter if the package is ever built
  with an isolated build backend before those stages land.