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

- **Stage 04** — 44% of `RANDOM` vs `RANDOM` games hit the 10,000-round cap and never
  terminate (2,617 of 6,000, measured). `RANDOM` takes constantly, and the rules allow taking
  forever once the draw pile is empty, so the strategy drives itself into a cycle. This breaks
  the project's only exact assertion: the symmetry test is currently computed over a biased
  56% subset. Blocking for the headline result. Three options costed in
  `.superpowers/reports/stage-04-strategy-registry.md`; the call belongs to the study owner and
  to Task 8. Provisional plan is option 2 (publish the termination rate beside the 0.5).
