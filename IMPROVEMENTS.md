# Improvement Backlog

Non-critical findings raised during development, appended as they are found. Each
line is prefixed with the stage that produced it and resolved in a later stage.

Resolution priority is the project owner's call, not the implementer's.

- **Stage 01** — `pyproject.toml` declares `durak.strategies`, `analysis`, and
  `reports` as packages before those directories exist (Tasks 5, 17, 19). Harmless
  for test runs from the source tree; will matter if the package is ever built
  with an isolated build backend before those stages land. *(Resolved at Stage 04
  for `durak.strategies`; `analysis` and `reports` still outstanding.)*

- **Stage 02** — `durak.rng` installs a NumPy overflow-warning filter scoped by message.
  A `pytest -W error::RuntimeWarning` run therefore fails under `NUMBA_DISABLE_JIT=1` by
  design (command-line filters take precedence). If a future contributor runs the suite
  that way and reads the failure as a bug, this line explains it.

- **Stage 04 — CORRECTED at Stage 05, do not act on the original text.** The original
  entry said 44% of `RANDOM` games never terminate and that a capped-game scoring rule was
  blocking the headline result. **That was wrong.** The 44% was caused by the trump
  representation bug fixed in Stage 05, not by the rules. `RANDOM` vs `RANDOM` is now
  0.5028 over 6,000 games with zero capped games.

- **Stage 05** — `GREEDY` caps out in 11 of 400 games (2.75%) and `AUTO_DEFEND` in 20 of
  400 (5%), from the genuine "taking is always legal" stall at low hand sizes with an empty
  deck. `RANDOM` reaches it 0 times in 400, so the project's only exact assertion is
  unaffected. Two follow-ups, both the study owner's call: the engine (Task 6) must record
  capped games as flagged anomalies rather than losses, and every report must publish the
  per-strategy termination rate beside the win rate. Whether to close the hole in the rules
  with a take cap is a spec question, carried to the rules report.

- **Stage 05** — spec section 3.5's termination argument ("once the draw pile is empty the
  attacker's hand can only shrink") is not sound in general. It now holds empirically for
  `RANDOM` because trumps beat off-suit cards and hands therefore shrink, but a real stall
  configuration exists (see the entry above). The spec statement should be corrected to say
  termination is bounded by the round cap, not guaranteed by the rules.
