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

- **Stage 06** — the win rule makes the endgame far more degenerate than the spec's "noted as a
  limitation" suggests. Measured: **68% of 36-card games and 70% of 52-card games** end with the
  winner attacking their last card, so the defender never responds and its entire hand was
  irrelevant. The primary comparison is between search strategies that share this rule, so the
  effect is common-mode rather than differential — but the rules report must state the number, not
  gesture at it. Why the 52-card figure is slightly *higher* is not established.

- **Stage 06** — `durak.rng`'s `game_stream_states` was the third site of the uint64-at-the-
  compiled-boundary defect. Two earlier ones are documented above. The pattern is now: any
  function taking a seed or stream state from Python must mask or wrap it before passing it into
  compiled code. A grep for `_fold(` and `next_below(` call sites across the codebase would be
  worth doing once at the end rather than discovering these one per stage.

- **Stage 06** — spec section 6.1 does not record where the remaining cards are at the end of a
  game (draw pile, discard, table). Five end-state fields were added to the schema so card
  conservation can be verified from the dataset alone rather than by re-running the game.
  Additive: no spec field renamed or removed. The spec's telemetry table should be updated.

- **Stage 07** — the `NUMBA_DISABLE_JIT=1` full suite has **not** been run to completion since
  Stage 6. It takes ~11 minutes, which is why it has not become routine. The per-file runs and the
  subprocess equivalence probe cover the same ground more cheaply, but the full interpreted pass is
  the project's formal cross-mode gate and must be run before Task 14 (the `prange` port), whose
  entire purpose is that guarantee. Running it in the background while working is the practical
  answer.

- **Stage 07** — spec section 6.1 does not describe the two telemetry-completeness flags added in
  Stage 7 (`flag_round_records_truncated`, `flag_move_records_truncated`), nor the five end-state
  fields from Stage 6. Both additions are necessary — without them a bounded dataset is
  indistinguishable from a complete one, and card conservation cannot be checked from the data.
  The spec's telemetry tables should be updated to match.

- **Stage 07** — `durak.engine.play_block` returns round and move records as NumPy arrays sized
  `game_count * limit` before trimming. For a 10,000-game tournament that is a 3,000,000-row
  allocation for moves and 3,000,000 for rounds even though level 3 is sampled 1-in-20. Fine at
  current block sizes (Task 8 splits into blocks); worth revisiting if block sizes grow.
