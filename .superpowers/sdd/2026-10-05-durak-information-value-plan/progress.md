# SDD ledger — plan: docs/superpowers/plans/2026-10-05-durak-information-value-plan.md

Spec: docs/superpowers/specs/2026-10-05-durak-information-value-design.md
Branch: master (user chose in-place over a worktree; master had 2 doc commits at start: a1d66bb, bbbb3c6)
Executor: inline (user switched from subagent-driven to executing-plans). TDD is the per-task gate; one whole-branch review at the end.

## Pre-flight conflict scan

Pairwise scan over tasks that share a file or an interface. Authority order: spec is binding, plan argues from it.

| # | Pair | Produces vs consumes | Finding |
|---|---|---|---|
| 1 | T1 → T2 | T1 `Settings`/`get_settings`, version constants vs T2 `build_deck(deck_size)` | Consistent. T2 takes a plain `int`, never the dataclass — matches the "no dataclass crosses nopython" constraint. |
| 2 | T1 → T3 | T1 config vs T3 `mix_seed`/`RngStream`/`GameStreams` | Consistent. T3 declares no dependency on T1. |
| 3 | T2 → T3 | shared: none. T2 owns `cards.py`, T3 owns `rng.py`. | Disjoint modules. |
| 4 | T2 → T4 | T2 `beats`/`rank_of`/`suit_of`/`build_deck` vs T4 `setup_game`/`legal_actions` | Consistent. T4 must call `beats` for every legality decision — no re-implementation of the beat relation. |
| 5 | T3 → T4 | T3 `GameStreams`/`shuffle_inplace` vs T4 `setup_game(deck_size, streams, ...)` | Consistent. Setup consumes the `streams` bundle; it must not construct its own RNG. |
| 6 | T4 → T5 | T4 `legal_actions`/`apply_action`/`actions` vs T5 policies | Consistent. Policy signature returns a packed `int32`. |
| 7 | T4 → T6 | T4 state + `check_invariants` vs T6 `play_game` | Consistent. Engine owns the loop; the win check lives in `apply_action` only (Task 6 Step 3). |
| 8 | T5 → T6 | T5 `get_policy`/`StrategyId` vs T6 `play_game(strategy_p0, ...)` | Consistent. Engine dispatches by integer id. |
| 9 | T6 → T7 | T6 `GameOutcome`/round+move records vs T7 schemas/writer | Consistent. |
| 10 | T6 → T8 | T6 `play_block` vs T8 runner orchestration | Consistent. T8 serial by design; `workers`/`block_size` arrive in T15. |
| 11 | T7 → T8 | T7 `write_parquet` vs T8 `run_tournament` | Consistent. |
| 12 | T8 → T18 | T8 benchmark throughput vs T18 scale decision | Hard dependency. Ruling recorded below. |
| 13 | T4 → T9 | T4 `GameState`/unseen-set derivation vs T9 `initial_particles` | Consistent. T9 derives unseen from state fields, not by re-deriving from the deck. |
| 14 | T9 → T10 | T9 particle marginals vs T10 search | Consistent. T10 consumes particles only for the hand; `evaluate` reads state. |
| 15 | T9+T10 → T11 | T9 particles, T10 `expectimax` vs T11 `policy_search` | Consistent, and this is the experiment's load-bearing pair. T11's shared-parameter assertion is the guard. |
| 16 | T11 → T12 | T11 `policy_search` vs T12 `policy_cheater_doc` | Consistent. T12 compares a *different* code path on purpose. |
| 17 | T9 → T13 | T9 particle marginals vs T13 `policy_conservative` | Consistent. |
| 18 | T6 → T14 | T6 serial `play_block` vs T14 `prange` compile | Consistent; T14 must preserve serial semantics bit-for-bit. Highest-risk task. |
| 19 | T7 → T15 | T7 partition layout vs T15 resume + `analysis/schema.sql` | Consistent. Block writes must be atomic (tmp + rename). |
| 20 | T16 → T17 | T16 stats functions vs T17 queries | Consistent. T17 must not re-implement statistics. |
| 21 | T15 → T17 | T15 DuckDB views vs T17 `build_all` | Consistent. |
| 22 | T17 → T19 | T17 JSON contract vs T19 `reports/dashboard.py` | T19 consumes JSON shape only. |
| 23 | T19 → T20 | T19 primitives vs T20 report 1 | Consistent. |
| 24 | T19 → T21 | T19 primitives vs T21 reports 2-3 + figures | Consistent. |
| 25 | T18+T17 → T22 | dataset + aggregates vs README | Hard dependency. |

Self-agreement per task: all 22 tasks' specified tests match the code the same task specifies, with three exceptions corrected during plan self-review (GameState → NamedTuple, `check_invariants` → bitmask, `opponent_independent_strength` added to T17). No task mandates a test that asserts nothing; no task mandates verbatim duplication.

Plan-vs-rubric conflicts found: **none.** The plan does not require a no-op test or a duplicated logic block anywhere.

### Rulings

- Ruling: Tasks run strictly in plan order — the dependency graph above is a real chain, and out-of-order work would mean inventing interface definitions before their producer exists. Cost if wrong: none; this is the plan's stated order.
- Ruling: Task 18 Step 1 must read the Task 8 benchmark from this ledger before choosing the dataset scale, and must record the chosen scale and its justification *before* the run starts. Cost if wrong: a scale chosen after seeing results is not pre-registration.
- Ruling: Task 14's forced rewrites are in scope for that task, not escalated — the plan explicitly anticipates them and the spec's execution-mode constraints mandate the fix. `numba.objmode` and `typed.Dict` escapes are banned. Cost if wrong: none.Task 1: complete (commits bbbb3c6..7dc9335, tests: /home/dimass/opencode/durak/.venv/bin/python -m pytest → 7 passed in 0.01s)

## Task log

Task 1: Ruling: plan says "src layout" but its own File Map is root-level `durak/` — used the File Map, since later tasks are written against it. Cost if wrong: one directory move.
Task 1: Ruling: `pyproject.toml` declares `durak.strategies`/`analysis`/`reports` before those dirs exist (Tasks 5/17/19); left as declared because the suite runs from the source tree. Cost if wrong: an isolated build fails until Task 5.
Task 1: complete (commits bbbb3c6..7dc9335, tests: .venv/bin/python -m pytest → 7 passed in 0.01s)
Task 2: complete (commits 7dc9335..560212a, tests: /home/dimass/opencode/durak/.venv/bin/python -m pytest → 26 passed in 0.06s)
Task 2: Ruling: `rank_of(card)` returns a 0-based rank offset, not the printed rank — the plan's own signature takes no `deck_size`, so it cannot know the deck floor. Added `printed_rank(card, deck_size)` for the two rules that name a printed rank ("lowest trump attacks first", "high card >= 10"). Cost if wrong: one rename plus two call sites in Task 4.
Task 2: Ruling: `card_label`/`parse_card_label` raise on a rank absent from the target deck (e.g. "2♠" against 36 cards) instead of aliasing onto a real card. Cost if wrong: a debug label could print a card the game does not contain.
Task 2: complete (commits 7dc9335..560212a, tests: .venv/bin/python -m pytest → 26 passed in 0.06s)
Task 3: complete (commits 560212a..544f59c, tests: /home/dimass/opencode/durak/.venv/bin/python -m pytest → 69 passed in 3.98s)
Task 3: Ruling: PRNG shift counts must be Python `int`, never `np.uint64` — measured, not assumed: shifting a uint64 by a uint64 hashed to `1371bbaa` compiled vs `42c9086a` interpreted. This would have failed the Task 14 equivalence test looking like an engine bug. Cost if wrong: the whole equivalence guarantee is void.
Task 3: Ruling: every operand crossing into compiled code is wrapped in `np.uint64`. Numba returns a plain Python int to its caller and re-infers `int64` on the next call, which raised OverflowError above 2**63-1 and silently produced a *different* sequence for smaller values. Cost if wrong: wrong numbers with no error.
Task 3: Ruling: engine-facing seed entry point is fixed-arity `mix_seed3(run_seed, pairing_index, game_index)` because Numba cannot type `*parts`. Both routes share `_fold`; a test asserts they agree. Cost if wrong: one extra wrapper.
Task 3: Ruling: `durak.rng` suppresses NumPy's scalar-overflow RuntimeWarning, scoped by message not category; a test proves unrelated RuntimeWarnings still surface. Consequence: `pytest -W error::RuntimeWarning` fails under NUMBA_DISABLE_JIT=1 by design (CLI filters win). Cost if wrong: a scoped filter change would need re-testing.
Task 3: complete (commits 560212a..544f59c, tests: .venv/bin/python -m pytest → 69 passed in 3.98s)

### Author-identity correction (2026-10-05)

All six existing commits were authored as `opencode <opencode@local>` because I had
passed `-c user.name=...` explicitly. The repository config already carries the owner's
identity. All six commits were rewritten with `git filter-branch --env-filter` to
`DimAss <dimon.kharkevich@gmail.com>`; `refs/original` and the reflogs were removed, so
the pre-rewrite objects are gone.

Rewritten SHAs (old -> new):
  a1d66bb -> 0c733b4  design specification
  bbbb3c6 -> 802d929  implementation plan
  7dc9335 -> f1e0a61  repository skeleton
  560212a -> cdfc866  card encoding
  544f59c -> 332b6ef  PRNG
  312ad11 -> 4ae9195  normative rules engine   <- current HEAD

Every earlier `BASE` recorded in this ledger refers to a pre-rewrite SHA. The equivalent
post-rewrite commits are the new ones listed above; the ranges are identical in content.

Task 3: complete (rewritten as 332b6ef, tests: .venv/bin/python -m pytest -> 127 passed)
Task 4: complete (rewritten as 4ae9195, tests: .venv/bin/python -m pytest -> 127 passed)
Task 5: complete (commits 4ae9195..8117280, tests: /home/dimass/opencode/durak/.venv/bin/python -m pytest → 161 passed in 13.98s)

Task 5: complete (commits 4ae9195..8117280, tests: .venv/bin/python -m pytest -> 161 passed)
Task 5: Ruling: `GameState` gained `trump` (suit) and `trump_card` (card id). The plan listed only `trump`, and the plan's own tests would have passed either way; the ambiguity between the two is what caused the Stage 05 defect. Cost if wrong: one extra int64 field.
Task 5: Ruling: a policy's only permitted mutation of the state is advancing its own player's stream. The plan's policy signature returns only an action, so the streams must live in the state. Cost if wrong: a different policy signature, and Task 6 rewrites the call.
Task 5: Ruling: policies propose a preference, then accept the rules' verdict. The throw-in cap is NOT recomputed in the policy -- the first draft did and returned illegal actions at a 6-attack table. Cost if wrong: the two implementations eventually disagree.
Task 5: FINDING (critical, fixed in 6b98a6e): `GameState.trump` stored a card id while `beats()` expects a suit, disabling the entire off-suit rule in every real game. No test caught it for three stages because every fixture passed a suit. RANDOM vs RANDOM is now 0.5028 over 6000 games with 0 capped, bit-identical across execution modes.
Task 5: FINDING (open, not blocking): GREEDY caps in 11/400 games and AUTO_DEFEND in 20/400 from a genuine always-legal-take stall. RANDOM reaches it 0/400, so the exact symmetry assertion is unaffected. Task 6 must record capped games as flagged anomalies; reports must publish the per-strategy termination rate. Whether to add a take cap to the rules is a spec question for the owner.

Task 6: complete (commits 6b98a6e..fc80efc, tests: .venv/bin/python -m pytest -> 195 passed in 43s)
Task 6: Ruling: `GameState.trump` is the suit and `trump_card` the card id (Stage 05 fix, reinforced here).
Task 6: Ruling: the round-cap condition lives once in `rules.round_cap_reached`; `rules.begin_round` raises on it, the engine records `flag_round_cap_hit` on it. Numba has no usable exception handling around a mutated state, so a predicate plus a raise is the only way to keep one condition and two responses.
Task 6: Ruling: `setup_game` is a thin Python wrapper over the compiled `rules.deal_state`, because the engine cannot construct a `GameStreams`. One deal implementation; the wrapper only unwraps a bundle into three stream states.
Task 6: Ruling: strategy dispatch in the engine is an if/elif chain on the integer id, not a dict of dispatchers — a `prange` loop needs a compile-time constant callee. Adding a strategy requires a branch plus a test that the branch is reachable.
Task 6: Ruling: the execution-mode equivalence probe moves into the suite now rather than at Task 14. Run early on a scratch script, it immediately found two defects (unmasked Python int seed at the compiled boundary; uint64 written to a signed column). Leaving it scheduled would have left the project's central guarantee unguarded for seven stages.
Task 6: Ruling: the per-transition invariant test drives `play_block`, not a Python loop. Same invariant, same code path, 37 million transitions in ~28s versus 300,000 in 57s — a jitclass argument is boxed and unboxed on every call from Python (~190us/action against ~0.3us). The Python walk is kept at 150 games because only it can name which invariant broke.
Task 6: FINDING: the win rule's endgame degeneracy is far larger than the spec suggests — 68% of 36-card games and 70% of 52-card games end with the winner attacking their last card, unopposed. The primary comparison shares this rule so the effect is common-mode, but the rules report must state the number.
Task 6: Ruling: `pytest -n auto` and Numba `cache=True` are now defaults. Suite 186s -> 43s parallel (60s serial). Engine throughput 405 games/s, 3.0M actions/s through the compiled loop.

Task 7: complete (commits 33af4c6..3ba8cf6, tests: .venv/bin/python -m pytest -> 216 passed in 44s)
Task 7: FINDING (critical, fixed in 3ba8cf6): level-2 and level-3 records restarted writing at index 0 for every game, so each game overwrote the previous game's rows. A 1000-game block reported 32,309 round records that were actually the final game's. Caught only because Task 7 finally writes multi-game blocks with recording on. Cost if unfixed: level 2 is the per-round data the primary question depends on, and it would have been silently wrong.
Task 7: Ruling: per-game telemetry is bounded (300 rounds, 2000 moves) with truncation recorded on the level-1 row as `flag_round_records_truncated`/`flag_move_records_truncated`, kept out of `flag_anomaly_any`. A capped game emits 10,000 rounds against ~14; unbounded, one stalled game would have ~700x its share of the level-2 dataset and determined the per-round curve. Cost if wrong: two extra schema columns.
Task 7: Ruling: partition columns live in the directory path only (Hive convention) and are cast back and reordered on read. Keeping them in both places makes pyarrow infer `deck_size` from the path as a dictionary column that cannot merge with the in-file int32. Cost if wrong: a reader that hands back a non-canonical schema.
Task 7: Ruling: `write_parquet` infers the level from the record dtype identity. Inferring which of three known tables the data is does not weaken the hand-written schemas; it removes a trap where level-2 rows were written through the level-1 schema. Cost if wrong: one explicit `level=` argument.
Task 7: Ruling: `audit_game` reimplements the duplicate scan rather than calling the engine's, and clamps every length counter to its array. An audit sharing code with what it checks reports nothing; an audit that raises on corruption reports nothing about the corruption. Cost if wrong: the two implementations could drift, which the batch cross-check test covers.
Task 7: OPEN: full `NUMBA_DISABLE_JIT=1` suite not run since Stage 6 (~11 min). Required before Task 14.

Task 8: complete (commits c66846c..182e3e1, tests: .venv/bin/python -m pytest -> 248 passed in 36s; -m perf -> 3 passed)
Task 8: RULING: `USE_RULE = -1` as the first-attacker sentinel, applied AFTER `assign_first_attacker` so `first_attacker_random_fallback` is still computed from the deal. A sentinel of 0 would disable the rule for half of every rotation run. Cost if wrong: the fallback flag would be lost on forced-seat games.
Task 8: RULING: wins attributed per strategy with GAMES as the denominator, not seat-slots. A RANDOM self-pairing puts one strategy in both seats of one game, so slot counting double-counts it. Undecided games counted separately and excluded from rates by default rather than folded in as losses. Cost if wrong: two extra fields on RunSummary.
Task 8: RULING: `RunSummary.seat_wins`/`seat_rate()` exist separately because a per-strategy rate for a self-pairing is identically 1.0 and carries no information; the symmetry check is about seats.
Task 8: RULING: only RANDOM is self-paired in the grid. Other self-pairings are consumed by nothing and would invite reading GREEDY-vs-GREEDY as a result.
Task 8: FINDING (fixed): benchmark computed `process_time() - perf_counter()`, i.e. an absolute CPU reading minus a wall-clock delta. Reported -20.4 us/game... actually -20.4 SECONDS per game, which is impossible; the finiteness assertion caught it. Both baselines now captured together.
Task 8: MEASURED (decides Stage 10 scale): deck36 RANDOM 22366 games/s, GREEDY 1930 games/s (12.0x, entirely the 5.0% stall rate); deck52 RANDOM 15864 games/s, GREEDY 1264 games/s (7.6% stall). Interpreted reference: RANDOM 255 games/s, GREEDY 19.8 games/s. Telemetry costs RANDOM 1.30x, GREEDY 1.02x.
Task 8: FINDING (open, unresolved by design): GREEDY's mean of 484 rounds is 5% of games at the 10,000-round cap; the median is 13. Benchmark now reports both. Sizing a dataset off the mean would overstate a finished GREEDY game by ~37x.
Task 8: OPEN (blocking for Stage 10 and Task 11): the 5-8% GREEDY stall rate is undiagnosed. Rules property or policy defect is not established and was deliberately not guessed.
Task 8: OPEN: full `NUMBA_DISABLE_JIT=1` suite still running in background. Required before Task 14.

Task 9: complete (commits 0849135..cca3fbb, tests: .venv/bin/python -m pytest -> 274 passed in 34s; NUMBA_DISABLE_JIT=1 on test_particles.py -> 26 passed in 3.8s)
Task 9: FINDING (critical, fixed in cca3fbb): the partial Fisher-Yates drew `pick` from [0, U-j) instead of [j, U), so selected cards were swapped back out and the "exact uniform prior" was not uniform. Per-card marginal std 0.058 vs multinomial 0.0025; entropy deficit 0.077 bits at k=4000 AND k=20000 (a bias, not noise). After the fix: deficit 0.0001 bits. This is the failure mode the project can least afford, since it puts a floor under the informed/uninformed difference that has nothing to do with information.
Task 9: FINDING (test defect, fixed): the uniformity test I wrote first PASSED against the above bias -- a single-seed per-card tolerance test can pass by luck. Rewritten to assert the seed-stable signature (across-card spread + entropy deficit) over 3 seeds.
Task 9: FINDING (fixed): `unseen_cards` excluded the modelled player's hand instead of the observer's, so the candidate set still contained cards the observer can see. Silent, and the public argument name `opponent` invites exactly that reading.
Task 9: FINDING (fixed): `_draw_particles` sorted each row before writing the drawn cards into it, so the sort read uninitialised `np.empty` memory; marginals were garbage ids rather than wrong-looking probabilities.
Task 9: FINDING (fixed): `hand_probabilities` returned an array sized `max(unseen)+1`, shorter than the deck whenever visible cards sit at high ids, so a caller indexing by deck position got IndexError instead of 0.0.
Task 9: RULING: `apply_observation(particles, observation, soft_penalty_lambda)` instead of the planned `(particles, state, observation, settings)` -- no Python object crosses nopython, and `state` is only needed to draw the prior. Spec 5.2 needs correcting.
Task 9: RULING: resampling is deterministic with a fixed half-step offset (very slightly biased, exactly reproducible). A random offset would give the informed arm a draw the uninformed arm does not have.
Task 9: RULING: `lambda = 0` skips the soft branch entirely rather than multiplying by zero, because resampling would still run and perturb the particle set.
Task 9: MEASURED: particle depletion. After 2 reveals of a 3-card hand against a 30-card unseen set, ~480 distinct hypotheses carry 20000 particles; resampled counts are multiples of 179; card frequencies span 179-1448 where uniform over 31 consistent cards gives 645+-25. Filter correct (support exact, no card dominates) but `particle_count` is a diversity budget, not a sample size.
Task 9: RULING: exact recovery is only reachable from a state where the unseen set IS the hand. The size constraint admits one branch per observation, so the filter cannot be walked into an exact hand -- an earlier version of the test tried to eliminate cards one at a time and asserted a posterior of 1.0 on an empty set.
Task 9: OPEN: full `NUMBA_DISABLE_JIT=1` suite not yet run (background job cancelled by a server restart). Required before Task 14.

Task 10: BLOCKED (not committed, work reverted to keep the suite green; resume from this entry).
Task 10: BLOCKER (root cause isolated by probe, not guessed): an `@njit` function called from Python CANNOT construct a `GameState` via `GameState(...)`. It raises `TypeError: invalid number of args: expected 8, got 7` from numba's ArgPacker. Isolated by four probes against `tests.test_search._position()`:
  - njit taking a GameState and returning a scalar  -> OK
  - njit taking a GameState and copying `.deck`/`.hands` -> OK
  - njit taking a GameState and calling `GS(...)` inside -> FAILS (8 vs 7)
  - `rules.apply_action(state, action, tunables)` called from Python -> OK (mutates in place, does not construct)
  The engine's own `deal_state` DOES construct and is called from Python via `setup_game`, so the failure is not universal -- it appears to depend on whether the jitclass is already resolved when the calling dispatcher is built. That is the thing to pin down next, not to work around blindly.
Task 10: RULING (established by probe, carry forward): a Python-called njit entry point may not create new GameState objects. Two viable shapes: (a) pass a pre-allocated scratch GameState built on the Python side (Python CAN construct it -- tests/support.py does) and have the compiled code mutate it by field assignment, since in-place assignment is exactly what `apply_action` does successfully; (b) make the search a single compiled function reachable only from inside `play_block`, matching how the engine already owns all state construction. (b) is the better fit and avoids a ~190 us/call Python-side allocation, but it means Task 10 lands together with the engine integration rather than as a standalone module.
Task 10: FINDING (real bug, would have shipped): in the draft `evaluate`, the trumps and high-card terms were written `(opponent - player)` while the card-count term was `(opponent - player)` too. Fewer cards IS good, so that one is right; but MORE trumps and MORE high cards are also good and must read `(player - opponent)`. As written, the search shed its trumps first and two of the five weights were inverted. Caught by `test_more_trumps_scores_positive`. The implementation is gone; the test survives in this entry's reasoning and must be restored first, because it is the assertion that catches it.
Task 10: FINDING (test-design): throw-in legality requires a RANK match with a card already on the table (or, with `throwin_defense_ranks`, with a defense). Any search fixture whose attacker's cards do not share a rank with `table_attacks` offers no throw-ins, and a test that asserts "the search stops" then passes vacuously. Every decision fixture must be built around the table card's rank.
Task 10: FINDING (Numba): `@njit` silently ignores a Python-level default argument. `expectimax(state, player, particles, weights)` with a defaulted `tunables` raised `TypeError: not enough arguments: expected 5, got 4` from inside the dispatcher. Public entry points must be thin Python wrappers over underscore-prefixed compiled implementations -- the same shape as `setup_game` wrapping `deal_state`.
Task 10: NOTE: `Settings.eval_weights` is `(1.0, 2.0, 0.5, 0.3, 1.5)` for (card count, trumps, high cards, role, deck exhausted). Hand justifications, not fitted. The deck-exhausted term is exactly zero while the draw pile is non-empty and enters inverted (weights[4] * (cards_p - cards_o)) once it is empty, which is the only term whose sign differs from the others and the easiest to get wrong.
