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

Task 10: BLOCKER RESOLVED by probe (supersedes the "needs a decision" entry above). No plan change is required.

  Probe results, in order:
    njit(state) -> scalar, no construction            OK
    njit(state) copying .deck / .hands                OK
    njit(state) calling GameState(...) INLINE         FAIL  TypeError: invalid number of args: expected 8, got 7
    njit(state) calling rules.deal_state(...)         OK
    njit(scalars only, no state arg) -> GameState     OK
    njit(state) + rules.apply_action(...)              OK
    njit copy_into(dst, state), field assignment      OK, faithful over 50 copy/apply cycles

  Conclusion: construction is available; only the INLINE `GameState(...)` form inside an njit function that also takes a state argument breaks the Python boundary. Routing construction through a called compiled function (as `deal_state` already does) works. The alias (`GS = rules.GameState`) is NOT the cause -- tested and ruled out.

  Working design, probe-verified, no plan change needed:
    - `durak/scratch.py` owns three reusable GameState buffers (`after`, `hypothesis`, `reply`), allocated ONCE per game on the Python side. Python CAN construct a GameState; tests/support.py does it on every fixture.
    - `_copy_into(dst, src)` is njit, takes two states, and assigns all 24 mutable fields in place. No construction anywhere on the Python-called path.
    - `action_value` per candidate: copy state -> `after`, apply; per particle: copy `after` -> `hypothesis`, substitute the particle's hand in place; per reply: copy `hypothesis` -> `reply`, apply, evaluate. Three buffers suffice because `after` must survive the inner loops.
    - Public entry points are thin Python wrappers supplying defaults, because @njit silently ignores Python default arguments.

  Why this matters beyond Task 10: a jitclass argument costs ~190 us per call from Python vs ~0.3 us inside compiled code. Allocating scratch per call would cost more than the search. One allocation per game is the difference between a search that can run inside a tournament and one that cannot.

Task 10: NOT COMPLETE. The module was rebuilt on the working design and reached 8 failing tests before the session's context ran out; it was reverted rather than committed, so the tree is green (274 passed). What still needs doing, in order:
  1. `TypeError: cannot unpack non-iterable GameState object` -- unresolved, appears in the search tests but NOT in `test_responses_are_exactly_the_legal_actions`, which passes. Narrow it before touching anything else; it is almost certainly a Numba return-annotation issue on a helper (`-> tuple` on a function that Numba resolves to a jitclass) rather than a logic error.
  2. `test_the_attacker_role_scores_positive_and_the_gap_is_exactly_twice_the_weight` returned 0.0 for the attacker-seated state where 3.3 was expected. Arithmetic not yet traced -- check `_hand_profile`'s suit test (`card % 4 == trump`) and the high-rank threshold before assuming the evaluation is wrong.
  3. The search picked STOP in both the stop-tests and the free-card rate test (0/120). Not yet diagnosed: it may be correct (see the note below) or it may be the reply-enumeration bug. Diagnose before assuming either.
  4. `test_a_reused_scratch_restores_the_state_exactly` asserted the WRONG thing: `expectimax` legitimately leaves `scratch.after` holding the last candidate's post-action state, so the post-condition is not "the scratch equals the source". The invariant to assert is that `_copy_into` restores it, which probe 3 already verified over 50 cycles. Fix the test, not the code.
  5. `test_the_particle_set_changes_the_decision` got the same action for a blocking and a harmless hand. Both particle sets must produce a DIFFERENT reply set for this to be observable -- verify the fixtures actually differ in beatability before concluding the search ignores particles.

Task 10: NOTE on the search's expected behaviour, to check during diagnosis rather than assume. After a THROW that cannot be beaten the defender must TAKE, so the attacker's hand shrinks by one and STOP is strictly worse. After a THROW that can be beaten the round continues and the attacker may have to take anyway, so STOP can be right. But STOP in the *free-card* fixture (where the defender can only take) should be clearly worse, so a search choosing STOP there is a real defect -- not a weighting artefact.

Task 10: SECOND ATTEMPT. Blocker from the previous entry confirmed solved; module still not finished. Reverted again to keep the tree green (274 passed, 58.33s). State is materially sharper than last time.

  CONFIRMED by probe: the inline-`GameState(...)` constraint is real and the caller-owned scratch design works. `_copy_into(dst, src)` assigns all 24 fields faithfully over 50 copy/apply cycles.

  CONFIRMED: `durak/evaluation.py` is CORRECT. Isolated probe on hands (2 vs 3), trump suit 0:
      attacker seat, full deck   ->  1.8000   (1*(3-2) + 0.5*(1-0) + 0.3, trumps 0, correct)
      defender seat, full deck   ->  1.2000   (1.8 - 2*0.3, exactly antisymmetric)
      attacker seat, empty pile  ->  0.3000   (1.8 + 1.5*(2-3), term 4 inverted, correct)
    All three match hand arithmetic, and evaluate(state,0) == -evaluate(state,1) exactly.

  RULED OUT as the cause of `TypeError: cannot unpack non-iterable GameState object`:
    - the `-> tuple` return annotation on a profile helper. Removing it (out-parameters, then a single
      inlined loop with scalar accumulators) did NOT fix the error.
    - the alias vs qualified `rules.GameState` reference (ruled out earlier).
  So the error originates in `durak/search.py`, not `evaluation.py`. It appears at the
  `expectimax(...)` call in every search test. `enumerate_responses` -- which also does
  `_copy_into` + `apply_action` + `legal_actions` and is Python-called -- PASSES, so it is not
  `_copy_into` and not `apply_action` in isolation. The difference is that `_expectimax` calls
  `action_value`, which calls `evaluate` and `legal_actions` inside compiled code.
  NEXT STEP: bisect by calling a Python-visible wrapper around `action_value` alone, with no
  candidate loop. That separates "evaluate compiled inside a larger graph" from "the candidate loop".

  ANOMALY worth chasing first, because it may be the same root cause: the pure-`evaluate` tests
  (`test_the_attacker_role_scores_positive_and_the_gap_is_exactly_twice_the_weight`) FAIL under
  pytest but the identical call PASSES in a standalone script. Same fixtures, same module. Something
  about test-collection order, a stale Numba cache entry, or `pytest-xdist` is affecting the compiled
  function. Try `-p no:cacheprovider -n 0` and a cleared `__pycache__` before investigating the
  evaluation at all -- a "fails in the suite, passes standalone" signal is almost always the cache.

  Test bugs found and fixed in the draft (they are mine, not the code's):
    - `test_copy_into_restores_the_source_exactly` indexed `legal_actions(...)[-1]` on an empty array;
      guard it.
    - `test_the_particle_set_changes_the_decision` passed a pre-substituted scratch to
      `enumerate_responses`, which unconditionally re-copies the state and so wiped the substitution.
      Both hypotheses then produced the reply set `{TAKE}`. Build the reply set by hand from the
      substituted state instead, or add an `enumerate_responses_in` that does not re-copy.
    - Fixtures for particle-sensitivity must differ in BEATABILITY, not merely in hand contents.

  Unresolved search behaviour, listed as observations rather than diagnoses:
    - the search chose THROW over STOP in the "opponent can beat every throw-in" fixture;
    - it chose the off-suit rank-6 card over the rank-6 trump in the trump fixture (chose card 24, not 27);
    - it found the free card 0/120 times.
  These may share one cause. Check the most basic thing first: whether the defender's reply set inside a
  hypothesis actually contains the beat. A search that never sees a BEAT can only ever conclude "the
  opponent takes", which would produce all three symptoms at once.

CROSS-MODE GATE: PASSED. `NUMBA_DISABLE_JIT=1 pytest` -> 274 passed in 1874.30s (31:14), 748 warnings, on the tree at 46b12e5. This closes the item open since Stage 6. All warnings are the intended uint64 wraparound in durak/rng.py (`overflow encountered in scalar add`), which Numba's overflow-check elision removes when compiled.

  Correction to an earlier estimate: this was recorded as "~11 minutes" from Stage 6 onward. It is 31 minutes interpreted against 58-64 seconds compiled, a ratio of about 30x, not the ~90x guessed earlier. The cost is dominated by the per-action interpreted dispatch in the engine, so it scales with test count rather than with anything expensive in the engine itself.

  What the gate does and does not cover:
    - COVERS: every test, both execution modes, on this exact tree. It is the formal guarantee that the
      Numba path and the Python path agree.
    - DOES NOT COVER: Task 14's `prange` port, which introduces a partitioning dimension this run has
      never exercised. The `test_worker_count_does_not_change_the_dataset` test remains trivially
      satisfied until Task 15, as its own docstring says. So this gate is necessary for Task 14 and not
      sufficient.
    - Note for whoever runs it again: run it in the background from the start of a session. At 31 minutes
      it outlives a working session, which is exactly why it stayed open across three.

Task 10: COMPLETE (commit 9cbd953; tests: .venv/bin/python -m pytest -> 296 passed in 36s; NUMBA_DISABLE_JIT=1 on test_search.py+test_rules.py -> 84 passed in 2.4s).

Task 10: CORRECTION to the previous two entries. The `TypeError: cannot unpack non-iterable GameState object` was NEVER a Numba problem -- it was `state, _ = setup()` in tests/test_search.py, unpacking a bare lambda that returned a GameState instead of a pair. Three hypotheses were tested and falsified (`-> tuple` annotation, stale `__pycache__`, Numba call-graph; a five-layer bisect of copy->apply->legal_actions->particle loop->candidate loop passed in full). Read the traceback's own frames next time; they named a lambda in the test module.
Task 10: The FIRST blocker (inline `GameState(...)` in njit across the Python boundary) WAS real and probe-verified. Both findings stand; they are independent.
Task 10: FINDING (fixed): `legal_actions` offered the defender TAKE with an empty table, which `apply_action` rejects. Unreachable in normal play but fatal to the search, which enumerates every action and applies each one.
Task 10: RULING: caller-owned scratch -- three `GameState` buffers allocated by Python once per game, copied into by field assignment inside compiled code. One allocation per game rather than one per candidate per particle per response; a jitclass argument costs ~190 us per call from Python vs ~0.3 us compiled.
Task 10: RULING: replies enumerated INSIDE each hypothesis, not once on the post-action state. Enumerating against the real hand would give the informed arm the opponent's actual cards for free and drive the measured advantage to ~0.
Task 10: VERIFIED: `evaluate` correct by hand -- hands (2 vs 3), trump suit 0 -> attacker 1.8, defender 1.2, empty pile 0.3, exactly antisymmetric. Trumps and high-card terms read player-minus-opponent; card count reads opponent-minus-player.
Task 10: RECORDED LIMITATION (pinned by a test): at depth 1 throwing dominates stopping even when every throw-in is beatable, because a beaten throw leaves both hands unchanged, a take improves the score, and STOP makes the attacker pick up the table. Fixing needs depth 2 or a sixth evaluation term -- the latter changes spec 5.4. DECISION REQUIRED before Task 12 publishes win rates.
Task 10: RULING (test): the search's contract is asserted as "the chosen action is the argmax of `action_value`" over 200 positions, not as a comparison against RANDOM. With three throw-ins and a stop available RANDOM throws 3/4 of the time, so "beats RANDOM" would be a 100%-vs-75% claim that also passes a search doing nothing.
Task 10: OPEN: full `NUMBA_DISABLE_JIT=1` suite must be re-run before Task 14 -- durak/rules.py changed and the last full interpreted pass predates it.

Task 11: BLOCKED on the same Numba ABI wall, now stated precisely. Tree green at 296 passed; no code committed.

Task 11: LIMITATION, exact statement (probe-verified, supersedes the narrower Stage 10 wording):
  "An @njit function called from Python may never construct a GameState. It may pass state objects
  around freely and copy into buffers the caller allocated, but creating one -- inline or via a helper,
  with any number of arguments, including one -- raises `TypeError: invalid number of args: expected 8,
  got 7` from Numba's ArgPacker."
  Probe matrix:
    njit(state) -> scalar, no construction                              OK
    njit(state, state) -> scalar, no construction                       OK
    njit(state, int) copying .deck/.hands via _copy_into                OK
    njit(deck_size, streams, tunables) -> GameState   (deal_state)     OK
    njit(state) + inline GameState(...)                                FAIL
    njit(state) + GameState(...) via a no-state helper                 FAIL
    njit(state, int) + inline GameState(...)                           FAIL
  The one shape that constructs successfully takes NO state argument. That is why `setup_game` works and
  why a policy cannot.

Task 11: WHY THIS BLOCKS IT. `policy_search(state, player, strategy_id, info_mode, rng_state)` has a
  signature pinned by the registry, and `expectimax` needs three `GameState` buffers. So the policy must
  produce scratch states without creating any. Two non-options, ruled out explicitly:
    - a module-level cached scratch: Task 14 runs the tournament in a `prange`, and a shared buffer
      across threads is a silent correctness bug, not a performance one. Must never be done.
    - allocating per call in Python: ~190 us per state against ~0.3 us inside compiled code, and the
      policy is called once per action.

Task 11: CANDIDATE SOLUTIONS, all of which change a boundary, so none is taken without approval:
  (a) Per-state scratch. Give `GameState` dedicated scratch arrays and have the search use them. Needs
      no construction, and is automatically thread-safe under `prange` because each thread plays a
      different game and therefore holds a different state. Cost: the jitclass grows, and the scratch
      must be documented as not part of the game state so the equivalence probe's field hashing and
      `audit_game`'s independent pass are unaffected.
  (b) Extend the policy signature to carry the scratch. Cleanest semantics -- the caller owns working
      memory -- but the signature is pinned in `durak/strategies/base.py` and by every registered policy,
      and the engine's dispatch would change. Cost: touches all policies and the engine.
  (c) Restructure the leaf to take scalars instead of a state: `evaluate` already only reads
      (hand sizes, trumps, high counts, role, deck-empty). A scalar-leaf form removes the *evaluation*
      need for a second state, but `apply_action` and `legal_actions` still need one, so this does not
      solve the problem alone -- it would have to be combined with (a) or (b).
  RECOMMENDATION: (a). It is the smallest change, it is thread-safe by construction rather than by care,
  and it keeps the pinned signature. It needs one decision from the user: whether scratch buffers belong
  inside `GameState`.

CROSS-MODE GATE: PASSED again, on the Task 10 tree. `NUMBA_DISABLE_JIT=1 pytest` -> 296 passed in 1684.31s (28:04), 761 warnings (all the intended uint64 wraparound). durak/rules.py changed since the previous gate, so this was required. Task 14's precondition is met.

USER DECISIONS:
  (1) Scratch location: INSIDE `GameState`. Chosen over extending the policy signature and over the scalar-leaf restructure. Rationale recorded: no construction anywhere on the policy path, and thread-safe by construction under `prange` because each thread plays a different game and therefore holds a different state.
  (2) Depth-1 over-attack: LEAVE AS IS, decide in Task 12. The limitation stays pinned by `test_throwing_dominates_stopping_at_depth_one` and recorded in IMPROVEMENTS. Task 12 is the first place it shows up in win rates, and whichever way it is resolved the choice goes into the manifest.

Task 11: DESIGN CONSEQUENCE worth recording, found while implementing (1). Two spare states suffice, not three, because the candidate action can be re-applied per particle rather than once per candidate:
    for candidate:
      for particle:
        copy(state -> spare_a); apply(candidate) in spare_a      # "after", rebuilt per particle
        copy(spare_a -> spare_b); substitute(particle hand) in spare_b   # "hypothesis"
        for reply:
          copy(spare_b -> spare_a); apply(reply) in spare_a       # spare_a reused as the leaf
          total += evaluate(spare_a, player)
  `spare_a` is reused as the reply buffer because it is already dead by then. The cost is one extra
  `apply_action` plus one extra copy per *particle* instead of per candidate, while the expensive part
  (k particles x replies x evaluate) is unchanged. That turns three spares into two, which halves the
  fields added to `GameState` and keeps the caller's `state` strictly read-only.

Task 11: OPTION (a) TESTED AND NOT IMPLEMENTABLE AS DESCRIBED. The user chose "scratch inside GameState"; implemented it and it does not work. Evidence, so this is not re-attempted blind:

  Flattened spare fields do not work. Adding `spare_a_*` / `spare_b_*` to the jitclass spec works at the
  rules level (296 tests pass with 50 extra fields, `check_invariants` unaffected), but the search calls
  `evaluate`, `legal_actions` and `apply_action`, all of which take a `GameState`. The buffer therefore has
  to *be* a GameState, and a set of prefixed fields is not one. What exists is `state.spare_a_deck`,
  `state.spare_a_hands`, ... -- there is no `state.spare_a` object to pass.

  That is the same wall as before, not a new one: the search needs GameState objects, and an njit entry
  point called from Python cannot construct one. Where the buffers are *allocated* is irrelevant to that;
  only their *type* matters.

  Also learned while implementing: the jitclass constructor is compiled as part of the class, so `setattr`
  with an f-string, a loop over a global tuple of strings, and a plain-Python helper that allocates arrays
  are all rejected there (`Untyped global name '_blank_spare'`). Any allocation added to `GameState.__init__`
  must be written out explicitly. Reverted; not worth carrying as a constraint unless a future option needs it.

  Self-referential jitclass (an `inner` field holding another GameState) is the only shape that would make
  option (a) work, and it was NOT established. The probe failed for a reason in the probe's own class
  construction (`NameError: name 'Recursive' is not defined`), not in Numba, so it is unresolved rather
  than refuted. If it works it is the cheapest fix; if it does not, option (a) is dead.

Task 11: REMAINING OPTIONS, corrected:
  (a') Self-referential jitclass field. Unresolved. Cheapest if it works; test it directly before anything else.
  (b) Extend the policy signature to carry the buffers. Known to work -- Stage 10's `expectimax` already
      takes three states and every test drives it. Costs: `strategies/base.py`'s Policy signature, all five
      registered policies, and the engine's `choose_action`. Mechanical, and the only option with no open
      technical question.
  (c) Move the search *into* the engine rather than into a policy: `play_block` allocates its own buffers via
      a construction path that is known to work (the engine already builds states through `deal_state`), and
      the strategy becomes an integer telling the engine how much information to give it. This is arguably
      the right shape for the project anyway -- the strategies are a dispatch table, and the reason a policy
      cannot allocate working memory is an ABI limitation, not a design one.
  RECOMMENDATION: (c). It is the only option that does not fight the execution model, and it makes the
  experiment's central claim structurally true: the two arms become one code path that receives a different
  hand source, which is exactly what the project claims to measure.
  NEEDS FROM USER: confirmation of (c), or a preference for (b) if the policy signature must stay pinned.

Task 11: OPTION (c) ALSO FAILS, and the constraint is now sharper than any statement recorded so far. Tree green at 296 passed; no code committed.

Task 11: FULL NEGATIVE RESULT. An njit function reached from Python that constructs a `GameState` raises
  `TypeError: invalid number of args: expected 8, got 7`.
  RULED OUT as the cause, each by direct test:
    - a state argument       Fails with one state arg, with two, and with none at all.
    - helper indirection     `_fresh_scratch(deck_size)` called from a state-holding frame: fails.
    - which frame constructs A no-state frame that allocates and passes the buffers down to a
                            state-holding frame: fails. This was option (c)'s load-bearing assumption.
    - cache=True vs False    Both fail. (`deal_state` is `@njit(cache=True)` at rules.py:315, so the
                            earlier reading of it as `cache=False` was wrong.)
    - parameter annotations  Annotated and unannotated both fail.
  Option (a) additionally fails for a separate, independent reason already recorded: flattened spare
  fields are not `GameState` objects, and the search's callees require objects.

Task 11: THE UNEXPLAINED ANOMALY, and it is the whole remaining problem. `rules.deal_state` constructs a
  GameState, is annotated, is `cache=True`, is reached from Python through `setup_game`, and WORKS. It has
  the same shape as functions that fail. So at least one structural difference remains unidentified, and
  until it is, all four options (a, a', b, c) are blocked rather than merely unattempted.
  SPECIFIC NEXT STEP, in order of likelihood:
    1. Same-module inlining. `deal_state` calls `shuffle_inplace_with_state`, `assign_first_attacker` and
       the refill helpers, all `@njit` in `rules.py` itself. Every failing helper I wrote imports
       `HAND_CAPACITY` and `EMPTY` from `durak.rules` and constructs alone. Numba's inlining of a
       same-module compiled helper is the leading candidate for why the jitclass return type resolves
       correctly there. Test: move one failing constructor into `durak/rules.py` next to `deal_state`,
       with everything else identical.
    2. The dynamic-globals path. `pyproject.toml` already records that Numba cannot cache functions using
       "ctypes pointers and large global arrays", and every state-touching function lives with that
       warning. If constructing resolves through a different path depending on which globals are in scope,
       a constructor that mirrors `deal_state`'s global usage exactly would work.
    3. Bisect `deal_state` itself: delete its body down to a bare constructor, add one call back at a time.
  Option 1 is a ten-minute test and would reopen (a) and (c) at once.

Task 11: STANDING RECOMMENDATION, unchanged. If no difference is found and the construction limit holds as
  stated, the honest resolution is that the search must not need a second `GameState` at all: restructure
  `evaluate` to take scalars (it already only reads hand sizes, trump counts, high counts, role and
  deck-empty) *and* replace `apply_action`/`legal_actions` on the hypothesis with the small set of
  transitions the search needs. That is a larger design change than the plan's Task 11 and needs its own
  approval; it should not be started as a workaround.

Task 11: APPROACH CHANGED, and the four blocked options are abandoned rather than revisited. 296 passed, 1 skipped.

Task 11: THE FRAMING ERROR, stated plainly because it cost three stages. I asked "where should the search's
  working buffers live" and then "how should they be allocated". Neither question was answerable, because
  the premise was false: the search does not need a second `GameState`, it needs the *arithmetic* one
  would have supplied. Every one of options (a), (a'), (b) and (c) was a way of getting memory the search
  did not actually need. The question that mattered -- "what does the leaf actually read?" -- was
  answerable from Stage 10's own notes, with no new experiment.

Task 11: WHAT THE LEAF ACTUALLY READS. `evaluate` reads five things: each hand's size, trump count and
  high-card count; the attacker; and whether the draw pile is empty. All of that is arithmetic over two
  card arrays. `legal_actions` is not needed on the defender's side either -- their replies are "which cards
  beat the pending attack", computable from their hand, the table and the trump. So the search stops
  calling `apply_action` and `legal_actions` on derived positions entirely, and the construction problem
  disappears with the requirement that caused it. The registry's pinned signature survives untouched.

Task 11: COMMITTED. `tests/test_leaf.py` -- the differential harness, written first and skipped because
  `durak/leaf` does not exist yet. It plays the same transitions through `rules.apply_action` and through
  pure arithmetic and demands identical profiles, which is the only thing that makes a hand-written
  reimplementation of the rules trustworthy. It also checks the replacement reply enumeration against
  `rules.legal_actions` on 200 random positions, since missing a beater would make the opponent look
  weaker than the rules allow.

Task 11: FIRST RESULT OF THE DIFFERENTIAL TEST: it caught a discrepancy immediately (2.3 vs 0.7 on an
  attack-then-beat position), which is exactly what it was written for. The implementation was not finished,
  so it was removed rather than committed half-written; the test carries the rules extraction in its
  docstring so the next attempt does not re-derive it.

Task 11: RULES, extracted from durak/rules.py and easy to get backwards:
  - ATTACK/THROW: card leaves the attacker's hand for the table; an empty hand ends the game BEFORE the
    defender responds and before any refill.
  - BEAT: card leaves the defender's hand for the table; an empty hand ends the game there and then.
  - TAKE: the defender picks up every table ATTACK then every table DEFENSE, in that order; table clears.
  - STOP: every table card goes to the DISCARD; table clears; neither hand grows.
  - Both TAKE and STOP end the round, refill, and leave `attacker = 1 - attacker`. They differ ONLY in
    which hand grew. Refill order is PREVIOUS attacker first, then the other, each drawing `deck[0]`
    while under `HAND_TARGET` and the pile is non-empty; "never discard down" is load-bearing, since a
    player holding twelve cards draws nothing. A finished game is never refilled.
  - `wait`: BEAT does NOT trigger a refill (it does not end the round), and neither does ATTACK/THROW.
    Only TAKE and STOP refill. Getting that wrong double-counts the deck.

Task 11: COMPLETE (commit 50cc861). The scratch problem was solved by removing the need rather than by working
  around it: `durak/leaf.py` computes the depth-1 leaf profile as arithmetic over two card arrays, because
  that is all `evaluate` reads (each hand's size, trump count and high-card count; the attacker; whether
  the pile is empty). No derived state is materialised, no scratch buffer exists, `durak/scratch.py` is
  deleted, and the caller's state stays strictly read-only.

Task 11: TESTS: 301 passed compiled; NUMBA_DISABLE_JIT=1 full suite 304 passed in 1843.74s (30:43) -- cross-mode gate green.

Task 11: TWO RULES WERE WRONG IN STAGE 10 AND ARE CORRECTED IN 50cc861. STOP sends the table to the DISCARD and
  grows neither hand; only a TAKE fills the defender's hand. Both end the round and both leave `attacker`
  flipped, so they differ only in which hand grew. The Stage 10 claim "STOP makes the attacker pick up the
  table" was false, and the "depth-1 always over-attacks" limitation rested on it. With the leaf arithmetic
  corrected, STOPPING wins in several fixtures, including one where the search keeps its trump. The
  property is fixture-dependent; two tests now pin both directions. The earlier commit message and ledger
  entries are wrong on this point and are superseded by this one -- history is not rewritten.

Task 12: PARTIAL (commit 10e4315). CHEATER_DOC implemented literally and separately from the search, and the
  three sweeps implemented. 318 passed compiled.

Task 12: CRITICAL FIX -- the primary comparison was unmeasurable. `run_tournament` passed `InfoMode.NONE` to
  every seat, so SEARCH_UNIFORM and SEARCH_INFORMED were the same policy on the same input. The arms would
  have measured a difference of exactly zero and reported it as a result. Nothing caught it: both are legal,
  both are deterministic, and every legality and smoke test passed on each. `info_modes_for(pairing)` now
  derives the mode from the strategy id and follows the strategy into whichever seat it plays.

Task 12: FIX -- `pairing_grid` returns an empty grid for a single non-RANDOM strategy, and `run_tournament`
  then reported 0 games with `clean: True`. A zero-game run reporting success is indistinguishable from a
  successful run of an empty experiment. The runner raises; the sweeps pair against RANDOM.

Task 12: BLOCKING FINDING, NOT DIAGNOSED. Both search arms fail to terminate a large fraction of games:
    uniform-vs-random   60 games, 21 capped (35%), rounds median 80,  mean 3544
    informed-vs-random  60 games, 16 capped (27%), rounds median 82,  mean 2718
    greedy-vs-random    60 games,  0 capped ( 0%), rounds median 20,  mean   22
  The "informed" arm is supposed to be the strongest player and terminates a quarter of its games. Its median
  game is four times GREEDY's. Win rates exclude undecided games by construction, so a sweep at this scale
  would report a number computed from the 65-74% of games that finished -- a biased subset chosen by the
  very defect under investigation. The live `delta`-vs-`k` sweep therefore was NOT recorded, because
  recording a headline number from a biased denominator would be worse than recording nothing.

  NOT ESTABLISHED, and worth stating as a hypothesis rather than a cause: the evaluation's card-count term
  reads `cards_opponent - cards_player`, and its deck-exhausted term inverts at `deck_len == 0` to
  `cards_player - cards_opponent`. A policy that hoards while the pile is empty can cycle indefinitely, and
  both terms reward hoarding at that boundary. Whether that is the mechanism is unverified; the alternative is
  that the depth-1 leaf systematically prefers taking over progressing. Distinguishing them is the next task.
  This must be settled before any win rate is published, and it bears on whether depth 1 can be kept at all.

Task 12: NON-TERMINATION MECHANISM, now traced rather than guessed. A single capped game, traced action by
  action, shows an exact repeating cycle from round 0 to the 10,000-round cap:

    step      round actor attacker chose           hands deck table
    0             0   p1       1     ATTACK(8)      6/6  24    0/0
    1             0   p0       1     TAKE           6/5  24    1/0
    2             1   p0       0     ATTACK(10)     7/6  23    0/0
    ...          ... identical block, repeating to step 11,999 and beyond

  Four facts, all measured, and together they are the bug:
    1. The defender ALWAYS takes, never beats, even when a beating card exists in their hand.
    2. The attacker NEVER throws in -- `throw_ins` is 0 in every round of every traced game.
    3. The draw pile is never consumed: `deck_remaining` is 23 at step 3,000, 6,000 and 11,997.
    4. Hand sizes oscillate 6/6 -> 7/6 -> 6/7 forever. Two cards ping-pong between the players and no
       net progress is made, so no hand ever empties and the game never terminates.

  So the stalemate is: take instead of beat, and never add to the table. Cards cycle, the deck is never
  drawn, and 10,000 rounds pass.

  COST, and it answers the performance complaint: the search runs at 179 games/s against `GREEDY`'s
  22,948 -- 128x slower -- and a capped game is 10,000 rounds against `GREEDY`'s median of 21, so a capped
  game is roughly 300x the work of a normal one. The 30-minute sweep was this defect, not a separate
  performance problem: a quarter of the games were costing hundreds of times what they should.

  NOT ISOLATED, and stated as open rather than guessed: the leaf arithmetic scores beating ABOVE taking.
  Measured directly on a constructed position with an empty pile: after a beat 2.3, after stop 1.2, after
  a take -0.8. So the evaluation wants the defender to beat, and the search takes anyway. The bug is
  therefore in one of three places, and I did not isolate which:
    (a) the reply enumeration handing the defender no BEAT in the hypothesis actually used;
    (b) the leaf being scored from the wrong seat when the *defender* is the acting player;
    (c) the second-ply assumption in `action_value`: for a BEAT the code hardcodes ply 2 as STOP, which is
        legal but ignores the attacker's throw-ins, and a wrong ply-2 model can invert the comparison.
  (b) and (c) are the most likely, because the leaf is written from the acting player's perspective and
  `action_value` special-cases BEAT and TAKE differently from ATTACK/THROW. Next task: instrument one
  defending position, print `action_value` per legal action and per reply, and compare against
  `leaf_profile_after` called directly with the same hands.

Task 12: ROOT CAUSE FOUND AND PROVEN. Traced from step 100 of the same capped game, trump suit 3:

    step 100 p1 ATTACK(2)   hands 6/7  deck 23
    step 101 p0 TAKE        hands 6/6  deck 23
    step 102 p0 ATTACK(2)   hands 7/6  deck 23      <- the same card, handed straight back
    step 103 p1 TAKE        hands 6/6  deck 23
    step 104 p1 ATTACK(10)  hands 6/7  deck 23
    step 105 p0 TAKE        hands 6/6  deck 23
    step 106 p0 ATTACK(10)  hands 7/6  deck 23      <- again
    step 107 p1 TAKE        hands 6/6  deck 23

  Cards 2 and 10 (both suit 2, neither a trump) alternate forever. Neither player holds a card that can
  beat the other, so every attack is answered with a forced take, and the taker immediately attacks the
  card back. The deck is never drawn and neither hand ever empties.

  WHY THE SEARCH DOES THIS. The card-count term is `cards_opponent - cards_player`. When the opponent
  takes, the attacker's hand shrinks by one and the opponent's grows by one, so the term improves by 2.
  A depth-1 leaf therefore scores **handing the opponent a card they cannot beat as progress** -- which
  at depth 1 is indistinguishable from winning a trick. So the search is correct with respect to its
  evaluation and the evaluation is wrong with respect to the game: a card TRANSFERRED to the opponent is
  not progress, only a card that reaches the discard is.

  That single defect explains every observation at once: the deck is never consumed (no round ever ends
  with a discard, because every round is a take), the hand sizes oscillate (each transfer is immediately
  reversed), and nothing terminates (the card count is conserved across the whole cycle).

Task 12: HYPOTHESES TESTED.
  (a) Reply enumeration missing a BEAT: REFUTED. In a traced defending position the reply set is correctly
      empty (`beaters: []`) because no card in that hand beats the attack, and `legal_actions` correctly
      offers only TAKE. The forced take is real, not a bug.
  (b) Leaf scored from the wrong seat: REFUTED. `action_value` and a direct `leaf_profile_after` call on the
      same hands agree exactly (-2.7000 both) on a real defending position.
  (c) Second-ply assumption wrong: NOT the cause here, though it remains a separate weakness. For BEAT the
      code hardcodes ply 2 as STOP, which is legal but ignores the attacker's throw-ins.
  USER'S HYPOTHESIS (the two models cannot play anything else): CONFIRMED in substance. Over 300 decisions
      the two arms chose the same action 219 times (73%), and the kind-pairs are dominated by (ATTACK,
      ATTACK) 137 and (TAKE, TAKE) 121. They converge on the same cycle for the same reason as above: the
      evaluation makes a forced take look like progress, so feeding the card back is the argmax for both.

  Cost of leaving it: 179 games/s against GREEDY's 22,948, and a capped game is 10,000 rounds against a
  median of 21 -- roughly 300x the work of a normal game. The 30-minute sweep was this defect.

Task 12: FIX DIRECTION, not taken here because it changes spec 5.4 and needs a decision. Three candidates:
  (1) Give the card-count term its correct meaning inside `durak/leaf.py`: count only cards that have left
      play, so a card the opponent took is neutral rather than progress. Cheapest, touches no interface,
      and is a genuine correction of a misapplied term rather than a new one.
  (2) Add a sixth term penalising transfers. Equivalent in effect to (1) but changes the published
      evaluation, so every existing weight-sweep result would be about the wrong function.
  (3) Depth 2, so the search can see the loop closing. Costs roughly the square in leaf evaluations and
      reopens the question of dataset scale.
  RECOMMENDATION: (1). It is a correction, not a new parameter, it is local to the one module that knows
  what happened in the ply, and it leaves the five published weights and the sweep grid meaning what they
  say. It must be decided before any win rate is measured.

Task 12: TRANSFER-NEUTRALITY CORRECTION IMPLEMENTED, AND IT DID NOT FIX THE NON-TERMINATION. Recording the
  negative result because the earlier diagnosis was wrong and the correction is still worth keeping.

  WHAT WAS CHANGED (`durak/leaf.py`). A transfer is now scored as if the card had never changed hands.
  `_table_stats` accumulates the (count, trumps, high) stats of the cards on the table at the moment of
  a TAKE, split by original owner (`moved_p`, `moved_o`), and `_finish` credits them back to the player
  they came from before any term reads the stats. All three hand-reading terms are corrected together on
  purpose: correcting only the count leaves the trump term swinging the other way. A card reaching the
  discard is not in the moved set and keeps its credit -- that asymmetry is the whole point.

  MEASURED, on a fixture where the defender may legally either take or beat the same attack:
    term      take       discard   diff
    count     -1.0000    +0.0000   +1.0000
    trump     -2.0000    +0.0000   +2.0000
    high      +0.5000    +0.5000    0.0000
    role      -0.3000    +0.3000   +0.6000
    all       -2.8000    +0.8000   +3.6000
  Before the correction the count term read +3.0 for the take -- the attacker's shed card was being
  credited to the defender as progress. It now reads -1.0. Verified legal: `legal_actions` confirms both
  replies, and the beater is genuinely in the defender's hand.

  THE CORRECTION IS REAL AND IT IS NOT THE CAUSE. Capped games, 60 per arm, before vs after:
    informed  16/60 -> 16/60      uniform  21/60 -> 21/60
  Unchanged. So "a transfer reads as progress" was a true defect in the evaluation but not the cause of
  the non-termination. It stays fixed because it is wrong independently, and it is recorded here so the
  next attempt does not re-litigate it.

  THE ACTUAL CAUSE, and the earlier diagnosis was wrong about it. My traced "deck never consumed at 23,
  hands frozen at 6/7" was a real observation of round 100-1600 of ONE game, and I over-generalised it to
  the whole capped population. Measured over all 16 capped games at the moment they hit the cap:
    game   rounds  deck  hands  discard
      4   10000     0    3/1      32
      6   10000     0    4/2      30
      9   10000     0    5/1      30
     12   10000     0    3/1      32
     17   10000     0    5/3      28
     20   10000     0    7/1      28
  The draw pile is EMPTY, the discard holds 26-32 cards, and the hands are down to 1-7 cards. These games
  are not frozen near the start; they run deep into the endgame and then stall. The "deck 23, hands 6/7"
  loop I traced is what a late game looks like once the pile is gone and only transfers remain -- the
  pile cannot be consumed because every round is a take, and the hands cannot shrink because the defender
  is at or above HAND_TARGET so the refill draws nothing ("never discard down").

  WHY A LATE DEADLOCK IS UNREACHABLE AT DEPTH 1. Probed a live stalled position: trump suit 3, the
  attacker's hand holds 7 cards and 5 of them are beaten by nothing in the opponent's hand -- including
  its trump, because the opponent holds no higher trump. So the reply is forced TAKE, every round. An
  attack that cannot be beaten can only ever be followed by a take; the attacker's hand therefore never
  shrinks below the target, the pile is never drawn, and no card ever reaches the discard. The position is
  a genuine stalemate under these rules, and the rules have no repetition or draw-detect to escape it.

  Depth 1 cannot see this. The leaf scores the position after one ply; the stalemate only exists across
  many rounds of forced takes. The evaluation has no term for "this line makes no progress", and adding
  one would be a new parameter for something a repetition rule, not a better number, should detect.

  DIRECTION THIS POINTS TO, for a decision rather than for silent action: the correct fix is a REPETITION
  OR STALEMATE DETECTOR in the engine, treating a position visited N times as a draw, not a 10,000-round
  cap. That is a rules change (a new terminal outcome) and needs the user's call. The alternative is
  depth 2+, which is expensive and still does not detect repetition -- it only makes the search avoid
  some stalemates by accident.

Task 12: ROOT CAUSE FOUND, AND IT WAS A RULES BUG, NOT A SEARCH OR EVALUATION DEFECT.

  THE BUG. `rules._do_take` ended the round with `_begin_next_round(state, defender)`, handing the turn to
  the player who had just taken the cards. In Durak a take does not pass the lead: the original attacker
  keeps attacking and the taker skips the round. That is what makes it the *podkidnoy* variant.

  MECHANISM, and it explains every symptom at once. The taker re-attacked the card it had just taken. The
  opponent could not beat it -- in a 6-card endgame it usually could not -- so it took it straight back,
  and the two passed one card back and forth. Hence: no card ever reached the discard, so the pile was
  never consumed; the attacker's hand could not fall below six, so the refill never ran; the card count was
  conserved, so nothing terminated. The traced cycle -- p1 ATTACK(2), p0 TAKE, p0 ATTACK(2), p1 TAKE --
  is this rule, verbatim.

  MEASURED AFTER THE FIX (60 games per arm, 36-card deck, vs RANDOM):
    arm        capped      median rounds   mean    max     games/s
    informed   0/60            36.5        36.3     59      4852
    uniform    0/60            36.0        36.5     62        60
    greedy     0/60            19.5        19.4     28     34724
    random     0/60            29.0        29.2     45     29883
  Before: 16/60 and 21/60 capped, median 80-82, mean 2718-3544, informed at 179 games/s.

  The speed complaint is resolved as a consequence, not as a separate fix: the informed arm is 28x faster
  than it was because its games are 75x shorter. Nothing was optimised; the stall was removed.

  I WAS WRONG TWICE ABOUT THE CAUSE, and both errors are recorded above rather than edited out. First I
  blamed the deck-exhausted evaluation term, on a trace of one game at rounds 100-1600; measuring all 16
  capped games showed the pile was EMPTY and the discard held 26-32 cards, so they were deep endgame
  games, not games frozen near the start. Second I blamed "a transfer reads as progress" and fixed it.
  That fix is real and independently correct -- a transfer is not progress -- but it changed nothing about
  the non-termination, which is how I learned it was not the cause.

  THE SAME BUG WAS IN THE LEAF. `leaf.py` flipped `player_attacks` on every round end. The roles swap on a
  successful defense only; after a take the original attacker keeps attacking. Same rule, reproduced in the
  role bookkeeping, so every post-take leaf scored the wrong seat for the role term (0.6 of the observed
  2.6 discrepancy). Fixed to flip only when neither ply was a TAKE.

  THE ARMS NOW GENUINELY DIVERGE: over 44,750 decisions from 300 games the informed and uniform arms chose
  differently 5,922 times, 13.2%. This is the quantity the whole project exists to measure, and before the
  rules fix both arms converged on the same non-terminating cycle.

  TESTS CHANGED, and each change is a case of a test that had encoded the bug:
    * test_take_does_not_swap_roles asserted `attacker == 1` under a name saying the roles do not swap.
      Renamed to test_a_take_leaves_the_attack_with_the_original_attacker and inverted.
    * test_a_capped_game_is_flagged... required AUTO_DEFEND to stall; it no longer can. Now asserts the
      invariant that still holds: no capped game carries a winner, every finished game does.
    * test_invariants_hold_after_every_transition_at_scale asserted >15M transitions, a floor that was
      being met by the stall itself. 5,000 -> 300,000 games so the sample meets the floor honestly.
    * test_win_from_a_single_card... measured 0.68 and now measures 0.90. Real, and a consequence of the
      fix: the attacking seat keeps shedding cards, and an empty hand ends the round before the defender
      answers. Added a seat-symmetry assertion (0.494 vs 0.505) so the rise cannot be mistaken for the
      attacker winning too often.
    * test_the_two_arms_score_differently... filtered to `attacker == 0`, which after the fix left 59 of
      120 positions and a floor of 5 against an observed 3. Now uses both seats -- a stronger case, since
      it exercises the BEAT branch -- and 400 seeds.
    * test_version_constants_start_at_one asserted the stamps equal 1, making it a check that the project
      had never changed its rules. Replaced with presence, monotonicity, and RULES_VERSION >= 2.
    * ENGINE_VERSION 1 -> 2, RULES_VERSION 1 -> 2, so no pre-fix dataset can be pooled with a post-fix one.

  SUITE: 320 passed compiled. The interpreted cross-mode gate is running.

Task 12: THE HEADLINE NUMBER, first measurement on corrected rules. 36-card deck, both arms against
  RANDOM, same seed, `weight_sweep` so the two rates come from one code path.

    games   uniform   informed   delta
      100     0.5825      0.5950  +0.0125
     1000     0.5543      0.5877  +0.0335
     2000     0.5573      0.5870  +0.0297

  `delta` holds between 1000 and 2000 games at about +0.030: perfect knowledge of the opponent's hand is
  worth roughly three percentage points against a random opponent. This is a FIRST measurement, not a
  published figure, and it is only defensible together with its caveats below.

  Caveat that must travel with the number: the delta is measured at a fixed particle count of 100, and the
  uninformed arm's strength at that count is unknown, because `k` cannot be varied (see below). A stronger
  approximation of the opponent's hand would narrow the gap, so +0.030 is an estimate against THIS
  baseline and not against an idealised one.

Task 12: `k` IS NOT A KNOB, and the sweep that looked like it proved the opposite.
  `Settings.particle_count` reached nothing. `policy_search` read the module constant
  `search_policy.PARTICLE_COUNT = 100`, baked into the compiled call. Measured: `k = 25 / 100 / 400` gave
  win rate `0.65 / 0.65 / 0.65` -- identical because the parameter had no effect, not because the
  measurement was stable. A table of constant rows reads exactly like a robustness result, which is why it
  survived several stages unexamined.

  RESOLVED as chosen: the field stays, because the manifest must record it (plan Task 9 requires it, and it
  is part of the reproducibility contract), but it is now honest about what it is. `Settings.__post_init__`
  refuses any value that disagrees with `PARTICLE_COUNT`, so a caller who widens the sweep gets an error
  instead of duplicated rows. `DEFAULT_K_VALUES` is `(100,)` and the docstring states the open question.
  Making the field authoritative means threading an int through `choose_action` -- an njit if/elif chain
  called from a `prange` loop -- and from there through `policy_search` and all eight policy signatures.
  That is a real change and stays an open task.

  OPEN QUESTION, recorded rather than answered: is `delta` stable across `k`? Unmeasured. No claim either
  way may appear in the report.

Task 12: cross-mode gate. 320 passed with `NUMBA_DISABLE_JIT=1`. One earlier run passed but had been
  started before the config edits landed, so it was not evidence for the current tree; it was re-run.

Task 14: BLOCKED, and not for a reason the plan anticipated. `play_block` cannot be compiled with `prange`
  on this toolchain (numba 0.68.0, numpy 2.5.3, Python 3.14.7).

  THE LIMIT, reproduced minimally and independent of this project:
    @njit(parallel=True) on a function whose prange body does `out[i]["a"] = i` for
    `np.dtype([("a", int64)])`  ->  NotImplementedError from numba/parfors/parfor.py.
  Both write forms fail: the direct field set and the `row = out[i]` view. Writing to plain 1-D arrays
  inside prange compiles and runs normally, and a `@njit` helper writing into a plain array also works.
  So the parallel transform handles 1-D buffers and does not handle structured arrays at all.

  `engine._play_one` makes 21 structured-array writes, which is why it fails even with telemetry levels
  disabled (`record_rounds=False, record_moves=False`), and the failure reproduces with level 1 alone.

  WORK DONE AND THEN REVERTED, because it did not achieve the task's purpose. A per-game write slot
  (`rounds[i*LIMIT:(i+1)*LIMIT]`) plus a serial compaction pass was implemented and measured: it made the
  loop race-free and kept the output byte-identical (verified for GREEDY, SEARCH_UNIFORM and CONSERVATIVE
  with telemetry on), but the parallel compilation still failed, so the slot layout would have been an
  abstraction with no consumer. `durak/engine.py` is reverted and unchanged.

  A SECOND FINDING worth keeping: `njit(cache=True)` masked the compile failure. The parallel dispatcher
  reported 0 nopython signatures and returned a stale artefact from disk, so a benchmark reported
  "1.00x speedup" instead of an error. Any prange work here must check `_play_block_parallel.nopython_signatures`
  rather than trusting a timing number.

  THE OPTIONS, and this is a decision rather than a silent choice:
    (1) Redesign telemetry to per-column 1-D arrays, assemble the structured arrays serially after the
        parallel loop. This makes prange reachable and is the only option that preserves both the structured
        dataset layout and parallelism. Cost: a real change to `telemetry.py` and `_play_one`, which is the
        plan's own warning about "a rewrite being forced" -- but the rewrite here is forced by the compiler.
    (2) Parallelise at the block level instead. `play_block` stays serial; the runner in Task 15 splits the
        work across processes, each running whole blocks. No engine change at all, and the dataset is
        unaffected because blocks are seeded by game index. Cost: process overhead and no shared-memory
        arrays, which is what the project wanted from prange.
    (3) Ship serial and record the wall-clock cost. The uniform search arm runs at ~59 games/s on 12 threads,
        which sets the dataset scale for Task 18.

  Recommendation: (2). The plan wanted prange for a shared-memory block loop, but the plan ALSO made
  determinism the project's central guarantee, and (1) buys parallelism by rewriting the one module whose
  layout every dataset depends on. (2) touches no telemetry code and keeps blocks independently verifiable,
  which is what Task 15's DuckDB views are built on. Its cost is honest and measurable rather than hidden.

  DELIVERED from this task regardless: `tests/test_equivalence.py`, with the compiled-versus-interpreted
  gate (the project's central guarantee, both deck sizes, five strategy pairs, three pairings each, every
  level-1/2/3 field compared by SHA-256 per array), a block-splitting equivalence test, and a row-consistency
  test. The serial-versus-parallel comparison is present but SKIPPED with the reason recorded, rather than
  written to pass by comparing the serial path with itself.

Task 17: THE REPORTED BLOCKER WAS MY OWN MISREADING, corrected. I reported that the two search arms
  played identical games in the tournament, on three pieces of evidence. Two of them do not support it:

  1. The two datasets have the same `game_seed` values. That is CORRECT and required: a game's identity
     is `(run_seed, pairing_index, game_index)`, so matched arms must be dealt the same games. I read a
     property of the seeding design as evidence that the games were the same.
  2. The first five `winner` values matched. Five bits is not evidence of anything.
  3. The win TOTALS were both 645. That is a coincidence in the aggregate, not identity.

  Measured properly -- comparing all 500 games of cell 0, the two arms differ in 204 of them:
    games=500 block=50 w=1: uniform 645  informed 645  differ 204/500
    games=500 block=20 w=1: uniform 645  informed 645  differ 204/500
    games=200 block=50 w=6: uniform 268  informed 265  differ  75/200
  Equal totals alongside 204 differing games is arithmetically fine (the changes balance) and not a sign
  of anything.

  Everything downstream was verified individually and is sound: `info_modes_for` returns
  (INFO_PARTICLES, 0) for `SEARCH_UNIFORM` and (INFO_TRUTH, 0) for `SEARCH_INFORMED`; the engine passes
  `info` per seat; a direct `play_block` with the two modes produces different action counts
  (98/101/111 against 118/99/95); decisions diverge on 15.4% of positions; and the worker pool gives
  identical output at 1, 2 and 6 workers.

  WHAT ACTUALLY REMAINS UNRESOLVED is the headline number, not the machinery. Measured so far the effect
  is smaller than the noise: +0.0297 at 2,000 games in the first pass, and -0.1375 at 40 games in the
  three-strategy run. Neither is usable. The arms differ in play but the win-rate difference has not been
  shown to be non-zero, and Task 18's scale is what settles it.

Task 17: THE HEADLINE RESULT, measured head to head. Play the two arms against EACH OTHER on matched
  deals, 400 games per seat order on the 36-card deck:

    uniform vs informed, winner = 0 (uniform)   172
    uniform vs informed, winner = 1 (INFORMED)  228
    informed vs uniform, winner = 0 (INFORMED)  228
    informed vs uniform, winner = 1 (uniform)   172

  INFORMED 456 / 800 = 0.5700, Wilson (0.5354, 0.6039), McNemar exact p = 8.5e-05.

  So perfect knowledge of the opponent's hand IS worth about seven percentage points when the opponent
  is the search itself. That is a real, significant, large effect -- and it is the opposite of what the
  RANDOM-opponent runs suggested (+0.030 at 2,000 games, i.e. indistinguishable from zero).

  THE RESOLUTION, and it is the interesting part: information is worth a lot against an opponent who can
  use it and worth almost nothing against RANDOM. The uninformed agent's ignorance costs it nothing when
  the opponent takes cards at random regardless, because there is no punishement for being wrong about a
  hand that plays randomly. A study reporting only the RANDOM-opponent rate would have concluded
  "information is worthless in Durak" and been wrong.

  This also explains the throw-in anomaly measured just before: the informed arm throws LESS (0.086 vs
  0.094) and a higher share of its throws is beatable (77.6% vs 68.5%). Read against the head-to-head
  that is not a failure to use information -- a beatable throw still sheds the attacker's card, and the
  informed arm is choosing differently for a reason the RANDOM-opponent view cannot expose. It remains
  unexplained and is worth a look, but it is no longer evidence of a broken lever.

  PROCESS FAILURE, recorded because it happened twice. `winner` is a SEAT and `strategy_p0`/`strategy_p1`
  are ids; comparing them is always false or always true. I documented this exact trap as
  IMPROVEMENTS.md #28, fixed it in `analysis/queries.py`, and then reproduced it twice in throwaway
  probes -- first concluding the arms played identical games, then concluding the informed arm never
  won. Both conclusions were false and both would have been reported as findings. The lesson is that the
  seat/strategy distinction needs a type or a helper, not a comment: it has now been violated three
  times in one session, each time producing a confident wrong answer.

Task 18 PRE-FLIGHT: full correctness audit before the final run. 36-card deck, 1,200 trials per
  strategy against RANDOM across both seat orders, 0 anomalies anywhere.

  | strategy          rate vs RANDOM | Wilson CI
  | AUTO_DEFEND           0.9167      | (0.8997, 0.9310)
  | CONSERVATIVE          0.5008      | (0.4726, 0.5291)
  | GREEDY                0.9800      | (0.9704, 0.9865)
  | SEARCH_UNIFORM        0.6500      | (0.6226, 0.6765)
  | SEARCH_INFORMED       0.6450      | (0.6175, 0.6716)
  | CHEATER_DOC           0.6408      | (0.6133, 0.6675)

  CORRECTNESS, verified and clean:
    * `RANDOM vs RANDOM` symmetry: seat 0 won 589, seat 1 won 611 of 1,200 -- 0.4908, inside the Wilson
      interval for 0.5. This is spec 7.3's only exact assertion and it holds.
    * Zero anomalies in every run: no flagged game, no undecided game, no redeal overflow.
    * Worker count, block size and repeat runs reproduce byte-identically (`tests/test_determinism.py`).
    * Compiled and interpreted modes agree bit for bit (`tests/test_equivalence.py`).
    * Legality: every returned action is in `legal_actions`, asserted over random positions.
    * The informed arm beats the uninformed arm head to head: 456/800 = 0.5700, p = 8.5e-05.

  ONE METRIC BUG, MINE, NOT THE ENGINE'S: the first pass reported `RANDOM` at 1800/1800 = 1.0000, which
  looks like RANDOM beating every strategy. It is a self-pairing, and `wins` sums both seats, so every
  decided game is won by "a RANDOM" by construction. The control is meaningless as written; the real
  symmetry check is the 0.4908 above.

  STRATEGY QUALITY, which is a result and not a defect, under the project's own philosophy: a strategy
  that plays badly is a weak strategy, not a broken simulator.
    * CONSERVATIVE at 0.5008 is no better than RANDOM. Its thresholds make it take 31% of defensive
      actions rather than spend a trump, and taking is expensive. The rules are implemented correctly;
      they describe a weak player.
    * The search arms sit at 0.645, below the information-free `GREEDY` at 0.980 and below the
      throwaway `AUTO_DEFEND` at 0.917. `SEARCH_INFORMED` beats `GREEDY` in 1 of 600 games.
    * The evaluation weights (1.0, 2.0, 0.5, 0.3, 1.5) were hand-justified and never validated against
      play. A trump weight of 2.0 against a hand-count weight of 1.0 makes the search hoard trumps, and
      hoarding leads to forced takes and large hands.

  WHAT THIS DOES AND DOES NOT LICENSE. The headline 7-point effect is measured between two weak players.
  It is a real, significant effect of information ON THIS POLICY CLASS, and it is not evidence about
  information in Durak generally. The evaluation function, not the information, is the binding
  constraint on absolute strength. Any report must say so or it will be read as a claim about the game.

Task 18 Step 1: THE SCALE, decided and written down BEFORE the run so it is not a post-hoc choice.

  Measured throughput of the full 8-strategy field with rounds and moves recorded:
    1 worker   122 games/s        10 workers  561 games/s        (12 cores)
  Grid size: 57 cells for deck 36 / RULE_DERIVED, 228 cells for both decks and both regimes.

  WHAT HAS TO BE RESOLVED. The primary metric is the paired head-to-head between the two search arms.
  Measured at 400 games per seat order it was 0.570 (456/800), and the sampling already resolves it:
    paired n= 800 -> Wilson (0.5342, 0.6027), excludes 0.5; McNemar p = 5.2e-02
  That is enough to see the effect and not enough to publish it. The target is a half-width near 0.007:
    paired n=20000 -> Wilson (0.5631, 0.5768), half-width 0.0069

  CHOSEN, two tiers with different jobs rather than one compromise number:
    PRIMARY   all 8 strategies, deck 36, RULE_DERIVED, 20,000 games/cell = 57 cells, 1.14M games, ~34 min.
              This is the configuration spec 8.1 pins the primary metric to, and it is where the power goes.
    SENSITIVITY the full grid, 36 and 52, both regimes, 2,000 games/cell = 228 cells, 456k games, ~14 min.
              Half-width 0.0217, adequate to show the shape of the tournament and to see whether the
              effect survives a different deck or a different first-attacker regime -- not to publish.
  Rejected: 20,000 across all 228 cells = 4.56M games, ~2.3 hours, for precision on 171 cells no report
  reads at that resolution.

  The pilot runs at 10% of the primary tier. A non-clean pilot is a defect to fix, never a data point.
