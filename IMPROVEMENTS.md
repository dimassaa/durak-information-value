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

- **Stage 08** — `GREEDY` fails to terminate in 5.0% of 36-card games and 7.6% of 52-card games
  (measured over 2,000 and 1,000 games respectively; `AUTO_DEFEND` was 5% in Stage 6). This is
  **not yet diagnosed**. Either it is a property of the rules, in which case every win rate needs a
  stated treatment of undecided games and Task 11's definitions must settle it; or it is a defect in
  `GREEDY`'s defence rule. Stage 08 deliberately did not guess which. It must be settled before any
  win rate is published, and before Stage 10's dataset scale is chosen, because a 5–8% stall rate
  inflates the cost of every run containing those policies by an order of magnitude.

- **Stage 08** — `ROTATION_BALANCED` calls `play_block` once per game because the first-attacker
  override is a single parameter but the rotation alternates on the game index. Correct, and
  roughly 3–5x slower than necessary: per-call overhead dominates at ~40 µs/game. The fix is to
  pass the *regime* into `play_block` and resolve the seat inside the compiled loop. Deferred to
  Task 15, where per-call overhead is the whole problem being solved.

- **Stage 08** — the benchmark runs with `-n 0`. Under xdist each worker is a separate process, so
  `time.process_time()` measures the worker rather than the benchmark and the numbers are
  meaningless. The `-m perf` marker is excluded from the default suite by `addopts`, but nothing
  stops someone running `-m perf` in parallel and getting plausible wrong figures. The benchmark
  module should detect a non-zero xdist worker count and skip with a clear message.

- **Stage 08** — spec section 8.4's rotation-balanced regime required an engine feature
  (`first_attacker_override` in `deal_state`) that the spec did not mention as a rules-layer change.
  The spec's rule section should state that the first attacker is overridable for experimental
  control, since it is now part of the rules surface and someone reading only the spec would not
  know it exists.

- **Stage 09** — a single-seed tolerance test on a Monte-Carlo quantity can pass against a real bug.
  `test_the_empirical_marginal_matches_the_analytic_uniform_prior` asserted each card's marginal
  against `m/U` within 0.02 at one RNG seed, and **passed** while the prior carried a real bias
  (per-card std 0.058 against a multinomial 0.0025). At that seed the largest deviation happened to
  land inside the band. The rewrite asserts the *signature* — across-card spread and the entropy
  deficit — over several seeds. The general rule: assert a seed-stable statistical signature, not a
  per-draw tolerance, and prefer a quantity that separates signal from noise by orders of magnitude.

- **Stage 09** — particle depletion means `particle_count` is a diversity budget, not a sample size.
  Systematic resampling duplicates survivors instead of re-drawing, so after two reveals ~480
  distinct hypotheses carry 20,000 particles and the resampled counts land on a lattice of 179. The
  filter is correct there, but posterior spread is not a standard error and must not be presented as
  one. Any report quoting uncertainty from the particle filter needs a diversity diagnostic —
  distinct-hypothesis count per step — alongside it, and none exists yet.

- **Stage 09** — `informativeness` is the entropy of the *normalised* marginal, which measures how
  many bits remain to identify the unseen set and is **blind to asymmetry**: a filter that knows the
  hand exactly scores the same as one that knows nothing. Report 1 uses this as the phase curve's
  input, so the curve must be read against round index and cards-left as well, and the report should
  say so on its face rather than in a footnote.

- **Stage 09** — spec 5.2 specifies `apply_observation(particles, state, observation, settings)`.
  Implemented as `(particles, observation, soft_penalty_lambda)`, because no Python object crosses
  into `nopython` and `state` is only needed when drawing the prior. The spec's telemetry and
  interface tables should be corrected to match, so a later reader does not "fix" it back.

- **Stage 09** — the size constraint admits exactly one branch per observation, which means the
  filter cannot be walked into an exact hand; it can only start there. `particle_count` sweeps (a
  planned sweep, `durak/config.py:particle_count`) should measure *diversity* at each step, not just
  the accuracy of the final marginal, or they will report that larger k helps without capturing why
  it stops helping.

- **Stage 10 (blocked)** — an `@njit` function called from Python cannot construct a `GameState`.
  Isolated by probe: passing a state in, copying `.deck`/`.hands`, and calling `rules.apply_action`
  from Python all work; calling `GameState(...)` inside a compiled function raises
  `TypeError: invalid number of args: expected 8, got 7` from Numba's `ArgPacker`. The engine's
  `deal_state` constructs and *is* called from Python, so this is not a blanket prohibition -- it
  appears to depend on whether the jitclass is already resolved when the calling dispatcher is built.
  That dependency must be pinned down before the search is written, because every design in the
  search depends on who is allowed to create states. Task 10 is blocked on it; see the ledger.

- **Stage 10 (blocked)** — `@njit` silently ignores Python-level default arguments. A function with a
  defaulted `tunables` raises `TypeError: not enough arguments: expected N, got N-1` from inside the
  dispatcher rather than applying the default. Every public entry point that needs a default must be
  a thin Python wrapper over an underscore-prefixed compiled implementation.

- **Stage 10 (bug found, work reverted)** — the draft evaluation had the trumps and high-card terms
  inverted. Fewer cards is good, so the card-count term reads opponent-minus-player; but *more* trumps
  and *more* high cards are also good, so those must read player-minus-opponent. As written the search
  shed its trumps first, and two of the five weights were reversed. `test_more_trumps_scores_positive`
  caught it. The lesson is worth keeping regardless of the code: with an even number of terms and a
  shared "opponent minus player" pattern, sign errors in the minority of terms are invisible until a
  test asserts each term's direction individually.

- **Stage 10 (test design)** — throw-in legality requires a *rank* match with a card already on the
  table (or, with `throwin_defense_ranks`, with a defense on it). A search fixture whose attacker's
  cards do not share a rank with `table_attacks` offers no throw-ins at all, and a test asserting "the
  search stops rather than throwing" then passes vacuously. Every decision fixture must be built around
  the table card's rank, and must assert that the throw-ins it expects actually exist.

- **Stage 10 (blocked, second attempt)** — `TypeError: cannot unpack non-iterable GameState object` is
  **not** caused by a `-> tuple` return annotation. Removing it (first to out-parameters, then to a
  single inlined loop with scalar accumulators) did not fix it. The error originates in the search
  module, not the evaluation module, and `enumerate_responses` — which does the same
  `_copy_into` + `apply_action` + `legal_actions` sequence from Python — passes. Bisect next with a
  Python-visible wrapper around `action_value` alone and no candidate loop.

- **Stage 10 (blocked, second attempt)** — a pure-`evaluate` test **fails under pytest but passes
  standalone** with the same fixtures and the same module. Until that is explained, do not debug the
  evaluation: try `-p no:cacheprovider -n 0` with a cleared `__pycache__` first. "Fails in the suite,
  passes in a script" is almost always a stale Numba cache entry or test-order pollution, and chasing
  the logic first wastes the whole diagnosis.

- **Stage 10 (blocked, second attempt)** — three search-behaviour failures share one symptom: the
  search chose THROW where STOP was right, chose the off-suit card where the unbeatable trump was
  right, and found the free card 0/120 times. All three are what a search that **never sees a BEAT
  reply** would produce, since it would conclude the opponent can only take. The first diagnostic
  should be whether the reply set inside a hypothesis actually contains the beat — not the weights.

- **Stage 10 (bug in my test)** — passing a pre-substituted scratch buffer to `enumerate_responses`
  silently does nothing: the function unconditionally re-copies the state first. Both particle
  hypotheses therefore produced the identical reply set `{TAKE}`, and the particle-sensitivity test
  failed for a reason that had nothing to do with the search. Any function taking a scratch and
  populating it should say whether it overwrites it.

- **Stage 10 (confirmed, carry forward)** — the leaf evaluation is **correct**, verified by hand:
  hands (2 vs 3), trump suit 0, weights (1.0, 2.0, 0.5, 0.3, 1.5) give 1.8 attacker / 1.2 defender with
  a full draw pile and 0.3 attacker with an empty one, exactly matching `1*(3-2) + 0.5*(1-0) + 0.3`,
  `1.8 - 2*0.3`, and `1.8 + 1.5*(2-3)`. Antisymmetry is exact. Do not re-derive it.

- **Stage 10** — the full `NUMBA_DISABLE_JIT=1` suite takes **31 minutes**, not the ~11 previously
  recorded. Against 58-64 seconds compiled that is about 30x, not the ~90x guessed when the estimate was
  first made. The cost is per-action interpreted dispatch in the engine, so it grows with test count
  rather than with anything expensive in the engine itself — meaning it gets *worse* as the suite grows,
  and must be run in the background from the start of a session rather than scheduled at the end of one.
  That misestimate is why this gate stayed open across three stages: it was quoted as a 10-minute job
  and quietly never scheduled.

- **Stage 10 — RESEARCH DECISION NEEDED, this bears on the headline result.** At depth 1 the informed
  arm **over-attacks by construction**: after a throw-in that gets beaten both hands are unchanged, so
  the leaf evaluation cannot distinguish it from not throwing; after a take the attacker has shed a card,
  which scores as an improvement; and STOP in a defended round makes the attacker pick up the table,
  which scores as a loss. So throwing dominates stopping even when every throw-in can be answered.

  Two ways out, and the second changes the spec:
  (a) depth 2, or
  (b) a leaf evaluation that charges for cards committed to the table — which **adds a sixth term to
      spec 5.4's five**.

  This is not a bug fix and was not taken unilaterally. It matters because an informed arm that
  over-attacks still measures something real, but not "the value of knowledge" — and the project's
  primary question is exactly that. Task 12 should decide this before win rates are published, and the
  choice belongs in the manifest either way.

- **Stage 10** — `legal_actions` offered the defender `TAKE` with an empty table, which `apply_action`
  rejects. Unreachable in normal play, but the two functions must agree exactly because the search
  enumerates every action `legal_actions` returns and applies each one. Any future helper that enumerates
  and then applies will hit the same class of disagreement, so `legal_actions`'s docstring now states the
  invariant and what an empty array means.

- **Stage 10 — process failure worth recording.** A `TypeError: cannot unpack non-iterable GameState
  object` consumed two sessions and three wrong hypotheses (`-> tuple` annotation, stale `__pycache__`,
  Numba call-graph). It was `state, _ = setup()` unpacking a bare lambda's return in the test itself. Each
  hypothesis was a reasonable story; none was evidence. **Read the traceback's own frames** — they named a
  lambda inside the test module — before hypothesising about a library. Two of the same file's "corrections"
  were also wrong twice: a role-term assertion that was simply false about the rules (recorded as a
  "cache anomaly" instead of being derived), and a "free card" fixture where nothing was free.

- **Stage 10** — `enumerate_responses` and `responses_in` now differ deliberately: the first *overwrites*
  the caller's buffer with the state it is given, the second acts on a buffer the caller already prepared.
  A silent overwrite in a performance helper cost a particle-sensitivity test its entire reason for
  existing. Every helper taking a scratch buffer documents which it does.

- **Stage 11 (blocked)** — exact Numba constraint, probe-verified: an `@njit` function called from Python
  may **never** construct a `GameState`. It may pass state objects around and copy into caller-allocated
  buffers, but creating one — inline, via a helper, with one argument or many — raises
  `TypeError: invalid number of args: expected 8, got 7`. The single shape that constructs successfully
  takes **no state argument**, which is why `setup_game` works and a policy cannot.

  Stage 10 recorded this as "no inline construction", which was too narrow and cost an extra round of
  hypotheses. Recorded the full matrix this time so it is not re-derived.

- **Stage 11 (blocked)** — a module-level cached scratch for the search is **not** an option and should
  never be written: Task 14 runs the tournament in a `prange`, and a buffer shared across threads is a
  silent correctness bug rather than a performance one. If a future change introduces one, it will pass
  every single-threaded test in the suite and corrupt the dataset.

- **Stage 11 (option tested and refuted)** — putting the search's working buffers inside `GameState` as
  *flattened fields* does not work, and the reason is worth not re-deriving: the search calls `evaluate`,
  `legal_actions` and `apply_action`, all of which take a `GameState`, so a buffer must **be** one. A set of
  `spare_a_*` / `spare_b_*` fields is not. Where the buffers are allocated is irrelevant — only their type is.
  The rules-level part works (50 extra fields, 296 tests pass, `check_invariants` unaffected), so the failure
  is entirely at the search boundary.

- **Stage 11 (learned while implementing)** — the `GameState` constructor is compiled as part of the jitclass,
  so it rejects `setattr` with an f-string, loops over a global tuple of strings, and plain-Python helpers
  that allocate arrays (`Untyped global name '_blank_spare'`). Anything added to `GameState.__init__` must be
  written out explicitly. Noted in case a future option needs allocation there.

- **Stage 11 (process)** — the chosen option was implemented before checking it against the one constraint
  that mattered. The constraint was already known and written down from Stage 10: an njit entry point cannot
  construct a `GameState`. Asking "where should the buffers live" was the wrong question; "the search needs
  `GameState` **objects**" was the right one, and it rules out every allocation-location answer.

- **Stage 11 (all options blocked)** — the `GameState` construction limit, final form. An njit function
  reached from Python that constructs a `GameState` raises
  `TypeError: invalid number of args: expected 8, got 7`. Ruled out by direct test: a state argument (fails
  with zero, one, or two), helper indirection, which frame constructs, `cache=True` vs `False`, and
  parameter annotations.

  **`rules.deal_state` constructs a `GameState`, is annotated, is `cache=True`, is reached from Python via
  `setup_game`, and works.** That contradiction is the entire remaining problem: one structural difference
  separates it from functions that fail, and it has not been identified. Options (a), (a'), (b) and (c) are
  all blocked on finding it, not merely unattempted.

  Leading candidate: `deal_state` calls other `@njit` helpers **in the same module**, and every failing
  helper imports `HAND_CAPACITY` and `EMPTY` and constructs alone. Moving one failing constructor into
  `durak/rules.py` beside `deal_state`, unchanged otherwise, is a ten-minute test that would reopen two
  options at once.

- **Stage 11 (process, and the expensive lesson)** — four options were designed, one was chosen, two more
  were ruled out by tests, and the option that was chosen *first* failed for a reason that was already
  written down in the ledger from the stage before. The question "where should the buffers live" was never
  the right question; "the search needs `GameState` **objects** that cannot be created" is, and it was
  answerable from Stage 10's own notes without a single new experiment. When a constraint is already
  recorded, read it before designing around it.
13. rules._do_take handed the turn to the taker; the take rule was wrong for two stages before anyone checked it against the actual game. It surfaced as "27-35% of search games never finish" and was misdiagnosed three times -- first as a particle-filter defect, then as the deck-exhausted evaluation term, then as "a transfer reads as progress". The correct fix was to compare the rule against the rulebook, which nobody did before building an analysis on top of it. A rules engine deserves a test that states each rule in the words of the game, not only in arithmetic.
14. tests/test_rules.py::test_take_does_not_swap_roles asserted `state.attacker == 1` -- the opposite of what its own name said. The contradiction was invisible because the name and the assertion were both read past. Test names that state the rule are worth writing, and asserting the opposite of the name is worse than having no test.
15. Numba's on-disk cache went stale after the rules fix and silently masked four real test failures plus a compiled-vs-interpreted divergence. Any change to a `@njit(cache=True)` module needs its __pycache__ cleared before the suite result is trusted.
16. tests/test_invariants.py calibrated its transition floor (>15M) while AUTO_DEFEND still stalled, so the floor measured the bug rather than the invariant; after the fix the same 5,000 games yield 300,000 transitions. Volume assertions that are met by pathological behaviour have to be re-derived when the pathology is removed.
17. `Settings.particle_count` looked like a live knob and reached nothing; `particle_sweep` reported it as a three-row robustness table and every row was the same number. A setting that is recorded for reproducibility must not also imply it configures the engine. `__post_init__` now refuses any value that disagrees with `search_policy.PARTICLE_COUNT`, so the disagreement is loud.
18. CONSERVATIVE's 20.7% take rate and 0.11 win rate against GREEDY say the hand rules are too passive to be a useful strength reference. Its own tests pass, because they pin the thresholds rather than the behaviour those thresholds produce: every rule fires, and together they lose. A strategy test suite should also pin that the policy is competitive against the reference it exists to be compared with, or a conservative strategy that is simply weak will sit in the dataset indistinguishable from a strong one.
19. njit rejects a Python class holding constants ("Untyped global name") and rejects imports inside a compiled function ("Unsupported opcode IMPORT_NAME"). Thresholds in a compiled policy must be module-level ints, and helpers must be the `@njit` variants (`_prior_particles`, `_unseen_into`), not the public Python wrappers that take an RngStream.
20. numba 0.68 (Python 3.14) cannot lower a write into a structured array inside a `prange` body: `@njit(parallel=True)` with `out[i]["a"] = i` raises `NotImplementedError` from `parfors/parfor.py`. Plain 1-D array writes inside prange work. The engine's telemetry is structured arrays written from inside the game loop, so Task 14's `prange` block runner is not reachable without redesigning the telemetry layout to per-column 1-D arrays assembled serially afterwards. A stale Numba `cache=True` artefact made this look like a silent 1.00x speedup rather than a compile failure -- check `nopython_signatures` before believing a benchmark.
21. Level-1 rows carried neither `pairing_index` nor `first_attacker_regime`, so a dataset holding several regimes could not be read by regime at all. Both regimes sat in one directory with nothing to tell them apart. Schema version 2 adds them; the regime is stamped by the runner because `njit` cannot write a string into a row.
22. Telemetry levels used to NEST: levels 2 and 3 were written under `games/`, so `read_parquet(out_dir / "games")` returned 1,061 rows where 48 games existed. Levels must be siblings, and `analysis/schema.sql` globs each level's own directory.
23. `write_manifest(out_dir)` overwrote the run's `manifest.json` with the dataset inventory, which has no `run_id` -- the reproducibility contract destroyed by the call meant to describe the data. The two documents need separate paths.
24. A process pool over blocks must be measured from a real file, not a heredoc: `spawn`/`forkserver` re-import `__main__` from its path, so a piped script makes the pool fail with BrokenProcessPool and looks like a pool bug.
25. FIXED. `write_parquet(ensure_empty=True)` writes a zero-row file carrying the level's schema, so a run with `record_rounds=False` no longer breaks every view in `analysis/schema.sql`. Pinned by `test_a_dataset_without_rounds_or_moves_still_loads_in_every_view`.
