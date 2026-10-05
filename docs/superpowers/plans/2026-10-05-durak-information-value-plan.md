# Durak Information-Value Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provably correct Durak simulation engine, five strategies, a deterministic parallel tournament runner, and a static HTML reporting pipeline that answers one question with statistics: how much win rate is perfect knowledge of the opponent's hand worth?

**Architecture:** One implementation of the rules and strategies, written in the Numba-compatible subset of Python, executed in two modes (interpreted via `NUMBA_DISABLE_JIT=1`, and compiled). Correctness is proven by running the same code in both modes on identical seeds and asserting bit-identical move trajectories. Three independent RNG streams per game (`deal`, `player0`, `player1`) keep strategies from perturbing each other's randomness. Telemetry lands in Parquet; DuckDB aggregates it into JSON; JSON renders three self-contained HTML dashboards.

**Tech Stack:** Python 3.11+, Numba, NumPy, PyArrow (Parquet), DuckDB, Pandas (report building only), pytest, Matplotlib (PNG figures only), Chart.js 4.5.1 pinned via CDN in generated HTML.

**Spec:** `docs/superpowers/specs/2026-10-05-durak-information-value-design.md` — the spec travels with this plan; executors read both. Where this plan pins a value the spec left open, this plan wins.

---

## Global Constraints

Values below are copied from the spec or pinned here where the spec deliberately left a choice. Every task's requirements implicitly include this section.

**Cards and deck**
- Deck sizes `{36, 52}`. 36 → ranks 6..14; 52 → ranks 2..14. Four suits. No jokers.
- `card_id = rank_offset * 4 + suit`, `suit ∈ 0..3`, `rank_offset` indexes the deck's rank range (0-based).
- Hand storage: fixed-capacity `int16` array, capacity **64**; empty slot is `-1`; size in `int8`. Hand order is not state identity.
- `beats(a, b, trump)`: same suit and higher rank, **or** `suit(a) == trump` and `suit(b) != trump`. Within trump, higher rank wins.
- Card conservation invariant: `deck_remaining + |hand_0| + |hand_1| + |discard| + |table| == deck_size`, checked after **every** transition.

**Rules**
- Trump is the **bottom** card of the draw pile, turned face up, drawn last.
- Redeal if either player holds all six trumps. Redeal cap **100**; on exceeding, set `flag_redeals_exceeded`, choose first attacker uniformly at random, continue.
- First attacker: both hold trumps → lowest-ranked trump; exactly one holds trumps → that player; **neither → uniform random**. Field `both_have_trumps` is recorded on every game (≈2.8% of deals have no trumps in either hand).
- Throw-in is legal iff `n_attacks_after_adding <= min(defender_hand_size_now, 6)` **and** the card's rank matches the rank of some card on the table. `n_attacks` counts attack cards only, never defense cards. The cap is **6**.
- Config flag `throwin_defense_ranks`, default **`True`** (defense-card ranks enable throw-ins, per spec R7). Recorded in the run manifest.
- Win check runs the moment a card leaves a hand, **before** any refill, and **after every beat as well as after every attack**. A defender who beats with their last card ends the game immediately; no throw-in follows.
- Refill: while hand `< 6`, draw one card; **never** discard down. Attacker draws first, then defender. Partial refill when the draw pile empties mid-refill.
- Round cap **10,000** per game; exceeding raises and sets `flag_round_cap_hit`.

**State representation on the hot path**
- `GameState` is a **`NamedTuple` of arrays and scalars**, not a class and not a dataclass. This is the concrete resolution of the spec's "no classes, state is flat arrays" constraint under Numba. Array fields mutate in place (arrays are references); field *rebinding* — needed by the roles swap and by every hypothetical branch in the search — happens by returning a new `GameState`. The per-action copy is a memcpy of a few hundred bytes across fixed-size arrays and is negligible.
- No dataclass, frozen or otherwise, crosses into `nopython` code. `Settings` is unpacked into plain scalars at the boundary by the caller; `legal_actions`, `apply_action`, `evaluate`, and `play_game` take scalars and booleans, never a config object.
- `check_invariants(state) -> int` returns a **bitmask** of violated invariants, with a module-level `INVARIANT_NAMES` mapping bit to name. It does not return a Python list, because a list of strings cannot cross a `nopython` boundary.

**Determinism**
- Three independent RNG streams per game: `deal`, `player0`, `player1`.
- Per-game seed derived by integer mixing from `(run_seed, pairing_index, game_index)` only — never from scheduling, wall clock, or process identity.
- Tournament seed bounded by `2**31 - 1`.
- No `hash()`. No transcendental functions (`sqrt`, `exp`, `log`, `pow`) on the hot path; arithmetic limited to IEEE-754 `+ - * /` on `float64` plus integer ops.
- Output must be byte-identical for any worker count.

**Versions and telemetry**
- `ENGINE_VERSION`, `RULES_VERSION`, `SCHEMA_VERSION` all start at `1` and are stamped on every row and displayed in every report.
- Level-1 and level-2 telemetry for **every** game. Level-3 sampling deterministic: `game_index % sample_modulus == 0`, default `sample_modulus = 20` (5%). RNG-based sampling is forbidden.
- No wall-clock latency in telemetry; compute cost is measured only in the benchmark.

**Strategies**
- `StrategyId`: `RANDOM`, `GREEDY`, `CONSERVATIVE`, `SEARCH_UNIFORM`, `SEARCH_INFORMED`, `CHEATER_DOC`.
- `InfoMode`: `NONE`, `HEURISTIC`, `UNIFORM`, `INFORMED`.
- Default particle count `K = 100`; swept in Task 12.
- Soft-penalty parameter `lambda = 0` by default; `lambda > 0` only behind an explicit flag recorded in the manifest.
- Evaluation weights fixed in code, never tuned against opponents; swept in Task 12 and reported.
- `SEARCH_UNIFORM` and `SEARCH_INFORMED` share one function, identical depth, K, weights, tie-break, and action ordering; only the opponent-hand source differs.
- Opponent reply is **averaged**, not maximised.

**Statistics** (pinned here; the spec left the values open)
- Wilson interval, `z = 1.96`, reported as `[lo, hi]`.
- Paired bootstrap: **10,000** resamples, percentile interval, paired on `game_seed`.
- McNemar **exact** test on discordant pairs (two-sided).
- Holm-Bonferroni over the family: 3 opponents × 2 directions × 2 deck sizes = **12** primary comparisons. Report raw and adjusted p-values.
- Pre-registered primary stratum: `deck_size = 36`, both directions aggregated.

**Phase cut** (pinned here; the spec required it be pre-registered, not chosen after seeing data)
- Based on deck remaining. `late` if `deck_remaining <= 0.15 * initial_deck`; `early` if `deck_remaining > 0.50 * initial_deck`; otherwise `mid`. Cut recorded in the run manifest.

**Process**
- Artifacts — code comments, docstrings, commit messages, plans, specs, README, stage reports — are **English**. Chat is Russian.
- One stage, one commit. Commit format `type: brief description`, `type ∈ {feat, fix, refactor, chore, docs, style, perf, test, ci, build}`.
- Each task ends by writing `.superpowers/reports/stage-NN-<name>.md` (Result / Verification / Problems / Notes and improvements) and appending any non-critical finding to `IMPROVEMENTS.md` at the project root. `.superpowers/reports/` is gitignored; `IMPROVEMENTS.md` is tracked.
- Stage numbering is zero-padded and continues across runs: read the highest existing `stage-NN` before writing the next.
- Benchmarks carry the `perf` marker and are excluded from the default run.
- No dead code, no commented-out blocks, no placeholder TODOs left in shipped code.

---

## Review Focus

Five input classes the spec implies that are most likely to bite. Each has a test pinned to the task that owns the code.

1. **A deal where neither player holds a trump** (≈2.8% of 36-card deals, and the trump card is still in the draw pile so this is normal, not an error). Expected: first attacker chosen uniformly at random, `both_have_trumps = False` recorded, no exception, game completes. A reasonable person would expect the engine to treat this as an ordinary deal — Task 4.
2. **A throw-in attempted while the defender holds more than 6 cards** (only possible after a take). Expected: the cap of 6 binds, not the hand size; at most 6 attack cards on the table. Task 4.
3. **The draw pile emptying mid-refill.** Expected: partial refill succeeds silently, both players keep what they drew, card conservation still holds, no exception. Task 4.
4. **Re-running the same dataset with a different worker count or a different block size.** Expected: byte-identical Parquet, because seeds derive from game index and never from scheduling. Task 15.
5. **Regenerating a report from an unchanged dataset.** Expected: byte-identical HTML — no `new Date()`, no locale-dependent number formatting, no unordered iteration leaking into output order. Task 19.

---

## File Map

Created by this plan. Each file has one responsibility.

```
pyproject.toml                     deps, [tool.pytest.ini_options], perf marker registration
.gitignore                         venv, __pycache__, .superpowers/, *.db, data/
README.md                          written in Task 23
IMPROVEMENTS.md                     tracked backlog of non-critical findings

durak/__init__.py
durak/config.py                     DURAK_-prefixed env settings; single source of tunables
durak/versions.py                   ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION
durak/cards.py                      card encode/decode, deck construction, beats(), rank/suit accessors
durak/rng.py                        splitmix64 seeding, PCG stream, three-stream-per-game bundle
durak/actions.py                    action encoding/decoding to int32
durak/rules.py                      normative rules: setup, legal moves, apply action, win check, refill
durak/opponent.py                   particle filter over hands consistent with observations
durak/evaluation.py                 fixed evaluation weights + the sweep grid
durak/engine.py                     play_game(), play_block() (prange over games)
durak/audit.py                      per-game invariant verification and anomaly flags
durak/telemetry.py                  columnar outcome assembly + Parquet writing
durak/tournament.py                 pairing grid, both directions, seed derivation, rotation regime
durak/runner.py                     orchestration, block writing, resume

durak/strategies/__init__.py
durak/strategies/base.py            StrategyId, InfoMode enums; policy signature
durak/strategies/registry.py        id -> policy table
durak/strategies/policies.py        AUTO_ATTACK/AUTO_DEFEND (stages 2), RANDOM, GREEDY
durak/strategies/search_policy.py   SEARCH_UNIFORM, SEARCH_INFORMED
durak/strategies/conservative.py    CONSERVATIVE
durak/strategies/cheater_doc.py     CHEATER_DOC

analysis/__init__.py
analysis/collect.py                 regenerate the dataset
analysis/stats.py                   Wilson, paired bootstrap, McNemar exact, Holm-Bonferroni
analysis/queries.py                 DuckDB queries -> analysis/data/*.json
analysis/sweeps.py                  particle-count, weight, and rule-variant sensitivity sweeps

reports/__init__.py
reports/dashboard.py                CSS tokens, KPI card, chart block, sortable table, JSON embed
reports/build.py                    JSON -> three dashboards + PNG figures

benchmarks/bench_engine.py          throughput + per-strategy cost, perf marker

tests/conftest.py                   fixtures: fixed seeds, small decks, run manifests
tests/test_config.py
tests/test_cards.py
tests/test_rng.py
tests/test_rules.py
tests/test_particles.py
tests/test_search.py
tests/test_sweeps.py
tests/test_conservative.py
tests/test_invariants.py
tests/test_equivalence.py
tests/test_determinism.py
tests/test_legality.py
tests/test_audit.py
tests/test_telemetry.py
tests/test_runner.py
tests/test_statistics.py
tests/test_queries.py
tests/test_reports.py
tests/test_readme.py

data/                               generated Parquet (gitignored)
analysis/data/*.json                generated aggregates (committed — they are report inputs)
reports/out/*.html                  generated dashboards (committed)
reports/out/figures/*.png           generated figures (committed)
```

Three additions to spec section 4.3, with reasons: `durak/actions.py` keeps the int32 action encoding in one place instead of split between `rules.py` and the strategies; `durak/evaluation.py` isolates the fixed weights and their sweep grid so the "weights are never tuned against opponents" rule is enforced by file boundary; `durak/tournament.py` splits pairing-grid construction from block orchestration in `runner.py`, because those change for different reasons and are reviewed separately.

---

## Stage 1 — Repository Skeleton

### Task 1: Repository skeleton and tooling

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `durak/__init__.py`, `durak/config.py`, `durak/versions.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `durak.versions.ENGINE_VERSION`, `RULES_VERSION`, `SCHEMA_VERSION` — all `int`, all `1`.
  - `durak.config.Settings` — frozen dataclass with `deck_size`, `games_per_pairing`, `seed`, `workers`, `particle_count`, `sample_modulus`, `throwin_defense_ranks`, `soft_penalty_lambda`, `eval_weights`, read from `DURAK_`-prefixed environment variables with the defaults pinned in Global Constraints.
  - `durak.config.get_settings()` -> `Settings`, cached.

- [ ] **Step 1: Write the failing test** — in `tests/test_config.py`, a test asserting `get_settings()` returns the pinned defaults for every field, a test asserting an environment override for `DURAK_SEED` takes effect and that `DURAK_SEED` above `2**31 - 1` raises `ValueError`, a test asserting the three version constants are `1`, and a test asserting `Settings` is frozen (assignment raises).
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_config.py -v`; expect collection failure with `ModuleNotFoundError: No module named 'durak'`.
- [ ] **Step 3: Create the package and configuration** — write `pyproject.toml` declaring the package under src layout with the dependencies listed in Tech Stack, and `[tool.pytest.ini_options]` registering the `perf` marker and excluding it via `addopts`. Write `durak/config.py` with a frozen `Settings` dataclass and `get_settings()`; validate the seed bound. Write `durak/versions.py` with the three constants. Write `.gitignore` covering `venv/`, `__pycache__/`, `.superpowers/`, `data/`, `*.db`, `*.parquet`, and explicitly **not** `analysis/data/`, `reports/out/`, or `IMPROVEMENTS.md`.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_config.py -v`; expect all tests PASS.
- [ ] **Step 5: Commit** — `git add pyproject.toml .gitignore durak tests` then commit `chore: scaffold package, config, and versioning`.

---

## Stage 2 — Rules Core

### Task 2: Card encoding and deck construction

**Files:**
- Create: `durak/cards.py`
- Test: `tests/test_cards.py`

**Interfaces:**
- Consumes: `durak.config.Settings` (for the default `deck_size`).
- Produces:
  - `build_deck(deck_size: int) -> np.ndarray` of `int16` card ids in ascending order, length 36 or 52.
  - `rank_of(card: int) -> int`, `suit_of(card: int) -> int`, `rank_min(deck_size: int) -> int`.
  - `beats(a: int, b: int, trump: int) -> bool`.
  - `card_label(card: int, deck_size: int) -> str` for debug output.

- [ ] **Step 1: Write the failing test** — in `tests/test_cards.py`: a test asserting `build_deck(36)` and `build_deck(52)` return strictly increasing, duplicate-free arrays of the right length; a test enumerating **every** `(a, b, trump)` triple in the 36-card deck and asserting `beats` is asymmetric where it should be, that no card beats itself, that a trump of any rank beats every non-trump, and that within a suit a card beats exactly the same-suit cards of lower rank; a test asserting `rank_min(36) == 6` and `rank_min(52) == 2`; a test asserting `card_label` round-trips through a documented format.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_cards.py -v`; expect `ImportError` on `durak.cards`.
- [ ] **Step 3: Implement `durak/cards.py`** — implement the encoding and `beats` exactly as pinned in Global Constraints. Every function must be pure, allocation-light, and safe to call from `njit` (`nopython`) mode: integer arithmetic and comparisons only, no Python objects.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_cards.py -v`; expect PASS. Then run `NUMBA_DISABLE_JIT=1 pytest tests/test_cards.py -v`; expect the same PASS (confirms the module is importable outside compiled mode too).
- [ ] **Step 5: Commit** — commit `feat: add card encoding, deck construction, and beat relation`.

### Task 3: PRNG and seed derivation

**Files:**
- Create: `durak/rng.py`
- Test: `tests/test_rng.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `mix_seed(*parts: int) -> int` — deterministic integer mixer (splitmix64-style), never `hash()`.
  - `RngStream` — a mutable `uint64` state holder with `next_u64()`, `next_below(bound: int) -> int` (unbiased, via rejection sampling), `next_float() -> float64`.
  - `GameStreams` — holds three `RngStream`s (`deal`, `player0`, `player1`), constructed from `(run_seed, pairing_index, game_index)`.
  - `shuffle_inplace(cards: np.ndarray, rng: RngStream) -> None` — Fisher-Yates using only `next_below`.

- [ ] **Step 1: Write the failing test** — in `tests/test_rng.py`: a test asserting `mix_seed` is deterministic and that different argument tuples give different seeds; a test asserting two `GameStreams` built from the same triple produce identical streams and that any difference in any component changes the streams; a test asserting `next_below(n)` returns values in `[0, n)` over a large sample and that its empirical distribution is within tolerance of uniform; a test asserting the three streams of one game are mutually uncorrelated (a naive test: the first 10,000 draws of `deal` and `player0` share no common prefix pattern); a test asserting `shuffle_inplace` produces a permutation of the input and is deterministic for a fixed stream.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_rng.py -v`; expect `ImportError` on `durak.rng`.
- [ ] **Step 3: Implement `durak/rng.py`** — implement splitmix64 mixing and a PCG/xoshiro stream. `next_below` must be unbiased via rejection, never modulo. All state is `uint64`; all code `nopython`-safe. Document in the module docstring **why** the project carries its own PRNG instead of `random.Random` (unsupported in `nopython` mode; the stream must match across both execution modes).
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_rng.py -v`; expect PASS in both execution modes.
- [ ] **Step 5: Commit** — commit `feat: add seeded PRNG with three independent streams per game`.

### Task 4: Normative rules engine

**Files:**
- Create: `durak/actions.py`, `durak/rules.py`
- Modify: `durak/config.py` (add the phase cut constants and `throwin_defense_ranks` handling)
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `durak.cards`, `durak.rng`, `durak.config.Settings`.
- Produces:
  - `durak.actions`: `ATTACK = 0`, `THROW = 1`, `STOP = 2`, `BEAT = 3`, `TAKE = 4`; `pack_action(kind, card) -> int32`, `unpack_action(a) -> (kind, card)`; `IS_ACTION_MASK` constant.
  - `durak.rules.GameState` — a `NamedTuple` of NumPy arrays and scalars: `deck`, `deck_len`, `trump`, `hands` (2×64 `int16`), `hand_sizes` (2×`int8`), `discard`, `discard_len`, `table_attacks`, `table_defenses`, `table_ranks_mask`, `attacker`, `round_idx`, `winner`, `flag_anomaly`.
  - `durak.rules.INVARIANT_NAMES` — tuple mapping invariant bit index to name, so a bitmask from `check_invariants` decodes to a readable list on the Python side.
  - `setup_game(deck_size, streams, settings) -> GameState` — shuffle, deal, reveal bottom card as trump, apply the redeal rule up to the cap, choose the first attacker per spec 3.2.
  - `legal_actions(state, player, settings) -> np.ndarray` of packed actions.
  - `apply_action(state, action, settings) -> GameState` — mutates array fields in place and returns `state` (rebinding field identity where the rules require it, as in the roles swap); runs the win check after every card leaving a hand.
  - `refill(state, settings) -> GameState`.
  - `phase_of(state, initial_deck) -> int` — `0` early, `1` mid, `2` late per the pinned cut.
  - `check_invariants(state) -> int` — bitmask of violated invariants, decoded via `INVARIANT_NAMES`; `0` means clean.

- [ ] **Step 1: Write the failing test** — in `tests/test_rules.py`, one test per normative rule, each pinning exact values:
  - `test_setup_deals_six_each_and_reveals_bottom_as_trump` — 12 cards dealt, draw pile length 24, `trump == draw_pile_bottom`.
  - `test_redeal_when_player_holds_all_six_trumps` — construct a deck whose dealing forces the redeal condition; assert a second deal occurred (`state.redeals == 1`).
  - `test_redeal_cap_flags_and_falls_back_to_random` — force 100+ redeal conditions; assert `flag_redeals_exceeded` is set and setup still returns a usable state.
  - `test_first_attacker_lowest_trump_when_both_have_trumps`, `test_first_attacker_only_holder_when_one_has_trumps`, and **`test_first_attacker_random_when_neither_has_trumps`** — assert the third case picks uniformly across many seeds, sets `both_have_trumps = False`, and raises nothing.
  - `test_throwin_limit_caps_at_six_when_defender_holds_more_than_six` — give the defender 9 cards, assert `legal_actions` offers at most 6 attack cards total and refuses the 7th.
  - `test_throwin_limit_scales_with_small_defender_hand` — defender holds 3, assert at most 3 attacks.
  - `test_throwin_requires_matching_rank_of_any_table_card` — assert a non-matching rank is rejected; assert the behaviour flips when `throwin_defense_ranks=False` and only attack-card ranks qualify.
  - `test_win_declared_when_attacker_empties_hand_before_defender_responds` — attacker holds 1 card, attacks, defender could beat; assert `winner == attacker` and the round ended.
  - `test_win_declared_when_defender_beats_with_last_card` — assert `winner == defender` immediately after the beat and that no throw-in action is offered afterwards.
  - `test_refill_never_discards_below_six` — give a player 12 cards, refill, assert size is still 12.
  - `test_refill_partial_when_draw_pile_exhausts_midway` — assert no exception, sizes are `<= 6`, and card conservation holds.
  - `test_attacker_draws_first` — both below 6; assert the attacker's hand grows first.
  - `test_roles_swap_only_on_successful_defense` — both sub-cases.
  - `test_round_cap_raises` — assert exceeding 10,000 rounds raises rather than loops.
  - `test_card_conservation_holds_after_every_transition` — randomised playout, assert `check_invariants` is empty at every step.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_rules.py -v`; expect `ImportError` on `durak.rules`.
- [ ] **Step 3: Implement `durak/actions.py` and `durak/rules.py`** — implement spec section 3 exactly, rule by rule, in the order the Global Constraints list them. Pay attention to four things the tests above pin and that are easy to get wrong: the win check fires after **every** card leaving a hand including a beat; refill never reduces a hand; the throw-in cap is `min(defender_hand_now, 6)` counted over attack cards only; and `table_ranks_mask` must be maintained across throw-ins so rank matching stays correct as the table grows. Every non-obvious rule carries a comment naming **why** it is that way, not what it does. All functions `nopython`-safe.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_rules.py -v`; expect PASS in both execution modes.
- [ ] **Step 5: Commit** — commit `feat: implement normative rules engine`.

---

## Stage 3 — First Strategies and Runner

### Task 5: Strategy registry and first policies

**Files:**
- Create: `durak/strategies/__init__.py`, `durak/strategies/base.py`, `durak/strategies/registry.py`, `durak/strategies/policies.py`
- Test: `tests/test_legality.py`

**Interfaces:**
- Consumes: `durak.rules`, `durak.actions`.
- Produces:
  - `durak.strategies.base`: `StrategyId` and `InfoMode` integer enums; `Policy` signature `(state, player, strategy_id, info_mode, rng_state) -> int32` returning a packed action.
  - `durak.strategies.registry.get_policy(strategy_id: int) -> Policy` — table lookup, no polymorphism.
  - Policies: `AUTO_ATTACK` and `AUTO_DEFEND` (throwaway drivers for stage 2, kept because Task 6 needs them to generate games before real policies exist), `RANDOM`, `GREEDY`.

- [ ] **Step 1: Write the failing test** — in `tests/test_legality.py`: a test asserting `get_policy` raises for an unknown id and returns a callable for each known id; a test asserting `RANDOM` only ever returns actions present in `legal_actions`; the same for `GREEDY` and for both auto policies; a test asserting `AUTO_ATTACK` picks the lowest non-trump when one exists and the lowest trump otherwise; a test asserting `GREEDY` on defense returns the lowest beating card when one exists and `TAKE` otherwise, and never declines an available beat; a test asserting `GREEDY` never plays a trump as its opening attack while it holds a non-trump.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_legality.py -v`; expect `ImportError` on `durak.strategies`.
- [ ] **Step 3: Implement the registry and policies** — dispatch by integer id, per spec 4.2. `RANDOM` samples uniformly from `legal_actions` using the player's own stream. `GREEDY` implements spec 5.5. Keep every policy free of Python objects so it stays `nopython`-safe.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_legality.py -v`; expect PASS in both modes.
- [ ] **Step 5: Commit** — commit `feat: add strategy registry with random and greedy policies`.

### Task 6: Game loop and outcome

**Files:**
- Create: `durak/engine.py`
- Test: `tests/test_invariants.py`

**Interfaces:**
- Consumes: `durak.rules`, `durak.strategies`, `durak.audit`.
- Produces:
  - `GameOutcome` — a flat NumPy-record container of all level-1 fields from spec 6.1.
  - `play_game(deck_size, strategy_p0, strategy_p1, info_p0, info_p1, seed, settings, record_rounds, record_moves) -> GameOutcome`.
  - `play_block(deck_size, strategy_p0, strategy_p1, info_p0, info_p1, block_seed, pairing_index, game_start, game_count, settings, record_rounds, record_moves) -> (GameOutcome batch, RoundRecord batch, MoveRecord batch)` — `game_index = game_start + i`.

- [ ] **Step 1: Write the failing test** — in `tests/test_invariants.py`: a test running 10,000 auto-policy games across both deck sizes and asserting `check_invariants` never reports a violation; a test asserting every game terminates within the round cap; a test asserting exactly one of `winner == 0` / `winner == 1` at the end and the loser's hand is non-empty; a test asserting card conservation across the whole game; a test asserting the same seed produces byte-identical outcome records on repeat.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_invariants.py -v`; expect `ImportError` on `durak.engine`.
- [ ] **Step 3: Implement `durak/engine.py`** — drive the loop from spec 4.3 / 3.3. The loop must call the win check through `rules.apply_action` on every action, not in the engine, so there is exactly one place where the condition lives. When `record_rounds` / `record_moves` are false, do not allocate those arrays at all. Decorate `play_block`'s inner loop for `prange` only in Task 14 — for now keep it serial so the single-threaded behaviour is independently verified before parallelism exists.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_invariants.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add game loop with invariant guarantees`.

### Task 7: Telemetry levels 1-2 and Parquet writer

**Files:**
- Create: `durak/audit.py`, `durak/telemetry.py`
- Test: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `durak.engine`, `durak.versions`.
- Produces:
  - `durak.audit.audit_game(state, outcome) -> list[str]` — runs every invariant from spec 7.1 and returns anomaly names; never swallows.
  - `durak.telemetry.RoundRecord` / `MoveRecord` columnar containers.
  - `write_parquet(records, path, schema_version) -> None` — partitioned by `run_id` and `deck_size`.
  - `level1_schema()`, `level2_schema()`, `level3_schema()` returning explicit PyArrow schemas including the three version columns.

- [ ] **Step 1: Write the failing test** — in `tests/test_telemetry.py`: a test asserting every level-1 field in spec 6.1 exists in `level1_schema()` with the right dtype; the same for level 2 and level 3; a test writing 1,000 games to a temp directory and reading them back asserting row count, dtypes, and that `engine_version`/`rules_version`/`schema_version` are populated; a test asserting level-3 sampling keeps exactly `game_index % 20 == 0` and is identical for 1 worker and 8 workers; a test asserting `audit_game` flags a deliberately corrupted state (a card in two hands) and does **not** flag a clean one.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_telemetry.py -v`; expect `ImportError` on `durak.telemetry`.
- [ ] **Step 3: Implement `durak/audit.py` and `durak/telemetry.py`** — schemas are explicit and hand-written, never inferred from a Python object, so a dtype drift is a schema diff rather than a silent change. Level 3 stores `chosen_index_in_legal_order`, not latency, and the module docstring says why.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_telemetry.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add data-quality audit and Parquet telemetry`.

### Task 8: Tournament runner and benchmark

**Files:**
- Create: `durak/tournament.py`, `durak/runner.py`, `benchmarks/bench_engine.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `durak.engine`, `durak.telemetry`, `durak.config`.
- Produces:
  - `durak.tournament.pairing_grid(strategy_ids, both_directions=True) -> list[Pairing]` where `Pairing` carries `(strategy_p0, strategy_p1, deck_size, first_attacker_regime)`; regime ∈ `{RULE_DERIVED, ROTATION_BALANCED}`.
  - `durak.tournament.game_seeds(run_seed, pairing_index, game_count) -> np.ndarray` of `int64`.
  - `durak.tournament.first_attacker_override(regime, game_index) -> int` — `ROTATION_BALANCED` alternates with `game_index % 2`.
  - `durak.runner.RunManifest` — run id, seed, strategy list, deck sizes, games per pairing, regimes, all settings, three versions, block size, timestamp. Written as JSON next to the dataset.
  - `durak.runner.run_tournament(manifest, out_dir, workers) -> RunSummary`.
  - `benchmarks.bench_engine.run(policies, deck_size, games) -> dict` with games/sec and CPU-seconds per game per policy.

- [ ] **Step 1: Write the failing test** — in `tests/test_runner.py`: a test asserting `pairing_grid` yields each unordered pairing exactly once per direction with both directions enabled, and zero self-pairings except `RANDOM` vs `RANDOM` (needed for the symmetry check); a test asserting seeds are distinct across pairings and identical for the same `(run_seed, pairing_index)`; a test asserting `first_attacker_override` alternates under `ROTATION_BALANCED` and returns a sentinel meaning "use the rule" under `RULE_DERIVED`; a test running a 200-game tournament with `RANDOM` vs `RANDOM` and asserting the Wilson interval for 0.5 **contains** 0.5; a test running the same tournament twice with different worker counts and asserting byte-identical Parquet; a test asserting the manifest round-trips through JSON with every field preserved.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_runner.py -v`; expect `ImportError` on `durak.tournament`.
- [ ] **Step 3: Implement the runner and benchmark** — serial workers only here. The manifest is the reproducibility contract: it must record `throwin_defense_ranks`, `soft_penalty_lambda`, `particle_count`, `sample_modulus`, and the phase cut alongside the seed and versions. `bench_engine` carries the `perf` marker and reports per-policy cost, because a win rate without its CPU cost is not a result.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_runner.py -v`; expect PASS. Then run `pytest -m perf benchmarks/bench_engine.py` and record the measured games/sec for each policy — **this number decides the stage-10 dataset scale**, so write it into the stage report.
- [ ] **Step 5: Commit** — commit `feat: add tournament runner, manifests, and throughput benchmark`.

---

## Stage 4 — Information Model

### Task 9: Opponent-hand particle filter

**Files:**
- Create: `durak/opponent.py`
- Test: `tests/test_particles.py`

**Interfaces:**
- Consumes: `durak.rules.GameState`, `durak.cards`, `durak.rng.RngStream`, `durak.config.Settings`.
- Produces:
  - `initial_particles(state, opponent, k, rng) -> np.ndarray` shape `(k, hand_size)` of card ids; prior uniform over all size-consistent subsets of the unseen set.
  - `apply_observation(particles, state, observation, settings) -> np.ndarray` — applies hard constraints (played, discarded, in my hand, on table excluded; size equals the publicly observed hand size), then systematic resampling back to `k`.
  - `hand_probabilities(state, opponent, k, rng) -> np.ndarray` of float64 per card id — marginal probability the card is in the opponent's hand.
  - `informateness(state, opponent, k, rng) -> float64` — mean Shannon entropy in bits over the particle set, used as the x-axis of the phase curve in report 1.

- [ ] **Step 1: Write the failing test** — in `tests/test_particles.py`: a test asserting every initial particle has the opponent's exact hand size and contains only unseen cards; a test asserting the empirical marginal from a large particle set matches the analytic uniform prior `hand_size / |unseen|` within tolerance **before any observation**; a test asserting that after observing a card played, no particle contains it; a test asserting systematic resampling preserves `k` and stays non-degenerate; a test asserting the particle set's entropy **decreases** as observations accumulate (monotone non-increasing across a scripted playout, allowing equality when nothing is learned); a test asserting `soft_penalty_lambda == 0` produces a model that never references opponent behaviour — i.e. feeding it a non-beat produces the same particles as feeding it a beat, since with `lambda = 0` both carry zero information beyond legality.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_particles.py -v`; expect `ImportError` on `durak.opponent`.
- [ ] **Step 3: Implement `durak/opponent.py`** — implement spec 5.2. The module docstring must state why the prior is uniform and why it is *exact* rather than an approximation, and why `lambda = 0` is the default (taking is a strategic choice, not evidence of absence; a behavioural prior would inject unmeasured information into the uninformed arm). `nopython`-safe: fixed-size arrays, no sets, no dicts.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_particles.py -v`; expect PASS in both modes.
- [ ] **Step 5: Commit** — commit `feat: add opponent-hand particle filter with exact uniform prior`.

### Task 10: Evaluation and expectimax search

**Files:**
- Create: `durak/evaluation.py`, `durak/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `durak.rules`, `durak.opponent`, `durak.cards`, `durak.config`.
- Produces:
  - `EVAL_WEIGHTS` — a frozen tuple of the five weights from spec 5.4 with hand-justified values and a comment per weight explaining the sign and scale.
  - `EVAL_SWEEP_GRID` — the grid of weight perturbations swept in Task 12.
  - `evaluate(state, player, weights) -> float64` — card-count differential, trump differential, high-card differential (high = rank ≥ 10), attacker/defender term, deck-exhausted term.
  - `enumerate_responses(state, action, player) -> np.ndarray` of the opponent's legal packed actions.
  - `expectimax(state, player, particles, k, weights, settings) -> int32` — mean over particles and over enumerated responses of `evaluate` on the resulting position; deterministic argmax with a documented tie-break.

- [ ] **Step 1: Write the failing test** — in `tests/test_search.py`: a test asserting `evaluate` returns a positive value for a player with fewer cards, more trumps, and the attacker role, and that flipping `player` negates the value; a test asserting the deck-exhausted term only activates when the draw pile is empty; a test asserting `expectimax` chooses to stop throwing rather than throw a card the opponent is certain to beat (constructed position); a test asserting `expectimax` throws a card the opponent is certain it cannot beat; a test asserting `expectimax` is deterministic across repeated calls on the same state and particle set; a test asserting the returned action is always in `legal_actions`.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_search.py -v`; expect `ImportError` on `durak.search`.
- [ ] **Step 3: Implement `durak/evaluation.py` and `durak/search.py`** — implement spec 5.3 and 5.4. Two modelling choices must be visible in the code as comments, not just in the spec: the opponent's response is **averaged** rather than maximised, which makes the model optimistic about moves with bad worst cases; and the leaf is scored without a recursive continuation, which bounds the informed strategy's strength from below and therefore makes the measured price of information a lower bound. `nopython`-safe throughout.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_search.py -v`; expect PASS in both modes.
- [ ] **Step 5: Commit** — commit `feat: add evaluation function and expectimax search`.

### Task 11: Informed and uninformed search policies

**Files:**
- Create: `durak/strategies/search_policy.py`
- Modify: `durak/strategies/registry.py`
- Test: `tests/test_search.py` (extend), `tests/test_legality.py` (extend)

**Interfaces:**
- Consumes: `durak.search`, `durak.opponent`, `durak.strategies.base`.
- Produces:
  - `policy_search(state, player, strategy_id, info_mode, rng_state) -> int32` — one function serving both `SEARCH_UNIFORM` and `SEARCH_INFORMED`. When `info_mode == INFORMED`, it reads the opponent's true hand with `k = 1`; otherwise it builds `k` particles. Every other parameter — depth, weights, tie-break, action ordering — is read from the same place in both branches, and a comment states that this is the entire point of the experiment.

- [ ] **Step 1: Write the failing test** — extend `tests/test_search.py` with a test asserting the two `info_mode` branches differ **only** in the hand source: capture the depth, weights, tie-break, and legal-action ordering used in each branch and assert they are identical objects/values. Extend `tests/test_legality.py` with legality tests for both new strategy ids.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_search.py tests/test_legality.py -v`; expect failure — the policy does not exist yet.
- [ ] **Step 3: Implement `policy_search` and register both ids** — the branching structure must make the shared parameters structurally shared, not merely equal by convention, so a future edit cannot silently give one arm a different search. The particle filter must be **rebuilt from scratch each call** rather than cached across calls; a cache is a correctness hazard because the observation history is part of its validity, and if you do cache it, the cache key must include the full observation history — prefer no cache and note why.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_search.py tests/test_legality.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add informed and uninformed search policies sharing one search contour`.

### Task 12: `CHEATER_DOC` and sensitivity sweeps

**Files:**
- Create: `durak/strategies/cheater_doc.py`, `analysis/sweeps.py`
- Test: `tests/test_sweeps.py`

**Interfaces:**
- Consumes: `durak.search`, `durak.opponent`, `durak.evaluation`, `durak.runner`.
- Produces:
  - `policy_cheater_doc(state, player, strategy_id, info_mode, rng_state) -> int32` — the source document's Cheater: perfect information, **no sampling**, deterministic search, same evaluation weights. Deliberately a different code path from `policy_search`, which is what makes the document's specification measurable.
  - `analysis/sweeps.particle_sweep(pairings, k_values, ...) -> DataFrame` — win rate per `k`.
  - `analysis/sweeps.weight_sweep(pairings, grid, ...) -> DataFrame` — win rate per weight setting, for both `SEARCH_INFORMED` and `SEARCH_UNIFORM`, plus the delta per setting.
  - `analysis/sweeps.rule_variant_sweep(...) -> DataFrame` — `throwin_defense_ranks` both ways.

- [ ] **Step 1: Write the failing test** — in `tests/test_sweeps.py`: a test asserting `policy_cheater_doc` is deterministic (calling it twice on the same state returns the same action) whereas `policy_search` is allowed to be stochastic only through particle sampling, and that with `k = 1` the two agree on at least one constructed position where sampling noise is absent; a test asserting `weight_sweep` returns a row per grid point with both arms and a delta column; a test asserting `rule_variant_sweep` produces a non-empty result for both settings of the flag.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_sweeps.py -v`; expect `ImportError` on `durak.strategies.cheater_doc`.
- [ ] **Step 3: Implement `cheater_doc` and the sweep drivers** — the sweeps are small, scriptable, and must write their settings into the manifest so a sweep result is reproducible from its own metadata. Default `k_values` to at least `[25, 50, 100, 200, 400]`, which is what makes the primary metric's robustness to `k` checkable.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_sweeps.py -v`; expect PASS. Then run a small live sweep at ~2,000 games per cell and record in the stage report whether `delta` is stable across `k`. If it is not, that is a finding for the report, not a reason to pick a favourable `k` — write it in `Notes and improvements` and carry it into report 1.
- [ ] **Step 5: Commit** — commit `feat: add document-faithful cheater and sensitivity sweeps`.

---

## Stage 5 — Conservative Strategy

### Task 13: `CONSERVATIVE`

**Files:**
- Create: `durak/strategies/conservative.py`
- Modify: `durak/strategies/registry.py`
- Test: `tests/test_conservative.py`

**Interfaces:**
- Consumes: `durak.opponent` (for estimated opponent trumps and high cards), `durak.search.evaluate`, `durak.strategies.base`.
- Produces:
  - `policy_conservative(state, player, strategy_id, info_mode, rng_state) -> int32` — refuses to spend a trump on a cheap take when it holds at least two other trumps and at most two cards are at stake; declines a beat when the beating card is a high non-trump needed later and few cards are at stake; throws in only non-trumps whose rank matches, and forces a take when it believes the opponent cannot beat.
  - `CONSERVATIVE_THRESHOLDS` — a frozen, documented constant set (trump reserve floor, take-size threshold, high-card rank floor) with a comment per value.

- [ ] **Step 1: Write the failing test** — in `tests/test_conservative.py`: a test asserting it declines an available beat by trump when it holds exactly two other trumps and only one card is at stake, and beats when it holds one other trump; a test asserting it always beats when no trump is required and the beating card is low; a test asserting it never throws in a trump even when the rank matches; a test asserting it throws in a matching non-trump when it estimates the opponent can beat it; a test asserting legality on every returned action.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_conservative.py -v`; expect `ImportError`.
- [ ] **Step 3: Implement the policy** — implement spec 5.5. The thresholds live in one frozen constant with a per-value justification comment; nothing is a bare magic number in the body. The module docstring states that this strategy uses the particle model's *aggregate* counts rather than a search, which is what makes it a meaningfully different point on the strength axis.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_conservative.py -v`; expect PASS in both modes.
- [ ] **Step 5: Commit** — commit `feat: add conservative strategy on the opponent model`.

---

## Stage 6 — Scale

### Task 14: Parallel block runner

**Files:**
- Modify: `durak/engine.py` (add `njit`/`prange` to the block loop)
- Modify: `pyproject.toml` (add `numba` as an explicit dependency)
- Test: `tests/test_equivalence.py`

**Interfaces:**
- Consumes: `durak.engine.play_block`.
- Produces: the same `play_block` signature, now `njit`-compiled with `prange` over game indices. Semantics must be **identical** to the serial version.

- [ ] **Step 1: Write the failing test** — in `tests/test_equivalence.py`, the test the whole project rests on: for at least 300 seeds across both deck sizes and all strategy pairs, run the same games twice — once in a subprocess with `NUMBA_DISABLE_JIT=1`, once compiled — and assert the two outcome batches are **bit-identical**, comparing every level-1 and level-2 field byte for byte, plus the full move sequence for the sampled games. Add a test asserting the serial and `prange` block paths agree for the same block.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_equivalence.py -v`; expect failure because only one execution mode exists yet.
- [ ] **Step 3: Compile the block loop** — apply `@njit` and `prange`. Expect the first stages of this task to fail with typing errors; each one is a real constraint violation of spec 4.2 and must be fixed **in the algorithm**, not papered over with `numba.objmode` or `typed.Dict`. If a rewrite is forced (for example an accidental Python `set` or a dict lookup in the hot path), that rewrite is the correct outcome and the spec's constraints explain why. Do not introduce a transcendental function on the hot path to satisfy the compiler.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_equivalence.py -v`; expect PASS. Then re-run the full suite `pytest -v`; expect all green.
- [ ] **Step 5: Commit** — commit `perf: compile the block runner with prange`.

### Task 15: Worker-count independence and DuckDB views

**Files:**
- Modify: `durak/runner.py` (block splitting and worker pool), `durak/telemetry.py` (partition layout)
- Create: `analysis/schema.sql`
- Test: `tests/test_determinism.py`

**Interfaces:**
- Consumes: `durak.engine.play_block`, `durak.telemetry`.
- Produces:
  - `run_tournament(manifest, out_dir, workers, block_size) -> RunSummary` — now with parallel workers and resumable blocks.
  - `durak.tournament.game_block_ranges(game_count, block_size) -> list[(start, end)]`.
  - `analysis/schema.sql` — DuckDB views over the Parquet dataset: `games`, `rounds`, `moves`, plus a `dataset_versions` view listing which `(engine_version, rules_version, schema_version)` triples are present. Views must filter by version so pre-fix and post-fix data cannot merge silently.

- [ ] **Step 1: Write the failing test** — in `tests/test_determinism.py`: a test running the same 400-game tournament at `workers=1` and `workers=8`, and at two different block sizes, asserting all four Parquet outputs are byte-identical; a test asserting an interrupted run resumes from its last completed block and produces output identical to an uninterrupted run; a test asserting `analysis/schema.sql` executes cleanly in DuckDB and that `dataset_versions` shows exactly the versions in the manifest.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_determinism.py -v`; expect failure — `workers` and `block_size` are not yet honoured and `analysis/schema.sql` does not exist.
- [ ] **Step 3: Implement block splitting, worker pool, resume, and the DuckDB views** — a block is the unit of parallelism and of resume. Write each block's rows under a temporary name and rename on completion so a partial block is never mistaken for a finished one. Record `block_size` in the manifest.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_determinism.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add resumable parallel runner and DuckDB views`.

---

## Stage 7 — Statistics and Analysis

### Task 16: Statistics module

**Files:**
- Create: `analysis/stats.py`
- Test: `tests/test_statistics.py`

**Interfaces:**
- Consumes: NumPy only (no SciPy dependency; the exact test is small enough to implement directly).
- Produces:
  - `wilson_interval(successes: int, trials: int, z: float = 1.96) -> (lo, hi)`.
  - `paired_bootstrap(deltas: np.ndarray, resamples: int = 10000, seed: int) -> (lo, hi)`.
  - `mcnemar_exact(b: int, c: int) -> p_value` — two-sided, on discordant pairs only.
  - `holm_bonferroni(p_values: list[float]) -> list[float]` — adjusted, monotonic.
  - `proportion_diff_interval(a_lo, a_hi, b_lo, b_hi) -> (lo, hi)` for reporting a delta directly from two independent Wilson intervals, clearly labelled as such.

- [ ] **Step 1: Write the failing test** — in `tests/test_statistics.py`, each function checked against a **hand-computed fixture** with the expected value written out in the test: Wilson for `0/10`, `5/10`, `50/100` (compare to values computed by hand in the test comment); a symmetric case where the interval is symmetric about the point estimate; McNemar exact for `(b, c) = (0, 0)`, `(10, 0)`, `(5, 5)` — the last must give `p = 1.0`; Holm-Bonferroni on `[0.01, 0.04, 0.03]` asserting monotonic non-decreasing adjusted values and that the largest is capped at 1; paired bootstrap on a constant delta array returning a degenerate interval at that constant.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_statistics.py -v`; expect `ImportError` on `analysis.stats`.
- [ ] **Step 3: Implement `analysis/stats.py`** — McNemar exact by summing the binomial tail directly; no normal approximation, because with small discordant counts the approximation is exactly wrong where it matters. `holm_bonferroni` must enforce monotonicity or the family is no longer valid.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_statistics.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add Wilson, paired bootstrap, exact McNemar, and Holm-Bonferroni`.

### Task 17: Analysis queries to JSON

**Files:**
- Create: `analysis/queries.py`, `analysis/collect.py`
- Test: `tests/test_queries.py`

**Interfaces:**
- Consumes: `analysis/stats.py`, `analysis/schema.sql`, the Parquet dataset.
- Produces:
  - `analysis.collect.run(manifest_path, out_dir, workers, block_size) -> RunSummary` — regenerates the dataset end to end.
  - `analysis.queries.build_all(dataset_dir, out_dir) -> list[Path]` — writes `analysis/data/*.json`, one file per report plus `meta.json` carrying run id, seed, versions, counts, and the full settings block.
  - Named query functions: `price_of_information`, `tournament_overview`, `rules_and_dealing`, `data_quality`, `opponent_independent_strength`.
  - `opponent_independent_strength` returns each strategy's win rate **against `RANDOM` only** — the metric that cannot be distorted by the composition of the field, and the one to quote when comparing against a different study with a different strategy palette. It is emitted as its own JSON file so report 2 can show it beside the marginal rates.
  - Every JSON document carries its own `ci` block per rate, its `n`, and for every delta its paired-test result and Holm-adjusted p-value where it belongs to the primary family.

- [ ] **Step 1: Write the failing test** — in `tests/test_queries.py`: build a small synthetic dataset with a **known** win rate (e.g. 60/100), run the query functions, and assert the emitted JSON contains `0.6` and a Wilson interval matching `analysis.stats.wilson_interval(60, 100)` to full precision; assert the paired-delta path emits `mcnemar` and `adjusted_p` keys; assert `data_quality` reports zero anomalies for a clean dataset and a non-zero count for one with a deliberately flagged game; assert `meta.json` round-trips the manifest settings exactly; assert `opponent_independent_strength` returns only `RANDOM`-opponent rows and that its win rates for the non-random strategies sum consistently with the recorded game counts.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_queries.py -v`; expect `ImportError` on `analysis.queries`.
- [ ] **Step 3: Implement the queries and the collect driver** — the primary metric is computed exactly as spec 8.1 pre-registers it: `deck_size = 36`, both directions aggregated, matched seeds, McNemar on discordant pairs, paired bootstrap for the effect interval, Holm-Bonferroni across the 12-cell family. Rounding happens **only** at JSON serialisation, never in intermediate arithmetic, and all floating-point formatting is locale-independent. `collect.py` must be runnable as `python -m analysis.collect --manifest <path>` and must write the manifest it used next to the output.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_queries.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add dataset collection and analysis queries emitting report JSON`.

---

## Stage 8 — Full Dataset

### Task 18: Final dataset run

**Files:**
- Create: `data/` output (gitignored), `analysis/data/*.json` (committed), the run manifest
- Test: no new test — this task is verified by running the existing suite plus the audit

**Interfaces:**
- Consumes: `analysis.collect.run`.
- Produces: the research dataset at the scale chosen from the Task 8 benchmark, its manifest, and the full JSON aggregate set that reports consume.

- [ ] **Step 1: Fix the scale before running anything** — read the Task 8 benchmark from the stage report and compute the target game count that keeps the Wilson interval on the primary delta narrow enough to separate the strategies. **Write the chosen scale and its justification into the stage report before the run starts**, so the scale is a stated decision and not a post-hoc choice.
- [ ] **Step 2: Run a pilot at 10% of that scale** — run `python -m analysis.collect` on a reduced manifest; check `data_quality` reports zero anomalies and that `RANDOM` vs `RANDOM` sits at 0.5 within its interval. A non-clean pilot means a defect, not a data point: fix it in the engine and re-run, never proceed and never filter.
- [ ] **Step 3: Run the full dataset** — run `python -m analysis.collect` with the fixed manifest.
- [ ] **Step 4: Verify** — run `pytest -v` (expect all green); run the audit query and expect zero anomalies; re-run the collection with the same seed and assert byte-identical Parquet; record wall-clock time and total CPU cost, which the reports must quote.
- [ ] **Step 5: Commit** — commit the generated `analysis/data/*.json` and the manifest as `chore: record research dataset and aggregates`. The Parquet files stay untracked.

---

## Stage 9 — Reports

### Task 19: Report primitives and determinism

**Files:**
- Create: `reports/__init__.py`, `reports/dashboard.py`
- Test: `tests/test_reports.py`

**Interfaces:**
- Consumes: JSON aggregates.
- Produces:
  - `reports.dashboard.Document` — accumulates sections and renders a complete standalone HTML file.
  - `Document.header(title, subtitle, meta)`, `.kpi_row(items)`, `.chart_block(chart_id, title, chart_spec)`, `.chart_row(left, right)`, `.table(columns, rows, sort_keys)`, `.quality_block(flags)`, `.render(path) -> None`.
  - `reports.dashboard.CHARTJS_VERSION = "4.5.1"` — pinned and written into every output file.
  - `reports.dashboard.style_tokens()` — the CSS custom properties lifted from the reference dashboard (`--bg-primary`, `--bg-card`, `--bg-header`, `--text-primary`, `--text-secondary`, `--color-accent1..4`, `--positive`, `--negative`, `--gap`, `--radius`), including its layout grids for the KPI row and chart rows.

- [ ] **Step 1: Write the failing test** — in `tests/test_reports.py`: a test asserting a rendered file is valid standalone HTML containing the Chart.js CDN tag with the pinned version, the CSS token block, the KPI markup, a `<table>` with `data-sort-key` attributes on sortable headers, and the embedded JSON; a test asserting **byte-identical** output when the same JSON is rendered twice; a test asserting no `new Date()`, `toLocaleString`, or `Intl.` appears anywhere in the output; a test asserting no Python `set` or `dict` iteration order leaks into the output by rendering a document built from inputs in two different insertion orders and asserting identical bytes (sort every mapping key before emitting).
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_reports.py -v`; expect `ModuleNotFoundError` on `reports.dashboard`.
- [ ] **Step 3: Implement `reports/dashboard.py`** — copy the visual language of the supplied reference: KPI row of six, two-column chart rows with a full-width variant, sortable paginated data table, filter selects where a dimension warrants it, a data-quality section. The dataset build date comes from `meta.json`, never from the clock. JSON is embedded as a `const` per dataset, exactly as in the reference, so the file needs no server. Every chart block gets a `title` that states the finding, not the chart type.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_reports.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add deterministic standalone HTML report primitives`.

### Task 20: Report 1 — The Price of Information

**Files:**
- Modify: `reports/build.py`
- Test: `tests/test_reports.py` (extend)

**Interfaces:**
- Consumes: `analysis/data/price_of_information.json`, `meta.json`.
- Produces: `reports/out/01_price_of_information.html` — the primary report.

- [ ] **Step 1: Write the failing test** — extend `tests/test_reports.py`: assert the file exists after `build.py` runs on fixture JSON, contains a KPI block whose values equal the fixture's numbers, contains the phase curve over deck-remaining rather than round index alone, shows the paired interval and the Holm-adjusted p-value for every primary-family comparison, shows compute cost next to every win rate, and contains a data-quality block listing the run's flags. Also assert the report states that the measured value is a **lower bound** on the price of information, because the informed search is depth-limited — the claim must be visible on the page, not only in the README.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_reports.py -v`; expect failure — report 1 is not built.
- [ ] **Step 3: Implement report 1** — blocks, in order: the headline `delta` with its paired interval and adjusted p-value, immediately followed by a one-line caveat that it is a lower bound; `delta` against each opponent; the phase curve (`delta` against deck remaining, with the entropy from `opponent.informativeness` as the second series, since the claim is that value tracks how much is actually knowable rather than how late it is); `delta` conditioned on opponent trump count and hand size; the `k`-sweep panel showing whether `delta` is stable across particle counts; and the compute-cost panel. Every chart title states the conclusion it shows.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_reports.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add the price-of-information report`.

### Task 21: Reports 2-3 and figures

**Files:**
- Modify: `reports/build.py`
- Test: `tests/test_reports.py` (extend)

**Interfaces:**
- Consumes: `analysis/data/tournament_overview.json`, `rules_and_dealing.json`.
- Produces: `reports/out/02_tournament_overview.html`, `reports/out/03_rules_and_dealing.html`, and `reports/out/figures/*.png`.

- [ ] **Step 1: Write the failing test** — extend `tests/test_reports.py`: assert both files are generated from fixture JSON, that report 2 contains the win-rate matrix, rankings with intervals, game-length distribution, throw-in efficiency by strategy, and the 36/52 split; that report 3 contains the trump-count distribution, the `both_have_trumps` share, empirical throw-in limits versus the rule, wins by final hand size, and take behaviour; that the PNG set exists and is non-empty.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_reports.py -v`; expect failure.
- [ ] **Step 3: Implement reports 2 and 3 and the figure renderer** — Matplotlib only for the PNGs that go into the README, rendered from the same JSON the HTML uses so a figure and its chart can never disagree. In report 3, the **wins-by-final-hand-size chart leads the page**, because the degenerate endgame (spec 3.4) is the most important known property of the chosen rule set and must not be buried.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_reports.py -v`; expect PASS.
- [ ] **Step 5: Commit** — commit `feat: add tournament and rules reports with README figures`.

---

## Stage 10 — README

### Task 22: README and final verification

**Files:**
- Create: `README.md`
- Test: `tests/test_readme.py`

**Interfaces:**
- Consumes: the guides `The Ultimate README Guide.md` and the generated reports and figures.
- Produces: `README.md`.

- [ ] **Step 1: Write the failing test** — in `tests/test_readme.py`: assert every section the guide marks mandatory exists as a heading (`Introduction`, `Tech Stack`, `Project Structure`, `Quick Start`, `Installation & Usage`, `Data`, `Methodology`, `Results`, `Testing`, `Limitations`, `Recommendations`); assert every internal file path mentioned in backticks exists on disk; assert every figure referenced exists in `reports/out/figures/`; assert every number quoted in `Results` matches the corresponding value in `analysis/data/*.json` (this is the test that stops the README from drifting from the data); assert the `Table of Contents` lists every heading it contains.
- [ ] **Step 2: Run the test to verify it fails** — run `pytest tests/test_readme.py -v`; expect failure — no README.
- [ ] **Step 3: Write the README** — follow the guide's structure. Three sections carry the project's honesty burden and must not be softened:
  - **Methodology** — the pre-registered primary metric, the `SEARCH_UNIFORM` / `SEARCH_INFORMED` shared contour, why the opponent reply is averaged rather than maximised, and why `lambda = 0`.
  - **Limitations** — all seven from spec 13, led by the degenerate endgame rule and the fact that the measured price of information is a **lower bound** because the informed search is depth-limited.
  - **Data-quality caveats** — mirror the reference project's convention: every rate carries its interval, the sample size is stated, and anomaly flags are surfaced.
  Also document the exact reproduction commands, the three version numbers, and the statement that `ENGINE_VERSION`/`RULES_VERSION` gate comparability.
- [ ] **Step 4: Run the test to verify it passes** — run `pytest tests/test_readme.py -v`; expect PASS. Then run the whole suite `pytest -v` and a full `python -m reports.build` followed by a re-render to confirm byte-identical HTML.
- [ ] **Step 5: Commit** — commit `docs: add README following the project guide`.

---

## Execution Order and Gates

Tasks run in the order listed. The dependencies that are not obvious:

- Task 8 must finish before Task 18, because the benchmark decides the dataset scale.
- Task 9 and Task 10 must finish before Task 11, Task 13.
- Task 14 must finish before Task 18, because the full dataset uses the parallel runner.
- Task 16 must finish before Task 17, because the queries call the statistics.
- Task 19 must finish before Tasks 20 and 21.
- Task 14 is the highest-risk task: it is where the spec's execution-mode constraints either hold or the code must be rewritten. If a rewrite is forced there, re-run the full suite before continuing — a divergence that only Task 14's test sees is a divergence the reports would inherit silently.

Each task ends with its own commit and a stage report. A rejected task result starts a new iteration of the same task with the same number; a plan change mid-task halts the task and returns here for approval.