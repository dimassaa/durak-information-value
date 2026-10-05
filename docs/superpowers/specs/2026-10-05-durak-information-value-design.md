# Durak Simulation: Design Specification

**Date:** 2026-10-05
**Status:** Awaiting review
**Supersedes:** `Durak AI Simulator and Web Application.md` (Section 8 "Web Application" and Phase 4 are dropped; the remainder is normalised and corrected here)

---

## 1. Purpose

This project is an analytical study, not a game. It builds a provably correct
simulation of two-player podkidnoy Durak, implements a palette of bot strategies on
top of it, runs a deterministic round-robin tournament, and answers one primary
research question with statistical rigour.

**Primary research question:** How much win rate is *worth* perfect knowledge of the
opponent's hand, holding everything else constant?

Everything else in the project — strategy rankings, deck-size effects, positional
advantage, throw-in efficiency — is a derived metric that exists to contextualise or
decompose the primary answer.

The project's own thesis is the phrase from the source document: **quality of data**
and **algorithmic correctness** come first. Every design decision below is subordinate
to those two.

### 1.1 Success criteria

1. A reader can verify engine correctness from the test suite alone, without reading
   the strategy code.
2. Re-running the dataset generation with the same seed reproduces the same Parquet
   bytes.
3. Regenerating a report from the same dataset reproduces the same HTML bytes.
4. The primary metric is stated with an interval, a pre-registration, and an explicit
   list of every comparison made (including multiplicity correction).
5. Known weaknesses of the chosen rule set are measured and reported, not hidden.

---

## 2. Scope

### 2.1 In scope

- Normative rules engine (two implementations of execution mode, one implementation of code).
- Five strategies: `RANDOM`, `GREEDY`, `CONSERVATIVE`, `SEARCH_UNIFORM`, `SEARCH_INFORMED`.
  Plus `CHEATER_DOC`, a faithful implementation of the source document's Cheater, retained
  only for comparison against that document's specification.
- Deterministic parallel tournament runner.
- Three-level telemetry (game / round / move) written to Parquet.
- Data-quality audit module with per-game invariant checks.
- Statistics module: Wilson intervals, paired bootstrap, McNemar exact test, Holm-Bonferroni.
- DuckDB query layer producing JSON aggregates.
- Three self-contained HTML dashboards plus PNG figures for the README.
- README following the supplied guide.

### 2.2 Out of scope

- **Web application, human play, UI, animations, `/bot-action` endpoint.** Dropped by the
  project owner. Section 8 and Phase 4 of the source document are void.
- **Machine learning.** All logic is rules, heuristics, and probability theory.
- **PostgreSQL.** Replaced by Parquet + DuckDB (rationale in Section 9).
- **Human baseline.** All opponents are scripted. This is a stated limitation.

### 2.3 Research design

The tournament is a full round-robin over the five primary strategies, played in **both
directions** (each unordered pairing played as `A-as-P0` and `A-as-P1`) so that marginal
win rates are averages over a balanced position assignment rather than selection-biased
ones. Mirror-image pairings share seeds, which makes them a paired comparison.

A fifth strategy, `RANDOM`, is a deliberate degenerate baseline. The reference project
this is modelled on shipped a degenerate strategy that distorted every marginal win rate
and had to confess it in its own limitations section. We keep a degenerate strategy, but
(1) `RANDOM` is symmetric rather than catastrophically bad, so `RANDOM` vs `RANDOM` is a
clean 50/50, (2) all marginal rates are reported both with and without it, and (3) the
opponent-independent metric defined in Section 8.3 is reported alongside marginal win rate so
no conclusion depends on the baseline's strength.

---

## 3. Normative Rules

This section is the single source of truth. The engine may not deviate from it. Where the
source document was ambiguous or self-contradictory, the resolution is stated explicitly
and marked **[RESOLVED]**. Where a rule has a configuration switch, both readings are
described and the default is stated.

### 3.1 Deck and encoding

- `deck_size` ∈ {36, 52}. 36 uses ranks 6..14; 52 uses ranks 2..14. Four suits, no jokers.
- A card is a single integer: `card_id = rank_offset * 4 + suit`, where `rank_offset`
  indexes the deck's rank range and `suit` ∈ 0..3. This encoding is dense and ordering by
  `card_id` is ordering by (rank, suit) within a suit-group, which lets beat-checks be
  integer comparisons instead of tuple comparisons.
- `beats(a, b, trump)`: `a` beats `b` if `suit(a) == suit(b) and rank(a) > rank(b)`, or
  `suit(a) == trump and suit(b) != trump`. Within the trump suit the higher rank wins.
- Hands are stored as fixed-capacity `int16` arrays (capacity 64; a legal hand can exceed
  6 after a take but cannot exceed the deck size, and 52 < 64) plus an `int8` size.
  An empty slot is `-1`. Hand order is not part of the state identity.

### 3.2 Setup

1. Shuffle the deck with the game RNG.
2. Deal 12 cards one at a time from the top of the shuffled deck, alternating, starting with
   P0, giving each player 6 cards. Alternation is a convention, not a statistical property; the
   resulting assignment is uniform either way.
3. The remaining cards form the draw pile. **The bottom card of the draw pile is turned
   face up and is the trump card.** It is therefore drawn last. *(Matches the source
   document; note this is the opposite of "top card is trump".)*
4. **Redeal rule:** if either player holds all six trump-suit cards, discard the deal,
   reshuffle, and redeal. **[RESOLVED]** The source document permits unbounded redealing,
   which is an infinite loop with a nonzero probability. A cap of **100 redeals** applies;
   on exceeding it the run records a `redeals_exceeded` anomaly, chooses the first attacker
   uniformly at random, and continues. In practice the event is vanishingly rare
   (≈1 in 1.9M deals) and the cap exists to make the failure loud rather than silent.
5. **First attacker [RESOLVED]:**
   - Both players hold at least one trump → the player holding the **lowest-ranked trump**
     attacks first.
   - Exactly one player holds a trump → that player attacks first.
   - **Neither player holds a trump → chosen uniformly at random.**

   The source document dismissed the third case as "impossible". It is not impossible: the
   trump card stays in the draw pile, and the probability that all six trumps land in the
   24 undealt cards is `C(30,12)/C(36,12) ≈ 2.8%`. This is a material frequency, so every
   game records a `both_have_trumps` boolean. Any analysis that ignores this field will
   silently average over two different starting distributions.

### 3.3 Round structure

A round is: one attack, zero or more throw-ins, then resolution.

1. **Attack.** The attacker plays one card from their hand.
2. **Win check** (see 3.4) runs immediately after the card leaves the hand.
3. **Defend.** The defender either beats the card or takes. *Taking is always legal,* at any
   point during the round, and ends the round immediately.
4. **Throw-ins.** If the attack was beaten, the attacker may add cards. A throw-in is legal
   when **both**:
   - its rank equals the rank of some card already on the table, and
   - `n_attacks_after_adding <= min(defender_hand_size_now, 6)`.

   `n_attacks` counts attack cards only, never defense cards. The `6` cap is part of the
   chosen rule. The cap matters only when the defender holds more than 6 cards, which
   happens after a take.

   **The win check (3.4) runs after every beat as well as after every attack**, so a defender
   who beats using their last card ends the game at that moment and no throw-in follows.
5. **Stop.** The attacker may stop voluntarily at any time.
6. **Resolution.** If every attack card on the table was beaten, the whole table goes to
   the discard pile and **roles swap** (the defender becomes the next attacker). If the
   defender took, the table goes into the defender's hand and **roles do not swap**: the
   defender, who now holds the cards, attacks next.
7. **Refill [RESOLVED].** If the round ended without a winner, the attacker draws first, up
   to 6 cards, then the defender draws up to 6 cards. "Up to 6" means **a refill never
   reduces a hand**: a player holding 12 cards draws nothing. The source document's
   "until they have 6 cards" is only correct under the reading "while below 6"; the
   explicit reading removes the ambiguity. Partial refills occur when the draw pile runs
   out mid-refill; both players keep whatever they drew.

### 3.4 Winning

**[RESOLVED by the project owner]** The win condition is checked the moment a card leaves a
hand, **before** any refill. If a player's hand is empty, that player wins immediately and
the round ends; the opponent does not respond.

Consequences of this rule, which must be measured and reported rather than discovered by a
reader:

- An attacker who plays their last card wins even though the defender then takes it.
- A defender who beats using their last card wins immediately, before the attacker can
  throw in.
- Positions with exactly one card in hand are strongly winning for whoever holds the
  decision.

This makes the endgame degenerate and short. The project does **not** pretend otherwise:
`win_from_single_card_p0` / `win_from_single_card_p1` are per-game telemetry fields, the
Rules & Dealing report leads with the distribution of wins by final hand size, and the
alternative rule (a last card that is taken does not win) is listed in Limitations as the
single highest-leverage sensitivity to re-run. This is the most important known property
of the chosen rule set and it must appear in the README, not in a footnote.

### 3.5 Termination

Termination is guaranteed. While the draw pile is non-empty the attacker's hand is
refilled to 6, so a round is bounded by the remaining draw pile, which is finite. Once the
draw pile is empty, no refill occurs and the attacker's hand can only shrink, so the
attacker must eventually empty their hand. A hard cap of 10,000 rounds per game is
enforced as a defensive assertion; exceeding it raises rather than hangs.

### 3.6 Resolved ambiguities, consolidated

| # | Source document text | Resolution |
|---|---|---|
| R1 | "if defender beats all cards, roles swap; if defender takes, the defender becomes attacker" preceded by an inline "Actually…" | Roles swap on successful defense only. The inline hesitation is resolved. |
| R2 | "the number of cards on the table cannot exceed the defender's hand size (including the card just beaten?)" | Count attack cards only, against the defender's **current** hand size, capped at 6. |
| R3 | "draw until they have 6 cards" | Refill only while below 6; never discard down. |
| R4 | "if neither player has a trump … first attacker is chosen randomly", justified as impossible | Kept as a real case (≈2.8% of deals) and recorded as a field. |
| R5 | Trump card "drawn last" / "placed face-up under the deck" | Bottom of the draw pile, drawn last. Consistent; no conflict. |
| R6 | "both players draw … starting with the attacker (the player who was the attacker in the turn that just ended)" | On a successful defense the roles swap, so "the player who was the attacker" means the attacker of the round just completed, who draws first even though the opponent attacks next. Confirmed intentional. |
| R7 | Throw-in rank match: "same rank as any card already on the table (either the initial attack card or any thrown-in card)" | Any card on the table, **including defense cards**. The parenthetical is narrower than the sentence; the sentence is followed. The narrower reading is available behind a config flag and is reported as a sensitivity, because defense cards are commonly higher than the attack card and this materially changes throw-in rates. |
| R8 | "The first player to get rid of all cards when the deck is empty wins" | Win is checked before refill, so the "when the deck is empty" qualifier is not a separate condition. See 3.4. |

---

## 4. Architecture

### 4.1 The correctness argument

There is **one implementation of the rules and strategies**, written in the subset of
Python that Numba accepts in `nopython` mode. Because Numba-decorated functions are
callable from ordinary Python, the same source runs in two execution modes:

- **interpreted** (`NUMBA_DISABLE_JIT=1`), which is the readable reference behaviour, and
- **compiled** (default).

The project's correctness test runs N games in both modes on identical seeds and asserts
bit-identical move trajectories and identical final counters. Correctness is therefore
proven over a real code path rather than asserted in a comment. This replaces the usual
"two implementations + equivalence test" approach, which doubles the code and doubles the
surface for the two copies to drift apart.

**Cost of this choice:** the code must stay inside Numba's subset, which is less pleasant
to write than ordinary Python. That constraint is documented in the package README and
enforced by review, not by convention. It is the right trade: the alternative is
correctness that nobody can check.

### 4.2 Constraints imposed by the execution mode

These are hard constraints on the hot path. Each exists for a stated reason.

- **No classes, no dicts, no object containers in `play_game`.** State is flat arrays.
  *Reason:* `numba.prange` and `nopython` cannot take Python objects cheaply, and per-move
  object allocation dominates runtime at this scale.
- **Actions are integers.** `ATTACK(card_id)`, `THROW(card_id)`, `STOP`, `BEAT(card_id)`,
  `TAKE`, packed into a single `int32`. *Reason:* avoids tuples in the inner loop.
- **Strategies are dispatched by integer id, not by object.** `decide(state, strategy_id,
  info_mode, player, rng_state) -> int32`. *Reason:* polymorphic dispatch over instances is
  not supported inside `prange`.
- **Own PRNG implementation** (PCG / splitmix64 on `uint64`), with the state threaded
  explicitly. *Reason:* `random.Random` is unsupported in `nopython` mode, and the stream
  must be identical in both execution modes.
- **No `hash()`.** Seed derivation uses an explicit integer mixer. *Reason:*
  `PYTHONHASHSEED` randomisation would silently destroy reproducibility across processes.
- **No transcendental functions** (`sqrt`, `exp`, `log`, `pow`) on the hot path;
  arithmetic is limited to IEEE-754 `+ - * /` on `float64` and integer ops. *Reason:* libm
  results are not guaranteed identical between interpreted and compiled paths, which would
  break the equivalence test. (This does not force the evaluation function onto integer
  arithmetic, because arithmetic operations themselves are bit-identical; only libm calls
  are excluded.)
- **Three independent RNG streams per game** — `deal`, `player0`, `player1`. *Reason:* if
  both players shared one stream, changing how many draws strategy A consumes would shift
  strategy B's randomness and destroy comparability between runs that differ in exactly one
  strategy. This is the single most important determinism detail in the project.

### 4.3 Module layout

```
durak/
  cards.py        card encoding, deck construction, beats(), rank/suit accessors
  rng.py          splitmix64 / PCG; seed derivation from (run_seed, pairing, game_index)
  rules.py        normative rules: legal moves, apply action, win check, refill, invariant checks
  opponent.py     opponent-hand model: particle filter over consistent hands
  search.py       expectimax over particles, parameterised by information mode
  strategies/
    base.py       StrategyId enum, info-mode enum, action encoding helpers
    registry.py   id -> decision function table
    random_policy.py, greedy.py, conservative.py, search_policy.py
  engine.py       play_game(...) -> GameOutcome (njit); play_block(...) (prange over games)
  audit.py        per-game invariant verification
  telemetry.py    GameOutcome / RoundRecord / MoveRecord assembly and Parquet writing
  runner.py       pairing grid, seed derivation, worker pool, resume support
  versions.py     ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION
analysis/
  collect.py      regenerate the dataset
  queries.py      DuckDB queries -> analysis/data/*.json
  stats.py        Wilson, paired bootstrap, McNemar, Holm-Bonferroni
reports/
  dashboard.py    shared HTML/CSS/table/chart primitives
  build.py        JSON -> the three dashboards + PNG figures
tests/
```

### 4.4 Engine contract

`play_game(deck_size, strategy_p0, strategy_p1, info_p0, info_p1, seed) -> GameOutcome`

- Returns a flat outcome record: winner, round count, action count, both strategies' counters,
  the game-level flags from Section 6, and the final hands.
- Optional `record_moves: bool` and `record_rounds: bool` flags; when off, the
  corresponding arrays are not allocated. Levels 1 and 2 are always recorded.
- `play_block(...)` runs a contiguous range of game indices under `prange` and returns
  columnar arrays for the whole block, so the caller writes one Parquet file per block and
  never touches a Python object per game.

### 4.5 Parallelism and determinism

- Parallelism is over **games**, never inside a game. A game is inherently sequential.
- Game `i` in a block derives its seed purely from `(run_seed, pairing_index, i)` through
  an integer mixer. Scheduling never influences a result, so a 1-worker run and an
  8-worker run produce byte-identical Parquet files.
- Block boundaries are part of the run configuration and are recorded in the run manifest, so
  a partially failed run can be resumed without duplicating or losing games.

---

## 5. Strategies

### 5.1 Information mode is the experimental variable

| Strategy id | Opponent-hand source |
|---|---|
| `RANDOM` | none (uniform over legal actions) |
| `GREEDY` | none (heuristic, ignores opponent cards) |
| `CONSERVATIVE` | heuristic counting of unseen trumps and high cards |
| `SEARCH_UNIFORM` | particle filter over hands consistent with observations |
| `SEARCH_INFORMED` | the true hand, one particle |

`SEARCH_UNIFORM` and `SEARCH_INFORMED` are the same function with the same depth, the same
particle budget, the same evaluation weights, the same tie-breaking, and the same action
ordering. **The only difference is the opponent-hand source.** Their win-rate difference is
therefore attributable to information and to nothing else.

The source document specified Cheater as "Probabilistic, but with perfect knowledge", which
in practice means different search, no sampling, and deterministic evaluation. Comparing
those two directly confounds information with sampling noise and with search quality, and
the resulting number cannot be called "the price of information". `CHEATER_DOC` is
implemented as the document describes it so the discrepancy can be *measured* and reported,
but it is not the headline comparison.

### 5.2 Opponent-hand model (particle filter)

- **Prior.** Uniform over all subsets of the unseen set having the opponent's observed hand
  size. This is exactly correct, not an approximation: the deck is uniformly shuffled, so any
  fixed-size subset of the unseen cards is equally likely to be the opponent's hand, with the
  remainder forming the draw pile in some order.
- **Hard constraints** applied after every observation: cards the opponent has played, cards
  in the discard pile, cards in my own hand, and cards on the table are excluded; the
  particle's size must equal the opponent's publicly observable hand size.
- **Resampling.** Systematic resampling of K particles after each observation, so the particle
  set does not degenerate.
- **No soft penalties by default [RESOLVED].** The source document suggests down-weighting
  particles when the opponent declines to beat a card it could have beaten. Taking is a
  strategic choice, not evidence of absence; modelling it requires an assumption about the
  opponent's behaviour, which would inject unmeasured information into the "uninformed"
  arm and destroy the interpretation. The default is therefore `lambda = 0`: the uninformed
  arm knows nothing beyond legality. A penalty variant exists behind a flag, with `lambda`
  recorded in the run manifest, and is reported separately if used.

### 5.3 Search: expectimax over particles

For each legal action of the bot, and for each of the K sampled opponent hands, the
opponent's legal responses to that action are enumerated. Each resulting position is scored
by the evaluation function. The score of an action is the mean over particles and over
responses. The bot picks the argmax, with a deterministic tie-break.

Two properties are load-bearing:

- **Branching is `|my actions| × K × |their responses|`**, with no recursive tree. This is
  what makes millions of games feasible at all.
- **The opponent's response is averaged, not maximised.** We are computing an expectation
  over opponent behaviour, not assuming an optimal opponent. This is a modelling choice and
  is stated as such in the README, along with its consequence: the model is systematically
  optimistic about moves whose worst case is bad.

Search depth beyond "my action → their responses → leaf evaluation" is out of scope and is a
stated limitation.

### 5.4 Evaluation function

Fixed, hand-justified weights: card-count differential, trump differential, high-card
differential, an attacker/defender term, and a deck-exhausted term. **The weights are not
tuned against opponents.** Auto-tuning was considered and rejected: it confounds algorithm
quality with tuning budget, and it invites an asymmetric budget between the informed and
uninformed arms, which would directly corrupt the primary metric.

Instead, a weight sweep is run as its own experiment and reported as a sensitivity: how much
does the win rate move across the weight grid, and does the informed-minus-uninformed delta
move with it? If the delta is stable across the grid, the primary metric is robust to
evaluation design.

### 5.5 Strategy behaviour notes

- `GREEDY`: beats with the lowest card that can beat, never declines an available beat;
  attacks with the lowest non-trump; throws in any rank-matching card while the throw-in limit
  allows. Its role is to be a strong, simple, information-free reference.
- `CONSERVATIVE`: reserves trumps and high cards, and takes rather than spending a trump when
  the cards at stake are few. Depends on the opponent model, so it is implemented after the
  particle filter.
- `RANDOM`: uniform over legal actions, including `TAKE` and `STOP`. It has no closed-form
  win rate against any non-random policy, so no theoretical value is asserted for it; its one
  exact property is that `RANDOM` vs `RANDOM` is 0.5, which is a symmetry test of the whole
  engine (Section 7.3).

---

## 6. Telemetry

Three levels. Levels 1 and 2 are written for every game; level 3 is written for a sampled
subset. All three are Parquet, partitioned by `run_id` and `deck_size`.

### 6.1 Level 1 — per game

Identity and configuration: `run_id`, `game_index`, `game_seed`, `deck_size`, `strategy_p0`,
`strategy_p1`, `info_mode_p0`, `info_mode_p1`, `engine_version`, `rules_version`,
`schema_version`.

Deal and rules: `first_attacker`, `both_have_trumps`, `initial_trumps_p0`, `initial_trumps_p1`,
`redeals`, `first_attacker_random_fallback`.

Outcome: `winner`, `rounds`, `actions`, `winner_final_hand`, `loser_final_hand`,
`deck_exhausted_at_round`, `winner_rounds_played`.

Resource usage: `cards_collected_p0`, `cards_collected_p1`, `voluntary_takes_p0`,
`voluntary_takes_p1`, `forced_takes_p0`, `forced_takes_p1`, `trumps_played_p0`,
`trumps_played_p1`, `throw_ins_p0`, `throw_ins_p1`, `cards_beaten_p0`, `cards_beaten_p1`,
`max_attacks_in_a_round`, `attack_turns_p0`, `attack_turns_p1`.

Degeneracy diagnostics: `win_from_single_card_p0`, `win_from_single_card_p1`,
`cards_left_when_deck_ran_out`.

Anomaly flags: `flag_redeals_exceeded`, `flag_round_cap_hit`, `flag_illegal_move`,
`flag_card_conservation_violation`, `flag_both_players_empty`, `flag_anomaly_any`.

### 6.2 Level 2 — per round

`run_id`, `game_index`, `round_idx`, `attacker`, `n_attacks`, `n_throw_ins`, `n_defenses`,
`defender_took`, `take_voluntary`, `hand_p0`, `hand_p1`, `deck_remaining`, `trumps_p0`,
`trumps_p1`, `roles_swapped`, `winner_in_round`, `phase` (`early` / `mid` / `late`, defined by
a fixed deck-remaining cut that is recorded in the run manifest, not chosen after seeing data).

### 6.3 Level 3 — per move (sampled)

`run_id`, `game_index`, `round_idx`, `seq`, `actor`, `action_type`, `card_id`, `info_mode`,
`legal_action_count`, `chosen_index_in_legal_order`.

Sampling is **deterministic**: `game_index % sample_modulus == 0`. Not RNG-based, because an
RNG-based sample would differ between a 1-worker and an 8-worker run and break
reproducibility. Wall-clock decision latency is deliberately **not** recorded, since it is not
reproducible; compute cost is measured separately in the benchmark (Section 9.4).

### 6.4 Why three levels

Level 1 alone answers "who wins". The primary question is "how the value of information
changes over the course of a game", and that is a per-round question: information is worth
little when the deck is full and the opponent's hand is a large fraction of the deck, and worth
a great deal at the end. Level 2 makes that curve computable. Level 3 exists to explain
*mechanism* — what the informed strategy actually does differently — and is sampled because it
is roughly two orders of magnitude larger than level 1.

---

## 7. Data Quality

### 7.1 Invariants checked on every game

1. No strategy ever returns an illegal action.
2. Card conservation: `deck_remaining + |hand_p0| + |hand_p1| + |discard| + |table| == deck_size`
   holds after every single state transition, not just at game end.
3. No card appears in two locations simultaneously.
4. The winner's final hand is empty and the loser's is not.
5. Both players are never empty simultaneously.
6. The game terminates within the round cap.
7. Redeals stay within the cap.
8. The same `(run_seed, pairing_index, game_index)` reproduces the same result.

Invariants 2 and 3 are the ones that catch the expensive class of bug — a card duplicated
into two hands silently corrupts every downstream metric while leaving the game playable.

### 7.2 Failure policy

An invariant violation **raises** and marks the game. It is never swallowed and averaged
over. A run with any flagged game is not publishable until the flags are explained; the
anomaly counts appear in every dashboard's data-quality block regardless, so a reader always
sees them.

### 7.3 Validation of the engine by its own statistics

Two large-sample checks act as end-to-end tests:

- `RANDOM` vs `RANDOM` must produce a win rate of 0.5. This is exact, not approximate: with
  symmetric agents and symmetric seeding, neither side is favoured. A Wilson interval that
  excludes 0.5 means position, dealing, or seeding is biased. This is the only assertion with
  a known closed-form answer, and it is the strongest single check in the project.
- `RANDOM`'s win rate against each non-random strategy must be reported but **not asserted**.
  There is no closed-form value for it; any expectation written down beforehand would be a
  guess wearing the costume of a test.

A third check is comparative rather than exact: `GREEDY` must beat `RANDOM` by a wide margin
at large sample. A near-50/50 result there would indicate that the action enumeration or the
strategies are not doing what their descriptions say.

---

## 8. Metrics and Statistics

### 8.1 Primary metric (pre-registered)

**`delta = WR(SEARCH_INFORMED vs X) − WR(SEARCH_UNIFORM vs X)`**, aggregated over both
directions, for `X` ∈ {`RANDOM`, `GREEDY`, `CONSERVATIVE`}.

Pre-registration fixes: the metric, the opponent set, `deck_size = 36` as the primary
stratum, and a two-sided alternative. Anything else — 52-card results, per-opponent
breakdowns, phase curves — is secondary and labelled as such in the reports.

Matched seeds make this a **paired** comparison: `SEARCH_INFORMED` and `SEARCH_UNIFORM` play
the identical deal for game `i`. Discordant pairs are reported as a 2×2 table and tested with
**McNemar's exact test**, which is the correct test for paired binary outcomes and is far more
sensitive than comparing two independent Wilson intervals. The paired bootstrap (10,000
resamples, percentile interval) is reported alongside as an effect-size interval.

### 8.2 Multiplicity

The primary metric is evaluated across 3 opponents × 2 directions × 2 deck sizes. Holm-Bonferroni
correction is applied to the family and both raw and adjusted p-values are reported. Without
this, a 0.05 threshold across 12 comparisons manufactures findings roughly 46% of the time, and
the report would be quietly lying about its own confidence.

### 8.3 Secondary metrics

- **Opponent-independent strength:** win rate of each strategy against `RANDOM` only. This is
  the metric that cannot be distorted by the composition of the field, and it is the one to
  quote when comparing across studies with different palettes.
- **Phase curve of information value:** `delta` as a function of round index and of cards left
  in the opponent's hand. The headline claim of the project is expected to be that this is not
  a constant.
- **Information value conditioned on opponent trumps held** and **on hand size**, to test
  whether information about *which* cards matters differently from information about *how many*.
- **Throw-in efficiency:** attacks, throw-ins, and defended throw-ins per round, by strategy.
- **Positional advantage:** win rate conditional on `first_attacker`, in both the rule-derived
  and the rotation-balanced regime (Section 8.4).
- **Deal structure:** distribution of initial trump counts; share of deals with no trumps in
  either hand (≈2.8%); redeal frequency.
- **Compute cost:** games/second and CPU-seconds per game per strategy, reported next to every
  win rate. A strategy that wins 1% more while costing 100× the CPU is a different result, not a
  better one.

### 8.4 Controlling for position

Two regimes:

1. **Rule-derived** — first attacker by the low-trump rule. This is the honest "as played"
   distribution and it is what the primary metric uses.
2. **Rotation-balanced** — the first attacker is forced alternately across the seed space, so
   each strategy starts attacking exactly half the time regardless of what the deal gave it.

Without regime 2, a positional effect and a dealing effect are confounded and cannot be
separated. Both regimes are run; the difference between them is itself reported as a metric.

---

## 9. Reproducibility, Storage, Performance

### 9.1 Versioning

Three versions stamped into every row and every report:

- `ENGINE_VERSION` — simulation behaviour (deal, rules, strategies, search). Any change that
  could alter a result bumps it.
- `RULES_VERSION` — the normative rules in Section 3. Bumped when a rule changes.
- `SCHEMA_VERSION` — column shape.

Dashboards filter by these versions, so a dataset produced before a rule fix can never be
silently merged with one produced after.

### 9.2 Storage

Parquet as the raw store; DuckDB for queries; JSON aggregates as the single input to report
generation. Rationale: the workload is columnar aggregation over tens of millions of rows with
no transactional requirements, which is exactly what DuckDB is for and exactly what Postgres is
not. The JSON boundary means report generation has no database dependency and is trivially
reproducible.

### 9.3 Seeds

Tournament seed bounded by `2^31 - 1` so derived seeds stay inside `int64`. Per-game seeds are
derived by integer mixing from `(run_seed, pairing_index, game_index)` and never from
scheduling, wall-clock time, or process identity.

### 9.4 Performance

The source document's estimate of "240,000 games at under 1 ms per action" is not verifiable
from the document and is not adopted. Instead: correctness first, then a benchmark at the end
of the runner stage, then the final dataset scale is chosen from measured throughput so the
Wilson intervals are as narrow as the project can afford. The benchmark result and the chosen
scale are recorded in the run manifest and in the README.

---

## 10. Reports

Three self-contained HTML files, Chart.js 4.x from a CDN with a **pinned** version, embedded
JSON constants, no build step, no server. Each file opens directly from disk and is committed
to the repository. A shared module holds the CSS tokens, KPI cards and data table so the three
reports look like one system, but no report depends on the others or on the repository to render.

Every report contains: a header with run id, seed, deck size, the three version numbers and the
dataset build date; a KPI row; charts; a sortable table carrying interval columns; and a
**data-quality block listing every anomaly flag in the run**, not merely asserting that
everything was fine.

**Determinism:** the dataset build date is injected from run metadata. The reference dashboard
writes `new Date()` into the header, which makes the artifact non-reproducible; that pattern is
not copied. Regenerating a report from the same dataset must produce identical HTML bytes, and
a test asserts it.

PNG versions of the figures are generated for embedding in the README.

1. **The Price of Information** — the primary report. `delta` with paired and corrected
   intervals, the phase curve, `delta` against each opponent, and the compute cost that comes
   with the win rate.
2. **Tournament Overview** — win-rate matrix, rankings with intervals, game length, throw-in
   efficiency, the 36 vs 52 split.
3. **Rules & Dealing** — trump-count distribution, the share of deals with no trumps in either
   hand, empirical throw-in limits versus the rule, wins by final hand size (the degeneracy
   diagnostic), and take behaviour.

---

## 11. Testing

pytest. No existing framework exists in this repository; pytest was confirmed by the owner.

- **Invariants**, Section 7.1, on randomly generated games. Invariants 2 and 3 are checked after
  every transition.
- **Execution-mode equivalence [critical].** Identical seeds run with `NUMBA_DISABLE_JIT=1` and
  with JIT must produce bit-identical move trajectories and identical counters. This is the
  single test the project's correctness claim rests on.
- **Determinism.** Same seed twice; same seed at 1 and at 8 workers.
- **Legality.** Every strategy's action is validated against the rule engine on every move of
  thousands of games, which surfaces rare positions that random play reaches only occasionally.
- **Statistical sanity.** `RANDOM` vs `RANDOM` = 0.5 (exact, Wilson interval must contain it);
  `GREEDY` clearly above `RANDOM`; `SEARCH_INFORMED` ≥ `SEARCH_UNIFORM` at large sample. The
  last one is a model self-check — if the informed search ever loses to the uninformed one, the
  search or the particle filter is broken, and the failure must be investigated rather than
  averaged in. No win rate is asserted for `RANDOM` against a non-random policy, because no
  closed-form value exists for it.
- **Termination.** No game exceeds the round cap.
- **Report determinism.** Byte-identical HTML from identical JSON.

Performance benchmarks run under a `perf` marker, excluded from the default run, as in the
reference project.

---

## 12. Stages

Ten stages. One reviewable outcome and one commit each.

| # | Stage | Verified by |
|---|---|---|
| 01 | Repository skeleton: `pyproject.toml`, src layout, environment, pytest, `.gitignore` (including `.superpowers/reports/`), config module | Package imports; `pytest` runs green on an empty suite |
| 02 | Cards, PRNG, normative rules, invariant audit, equivalence harness; throwaway auto-policies drive the games | Invariants hold across 10k random games; both execution modes agree |
| 03 | `RANDOM` and `GREEDY`, strategy registry, single-threaded runner, telemetry levels 1-2, first 1k-game dataset, benchmark | `RANDOM` vs `RANDOM` = 0.5; `GREEDY` well above `RANDOM`; throughput measured |
| 04 | Particle filter, expectimax, `SEARCH_UNIFORM`, `SEARCH_INFORMED`, `CHEATER_DOC`; particle-count sweep | Equivalence holds; `INFORMED` ≥ `UNIFORM` on a sample; `delta` stable across K; cost measured |
| 05 | `CONSERVATIVE` on top of the opponent model | Legality, equivalence, and a plausible win-rate delta |
| 06 | Numba `prange` runner, telemetry level 3, Parquet schema, DuckDB views, resume support | Identical output at 1 and 8 workers |
| 07 | Statistics module and analysis queries producing `analysis/data/*.json` | Intervals match a hand-computed fixture |
| 08 | Full dataset run at the scale chosen from the stage-06 benchmark | Audit clean; rerun byte-identical |
| 09 | Report pipeline, three dashboards, PNG figures | Each report opens from disk; rebuild is byte-identical |
| 10 | README per the supplied guide: methodology, data, findings, limitations, reproducibility | Links resolve; every figure in the README matches a report |

Dependency order is not arbitrary: stages 04 and 05 need the opponent model, stage 06 needs a
correct single-threaded runner to parallelise, and stage 08 needs the stage-06 benchmark. Nothing
in stage 08 can be interpreted without stage 07's statistics.

---

## 13. Known Risks and Limitations

Stated up front so they are not discovered late:

1. **The win rule is endgame-degenerate** (Section 3.4). Attacks with one card in hand win
   unconditionally. This compresses the endgame and inflates single-card win rates. The
   alternative rule is the highest-value sensitivity to re-run.
2. **Expectimax over averaged opponent responses** (Section 5.3) is optimistic about moves with
   bad worst cases. It is a modelling choice, not an oversight.
3. **Particle count K is a confound** if the uninformed arm's strength varies materially with
   K. K is therefore swept, and the primary metric is reported across the sweep, not at a single
   K.
4. **Rule variant R7** (whether defense-card ranks enable throw-ins) materially changes
   throw-in rates. The default is fixed; the variant is a reported sensitivity.
5. **No human baseline.** All opponents are scripted.
6. **Fixed deck definitions.** Results do not generalise to other deck sizes or to variant rules.
7. **Search depth is bounded.** The informed strategy is not playing optimally; its win rate is a
   lower bound on the value of perfect information, which is a statement worth making in the
   report because it bounds the primary result from one side.