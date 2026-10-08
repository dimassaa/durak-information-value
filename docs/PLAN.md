# Plan: from here to the end

Replaces the previous plan. Written after the depth-2 control run showed the published headline is
not reproducible with current code. Decisions here are stated as **branches**: where an interpretation
changes what we build, both readings and their consequences are given up front, so nothing has to be
re-argued when the measurement lands.

Chat reporting for this plan is deliberately short. Detail lives in this file and in `IMPROVEMENTS.md`.

---

## 1. What is actually true right now

Verified, with the path that verified it.

| claim | evidence |
|---|---|
| the measurement instrument is sound | 0 anomalies in 1.14M games; compiled == interpreted bit for bit; byte-identical at 1 and 10 workers; `RANDOM` vs `RANDOM` symmetry inside its interval |
| 8 of 9 externally checkable rules match the engine | audit against Wikipedia / TreeGame / GAMBITER / stavchess / lenta; the take-rule bug is fixed and its differential test is exact |
| the depth-2 leaf is correct | four differential tests against `rules.apply_action`, exact equality, plus enumeration tests against `rules.legal_actions` |
| the search arms are very weak | 0.0003 against `GREEDY` at depth 1, 40,000 games — 10 wins |
| **the published headline is withdrawn** | one harness, three points: old dataset +0.0799, current depth 1 **+0.0176** [+0.0078, +0.0274], current depth 2 **−0.0527** [−0.0625, −0.0429] |
| no valid dataset exists | the 1.14M-game dataset was lost when `/tmp` was cleared, and it predated the leaf fix anyway |

**The headline finding so far is not a number, it is a direction:** every improvement made to the
search has *reduced* the measured information effect, +0.080 → +0.018 → −0.053. Whether that
continues downward or turns around is the central open question of the project.

## 1a. Status, and one hard scheduling constraint

**Done:** the leaf and the ply enumeration are depth-parameterised (`leaf_profile_after_n`,
`plies_after`); the search is depth-parameterised by an ITERATIVE explicit-stack walk, because a
recursive `@njit` kills every spawned worker; depths 1-4 are pinned for legality and determinism, and the
four-ply leaf and the three-ply enumeration are pinned against the engine. The depth-4 test found a real
bug -- the leaf's loop dropped every ply after a round end, which is invisible at depths 1 to 3.

**Two measurement traps found the hard way, both now in `IMPROVEMENTS.md`:**
- numba's disk cache does NOT invalidate when a module global changes value, so a depth sweep with a warm
  cache returns *identical* numbers at every depth -- a flat curve that looks like a finding. `SEARCH_DEPTH`
  is now read from `DURAK_SEARCH_DEPTH` at import, and the sweep clears `__pycache__` before each depth.
- the depth-curve harness passed the info modes as a fixed pair to both seat orders, which handed the
  UNIFORM arm the truth in the mirror game; the two directions cancelled to a flat `+0.0000`. Info modes
  must follow the ARM, not the seat.

**The constraint.** Every remaining item -- the two unimplemented rules, the `pairing_index` schema fix --
edits `durak/rules.py`, `durak/leaf.py`, `durak/engine.py` or `durak/telemetry.py`. Those are exactly the
files the depth-curve run is reading. **They must not be edited while a measurement is in flight**, because
the run would silently measure a mixture of two code versions and the result would be uninterpretable.
So the depth curve runs first, and the remaining engine edits are batched behind it rather than interleaved.

## 2. What remains

Ordered. Items 1–3 are blocking; the rest are not.

| # | item | blocks | effort |
|---|---|---|---|
| 1 | rebuild a baseline dataset into `datasets/`, at current code | everything | 35 min run |
| 2 | **depth curve {1, 2, 3, 4}** — the delta at each depth | the headline, item 5, the reports | ~2 h incl. optimisation |
| 3 | depth as configuration + differential tests parameterised over depths | 2 | ~3 h |
| 4 | speed: `prange` over particles, exact dominance pruning | 2 and 5 | ~2 h |
| 5 | weight sweep, selected on absolute strength | 6, 7 | ~1 h run |
| 6 | oracle / ceiling | the interpretation of the headline | ~4 h build |
| 7 | reports, README, final write-up | — | ~3 h |
| 8 | rule variants: 2 unimplemented rules, 6-cell grid | 7 | ~4 h, touches `rules.py` |
| 9 | schema fix: `pairing_index` on levels 2 and 3 | any phase analysis | ~2 h + re-run |
| 10 | three trivial renames in the queries (`IMPROVEMENTS.md` 6.2–6.4) | nothing | 20 min |

Item 1 is first because there is currently no dataset at all, and every later measurement writes one.

## 3. The central experiment, item 2

Depth is not a performance knob. It is a **research parameter**, and the project's answer moved sign
when it changed. So the depth curve is measured before any depth is adopted.

**Design.** For each depth in {1, 2, 3, 4}: the informed-versus-uniform head-to-head on one fixed
seed, both seat orders, plus the four-opponent panel. Report at each depth: the delta with its
interval, the win rate against `GREEDY`, the take rate, the final hand size, and the wall clock.

**Cost, and the reason item 4 exists.** Measured branching factor is 4.80. At 100 particles that is
11,052 leaf evaluations per decision at depth 2 and 254,541 at depth 4. Depth 7 is 28 million —
about 86 games in five minutes, which measures nothing. **Depth 4 is the deepest usable depth.**
Runs may take hours, so this is not a wall-clock constraint; it is a compute one, and item 4 exists to
push depth 4 down from hours to minutes.

### Branch R1 — the curve turns around (non-monotone)

The likely case if the depth-1 figure was an artifact of a weak evaluation. Information needs a
competent player to pay off; depth 3–4 supplies one.

**Then:** adopt the depth that maximises the delta, not the depth that is cheapest. Report the whole
curve, and state that the effect is depth-dependent. The oracle becomes the next blocking item,
because the ceiling now bounds a number worth bounding.

### Branch R2 — the curve is flat at or below zero

Perfect knowledge is worth nothing to a player that reasons two plies ahead.

**Then:** the honest result is *"under this rule set and this evaluation, perfect knowledge of the
opponent's hand is worth approximately nothing"*, and it is a **negative result worth publishing**,
not a failure. Before saying it: sweep the weights (item 5) and the rule readings (item 8), because a
null that survives both is a finding, and a null that disappears under either is a statement about the
weights or the rules instead.

### Branch R3 — depth 3 or 4 costs more compute than is worth

**Then:** cap the depth, say so in the report as a stated limit, and run the curve at the depths that
fit. A truncated curve is still a curve; an unmeasured depth is a guess.

## 4. Item 4 — speed, without touching the model

Rejected on principle: reducing the particle count. It changes the model of the opponent.

### Accepted, in order

| # | technique | gain | why it is safe |
|---|---|---|---|
| 4a | `prange` over the particle loop | **an order of magnitude** | the particle array is a plain 2-D `int`, not a structured array, so the numba limitation that blocked `prange` in the engine may not apply. One test decides it. |
| 4b | exact dominance pruning | ×1.5–2 | `evaluate` is a weighted sum of five terms. If one action's profile is no worse on **all five**, the other cannot win and is dropped. This is proof, not heuristic. |
| 4c | transposed evaluation order for cache locality | ×1.2–1.5 | pure layout change, identical arithmetic |

### Measured before deciding

| technique | verdict |
|---|---|
| alpha-beta | **inapplicable.** It requires minimax; this is expectimax, and an average cannot be bounded by a partial subtree. Using it would corrupt the model. |
| cache the take rate as an average | **void.** A predecessor averaged the leaf over sampled opponent moves; the leaf no longer depends on the opponent's move, so the average is of a constant. |
| transposition table | **undecided, needs one measurement.** The leaf is deterministic given (hands, table, deck, kind, card), so it is cacheable — but hashing ~100 cards may cost more than the leaf itself, and a mutable global is exactly what numba refused before. Measure the repeat rate at depth 3–4 first: under 5% repeats means no table. |
| determinisation / ShuffleCopy | **already implemented.** The particle filter is precisely this. |
| MCTS / IS-MCTS | **deferred.** Naive MCTS is wrong under imperfect information — it merges distinct opponent hands in one node and wins by concealment. Proper IS-MCTS would attack both item 6 and the weakness at once, but it is a project of its own and should not be started before the depth curve is known. |

## 5. Item 3 — depth as configuration

Today depth is not a parameter, it is a separate code path per depth: `second_plies`,
`third_plies`, `leaf_profile_after_ply3`. Depth 4 means two more functions and no end to it. This
does not scale and it does not make runs comparable.

**Design.** One recursive `@njit` implementation with `depth` as an argument. The state is passed as
explicit arrays, as the leaf already does — `GameState` cannot be constructed inside compiled code,
which cost time already. Depth arrives through the run configuration, is written into the manifest,
and **appears next to every published number**, because depth 4 against depth 1 is a comparison of
two different policies.

**Risk R4.** Generalising the leaf may break the exact differential tests, which are the only thing
that makes depth 2 trustworthy. Mitigation: parameterise those tests over depths {1, 2, 3, 4} and
require exact equality at each before the generalisation is considered done.

## 6. Item 5 — the weight sweep, and the trap in it

The weights `(1.0, 2.0, 0.5, 0.3, 1.5)` have never been checked against play, and the diagnosis says
they are the prime remaining suspect: `GREEDY` never takes (1.3% of defensive actions), the search
took 21.4% at depth 1, and taking is the only way to add to your own hand in a game won by emptying
it.

**The trap: the weights must be selected on absolute strength against a fixed panel, never on the
delta.** Choosing weights until the headline grows is fitting the answer to the question, and the two
arms must then share one frozen vector. The delta is recorded at every variant and **published
across all of them**, so the headline is a range over the weight space rather than a point.

Twelve variants, 10,000 games per opponent per variant, one fixed seed. ≈ 600,000 games.

**Risk R5.** If no variant beats `GREEDY` by more than noise, the weakness is not the weights and the
next suspect is the search's structure. That is a real possibility and the sweep is designed to
detect it rather than to assume an answer.

## 7. Items 6 and 8 — the ceiling and the rules

**Oracle (item 6).** Knows the opponent's hand and the deck order, depth 4–5, one most-global
tournament. It exists to decide whether the headline is a strong result or a fraction of one. This is
**blocking for interpretation, not for measurement**: any number can be produced without it, but no
number can be *characterised* without it.

**Rules (item 8).** The audit found one implemented convention that differs from the sources (trump
revealed from the last card rather than the next — statistically identical under a uniform shuffle,
so no result depends on it) and two rules not implemented at all:

- the first defence round capped at five cards, which several sources state;
- throwing in more cards than the defender holds once the draw pile is empty.

The second matters more than it looks: it acts in the endgame, where 24.6% of games end with the loser
still holding 32–35 cards. This is the only work that touches `rules.py`, so it is scheduled last and
behind its own review, with its own differential tests against the engine.

**Already measured:** the throw-in-rank reading moves the win rate by 7.9 points, against a headline of
8.0. **The choice of rule reading moves the result as much as the thing being measured.** That
sentence belongs in the report whatever the final number is.

## 8. Item 9 — the schema fix

`game_index` is cell-local, so levels 2 and 3 cannot be joined to level 1: a naive join of 40,000
games returned 39,073,046 round rows instead of ~2.4 million, fanning out by 57. Every level-2 number
is an average over 57 unrelated games, and none can be attributed to a strategy.

Fix: stamp `pairing_index` onto the round and move schemas, bump `SCHEMA_VERSION`, re-run. **Risk R6:**
it invalidates every existing dataset, so it is done *after* the depth curve, or the depth curve would
have to be run twice.

## 9. What may never be claimed

- **That "the price of information" is a number.** It is a function of depth, of the weight vector, of
  the rule reading, and of the trump composition of the deal. All four are being measured; the range
  must be published.
- **That the search arms are strong.** They lose ~99.5% of their games to a plain heuristic. Until a
  weight vector changes that, every delta describes a policy class, not the quantity of interest.
- **That the rules engine is correct because it is internally consistent.** Eight of nine externally
  checkable rules match, which is evidence and not proof — the take-rule bug was logically coherent for
  thirteen stages.
- **Anything about particle count.** It is fixed by a compiled constant and cannot be varied. Stability
  of the delta across `k` is unmeasured and no claim is available in either direction.