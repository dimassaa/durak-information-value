# Measurement plan: what has to be run, what data comes out, what it decides

Written after Task 21 and the depth-2 verdict, while the depth-2 confirmation run was still in
flight. Its purpose is to make the remaining work an explicit budget rather than an open-ended
sequence of "and then we should probably also check...".

The plan is ordered by what each experiment can *decide*, not by effort. Section 1 comes first
because until the search arms are competent players, every number measured between them is a number
about a weak policy class, and a weak policy class cannot answer the research question.

Datasets live in `datasets/`, which is gitignored: they are large and reproducible from a seed.

---

## 0. Ground rules that apply to every experiment

Four constraints discovered the hard way. Each has already produced one wrong conclusion, so they
are stated before the experiments rather than after.

**0.1 The stratifier must be fixed before any move is made.** Rounds, actions, takes and
"informational density" are all counts of actions taken, so they are functions of the treatment.
Bucketing an outcome on them selects differentially between arms and manufactures a curve — it
produced a spurious "information is worth minus 2.5 points" that survived into a committed file.
Pre-treatment handles: `initial_trumps_p0/p1`, `both_have_trumps`, `redeals`, `first_attacker`.

**0.2 `winner` is a seat; `strategy_p0` is an id.** Comparing them has produced three confident false
findings in this project, and one of them was a whole measurement harness. Every win-rate code path
resolves the winner through the seat the arm occupied. A harness that reports `games: 1500` next to
`trials_a: 1000` is not to be trusted whatever else it says.

**0.3 Compare the same seed and the same games across arms.** Fixing the seed makes arms paired in
the strongest available sense, which is what allows a small sample to detect a small effect. Two
separate tournaments are two separate samples and their difference is not a significance test.

**0.4 A missing measurement is reported as missing, never as zero.** A zero on a chart beside real
numbers reads as "measured, and it is zero", which is a different and false claim.

---

## 1. The evaluation weights (blocking)

**Why first.** The depth-2 verdict left the search losing 0.0003 of its games to `GREEDY` — 10 wins
in 40,000, measured through the trusted path. The diagnosed cause is not subtle: `GREEDY` never
takes (1.3% of its defensive actions), the search took 21.4% at depth 1 and still takes 12.5% at
depth 2. In a game whose win condition is emptying your hand, taking is the only way to add to your
own hand, so take discipline is worth more than card selection. The weights
`(1.0, 2.0, 0.5, 0.3, 1.5)` — card count, trumps, high cards, role, deck exhausted — have never
been checked against play.

**The methodological constraint that matters most.** The weights must be selected on **absolute
strength against a fixed reference panel**, never on the information delta. Tuning the weights until
the headline delta grows is fitting the answer to the question. The two search arms must share one
weight vector chosen before any delta is looked at, or the primary result is circular.

### 1.1 The sweep

| | |
|---|---|
| variants | 12 weight vectors, see 1.2 |
| reference panel | `GREEDY`, `CONSERVATIVE`, `AUTO_DEFEND`, `RANDOM` — fixed, unchanging |
| per variant | 4 panel opponents × both seat orders × **10,000 games** |
| head-to-head | `SEARCH_UNIFORM` vs `SEARCH_INFORMED` at each variant × **10,000 games** |
| total | ≈ 600,000 games, ≈ 15 min at the measured 1.3 ms/game effective |
| seed | one fixed seed, identical across every arm |

### 1.2 The variants

A one-at-a-time sweep cannot see interactions, so: five single-term perturbations to locate each
term's effect, then five combinations, then two deliberately extreme points to check the corners.

| # | vector | purpose |
|---|---|---|
| 1 | `(1.0, 2.0, 0.5, 0.3, 1.5)` | the incumbent |
| 2 | `(3.0, 2.0, 0.5, 0.3, 1.5)` | card count tripled |
| 3 | `(1.0, 1.0, 0.5, 0.3, 1.5)` | trumps halved |
| 4 | `(1.0, 0.0, 0.5, 0.3, 1.5)` | no trump term at all |
| 5 | `(1.0, 2.0, 2.0, 0.3, 1.5)` | high cards quadrupled |
| 6 | `(1.0, 2.0, 0.5, 0.0, 1.5)` | no role term |
| 7 | `(5.0, 2.0, 0.5, 0.3, 1.5)` | card count 5×, to see whether a floor exists |
| 8 | `(3.0, 0.5, 1.0, 0.0, 1.0)` | cards and trumps only |
| 9 | `(2.0, 0.0, 0.0, 0.0, 0.0)` | pure card count, nothing else |
| 10 | `(1.0, 2.0, 0.5, 0.3, 3.0)` | deck-exhaustion term tripled |
| 11 | `(2.0, 1.0, 0.5, 0.3, 1.5)` | cards up, trumps down |
| 12 | `(0.5, 3.0, 0.5, 0.3, 1.5)` | the opposite corner: trumps up, cards down |

Variant 9 and 12 are the corners that matter most: if pure card count beats the incumbent against
the panel, the incumbent's trump term is actively harmful, which is the hypothesis the depth-2
analysis produced.

### 1.3 What each variant is measured on

| metric | why |
|---|---|
| win rate vs `GREEDY` | the current failure, 0.0003 |
| win rate vs panel | absolute strength, independent of which opponent |
| take rate | the diagnosed axis; the target is `GREEDY`'s 0.013 |
| final hand size | the symptom, 4.88 against `GREEDY`'s 0.38 |
| rounds per game | a proxy for whether the game is being played at all |
| information delta at that variant | recorded, but **not** used to choose |

### 1.4 What it decides

- **Whether a competent search arm is reachable.** If no variant beats `GREEDY` by more than noise,
  the weakness is not the weights and the next suspect is the search's structure, not its evaluation.
- **Whether the information delta is weight-sensitive.** This is a *scientific* result in its own
  right: if the delta moves from 0.04 to 0.16 across the panel of plausible weightings, then "the
  price of information" is not a number but a function of the evaluator, and no report may state it
  as a single figure.
- **Which vector to freeze.** One vector, both arms, before any delta is published.

### 1.5 Cost of being wrong

A sweep that selects on absolute strength can still land on a vector that is strong *and* gives an
unrepresentative delta. The guard is to report the delta across **all twelve** variants alongside the
chosen one, so the headline is published as a range over the weight space rather than a point.

---

## 2. The oracle / ceiling (blocking for interpretation)

**Why.** "+0.08" currently means "at least 0.08, to this policy class". Without an upper bound that
sentence is the most the data supports, and it is not the same claim as the research question.

| | |
|---|---|
| model | knows the opponent's hand **and the deck order** |
| search | depth 4–5 |
| tournament | one most-global tournament, all strategies, honest-model tournaments kept separate |
| budget | 200,000 games, single tier |
| seed | fixed |

**Decides:** the achievable ceiling. If the ceiling is 0.10, then 0.08 is 80% of everything
attainable and the result is strong. If it is 0.40, the headline is a tenth of the truth and the
honest report must say so. **This experiment cannot be skipped and its answer changes the tone of
every report.**

---

## 3. Rule variants (the engine is the largest single source of effect)

The external audit (see `IMPROVEMENTS.md` 49–50) found one implemented convention that differs from
the sources and two rules that are not implemented at all. The measured sensitivity to the throw-in
rule is already **7.9 points**, against a headline of 8.0 — the rule choice moves the result as much
as the thing being measured.

### 3.1 The variant grid

| variant | values | status | expected effect |
|---|---|---|---|
| `throwin_defense_ranks` | `False`, `True` | measured | 0.19175 → 0.27075 |
| first-defence cap | 5, 6 cards | **not implemented** | unknown; sources say the first round is 5 |
| throw beyond hand when deck empty | allowed, forbidden | **not implemented** | unknown; touches the endgame, where 24.6% of games end with the loser on 32–35 cards |
| trump revealed from | next card, last card | implemented (last) | **nil**, statistically identical under a uniform shuffle |
| redeal on all six trumps | on, off | implemented | unknown |
| deck | 36, 52 | only 36 ever played | unknown |

### 3.2 Design and cost

Two settings each, 10,000 games per cell, **paired** — the same seed and the same games under both
settings, so the difference is a within-game contrast and not two samples. Six variants × 2 × 10,000
≈ 120,000 games ≈ 3 min. Cheap enough that there is no excuse for leaving any of it unmeasured.

### 3.3 What it decides

- Which readings are *conventions* (no effect) and which are *ambiguities* (large effect). The
  distinction is the whole point: a convention needs no caveat, an ambiguity needs both readings.
- Whether the primary result is robust to rule interpretation, or is an artifact of one reading.
- Whether the endgame rules, which concentrate 24.6% of games, belong in the main comparison at all.

**Cost of implementing the two missing rules:** each is a `RuleTunables` field and a branch in
`rules.py`, plus the differential test against the engine that already exists. This is the only work
in this plan that touches the rules engine, and it is the riskiest, so it is scheduled after
everything else and behind its own review.

---

## 4. Stratification: where the information is actually worth something

The trump-abundance result (delta +0.166 with no trumps dealt down to +0.043 with five or more, all
intervals excluding zero, monotone) is the project's most interesting finding and it is currently
based on one dataset at one weight vector, with **no identified mechanism**.

| experiment | games | decides |
|---|---|---|
| Re-measure the trump curve at each of the 12 weight vectors | 10,000 per bucket per variant | is the 4× spread a property of the game or of the evaluator? |
| Trump count **per player**, not per deal | free, level 1 | separates "one player is short of trumps" from "both are" |
| Both-deals-have-no-trump cell | 859 games in the primary dataset | too small; needs a dedicated run to say anything |
| Deck 52 at the same stratification | 10,000 per bucket | does the curve survive a second deck? |

**Decides:** whether the report may claim information is worth more when trumps are scarce. It may
not claim the *mechanism* — that no experiment here tests.

---

## 5. What must be re-measured once the weights are frozen

Nothing from the current 1.14M-game dataset survives a weight change. It is spent as soon as variant
2 differs from variant 1.

| tier | games | purpose |
|---|---|---|
| pilot | 114,000 | the 10% tier; catches join, schema and serialisation defects, which it has caught three times |
| **primary** | **1,140,000** | the headline, 20,000 per cell on deck 36, both seat orders |
| sensitivity | 228,000 | the full deck × rules × position grid, 2,000 per cell |
| total | ≈ 1,482,000 | ≈ 35 min at 12 workers, given the measured 1.3 ms/game |

The pilot is not optional. Every defect in section 6 was caught by a pilot, and none would have been
caught by a full run.

---

## 6. Data defects that block analysis regardless of weights

None of these are caused by the weights, none are fixed by a bigger sample, and three of them make
reported numbers wrong rather than merely unavailable.

| # | defect | consequence | fix | cost |
|---|---|---|---|---|
| 6.1 | `game_index` is cell-local; levels 2 and 3 carry no `pairing_index` | every level-2/3 aggregate is an average over 57 unrelated games; `game_index` fans out 39,073,046 → 2,227,163,622 | stamp `pairing_index` onto the round and move schemas, bump `SCHEMA_VERSION`, re-run | schema change; blocks all phase analysis |
| 6.2 | `first_attacker_p0/p1_wins` count **seat** wins, not first-attacker wins | reads a clean 570,154/569,846 while the real first-attacker rate is 0.47999 | rename to `seat0_wins`/`seat1_wins`, add a real `first_attacker_wins` | trivial |
| 6.3 | `ROUND_DTYPE.n_attacks` is 1 in all 39M rows | a report built on it understates round activity by the throw-in count | rename, or count attacks rather than attack-openings | trivial |
| 6.4 | levels emit one extra round record per game | a per-game round count from level 2 is wrong by one | the end-of-game flush writes `round_idx` where the in-game flush writes `round_idx - 1` | trivial |
| 6.5 | determinism tests fail on a cold Numba cache | workers race on the cache index; the project's own "clear `__pycache__`" advice collides with it | pre-warm in the parent before the pool, or drop `cache=True` from worker-compiled functions | small |

**6.1 is the only one worth scheduling on its own.** The plan's per-phase analysis is impossible
without it, and the phase breakdown is the natural place to show *when* information pays.

---

## 7. Statistical power, stated before the runs

Recorded so a scale can never again be chosen after seeing what it produced.

| target | required n | basis |
|---|---|---|
| Wilson half-width 0.007 on a rate near 0.5 | 20,000 per cell | the primary tier's design point |
| Wilson half-width 0.02 | 2,400 per cell | the sensitivity grid |
| detect a paired delta of 0.02 with 80% power at α=0.05 | ≈ 9,800 pairs | `proportion_diff_interval` |
| detect a paired delta of 0.05 | ≈ 1,600 pairs | why 10,000 per weight variant is generous |
| a difference that is *not* real must not look real | Holm–Bonferroni over the family | already implemented |

The multiplicity point matters for section 1: twelve weight variants is a family of twelve, and the
adjusted values must be reported, not just the raw ones.

---

## 8. Total budget

| section | games | wall clock (12 workers) |
|---|---|---|
| 1. weights | 600,000 | ≈ 15 min |
| 2. oracle | 200,000 | ≈ 5 min |
| 3. rule variants | 120,000 + implementation | ≈ 3 min + a rules-engine change |
| 4. stratification | 480,000 | ≈ 12 min |
| 5. full re-run at frozen weights | 1,482,000 | ≈ 35 min |
| **total** | **≈ 2,882,000** | **≈ 70 min + the rules-engine work** |

The games are not the cost. The costs are the oracle model's implementation, the schema change in 6.1,
and the judgement calls in section 1.4.

---

## 9. The order, and why

1. **Section 1** — a competent search arm, or every number afterwards is about a weak policy class.
2. **Section 2** — the ceiling, because it decides whether the headline is a strong result or a
   tenth of one.
3. **Section 5** — the full dataset at frozen weights, since everything downstream depends on it.
4. **Sections 3 and 4** — sensitivity and stratification, on the frozen configuration.
5. **Section 6** — the two trivial renames any time; 6.1 when the phase analysis is actually wanted.

## 10. What must never be claimed from any of this

- That "the price of information" is a single number. It is a function of the weight vector, the rule
  reading, and the trump composition of the deal. All three are measured; the range must be reported.
- That the search arms are strong. Until section 1 produces one that beats `GREEDY` at a rate a
  mediocre strategy would, they are a policy class and the delta describes them.
- That the rules engine is correct because it is internally consistent. Eight of nine externally
  checkable rules match, which is evidence and not proof; the take-rule bug was logically coherent
  for thirteen stages.
- Anything about particle count `k`. It is fixed by a compiled constant and cannot be varied, so
  stability of the delta across `k` is **unmeasured** and no claim is available in either direction.