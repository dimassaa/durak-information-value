# Run catalogue

Every measurement this project intends to make, with its configuration, its scale, and — the part that
matters — **what it decides**. An experiment with no decision attached is a number nobody will act on.

Chosen scope: all eleven experiment families, a two-tier dataset (1.14M primary + 228k sensitivity), the
"first defence is five cards" rule variant, and all four deliverable formats.

**Hard ordering rule.** Runs R1–R4 read `durak/rules.py`, `durak/leaf.py`, `durak/engine.py` and
`durak/telemetry.py`. Nothing in that list may be edited while any of them is in flight: the run would
silently measure a mixture of two code versions and the result would be uninterpretable. Engine edits are
therefore batched between measurement blocks, never interleaved with them.

**Standing rule.** Clear `durak/__pycache__` and `durak/strategies/__pycache__` before any run whose result
will be believed, and treat four identical numbers across a parameter sweep as a broken sweep until proven
otherwise. Numba's disk cache does not invalidate when a module global changes value.

---

## Phase A — unblock the model (no measurement)

### A1. Bounded memory

**Why.** Every model currently remembers every card ever played: the particle filter excludes the entire
discard pile from the opponent's candidate set. That makes memory perfect and the information question
binary — know the hand exactly, or don't. Real players remember the last few cards and reconstruct the
rest. Making memory a parameter turns "information" into a spectrum and makes the project answer a question
about *how much* history is worth, not just whether the present is.

**What changes.** `opponent._unseen_into` excludes only the cards played within the last `M` plies; cards
discarded earlier return to the candidate set, so the filter must also place them somewhere — deck or
opponent's hand — which is precisely the uncertainty being modelled. `M = ∞` reproduces the current
behaviour and `M = 0` reproduces no memory of played cards.

**Decides.** Whether the measured effect comes from the opponent's hand specifically or from *any*
information about card flow. If a bounded-memory arm matches the perfect-memory arm, the project's headline
is about bookkeeping rather than about the opponent.

**Cost.** One parameter threaded through the filter plus a test that `M = ∞` reproduces today's numbers
exactly. Two days' risk, no measurement.

### A2. Make the evaluation weights a run parameter

**Why.** `EVAL_WEIGHTS` is a module constant, and `@njit` freezes globals at compile time, so the weight
sweep as it stands would measure the same policy twelve times — the same trap that flattened the depth
sweep. Weights must arrive per run, from configuration, and be recorded in the manifest.

**Decides.** Nothing; it unblocks R3 and R4.

---

## Phase B — the experiments

### R1. Depth curve {1, 2, 3, 4}

| | |
|---|---|
| question | does the information effect fall, stay flat, or turn around with depth? |
| config | one fixed seed per depth, **cache cleared before each**, `SEARCH_DEPTH` from `DURAK_SEARCH_DEPTH` |
| scale | games per direction **falling with depth**, because cost rises by the branching factor (~4.80) per ply: 2000 / 1000 / 500 / 250. Equal counts would take about five hours; this buys the sign at every depth and the precision achieved is printed, not assumed |
| also measured | win rate against `GREEDY`, `AUTO_DEFEND`; take rate; final hand; wall clock |
| decides | H1 and H2 in `RESEARCH.md`. Whether any headline number can be published at all |
| precedent | depths 1–3 gave −0.0120, −0.0580, −0.0840 on the pre-deck-hiding model: monotone, not turning around |

**Runs first.** Everything downstream inherits a chosen depth, and depth is a research parameter — picking
it for speed before measuring the curve is how the project's answer gets changed by accident.

### R2. Oracle as ceiling

| | |
|---|---|
| question | what is the most perfect knowledge of the hand could possibly be worth? |
| config | `ORACLE` against `ORACLE_UNIFORM` — same depth, same weights, one line differing — plus both against the fixed panel |
| scale | 20,000 games per direction |
| decides | whether a measured delta is a strong result or a fraction of one |
| status | **must be re-measured.** The first oracle was void: with the pile public, the deck-order axis did not exist and the hand was derivable |

Bounding a difference needs both arms at the depth being bounded. `ORACLE` against a shallower
`SEARCH_UNIFORM` would confound "does knowledge pay" with "does depth pay", and those point in opposite
directions.

### R3. Weight sweep, twelve vectors

| | |
|---|---|
| question | does the evaluator move the measured effect, or the information? |
| config | one fixed seed, **one shared random draw-order hypothesis per decision**, so the arms differ in the hand and in nothing else |
| variants | the incumbent `(1.0, 2.0, 0.5, 0.3, 1.5)`; card count 3× and 5×; trumps halved and removed; high cards 4×; role removed; deck term 3×; cards-up/trumps-down; cards-down/trumps-up; pure card count |
| scale | 10,000 games per opponent per variant against a fixed panel, **≈ 600,000 games** |
| decides | H1. Whether a competent search arm exists at all |

**The trap.** Selected on **absolute strength against the fixed panel, never on the delta**. Tuning weights
until the headline grows is fitting the answer to the question, and both arms must then share one frozen
vector. The delta is *recorded* at every variant and published across all twelve, so the headline is a
range over the weight space rather than a point.

**Risk.** If no variant beats `GREEDY` comfortably, every earlier number was measured with a broken
instrument and the programme restarts from the pilot.

### R4. Extended weight surface

A full grid over the five terms plus random interior points, so interactions are visible rather than
inferred from rays. Only worth running if R3 shows a variant worth refining; a surface over a space where
nothing is competitive is decoration.

### R5. Memory horizon

The same head-to-head at `M ∈ {0, 2, 4, 8, 16, ∞}`, everything else frozen. Decides how the value of
information distributes across history, and whether the hand specifically matters or card flow in general.

### R6. Trump stratification

Delta by trumps in the opening deal, at each weight vector, on deck 36 and 52. Trumps in the deal are fixed
**before any move**, which is the only kind of stratifier that does not select on the treatment — the
project has already retracted one curve for using a stratifier produced by the arms' own actions.

**Reports the curve, never the mechanism.** No trump hierarchy versus hand homogeneity is untested and must
not appear in a report.

### R7. Deck 52

Never played. Answers whether the trump curve and the effect survive a second deck, and whether anything
measured is an artifact of a 36-card deck.

### R8. Phase analysis

Blocked on **A3**: levels 2 and 3 carry no `pairing_index` and `game_index` is cell-local, so any join fans
out by 57 and no level-2 quantity can be attributed to a strategy. Fixing it is a schema version bump and
invalidates existing datasets, so it comes after the schema is otherwise settled. Decides *when* in a game
information pays, which is the natural home for a bounded-memory result.

### R9. Seed robustness

The same configuration under several seeds. A conclusion that depends on one seed is a conclusion about that
seed. Cheap, and it is the check most often skipped.

### R10. Self-play

Each arm against itself at each depth, to separate "this depth is stronger" from "this depth is stronger
against these particular opponents". Two non-comparable rates, both needed.

---

## Phase C — publication runs

### R11. Pilot at 10%

**114,000 games. Not optional.** Every one of the six instrument defects in this project was caught by a
pilot, and none would have been caught by a full run. The pilot runs first, every time.

### R12. Primary dataset — 1.14M games

20,000 games per cell on deck 36, both seat orders, at frozen weights and chosen depth. Wilson half-width
≈ 0.007. ≈ 35 minutes at 12 workers.

### R13. Sensitivity grid — 228,000 games

2,000 per cell across the deck × rule × position grid, including the rule variants of R14. ≈ 7 minutes.

---

## R14. Rule variant: first defence is five cards

Several sources state the first defence round is capped at five cards rather than six. A `RuleTunables` field
and a branch in `_throwin_legal` — and in `leaf._throwin_legal_in`, which is a **second implementation that
must agree with the first**. Measured under both readings, paired, same seed and same games.

**The second implementation is the risk.** The leaf exists because `GameState` cannot be constructed inside
compiled code, so the throw-in legality test is duplicated. A change to one and not the other is a silent
model divergence, and its differential tests against the engine would pass on the path that was updated.

**Why this variant and not the other.** The deck-empty throw-in variant acts in the endgame, where a
quarter of games end with the loser holding 32+ cards — but those are `AUTO_ATTACK` games, and that
baseline is degenerate (it never defends, and loses 100% of them). Measuring a rule against a null opponent
would produce a number about the null opponent. The five-card cap is the variant with a clean read.

---

## Order and budget

| block | runs | wall clock | gate |
|---|---|---|---|
| A | A1, A2, A3 | — | model correct and configurable |
| B1 | R1, R2, R11 | ≈ 60 min | depth chosen, ceiling known |
| B2 | R3, R5, R9 | ≈ 90 min | weights frozen |
| C | R12, R13 | ≈ 45 min | headline published |
| D | R6, R7, R8, R10, R14 | ≈ 40 min + schema work | robustness published |

≈ 2.9M games, ≈ 4 hours of measurement plus the model work in block A. **The games are not the cost** — the
cost is making depth, weights and memory configurable, and the duplicated legality test in the leaf.

## What may not be published regardless of results

- The withdrawn `+0.0799`.
- "The price of information" as a single number. It is a function of depth, of the weights, of the rule
  reading, of the memory horizon and of the trump composition. All are being measured; the range is the
  finding.
- That the search arms are strong. They lose ~99.5% of their games to a plain heuristic.
- Anything about particle count. It is fixed by a compiled constant and cannot be varied.
- A causal explanation of the trump curve.