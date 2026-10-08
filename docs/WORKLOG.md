# Work log: what was built, what broke, what was decided

A narrative of the project so far, written after Task 18's first full dataset run. It exists because
the ledger (`progress.md`) records decisions one at a time and is not readable as a whole, and because
several of the wrong turns below are worth keeping visible.

Scope: a simulation of подкидной дурак (Durak) and an analysis of one question — **what is perfect
knowledge of the opponent's hand worth?** Two search policies are the same expectimax over a particle
filter; they differ only in where the opponent's hand comes from. One is fed the real hand.

---

## 1. The shape of the answer so far

| | |
|---|---|
| informed (knows the hand) | 0.5399 — CI (0.5350, 0.5448) |
| uninformed (samples it) | 0.4601 — CI (0.4552, 0.4650) |
| **delta** | **+0.0799** — paired CI (0.0702, 0.0894) |
| McNemar exact | **1.96e-57** |
| games | 40,000 head-to-head, 1,140,000 in the dataset |
| anomalies | 0 |

**This is a lower bound, and the reason is structural.** The informed arm is depth-limited: it looks one
ply ahead and evaluates the result statically, so it does not convert everything it knows into wins. A
stronger player with the same information would score higher. That sentence belongs next to the number
in any report, not only in a README.

The dataset is clean on every check that exists: `RANDOM vs RANDOM` symmetry 0.4908 against a 0.5
interval, zero anomalies in 1.14M games, byte-identical output at 1 and 10 workers, compiled and
interpreted execution bit-for-bit identical.

## 2. What went wrong, in the order it went wrong

This is the part worth reading. Most of the project's real defects were not found by tests; they were
found by noticing that a number was strange and then measuring instead of theorising.

### 2.1 The rules were wrong for thirteen stages

`_do_take` handed the turn to the player who had just took the cards. In Durak a take does not pass the
lead: the original attacker keeps attacking and the taker skips. This is what makes it the *podkidnoy*
variant.

Symptom: 27–35% of search games never terminated, at 10,000 rounds, with the draw pile untouched and
two cards passing between the players forever. The first four-stage fix made games 28× faster as a side
effect, because games got 75× shorter.

I misdiagnosed this three times before finding it, and each misdiagnosis is recorded rather than edited
out: I blamed the particle filter, then the deck-exhausted evaluation term, then "a transfer reads as
progress". The third was a real bug — but fixing it changed nothing about the non-termination, which is
how I learned it was not the cause. **The rules engine was never compared against the game.** Building
a serious analysis on top of an unverified rule set is the single most expensive mistake in this project.

### 2.2 The leaf credited a take to the wrong player

`leaf.py`'s TAKE branch collected the picked-up cards into the opponent's hand. The evaluation reads
`cards_opponent - cards_player`, so handing the opponent cards *raised* the player's score — taking beat
every alternative.

Symptom: the search took 49.9% of its defensive actions, finished with 11.8 cards, and lost **100%** of
its games to a plain heuristic. A weak strategy loses about 55–60%; losing every single game is not a
strength gradient.

The transfer correction introduced at Stage 12 was **masking** this, not independent of it: it was
derived from the same wrong attribution, so it cancelled the error instead of exposing it. I had also
weakened the differential test from exact equality against the engine to "equality plus a residual" —
removing the one assertion that would have caught it. Fixed by correcting the attribution and **deleting**
the correction, not extending it; the differential tests are exact again.

### 2.3 The seat/strategy confusion, three times

`winner` in the telemetry is a **seat** (0 or 1). `strategy_p0` is a **strategy id**. Comparing them is
always false or always true. I documented this exact trap as a lessons-learned entry, fixed it in
`analysis/queries.py`, and then reproduced it twice in throwaway probes — concluding first that the two
arms played identical games, and then that the informed arm never won. Both conclusions were false and
both would have been reported as findings. Three violations in one session, each producing a confident
wrong answer.

This argues for a type or a helper, not a comment. It is still only a comment.

### 2.4 Silently stale numbers, three times

- **Disk cache masked a compile failure.** `njit(cache=True)` returned a stale artefact, so a `prange`
  benchmark reported "1.00× speedup" instead of an error, and a compiled-vs-interpreted divergence hid
  behind four real test failures.
- **Schema version drift.** `analysis/schema.sql`'s version macros fell behind `SCHEMA_VERSION`; every
  view silently returned zero rows, so reports showed empty figures with no error anywhere. This
  happened twice, and there is now a test pinning the literals to `durak.versions`.
- **A long run writes its aggregates when it FINISHES.** The 1.14M-game run was started before a
  serialisation fix and wrote `p = 0.0`; re-deriving the aggregates from the same Parquet afterwards
  gave `1.95615e-57`. The dataset was fine, the directory was older than the code that produced it.

### 2.5 Defects found by the pilot, which is what a pilot is for

The 10% pilot caught two things that a full run would have buried:

- `mcnemar_exact` computed `2**n` and raised `OverflowError` past n ≈ 1023. A 20,000-game cell has
  thousands of discordant pairs, so the exact test died on the very dataset it exists to judge. Now
  summed in log space.
- `price_of_information` measured each arm against the *whole field*, which mixes the information effect
  with how strong each arm's other opponents were. It read **+0.012** where the head-to-head reads
  **+0.080**. A composite query also grouped paired games by `game_index` alone, which in an
  eight-strategy tournament merges unrelated games — the bootstrap interval collapsed to `(0.0, 0.0)`,
  a broken join that looked exactly like a vanishing effect.

### 2.6 Rounding deleted the finding

The JSON writer rounded to six **decimal places**, turning a p-value of `2.302e-07` into `0.0` and
printing `0.000e+00` in the report's KPI tile. The number was correct in the query and wrong on the
page — invisible to any test of the arithmetic. Now rounded to six **significant digits**.

## 3. Decisions that shaped the result

- **Information is isolated to one variable.** `SEARCH_UNIFORM` and `SEARCH_INFORMED` are the same
  function; `info_mode` is the only difference. Any behavioural difference is a difference in information.
- **The primary metric is head-to-head, not marginal.** Marginal rates depend on who else played;
  only a fixed-opponent rate is comparable across studies. This was wrong in the code until the pilot
  forced it.
- **Transfers are not progress** — but neither are they free. A take is genuinely worth +1 on the card
  count (the opponent holds one more card to shed). What depth 1 cannot represent is the *deferred*
  cost: the same opponent attacks again immediately with the larger hand. This is the strongest argument
  for depth 2, and it is being measured rather than assumed.
- **Parallelism at the block level, not `prange`.** Numba 0.68 does not lower writes into structured
  arrays inside a parallel loop, and the engine's telemetry is structured arrays. Rather than redesign
  the layout that every dataset depends on, blocks go to worker processes. Result: 2.98× at 4 workers.
- **Scale stated before the run.** The primary tier is 20,000 games/cell on deck 36, chosen so the
  Wilson half-width lands near 0.007. The sensitivity tier is 2,000/cell over the full 228-cell grid,
  because 20,000 everywhere would take 2.3 hours for precision no report reads.

## 4. Open, and deliberately so

These are not done. Each is recorded rather than quietly dropped.

1. **The search policies are still weak.** They lose ~100% of their games to `GREEDY`, and finish with
   ~10.5 cards against its 0.42. The leaf fix reduced the take rate from 49.9% to 21.2% but did not
   resolve it. Depth 2 is being measured. If it does not help, the prime suspect is the weight vector
   `(1.0, 2.0, 0.5, 0.3, 1.5)` — hand-picked and never validated against play — which is a separate
   decision, not a bug to be quietly fixed.
2. **The ceiling is unknown.** A perfect-information minimax player that also knows the deck order, at
   depth 4–5, would give the upper bound. Until it exists, "+0.080" cannot be read as "information is
   worth about eight points" — only as "worth at least that much, to this class of policy".
3. **`k` cannot be varied.** `particle_count` is recorded for reproducibility and refused if it disagrees
   with the compiled constant. Whether `delta` is stable across particle counts is **unmeasured**, and
   no claim may be made in either direction.
4. **The endgame is not degenerate — that was my mischaracterisation.** An empty hand ends the round
   before the defender responds, so 87.9% of games end with the winner attacking an unopposed last
   card. That is the ordinary rule of подкидной дурак, correctly implemented, not a defect. I earlier
   called it "degenerate" and inferred a ceiling on the information effect from it. **The ceiling
   argument was wrong and is refuted by measurement**: zero of 1,140,000 games contained no
   informational decision, so the endgame rule bounds nothing. The true wording is that the last
   attack is uninformative, not that games are decided without information.
5. **`informativeness` is blind to asymmetry.** It is the entropy of the *normalised* marginal, so a
   filter that knows the hand exactly scores the same as one that knows nothing. The phase curve is
   suggestive, not proof.
6. **No diversity diagnostic for the particle filter.** Resolved at schema version 4: level 1 now records
   `distinct_hypotheses_sum` and `search_decisions`, whose ratio is the effective hypotheses per
   decision. Measured at 78.91 against a nominal 100. Note the original diagnosis was wrong in its
   mechanism -- accumulated depletion belongs to `update_particles`, which the search does not call --
   and wrong in its consequence, since a resampling interval over game outcomes cannot be biased by a
   property of the search's internal belief.

## 5. The through-line

Almost nothing above was caught by a failing test. It was caught by measuring a number that looked
wrong and refusing to accept the first explanation. The three most expensive mistakes — the unverified
rules, the masked leaf bug, the repeated seat/strategy confusion — all shared one shape: **a plausible
model of why something was happening, accepted without a measurement that could have refuted it.** The
correction was always to measure the specific thing rather than to reason further, and every time the
measurement was cheap.