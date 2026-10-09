# Research design: hypotheses, threats, metrics

Written while the depth-curve measurement and the oracle build were both in flight, and deliberately
touching no code — a document written mid-measurement that edits an engine file corrupts the run.

This is not a restatement of `PLAN.md`. `PLAN.md` says what will be built; this says what could be wrong
with the answer, what each possible result would mean, and what each conclusion would and would not license.

---

## 1. The question, stated so it can fail

**What is perfect knowledge of the opponent's hand worth, in подкидной дурак?**

The form that matters is the one that can come back negative. "Is information valuable?" invites a yes.
"What does a player gain from knowing the opponent's hand exactly, rather than from a consistent guess?"
admits zero, a gain, and a loss, and the answer we hold is negative.

Three ways this question is answerable, in increasing strength:

| | claim | what would establish it |
|---|---|---|
| weak | "in this tournament, arm A beat arm B 54% of the time" | a paired comparison and a p-value |
| real | "knowing the hand is worth +0.08 to these two policies under these rules and weights" | the number, plus the depth, weights, rule reading and deal composition stated |
| complete | "knowing the hand is worth at most X and at least Y, over this whole family of evaluators and rules" | the ceiling (oracle) and the range across the weight and rule space |

The project is currently at **real**, and only because the depth curve and the oracle are running. It was
briefly at weak, publishing a bare +0.08 that turned out to describe code which no longer existed.

---

## 2. Hypotheses, with what each outcome would mean

### H1 — the information effect is an artifact of a weak evaluation

**Claim.** The +0.08 measured at depth 1 was not the value of information; it was the value of a poor
evaluator being helped by a shortcut. A better evaluator would show less.

**Already supported.** Depth 1 gave +0.0176 on the current code, depth 2 gave −0.0527. The effect fell as
the search improved, which is the direction H1 predicts.

**What would confirm it.** The weight sweep moves the delta substantially across plausible weight vectors,
and the vector chosen for strength is not the vector that maximises the delta. That would show the delta is
a property of the evaluator rather than of the information.

**What would refute it.** The delta stays put across the whole weight space. Then the evaluator is not what
moves it, and something else does.

**Why this matters more than it looks.** If H1 holds, the project's headline number is a measurement of the
evaluator, and the honest report says so in the first paragraph rather than in a footnote.

### H2 — information pays only for a competent player

**Claim.** There is a threshold: below it, knowledge cannot be converted into play and is worth nothing;
above it, knowledge is worth something real and rising.

**Predicted curve.** Non-monotone in depth: high at 1 (shortcut), falling through 2, recovering at 3–4.

**Why it is a real hypothesis and not a hope.** The search loses ~99.5% of its games to a plain heuristic,
so at depth 1–2 it is not a player that could use what it knows. H2 predicts the effect returns only once
the policy is competent — which is also the only configuration in which the oracle is a meaningful bound.

**What would refute it.** A flat-to-falling curve through depth 4. Then depth is not the missing ingredient,
and the remaining candidates are the weights (H1) and the rules.

### H3 — the effect is a function of the rule reading, not of the information

**Claim.** The throw-in rule ambiguity moves the win rate by 7.9 points. The headline is of the same order.
The measurement is therefore substantially a statement about which reading of an ambiguous rule was
implemented.

**Already supported.** 0.19175 → 0.27075, a 7.9-point swing, against a headline of 8.0.

**What would confirm it.** Re-measuring the effect under both readings moves it by a comparable amount.

**What would refute it.** The effect is stable across readings while the absolute rates move.

**Consequence either way.** Both readings must be reported. A number reported under one reading and not
the other is a number about a rule choice.

### H4 — the effect is a function of the deal, chiefly of trump abundance

**Claim.** Information is worth 0.166 with no trumps dealt and 0.043 with five or more — monotone, all
intervals excluding zero, a 3.9× spread.

**What would confirm it.** The curve survives the weight sweep and reappears on a second deck.

**What would refute it.** The curve is an artifact of the one weight vector, or of the 36-card deck.

**What must NOT be claimed.** Any *mechanism*. The plausible reading — no trump hierarchy means each card's
identity matters, abundant trumps make hands homogeneous — is untested. A report may state the curve and may
not state the reason.

### H5 — the rules engine is right

**Claim.** The engine implements the real game.

**Already supported.** Eight of nine externally checkable rules match, checked against Wikipedia, TreeGame,
GAMBITER, stavchess and lenta independently. The trump ace is unbeaten by construction, refill order is
attacker-first, the trump card stays in the draw pile and is drawn last, six-card hands, redeal on six trumps.

**What would refute it.** It already was, once: the take rule was wrong for thirteen stages. It was
internally coherent, consistent with the spec as written, and produced a plausible number. This hypothesis
is not safe because it is central, and the counterexample is on the record.

**Two known divergences, both real.** The trump is revealed from the LAST card where sources say the next
(statistically identical under a uniform shuffle, so no result depends on it). And two rules are not
implemented: a first-defence cap of five cards, and throwing in beyond the defender's hand once the draw
pile is empty — the second acting in the endgame. The first is **withdrawn as a measurement**
(`RUNS.md` R14): a rule-reading coordinate on a range already reported, not a result that could change a
conclusion. The second remains the live divergence, and it is the one that matters: it acts where the
games actually end.

### H6 — the measurement machinery is honest

**Claim.** The numbers mean what they say.

**Guaranteed by test.** `RANDOM` vs `RANDOM` symmetry. Compiled and interpreted execution bit-for-bit
identical. Byte-identical output at 1 and 10 workers. Card conservation across 1.14M games with zero
anomalies. Exact differential agreement between the leaf and the engine at 1, 2, 3 and 4 plies.

**Six ways it has been untrue in practice**, all recorded with the mechanism:
1. `game_index` is cell-local, so every level-2 aggregate was an average over 57 unrelated games.
2. Seeding with a python-global seed ignored rng draws, so "seeded" runs repeated the same games.
3. `pairing_index` was written globally rather than per pairing, collapsing distinct cells together.
4. numba's disk cache does not invalidate on a changed global, which flattened a depth sweep.
5. A cold Numba cache races across worker processes and kills pools.
6. Paired games shared seeds, so a "paired" bootstrap measured the wrong quantity.

**This is the most productive thing the project has produced.** Each of the six was found by a number that
looked wrong, and each is now a test.

---


### H7 — the effect is about card flow in general, not about the opponent's hand

**Claim.** Every model currently remembers every card ever played: the particle filter excludes the whole
discard pile from the opponent's candidate set. That is *perfect memory*, and it makes the information
question binary — know the hand exactly, or know nothing. Real players remember the last few cards and
reconstruct the rest. So the measured effect may not be about the opponent at all. It may be about knowing
which cards are gone, which is a different quantity with a different policy consequence.

**Why this is the sharpest remaining question.** The particle filter does two jobs at once: it estimates the
opponent's hand, and it uses the discard pile to eliminate candidates. Separating them is the difference
between "perfect knowledge of the opponent" and "perfect bookkeeping". A bounded-memory arm keeps the first
and loses the second, so the two decompose.

**The design.** A memory horizon `M`, the number of plies back a model remembers. Cards discarded more
than `M` plies ago return to the candidate set, and the filter must place them — in the deck or in the
opponent's hand — which is exactly the uncertainty being modelled. `M = ∞` reproduces today's numbers
exactly and is the regression guard for the change; `M = 0` remembers nothing about played cards.

**Predictions, and they differ.**

| outcome | reading |
|---|---|
| a bounded arm matches the perfect-memory arm | the effect is bookkeeping. The headline is about tracking cards, not about the opponent |
| the effect decays smoothly with `M` | information about the opponent is worth a few plies of history and no more |
| a bounded arm BEATS the perfect-memory arm | a sharper prior beats a complete one, and the project's whole framing is wrong |

**The third row is the one worth being ready for.** It is not a fantasy: the search already loses to a
100-particle prior arm despite the oracle arm holding the exact hand, so in this engine a sharper belief has
beaten a perfect one before.

**Cost.** One parameter through the filter, plus a test that `M = ∞` reproduces the current numbers exactly.
That regression test is the whole safety margin — the parameter changes what every belief in the project is,
and a silent off-by-one would look like a scientific result.

## 3. Threats to validity, ranked by how much damage they do

| # | threat | damage | status |
|---|---|---|---|
| V1 | the search arms are far too weak to be "players" | total — every delta describes them, not information | open; the weight sweep is the test |
| V2 | the evaluator is what moves the delta, not the information | total — H1 | open; the weight sweep is the test |
| V3 | the rule reading moves the result as much as the effect | large — the number is about a rule choice | partially measured, 7.9 points |
| V4 | depth is an uncontrolled variable | large — the delta changed sign with it | being measured now |
| V5 | the effect is not uniform across deals | moderate — a single average hides 4× | measured on trumps, mechanism unknown |
| V6 | a paired bootstrap on unpaired games | moderate — false precision in the interval | fixed, pinned |
| V7 | `k` (particle count) cannot be varied | unknown — stability unmeasured | **no claim available either way** |
| V8 | level-2 data cannot be joined to games | moderate — no phase analysis possible | open; needs `pairing_index` |
| V9 | the dataset predates the current code | total for any published number | closed by withdrawal; a rebuild is required |
| V10 | endgame collapse: 87.9% of games end on an unopposed last card | moderate — compresses the decisive phase | measured, stated on the report |

**V7 deserves emphasis.** Particle count is fixed by a compiled constant and cannot be varied at all. So
whether the delta is stable across `k` is **unmeasured**, and the project may make no claim in either
direction. This is the only significant threat with no plan attached, and the only honest resolution is to
either make `k` configurable or to state the omission in every report.

---

## 4. Metrics: what exists, what is missing, what to add

### Exists and is trustworthy
Level-1 per-game outcome and dealing statistics; strategy strength against `RANDOM` (the only rate
comparable across studies); pairwise matchups with both seat orders folded in; the paired information delta
with a bootstrap interval and an exact McNemar test; take, throw-in and trump profiles per strategy; per-phase
win rates if and only if the join is fixed; rule-variant comparisons.

### Exists and is unusable
Anything requiring a level-2 or level-3 join: per-phase breakdowns, per-round analysis, move-level
legality or latency. `game_index` is cell-local and levels 2 and 3 carry no `pairing_index`, so the join
fans out by 57 and nothing can be attributed to a strategy.

### Missing, and why each matters
| metric | why it is needed |
|---|---|
| strength against a fixed panel, not just `RANDOM` | the only strength rate that survives a change of field |
| take rate per strategy, by depth | the diagnosed axis: `GREEDY` takes 1.3%, the search 12.5–21% |
| the depth curve of the delta | decides H1 and H2, and whether any number can be published |
| the ceiling | without it, every number is a lower bound |
| delta across the weight space | decides whether +0.08 is a number or a range |
| delta across the rule readings | decides H3 |
| delta by trump composition | H4 |
| distinct-hypothesis count for the particle filter | DONE at schema 4: `distinct_hypotheses_sum / search_decisions`, measured at 78.91 against a nominal 100, so the budget overstates real resolution by about a fifth. It bounds the search's resolution; it does not touch any quoted interval |
| time-to-empty-hand per strategy | the direct measure of what the game rewards, and it is cheap |

---

## 5. Predicting the outcomes

Stated before the runs finish, so they can be compared against what comes out. A prediction recorded after
the result is a rationalisation, not a prediction.

**Likely, and why:** the depth curve is flat or slightly falling through depth 4, and the ceiling is well
above the current delta. Both because the search is far too weak for extra depth to pay off (V1), and because
the evaluator is what moves the delta (V2). If that is right, the project's honest conclusion is a *negative*
result: perfect knowledge of the hand is worth approximately nothing to a policy that reasons even two plies
ahead, and the +0.08 that started this was a measure of a weak evaluator being helped by a shortcut.

**Possible, and it would be the most interesting outcome:** the curve dips at depth 2 and recovers at 3–4
(H2). Then the effect is real and gated on competence, the ceiling matters, and the project's headline becomes
a floor rather than an artifact.

**Possible and would be a hard stop:** the weight sweep finds a vector where the search beats `GREEDY`
comfortably. Every earlier number was then measured with a broken instrument, and the whole measurement
programme has to be re-run from the pilot.

**What must not be published under any branch:** the withdrawn +0.0799; any statement that the price of
information is a single number; any statement about particle count; any causal explanation of the trump curve;
any claim that the rules engine is verified.

---

## 6. Decisions taken, with the alternative rejected

Recorded because the rejected alternative is usually the more instructive half.

| decision | rejected alternative | consequence if the choice was wrong |
|---|---|---|
| head-to-head as the primary metric | field-wide marginal rates | the field-wide version read +0.012 against a head-to-head +0.080; it mixes the information effect with how strong the other opponents were |
| both arms differ only in the particle source | giving one arm a stronger reply model | the comparison measures the reply model instead of the information |
| exact differential tests against the engine | approximate agreement | the leaf's take-attribution bug survived a weakened assertion and reached a committed finding |
| process pool over blocks | `prange` over the engine loop | numba cannot lower structured-array writes in `prange`; the pool gave 2.98× |
| integer-only mid/section rollbacks | float-based | — |
| a missing measurement reported as missing | emitting a zero | a zero on a chart reads as "measured, and it is zero" |
| process-level parallelism | threads | numba releases the GIL, but the structured telemetry does not thread safely |
| one search depth as a parameter | one code path per depth | a path per depth makes runs incomparable and does not scale |
| an iterative explicit-stack walk | recursion | recursion compiles and then kills every spawned worker |

---

## 7. The honest summary

The instrument is in good shape and has been hardened by six caught defects. **The subject is not.** The
policies under study lose ~99.5% of their games to a plain heuristic, the evaluator was never validated
against play, and the two quantities most likely to explain the headline — the weights and the depth — are
exactly the two now being measured.

If both measurements come back showing the delta is small or unstable across them, the correct outcome is a
published negative result about this class of policies, and the value of the project is that it says so
without a single unearned number in it.