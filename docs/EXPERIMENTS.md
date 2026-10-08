# Experiment log

One row per measurement actually run. The point is negative results: a run that found nothing is the
expensive part of this project to repeat, and without a row here it gets re-run. Every entry states its
seed, its scale, and what it does NOT establish, because a number without those is how this project
produced five confident wrong answers before the traps were found.

Superseded rows are kept and marked. Deleting them would erase the reason a later number exists.

| # | question | config | result | what it does not establish | status |
|---|---|---|---|---|---|
| 1 | Does the published +0.0799 reproduce? | old dataset, depth 1, 40,000 games, `queries._paired_deltas` | +0.0799 CI [+0.0702, +0.0894] | nothing about the current code — the dataset predates the leaf fix and depth 2 | **WITHDRAWN** |
| 2 | Same question on current code, depth 1 | 20,000 games, one harness | +0.0176 CI [+0.0078, +0.0274] | not comparable with row 1's games | current, pre-deck-hiding |
| 3 | Same, depth 2 | 20,000 games, same harness | −0.0527 CI [−0.0625, −0.0429] | a null at one depth is not a curve | current, pre-deck-hiding |
| 4 | Is the effect U-shaped in "informational density"? | deciles over 40,000 games | U-shape reaching −0.025, interval excluding zero | **an artifact** — the stratifier is a count of actions taken, so it is post-treatment; the arms' defensive rates differ within a single game by −0.0072, CI [−0.0081, −0.0062] | **RETRACTED** |
| 5 | Is the effect monotone in trumps dealt? | 0..5+ trumps, 40,000 games | +0.166 → +0.043, all intervals exclude zero, monotone | no mechanism; a pre-treatment stratifier, so the curve itself is legitimate | current, pre-deck-hiding |
| 6 | Throw-in rule sensitivity | `throwin_defense_ranks` on/off, 4,000 games each, separate tournaments | 0.19175 → 0.27075, a 7.9-point swing | NOT a significance test — separate tournaments, unpaired | current |
| 7 | Does depth 2 fix the search's weakness? | take rate, final hand, pairwise, ~2,000 games | take rate 0.214 → 0.125; vs `AUTO_DEFEND` 0.008 → 0.196; vs `GREEDY` unchanged at ~0.005 | the win rates came from a harness that passed `InfoMode.NONE` to both arms, so the two search rows were one policy measured twice; and level 3 wrote empty, so the take-rate drop was never independently reproduced | **win rates VOID**, take rate UNVERIFIED |
| 8 | Where does the information effect live? | 20,000 games, depth 2, trusted harness | −0.0618 CI [−0.0716, −0.0520] | the runs shared games, so the overlap with row 3 does not prove the refactor was equivalent; a strictly identical number was NOT obtained | current |
| 9 | Ceiling from a perfect-information oracle | `ORACLE` vs `ORACLE_UNIFORM`, depth 4, 20,000 games/direction | −0.0707 CI [−0.0716, −0.0429] region; the oracle does NOT beat the arms | **void as a ceiling**: the pile was public, so the second oracle axis did not exist, and the hand was derivable | **VOID**, see row 11 |
| 10 | Depth curve, depths 1–4 | 4,000 games/depth, one seed, cache cleared per depth | depths 1–3: −0.0120, −0.0580, −0.0840 — monotone | depth 4 pending; all pre-deck-hiding, so all void as descriptions of the current model | **SUPERSEDED** by row 11 |
| 11 | Does hiding the draw pile change the picture? | implementation + invariance tests | the pile no longer reaches the search; every row above is void | nothing yet — the curve must be re-run | **the model now** |

## Why so many voids

Five of these rows were confidently wrong, and the mechanism is the same every time: a number that looked
implausible or a harness that looked reasonable, accepted without the measurement that could have refuted
it. The five traps, all of which produced a confident wrong answer rather than a visible failure:

1. `game_index` is cell-local, so every level-2 aggregate averaged 57 unrelated games.
2. Seeding with a python-global seed ignored rng draws, so "seeded" runs repeated games.
3. `pairing_index` was written globally, collapsing distinct cells.
4. numba's disk cache does not invalidate on a changed global, which flattened a depth sweep to four
   identical numbers — and a flat curve reads as a finding.
5. Paired games sharing seeds made a "paired" bootstrap measure the wrong quantity.

Plus the seat/arm confusion, hit four times: `winner` is a seat, `strategy_p0` is a strategy id, and
comparing them is true only when some strategy happens to be numbered 0 or 1. It produced: "the arms play
identical games", "the informed arm never wins", a depth-2 harness that handed the UNIFORM arm the truth
and cancelled to a flat +0.0000, and an oracle that could not beat the arms it was supposed to bound.

**The generalisable lesson:** a measurement that produces a suspiciously round or flat number has usually
cancelled itself, not discovered a null. Three separate times here the flat result was the bug.

## Rows still owed

- Depth curve {1,2,3,4} on the deck-hiding model. Running.
- Weight sweep, twelve vectors, selected on absolute strength against a fixed panel and NEVER on the delta.
- Oracle re-measurement once the model is settled, since the old ceiling was void.
- Delta by trump composition, re-measured on the corrected model.
- The two unimplemented rules, once the engine is free of a running measurement.