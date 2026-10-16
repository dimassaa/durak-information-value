"""R3 -- twelve evaluation-weight vectors, one process each.

Selection is on ABSOLUTE STRENGTH against a fixed panel. The informed-minus-uniform delta is recorded
at every vector and printed across all twelve, and is never the criterion: tuning weights until the
headline grows fits the answer to the question. If both arms must then share one frozen vector, the
choice of vector has to be made without looking at the delta, or the headline is a function of the
search over weights rather than a measurement.

Each vector gets its OWN process and its OWN numba cache directory, because `@njit` freezes module
globals at compile time: two vectors in one process would both play whichever compiled first. That is
also why the sweep refuses to run without a private `NUMBA_CACHE_DIR` -- sharing a cache with another
experiment makes one of them silently play the other's weights.

Three gates run BEFORE the 600,000 games, because each one produces a plausible-looking table of wrong
numbers rather than an error:

  1. **Every vector must change the decision somewhere.** A term set to zero can leave all five weights
     proportional to the incumbent, in which case the search returns the same action on every position
     and the row is a duplicate wearing a new label. Twelve identical numbers mean a broken sweep, not a
     flat result.
  2. **The incumbent must itself change decisions**, or gate 1 is passing for the wrong reason.
  3. **The two arms must not produce identical games** under one seed. They differ in one line, and if
     that line stops mattering the delta cancels to exactly 0.0000 and reads as "information is worth
     nothing" -- the failure this project has already mistaken for a result once.

Written to `results/weights/`, never to /tmp.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CACHE_DIR = os.environ.get("NUMBA_CACHE_DIR", "")
if not CACHE_DIR:
    sys.exit(
        "refusing to run without a private NUMBA_CACHE_DIR.\n"
        "  Numba's cache does NOT invalidate when a module global changes, so a shared cache means\n"
        "  every vector after the first silently plays the first one's weights, and the sweep\n"
        "  returns the same number twelve times under twelve names.\n"
        "  Use:  NUMBA_CACHE_DIR=.numba_cache/<variant> .venv/bin/python _weight_sweep.py"
    )

from analysis.stats import wilson_interval
from durak import rules
from durak.engine import play_block
from durak.evaluation import EVAL_WEIGHTS
from durak.runner import _BlockJob, _run_block
from durak.search import expectimax
from durak.strategies import (
    AUTO_DEFEND,
    CHEATER_DOC,
    CONSERVATIVE,
    GREEDY,
    INFO_PARTICLES,
    INFO_TRUTH,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
)
from durak.strategies.search_policy import _truth_particles

DECK = 36
NONE = 0
BLOCK = int(os.environ.get("SWEEP_BLOCK", "250"))
WORKERS = int(os.environ.get("SWEEP_WORKERS", str(max(1, (os.cpu_count() or 2) - 4))))
GAMES = int(os.environ.get("SWEEP_GAMES", "10000"))
DELTA_GAMES = int(os.environ.get("SWEEP_DELTA_GAMES", "4000"))
PROBE_POSITIONS = int(os.environ.get("SWEEP_PROBE", "30"))
SEED = int(os.environ.get("SWEEP_SEED", "1"))

#: Order matches `durak.evaluation.evaluate`: card count, trumps, high cards, role, deck exhausted.
#: The incumbent is first and every other vector is a ray from it, so the sweep shows which direction
#: each term pushes rather than twelve unrelated points.
VARIANTS = [
    ("incumbent", EVAL_WEIGHTS),
    ("cards_3x", (3.0, 2.0, 0.5, 0.3, 1.5)),
    ("cards_5x", (5.0, 2.0, 0.5, 0.3, 1.5)),
    ("trumps_half", (1.0, 1.0, 0.5, 0.3, 1.5)),
    ("trumps_off", (1.0, 0.0, 0.5, 0.3, 1.5)),
    ("trumps_3x", (1.0, 6.0, 0.5, 0.3, 1.5)),
    ("high_4x", (1.0, 2.0, 2.0, 0.3, 1.5)),
    ("role_off", (1.0, 2.0, 0.5, 0.0, 1.5)),
    ("deck_3x", (1.0, 2.0, 0.5, 0.3, 4.5)),
    ("cards_up_trumps_down", (3.0, 1.0, 0.5, 0.3, 1.5)),
    ("cards_down_trumps_up", (1.0, 4.0, 0.5, 0.3, 1.5)),
    ("cards_only", (1.0, 0.0, 0.0, 0.0, 0.0)),
]

#: Fixed yardstick. Its members do not consult the weights at all, so a vector's strength against them
#: measures the vector rather than anything it was tuned against.
PANEL = (RANDOM, GREEDY, AUTO_DEFEND, CONSERVATIVE, CHEATER_DOC)
PANEL_NAMES = {RANDOM: "RANDOM", GREEDY: "GREEDY", AUTO_DEFEND: "AUTO_DEFEND",
               CONSERVATIVE: "CONSERVATIVE", CHEATER_DOC: "CHEATER_DOC"}

ARM_MODES = {SEARCH_INFORMED: INFO_TRUTH, SEARCH_UNIFORM: INFO_PARTICLES}


def probe_positions(count):
    """Opening positions of `count` deals, with the truth particles and a shared deck hypothesis.

    Truth particles on purpose. This is a differential test of the EVALUATOR, not a play measurement:
    it asks whether a weight vector reaches the chosen action at all, and using the informed belief
    makes the answer sharper rather than noisier.
    """
    out = []
    seed = 0
    while len(out) < count:
        state = rules.deal_state(DECK, np.uint64(seed), np.uint64(seed + 7),
                                 np.uint64(seed + 13), rules.DEFAULT_SETTINGS, False, -1)
        seed += 1
        if int(state.winner) != rules.EMPTY:
            continue
        player = int(state.attacker)
        particles = _truth_particles(state, 1 - player)
        deck = np.repeat(state.deck[:state.deck_len][::-1][None, :], particles.shape[0], axis=0)
        out.append((state, player, particles, deck))
    return out


def differing(positions, variant, incumbent, depth) -> int:
    """How many of `positions` the two weight vectors resolve to different actions."""
    diff = 0
    for state, player, particles, deck in positions:
        if int(expectimax(state, player, particles, deck, variant, depth)) != \
           int(expectimax(state, player, particles, deck, incumbent, depth)):
            diff += 1
    return diff


def _jobs(matchup_index, arm, opponent, game_count):
    """Both seat orders, as the runner's own picklable block jobs.

    `seat_of_arm` travels with the job rather than being recomputed at the tally, so the resolution of
    the outcome cannot drift between the two directions.
    """
    out = []
    for seat_of_arm, (p0, p1) in ((0, (arm, opponent)), (1, (opponent, arm))):
        for start in range(0, game_count, BLOCK):
            stop = min(start + BLOCK, game_count)
            out.append((
                _BlockJob(
                    pairing_index=1000 + matchup_index, deck_size=DECK,
                    strategy_p0=p0, strategy_p1=p1,
                    info_p0=int(ARM_MODES.get(p0, NONE)), info_p1=int(ARM_MODES.get(p1, NONE)),
                    regime="RULE_DERIVED", seed=SEED, game_start=start, game_count=stop - start,
                    sample_modulus=20,
                ),
                seat_of_arm,
            ))
    return out


def head_to_head(matchup_index, arm, opponent, game_count):
    """`arm`'s win rate against `opponent` over both seat orders. `(wins, decided, capped)`."""
    jobs = _jobs(matchup_index, arm, opponent, game_count)
    if WORKERS == 1:
        results = [(seat, _run_block(job, rules.DEFAULT_SETTINGS, False, False)) for job, seat in jobs]
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [(seat, pool.submit(_run_block, job, rules.DEFAULT_SETTINGS, False, False))
                       for job, seat in jobs]
            results = [(seat, f.result()) for seat, f in futures]

    wins = decided = capped = 0
    for seat_of_arm, (outcomes, _rounds, _moves) in results:
        # `winner` is a SEAT, resolved through the seat the ARM occupied in this direction. A mirrored
        # run without this cancels to exactly zero and looks like a null result.
        wins += int(np.count_nonzero(outcomes["winner"] == seat_of_arm))
        decided += int(np.count_nonzero(outcomes["winner"] >= 0))
        capped += int(np.count_nonzero(outcomes["winner"] < 0))
    return wins, decided, capped


def gates(positions, depth):
    """Refuse to spend 600,000 games on a sweep that cannot tell its vectors apart.

    Two different failures, deliberately not treated the same way.

    A vector IDENTICAL to the incumbent is a bug in the vector list: its row would be a duplicate wearing
    a new label, and there is nothing to learn from paying for it.

    A vector that differs on NO position is not that. It means the term has not reached the chosen action
    at this depth on these positions, which is a finding about the evaluator -- the deck-exhausted term is
    the obvious candidate, since it only bites when the draw pile is nearly out. Refusing to run it would
    suppress exactly the result the sweep exists to produce, so it is reported loudly and measured anyway.
    The incumbent is compared against a FLAT vector instead of against itself, so the probe's own
    sensitivity is measured rather than assumed.
    """
    flat = (7.0, 7.0, 7.0, 7.0, 7.0)
    incumbent = tuple(float(v) for v in EVAL_WEIGHTS)
    incumbent_diff = differing(positions, incumbent, flat, depth)
    print(f"  gate probe sensitivity: the incumbent differs from a flat vector in "
          f"{incumbent_diff:>3} of {len(positions)} positions", flush=True)
    if incumbent_diff == 0:
        sys.exit(f"the incumbent changes no decision against a flat vector on {len(positions)} positions, "
                 f"so the probe cannot detect any vector. The gate is broken, not the weights.")

    # Gate the vector THIS PROCESS IS PLAYING, taken from the environment, not the R3 list.
    #
    # It used to iterate `VARIANTS`, which is the sweep's own twelve-point list. On the R3 sweep that is
    # the same set, so the gate did its job. On the surface sweep it is a DIFFERENT set, and every one of
    # the nine surface runs printed its warning about `deck_3x` -- a vector from R3, not the one being
    # measured. So the only check that could catch a no-op vector was examining a different vector, and
    # all nine surface runs passed a gate that never looked at them.
    measured = tuple(float(v) for v in os.environ["DURAK_EVAL_WEIGHTS"].split(","))
    label = os.environ.get("SWEEP_LABEL", "measured vector")
    diff = differing(positions, measured, incumbent, depth)
    print(f"  gate measured vector {label:<24} differs from the incumbent in {diff:>3} of "
          f"{len(positions)} positions", flush=True)
    if diff == 0 and max(abs(a - b) for a, b in zip(measured, incumbent)) > 1e-12:
        sys.exit(f"the vector being measured ({label} = {measured}) changes no decision against the "
                 f"incumbent on {len(positions)} opening positions, so its row would be a duplicate of "
                 f"the incumbent's wearing a new label. Fix the vector.")


def arms_differ() -> None:
    """Refuse if the two arms play identical games under one seed.

    With the deck hidden they must differ; if they do not, the belief line has stopped mattering and the
    delta would cancel to 0.0000, which reads as a finding.
    """
    played = {}
    for arm, mode in ((SEARCH_INFORMED, INFO_TRUTH), (SEARCH_UNIFORM, INFO_PARTICLES)):
        outcomes, _, _ = play_block(DECK, arm, GREEDY, mode, NONE, SEED, 0, 0, 200,
                                    rules.DEFAULT_SETTINGS, False, False)
        played[arm] = outcomes
    if np.array_equal(played[SEARCH_INFORMED]["winner"], played[SEARCH_UNIFORM]["winner"]) \
            and np.array_equal(played[SEARCH_INFORMED]["actions"], played[SEARCH_UNIFORM]["actions"]):
        sys.exit("the two arms produced byte-identical games under one seed, so the delta would cancel "
                 "to 0.0000 and read as 'perfect information is worth nothing'.")
    print("  gate arms differ under one seed: ok", flush=True)


def main():
    from durak.strategies.search_policy import PARTICLE_COUNT, SEARCH_DEPTH

    label = os.environ.get("SWEEP_LABEL", "unknown")
    weights = tuple(float(v) for v in os.environ["DURAK_EVAL_WEIGHTS"].split(","))
    depth = SEARCH_DEPTH
    started = time.perf_counter()

    print(f"=== weight sweep: {label} ===", flush=True)
    print(f"  weights (cards,trumps,high,role,deck) = {weights}", flush=True)
    print(f"  SEARCH_DEPTH={depth}  PARTICLE_COUNT={PARTICLE_COUNT}  workers={WORKERS}", flush=True)
    print(f"  panel strength at {GAMES} games/opponent, delta at {DELTA_GAMES}", flush=True)

    incumbent = tuple(float(v) for v in EVAL_WEIGHTS)
    if max(abs(a - b) for a, b in zip(weights, incumbent)) < 1e-12 and label != "incumbent":
        sys.exit(f"the weights in the environment are the incumbent but the label is {label!r}; the "
                 f"process is not playing what its own name says.")

    positions = probe_positions(PROBE_POSITIONS)
    gates(positions, depth)
    arms_differ()

    strengths = {}
    for i, opponent in enumerate(PANEL):
        wins, decided, capped = head_to_head(i, SEARCH_UNIFORM, opponent, GAMES // 2)
        rate = wins / decided if decided else float("nan")
        lo, hi = wilson_interval(wins, decided) if decided else (float("nan"), float("nan"))
        strengths[PANEL_NAMES[opponent]] = rate
        print(f"  strength vs {PANEL_NAMES[opponent]:<14} {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"({wins}/{decided} decided, {capped} capped)   [{time.perf_counter() - started:6.0f}s]",
              flush=True)

    # Recorded, never the selection criterion. The docs publish this across all twelve vectors, so the
    # headline is a range over the weight space rather than a point.
    wins_i, decided_i, _ = head_to_head(90, SEARCH_INFORMED, SEARCH_UNIFORM, DELTA_GAMES // 2)
    wins_u, decided_u, _ = head_to_head(91, SEARCH_UNIFORM, SEARCH_INFORMED, DELTA_GAMES // 2)
    delta = (wins_i / decided_i if decided_i else float("nan")) - \
            (wins_u / decided_u if decided_u else float("nan"))
    print(f"  RECORDED delta (not the criterion) {delta:+.4f}  at {DELTA_GAMES} games", flush=True)
    print(f"  RESULT label={label} weights={','.join(str(v) for v in weights)} "
          f"depth={depth} delta={delta:+.4f} "
          f"panel_mean={np.mean([strengths[n] for n in strengths]):.4f} "
          f"wall={time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()