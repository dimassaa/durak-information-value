"""The ceiling measurement: `ORACLE` against the fixed panel, and against both search arms.

What this decides
-----------------
The project's headline -- the value of perfect knowledge of the opponent's hand -- is currently a
lower bound, because it is measured between two arms that lose ~99.5% of their games to `GREEDY`.
`ORACLE` is the intended upper bound: the same search, the opponent's real hand, four plies deep. If
the achievable ceiling is near the measured delta then the headline is most of what is attainable;
if it is far above, the headline is a fraction. This number decides how every other number in the
project may be described, so it is reported whether or not it flatters the result.

Design commitments, each one bought with a wrong finding earlier in this project
-------------------------------------------------------------------------------
* **`winner` is a SEAT, never a strategy id.** Every rate is resolved through the seat the arm
  actually occupied, per direction, exactly as `analysis.queries._head_to_head` does it. Comparing
  the two directly makes the directions cancel to a flat 0.0000, which reads as a null result.
* **Information modes follow the ARM, not the seat.** `SEARCH_INFORMED` runs with `INFO_TRUTH`,
  `SEARCH_UNIFORM` with `INFO_PARTICLES`, in both directions. Giving the mirror game the same mode
  pair hands the opponent the truth and cancels the comparison against itself.
* **One seed per matchup, shared by both directions**, so the two directions play the same deals and
  a difference between them is a seat effect rather than a dealing effect.
* **Blocks, not games, are the unit of parallelism.** A game's seed derives from
  `(block_seed, pairing_index, game_index)` and nothing else, so the result cannot depend on how
  many workers ran it. Checked rather than assumed, on a probe, before the long runs.
* **Depth is frozen at compile time** and Numba's on-disk cache does NOT invalidate when a module
  global changes, so a depth run that reuses another run's cache silently plays the first depth
  compiled. This script therefore *requires* a private `NUMBA_CACHE_DIR` and refuses to run without
  one, and it re-reads `SEARCH_DEPTH` / `ORACLE_DEPTH` after import and prints them beside every
  rate.
* **The depth actually reaches the policy**, checked in-process against the shared search at both
  depths. A cache collision that froze the oracle at the arms' depth would otherwise print a flat
  0.0000 against `SEARCH_INFORMED` and read as "perfect information is worth nothing".

Run as a real file, never piped: `spawn` re-imports `__main__`, and a heredoc gives
`BrokenProcessPool`.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Read and enforced before any durak import: a private numba cache is the only thing that keeps this
# run's compiled depths apart from another experiment's.
CACHE_DIR = os.environ.get("NUMBA_CACHE_DIR", "")
if not CACHE_DIR:
    sys.exit(
        "refusing to run without a private NUMBA_CACHE_DIR.\n"
        "  Numba's on-disk cache does NOT invalidate when a module global changes, so sharing\n"
        "  `durak/__pycache__` with another experiment means one of the two silently plays the\n"
        "  other's depth -- and a depth sweep then comes out perfectly flat, which looks like a\n"
        "  finding and is not one.\n"
        "  Use:  NUMBA_CACHE_DIR=/tmp/<something> .venv/bin/python _oracle_ceiling.py"
    )

from analysis.stats import wilson_interval
from durak.engine import play_block
from durak.rules import DEFAULT_SETTINGS, EMPTY, acting_player, legal_actions
from durak.runner import _BlockJob, _run_block
from durak.strategies import (
    AUTO_DEFEND,
    CONSERVATIVE,
    GREEDY,
    INFO_PARTICLES,
    INFO_TRUTH,
    ORACLE,
    ORACLE_UNIFORM,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
)

DECK = 36
NONE = 0

#: Games per direction. 20,000 is the project's usual scale. The block size is only the unit of
#: parallelism and does not change which games are played.
GAMES = int(os.environ.get("ORACLE_GAMES", "20000"))
BLOCK = int(os.environ.get("ORACLE_BLOCK", "250"))
WORKERS = int(os.environ.get("ORACLE_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))

#: The information mode each ARM runs with, resolved before the seat assignment is made. The oracle
#: ignores its mode by construction (`tests/test_oracle.py::test_the_oracle_ignores_the_information_
#: mode`), and gets `NONE` here because that is what `runner.info_modes_for` hands it.
INFO = {
    ORACLE: NONE,
    SEARCH_INFORMED: INFO_TRUTH,
    SEARCH_UNIFORM: INFO_PARTICLES,
}


def _jobs(matchup_index, arm, opponent, game_count):
    """Both seat orders of one matchup, as the runner's own picklable block jobs.

    `seat_of_arm` travels with the job rather than being recomputed at the tally, so the resolution
    of the outcome cannot drift between the two directions.
    """
    out = []
    for seat_of_arm, (p0, p1) in ((0, (arm, opponent)), (1, (opponent, arm))):
        for start in range(0, game_count, BLOCK):
            stop = min(start + BLOCK, game_count)
            out.append((
                _BlockJob(
                    pairing_index=1000 + matchup_index, deck_size=DECK,
                    strategy_p0=p0, strategy_p1=p1,
                    info_p0=int(INFO.get(p0, NONE)), info_p1=int(INFO.get(p1, NONE)),
                    regime="RULE_DERIVED", seed=1, game_start=start, game_count=stop - start,
                    sample_modulus=20,
                ),
                seat_of_arm,
            ))
    return out


def head_to_head(matchup_index, arm, opponent, game_count, workers):
    """`arm`'s win rate against `opponent` over both seat orders. `(wins, decided, capped)`."""
    jobs = _jobs(matchup_index, arm, opponent, game_count)
    if workers == 1:
        results = [(seat, _run_block(job, DEFAULT_SETTINGS, False, False)) for job, seat in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [(seat, pool.submit(_run_block, job, DEFAULT_SETTINGS, False, False))
                       for job, seat in jobs]
            results = [(seat, f.result()) for seat, f in futures]

    wins = decided = capped = 0
    for seat_of_arm, (outcomes, _rounds, _moves) in results:
        # `winner` is a SEAT. Resolved through the seat the ARM occupied in this direction, which is
        # the entire point: a mirrored run without it cancels to exactly zero and looks like a null.
        wins += int(np.count_nonzero(outcomes["winner"] == seat_of_arm))
        # A game that hit the round cap has no winner to attribute. Excluded from the denominator
        # rather than counted as a loss, which would invent an outcome the engine never decided.
        decided += int(np.count_nonzero(outcomes["winner"] >= 0))
        capped += int(np.count_nonzero(outcomes["winner"] < 0))
    return wins, decided, capped


def _report(label, wins, decided, capped, elapsed):
    rate = wins / decided if decided else float("nan")
    lo, hi = wilson_interval(wins, decided) if decided else (float("nan"), float("nan"))
    print(f"  {label:<34} {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"DELTA {2 * rate - 1:+.4f}  ({wins}/{decided} decided, {capped} capped)   "
          f"[{elapsed:6.0f}s]", flush=True)
    return rate


def depth_reach_probe(game_count=400):
    """Two sharp checks on whether the oracle is running at its own depth, here and now.

    **Structural.** The oracle must answer exactly as the shared search does at `ORACLE_DEPTH`, on
    every position. If it does not, the depth is not reaching the policy.

    **Discriminating.** `ORACLE` and `SEARCH_INFORMED` are two policies, so playing them against the
    same opponent with the same seeds must produce different games. Identical bytes would mean the
    oracle *is* the informed arm -- the numba cache collision this script exists to catch -- and the
    comparison against it would then cancel to a flat 0.0000 and read as "perfect information is
    worth nothing".

    The obvious version of the second check -- count positions where the two arms choose different
    actions -- is not sharp enough and was tried first: over ten real positions the depth-2 and
    depth-4 searches happen to agree on every one of them, so the check reported a collision where
    there was none. Byte-identical games under an identical seed have no such false positive.
    """
    from durak.evaluation import EVAL_WEIGHTS
    from durak.search import expectimax
    from durak.strategies import get_policy
    from durak.strategies.oracle import ORACLE_DEPTH
    from durak.strategies.search_policy import _truth_particles
    from tests.support import random_position

    oracle = get_policy(ORACLE)
    compared = matched = 0
    for seed in range(10):
        state = random_position(seed=seed)
        player = acting_player(state)
        if state.winner != EMPTY or legal_actions(state, player).size == 0:
            continue
        expected = int(expectimax(state, player, _truth_particles(state, 1 - player),
                                  EVAL_WEIGHTS, ORACLE_DEPTH))
        chosen = int(oracle(state, player, ORACLE, NONE, np.uint64(seed)))
        matched += int(chosen == expected)
        compared += 1

    def _games(strategy, info):
        outcomes, _r, _m = play_block(DECK, strategy, AUTO_DEFEND, info, NONE, 1, 7, 0, game_count,
                                     DEFAULT_SETTINGS, False, False)
        return np.ascontiguousarray(outcomes).tobytes()

    oracle_games = _games(ORACLE, NONE)
    informed_games = _games(SEARCH_INFORMED, INFO_TRUTH)

    return {
        "positions": compared,
        "positions_matching_own_depth": matched,
        "oracle_depth": ORACLE_DEPTH,
        "games_differ": oracle_games != informed_games,
    }


def main():
    from durak.strategies.oracle import ORACLE_DEPTH
    from durak.strategies.search_policy import SEARCH_DEPTH

    started = time.perf_counter()
    probe = depth_reach_probe()

    print(f"=== oracle ceiling: deck {DECK}, {GAMES} games per direction, {WORKERS} workers ===",
          flush=True)
    print(f"  SEARCH_DEPTH={SEARCH_DEPTH}  ORACLE_DEPTH={ORACLE_DEPTH}  cache={CACHE_DIR}",
          flush=True)
    print("  depth reach: matches the shared search at its own depth on "
          f"{probe['positions_matching_own_depth']}/{probe['positions']} positions; its games differ "
          f"from SEARCH_INFORMED's under the same seeds: {probe['games_differ']}", flush=True)
    if probe["positions_matching_own_depth"] != probe["positions"]:
        sys.exit("the oracle does not answer as the shared search at its OWN depth does, so the "
                 "depth is not reaching the policy and every number below is some other depth's")
    if not probe["games_differ"]:
        sys.exit("ORACLE and SEARCH_INFORMED produced byte-identical games under identical seeds, so "
                 "the oracle IS the informed arm; the comparison against it would cancel to 0.0000 "
                 "and read as 'perfect information is worth nothing'")

    # Worker-count invariance, measured on a probe rather than assumed, before spending hours.
    one = head_to_head(900, ORACLE, GREEDY, 500, 1)
    many = head_to_head(900, ORACLE, GREEDY, 500, 4)
    print(f"  worker invariance probe: 1 worker {one[0]}/{one[1]} vs 4 workers {many[0]}/{many[1]}",
          flush=True)
    if one != many:
        sys.exit("1 worker and 4 workers disagree, so the block decomposition is not seed-stable "
                 "and the run below would not be reproducible")

    print("\n--- ORACLE against the fixed panel (both seat orders) ---", flush=True)
    panel = {}
    for index, (name, opponent) in enumerate((("GREEDY", GREEDY), ("CONSERVATIVE", CONSERVATIVE),
                                              ("AUTO_DEFEND", AUTO_DEFEND), ("RANDOM", RANDOM))):
        wins, decided, capped = head_to_head(index, ORACLE, opponent, GAMES, WORKERS)
        panel[name] = _report(f"ORACLE vs {name}", wins, decided, capped,
                              time.perf_counter() - started)

    print("\n--- ORACLE against the other search arms (both seat orders) ---", flush=True)
    arms = {}
    for index, (name, opponent) in enumerate((("SEARCH_UNIFORM", SEARCH_UNIFORM),
                                              ("SEARCH_INFORMED", SEARCH_INFORMED)), start=10):
        wins, decided, capped = head_to_head(index, ORACLE, opponent, GAMES, WORKERS)
        arms[name] = _report(f"ORACLE vs {name}", wins, decided, capped,
                             time.perf_counter() - started)

    print("\n--- THE CEILING ON THE HEADLINE ITSELF: same depth, same class, belief the only "
          "difference ---", flush=True)
    # The headline is a DIFFERENCE. Bounding it needs both arms at the depth being bounded, or the
    # number confounds "does perfect knowledge pay" with "does depth pay" -- and in this engine those
    # two point in opposite directions.
    wins, decided, capped = head_to_head(30, ORACLE, ORACLE_UNIFORM, GAMES, WORKERS)
    arms["ORACLE_UNIFORM"] = _report("ORACLE vs ORACLE_UNIFORM", wins, decided, capped,
                                    time.perf_counter() - started)

    print("\n--- how weak is what is being bounded, against GREEDY ---", flush=True)
    for index, (name, arm) in enumerate((("SEARCH_INFORMED", SEARCH_INFORMED),
                                         ("SEARCH_UNIFORM", SEARCH_UNIFORM),
                                         ("ORACLE_UNIFORM", ORACLE_UNIFORM)), start=20):
        wins, decided, capped = head_to_head(index, arm, GREEDY, GAMES, WORKERS)
        _report(f"{name} vs GREEDY", wins, decided, capped, time.perf_counter() - started)

    print(f"\nRESULT games_per_direction={GAMES} workers={WORKERS} "
          f"search_depth={SEARCH_DEPTH} oracle_depth={ORACLE_DEPTH}", flush=True)
    print("RESULT panel " + " ".join(f"{n}={r:.4f}" for n, r in panel.items()), flush=True)
    print("RESULT arms " + " ".join(f"oracle_vs_{n}={r:.4f}" for n, r in arms.items()), flush=True)
    print(f"wall clock {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()