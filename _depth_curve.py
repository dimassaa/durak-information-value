"""The depth curve: does the information effect fall, stay flat, or turn around?

This is the project's central experiment. Depth is a RESEARCH parameter, not a performance knob: on one
harness the information effect was +0.0176 at depth 1 and -0.0527 at depth 2, so the value changes the
answer. Until this curve is measured no number may be published, including the withdrawn +0.0799.

Two readings remain open and the data chooses between them:
  * depth makes perfect knowledge genuinely less useful, because a rational player already does nearly
    the same thing -> the curve falls and stays down;
  * at depth 1 the effect was an artifact of a weak evaluation, and a competent player converts knowledge
    into play differently -> the curve is non-monotone, falling then recovering.

Run as a real file: the process-pool path re-imports `__main__`.

Ground rules, each of which has already produced one wrong conclusion in this project:
  * `winner` is a SEAT; the arm is resolved through the seat it occupied, per direction.
  * The two arms get DIFFERENT info modes. Passing NONE to both measures one policy twice, which is
    exactly the error that produced a wrong earlier verdict.
  * One seed per depth, and the SAME seed at every depth, so a difference between depths is a difference
    between policies rather than between samples.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from durak.engine import play_block
from durak.rules import DEFAULT_SETTINGS
from durak.runner import _BlockJob, _run_block
from durak.strategies import (AUTO_DEFEND, GREEDY, INFO_PARTICLES, INFO_TRUTH, InfoMode,
                              SEARCH_INFORMED, SEARCH_UNIFORM)

GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
#: Run seed. An argument with a default of 1, so an existing two-argument invocation plays exactly the
#: same games it always did. R9 re-runs one configuration under several seeds, and a conclusion that
#: depends on one seed is a conclusion about that seed.
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TRUTH, PARTICLES = 2, 1


def set_depth(d):
    """Depth comes from the environment BEFORE any compilation: `@njit` freezes globals at compile time."""
    os.environ["DURAK_SEARCH_DEPTH"] = str(d)
    import durak.strategies.search_policy as sp
    assert sp.SEARCH_DEPTH == d, (
        f"the policy compiled at depth {sp.SEARCH_DEPTH}, not {d}: numba freezes module globals, so the "
        f"depth must be set before the first compile, in this process"
    )
    return d


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


#: Processes used for one head-to-head. `play_block` is SERIAL by design -- `engine.py` records that
#: `prange` "arrives in Task 14" -- so the only parallelism available is a pool over game blocks, which is
#: what `run_tournament` uses and why a 113,950-game tournament measured 339 games/s against this
#: script's ~2 games/s at depth 4 when it ran one process. Set to 1 to reproduce the sequential path.
WORKERS = int(os.environ.get("CURVE_WORKERS", str(max(1, (os.cpu_count() or 2) - 4))))
BLOCK = int(os.environ.get("CURVE_BLOCK", "250"))


def _blocks(a, b, ip_a, ip_b, games):
    """Both seat orders as the runner's own picklable block jobs, one per `BLOCK` games.

    `seat_of_a` travels with the job rather than being recomputed at the tally, so the resolution of the
    outcome cannot drift between the two directions.
    """
    jobs = []
    for index, (p0, p1, i0, i1, seat_of_a) in enumerate(
            ((a, b, ip_a, ip_b, 0), (b, a, ip_b, ip_a, 1))):
        for start in range(0, games, BLOCK):
            stop = min(start + BLOCK, games)
            jobs.append((_BlockJob(pairing_index=700 + index, deck_size=36, strategy_p0=p0,
                                   strategy_p1=p1, info_p0=i0, info_p1=i1, regime="RULE_DERIVED",
                                   seed=SEED, game_start=start, game_count=stop - start,
                                   sample_modulus=20),
                         seat_of_a))
    return jobs


def head_to_head(a, b, ip_a, ip_b, games=GAMES):
    """Arm `a`'s win rate against arm `b`, both seat orders, winner resolved per direction.

    The info modes follow the ARM, not the seat. Passing the same pair in both directions hands arm `b`
    the truth in the mirror game, the two directions cancel exactly, and the delta comes out a flat
    +0.0000 -- which looks like a null result and is the harness cancelling itself.

    A game's seed derives from `(seed, pairing_index, game_start)`, so the block decomposition plays the
    SAME games the serial version plays. That is checked rather than assumed: `pool_is_stable` runs both
    and refuses to continue if they disagree.
    """
    jobs = _blocks(a, b, ip_a, ip_b, games)
    if WORKERS == 1:
        results = [(seat, _run_block(job, DEFAULT_SETTINGS, True, False)) for job, seat in jobs]
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [(seat, pool.submit(_run_block, job, DEFAULT_SETTINGS, True, False))
                       for job, seat in jobs]
            results = [(seat, f.result()) for seat, f in futures]

    wins = n = 0
    for seat_of_a, (out, _r, _m) in results:
        wins += int((out["winner"] == seat_of_a).sum())
        n += len(out)
    return wins, n


def pool_is_stable(a, b, ip_a, ip_b, games=2 * BLOCK):
    """Refuse to report a pooled number unless one worker and several agree exactly.

    Cheap, and it is the check that makes the pool usable at all: a block decomposition that shifted a
    single game would change the interval by a hair and produce a plausible, wrong curve.
    """
    global WORKERS
    kept = WORKERS
    try:
        WORKERS = 1
        serial = head_to_head(a, b, ip_a, ip_b, games)
        WORKERS = kept
        pooled = head_to_head(a, b, ip_a, ip_b, games)
    finally:
        WORKERS = kept
    if serial != pooled:
        sys.exit(f"one worker and {kept} workers disagree ({serial} against {pooled}), so the block "
                 f"decomposition is not seed-stable and every number below would be unreproducible")
    print(f"  pool stable: 1 worker and {kept} workers both {serial}", flush=True)


def main():
    set_depth(DEPTH)
    print(f"=== depth {DEPTH}, {GAMES} games per direction ===", flush=True)
    started = time.perf_counter()

    print(f"  workers={WORKERS} block={BLOCK}", flush=True)
    pool_is_stable(SEARCH_INFORMED, SEARCH_UNIFORM, TRUTH, PARTICLES)

    i_wins, n = head_to_head(SEARCH_INFORMED, SEARCH_UNIFORM, TRUTH, PARTICLES)
    delta = 2 * i_wins / n - 1.0
    se = np.sqrt(2 * (i_wins / n) * (1 - i_wins / n) / n)
    lo, hi = wilson(i_wins, n)
    print(f"  informed {i_wins}/{n} = {i_wins / n:.4f}  CI [{lo:.4f}, {hi:.4f}]", flush=True)
    print(f"  DELTA {delta:+.4f}  95% CI [{delta - 1.96 * se:+.4f}, {delta + 1.96 * se:+.4f}]",
          flush=True)

    for name, opp in (("GREEDY", GREEDY), ("AUTO_DEFEND", AUTO_DEFEND)):
        k, m = head_to_head(SEARCH_INFORMED, opp, TRUTH, PARTICLES)
        print(f"  informed vs {name:<12} {k / m:.4f}  ({k}-{m - k})", flush=True)
        k, m = head_to_head(SEARCH_UNIFORM, opp, PARTICLES, PARTICLES)
        print(f"  uniform  vs {name:<12} {k / m:.4f}  ({k}-{m - k})", flush=True)

    print(f"  wall clock {time.perf_counter() - started:.0f}s", flush=True)
    print(f"RESULT depth={DEPTH} delta={delta:+.4f} games={n}", flush=True)


if __name__ == "__main__":
    main()