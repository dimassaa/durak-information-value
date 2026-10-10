"""R10 -- self-play: is a depth stronger, or stronger against THESE opponents?

Two rates that are not comparable and are both needed. `SEARCH_UNIFORM` at depth 4 beating `GREEDY` is a
statement about a matchup; the same arm beating itself at depth 4 is a statement about the policy. A depth
can improve the second and not the first, and a programme that reports only the first will read the
difference as depth buying strength when it bought specialisation.

Reported per depth, both seat orders, resolved through the seat the arm occupied. The games come from the
same pooled block decomposition as the depth curve, gated the same way -- one worker must equal several
before any of these numbers is used.

Written to stdout; the caller redirects into `results/`.
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
        "refusing to run without a private NUMBA_CACHE_DIR: the depth is read from the environment and\n"
        "  numba's cache does not invalidate on a changed global, so two depths sharing a cache mean one\n"
        "  of them silently plays the other's depth."
    )

from analysis.stats import wilson_interval
from durak.rules import DEFAULT_SETTINGS
from durak.runner import _BlockJob, _run_block
from durak.strategies import GREEDY, INFO_PARTICLES, INFO_TRUTH, SEARCH_INFORMED, SEARCH_UNIFORM

DECK = int(os.environ.get("SELFPLAY_DECK", "36"))
BLOCK = int(os.environ.get("SELFPLAY_BLOCK", "250"))
GAMES = int(os.environ.get("SELFPLAY_GAMES", "4000"))
WORKERS = int(os.environ.get("SELFPLAY_WORKERS", str(max(1, (os.cpu_count() or 2) - 4))))
DEPTHS = tuple(int(d) for d in os.environ.get("SELFPLAY_DEPTHS", "1,2,3,4").split(","))
SEED = int(os.environ.get("SELFPLAY_SEED", "1"))

NONE = 0
ARMS = {SEARCH_UNIFORM: INFO_PARTICLES, SEARCH_INFORMED: INFO_TRUTH}
MODES = dict(ARMS)


def head_to_head(a, b, games):
    """`a`'s win rate against `b` over both seat orders. Modes follow the ARM, not the seat."""
    jobs = []
    for index, (p0, p1, seat_of_a) in enumerate(((a, b, 0), (b, a, 1))):
        for start in range(0, games, BLOCK):
            stop = min(start + BLOCK, games)
            jobs.append((_BlockJob(
                pairing_index=800 + index, deck_size=DECK, strategy_p0=p0, strategy_p1=p1,
                info_p0=int(MODES.get(p0, NONE)), info_p1=int(MODES.get(p1, NONE)),
                regime="RULE_DERIVED", seed=SEED, game_start=start, game_count=stop - start,
                sample_modulus=20), seat_of_a))
    if WORKERS == 1:
        results = [(seat, _run_block(job, DEFAULT_SETTINGS, False, False)) for job, seat in jobs]
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [(seat, pool.submit(_run_block, job, DEFAULT_SETTINGS, False, False))
                       for job, seat in jobs]
            results = [(seat, f.result()) for seat, f in futures]

    wins = decided = 0
    for seat_of_a, (out, _r, _m) in results:
        wins += int((out["winner"] == seat_of_a).sum())
        decided += int((out["winner"] >= 0).sum())
    return wins, decided


def stable(a, b):
    kept = WORKERS
    try:
        globals()["WORKERS"] = 1
        serial = head_to_head(a, b, 2 * BLOCK)
        globals()["WORKERS"] = kept
        pooled = head_to_head(a, b, 2 * BLOCK)
    finally:
        globals()["WORKERS"] = kept
    if serial != pooled:
        sys.exit(f"one worker and {kept} workers disagree ({serial} against {pooled}); the block "
                 f"decomposition is not seed-stable and every number below would be unreproducible")
    print(f"  pool stable: 1 worker and {kept} workers both {serial}", flush=True)


def main():
    # ONE DEPTH PER PROCESS. `@njit` freezes module globals at compile time, so a loop that "sets" the
    # depth after the first compile would keep measuring the first depth while printing the later one --
    # a sweep that comes out flat and reads as a finding. Depth arrives as an argument and is applied
    # before anything is compiled.
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else DEPTHS[0]
    os.environ["DURAK_SEARCH_DEPTH"] = str(depth)
    import durak.strategies.search_policy as sp
    if sp.SEARCH_DEPTH != depth:
        sys.exit(f"the policy compiled at depth {sp.SEARCH_DEPTH}, not {depth}")

    print(f"=== R10 self-play: depth {depth}, deck {DECK}, {GAMES} games per direction, "
          f"workers {WORKERS} ===", flush=True)
    started = time.perf_counter()

    stable(SEARCH_UNIFORM, SEARCH_UNIFORM)
    for arm, label in ((SEARCH_UNIFORM, "SEARCH_UNIFORM"), (SEARCH_INFORMED, "SEARCH_INFORMED")):
        wins, decided = head_to_head(arm, arm, GAMES)
        rate = wins / decided if decided else float("nan")
        lo, hi = wilson_interval(wins, decided) if decided else (float("nan"), float("nan"))
        print(f"  SELFPLAY depth={depth} arm={label:<16} {rate:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"({wins}/{decided})", flush=True)
        # The contrast that makes self-play worth running: the same arm against a FIXED opponent, so a
        # rise in one and not the other is specialisation rather than strength.
        w2, d2 = head_to_head(arm, GREEDY, GAMES)
        r2 = w2 / d2 if d2 else float("nan")
        print(f"  vs GREEDY depth={depth} arm={label:<16} {r2:.4f}  ({w2}/{d2})   "
              f"[{time.perf_counter() - started:6.0f}s]", flush=True)

    print(f"=== self-play depth {depth} done in {time.perf_counter() - started:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()