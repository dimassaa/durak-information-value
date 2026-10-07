"""Throughput benchmark: games per second and CPU-seconds per game, per policy.

Why this exists
---------------
Spec 8.4 makes compute cost a first-class result: a strategy that wins 1% more while costing 100x
the CPU is a *different* result, not a better one. A win rate without its cost is half a number.

And it decides the dataset scale. Stage 10 cannot choose a sample size without knowing what one
game costs, and a scale chosen after seeing throughput is not a pre-registration. The measured
numbers therefore go into the stage report *before* the scale is picked.

Why a separate `perf` marker
----------------------------
Excluded from the default suite by `addopts`. Wall-clock thresholds fail on shared hardware for
reasons unrelated to the code under test, so a routine `pytest` must stay a correctness signal
only. Run deliberately, with `pytest -m perf benchmarks/bench_engine.py`.

Measured under `NUMBA_DISABLE_JIT=1` as well as compiled
-------------------------------------------------------
The interpreted column is the cost of the *reference* path, not a production option: it is what
equivalence testing runs. Reporting only the compiled number would hide that the guarantee itself
costs about 90x.

What is measured
----------------
Wall-clock throughput of `play_block` with telemetry off, which is the configuration the benchmark
that decides dataset scale cares about. A warm-up call precedes every measurement so Numba
compilation is not counted as game cost -- the difference between the first and later runs is
roughly twelve seconds, which would otherwise dominate a short measurement.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from durak import rules
from durak.engine import play_block
from durak.strategies import AUTO_ATTACK, AUTO_DEFEND, GREEDY, RANDOM, InfoMode

POLICIES = {
    "RANDOM": RANDOM,
    "GREEDY": GREEDY,
    "AUTO_ATTACK": AUTO_ATTACK,
    "AUTO_DEFEND": AUTO_DEFEND,
}

# Small enough that the benchmark runs in seconds on a laptop and still yields a stable rate.
DEFAULT_GAMES = 2_000


def _under_xdist() -> bool:
    """Whether pytest is distributing this run across worker processes.

    Under xdist each worker is a separate process, so a per-process CPU measure times the worker rather
    than the work, and the reported games/sec is plausible and wrong. The suite's default `addopts`
    excludes `-m perf`, so this only bites when someone runs the benchmark deliberately in parallel -- which
    is exactly the moment a plausible wrong throughput number would be believed.
    """
    return bool(os.environ.get("PYTEST_XDIST_WORKER") or os.environ.get("PYTEST_XDIST_WORKER_COUNT"))


def run(policies=("RANDOM", "GREEDY"), deck_size: int = 36, games: int = DEFAULT_GAMES,
        record_telemetry: bool = False) -> dict:
    """Measure each policy's cost. Returns games/sec and CPU-seconds per game.

    Both directions are played, because that is what the tournament does and a one-directional
    measurement would not describe the run it is meant to size.
    """
    tunables = rules.DEFAULT_SETTINGS
    result = {"games_per_policy": int(games), "deck_size": deck_size,
              "record_telemetry": bool(record_telemetry), "policies": {}}

    for name in policies:
        strategy = POLICIES[name]

        # Warm-up: compilation is not game cost.
        play_block(deck_size, strategy, strategy, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 20,
                   tunables, record_telemetry, record_telemetry)

        started = time.perf_counter()
        # Both baselines captured together. Taking only the wall-clock baseline and then
        # differencing an absolute CPU reading compares two different quantities -- which is how
        # this first reported a *negative* cost of -20 seconds.
        cpu_started = time.process_time()
        outcomes, _, _ = play_block(
            deck_size, strategy, strategy, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, games,
            tunables, record_telemetry, record_telemetry,
        )
        elapsed = time.perf_counter() - started

        played = int(len(outcomes))
        cpu_elapsed = time.process_time() - cpu_started
        result["policies"][name] = {
            "games": played,
            "seconds": elapsed,
            "games_per_second": played / elapsed if elapsed > 0 else float("inf"),
            "cpu_seconds_per_game": cpu_elapsed / played if played else float("nan"),
            "mean_rounds": float(np.mean(outcomes["rounds"])),
            # Median alongside the mean, and it is not optional. A game stopped at the round cap
            # contributes 10,000 rounds, so a 5% stall rate moves the mean from 13 to 484 while the
            # median barely moves -- which reads as "GREEDY games are long" when the truth is
            # "5% of them never finish". Sizing a dataset off that mean would be badly wrong.
            "median_rounds": float(np.median(outcomes["rounds"])),
            "mean_actions": float(np.mean(outcomes["actions"])),
            "undecided": int(np.count_nonzero(outcomes["winner"] < 0)),
            "undecided_share": float(np.mean(outcomes["winner"] < 0)),
            # Cost per *decided* game, which is the number that describes a finished game rather
            # than the cost of a strategy that also produces a stall.
            "cpu_seconds_per_decided_game": (
                cpu_elapsed / int(np.count_nonzero(outcomes["winner"] >= 0))
                if np.any(outcomes["winner"] >= 0) else float("nan")
            ),
        }

    # Relative cost, which is the number the reports quote beside every win rate.
    if "GREEDY" in result["policies"] and "RANDOM" in result["policies"]:
        greedy = result["policies"]["GREEDY"]["cpu_seconds_per_game"]
        random = result["policies"]["RANDOM"]["cpu_seconds_per_game"]
        result["greedy_over_random"] = greedy / random if random > 0 else float("inf")

    return result


@pytest.mark.perf
def test_throughput_is_measured_for_every_policy(capsys):
    """The benchmark itself.

    Prints rather than asserts a rate: wall-clock thresholds fail on shared hardware for reasons
    unrelated to the code, and a benchmark that asserts its own speed is a benchmark that fails
    when the machine is busy. The printed numbers are the deliverable.

    Skipped under xdist, where each worker is a separate process and the CPU figures describe the
    worker rather than the work -- a plausible wrong number is worse than no number.
    """
    if _under_xdist():
        pytest.skip("throughput is meaningless under xdist: each worker times itself, not the run")
    report = run()
    print("\n=== engine throughput (deck 36, telemetry off) ===")
    for name, values in report["policies"].items():
        print(
            f"{name:12s} {values['games_per_second']:9.0f} games/s   "
            f"{values['cpu_seconds_per_game'] * 1e6:8.1f} us CPU/game   "
            f"rounds mean {values['mean_rounds']:7.1f} median {values['median_rounds']:5.1f}   "
            f"undecided {values['undecided_share'] * 100:4.1f}%"
        )
    if "greedy_over_random" in report:
        print(f"GREEDY costs {report['greedy_over_random']:.2f}x RANDOM in CPU time")
    print(f"total {sum(v['games'] for v in report['policies'].values())} games")


@pytest.mark.perf
def test_telemetry_cost_is_measured_separately():
    """Level 2 and level 3 are not free, and the dataset scale depends on knowing by how much."""
    if _under_xdist():
        pytest.skip("throughput is meaningless under xdist: each worker times itself, not the run")
    without = run(games=1_000, record_telemetry=False)
    with_telemetry = run(games=1_000, record_telemetry=True)

    for name in without["policies"]:
        plain = without["policies"][name]["cpu_seconds_per_game"]
        full = with_telemetry["policies"][name]["cpu_seconds_per_game"]
        print(
            f"{name:12s} telemetry off {plain * 1e6:8.1f} us   on {full * 1e6:8.1f} us   "
            f"({full / plain if plain else float('inf'):.2f}x)"
        )


@pytest.mark.perf
def test_per_game_cost_is_positive_and_finite():
    """The only assertion here: a cost of zero would mean the measurement is broken.

    Deliberately not a floor or a ceiling. A minimum rate here would fail on a loaded machine, and a
    maximum would fail the moment a strategy got slower -- which is information a benchmark should
    report rather than reject.
    """
    report = run(games=200)

    for name, values in report["policies"].items():
        cost = values["cpu_seconds_per_game"]
        assert np.isfinite(cost), f"{name} reported a non-finite cost: {cost}"
        assert cost > 0, f"{name} reported a zero cost, so nothing was measured"


if __name__ == "__main__":
    print(f"NUMBA_DISABLE_JIT={os.environ.get('NUMBA_DISABLE_JIT', '0')} "
          f"policy count={len(POLICIES)}")
    report = run()
    for name, values in report["policies"].items():
        print(f"{name:12s} {values['games_per_second']:9.0f} games/s  "
              f"{values['cpu_seconds_per_game'] * 1e6:8.1f} us CPU/game  "
              f"median {values['median_rounds']:.0f} rounds  "
              f"{values['undecided_share'] * 100:.1f}% undecided")