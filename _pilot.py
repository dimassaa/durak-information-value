"""The 10% pilot. Runs BEFORE everything, because all six instrument defects were caught by one.

`IMPROVEMENTS.md` and `docs/EXPERIMENTS.md` record what a pilot has already saved this project from; a
full run would have buried every one of them. Written to datasets/, never to /tmp.

Scale and worker count come from the environment, so the same script produces the development pilot and
the 10% one without a second copy whose only difference is a number. `AUTO_ATTACK` is absent by design:
`RunManifest` refuses it in a tournament, because pooling a driver that never defends into a descriptive
statistic produces a statistic about the driver.
"""
import os
from dataclasses import replace
from pathlib import Path

from durak.config import Settings
from durak.runner import RunManifest, run_tournament
from durak.strategies import (AUTO_DEFEND, CHEATER_DOC, CONSERVATIVE, GREEDY,
                              RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM)

GAMES_PER_PAIRING = int(os.environ.get("PILOT_GAMES_PER_PAIRING", "1000"))
# `run_tournament(workers=0)` resolves to `os.cpu_count()`, i.e. every core, which leaves the machine
# unresponsive for the duration and makes the wall clock a measure of the scheduler rather than of the
# run. Four cores are held back unless told otherwise.
WORKERS = int(os.environ.get("DURAK_WORKERS", str(max(1, (os.cpu_count() or 2) - 4))))


def main():
    out = Path("datasets/pilot")
    # `Settings` is a frozen dataclass, so a pilot cannot be produced by mutating one. The scale is a field,
    # which is correct: the scale of a run is part of its configuration and belongs in the manifest, not in a
    # global someone can change after the fact.
    settings = replace(Settings(), games_per_pairing=GAMES_PER_PAIRING)

    manifest = RunManifest(
        run_id=0, seed=0,
        strategies=(RANDOM, GREEDY, AUTO_DEFEND, CONSERVATIVE,
                    SEARCH_UNIFORM, SEARCH_INFORMED, CHEATER_DOC),
        deck_sizes=(36,), regimes=("RULE_DERIVED",), settings=settings, block_size=100,
        games_per_pairing=GAMES_PER_PAIRING,
    )
    print(f"pilot: {GAMES_PER_PAIRING} games/pairing, {WORKERS} workers, search depth from environment",
          flush=True)
    summary = run_tournament(manifest, out, workers=WORKERS)
    print("games", summary.games, "anomalies", summary.anomalies, "wins", summary.wins)


if __name__ == "__main__":
    main()
