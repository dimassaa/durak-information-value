"""R12/R13 -- the publication datasets. Written to `datasets/`, never to /tmp.

Generalises `_pilot.py`, which is the same manifest builder at the pilot's scale. It exists separately
only because the pilot is already queued in a chain that has not run yet, and editing a file a queued
step depends on trades a verified run for an unverified one. Once the pilot has landed, this replaces it.

Everything that decides a number is read from the environment and printed next to the result, because a
dataset that does not say which depth, weights and memory horizon produced it is a dataset whose numbers
cannot be attributed -- and the depth is a research parameter, not a performance knob, so it has to travel
with every figure taken from this.

`AUTO_ATTACK` is absent because `RunManifest` refuses it in a tournament. It is a driver that never
defends; pooled into a descriptive statistic it supplies about a quarter of the games and nearly triples
the mean loser's final hand, so the statistic describes the driver rather than the game.
"""

import os
from dataclasses import replace
from pathlib import Path

from durak.config import Settings
from durak.runner import RunManifest, run_tournament
from durak.strategies import (
    AUTO_DEFEND,
    CHEATER_DOC,
    CONSERVATIVE,
    GREEDY,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
)

NAME = os.environ.get("DATASET_NAME", "primary")
GAMES_PER_PAIRING = int(os.environ.get("DATASET_GAMES_PER_PAIRING", "6500"))
DECK_SIZES = tuple(int(v) for v in os.environ.get("DATASET_DECKS", "36").split(","))
#: Four cores are held back unless told otherwise. `run_tournament(workers=0)` resolves to
#: `os.cpu_count()`, i.e. every core, which makes the wall clock a measure of the scheduler rather than
#: of the run.
WORKERS = int(os.environ.get("DURAK_WORKERS", str(max(1, (os.cpu_count() or 2) - 4))))
RECORD_LEVELS = os.environ.get("DATASET_RECORD_LEVELS", "1") != "0"

STRATEGIES = (RANDOM, GREEDY, AUTO_DEFEND, CONSERVATIVE,
              SEARCH_UNIFORM, SEARCH_INFORMED, CHEATER_DOC)


def main():
    # Read the arms' own frozen value, NOT `EVAL_WEIGHTS`. It printed the incumbent while the search
    # played the selected vector, so the console claimed a configuration the dataset did not use -- the
    # same defect as the oracle's. The manifest was always right; only the line a reader trusted was wrong.
    from durak.opponent import MEMORY_PLY
    from durak.strategies.search_policy import PARTICLE_COUNT, SEARCH_DEPTH, WEIGHTS

    out = Path("datasets") / NAME
    settings = replace(Settings(), games_per_pairing=GAMES_PER_PAIRING)

    manifest = RunManifest(
        run_id=0, seed=1, strategies=STRATEGIES,
        deck_sizes=DECK_SIZES, regimes=("RULE_DERIVED",), settings=settings, block_size=100,
        games_per_pairing=GAMES_PER_PAIRING,
    )

    print(f"=== dataset {NAME} ===", flush=True)
    print(f"  games/pairing {GAMES_PER_PAIRING}  decks {DECK_SIZES}  workers {WORKERS}  -> {out}",
          flush=True)
    # The configuration that decides every number in the dataset, printed before the run so it cannot
    # be separated from the result.
    print(f"  SEARCH_DEPTH={SEARCH_DEPTH}  MEMORY_PLY={MEMORY_PLY}  PARTICLE_COUNT={PARTICLE_COUNT}", flush=True)
    print(f"  weights={WEIGHTS}  records levels<={2 if RECORD_LEVELS else 1}", flush=True)

    summary = run_tournament(manifest, out, workers=WORKERS,
                             record_rounds=RECORD_LEVELS, record_moves=RECORD_LEVELS)
    print("games", summary.games, "anomalies", summary.anomalies, flush=True)
    print("wins", summary.wins, flush=True)
    if summary.anomalies:
        print(f"REFUSING TO CALL THIS PUBLISHABLE: {summary.anomalies} anomalies", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()