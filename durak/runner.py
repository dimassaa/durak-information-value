"""The run manifest and the tournament runner.

The manifest is the reproducibility contract
--------------------------------------------
Everything needed to rebuild a run bit-for-bit, in one JSON file written beside the data: the run
id, the seed, the strategy list, the deck sizes, the games per pairing, the regimes, *every*
setting that can change a result, the three version stamps, the block size, and the phase cut.

Two decisions inside it are worth stating:

Every setting that can change a result is recorded explicitly -- `throwin_defense_ranks`,
`soft_penalty_lambda`, `particle_count`, `sample_modulus`, and the phase cut. A sweep over the
throw-in rule whose arms cannot be told apart produces two numbers and one conclusion that is
wrong. The test names these fields rather than iterating the dataclass, so a new setting that is
not recorded fails a test instead of being silently added.

The timestamp is in the file and excluded from equality. A dataset with no generation date cannot
be cited; two manifests that differ only by a timestamp cannot be compared, and comparing them is
the whole point.

Identity
--------
A game is `(run_seed, pairing_index, game_index)` and nothing else -- not scheduling, not
wall-clock, not the block's position in the run. That is what makes a run resumable, makes a
1-worker run equal an 8-worker run, and makes the manifest sufficient on its own.

Worker counts
-------------
`workers` is accepted and validated here and currently runs serially. Task 15 makes it select a
`prange` partitioning. Until then the corresponding test is satisfied trivially, and says so; the
invariance that is real today is tested over block sizes, which is the partition boundary that
exists now.

Audit
-----
Every run is audited before it is summarised, and the anomaly counts travel with the summary.
Spec 7.2 makes a run with any flagged game unpublishable until the flags are explained, which
requires that a reader can see them without re-running anything.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

from durak import rules
from durak.audit import audit_outcomes
from durak.config import Settings
from durak.engine import play_block
from durak.rules import (
    EARLY_FRACTION,
    LATE_FRACTION,
    RuleTunables,
    settings_from_config,
)
from durak.strategies import InfoMode
from durak.strategies.base import StrategyId
from durak.strategies.registry import registered_ids
from durak.telemetry import read_parquet, write_parquet, write_manifest
from durak.tournament import (
    REGIMES,
    RULE_DERIVED,
    ROTATION_BALANCED,
    Pairing,
    first_attacker_override,
    pairing_grid,
)
from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

# Wilson score, used for every win rate the reports quote. 1.96 is the two-sided 95% interval,
# pre-registered so no interval width is chosen after seeing a p-value.
WILSON_Z = 1.96


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves at small
    counts -- and because a symmetric interval around 0.5 is exactly what a symmetry check needs
    to be able to contain.
    """
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to rebuild a run. Serialised to JSON beside the dataset."""

    run_id: str
    seed: int
    strategies: list
    deck_sizes: list
    games_per_pairing: int
    regimes: list
    settings: Settings
    block_size: int
    engine_version: int = ENGINE_VERSION
    rules_version: int = RULES_VERSION
    schema_version: int = SCHEMA_VERSION
    #: Excluded from equality, present in the file. See the module docstring.
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        compare=False,
    )

    def __post_init__(self):
        known = set(registered_ids())
        for strategy_id in self.strategies:
            if int(strategy_id) not in known:
                raise ValueError(
                    f"unknown strategy id {strategy_id}; registered ids are {sorted(known)}"
                )
        for regime in self.regimes:
            if regime not in REGIMES:
                raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
        if self.games_per_pairing < 0:
            raise ValueError("games_per_pairing must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "strategies": [int(s) for s in self.strategies],
            "deck_sizes": [int(d) for d in self.deck_sizes],
            "games_per_pairing": self.games_per_pairing,
            "regimes": list(self.regimes),
            "settings": asdict(self.settings),
            "block_size": self.block_size,
            "engine_version": self.engine_version,
            "rules_version": self.rules_version,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            # The phase cut is a pre-registered choice. It belongs in the manifest because a
            # per-round curve is meaningless without knowing where its boundaries were, and a
            # boundary in a code comment is not a boundary a reader can verify.
            "phase_cut": {
                "late_fraction": LATE_FRACTION,
                "early_fraction": EARLY_FRACTION,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        settings_fields = {f for f in Settings.__dataclass_fields__}
        settings_values = {k: v for k, v in data["settings"].items() if k in settings_fields}
        # JSON has no tuple: `eval_weights` comes back as a list and would not compare equal to
        # the tuple the dataclass holds. Converted back rather than left to compare unequal,
        # because a manifest that does not round-trip is a run nobody can reproduce.
        if "eval_weights" in settings_values:
            settings_values["eval_weights"] = tuple(settings_values["eval_weights"])
        return cls(
            run_id=data["run_id"],
            seed=data["seed"],
            strategies=list(data["strategies"]),
            deck_sizes=list(data["deck_sizes"]),
            games_per_pairing=data["games_per_pairing"],
            regimes=list(data["regimes"]),
            settings=Settings(**settings_values),
            block_size=data["block_size"],
            engine_version=data["engine_version"],
            rules_version=data["rules_version"],
            schema_version=data["schema_version"],
            generated_at=data.get("generated_at", ""),
        )

    @property
    def tunables(self) -> RuleTunables:
        return settings_from_config(self.settings)

    @property
    def grid(self) -> list:
        return pairing_grid(self.strategies, deck_sizes=self.deck_sizes, regimes=self.regimes)


@dataclass
class RunSummary:
    """What a run produced, and whether it may be published."""

    games: int
    wins: dict
    anomalies: int
    audit: dict
    elapsed_seconds: float
    rows: dict = field(default_factory=dict)
    #: Games each strategy actually appeared in, so a win rate has the right denominator.
    trials: dict = field(default_factory=dict)
    strategy_ids: list = field(default_factory=list)
    #: Wins by seat, which is what spec 7.3's symmetry check consumes. Distinct from `wins` by
    #: strategy: for the `RANDOM` self-pairing a per-strategy rate is identically 1.0, because
    #: every decided game was won by *a* `RANDOM`. The symmetry claim is about seats.
    seat_wins: dict = field(default_factory=dict)

    @property
    def publishable(self) -> bool:
        """Spec 7.2: a run with any flagged game is not publishable until the flags are explained."""
        return self.anomalies == 0

    @property
    def undecided(self) -> int:
        """Games stopped at the round cap, which have no winner to attribute."""
        return int(self.wins.get("undecided", 0))

    def games_for(self, strategy_id: int) -> int:
        """Games a strategy appeared in, counted once per game."""
        return self.decided + self.undecided // max(1, len(self.strategy_ids))

    def win_rate(self, strategy_id: int, undecided_counts_as: str = "excluded") -> float:
        """A strategy's share of decided games.

        `undecided_counts_as` is `"excluded"` (the default) or `"loss"`. Excluded is the honest
        choice and the one the reports use: a game that hit the round cap is a known gap in the
        evidence, not a result. Counting it as a loss would quietly import a rule -- whoever was
        attacking lost -- that the engine never decided.
        """
        strategy_id = int(strategy_id)
        wins = int(self.wins.get(strategy_id, 0))
        trials = self._trials_for(strategy_id)
        if undecided_counts_as == "loss":
            trials += self._share_of_undecided(strategy_id)
        return wins / trials if trials else float("nan")

    def _trials_for(self, strategy_id: int) -> int:
        return int(self.trials.get(int(strategy_id), 0))

    def seat_rate(self, seat: int) -> float:
        """Seat `seat`'s share of *decided* games. The symmetry check reads this.

        Excludes games that hit the round cap, because those have no seat that won. Reported
        alongside the count of excluded games, so the reader can see what was dropped rather than
        having to infer it.
        """
        decided = self.seat_wins.get(0, 0) + self.seat_wins.get(1, 0)
        if not decided:
            return float("nan")
        return self.seat_wins.get(int(seat), 0) / decided

    def _share_of_undecided(self, strategy_id: int) -> int:
        if not self.strategy_ids:
            return 0
        return self.undecided // len(self.strategy_ids)

    def wilson(self, successes: int, trials: int) -> Tuple[float, float]:
        return wilson_interval(successes, trials)


def _write_levels(outcomes, rounds, moves, out_dir: Path, level: int, schema_version: int):
    write_parquet(outcomes, out_dir, schema_version=schema_version, level=1)
    if rounds.size:
        write_parquet(rounds, out_dir / "rounds", schema_version=schema_version, level=2)
    if moves.size:
        write_parquet(moves, out_dir / "moves", schema_version=schema_version, level=3)


def run_tournament(manifest: RunManifest, out_dir, workers: int = 0,
                   record_rounds: bool = True, record_moves: bool = True) -> RunSummary:
    """Play every cell of the grid and write the dataset.

    `workers` is accepted and validated, and runs serially for now; Task 15 gives it teeth. The
    result must not depend on it either way, which is what makes the parallel port safe to attempt.

    Blocks are cut by `game_index` and nothing else, so the same run at any block size produces the
    same games. That is asserted in the tests and is the property the `prange` port must preserve.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workers < 0:
        raise ValueError(f"workers must be non-negative, got {workers}")

    grid = manifest.grid
    tunables = manifest.tunables
    started = time.perf_counter()

    all_outcomes = []
    all_rounds = []
    all_moves = []

    for pairing in grid:
        override = None  # resolved per game below
        for start in range(0, manifest.games_per_pairing, manifest.block_size):
            count = min(manifest.block_size, manifest.games_per_pairing - start)

            if pairing.first_attacker_regime == ROTATION_BALANCED:
                # The rotation alternates on the absolute game index, so a block must apply it
                # game by game rather than once for the whole block.
                block_outcomes = []
                block_rounds = []
                block_moves = []
                for offset in range(count):
                    game_index = start + offset
                    one, rounds, moves = play_block(
                        pairing.deck_size, pairing.strategy_p0, pairing.strategy_p1,
                        InfoMode.NONE, InfoMode.NONE, manifest.seed, pairing.pairing_index,
                        game_index, 1, tunables, record_rounds, record_moves,
                        run_id=0, sample_modulus=manifest.settings.sample_modulus,
                        first_attacker_override=first_attacker_override(
                            pairing.first_attacker_regime, game_index),
                    )
                    block_outcomes.append(one)
                    if rounds.size:
                        block_rounds.append(rounds)
                    if moves.size:
                        block_moves.append(moves)
                outcomes = np.concatenate(block_outcomes)
                rounds = (np.concatenate(block_rounds) if block_rounds
                          else np.zeros(0, dtype=rounds.dtype if hasattr(rounds, "dtype")
                                         else np.dtype([])))
                moves = (np.concatenate(block_moves) if block_moves
                         else np.zeros(0, dtype=moves.dtype if hasattr(moves, "dtype")
                                        else np.dtype([])))
            else:
                outcomes, rounds, moves = play_block(
                    pairing.deck_size, pairing.strategy_p0, pairing.strategy_p1,
                    InfoMode.NONE, InfoMode.NONE, manifest.seed, pairing.pairing_index,
                    start, count, tunables, record_rounds, record_moves,
                    run_id=0, sample_modulus=manifest.settings.sample_modulus,
                    first_attacker_override=first_attacker_override(
                        pairing.first_attacker_regime, start),
                )

            all_outcomes.append(outcomes)
            if rounds.size:
                all_rounds.append(rounds)
            if moves.size:
                all_moves.append(moves)

    if not all_outcomes:
        (out_dir / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
        return RunSummary(games=0, wins={}, anomalies=0, audit={"games": 0, "clean": True},
                          elapsed_seconds=time.perf_counter() - started)

    outcomes = np.concatenate(all_outcomes)
    rounds = np.concatenate(all_rounds) if all_rounds else np.zeros(0, dtype=_round_dtype())
    moves = np.concatenate(all_moves) if all_moves else np.zeros(0, dtype=_move_dtype())

    # `run_id` is stamped after the fact rather than threaded through every call: a run is one
    # dataset, and the id identifies the run, not the cell.
    for column in ("run_id",):
        if column in outcomes.dtype.names:
            outcomes[column] = 0

    _write_levels(outcomes, rounds, moves, out_dir / "games", 1, manifest.schema_version)
    if rounds.size:
        write_parquet(rounds, out_dir / "rounds", schema_version=manifest.schema_version, level=2)
    if moves.size:
        write_parquet(moves, out_dir / "moves", schema_version=manifest.schema_version, level=3)

    (out_dir / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
    write_manifest(out_dir / "games")

    audit = audit_outcomes(outcomes)
    wins = wins_by_strategy(outcomes)
    strategy_ids = sorted(set(int(v) for v in np.unique(outcomes["strategy_p0"]))
                          | set(int(v) for v in np.unique(outcomes["strategy_p1"])))
    decided = int(np.count_nonzero(outcomes["winner"] >= 0))
    # Games, not seat-slots: a `RANDOM` self-pairing puts the same strategy in both seats of one
    # game, and counting slots would double it.
    trials = {
        strategy: int(np.count_nonzero(
            (outcomes["strategy_p0"] == strategy) | (outcomes["strategy_p1"] == strategy)
        ))
        for strategy in strategy_ids
    }
    seat_wins = {
        0: int(np.count_nonzero(outcomes["winner"] == 0)),
        1: int(np.count_nonzero(outcomes["winner"] == 1)),
    }
    if decided != sum(seat_wins.values()):
        raise AssertionError(
            f"seat wins {seat_wins} do not sum to the decided count {decided}; a winner value "
            f"outside {{0, 1}} is leaking into the tally"
        )

    return RunSummary(
        games=len(outcomes),
        wins=wins,
        trials=trials,
        strategy_ids=strategy_ids,
        seat_wins=seat_wins,
        anomalies=audit["anomalous_games"],
        audit=audit,
        elapsed_seconds=time.perf_counter() - started,
        rows={"rounds": len(rounds), "moves": len(moves)},
    )


def wins_by_strategy(outcomes) -> dict:
    """Wins per strategy id, plus an ``undecided`` count for games that hit the round cap.

    Keyed by strategy rather than by seat, because a seat count says nothing about a strategy: it
    is a function of which cells the grid happened to contain. Each row carries the strategy in
    each seat, so seat 0's wins are attributed to ``strategy_p0`` and seat 1's to ``strategy_p1``.

    Games with no winner are counted separately rather than folded into either strategy. Dropping
    them would quietly raise every strategy's apparent win rate by the cap rate, which for
    ``GREEDY`` is around 3% -- large enough to matter and invisible in the result.
    """
    strategies = sorted(set(int(v) for v in np.unique(outcomes["strategy_p0"]))
                        | set(int(v) for v in np.unique(outcomes["strategy_p1"])))
    wins = {strategy: 0 for strategy in strategies}

    decided = outcomes["winner"] >= 0
    seat0 = decided & (outcomes["winner"] == 0)
    seat1 = decided & (outcomes["winner"] == 1)

    for strategy in strategies:
        wins[strategy] = int(np.count_nonzero(seat0 & (outcomes["strategy_p0"] == strategy)))
        wins[strategy] += int(np.count_nonzero(seat1 & (outcomes["strategy_p1"] == strategy)))

    undecided = int(np.count_nonzero(~decided))
    if undecided:
        wins["undecided"] = undecided

    return wins


def _round_dtype():
    from durak.engine import ROUND_DTYPE

    return ROUND_DTYPE


def _move_dtype():
    from durak.engine import MOVE_DTYPE

    return MOVE_DTYPE