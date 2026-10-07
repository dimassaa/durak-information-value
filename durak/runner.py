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
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

from durak import rules
#: Imported, not reimplemented. The runner needs an interval for `RunSummary.wilson`, and a second
#: copy of the formula could drift from the one the reports publish -- the project's stated
#: intervals and its internal sanity checks would then be two statistics with one name.
from analysis.stats import wilson_interval
from durak.audit import audit_outcomes
from durak.config import Settings
from durak.engine import play_block
from durak.tournament import game_block_ranges
from durak.rules import (
    EARLY_FRACTION,
    LATE_FRACTION,
    RuleTunables,
    settings_from_config,
)
from durak.strategies import (
    CHEATER_DOC,
    INFO_PARTICLES,
    INFO_TRUTH,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
    InfoMode,
)


def info_modes_for(pairing) -> tuple:
    """The information mode each seat runs with, from the strategy it plays.

    This is the project's experimental variable and it was, until now, hard-coded to ``NONE`` for every
    seat -- which made ``SEARCH_UNIFORM`` and ``SEARCH_INFORMED`` the *same* policy on the same input, so
    the runner measured a difference of exactly zero between the arms of the primary comparison and would
    have reported it as a result. Deriving the modes from the strategy ids makes the difference structural:
    a new search arm has to say which mode it wants or it inherits the prior.

    ``SEARCH_INFORMED`` reads the opponent's real hand; every other strategy gets the particle set or
    nothing, as appropriate. Neither search strategy draws from a stream, so the ``rng_state`` the runner
    threads through is unused for them -- but the signature is uniform, so it is passed regardless.
    """
    def mode_for(strategy_id: int) -> int:
        if strategy_id == SEARCH_INFORMED or strategy_id == CHEATER_DOC:
            return INFO_TRUTH
        if strategy_id == SEARCH_UNIFORM:
            return INFO_PARTICLES
        return int(InfoMode.NONE)

    return mode_for(pairing.strategy_p0), mode_for(pairing.strategy_p1)
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
    """Write one run's three telemetry levels under `out_dir`.

    `out_dir` is the run root, and each level gets its own subdirectory. The levels used to nest --
    level 2 and 3 written under `games/` -- which put round and move rows inside the level-1 directory,
    so `read_parquet(out_dir / "games")` returned 1,061 rows where 48 games existed. `read_parquet`
    deliberately reads every `*.parquet` under a path recursively, so nesting levels is silently
    incompatible with reading one level. `analysis/schema.sql` globs each level's own directory, which
    only works once the levels are siblings.
    """
    write_parquet(outcomes, out_dir / "games", schema_version=schema_version, level=1)
    # `ensure_empty` on levels 2 and 3: a run that recorded no rounds or no moves still gets a zero-row
    # file for that level, because `analysis/schema.sql` globs `rounds/**` and `moves/**` and DuckDB
    # raises on a glob matching nothing. Without it, `record_rounds=False` made every view fail, so the
    # cheapest way to read a dataset was to record everything.
    write_parquet(rounds, out_dir / "rounds", schema_version=schema_version, level=2,
                  run_id=0, ensure_empty=True)
    write_parquet(moves, out_dir / "moves", schema_version=schema_version, level=3,
                  run_id=0, ensure_empty=True)


@dataclass(frozen=True)
class _BlockJob:
    """One unit of work: a contiguous range of games within one grid cell.

    Picklable by construction -- every field is a plain scalar or tuple -- which is what lets a block
    run in a worker process. Carries no output paths: the worker returns arrays, and the parent decides
    where they land, so a worker cannot write into the dataset layout it does not own.
    """

    pairing_index: int
    deck_size: int
    strategy_p0: int
    strategy_p1: int
    info_p0: int
    info_p1: int
    regime: str
    seed: int
    game_start: int
    game_count: int
    sample_modulus: int


def _run_block(job: _BlockJob, tunables, record_rounds: bool, record_moves: bool):
    """Play one block and return `(outcomes, rounds, moves)`.

    Module level, so a `ProcessPoolExecutor` can pickle it by reference. `tunables` is passed by the
    parent rather than reconstructed in the child: a `RuleTunables` rebuilt from defaults would be a
    different rule set whenever the manifest was not the default one, and the dataset would silently
    claim a configuration it did not use.
    """
    override = first_attacker_override(job.regime, job.game_start)
    if job.regime == RULE_DERIVED:
        return play_block(
            job.deck_size, job.strategy_p0, job.strategy_p1, job.info_p0, job.info_p1,
            job.seed, job.pairing_index, job.game_start, job.game_count, tunables,
            record_rounds, record_moves, run_id=0, sample_modulus=job.sample_modulus,
            first_attacker_override=override,
        )

    # ROTATION_BALANCED alternates on the ABSOLUTE game index, so the override differs within a block
    # and the block must be played one game at a time. `play_block` with a constant override would give
    # every game in the block the same seat, which quietly turns the balanced regime into a
    # single-seat regime.
    outcomes, rounds, moves = [], [], []
    for offset in range(job.game_count):
        one, r, m = play_block(
            job.deck_size, job.strategy_p0, job.strategy_p1, job.info_p0, job.info_p1,
            job.seed, job.pairing_index, job.game_start + offset, 1, tunables,
            record_rounds, record_moves, run_id=0, sample_modulus=job.sample_modulus,
            first_attacker_override=first_attacker_override(job.regime, job.game_start + offset),
        )
        outcomes.append(one)
        if r.size:
            rounds.append(r)
        if m.size:
            moves.append(m)

    round_dtype = rounds[0].dtype if rounds else _round_dtype()
    move_dtype = moves[0].dtype if moves else _move_dtype()
    return (np.concatenate(outcomes),
            np.concatenate(rounds) if rounds else np.zeros(0, dtype=round_dtype),
            np.concatenate(moves) if moves else np.zeros(0, dtype=move_dtype))


def _stamp_regime(outcomes, grid):
    """Add `first_attacker_regime` to level-1 rows, derived from their `pairing_index`.

    Added here rather than in the engine because the regime is a string and `njit` cannot write one
    into a structured row. Deriving it from `pairing_index` rather than carrying it through the call is
    what keeps the compiled loop untouched -- and `pairing_index` is a bijection onto the grid cells, so
    the derivation is exact rather than a lookup that could miss.

    Without this column a dataset holding both regimes is unreadable: nothing distinguishes a
    `RULE_DERIVED` game from a `ROTATION_BALANCED` one except its seat sequence, and a phase comparison
    across regimes silently mixes them. Schema version 2.
    """
    names = list(outcomes.dtype.names)
    if "first_attacker_regime" in names:
        return outcomes

    by_index = {p.pairing_index: p.first_attacker_regime for p in grid}
    regime_dtype = np.dtype([("first_attacker_regime", "U32")])
    regimes = np.zeros(len(outcomes), dtype=regime_dtype)
    for i in range(len(outcomes)):
        index = int(outcomes["pairing_index"][i])
        regimes["first_attacker_regime"][i] = by_index.get(index, "UNKNOWN")

    combined = np.dtype(outcomes.dtype.descr + [("first_attacker_regime", "U32")])
    merged = np.empty(len(outcomes), dtype=combined)
    for name in names:
        merged[name] = outcomes[name]
    merged["first_attacker_regime"] = regimes["first_attacker_regime"]
    return merged


def _resolve_workers(workers: int) -> int:
    """`0` means "decide at runtime", mirroring the manifest's convention, and is not zero workers."""
    if workers < 0:
        raise ValueError(f"workers must be non-negative, got {workers}")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def run_tournament(manifest: RunManifest, out_dir, workers: int = 0,
                   record_rounds: bool = True, record_moves: bool = True) -> RunSummary:
    """Play every cell of the grid and write the dataset.

    `workers` selects a process pool over whole blocks. It does not change the dataset, and that is not
    a hope but a property of the decomposition: a job is `(pairing_index, game_start, game_count)` and a
    game's seed is derived from `(run_seed, pairing_index, game_index)` and nothing else, so a block
    plays the same games on any worker in any order. The pool returns arrays to the parent, which
    concatenates them in grid-then-block order, so even the row order is fixed.

    Why processes and not `prange`
    ------------------------------
    `numba` 0.68 does not lower writes into a structured array inside a parallel loop, and the engine's
    telemetry is structured arrays written from inside the game loop (21 such writes in `_play_one`).
    Parallelism is therefore taken one level up rather than by redesigning the telemetry layout: blocks
    stay independent, and each writes its own Parquet partition, so there is no shared mutable state for
    a race to corrupt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workers < 0:
        raise ValueError(f"workers must be non-negative, got {workers}")

    grid = manifest.grid
    if not grid:
        # A zero-game run that reports `clean: True` is the worst possible outcome: it looks like a
        # successful run of an empty experiment. Every tournament needs at least one pairing, and the
        # single-strategy case cannot produce one unless the strategy is `RANDOM`.
        raise ValueError(
            f"the manifest for run {manifest.run_id!r} produced an empty pairing grid: strategies "
            f"{manifest.strategies} generate no matchups. Include at least two strategies, or `RANDOM`."
        )
    tunables = manifest.tunables
    started = time.perf_counter()

    all_outcomes = []
    all_rounds = []
    all_moves = []

    # One job per block, in grid order then block order. The order is the row order of the finished
    # dataset, so a pool that completes blocks out of order still concatenates deterministically.
    jobs = []
    for pairing in grid:
        info_p0, info_p1 = info_modes_for(pairing)
        for start, stop in game_block_ranges(manifest.games_per_pairing, manifest.block_size):
            jobs.append(_BlockJob(
                pairing_index=pairing.pairing_index, deck_size=pairing.deck_size,
                strategy_p0=pairing.strategy_p0, strategy_p1=pairing.strategy_p1,
                info_p0=int(info_p0), info_p1=int(info_p1),
                regime=pairing.first_attacker_regime, seed=manifest.seed,
                game_start=start, game_count=stop - start,
                sample_modulus=manifest.settings.sample_modulus,
            ))

    all_outcomes = []
    all_rounds = []
    all_moves = []

    if not jobs:
        (out_dir / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
        return RunSummary(games=0, wins={}, anomalies=0, audit={"games": 0, "clean": True},
                          elapsed_seconds=time.perf_counter() - started)

    worker_count = _resolve_workers(workers)
    if worker_count == 1 or len(jobs) == 1:
        # Serial on purpose for one worker: a pool of one pays process startup and pickling for no
        # parallelism, and this is the path the tests and every interactive run take.
        results = [_run_block(job, tunables, record_rounds, record_moves) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_run_block, job, tunables, record_rounds, record_moves)
                       for job in jobs]
            # Collected by index, not by completion: `as_completed` would make the row order depend on
            # which worker happened to finish first, and the dataset must not depend on scheduling.
            results = [f.result() for f in futures]

    for outcomes, rounds, moves in results:
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

    outcomes = _stamp_regime(outcomes, grid)

    _write_levels(outcomes, rounds, moves, out_dir, 1, manifest.schema_version)

    (out_dir / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
    # A different document under a different name. `write_manifest` inventories the dataset as written
    # (partitions, observed columns, versions), and it used to be called on `out_dir / "games"` so the two
    # never collided. Writing it to `out_dir` alongside the run manifest overwrote `manifest.json` with
    # the inventory -- which has no `run_id`, so the reproducibility contract was destroyed by the very
    # call meant to describe the data.
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