"""The tournament grid, the reproducibility manifest, and the runner.

The runner's job is to make a run reproducible from its own metadata, so the tests are about
identity and invariance rather than about win rates:

* a given pairing must occur exactly once per direction, and `RANDOM` must be paired with itself
  -- that self-pairing is what the 0.5 symmetry check exists to consume, and omitting it would
  leave the project's only exact assertion with nothing to run on;
* the same `(run_seed, pairing_index)` must give the same seeds on any machine, in any order;
* the manifest must round-trip through JSON with every field intact, because a manifest that
  loses a field is a run nobody can reproduce.

On worker counts: `run_tournament` accepts `workers` and Task 15 makes it mean something. Until
then it is honoured only as a validation input, so the invariance test below is currently
satisfied trivially. That is stated in the test rather than hidden, and the equivalent test over
*block sizes* is real today -- block boundaries are what actually partition the work.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from durak.config import Settings
from durak.rules import EMPTY, USE_RULE, acting_player, check_invariants
from durak.runner import RunManifest, RunSummary, run_tournament
from durak.strategies import AUTO_DEFEND, GREEDY, RANDOM, InfoMode, StrategyId
from durak.tournament import (
    REGIMES,
    RULE_DERIVED,
    ROTATION_BALANCED,
    Pairing,
    first_attacker_override,
    game_seeds,
    pairing_grid,
)

DECK = 36


# --- pairing grid ------------------------------------------------------------


def test_grid_covers_each_pairing_once_per_direction():
    """Both directions, because seat is part of what is being measured.

    Running only `A` vs `B` and not `B` vs `A` would confound the two strategies' relative
    strength with the first-attacker advantage, which is exactly the confound spec 8.4 introduces
    the rotation-balanced regime to break. Restricted to one regime here so the cell count is the
    pairing count; the regime axis is pinned separately.
    """
    grid = pairing_grid([RANDOM, GREEDY], regimes=(RULE_DERIVED,))

    assert len(grid) == 3, "expected RANDOM-self plus both directions of RANDOM vs GREEDY"
    assert set((p.strategy_p0, p.strategy_p1) for p in grid) == {
        (RANDOM, RANDOM), (RANDOM, GREEDY), (GREEDY, RANDOM),
    }


def test_grid_is_unique_per_deck_and_regime():
    grid = pairing_grid([RANDOM, GREEDY, AUTO_DEFEND], deck_sizes=(36, 52))

    keys = {(p.strategy_p0, p.strategy_p1, p.deck_size, p.first_attacker_regime) for p in grid}

    assert len(keys) == len(grid), "the grid contains a duplicate cell"


def test_grid_includes_random_against_itself():
    """The symmetry check needs `RANDOM` vs `RANDOM`, so the self-pairing must exist.

    It is the only self-pairing: no other strategy is its own opponent, and adding one would mean
    running a matchup whose result the report has no use for.
    """
    grid = pairing_grid([RANDOM, GREEDY], regimes=(RULE_DERIVED,))

    self_pairings = [p for p in grid if p.strategy_p0 == p.strategy_p1]

    assert len(self_pairings) == 1
    assert self_pairings[0].strategy_p0 == RANDOM
    # Exactly one form: seat is already covered by playing the game, and emitting the self-pairing
    # reversed would double the sample of the symmetry check for no added information.
    assert self_pairings[0].strategy_p1 == RANDOM


def test_grid_omits_other_self_pairings():
    grid = pairing_grid([RANDOM, GREEDY, AUTO_DEFEND])

    for pairing in grid:
        if pairing.strategy_p0 == pairing.strategy_p1:
            assert pairing.strategy_p0 == RANDOM, (
                f"{pairing.strategy_p0} was paired against itself, which nothing consumes"
            )


def test_grid_covers_both_regimes():
    grid = pairing_grid([RANDOM, GREEDY])

    assert {p.first_attacker_regime for p in grid} == set(REGIMES)


def test_pairing_indices_are_dense_and_ordered():
    """`pairing_index` is a seed coordinate, so it must be a stable ordinal.

    A gap or a reordering between two runs of the same configuration would change every seed
    derived from it and silently produce a different dataset.
    """
    grid = pairing_grid([RANDOM, GREEDY])

    assert [p.pairing_index for p in grid] == list(range(len(grid)))


def test_grid_is_deterministic():
    assert pairing_grid([RANDOM, GREEDY, AUTO_DEFEND]) == \
        pairing_grid([RANDOM, GREEDY, AUTO_DEFEND])


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        pairing_grid([RANDOM, 99])


# --- seeds -------------------------------------------------------------------


def test_seeds_are_identical_for_the_same_coordinates():
    """The whole reproducibility contract, in one assertion."""
    first = game_seeds(7, 3, 100)
    second = game_seeds(7, 3, 100)

    assert np.array_equal(first, second)


def test_seeds_differ_across_pairings():
    a = game_seeds(7, 0, 100)
    b = game_seeds(7, 1, 100)

    assert len(np.unique(np.concatenate([a, b]))) == 200


def test_seeds_differ_across_runs():
    assert not np.array_equal(game_seeds(7, 0, 100), game_seeds(8, 0, 100))


def test_seeds_are_distinct_within_a_pairing():
    seeds = game_seeds(7, 0, 5_000)

    assert len(np.unique(seeds)) == 5_000


def test_seeds_fit_int64():
    """Every seed must fit the column it is written to.

    A uint64 seed written into an int64 column wraps silently when compiled and raises when
    interpreted -- the third site of that defect. Asserted here so the property is pinned at the
    point it is produced rather than discovered at the writer.
    """
    seeds = game_seeds(2**31 - 1, 2**31 - 1, 500)

    assert seeds.dtype == np.int64
    assert np.all(seeds < 2**63)
    assert np.all(seeds != np.iinfo(np.int64).min)


# --- first-attacker regimes --------------------------------------------------


def test_rule_derived_defers_to_the_rules():
    for game_index in (0, 1, 2, 3, 17):
        assert first_attacker_override(RULE_DERIVED, game_index) == USE_RULE


def test_rotation_balanced_alternates():
    overrides = [first_attacker_override(ROTATION_BALANCED, i) for i in range(20)]

    assert all(o in (0, 1) for o in overrides), "a rotation override named a non-seat"
    assert overrides[:4] == [0, 1, 0, 1]
    assert overrides.count(0) == overrides.count(1) == 10


def test_rotation_balanced_is_exactly_balanced_over_an_even_count():
    """Half the games each seat, so a positional effect cannot hide in the dealing."""
    for count in (2, 100, 1_000):
        overrides = [first_attacker_override(ROTATION_BALANCED, i) for i in range(count)]
        assert overrides.count(0) == count // 2


def test_the_sentinel_is_not_a_seat():
    """A sentinel of 0 would silently disable the rule for half of every rotation run."""
    assert USE_RULE not in (0, 1)
    assert first_attacker_override(RULE_DERIVED, 0) == USE_RULE


def test_rotation_balanced_actually_forces_the_seat_in_play():
    """The override must reach the engine, not just the pairing record.

    Checked by playing, not by inspecting a function: an override the engine ignored would leave
    every regime-2 row mislabelled and the positional metric meaningless.
    """
    from durak.engine import play_block
    from durak import rules

    from durak.engine import play_block

    # Resolved per game, exactly as the runner does: alternating on the *absolute* game index is
    # what makes the rotation hold under any block partitioning.
    counts = {0: 0, 1: 0}
    for game_index in range(200):
        outcomes, _, _ = play_block(
            DECK, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 5, 0, game_index, 1,
            rules.DEFAULT_SETTINGS, record_rounds=False, record_moves=False,
            first_attacker_override=first_attacker_override(ROTATION_BALANCED, game_index),
        )
        counts[int(outcomes["first_attacker"][0])] += 1

    assert counts == {0: 100, 1: 100}

    # And under the rule-derived regime the same games must NOT be balanced, otherwise the flag
    # would be doing nothing and the two regimes would be indistinguishable.
    derived = {0: 0, 1: 0}
    for game_index in range(200):
        outcomes, _, _ = play_block(
            DECK, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 5, 0, game_index, 1,
            rules.DEFAULT_SETTINGS, record_rounds=False, record_moves=False,
            first_attacker_override=first_attacker_override(RULE_DERIVED, game_index),
        )
        derived[int(outcomes["first_attacker"][0])] += 1

    assert derived[0] != 100 or derived[1] != 100, (
        "the rule-derived regime came out exactly balanced, which means the override flag is "
        "being applied unconditionally"
    )


# --- manifest ----------------------------------------------------------------


def test_manifest_round_trips_through_json():
    """Every field, or the run is not reproducible.

    The reproducibility contract is the manifest. A field that does not survive a JSON round trip
    is a field nobody can use to rebuild the dataset, and the failure surfaces months later as a
    question that cannot be answered.
    """
    manifest = RunManifest(
        run_id="study-2026-10",
        seed=12345,
        strategies=[RANDOM, GREEDY],
        deck_sizes=[36],
        games_per_pairing=100,
        regimes=[RULE_DERIVED],
        settings=Settings(),
        block_size=500,
        engine_version=1,
        rules_version=1,
        schema_version=1,
    )

    restored = RunManifest.from_dict(json.loads(manifest.to_json()))

    assert restored == manifest


def test_manifest_records_every_setting_that_can_change_a_result():
    """The settings the manifest must carry, named explicitly rather than by iterating the dataclass.

    Spec 7 and 8 depend on these: a sweep over `throwin_defense_ranks` or `soft_penalty_lambda`
    that was not recorded cannot be interpreted later, because the two arms cannot be told apart.
    """
    manifest = RunManifest(
        run_id="r", seed=0, strategies=[RANDOM], deck_sizes=[36], games_per_pairing=1,
        regimes=[RULE_DERIVED], settings=Settings(), block_size=1,
        engine_version=1, rules_version=1, schema_version=1,
    )
    recorded = json.loads(manifest.to_json())["settings"]

    for name in ("throwin_defense_ranks", "soft_penalty_lambda", "particle_count",
                 "sample_modulus", "deck_size", "games_per_pairing", "seed"):
        assert name in recorded, f"{name} is not recorded in the manifest"


def test_manifest_records_the_phase_cut():
    """The phase cut is a pre-registered choice and must be in the manifest, not in a comment."""
    recorded = json.loads(RunManifest(
        run_id="r", seed=0, strategies=[RANDOM], deck_sizes=[36], games_per_pairing=1,
        regimes=[RULE_DERIVED], settings=Settings(), block_size=1,
        engine_version=1, rules_version=1, schema_version=1,
    ).to_json())

    assert "phase_cut" in recorded
    assert recorded["phase_cut"]["late_fraction"] == pytest.approx(0.15)
    assert recorded["phase_cut"]["early_fraction"] == pytest.approx(0.50)


def test_manifest_has_a_timestamp_that_is_not_part_of_equality():
    """A timestamp makes two identical runs differ, so it is excluded from comparison.

    Kept in the file because a dataset with no generation date cannot be cited, and excluded from
    equality because otherwise no reproducibility check could ever compare two manifests.
    """
    first = RunManifest(run_id="r", seed=0, strategies=[RANDOM], deck_sizes=[36],
                        games_per_pairing=1, regimes=[RULE_DERIVED], settings=Settings(),
                        block_size=1, engine_version=1, rules_version=1, schema_version=1)
    second = RunManifest(run_id="r", seed=0, strategies=[RANDOM], deck_sizes=[36],
                         games_per_pairing=1, regimes=[RULE_DERIVED], settings=Settings(),
                         block_size=1, engine_version=1, rules_version=1, schema_version=1)

    assert first == second, "two identical runs produced unequal manifests"
    assert "generated_at" in json.loads(first.to_json())


def test_manifest_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        RunManifest(run_id="r", seed=0, strategies=[99], deck_sizes=[36], games_per_pairing=1,
                     regimes=[RULE_DERIVED], settings=Settings(), block_size=1,
                     engine_version=1, rules_version=1, schema_version=1)


# --- the runner --------------------------------------------------------------


def _manifest(games=200, strategies=(RANDOM, GREEDY), block_size=100, **kwargs):
    return RunManifest(
        run_id="test", seed=99, strategies=list(strategies), deck_sizes=[DECK],
        games_per_pairing=games, regimes=[RULE_DERIVED], settings=Settings(),
        block_size=block_size, engine_version=1, rules_version=1, schema_version=1, **kwargs
    )


def test_a_tournament_writes_every_level(tmp_path):
    summary = run_tournament(_manifest(games=40, strategies=(RANDOM,)), tmp_path)

    assert summary.games == 40
    assert summary.anomalies == 0
    assert (tmp_path / "manifest.json").exists()
    for name in ("games", "rounds", "moves"):
        assert list((tmp_path / name).rglob("*.parquet")), f"{name} produced no files"


def test_random_against_random_is_approximately_half(tmp_path):  # noqa: C901
    """Spec 7.3's only exact assertion, on a real run through the runner.

    Wilson interval, not a point comparison: with 200 games the sampling error is about 7
    percentage points, so an exact equality test would be a test of luck. The interval *containing*
    0.5 is the real claim -- a seat bias, a shared stream, or a non-uniform deal would push it out.
    """
    summary = run_tournament(_manifest(games=2_000, strategies=(RANDOM,)), tmp_path)

    decided = summary.seat_wins[0] + summary.seat_wins[1]
    assert decided == 2_000, f"expected 2000 decided games, got {decided}"
    assert summary.undecided == 0, (
        f"{summary.undecided} games hit the round cap; the interval below would be computed over "
        f"a subset and the check would no longer mean what it says"
    )

    games = summary.seat_wins[0]
    rate = summary.seat_rate(0)
    low, high = summary.wilson(games, decided)

    assert low <= 0.5 <= high, (
        f"seat-0 rate {rate:.4f} with Wilson interval [{low:.4f}, {high:.4f}] excludes 0.5, "
        f"which indicates position, dealing, or seeding is biased"
    )
    assert 0.4 < rate < 0.6


def test_the_same_run_twice_is_byte_identical(tmp_path):
    """The reproducibility guarantee, end to end through files on disk."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_tournament(_manifest(games=60, block_size=20), first)
    run_tournament(_manifest(games=60, block_size=20), second)

    for level in ("games", "rounds", "moves"):
        left = sorted(p.read_bytes() for p in (first / level).rglob("*.parquet"))
        right = sorted(p.read_bytes() for p in (second / level).rglob("*.parquet"))
        assert left == right, f"{level} differ between two identical runs"


def test_block_boundaries_do_not_change_the_dataset(tmp_path):
    """The invariance that is real today.

    A game is identified by `(seed, pairing_index, game_index)` and nothing else, so how the work
    is divided cannot change the result. This is the property Task 15's `prange` port must
    preserve, tested now with the one partition boundary that exists today.
    """
    coarse = tmp_path / "coarse"
    fine = tmp_path / "fine"

    run_tournament(_manifest(games=60, block_size=60), coarse)
    run_tournament(_manifest(games=60, block_size=13), fine)

    for level in ("games", "rounds"):
        left = sorted(p.read_bytes() for p in (coarse / level).rglob("*.parquet"))
        right = sorted(p.read_bytes() for p in (fine / level).rglob("*.parquet"))
        # Different block sizes write different file layouts, so compare contents rather than
        # bytes. Equality of the concatenated rows is the claim.
        from durak.telemetry import read_parquet

        assert read_parquet(coarse / level).to_pandas().equals(
            read_parquet(fine / level).to_pandas()
        ), f"{level} differ when the work is partitioned differently"
        assert left and right


def test_worker_count_does_not_change_the_dataset(tmp_path):
    """Accepted now, meaningful from Task 15.

    Stated plainly rather than presented as a stronger guarantee than it is: with serial-only
    execution, two different `workers` values run the same code. The assertion is real from
    Task 15, when `workers` selects a `prange` partitioning.
    """
    one = tmp_path / "w1"
    eight = tmp_path / "w8"

    run_tournament(_manifest(games=40, block_size=40), one, workers=1)
    run_tournament(_manifest(games=40, block_size=40), eight, workers=8)

    from durak.telemetry import read_parquet

    assert read_parquet(one / "games").to_pandas().equals(
        read_parquet(eight / "games").to_pandas()
    )


def test_the_summary_counts_every_game_once(tmp_path):
    manifest = _manifest(games=30, strategies=(RANDOM, GREEDY, AUTO_DEFEND))
    summary = run_tournament(manifest, tmp_path)

    # Two directions per unordered pairing, plus the single RANDOM self-pairing; no other
    # self-pairing exists, which the grid test pins separately.
    pairings = len(manifest.strategies) * (len(manifest.strategies) - 1) + 1
    assert summary.games == pairings * len(manifest.regimes) * manifest.games_per_pairing


def test_an_audited_summary_reports_no_anomalies(tmp_path):
    """Spec 7.2: the anomaly count travels with every run, not only the ones someone inspected."""
    # `RANDOM` alone: `GREEDY` caps about 3% of games by measurement, and a capped game is a
    # legitimate anomaly, so a "no anomalies" assertion over it would be asserting something false.
    summary = run_tournament(_manifest(games=200, strategies=(RANDOM,)), tmp_path)

    assert summary.anomalies == 0
    assert summary.audit["clean"] is True
    assert summary.audit["games"] == summary.games


def test_the_manifest_is_written_next_to_the_data(tmp_path):
    """A dataset with no manifest beside it is a dataset nobody can reproduce."""
    manifest = _manifest(games=10)
    run_tournament(manifest, tmp_path)

    on_disk = json.loads((tmp_path / "manifest.json").read_text())

    assert on_disk["run_id"] == manifest.run_id
    assert on_disk["seed"] == manifest.seed
    assert on_disk["strategies"] == manifest.strategies
    assert on_disk["games_per_pairing"] == manifest.games_per_pairing
    assert on_disk["settings"]["sample_modulus"] == manifest.settings.sample_modulus


def test_an_empty_run_writes_nothing_and_says_so(tmp_path):
    """Zero games must not leave a dataset that a later read treats as real."""
    summary = run_tournament(_manifest(games=0), tmp_path)

    assert summary.games == 0
    assert not list(tmp_path.rglob("*.parquet"))