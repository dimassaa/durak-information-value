"""The analysis queries, checked against a dataset whose answers are known in advance.

A synthetic dataset with an exactly known win rate is the only kind that can test an aggregation. A
real run's win rate is whatever it is, so an assertion against it is either circular or a change
detector. Here 60 of 100 games are wins, and the emitted JSON must contain exactly that, with a Wilson
interval matching `analysis.stats.wilson_interval(60, 100)` to full precision -- full precision being
the point, because rounding inside the arithmetic makes the interval disagree with the function it
claims to have called.

Rounding happens at serialisation and nowhere else. A report that rounds a delta before testing it
against zero can cross the line from "not significant" to "significant", and there is no way to see
that from the JSON.
"""

import json

import numpy as np
import pytest

from analysis.stats import mcnemar_exact, wilson_interval
from durak.strategies import RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM
from analysis.queries import (
    build_all,
    data_quality,
    opponent_independent_strength,
    price_of_information,
    rules_and_dealing,
    tournament_overview,
)


def _synthetic_dataset(tmp_path, wins_p0=60, total=100, flags=0, opponent="RANDOM"):
    """A level-1-only dataset with an exactly known rate, written the way the runner writes it.

    Both arms play the SAME `game_index` values, which is what makes the comparison paired: game `i` of
    the uninformed arm and game `i` of the informed arm are the same deal with the same seed. Writing
    only one arm produces a dataset where the pairing cannot be formed at all, and the query under test
    would be returning `None` for reasons that have nothing to do with the query.

    Written directly rather than by running games, because the point is to know the answer before the
    query runs. Levels 2 and 3 are written empty so the views load without the real engine being involved.
    """
    from durak.engine import GAME_DTYPE
    from durak.telemetry import write_parquet
    from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

    # The runner stamps `first_attacker_regime` after play, so the engine dtype lacks it. The fixture
    # builds the published shape directly rather than going through the engine, and so must carry the
    # column itself.
    base = GAME_DTYPE
    records = np.zeros(2 * total, dtype=np.dtype(base.descr + [("first_attacker_regime", "U32")]))
    opponent_id = 2 if opponent == "RANDOM" else 3
    for arm, strategy in ((SEARCH_UNIFORM, 5), (SEARCH_INFORMED, 6)):
        for i in range(total):
            row = i if strategy == SEARCH_UNIFORM else total + i
            won = i < wins_p0
            records[row]["run_id"] = 0
            records[row]["game_index"] = i
            records[row]["game_seed"] = 1000 + i
            records[row]["deck_size"] = 36
            records[row]["strategy_p0"] = strategy
            records[row]["strategy_p1"] = opponent_id
            records[row]["info_mode_p0"] = 1 if strategy == SEARCH_INFORMED else 0
            records[row]["info_mode_p1"] = 0
            records[row]["engine_version"] = ENGINE_VERSION
            records[row]["rules_version"] = RULES_VERSION
            records[row]["schema_version"] = SCHEMA_VERSION
            records[row]["first_attacker"] = i % 2
            records[row]["winner"] = 0 if won else 1
            records[row]["rounds"] = 30
            records[row]["actions"] = 80
            records[row]["final_hand_p0"] = 0 if won else 3
            records[row]["final_hand_p1"] = 3 if won else 0
            records[row]["deck_remaining_at_end"] = 4
            records[row]["first_attacker_regime"] = "RULE_DERIVED"
            records[row]["pairing_index"] = 0
            records[row]["both_have_trumps"] = True
            records[row]["flag_redeals_exceeded"] = False
            records[row]["winner_rounds_played"] = 30
            if i < flags:
                records[row]["winner"] = -1          # an undecided game, which the audit must count

    write_parquet(records, tmp_path / "games", level=1)
    from durak.telemetry import write_parquet as wp
    wp(np.zeros(0, dtype=__import__("durak.engine", fromlist=["ROUND_DTYPE"]).ROUND_DTYPE),
       tmp_path / "rounds", level=2, run_id=0, ensure_empty=True)
    wp(np.zeros(0, dtype=__import__("durak.engine", fromlist=["MOVE_DTYPE"]).MOVE_DTYPE),
       tmp_path / "moves", level=3, run_id=0, ensure_empty=True)
    return tmp_path


# --- price of information ------------------------------------------------------


def test_price_of_information_reports_a_known_rate_with_the_matching_interval(tmp_path):
    """60 of 100 wins must come out as 0.6 and carry exactly `wilson_interval(60, 100)`."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = price_of_information(root)

    assert result["informed"]["n"] == 100
    assert result["informed"]["win_rate"] == pytest.approx(0.6, abs=1e-12)
    assert result["informed"]["ci"] == pytest.approx(wilson_interval(60, 100), abs=1e-12)


def test_price_of_information_carries_its_paired_tests(tmp_path):
    """The primary result is a PAIRED comparison, so the p-value and the adjusted p-value must travel with it."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = price_of_information(root)

    assert "mcnemar" in result["delta"]
    assert "adjusted_p" in result["delta"]
    assert 0.0 <= result["delta"]["mcnemar"] <= 1.0
    assert 0.0 <= result["delta"]["adjusted_p"] <= 1.0


def test_the_interval_is_not_rounded_before_it_is_serialised(tmp_path):
    """The published interval must be the function's, not a 4-digit rendering of it."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = price_of_information(root)
    expected = wilson_interval(60, 100)
    assert result["informed"]["ci"][0] == pytest.approx(expected[0], abs=0.0)
    assert result["informed"]["ci"][1] == pytest.approx(expected[1], abs=0.0)


def test_an_undecided_game_leaves_the_denominator(tmp_path):
    """`winner = -1` has no outcome. Counting it as a loss would invent a result."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100, flags=5)
    result = price_of_information(root)
    assert result["informed"]["n"] == 95, "undecided games must not enter the denominator"
    # Five per arm, and the flag is applied to both arms' copy of each flagged game, so ten in all.
    assert result["undecided_games"] == 10


# --- data quality ---------------------------------------------------------------


def test_data_quality_is_clean_for_a_clean_dataset(tmp_path):
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = data_quality(root)
    assert result["anomalies"] == 0
    assert result["clean"] is True


def test_data_quality_counts_a_flagged_game(tmp_path):
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100, flags=5)
    result = data_quality(root)
    assert result["anomalies"] > 0
    assert result["clean"] is False


# --- the other three ------------------------------------------------------------


def test_tournament_overview_lists_every_strategy_with_a_rate(tmp_path):
    root = _synthetic_dataset(tmp_path)
    overview = tournament_overview(root)
    ids = {row["strategy_id"] for row in overview["rows"]}
    assert ids == {2, 5, 6}   # both arms plus RANDOM
    row = next(r for r in overview["rows"] if r["strategy_id"] == 5)
    assert row["win_rate"] == pytest.approx(0.6, abs=1e-12)
    assert row["n"] == 100, "each strategy played 100 games as strategy_p0"


def test_opponent_independent_strength_uses_only_random_opponents(tmp_path):
    """The metric that cannot be distorted by the composition of the field.

    A strategy's rate against a mixed field changes when another strategy joins the tournament, so it
    cannot be compared across studies with different palettes. Against `RANDOM` only, it can.
    """
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100, opponent="GREEDY")
    result = opponent_independent_strength(root)
    assert result["rows"] == [], "a dataset with no RANDOM opponent yields no such rows"
    assert result["opponent"] == 2


def test_rules_and_dealing_reports_the_first_attacker_split(tmp_path):
    root = _synthetic_dataset(tmp_path)
    result = rules_and_dealing(root)
    assert result["deck_sizes"] == [36]
    assert result["regimes"] == ["RULE_DERIVED"]
    assert result["first_attacker_p0_wins"] + result["first_attacker_p1_wins"] == 200


# --- build_all ------------------------------------------------------------------


def test_build_all_writes_one_json_per_report_plus_meta(tmp_path):
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    out = tmp_path / "json"
    written = build_all(root, out)

    names = {p.name for p in written}
    assert "meta.json" in names
    for report in ("price_of_information", "tournament_overview", "rules_and_dealing",
                   "data_quality", "opponent_independent_strength"):
        assert f"{report}.json" in names
        # Every file must be valid JSON: a report that writes prose where a number belongs fails at the
        # first figure, which is after the dataset is already built.
        json.loads((out / f"{report}.json").read_text())


def test_meta_round_trips_the_settings_block_exactly(tmp_path):
    """The settings are the reproducibility contract. A float rounded in JSON cannot rebuild the run."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    out = tmp_path / "json"
    build_all(root, out)
    meta = json.loads((out / "meta.json").read_text())

    assert meta["engine_version"] is not None
    assert meta["rules_version"] is not None
    assert meta["schema_version"] is not None
    assert meta["games"] == 200
    assert "settings" in meta


def test_every_emitted_number_is_finite(tmp_path):
    """A NaN serialises as `NaN` in JSON, which most parsers reject and every chart mishandles."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    out = tmp_path / "json"
    build_all(root, out)

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, float):
            yield node

    for path in out.glob("*.json"):
        for value in walk(json.loads(path.read_text())):
            assert np.isfinite(value), f"{path.name} contains a non-finite number: {value}"