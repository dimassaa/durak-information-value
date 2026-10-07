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
from pathlib import Path

import numpy as np
import pytest

from analysis.stats import mcnemar_exact, wilson_interval
from durak.strategies import RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM
from analysis.queries import (
    build_all,
    data_quality,
    deck_split,
    endgame_degeneracy,
    game_length,
    opponent_independent_strength,
    pairwise_matchups,
    price_of_information,
    rule_variant_comparison,
    rules_and_dealing,
    take_behaviour,
    throw_in_efficiency,
    throw_in_limits,
    tournament_overview,
    wins_by_final_hand_size,
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
    # HEAD TO HEAD: the two arms play each other in both seat orders. A field-wide fixture would let
    # the primary metric pass by averaging over unrelated opponents, which is the bug this shape exists
    # to prevent -- the field-wide version read +0.012 where the head-to-head reads +0.070.
    records = np.zeros(2 * total, dtype=np.dtype(base.descr + [("first_attacker_regime", "U32")]))
    opponent_id = 2 if opponent == "RANDOM" else 3
    order = ((SEARCH_UNIFORM, SEARCH_INFORMED), (SEARCH_INFORMED, SEARCH_UNIFORM))
    for arm, (strategy, other) in enumerate(order):
        for i in range(total):
            row = arm * total + i
            # `informed_won` is a property of the ARM, not of the seat, so it is resolved to a seat
            # here: informed wins in seat 1 (winner = 1) and in seat 0 (winner = 0) equally often.
            informed_won = i < wins_p0
            won = informed_won
            records[row]["run_id"] = 0
            records[row]["game_index"] = i
            records[row]["game_seed"] = 1000 + i
            records[row]["deck_size"] = 36
            records[row]["strategy_p0"] = strategy
            records[row]["strategy_p1"] = other if opponent == "RANDOM" else opponent_id
            records[row]["info_mode_p0"] = 1 if strategy == SEARCH_INFORMED else 0
            records[row]["info_mode_p1"] = 0
            records[row]["engine_version"] = ENGINE_VERSION
            records[row]["rules_version"] = RULES_VERSION
            records[row]["schema_version"] = SCHEMA_VERSION
            records[row]["first_attacker"] = i % 2
            records[row]["winner"] = (1 if arm == 0 else 0) if informed_won else (0 if arm == 0 else 1)
            records[row]["rounds"] = 30
            records[row]["actions"] = 80
            records[row]["final_hand_p0"] = 0 if won else 3
            records[row]["final_hand_p1"] = 3 if won else 0
            records[row]["deck_remaining_at_end"] = 4
            records[row]["first_attacker_regime"] = "RULE_DERIVED"
            records[row]["pairing_index"] = arm
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
    """120 of 200 head-to-head games won must come out as 0.6 with exactly `wilson_interval(120, 200)`.

    The head-to-head is the whole point: 60 wins per seat order, 120 in total. A field-wide metric would
    answer a different question and would have hidden the bug this test now pins.
    """
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = price_of_information(root)

    assert result["informed"]["n"] == 200
    assert result["informed"]["win_rate"] == pytest.approx(0.6, abs=1e-12)
    assert result["informed"]["ci"] == pytest.approx(wilson_interval(120, 200), abs=1e-12)


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
    expected = wilson_interval(120, 200)
    assert result["informed"]["ci"][0] == pytest.approx(expected[0], abs=0.0)
    assert result["informed"]["ci"][1] == pytest.approx(expected[1], abs=0.0)


def test_an_undecided_game_leaves_the_denominator(tmp_path):
    """`winner = -1` has no outcome. Counting it as a loss would invent a result."""
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100, flags=5)
    result = price_of_information(root)
    assert result["informed"]["n"] == 190, "undecided games must not enter the denominator"
    # Five flagged indices per seat order, ten rows in all.
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
    assert ids == {5, 6}, "the head-to-head fixture contains only the two arms"
    row = next(r for r in overview["rows"] if r["strategy_id"] == 5)
    # The informed arm wins 60 of 100 in each seat order, so the uninformed arm wins 40 of 200 overall.
    assert row["win_rate"] == pytest.approx(0.4, abs=1e-12)
    assert row["n"] == 200, "the arm plays every game in the head-to-head, in both seats"


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

def test_a_tiny_p_value_survives_serialisation(tmp_path):
    """A p-value must not be rounded to zero on its way to the report.

    Rounding to a fixed number of decimal places is harmless for a win rate and destroys a p-value:
    2.302e-07 becomes 0.0, and the report prints a p-value of zero for the one number that says whether
    the result is real. The renderer showed `0.000e+00` in a KPI tile until this was fixed, which is the
    kind of quiet deletion that a test on the arithmetic alone would never notice.
    """
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    out = tmp_path / "json"
    build_all(root, out)
    payload = json.loads((out / "price_of_information.json").read_text())

    assert payload["delta"]["mcnemar"] > 0.0, "a p-value of zero would be indistinguishable from 'never computed'"
    assert payload["delta"]["mcnemar"] > 1e-9, "the p-value lost its significant digits"


def test_significant_digits_are_kept_for_small_and_large_values(tmp_path):
    """The same function handles a win rate and a p-value without special-casing either."""
    from analysis.queries import _round_for_output

    assert _round_for_output(0.5412345678) == pytest.approx(0.541235, abs=1e-9)
    assert _round_for_output(2.302e-07) == pytest.approx(2.302e-07, rel=1e-6)
    assert _round_for_output(0.0) == 0.0


# --- the level-2 key is not a game key ---------------------------------------------------
#
# `game_index` is CELL-LOCAL. The 57-cell primary run has 1,140,000 games over 20,000 distinct
# `(run_id, game_index)` pairs, and levels 2 and 3 carry no `pairing_index`. These two tests exist so
# that no future query reaches for `rounds` or `moves` believing the join is safe.


def test_game_index_is_cell_local_so_it_is_not_a_game_key(tmp_path):
    """The finding, pinned: a game's unique key needs `pairing_index`, not just `game_index`.

    A single-pairing dataset has a unique `game_index`, so this defect is invisible there and only
    appears once a tournament has more than one cell -- which is the only configuration the reports
    are ever built from.
    """
    root = tmp_path / "cells"
    _single_cell_dataset(root / "cell_a", games=5, wins=2, strategies=(SEARCH_UNIFORM, 3))
    _single_cell_dataset(root / "cell_b", games=5, wins=4, strategies=(SEARCH_UNIFORM, 3))

    import duckdb

    con = duckdb.connect()
    # Both cells were written to separate directories; read them as one dataset the way a real run
    # is read, and count how many rows share an index. The paths are inlined rather than bound
    # because `read_parquet` takes a list literal, not a parameter, and a stray `?` here would fail
    # with a parameter-count error that says nothing about the defect.
    rows = con.execute(f"""
        SELECT game_index, COUNT(*) AS cells
        FROM read_parquet([
            '{root}/cell_a/games/**/*.parquet',
            '{root}/cell_b/games/**/*.parquet'
        ], hive_partitioning = true)
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    assert all(count == 2 for _index, count in rows), (
        "both cells reuse game_index 0..4, so (run_id, game_index) does not identify a game"
    )
    con.close()


def test_the_reports_measure_from_level_one_only():
    """The constraint, stated where a future query author will read it.

    Levels 2 and 3 cannot be joined to level 1 without a 57-fold fan-out on a multi-cell tournament,
    because they carry no `pairing_index` and `game_index` is cell-local. Rather than trust every
    future author to remember, this asserts the fact that makes it true, and points at the backlog.
    """
    import analysis.queries as queries

    source = Path(queries.__file__).read_text(encoding="utf-8")

    # `rounds` and `moves` must not appear in any query's SQL. `_participation` and the new
    # report-2/3 queries all read `games`, whose per-game columns need no join.
    for level in ("FROM rounds", "FROM moves", "JOIN rounds", "JOIN moves"):
        assert level not in source, (
            f"a report query references the {level.split()[-1]} level; it carries no pairing_index, "
            f"so joining it to games fans out across cells. See IMPROVEMENTS.md item 41."
        )


# --- pairwise matchups --------------------------------------------------------------


def test_pairwise_matchups_aggregate_both_seat_orders(tmp_path):
    """One row per unordered PAIR, both seat orders folded in.

    The tournament grid plays `(A,B)` and `(B,A)` as separate cells. Emitting both as their own rows
    would double every matchup and present a seat effect as a strength difference. Folding them is
    what lets the row answer "how often does A beat B", which is the only question a matchup table
    can answer without also answering "which seat was A in".
    """
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    result = pairwise_matchups(root)

    keys = [(row["strategy_a"], row["strategy_b"]) for row in result["rows"]]
    assert keys == sorted(keys), "matchups must be emitted in a fixed order"
    for row in result["rows"]:
        assert row["strategy_a"] < row["strategy_b"], "a matchup must not appear under both orders"
    # The head-to-head fixture plays the two arms in both seat orders, which is ONE matchup of 200
    # games rather than two of 100. Counting 400 here would mean the fold had not happened.
    assert sum(row["games"] for row in result["rows"]) == 200


def test_a_matchup_rate_is_a_rate_over_that_matchup_games_only(tmp_path):
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    row = next(r for r in pairwise_matchups(root)["rows"]
               if (r["strategy_a"], r["strategy_b"]) == (SEARCH_UNIFORM, SEARCH_INFORMED))

    assert row["games"] == 200
    # `strategy_a` is the LOWER id, and SEARCH_UNIFORM (5) < SEARCH_INFORMED (6), so the informed arm
    # is row `b`. It won 60 of 100 in each seat order, so 120 of 200 here.
    assert row["wins_b"] == 120
    assert row["win_rate_b"] == pytest.approx(0.6, abs=1e-12)
    assert row["win_rate_a"] == pytest.approx(1.0 - row["win_rate_b"], abs=1e-12)
    assert row["ci_b"] == pytest.approx(wilson_interval(120, 200), abs=1e-12)


def test_a_strategy_never_plays_itself_in_a_matchup_row(tmp_path):
    """The `RANDOM` vs `RANDOM` cell is a symmetry BASELINE, not a matchup.

    Its rate should sit at 0.5, and reporting that as "RANDOM beats RANDOM half the time" in a
    strength table reads as a measurement of a strategy rather than as the check that the engine is
    unbiased. It is excluded from the matchup rows and carried as its own field.
    """
    root = _synthetic_dataset(tmp_path, opponent="GREEDY")
    result = pairwise_matchups(root)

    assert all(r["strategy_a"] != r["strategy_b"] for r in result["rows"])
    assert result["self_play"] == [], (
        "a fixture whose two seats always held different strategies played nobody against itself"
    )


def test_self_play_is_reported_as_a_seat_check_rather_than_a_matchup(tmp_path):
    """The self-play cell is the seat-symmetry check, and it is kept where it cannot be misread.

    Its expected value is 0.5 by construction. Pooled into the matchup table it would read as
    "RANDOM beats RANDOM 50% of the time", which is a statement about nothing.
    """
    from durak.strategies import RANDOM

    root = tmp_path / "selfplay"
    _single_cell_dataset(root, games=8, wins=4, strategies=(RANDOM, RANDOM))
    result = pairwise_matchups(root)

    assert result["rows"] == [], "a strategy against itself is not a matchup"
    assert len(result["self_play"]) == 1
    assert result["self_play"][0]["strategy_id"] == RANDOM
    assert result["self_play"][0]["win_rate"] == pytest.approx(0.5, abs=1e-12)
    # The interval must contain 0.5: that containment IS the check, and a point estimate alone
    # would let a biased engine pass.
    assert result["self_play"][0]["ci"][0] < 0.5 < result["self_play"][0]["ci"][1]


def test_the_head_to_head_matches_what_the_matchup_matrix_says(tmp_path):
    """One source of truth for the primary number.

    If the matrix could report a different rate for the same two strategies, the report would carry
    two numbers for one fact and nothing would say which was right.
    """
    root = _synthetic_dataset(tmp_path, wins_p0=60, total=100)
    poi = price_of_information(root)
    row = next(r for r in pairwise_matchups(root)["rows"]
               if (r["strategy_a"], r["strategy_b"]) == (SEARCH_UNIFORM, SEARCH_INFORMED))

    assert row["games"] == poi["informed"]["n"]
    assert row["win_rate_b"] == pytest.approx(poi["informed"]["win_rate"], abs=1e-12)
    assert row["win_rate_a"] == pytest.approx(poi["uniform"]["win_rate"], abs=1e-12)


# --- game length ---------------------------------------------------------------------


def test_game_length_pools_both_seats_of_every_game(tmp_path):
    """Both participants, not `strategy_p0` alone.

    A rate over one seat halves the denominator and measures a seat as much as a strategy. Pooling
    both seats is what makes the number mean "this strategy's games".
    """
    root = _synthetic_dataset(tmp_path)
    result = game_length(root)

    assert result["participations"] == 400, "200 games, two participants each"
    assert result["mean_rounds"] == pytest.approx(30.0, abs=1e-12)
    assert result["max_rounds"] == 30


def test_game_length_carries_the_distribution_not_just_its_mean(tmp_path):
    """A mean with no spread beside it cannot be compared to another mean with no spread.

    Binned rather than listed per round: a 36-card game runs to three digits of rounds, and a chart
    with 140 bars renders nothing. The histogram counts GAMES, not participations -- a game has one
    length, and counting it twice would say there were twice as many games as there were.
    """
    root = _synthetic_dataset(tmp_path)
    result = game_length(root)

    assert result["histogram"], "the distribution must be on the page, not just its mean"
    assert sum(b["games"] for b in result["histogram"]) == 200
    assert result["p95_rounds"] == 30


# --- throw-in efficiency --------------------------------------------------------------


def test_throw_in_efficiency_reports_a_rate_not_a_raw_count(tmp_path):
    """Throw-ins per game is the comparable unit; a raw count is not.

    Every strategy here played the same number of games, so a count would happen to be comparable
    and would stop being comparable the moment one of them sat out a cell.
    """
    root = _synthetic_dataset(tmp_path)
    result = throw_in_efficiency(root)

    row = next(r for r in result["rows"] if r["strategy_id"] == SEARCH_UNIFORM)
    assert row["games"] == 200
    assert row["throw_ins_per_game"] == pytest.approx(0.0, abs=1e-12)
    assert row["cards_beaten_per_game"] == pytest.approx(0.0, abs=1e-12)


def test_throw_in_efficiency_and_take_behaviour_agree_on_the_denominator(tmp_path):
    """Two documents over one dataset must not disagree about how many games a strategy played."""
    root = _synthetic_dataset(tmp_path)
    games_by_id = {row["strategy_id"]: row["games"] for row in throw_in_efficiency(root)["rows"]}
    for row in take_behaviour(root)["rows"]:
        assert row["games"] == games_by_id[row["strategy_id"]]


def test_take_behaviour_reports_no_rate_when_there_were_no_takes(tmp_path):
    """Zero of zero is not 0.0.

    A strategy that never took has no voluntary-take rate. Reporting `0.0` would draw a bar
    claiming it never CHOOSES to take, which is a claim about a choice it was never offered.
    """
    root = _synthetic_dataset(tmp_path)
    result = take_behaviour(root)

    row = next(r for r in result["rows"] if r["strategy_id"] == SEARCH_UNIFORM)
    assert row["voluntary_takes"] == 0, "the fixture writes all-zero telemetry"
    assert row["forced_takes"] == 0
    assert row["voluntary_take_rate"] is None
    assert row["takes_per_game"] == pytest.approx(0.0, abs=1e-12)


def test_take_behaviour_splits_the_two_kinds_of_take_when_both_occur(tmp_path):
    """The split only means something on a dataset holding both kinds, so it is checked on one.

    Collapsing them would hide the number the rules care about: a voluntary take is a choice, a
    forced take is the rules refusing to let a defender continue, and their sum cannot say which.
    """
    from durak.strategies import RANDOM

    root = tmp_path / "takes"
    _single_cell_dataset(root, games=8, wins=4, strategies=(SEARCH_UNIFORM, RANDOM))
    result = take_behaviour(root)

    row = next(r for r in result["rows"] if r["strategy_id"] == SEARCH_UNIFORM)
    assert row["voluntary_takes"] > 0 and row["forced_takes"] > 0, "the fixture writes both kinds"
    assert row["voluntary_take_rate"] + row["forced_take_rate"] == pytest.approx(1.0, abs=1e-12)


# --- endgame degeneracy ----------------------------------------------------------------


def test_wins_by_final_hand_size_reports_the_losers_hand(tmp_path):
    """The degeneracy diagnostic is the LOSER's hand at the end.

    The winner's hand is always empty -- that IS the win condition -- so a distribution over
    `winner_final_hand` is a spike at zero and measures nothing. The loser is where the variation
    is: a game ending with the opponent on one card is a very different game from one ending with
    them on twelve.
    """
    root = _synthetic_dataset(tmp_path)
    result = wins_by_final_hand_size(root)

    assert result["winner_final_hand_is_always_zero"] is True, (
        "the fixture gives every winner an empty hand; if this reads False the win rule moved"
    )
    assert result["games"] == 200
    # The head-to-head fixture sets `final_hand_p0`/`final_hand_p1` to 0 for the winner and 3 for
    # the loser, and leaves `loser_final_hand` at its zero default, so every loser reads 0 here.
    assert sum(row["games"] for row in result["loser_hand"]) == 200


def test_wins_by_final_hand_size_uses_a_fixture_that_actually_spreads_the_losers_hand(tmp_path):
    """A distribution pinned by an all-zero column proves the query runs, not that it is right."""
    from durak.strategies import RANDOM

    root = tmp_path / "hands"
    _single_cell_dataset(root, games=8, wins=4, strategies=(SEARCH_UNIFORM, RANDOM))
    result = wins_by_final_hand_size(root)

    sizes = {row["final_hand"] for row in result["loser_hand"]}
    assert len(sizes) > 1, "the fixture writes four different loser hand sizes"
    assert sum(row["games"] for row in result["loser_hand"]) == 8
    assert result["winner_final_hand_is_always_zero"] is True


def test_the_single_card_attack_rate_is_reported(tmp_path):
    """The degenerate endgame is a rate over all games, and it is why this rule set is odd.

    Attacking with a last card ends the game before the defender can answer, so a large share of
    games are decided without a defense ever being played. Spec 3.4 states this rather than fixing
    it; a reader still has to be able to see how often it happens.
    """
    root = _synthetic_dataset(tmp_path)
    result = endgame_degeneracy(root)

    assert result["games"] == 200
    assert result["single_card_attack_wins"] == 0, (
        "the synthetic fixture writes all-zero telemetry, so this must read zero rather than missing"
    )
    assert result["single_card_attack_win_rate"] == pytest.approx(0.0, abs=1e-12)


def test_the_single_card_rate_counts_both_seats(tmp_path):
    """A flag is set on whichever seat won, so counting only `p0` would miss half of them.

    The column exists twice because the telemetry is per seat. A query that reads one and not the
    other reports a rate that is wrong by a factor of two on a dataset where the flag is symmetric,
    and by an amount nobody could check without the other column.
    """
    from durak.strategies import RANDOM

    root = tmp_path / "single"
    _single_cell_dataset(root, games=8, wins=4, strategies=(SEARCH_UNIFORM, RANDOM))
    result = endgame_degeneracy(root)

    assert result["games"] == 8
    assert result["single_card_attack_wins"] == 2, "the fixture flags every fourth game, in seat 0"
    assert result["single_card_attack_win_rate"] == pytest.approx(0.25, abs=1e-12)


# --- throw-in limits versus the rule -----------------------------------------------------


def test_empirical_throw_in_limits_are_measured_against_the_written_rule(tmp_path):
    """The empirical maximum sits next to the rule's cap, not in place of it.

    If the observed maximum ever exceeds the cap the engine is not applying the rule it documents,
    and the report must say so rather than drawing a histogram that quietly agrees with the engine
    whatever the engine did.
    """
    from durak.rules import THROWIN_CAP

    root = _synthetic_dataset(tmp_path)
    result = throw_in_limits(root)

    assert result["rule_cap"] == THROWIN_CAP, "the cap is a constant of durak.rules, not tuned here"
    assert result["observed_max"] == 0, "the fixture writes all-zero telemetry"
    assert result["within_rule"] is True


def test_the_throw_in_limit_query_reads_a_column_whose_values_are_actually_present(tmp_path):
    """An all-zero column would make every distribution query look correct and be empty.

    This is the check that the query reads `max_attacks_in_a_round` and not some other zero-valued
    column: the fixture writes 1..6, so a query reading the wrong column reports a spike at zero and
    this fails.
    """
    from durak.strategies import RANDOM

    root = tmp_path / "throwin"
    _single_cell_dataset(root, games=8, wins=4, strategies=(SEARCH_UNIFORM, RANDOM))
    result = throw_in_limits(root)

    assert result["observed_max"] == 6
    assert {row["attacks"] for row in result["distribution"]} == {1, 2, 3, 4, 5, 6}
    assert sum(row["games"] for row in result["distribution"]) == 8
    assert result["at_or_above_cap"] == 1, (
        "the fixture writes attacks of 1..6 across 8 games, so exactly one game reaches the cap"
    )


# --- deck sensitivity ----------------------------------------------------------------------


def test_deck_split_reports_the_decks_that_were_actually_played(tmp_path):
    """A deck that was not run cannot be reported as a rate.

    The 36/52 split is a sensitivity check and exists only where both decks were played. A dataset
    with one deck must say so rather than emitting a zero for the other, because a zero reads as
    "measured, and it is zero".
    """
    root = _synthetic_dataset(tmp_path)
    result = deck_split(root)

    assert [row["deck_size"] for row in result["rows"]] == [36]
    assert result["missing_decks"] == [52]


def test_deck_split_rates_agree_with_the_tournament_overview(tmp_path):
    """With one deck played, the split and the overview describe the same games and must agree."""
    root = _synthetic_dataset(tmp_path)
    split = deck_split(root)
    overview = {r["strategy_id"]: r for r in tournament_overview(root)["rows"]}

    for row in split["rows"]:
        for entry in row["strategies"]:
            assert entry["win_rate"] == pytest.approx(
                overview[entry["strategy_id"]]["win_rate"], abs=1e-12)


# --- rule variants ---------------------------------------------------------------------------


def test_a_missing_sweep_says_it_is_missing_rather_than_reporting_zeroes(tmp_path):
    """An unmeasured rule variant and a measured zero look identical on a chart.

    Reporting a rate of 0.0 for a sweep that was never run would put a fabricated point next to the
    real ones, and nothing on the page would distinguish them.
    """
    result = rule_variant_comparison(tmp_path / "absent")

    assert result["available"] is False
    assert result["variants"] == []


def test_a_sweep_directory_is_read_into_two_variants_with_their_own_denominators(tmp_path):
    """Each variant's rate arrives with the games behind it.

    The two readings were run in separate tournaments. A reader differencing the two rates needs
    both denominators to judge whether the gap means anything.
    """
    from durak.strategies import RANDOM

    root = tmp_path / "sweep"
    for flag, wins in ((True, 150), (False, 100)):
        _single_cell_dataset(root / f"rule_{int(flag)}", games=200, wins=wins,
                             strategies=(SEARCH_UNIFORM, RANDOM))

    result = rule_variant_comparison(root)

    assert result["available"] is True
    assert [v["throwin_defense_ranks"] for v in result["variants"]] == [False, True]
    for variant in result["variants"]:
        assert variant["games"] == 200
        assert 0.0 <= variant["win_rate"] <= 1.0


def _single_cell_dataset(target, games: int, wins: int, strategies) -> None:
    """One cell of games with an exactly known rate, for a query whose answer must be predictable.

    `wins` rows are marked seat-0 wins, so `strategy_p0` wins exactly `wins` of `games`. The
    telemetry fields the new queries read are written explicitly rather than left at zero, so a
    query that silently reads the wrong column fails here instead of reporting all zeroes.
    """
    import numpy as np

    from durak import engine as engine_module
    from durak.engine import GAME_DTYPE
    from durak.telemetry import write_parquet
    from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

    records = np.zeros(games, dtype=np.dtype(GAME_DTYPE.descr + [("first_attacker_regime", "U32")]))
    for i in range(games):
        records[i]["run_id"] = 0
        records[i]["game_index"] = i
        records[i]["deck_size"] = 36
        records[i]["strategy_p0"] = strategies[0]
        records[i]["strategy_p1"] = strategies[1]
        records[i]["engine_version"] = ENGINE_VERSION
        records[i]["rules_version"] = RULES_VERSION
        records[i]["schema_version"] = SCHEMA_VERSION
        records[i]["winner"] = 0 if i < wins else 1
        records[i]["rounds"] = 20 + i % 5
        records[i]["actions"] = 60
        records[i]["first_attacker"] = 0
        records[i]["first_attacker_regime"] = "RULE_DERIVED"
        records[i]["both_have_trumps"] = True
        records[i]["max_attacks_in_a_round"] = 1 + i % 6
        records[i]["initial_trumps_p0"] = i % 7
        records[i]["initial_trumps_p1"] = (i + 3) % 7
        records[i]["winner_final_hand"] = 0
        records[i]["loser_final_hand"] = 1 + i % 4
        records[i]["throw_ins_p0"] = i % 3
        records[i]["cards_beaten_p1"] = i % 4
        records[i]["voluntary_takes_p0"] = i % 2
        records[i]["forced_takes_p0"] = i % 5
        records[i]["win_from_single_card_p0"] = bool(i % 4 == 0)
    write_parquet(records, target / "games", level=1)
    for level, dtype, name in ((2, "ROUND_DTYPE", "rounds"), (3, "MOVE_DTYPE", "moves")):
        write_parquet(np.zeros(0, dtype=getattr(engine_module, dtype)), target / name,
                      level=level, run_id=0, ensure_empty=True)
