"""The report primitives, and the determinism they exist to guarantee.

Every assertion here is about the OUTPUT BYTES, not about appearance. A dashboard is regenerated when
the dataset changes, and a report that differs between two runs of the same data cannot be reviewed:
a reviewer cannot tell a real change from a timestamp. So the file must depend on the JSON and nothing
else.

The three specific leaks this pins:

* the clock -- `new Date()`, `toLocaleString`, `Intl`. A build date comes from `meta.json`, which is a
  recorded fact, not the moment someone pressed a button.
* iteration order -- Python dicts preserve insertion order, so two callers building the same document
  from differently ordered inputs would produce different bytes. Every mapping key is sorted before it
  is emitted.
* the Chart.js version -- a CDN tag that floats means a report's appearance depends on when it was
  opened. It is pinned and asserted.
"""

import json
import re
from pathlib import Path

import pytest

from reports.dashboard import CHARTJS_VERSION, Document, style_tokens


def _document() -> Document:
    doc = Document()
    doc.header("The Price of Information", "36-card deck, head to head",
               {"games": 40000, "engine_version": 2, "build_date": "2026-10-07"})
    doc.kpi_row([("Delta", "+0.082"), ("n", "40 000"), ("p", "2.3e-07")])
    doc.chart_block("chart-poi", "Perfect knowledge is worth about eight points",
                    {"type": "bar", "data": {"labels": ["uniform", "informed"], "values": [0.459, 0.541]}})
    doc.table(
        ["Strategy", "Win rate"],
        [["GREEDY", 0.953], ["SEARCH_UNIFORM", 0.502], ["SEARCH_INFORMED", 0.513]],
        sort_keys=[0],
    )
    doc.quality_block({"anomalies": 0, "clean": True, "games": 40000})
    return doc


def test_a_rendered_file_is_valid_standalone_html(tmp_path):
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert f'<script src="https://cdn.jsdelivr.net/npm/chart.js@{CHARTJS_VERSION}"></script>' in html
    assert CHARTJS_VERSION == "4.5.1"


def test_the_css_token_block_is_present(tmp_path):
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    for token in ("--bg-primary", "--bg-card", "--text-primary", "--color-accent1",
                  "--positive", "--negative", "--gap", "--radius"):
        assert token in html, f"{token} is missing from the rendered CSS"
    # The token block must be the source of the styling, not a comment describing it.
    assert style_tokens()["--bg-primary"].startswith("#")


def test_kpi_and_table_markup_are_emitted(tmp_path):
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    assert 'class="kpi-row"' in html
    assert re.search(r'class="kpi"', html), "each KPI needs its own element to be styled"
    assert "<table" in html and "</table>" in html
    assert "GREEDY" in html


def test_sortable_headers_carry_their_sort_key(tmp_path):
    """A header the reader can click has to say what it sorts by.

    Without `data-sort-key` the attribute the client-side handler reads is absent, the control renders,
    and clicking it does nothing -- which reads as a working table that is not one.
    """
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    matches = re.findall(r"<th[^>]*data-sort-key=\"(\d+)\"", html)
    assert matches, "no sortable header was emitted"
    assert matches == ["0"], f"only the declared sort key should be sortable, got {matches}"


def test_the_chart_payload_is_embedded_so_the_file_needs_no_server(tmp_path):
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    assert "const " in html and "chart-poi" in html
    assert '"labels"' in html or "'labels'" in html or html.count("labels") > 0
    assert "http://" not in html.replace("https://cdn.jsdelivr.net", ""), (
        "the page must not fetch anything beyond the pinned CDN script"
    )


def test_rendering_the_same_input_twice_is_byte_identical(tmp_path):
    first = tmp_path / "a.html"
    second = tmp_path / "b.html"
    _document().render(first)
    _document().render(second)

    assert first.read_bytes() == second.read_bytes(), (
        "two renders of the same document differ; something in the output depends on more than the input"
    )


def test_nothing_in_the_output_reads_a_clock(tmp_path):
    """The build date is a recorded fact from `meta.json`, not the moment of rendering."""
    path = tmp_path / "report.html"
    _document().render(path)
    html = path.read_text(encoding="utf-8")

    for forbidden in ("new Date", "toLocaleString", "Intl.", "Date.now", "getTime"):
        assert forbidden not in html, f"`{forbidden}` makes the output depend on when it was rendered"


def test_insertion_order_of_the_inputs_does_not_reach_the_output(tmp_path):
    """Two callers building the same document from differently ordered MAPPINGS get identical bytes.

    Python dicts iterate in insertion order, so a document that emits a mapping as given will differ
    between callers that assembled it differently. Every mapping is sorted before it is written.

    Table ROW order is deliberately NOT normalised: row order carries meaning in a results table, and
    sorting it would silently discard the ranking the report chose to show. The caller owns that.
    """
    forward = Document()
    forward.header("T", "s", {"a": 1, "b": 2, "c": 3})
    forward.quality_block({"z": 1, "y": 2, "x": 3})
    forward.table(["A", "B"], [["x", 1], ["y", 2]], sort_keys=[])

    reverse = Document()
    reverse.header("T", "s", {"c": 3, "b": 2, "a": 1})
    reverse.quality_block({"x": 3, "y": 2, "z": 1})
    reverse.table(["A", "B"], [["x", 1], ["y", 2]], sort_keys=[])

    a, b = tmp_path / "f.html", tmp_path / "r.html"
    forward.render(a)
    reverse.render(b)

    assert a.read_bytes() == b.read_bytes(), (
        "mapping insertion order leaked into the output; every key must be sorted before it is emitted"
    )


def test_html_special_characters_are_escaped(tmp_path):
    """A strategy name or a title with `<` in it must not inject markup."""
    doc = Document()
    doc.header("a < b & c", "x < y", {})
    doc.table(["Name"], [["<script>alert(1)</script>"]], sort_keys=[])
    path = tmp_path / "escaped.html"
    doc.render(path)
    html = path.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_a_note_block_is_distinct_from_a_quality_block(tmp_path):
    """Prose caveats and quality flags are different things and must not share one heading.

    Filing an interpretive caveat under a heading that reads "Data quality" tells a reader it is an
    audit finding. It is not: "the search arms are weaker than the heuristic" is an interpretation of
    the numbers, and a reader scanning the data-quality section for problems would either skip it or
    read it as a defect in the run.
    """
    doc = Document()
    doc.note_block("An interpretation", "This rate means something other than what it looks like.")
    doc.quality_block({"anomalies": 0})
    path = tmp_path / "note.html"
    doc.render(path)
    html = path.read_text(encoding="utf-8")

    assert 'class="note"' in html
    assert html.count("Data quality") == 1, (
        "a caveat must not add a second 'Data quality' heading; the two block kinds are distinct"
    )
    assert "An interpretation" in html


def test_a_note_block_stays_where_it_was_added(tmp_path):
    """A caveat is only useful next to the number it qualifies.

    `quality_block` is documented as rendering last; a note is not, and reordering it would put the
    lower-bound caveat on report 1 a screen away from the delta it qualifies.
    """
    forward = Document()
    forward.note_block("First", "one")
    forward.chart_block("c", "t", {"type": "bar", "data": {}})
    forward.note_block("Second", "two")

    reversed_ = Document()
    reversed_.note_block("Second", "two")
    reversed_.chart_block("c", "t", {"type": "bar", "data": {}})
    reversed_.note_block("First", "one")

    a, b = tmp_path / "f.html", tmp_path / "r.html"
    forward.render(a)
    reversed_.render(b)

    assert a.read_text(encoding="utf-8").index("one") < a.read_text(encoding="utf-8").index("two")
    assert a.read_bytes() != b.read_bytes(), (
        "note position is part of the document's meaning; two orders must not render identically"
    )


def test_a_note_block_escapes_its_text(tmp_path):
    doc = Document()
    doc.note_block("t < b", "x <script>alert(1)</script>")
    path = tmp_path / "escaped_note.html"
    doc.render(path)
    html = path.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_an_empty_document_still_renders_valid_html(tmp_path):
    path = tmp_path / "empty.html"
    Document().render(path)
    html = path.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")

# --- report 1: the price of information -----------------------------------------


def _fixture_json(tmp_path) -> Path:
    """Minimal aggregate set shaped exactly like `analysis/data/*.json`."""
    root = tmp_path / "agg"
    root.mkdir()
    payload = {
        "meta.json": {
            "engine_version": 2, "rules_version": 2, "schema_version": 2,
            "games": 40000, "build_date": "2026-10-07", "dataset_dir": "/tmp/run",
            "primary_deck_size": 36, "settings": {"seed": 3},
        },
        "price_of_information.json": {
            "metric": "price_of_information", "deck_size": 36, "comparison": "head_to_head",
            "informed": {"n": 20000, "wins": 10820, "win_rate": 0.541,
                         "ci": [0.5297, 0.5522]},
            "uniform": {"n": 20000, "wins": 9180, "win_rate": 0.459,
                        "ci": [0.4478, 0.4703]},
            "delta": {"point": 0.082, "ci": [0.0620, 0.1024],
                      "ci_kind": "paired_bootstrap_percentile",
                      "discordant_b": 10820, "discordant_c": 9180,
                      "mcnemar": 2.302e-07, "adjusted_p": 2.302e-07},
            "undecided_games": 0,
            "undecided_note": "Undecided games are excluded from every rate.",
        },
        "data_quality.json": {"metric": "data_quality", "games": 40000, "anomalies": 0,
                             "undecided_games": 0, "flag_redeals_exceeded": 0, "clean": True},
        "tournament_overview.json": {
            "metric": "tournament_overview",
            "rows": [{"strategy_id": 5, "n": 40000, "wins": 20080, "win_rate": 0.502,
                      "ci": [0.497, 0.507], "undecided_games": 0}],
            "note": "Marginal rates.",
        },
        "opponent_independent_strength.json": {"metric": "opponent_independent_strength",
                                               "opponent": 2, "rows": [], "note": "n/a"},
        "rules_and_dealing.json": {"metric": "rules_and_dealing", "deck_sizes": [36],
                                    "regimes": ["RULE_DERIVED"], "first_attacker_p0_wins": 20000,
                                    "first_attacker_p1_wins": 20000, "games": 40000,
                                    "both_have_trumps": 39900, "at_least_one_without_trump_rate": 0.0025},
    }
    for name, body in payload.items():
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    return root


def test_report_one_renders_from_aggregate_json(tmp_path):
    from reports.build import build_report_one

    agg = _fixture_json(tmp_path)
    out = build_report_one(agg, tmp_path / "out" / "01_price_of_information.html")
    html = out.read_text(encoding="utf-8")

    assert out.name == "01_price_of_information.html"
    assert out.exists()
    assert html.lstrip().startswith("<!DOCTYPE html>")
    # The headline must be the measured number, not a restatement of the question.
    assert "+0.0820" in html
    assert "0.0620" in html and "0.1024" in html, "the paired interval must be on the page"
    assert "2.302e-07" in html, "the Holm-adjusted p-value must be on the page"


def test_report_one_states_that_the_value_is_a_lower_bound(tmp_path):
    """The claim has to be visible where the number is read, not only in the README.

    The informed search is depth-limited, so the measured effect is a floor on what perfect knowledge is
    worth. A reader who sees "+0.082" without that qualification will quote it as the price.
    """
    from reports.build import build_report_one

    agg = _fixture_json(tmp_path)
    html = build_report_one(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "lower bound" in html.lower()


def test_report_one_carries_a_data_quality_block(tmp_path):
    from reports.build import build_report_one

    agg = _fixture_json(tmp_path)
    html = build_report_one(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "Data quality" in html
    assert "anomalies" in html and "clean" in html


def test_report_one_is_byte_identical_across_two_builds(tmp_path):
    from reports.build import build_report_one

    agg = _fixture_json(tmp_path)
    first = tmp_path / "one.html"
    second = tmp_path / "two.html"
    build_report_one(agg, first)
    build_report_one(agg, second)

    assert first.read_bytes() == second.read_bytes()


def test_report_one_reports_the_build_date_from_meta_not_the_clock(tmp_path):
    from reports.build import build_report_one

    agg = _fixture_json(tmp_path)
    html = build_report_one(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "2026-10-07" in html
    for forbidden in ("new Date", "toLocaleString", "Intl."):
        assert forbidden not in html


# --- reports 2-3 and figures ----------------------------------------------------------
#
# The fixture below carries the aggregates reports 2 and 3 consume, written with values chosen so
# that a number which appears on the page can be matched to a specific JSON field by string. A
# report that quietly invented a number could not be caught by an existence check.


def _fixture_json_reports_2_3(tmp_path) -> Path:
    """Aggregate set for reports 2 and 3, plus the reports-1 documents report 2 refers to."""
    root = _fixture_json(tmp_path)
    payload = {
        "tournament_overview.json": {
            "metric": "tournament_overview",
            "note": "Marginal rates.",
            "rows": [
                {"strategy_id": 3, "n": 40000, "wins": 35920, "win_rate": 0.898,
                 "ci": [0.894, 0.902], "undecided_games": 0},
                {"strategy_id": 5, "n": 40000, "wins": 20032, "win_rate": 0.5008,
                 "ci": [0.496, 0.506], "undecided_games": 0},
                {"strategy_id": 6, "n": 40000, "wins": 20120, "win_rate": 0.503,
                 "ci": [0.498, 0.508], "undecided_games": 0},
            ],
        },
        "opponent_independent_strength.json": {
            "metric": "opponent_independent_strength", "opponent": 2,
            "note": "Against RANDOM only.",
            "rows": [
                {"strategy_id": 3, "n": 20000, "wins": 17824, "win_rate": 0.8912,
                 "ci": [0.887, 0.895]},
                {"strategy_id": 5, "n": 20000, "wins": 10320, "win_rate": 0.516,
                 "ci": [0.509, 0.523]},
                {"strategy_id": 6, "n": 20000, "wins": 10140, "win_rate": 0.507,
                 "ci": [0.500, 0.514]},
            ],
        },
        "pairwise_matchups.json": {
            "metric": "pairwise_matchups", "note": "Both seat orders folded in.",
            "rows": [
                {"strategy_a": 3, "strategy_b": 5, "games": 40000, "wins_a": 39990, "wins_b": 10,
                 "win_rate_a": 0.99975, "ci_a": [0.9995, 0.9999],
                 "win_rate_b": 0.00025, "ci_b": [0.0001, 0.0005]},
                {"strategy_a": 5, "strategy_b": 6, "games": 40000, "wins_a": 20100, "wins_b": 19900,
                 "win_rate_a": 0.5025, "ci_a": [0.4975, 0.5075],
                 "win_rate_b": 0.4975, "ci_b": [0.4925, 0.5025]},
            ],
            "self_play": [
                {"strategy_id": 2, "games": 20000, "win_rate": 0.5039,
                 "ci": [0.4977, 0.5101]},
            ],
        },
        "game_length.json": {
            "metric": "game_length", "games": 40000, "participations": 80000,
            "mean_rounds": 33.27, "median_rounds": 29.0, "p95_rounds": 61.0, "max_rounds": 145,
            "bin_width_rounds": 5,
            "histogram": [{"bin_low": 25, "games": 20000}, {"bin_low": 30, "games": 14000},
                          {"bin_low": 60, "games": 6000}],
            "per_strategy": [
                {"strategy_id": 3, "games": 20000, "mean_rounds": 21.2, "p95_rounds": 40.0},
                {"strategy_id": 5, "games": 20000, "mean_rounds": 41.6, "p95_rounds": 70.0},
            ],
        },
        "throw_in_efficiency.json": {
            "metric": "throw_in_efficiency", "note": "Per game.",
            "rows": [
                {"strategy_id": 3, "games": 20000, "throw_ins": 10194, "cards_beaten": 14067,
                 "cards_collected": 26909, "throw_ins_per_game": 0.5097,
                 "cards_beaten_per_game": 0.7034, "cards_collected_per_game": 1.3454},
                {"strategy_id": 5, "games": 20000, "throw_ins": 9264, "cards_beaten": 15602,
                 "cards_collected": 11641, "throw_ins_per_game": 0.4632,
                 "cards_beaten_per_game": 0.7801, "cards_collected_per_game": 0.5821},
            ],
        },
        "take_behaviour.json": {
            "metric": "take_behaviour", "note": "Voluntary and forced kept apart.",
            "rows": [
                {"strategy_id": 3, "games": 20000, "voluntary_takes": 0, "forced_takes": 2199,
                 "cards_collected": 26909, "takes_per_game": 0.11,
                 "voluntary_take_rate": 0.0, "forced_take_rate": 1.0},
                {"strategy_id": 5, "games": 20000, "voluntary_takes": 9759, "forced_takes": 985,
                 "cards_collected": 11641, "takes_per_game": 0.537,
                 "voluntary_take_rate": 0.908, "forced_take_rate": 0.092},
            ],
        },
        "wins_by_final_hand_size.json": {
            "metric": "wins_by_final_hand_size", "games": 40000,
            "winner_final_hand_is_always_zero": True,
            "loser_hand": [{"final_hand": 1, "games": 800}, {"final_hand": 3, "games": 7300},
                           {"final_hand": 6, "games": 990}, {"final_hand": 9, "games": 1050},
                           {"final_hand": 12, "games": 830}],
            "note": "The loser's final hand.",
        },
        "endgame_degeneracy.json": {
            "metric": "endgame_degeneracy", "games": 40000,
            "single_card_attack_wins": 20000, "single_card_attack_win_rate": 0.5,
            "per_strategy": [
                {"strategy_id": 3, "games": 20000, "single_card_wins": 7000,
                 "single_card_win_rate": 0.35},
                {"strategy_id": 5, "games": 20000, "single_card_wins": 13000,
                 "single_card_win_rate": 0.65},
            ],
            "note": "Win check runs before the defender responds.",
        },
        "throw_in_limits.json": {
            "metric": "throw_in_limits", "games": 40000, "rule_cap": 6,
            "rule_expression": "min(defender's current hand, THROWIN_CAP)",
            "observed_max": 6, "at_or_above_cap": 240, "within_rule": True,
            "distribution": [{"attacks": 1, "games": 5800}, {"attacks": 3, "games": 9500},
                             {"attacks": 6, "games": 240}],
        },
        "deck_split.json": {
            "metric": "deck_split", "missing_decks": [52],
            "rows": [{
                "deck_size": 36, "games": 40000,
                "strategies": [
                    {"strategy_id": 3, "games": 20000, "wins": 17960, "win_rate": 0.898},
                    {"strategy_id": 5, "games": 20000, "wins": 10020, "win_rate": 0.501},
                ],
            }],
            "note": "Marginal within a deck.",
        },
        "rule_variant_comparison.json": {
            "metric": "rule_variant_comparison", "available": True,
            "variants": [
                {"throwin_defense_ranks": False, "games": 6000, "wins": 1150,
                 "win_rate": 0.19175, "ci": [0.1774, 0.2068]},
                {"throwin_defense_ranks": True, "games": 6000, "wins": 1624,
                 "win_rate": 0.27075, "ci": [0.2548, 0.2874]},
            ],
            "note": "Two readings of the throw-in rule.",
        },
        "rules_and_dealing.json": {
            "metric": "rules_and_dealing", "deck_sizes": [36],
            "regimes": ["RULE_DERIVED"], "first_attacker_p0_wins": 20000,
            "first_attacker_p1_wins": 20000, "games": 40000, "both_have_trumps": 26216,
            "at_least_one_without_trump_rate": 0.3446,
        },
    }
    for name, body in payload.items():
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    return root


def test_report_two_renders_from_aggregate_json(tmp_path):
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    out = build_report_two(agg, tmp_path / "out" / "02_tournament_overview.html")
    html = out.read_text(encoding="utf-8")

    assert out.name == "02_tournament_overview.html"
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")


def test_report_two_shows_the_matchup_matrix_with_both_rates(tmp_path):
    """The matrix is the one view that does not move with the field, so it is the centre of the page.

    Both rates are shown. One would force a reader to difference them by hand, and a reader who
    does that gets a different number from the one the page claims.
    """
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "GREEDY" in html and "SEARCH_UNIFORM" in html
    assert "0.99975" in html, "the matrix rate must be on the page at full precision"
    assert "0.5025" in html


def test_report_two_carries_the_intervals_not_just_the_rates(tmp_path):
    """Every rate on this page moves with the field composition; an interval is the only thing that
    says whether a gap between two of them is a difference or a coincidence."""
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "Wilson" in html or "95%" in html
    assert "0.4975" in html, "an interval bound must be on the page"


def test_report_two_says_the_search_arms_are_weaker_than_the_heuristic(tmp_path):
    """The headline weakness is a finding about the policies and belongs on the page.

    The search arms lose essentially every game to `GREEDY` on this dataset. A tournament report that
    ranked strategies without saying so would let a reader conclude the search is competitive. It
    is not, and the report is where that has to be visible.

    The assertion is on the WORD, not on the number: the number is already pinned by the matrix
    test, and a test that passed on the number alone would still pass if the sentence were deleted.
    """
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "weaker" in html.lower(), (
        "the report must state that the search arms are weaker than the heuristic"
    )
    # It has to say what kind of finding that is, or a reader takes it as a verdict on the engine.
    assert "policies" in html.lower()
    # And it has to say the primary result survives, or the weakness reads as disowning report 1.
    assert "unaffected" in html.lower()


def test_report_two_distinguishes_marginal_rates_from_opponent_independent_ones(tmp_path):
    """The two rates answer different questions and cannot be quoted interchangeably.

    A marginal rate moves when a strategy joins the tournament; the `RANDOM`-only rate does not.
    Presenting them under one heading would let a reader quote one as the other.
    """
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "0.8912" in html, "the RANDOM-only rate must be on the page"
    assert "marginal" in html.lower()


def test_report_two_carries_a_data_quality_block(tmp_path):
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "Data quality" in html
    assert "anomalies" in html


def test_report_three_renders_and_leads_with_the_degenerate_endgame(tmp_path):
    """Spec 3.4 is the most important known property of the chosen rule set.

    Attacking with a last card ends the game before the defender can answer, which compresses the
    endgame and inflates single-card win rates. The plan puts it first for that reason, so the test
    pins the ORDER and not merely the presence: a page that carries the number but buries it after
    the throw-in sweep is a page that has reversed the priority while still passing a presence check.
    """
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    out = build_report_three(agg, tmp_path / "out" / "03_rules_and_dealing.html")
    html = out.read_text(encoding="utf-8")

    assert out.name == "03_rules_and_dealing.html"
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "single-card" in html.lower(), "the single-card-attack rate must be on the page"
    assert "last card" in html.lower(), "the rule itself must be stated, not just its rate"

    # ORDER, not presence. A page carrying both numbers but showing the rule-variant sweep first has
    # reversed the priority the plan set, and a presence check would not notice.
    single_card_chart = html.index("decided by an attack the defender never answered")
    loser_hand_chart = html.index("loser still holding cards")
    trumps_chart = html.index("leave at least one")
    rule_ambiguity = html.index("What this comparison does and does not establish")

    assert single_card_chart < loser_hand_chart < trumps_chart < rule_ambiguity, (
        "the degenerate endgame leads, the dealing statistics follow, and the rule-variant "
        "sensitivity closes the page"
    )


def test_report_three_shows_wins_by_final_hand_size(tmp_path):
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    for size in ("1", "3", "6", "9", "12"):
        assert size in html, f"loser hand size {size} must be on the page"


def test_report_three_shows_the_trump_distribution_and_the_no_trump_share(tmp_path):
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "0.3446" in html, "the no-trump deal rate must be on the page"
    assert "trump" in html.lower()


def test_report_three_puts_the_empirical_throw_in_limit_beside_the_written_rule(tmp_path):
    """The empirical distribution and the rule's cap are two different claims.

    Reporting only the empirical one would agree with the engine whatever the engine did; reporting
    only the rule would assert a cap the data never reached. Both, side by side, is the comparison.
    """
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "THROWIN_CAP" in html
    assert "6" in html


def test_report_three_names_the_unplayed_deck_rather_than_reporting_a_zero(tmp_path):
    """The 36/52 sensitivity check needs both decks.

    The primary run played the 36-card deck only. Emitting a rate for the 52-card deck would put a
    fabricated zero next to real numbers, so the page names it as unmeasured instead.
    """
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "52" in html
    assert "not" in html.lower()


def test_the_rule_variant_sweep_is_shown_with_both_arms_and_their_denominators(tmp_path):
    """The throw-in rule is a genuine ambiguity in the source document, so both readings are reported.

    Each was measured in its own tournament with its own denominator. The page has to show both
    denominators, because a reader who differences two bare rates is reading noise if the gap is
    inside the intervals.
    """
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "0.27075" in html and "0.19175" in html
    assert "6000" in html, "each variant's game count must be on the page"


def test_an_unavailable_rule_variant_sweep_is_stated_as_unmeasured(tmp_path):
    """An unmeasured sweep and a measured zero look identical on a chart.

    If the sweep was not run, the page must say so. Reporting `0.0` for it would place a fabricated
    point beside the real ones with nothing to distinguish them.
    """
    from reports.build import build_report_three

    agg = _fixture_json_reports_2_3(tmp_path)
    (agg / "rule_variant_comparison.json").write_text(json.dumps({
        "metric": "rule_variant_comparison", "available": False, "variants": [],
    }))
    html = build_report_three(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    assert "not" in html.lower()


def test_reports_two_and_three_are_byte_identical_across_two_builds(tmp_path):
    from reports.build import build_report_three, build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    for builder in (build_report_two, build_report_three):
        first = tmp_path / f"{builder.__name__}-a.html"
        second = tmp_path / f"{builder.__name__}-b.html"
        builder(agg, first)
        builder(agg, second)
        assert first.read_bytes() == second.read_bytes(), (
            f"{builder.__name__} is not reproducible; two builds of the same JSON differ"
        )


def test_reports_two_and_three_read_no_clock(tmp_path):
    from reports.build import build_report_three, build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    for builder in (build_report_two, build_report_three):
        html = builder(agg, tmp_path / "out.html").read_text(encoding="utf-8")
        for forbidden in ("new Date", "toLocaleString", "Intl.", "Date.now", "getTime"):
            assert forbidden not in html, f"{builder.__name__} leaks the clock via {forbidden}"


def test_reports_two_and_three_escape_what_reaches_the_page(tmp_path):
    """A value carrying markup must not become markup.

    The injected surface is `data_quality.json`, because that document is rendered as free text on
    every report. It is a real path: the audit's flag values are strings a future anomaly check
    could fill with anything, and the report copies them onto the page verbatim. Strategy NAMES are
    not a surface -- they come from the registry, not the dataset -- so injecting there would have
    tested a field the report never reads.
    """
    from reports.build import build_report_three, build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    payload = json.loads((agg / "data_quality.json").read_text())
    payload["anomalies"] = "<script>alert(1)</script>"
    (agg / "data_quality.json").write_text(json.dumps(payload))

    for builder in (build_report_two, build_report_three):
        html = builder(agg, tmp_path / "out.html").read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html, f"{builder.__name__} let markup through"
        assert "&lt;script&gt;" in html, f"{builder.__name__} neither escaped nor dropped the value"


def test_table_row_order_is_the_callers_and_not_a_sort(tmp_path):
    """Report 2's matchup table is ordered by strength, and sorting it by a column would destroy that.

    The dashboard's `sort_keys` controls which COLUMN headers are clickable; it is not a row sort.
    If a report emitted its rows sorted by strategy id, the ranking the page chose to show would be
    replaced by an arbitrary one, and nothing on the page would say so.
    """
    from reports.build import build_report_two

    agg = _fixture_json_reports_2_3(tmp_path)
    html = build_report_two(agg, tmp_path / "out.html").read_text(encoding="utf-8")

    # GREEDY (3) vs SEARCH_UNIFORM (5) is the fixture's first row, and it is the extreme matchup.
    assert html.index("0.99975") < html.index("0.5025"), (
        "the strongest matchup must come first; the rows were reordered"
    )


# --- figures ---------------------------------------------------------------------------


def test_the_figure_renderer_writes_a_png_set(tmp_path):
    from reports.figures import build_figures

    agg = _fixture_json_reports_2_3(tmp_path)
    written = build_figures(agg, tmp_path / "figures")

    assert written, "no figures were written"
    for path in written:
        assert path.exists() and path.stat().st_size > 0, f"{path.name} is empty"
        assert path.suffix == ".png"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"


def test_figures_are_byte_identical_across_two_renders(tmp_path):
    """The README embeds these files, so they are committed artefacts and must not churn.

    A PNG whose bytes move on every build turns every regeneration into a diff nobody reads, which
    is the same failure mode the HTML determinism requirement exists to prevent.
    """
    from reports.figures import build_figures

    agg = _fixture_json_reports_2_3(tmp_path)
    first = build_figures(agg, tmp_path / "a")
    second = build_figures(agg, tmp_path / "b")

    assert [p.name for p in first] == [p.name for p in second]
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} is not reproducible"


def test_figures_are_rendered_from_the_json_and_not_recomputed(tmp_path):
    """A figure and its chart must never disagree.

    Both are drawn from the same aggregate document, so a change to the query moves both together.
    The test proves it by perturbing the JSON and requiring the PNG to change.
    """
    from reports.figures import build_figures

    agg = _fixture_json_reports_2_3(tmp_path)
    before = {p.name: p.read_bytes() for p in build_figures(agg, tmp_path / "before")}

    payload = json.loads((agg / "endgame_degeneracy.json").read_text())
    payload["single_card_attack_win_rate"] = 0.5
    payload["per_strategy"][0]["single_card_win_rate"] = 0.9
    (agg / "endgame_degeneracy.json").write_text(json.dumps(payload))
    after = {p.name: p.read_bytes() for p in build_figures(agg, tmp_path / "after")}

    changed = [name for name in before if before[name] != after[name]]
    assert changed, "no figure responded to a change in the JSON it is supposed to be drawn from"
