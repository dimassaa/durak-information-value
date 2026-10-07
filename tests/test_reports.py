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
                                    "both_have_trumps": 39900, "no_trump_deal_rate": 0.0025},
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
