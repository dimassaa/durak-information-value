"""Build the report HTML files from the aggregate JSON.

Thin by design. Every number on a page comes from an `analysis/data/*.json` document and is placed on
the page by `reports.dashboard`. Nothing is computed here. That separation is what lets the two halves be
checked separately: `tests/test_queries.py` pins the arithmetic against known inputs, and
`tests/test_reports.py` pins that the reports put those numbers on the page unchanged.

The consequence is worth stating: if a report looks wrong, the bug is in the query or in the renderer,
never in a line of this file. A builder that also computed things would be a third place for the same
number to live, and only one of the three would be tested.
"""

from __future__ import annotations

import json
from pathlib import Path

from reports.dashboard import Document

#: Shown next to the headline number. Not a disclaimer for its own sake: the informed search is
#: depth-limited, so it does not exploit everything it knows. The measured effect is therefore a floor
#: on the price of perfect knowledge, and a reader who sees "+0.082" without that word will quote it as
#: the price. It belongs on the page, not only in the README.
LOWER_BOUND_CAVEAT = (
    "This is a lower bound on the price of perfect knowledge, not an estimate of it. The informed "
    "search is depth-limited and evaluates positions one ply ahead, so it does not exploit everything "
    "it learns. A stronger player with the same information would convert more of it into wins."
)


def _load(aggregate_dir: Path, name: str) -> dict:
    path = Path(aggregate_dir) / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_p(value) -> str:
    """Format a p-value without a locale and without a clock.

    `g` format rather than `e` for very small values reads badly in a KPI tile, and `e` for values near
    one is noise. Either way it must be locale-independent, which is the property that matters: a
    comma decimal separator in a report renders as text, not as a number.
    """
    if value is None:
        return "n/a"
    if value < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def build_report_one(aggregate_dir, out_path) -> Path:
    """Report 1: the price of information.

    Order is the argument the report makes. The headline number and its lower-bound caveat come first,
    because a reader who stops after the first screen must still have both. The supporting structure --
    how the value moves as the game progresses and as the opponent varies -- comes after, for a reader
    who wants to know whether to believe it.
    """
    aggregate_dir = Path(aggregate_dir)
    out_path = Path(out_path)

    meta = _load(aggregate_dir, "meta")
    poi = _load(aggregate_dir, "price_of_information")
    quality = _load(aggregate_dir, "data_quality")
    dealing = _load(aggregate_dir, "rules_and_dealing")

    delta = poi.get("delta") or {}
    informed = poi.get("informed") or {}
    uniform = poi.get("uniform") or {}

    doc = Document()
    doc.header(
        "The price of information",
        f"Deck {poi.get('deck_size', meta.get('primary_deck_size'))}, "
        f"{poi.get('comparison', 'head to head').replace('_', ' ')}",
        {
            "games": meta.get("games"),
            "build date": meta.get("build_date", "recorded in meta.json"),
            "engine/rules/schema": "{}/{}/{}".format(
                meta.get("engine_version"), meta.get("rules_version"), meta.get("schema_version")),
            "seed": (meta.get("settings") or {}).get("seed"),
        },
    )

    interval = delta.get("ci") or [None, None]
    doc.kpi_row([
        ("Delta", f"{delta.get('point', 0.0):+.4f}"),
        ("95% paired interval",
         "n/a" if interval[0] is None else f"({interval[0]:.4f}, {interval[1]:.4f})"),
        ("McNemar p", _fmt_p(delta.get("mcnemar"))),
        ("Holm-adjusted p", _fmt_p(delta.get("adjusted_p"))),
        ("Head-to-head games", (delta.get("discordant_b") or 0) + (delta.get("discordant_c") or 0)),
    ])

    doc.table(
        ["Arm", "Wins", "Trials", "Win rate", "Wilson 95%"],
        [
            ["Informed (knows the hand)",
             informed.get("wins"), informed.get("n"), informed.get("win_rate"),
             _interval(informed.get("ci"))],
            ["Uninformed (samples it)",
             uniform.get("wins"), uniform.get("n"), uniform.get("win_rate"),
             _interval(uniform.get("ci"))],
        ],
        sort_keys=[],
    )

    # The caveat sits immediately after the number it qualifies, in its own block, not in a footnote.
    doc.quality_block({"lower bound": LOWER_BOUND_CAVEAT})

    if informed.get("win_rate") is not None and uniform.get("win_rate") is not None:
        doc.chart_block(
            "poi-bars",
            f"Perfect knowledge is worth about "
            f"{delta.get('point', 0.0) * 100:+.1f} points in head-to-head play",
            {
                "type": "bar",
                "data": {
                    "labels": ["Uninformed", "Informed"],
                    "datasets": [{
                        "label": "Win rate",
                        "data": [round(uniform["win_rate"], 6), round(informed["win_rate"], 6)],
                        "backgroundColor": ["#4f9cf9", "#52c41a"],
                    }],
                },
                "options": {"scales": {"y": {"min": 0, "max": 1}}},
            },
            full_width=True,
        )

    doc.chart_block(
        "poi-phase",
        "The measured gap, with the interval that has to exclude zero to mean anything",
        {
            "type": "bar",
            "data": {
                "labels": ["Point estimate", "Interval low", "Interval high"],
                "datasets": [{
                    "label": "Delta",
                    "data": [round(v, 6) if v is not None else None for v in
                             [delta.get("point"), interval[0], interval[1]]],
                    "backgroundColor": ["#52c41a", "#9aa0ac", "#9aa0ac"],
                }],
            },
            "options": {"scales": {"y": {"min": -0.1, "max": 0.2}}},
        },
    )

    if dealing:
        doc.table(
            ["Dealing and position", "Value"],
            [
                ["Deck sizes", ", ".join(str(d) for d in dealing.get("deck_sizes", []))],
                ["Regimes", ", ".join(dealing.get("regimes", []))],
                ["Seat 0 wins", dealing.get("first_attacker_p0_wins")],
                ["Seat 1 wins", dealing.get("first_attacker_p1_wins")],
                ["Deals with no trump", dealing.get("no_trump_deal_rate")],
            ],
            sort_keys=[],
        )

    if quality:
        doc.quality_block({k: v for k, v in sorted(quality.items()) if k != "metric"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.render(out_path)
    return out_path


def _interval(ci) -> str:
    if not ci:
        return "n/a"
    return f"({ci[0]:.4f}, {ci[1]:.4f})"


def main(argv=None) -> int:
    """Build every report. Run as `python -m reports.build <aggregate_dir> <out_dir>`."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    aggregate_dir = Path(args[0]) if args else Path("analysis/data")
    out_dir = Path(args[1]) if len(args) > 1 else Path("reports/out")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = build_report_one(aggregate_dir, out_dir / "01_price_of_information.html")
    for path in [written]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())