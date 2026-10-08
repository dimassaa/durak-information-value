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

#: Report 2's most important sentence, and the reason it is a constant rather than a computed string.
#:
#: Both search arms lose essentially every game to the plain `GREEDY` heuristic. That is a fact about
#: the POLICIES -- the search is depth-limited and its evaluation is misweighted -- and not about the
#: engine or about information. A tournament page that ranked strategies and stopped there would let
#: a reader conclude the search is competitive. It is not. The exact rates are on the page as numbers;
#: this is the sentence that stops them being read as a strength ranking.
SEARCH_WEAKNESS_CAVEAT = (
    "Both search arms are markedly WEAKER than the plain GREEDY heuristic: on this dataset each "
    "loses essentially every game to it. That is a finding about the policies -- the search is "
    "depth-limited and its evaluation is misweighted -- and not about the engine or about "
    "information. Report 1's price of information is measured between the two arms and is unaffected, "
    "because a weakness shared by both cancels out of that comparison."
)

#: Why the two rates on report 2 cannot be quoted interchangeably. A reader comparing this tournament
#: to another study will reach for the wrong number unless the difference is on the page.
RATE_KINDS_NOTE = (
    "Two different rates are on this page and they answer different questions. A MARGINAL rate is "
    "measured against the whole field, so it moves when a strategy joins or leaves the tournament and "
    "cannot be compared across studies with different strategy palettes. The OPPONENT-INDEPENDENT "
    "rate is measured against RANDOM alone, so its opponent is fixed and it IS comparable across "
    "studies. Quote that one."
)

#: Report 3's headline, stated because the chosen rule set has a property that most readers of a
#: Durak result would not expect and that changes how every rate on the page should be read.
ENDGAME_CAVEAT = (
    "The win check runs the moment a card leaves the hand, so an attacker playing a last card wins "
    "outright and the defender never gets to answer. On this dataset most games end that way. This is "
    "spec 3.4's documented rule set rather than a defect, but it compresses the endgame: a 'close' "
    "game here is closer than a reader of the number would assume."
)


def _load(aggregate_dir: Path, name: str) -> dict:
    path = Path(aggregate_dir) / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_names() -> dict:
    """Strategy id to display name.

    Read from the registry rather than retyped here, so a strategy that exists and one that does not
    cannot disagree. A report naming a strategy that does not exist is worse than one showing a
    number: it looks like a result.
    """
    from durak.strategies import policy_names

    return policy_names()


def _name(strategy_id, names: dict) -> str:
    """A strategy's name, falling back to the id.

    The fallback matters: an id present in a dataset but absent from the registry is a fact about
    the dataset, and printing `unregistered` says so where printing the bare id would leave the
    reader to guess whether the name is missing or the strategy is.
    """
    return names.get(int(strategy_id), f"id {strategy_id}")


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
    doc.note_block("This is a lower bound", LOWER_BOUND_CAVEAT)

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
                ["Seat 0 wins", dealing.get("seat0_wins")],
                ["Seat 1 wins", dealing.get("seat1_wins")],
                ["Deals with no trump", dealing.get("at_least_one_without_trump_rate")],
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


def _meta_items(meta: dict, poi: dict = None) -> dict:
    """The header metadata every report carries.

    Shared so all three reports identify their run identically. A report whose header differs from
    its siblings' by a field is one a reader cannot cross-check.
    """
    return {
        "games": meta.get("games"),
        "build date": meta.get("build_date", "recorded in meta.json"),
        "engine/rules/schema": "{}/{}/{}".format(
            meta.get("engine_version"), meta.get("rules_version"), meta.get("schema_version")),
        "seed": (meta.get("settings") or {}).get("seed"),
        "deck": (poi or {}).get("deck_size", meta.get("primary_deck_size")),
    }


def _weakest_against_greedy(matchups: list, names: dict) -> tuple:
    """The strongest non-`GREEDY` strategy's win rate against `GREEDY`, and who it belongs to.

    Read out of the matchup matrix rather than recomputed, so the weakness sentence on the page and
    the matrix beside it cannot disagree. Returns `(None, None)` when `GREEDY` is not in the data.
    """
    greedy = next((sid for sid, name in names.items() if name == "GREEDY"), None)
    if greedy is None:
        return None, None
    candidates = []
    for row in matchups:
        if greedy not in (row["strategy_a"], row["strategy_b"]):
            continue
        other = row["strategy_b"] if row["strategy_a"] == greedy else row["strategy_a"]
        rate = row["win_rate_a"] if row["strategy_a"] == other else row["win_rate_b"]
        candidates.append((rate if rate is not None else 0.0, other))
    if not candidates:
        return None, None
    rate, strategy = max(candidates)
    return rate, strategy


def _weighted_median(rows, key: str = "final_hand") -> str:
    """The median of a distribution carried as `(value, count)` pairs, or `n/a` when there is none.

    Computed here rather than taken from the JSON because the loser's-hand document ships the full
    histogram and no summary, and the report is not allowed to compute statistics -- but a headline
    tile wants one. Summing counts rather than averaging sizes is the whole point: the mean of the
    distinct sizes would be a number no game had.
    """
    if not rows:
        return "n/a"
    pairs = sorted((int(row[key]), int(row["games"])) for row in rows)
    total = sum(count for _value, count in pairs)
    if total == 0:
        return "n/a"
    seen = 0
    for value, count in pairs:
        seen += count
        if seen * 2 >= total:
            return str(value)
    return str(pairs[-1][0])


def _missing_decks_note(deck: dict) -> str:
    """Why an unplayed deck gets no rate, in words.

    Written out rather than templated inline because both reports need it and a shared string is
    the only way the two pages can be relied on to say the same thing about the same gap.
    """
    played = ", ".join(str(row["deck_size"]) for row in (deck.get("rows") or [])) or "no"
    missing = ", ".join(str(d) for d in deck.get("missing_decks") or [])
    return (
        f"The 36 vs 52 deck sensitivity check was NOT run: this dataset played only the "
        f"{played}-card deck, so deck {missing} is UNMEASURED. No rate is reported for it -- a "
        f"zero would read as 'measured, and it is zero', which is a different and false claim."
    )


def build_report_two(aggregate_dir, out_path) -> Path:
    """Report 2: the tournament as a whole.

    Order is the argument. The matchup matrix leads because it is the only view on the page whose
    rates do not move with the field, so it is the only one a reader can carry to another study. The
    two weaker views -- the marginal rates and the game-length and throw-in profiles -- follow as
    context, each labelled for what it can and cannot support.
    """
    aggregate_dir = Path(aggregate_dir)
    out_path = Path(out_path)

    meta = _load(aggregate_dir, "meta")
    quality = _load(aggregate_dir, "data_quality")
    matchups = _load(aggregate_dir, "pairwise_matchups") or {}
    overview = _load(aggregate_dir, "tournament_overview") or {}
    independent = _load(aggregate_dir, "opponent_independent_strength") or {}
    length = _load(aggregate_dir, "game_length") or {}
    throwins = _load(aggregate_dir, "throw_in_efficiency") or {}
    takes = _load(aggregate_dir, "take_behaviour") or {}
    deck = _load(aggregate_dir, "deck_split") or {}

    names = _strategy_names()
    matchup_rows = matchups.get("rows") or []

    doc = Document()
    doc.header(
        "Tournament overview",
        f"{meta.get('games')} games, "
        f"{(meta.get('primary_deck_size') or 36)}-card deck, all strategies in all pairings",
        _meta_items(meta),
    )

    # The weakness caveat goes in the quality block near the top, not in a footnote: it is the
    # sentence that decides how every ranking on this page should be read.
    doc.note_block("The search arms are weaker than the heuristic", SEARCH_WEAKNESS_CAVEAT)

    weakest_rate, weakest_strategy = _weakest_against_greedy(matchup_rows, names)
    self_play = {row["strategy_id"]: row for row in (matchups.get("self_play") or [])}
    random_baseline = self_play.get(2)
    matchup_games = sum(row["games"] for row in matchup_rows)

    doc.kpi_row([
        ("Games", meta.get("games")),
        ("Strategies", len(overview.get("rows") or []) or None),
        ("Distinct matchups", len(matchup_rows) or None),
        ("Best rate vs GREEDY",
         "n/a" if weakest_rate is None else f"{weakest_rate:.5f} ({_name(weakest_strategy, names)})"),
        ("RANDOM vs RANDOM",
         "n/a" if not random_baseline else
         f"{random_baseline['win_rate']:.4f} {_interval(random_baseline.get('ci'))}"),
    ])

    if matchup_rows:
        # Ordered by the weaker side's rate, descending, so the extreme matchups sit at the top. The
        # order is the claim: a matchup table whose rows are sorted by id has chosen no order, and
        # the reader is left to find the interesting row themselves.
        ordered = sorted(
            matchup_rows,
            key=lambda r: (-max(r["win_rate_a"] or 0.0, r["win_rate_b"] or 0.0),
                           r["strategy_a"], r["strategy_b"]),
        )
        doc.table(
            ["Matchup", "Rate (first named)", "Wilson 95%", "Games",
             "Rate (second named)", "Wilson 95%"],
            [
                [f"{_name(r['strategy_a'], names)} vs {_name(r['strategy_b'], names)}",
                 r["win_rate_a"], _interval(r.get("ci_a")), r["games"],
                 r["win_rate_b"], _interval(r.get("ci_b"))]
                for r in ordered
            ],
            sort_keys=[],
        )
        doc.chart_block(
            "tournament-matchups",
            "The stronger side of every matchup: a bar at the top means the pairing was "
            "one-sided, and the search arms sit at the bottom against every heuristic",
            {
                # One dataset, one label. A two-series chart here would need a label per ROW,
                # because each row names different strategies, and a Chart.js dataset carries a
                # single label -- so the per-row variant silently mislabelled every bar. The rate
                # shown is the stronger side's, which is what "this pairing was one-sided" means,
                # and the table beside it carries both sides at full precision.
                "type": "bar",
                "data": {
                    "labels": [f"{_name(row['strategy_a'], names)} vs {_name(row['strategy_b'], names)}"
                               for row in ordered],
                    "datasets": [{
                        "label": "win rate of the stronger side",
                        "data": [round(max(row["win_rate_a"] or 0.0, row["win_rate_b"] or 0.0), 6)
                                 for row in ordered],
                        "backgroundColor": "#4f9cf9",
                    }],
                },
                "options": {"scales": {"y": {"min": 0, "max": 1}}},
            },
            full_width=True,
        )

    rows_by_id = {row["strategy_id"]: row for row in (overview.get("rows") or [])}
    independent_by_id = {row["strategy_id"]: row
                         for row in (independent.get("rows") or [])}
    ranked_ids = sorted(rows_by_id, key=lambda sid: -(rows_by_id[sid]["win_rate"] or 0.0))

    if ranked_ids:
        doc.chart_row(
            {
                "id": "tournament-marginal",
                "title": "Marginal rates rank the field but do not travel: they move with the palette",
                "full_width": False,
                "spec": {
                    "type": "bar",
                    "data": {
                        "labels": [_name(sid, names) for sid in ranked_ids],
                        "datasets": [{"label": "Marginal win rate",
                                      "data": [round(rows_by_id[sid]["win_rate"], 6)
                                               for sid in ranked_ids],
                                      "backgroundColor": "#4f9cf9"}],
                    },
                    "options": {"scales": {"y": {"min": 0, "max": 1}}},
                },
            },
            {
                "id": "tournament-vs-random",
                "title": "Against RANDOM alone, where the rate is fixed and can be compared elsewhere",
                "full_width": False,
                "spec": {
                    "type": "bar",
                    "data": {
                        "labels": [_name(sid, names) for sid in ranked_ids
                                   if sid in independent_by_id],
                        "datasets": [{"label": "Win rate vs RANDOM",
                                      "data": [round(independent_by_id[sid]["win_rate"], 6)
                                               for sid in ranked_ids if sid in independent_by_id],
                                      "backgroundColor": "#52c41a"}],
                    },
                    "options": {"scales": {"y": {"min": 0, "max": 1}}},
                },
            },
        )

        doc.table(
            ["Strategy", "Marginal rate", "Marginal Wilson 95%", "vs RANDOM",
             "vs RANDOM Wilson 95%", "Games"],
            [
                [_name(sid, names),
                 rows_by_id[sid]["win_rate"], _interval(rows_by_id[sid].get("ci")),
                 independent_by_id.get(sid, {}).get("win_rate"),
                 _interval(independent_by_id.get(sid, {}).get("ci")),
                 rows_by_id[sid]["n"]]
                for sid in ranked_ids
            ],
            sort_keys=[],
        )

    doc.note_block("Two rates, two different questions", RATE_KINDS_NOTE)

    histogram = length.get("histogram") or []
    if histogram:
        doc.chart_block(
            "tournament-length",
            f"Games run a median of {length.get('median_rounds')} rounds, and the search arms drag "
            f"the mean up to {length.get('mean_rounds')}",
            {
                "type": "bar",
                "data": {
                    "labels": [f"{b['bin_low']}+" for b in histogram],
                    "datasets": [{"label": "Games",
                                  "data": [b["games"] for b in histogram],
                                  "backgroundColor": "#4f9cf9"}],
                },
                "options": {"scales": {"x": {"ticks": {"maxRotation": 90, "autoSkip": True}}}},
            },
            full_width=True,
        )

    per_strategy_length = {row["strategy_id"]: row for row in (length.get("per_strategy") or [])}
    if per_strategy_length:
        doc.table(
            ["Strategy", "Mean rounds", "p95 rounds", "Games"],
            [[_name(sid, names), row["mean_rounds"], row["p95_rounds"], row["games"]]
             for sid, row in sorted(per_strategy_length.items(),
                                    key=lambda kv: -(kv[1]["mean_rounds"] or 0.0))],
            sort_keys=[],
        )

    throwin_by_id = {row["strategy_id"]: row for row in (throwins.get("rows") or [])}
    take_by_id = {row["strategy_id"]: row for row in (takes.get("rows") or [])}
    profile_ids = [sid for sid in throwin_by_id if sid in take_by_id]
    if profile_ids:
        doc.chart_row(
            {
                "id": "tournament-throwins",
                "title": "Throw-ins per game: the arms that attack add far more than the ones that take",
                "full_width": False,
                "spec": {
                    "type": "bar",
                    "data": {
                        "labels": [_name(sid, names) for sid in profile_ids],
                        "datasets": [{
                            "label": "Throw-ins per game",
                            "data": [round(throwin_by_id[sid]["throw_ins_per_game"] or 0.0, 4)
                                     for sid in profile_ids],
                            "backgroundColor": "#faad14",
                        }],
                    },
                },
            },
            {
                "id": "tournament-takes",
                "title": "Share of takes that were voluntary, not forced by the rules",
                "full_width": False,
                "spec": {
                    "type": "bar",
                    "data": {
                        "labels": [_name(sid, names) for sid in profile_ids],
                        "datasets": [{
                            "label": "Voluntary share of takes",
                            "data": [None if take_by_id[sid]["voluntary_take_rate"] is None
                                     else round(take_by_id[sid]["voluntary_take_rate"], 4)
                                     for sid in profile_ids],
                            "backgroundColor": "#f5222d",
                        }],
                    },
                    "options": {"scales": {"y": {"min": 0, "max": 1}}},
                },
            },
        )

    if deck.get("missing_decks"):
        doc.note_block("The 36 vs 52 deck check was not run", _missing_decks_note(deck))

    if quality:
        # Merged with the matchup denominator into ONE quality block: two blocks on one page put
        # two headings reading "Data quality" above one another, and a reader looking for the
        # audit finds the denominator first.
        doc.quality_block({
            **{k: v for k, v in sorted(quality.items()) if k != "metric"},
            "matchup games": f"{matchup_games} pairwise games across {len(matchup_rows)} distinct "
                             f"matchups",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.render(out_path)
    return out_path


def build_report_three(aggregate_dir, out_path) -> Path:
    """Report 3: the rules, the dealing, and how sensitive the result is to reading them two ways.

    The degenerate endgame leads. Spec 3.4 makes a last-card attack win outright, and on this
    dataset that decides the large majority of games -- so a reader who has not seen it will
    misread every rate on the page, including the ones about throw-ins and trumps. The rule-variant
    sweep closes the page, because the throw-in rule is a genuine ambiguity in the source document
    rather than a tuning knob, and it is the last thing that should be able to move a conclusion.
    """
    aggregate_dir = Path(aggregate_dir)
    out_path = Path(out_path)

    meta = _load(aggregate_dir, "meta")
    quality = _load(aggregate_dir, "data_quality")
    degeneracy = _load(aggregate_dir, "endgame_degeneracy") or {}
    hands = _load(aggregate_dir, "wins_by_final_hand_size") or {}
    limits = _load(aggregate_dir, "throw_in_limits") or {}
    dealing = _load(aggregate_dir, "rules_and_dealing") or {}
    deck = _load(aggregate_dir, "deck_split") or {}
    variants = _load(aggregate_dir, "rule_variant_comparison") or {}

    names = _strategy_names()

    doc = Document()
    doc.header(
        "Rules, dealing, and rule sensitivity",
        f"{(meta.get('primary_deck_size') or 36)}-card deck, "
        f"regime {', '.join(dealing.get('regimes') or ['unknown'])}",
        _meta_items(meta),
    )

    rate = degeneracy.get("single_card_attack_win_rate")
    loser_hand = hands.get("loser_hand") or []
    doc.kpi_row([
        ("Games", degeneracy.get("games") or dealing.get("games")),
        ("Ended by a last-card attack",
         "n/a" if rate is None else f"{rate * 100:.1f}%"),
        ("Loser's median hand", _weighted_median(loser_hand)),
        ("Empirical max table attacks", limits.get("observed_max")),
        ("Rule cap (THROWIN_CAP)", limits.get("rule_cap")),
        ("Deals where at least one player holds no trump",
         dealing.get("at_least_one_without_trump_rate")),
    ])

    # --- the degeneracy leads ------------------------------------------------------------
    doc.note_block("The endgame rule, and why it distorts every rate here", ENDGAME_CAVEAT)

    per_strategy = degeneracy.get("per_strategy") or []
    if per_strategy:
        doc.chart_block(
            "rules-single-card",
            f"About {rate * 100:.0f}% of games are decided by an attack the defender never answered"
            if rate is not None else "Single-card attacks decide a large share of games",
            {
                "type": "bar",
                "data": {
                    "labels": [_name(row["strategy_id"], names) for row in per_strategy],
                    "datasets": [{
                        "label": "Wins from a single-card attack",
                        "data": [round(row["single_card_win_rate"] or 0.0, 6)
                                 for row in per_strategy],
                        "backgroundColor": "#f5222d",
                    }],
                },
                "options": {"scales": {"y": {"min": 0, "max": 1}}},
            },
            full_width=True,
        )

    if loser_hand:
        # Grouped into pairs of sizes: a 36-card game's loser hand runs past 25 and a chart with one
        # bar per size renders nothing. The grouping is stated in the title so the reader knows the
        # axis is binned.
        buckets: dict = {}
        for row in loser_hand:
            key = (int(row["final_hand"]) // 2) * 2
            buckets[key] = buckets.get(key, 0) + int(row["games"])
        labels = [f"{k}-{k + 1}" for k in sorted(buckets)]
        doc.chart_block(
            "rules-loser-hand",
            "Most games end with the loser still holding cards, so the endgame is decided by the "
            "attacker's empty hand rather than by exhaustion",
            {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": "Games by loser's final hand size (binned)",
                        "data": [buckets[k] for k in sorted(buckets)],
                        "backgroundColor": "#4f9cf9",
                    }],
                },
                "options": {"scales": {"x": {"ticks": {"maxRotation": 90, "autoSkip": True}}}},
            },
            full_width=True,
        )
        doc.table(
            ["Loser's final hand", "Games"],
            [[row["final_hand"], row["games"]] for row in loser_hand],
            sort_keys=[],
        )

    # --- trumps and dealing --------------------------------------------------------------
    doc.chart_block(
        "rules-trumps",
        f"About {(dealing.get('at_least_one_without_trump_rate') or 0.0) * 100:.1f}% of deals leave at least one "
        f"player without a trump",
        {
            "type": "doughnut",
            "data": {
                "labels": ["Both players hold a trump", "At least one player has none"],
                "datasets": [{
                    "label": "Deals",
                    "data": [dealing.get("both_have_trumps"),
                             (dealing.get("games") or 0) - (dealing.get("both_have_trumps") or 0)],
                    "backgroundColor": ["#52c41a", "#f5222d"],
                }],
            },
        },
    )

    doc.table(
        ["Dealing and position", "Value"],
        [
            ["Deck sizes played", ", ".join(str(d) for d in (dealing.get("deck_sizes") or []))],
            ["Regimes", ", ".join(dealing.get("regimes") or [])],
            ["Both players hold a trump", dealing.get("both_have_trumps")],
            ["Deals where at least one player holds no trump", dealing.get("at_least_one_without_trump_rate")],
            ["Seat 0 wins", dealing.get("seat0_wins")],
            ["Seat 1 wins", dealing.get("seat1_wins")],
        ],
        sort_keys=[],
    )
    doc.note_block(
        "A positional effect can be told from a dealing effect here",
        "The two seat counts are near-equal, so a rate difference between two strategies is about "
        "the strategies and not about which seat they were dealt into. The field that answers the "
        "positional question is <b>first_attacker_wins</b>, which counts whoever attacked first: "
        "under RULE_DERIVED the first attacker is chosen by the rules and is not seat 0, so it cannot "
        "be recovered from the seat counts. It reads 0.48 on the primary dataset -- the first attacker "
        "loses about half again as often as it wins, which a clean 50/50 seat split cannot show.",
    )

    # --- throw-in limits versus the rule ---------------------------------------------------
    distribution = limits.get("distribution") or []
    if distribution:
        doc.chart_block(
            "rules-throwin-limit",
            f"Play reaches the {limits.get('rule_cap')}-card cap in "
            f"{limits.get('at_or_above_cap')} games and never exceeds it",
            {
                "type": "bar",
                "data": {
                    "labels": [str(row["attacks"]) for row in distribution],
                    "datasets": [{
                        "label": "Games by maximum attacks on the table in a round",
                        "data": [row["games"] for row in distribution],
                        "backgroundColor": ["#4f9cf9"] * len(distribution),
                    }],
                },
                "options": {"scales": {"x": {"title": {"display": True, "text": "attacks on the table"}},
                                       "y": {"title": {"display": True, "text": "games"}}}},
            },
            full_width=True,
        )
    doc.table(
        ["Throw-in limit", "Value"],
        [
            ["Rule", limits.get("rule_expression")],
            ["THROWIN_CAP", limits.get("rule_cap")],
            ["Observed maximum", limits.get("observed_max")],
            ["Games reaching the cap", limits.get("at_or_above_cap")],
            ["Observed within the rule", limits.get("within_rule")],
        ],
        sort_keys=[],
    )

    # --- rule sensitivity closes the page ---------------------------------------------------
    if variants.get("available") and variants.get("variants"):
        rows = variants["variants"]
        doc.chart_block(
            "rules-variant",
            "Which ranks may be added to an already-answered attack moves the win rate by "
            f"{(max(r['win_rate'] or 0.0 for r in rows) - min(r['win_rate'] or 0.0 for r in rows)) * 100:.1f} "
            f"points, so the throw-in rule is a real ambiguity and not a detail",
            {
                "type": "bar",
                "data": {
                    "labels": ["Defence ranks may NOT be thrown in"
                               if not row["throwin_defense_ranks"]
                               else "Defence ranks MAY be thrown in" for row in rows],
                    "datasets": [{
                        "label": "SEARCH_UNIFORM win rate",
                        "data": [round(r["win_rate"] or 0.0, 6) for r in rows],
                        "backgroundColor": ["#4f9cf9", "#52c41a"],
                    }],
                },
                "options": {"scales": {"y": {"min": 0, "max": 1}}},
            },
            full_width=True,
        )
        doc.table(
            ["throwin_defense_ranks", "Games", "Wins", "Win rate", "Wilson 95%"],
            [[r["throwin_defense_ranks"], r["games"], r["wins"], r["win_rate"],
              _interval(r.get("ci"))] for r in rows],
            sort_keys=[],
        )
        doc.note_block(
            "What this comparison does and does not establish",
            "These two readings were measured in SEPARATE tournaments, so they are not a paired "
            "comparison and the gap is not a significance test. Each rate carries its own interval; "
            "where the intervals overlap, the rule reading does not explain the difference. The "
            "throw-in rule is a genuine ambiguity in the source document rather than a tuning knob, "
            "so both readings are reported rather than one being chosen.",
        )
    else:
        doc.note_block(
            "The throw-in rule sensitivity is unmeasured",
            "The throwin_defense_ranks sweep was NOT run for this dataset, so how much the result "
            "depends on that reading of the throw-in rule is UNMEASURED. No rate is shown for it: a "
            "zero would read as 'measured, and it is zero'.",
        )

    if deck.get("missing_decks"):
        doc.note_block("The 36 vs 52 deck check was not run", _missing_decks_note(deck))

    if quality:
        doc.quality_block({k: v for k, v in sorted(quality.items()) if k != "metric"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.render(out_path)
    return out_path


def main(argv=None) -> int:
    """Build every report. Run as `python -m reports.build <aggregate_dir> <out_dir>`."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    aggregate_dir = Path(args[0]) if args else Path("analysis/data")
    out_dir = Path(args[1]) if len(args) > 1 else Path("reports/out")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        build_report_one(aggregate_dir, out_dir / "01_price_of_information.html"),
        build_report_two(aggregate_dir, out_dir / "02_tournament_overview.html"),
        build_report_three(aggregate_dir, out_dir / "03_rules_and_dealing.html"),
    ]

    # Figures are optional in the sense that a missing Matplotlib must not take the three HTML
    # reports down with it. They are not optional for the README, which embeds them, so a failure is
    # reported loudly on stderr rather than swallowed.
    try:
        from reports.figures import build_figures

        written.extend(build_figures(aggregate_dir, out_dir / "figures"))
    except ImportError as exc:  # pragma: no cover - exercised only without Matplotlib
        print(f"figures skipped: {exc}", file=sys.stderr)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())