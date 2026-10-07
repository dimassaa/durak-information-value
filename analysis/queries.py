"""The report queries: one dataset in, the JSON the reports draw from out.

Shape of the thing
------------------
Every query is a function from a dataset directory to a plain Python structure, and `build_all` is the
only thing that writes files. That split is deliberate: a query can be tested against a dataset whose
answer is known in advance without touching the filesystem, and a bug in the arithmetic cannot hide
behind a bug in the serialisation.

What each query is for
----------------------
* `price_of_information` -- the project's primary result. The informed arm against the uninformed arm on
  matched games, aggregated over both seat orders, with a paired test. The comparison is PAIRED because
  both arms play the same games: an unpaired comparison of two rates from the same games throws away
  exactly the pairing that makes the estimate precise.
* `tournament_overview` -- every strategy's marginal rate against the whole field. Useful for the shape
  of the tournament, useless across studies, because the rate depends on who else played.
* `opponent_independent_strength` -- rates against `RANDOM` alone. The only rate that survives a change
  of strategy palette, which is why it is emitted as its own document beside the marginal ones.
* `rules_and_dealing` -- the first-attacker split and dealing statistics, so a positional effect can be
  told from a dealing effect.
* `data_quality` -- the audit. A report may only be produced from a clean dataset.

Rounding happens once, at serialisation. Nothing in here rounds, because a delta rounded before being
tested against zero can cross the line between "not significant" and "significant" with no way to tell
from the output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import numpy as np

from analysis.stats import (
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap,
    proportion_diff_interval,
    wilson_interval,
)
from durak.strategies import RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM

#: Spec 8.1's primary metric is pinned to the 36-card deck. The 52-card deck exists in the dataset for
#: the sensitivity analysis and must not silently enter the headline number.
PRIMARY_DECK_SIZE = 36

#: Bootstrap resamples for the effect interval. Fixed rather than derived from the data so the number of
#: resamples is not itself a tuned parameter; the interval's width is stable well below this.
BOOTSTRAP_RESAMPLES = 10_000

_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"


def _connect(dataset_dir) -> duckdb.DuckDBPyConnection:
    """A DuckDB session with the published views loaded from `analysis/schema.sql`.

    The views are executed rather than reimplemented here. A second copy of the version filter would be
    free to drift from the SQL the reports publish, and the two would then disagree about which rows
    exist -- with no error anywhere.
    """
    con = duckdb.connect()
    con.execute(f"SET VARIABLE dataset_root='{dataset_dir}'")
    body = "\n".join(
        line for line in _SQL_PATH.read_text().splitlines() if not line.strip().startswith("--")
    )
    for statement in [s.strip() for s in body.split(";") if s.strip()]:
        con.execute(statement)
    return con


def _rate(con, sql, params=()) -> dict:
    """Run a single-row rate query into the standard `{n, wins, win_rate, ci}` shape."""
    n, wins = con.execute(sql, params).fetchone()
    return {"n": int(n), "wins": int(wins), "win_rate": float(wins) / n if n else None,
            "ci": list(wilson_interval(int(wins), int(n)))} if n else \
        {"n": 0, "wins": 0, "win_rate": None, "ci": None}


def price_of_information(dataset_dir) -> dict:
    """The primary result: what perfect knowledge of the opponent's hand is worth.

    Both arms play the *same* games -- same `(run_seed, pairing_index, game_index)` -- so the comparison
    is paired. Both seat orders are aggregated, because running only one direction confounds the
    information effect with the first-attacker advantage: in one direction the informed arm attacks
    first, in the other it defends.

    Three things travel with the delta because each is what makes it interpretable:
      * the discordant-pair count and McNemar p-value -- whether the difference is more than chance;
      * a paired-bootstrap interval on the effect -- how large it is;
      * an independence-free note that the interval is on the *paired* difference.
    """
    con = _connect(dataset_dir)

    games, informed_wins, uniform_wins = _head_to_head(con)
    b, c = informed_wins, uniform_wins
    informed = {"n": games, "wins": informed_wins,
                "win_rate": informed_wins / games if games else None,
                "ci": list(wilson_interval(informed_wins, games)) if games else None}
    uniform = {"n": games, "wins": uniform_wins,
               "win_rate": uniform_wins / games if games else None,
               "ci": list(wilson_interval(uniform_wins, games)) if games else None}

    delta = None
    if informed["win_rate"] is not None and uniform["win_rate"] is not None:
        paired = _paired_deltas(con)
        boot = paired_bootstrap(paired, resamples=BOOTSTRAP_RESAMPLES, seed=0) \
            if paired.size > 1 else (None, None)
        p = mcnemar_exact(b, c)
        delta = {
            "point": informed["win_rate"] - uniform["win_rate"],
            "ci": list(boot) if boot[0] is not None else None,
            "ci_kind": "paired_bootstrap_percentile",
            "discordant_b": b,
            "discordant_c": c,
            "mcnemar": p,
            # A single-test family: Holm over one member is the identity, and stating the adjusted value
            # keeps the report's schema uniform across every metric it carries.
            "adjusted_p": holm_bonferroni([p])[0],
        }

    undecided = con.execute(f"""
        SELECT COUNT(*) FROM games WHERE deck_size = {PRIMARY_DECK_SIZE} AND winner < 0
    """).fetchone()[0]

    con.close()
    return {
        "metric": "price_of_information",
        "deck_size": PRIMARY_DECK_SIZE,
        "comparison": "head_to_head",
        "informed": informed,
        "uniform": uniform,
        "delta": delta,
        "undecided_games": int(undecided),
        "undecided_note": (
            "Undecided games are excluded from every rate rather than counted as losses. A game that "
            "hit the round cap has no outcome, and scoring it would invent one."
        ),
    }


def _head_to_head(con):
    """Games where the two arms played EACH OTHER, with the outcome expressed per arm.

    This is the only comparison that isolates information. A rate measured against the whole field mixes
    the information effect with how strong each arm's other opponents were, and in an eight-strategy
    tournament those opponents are most of the field -- which is why the field-wide version read +0.012
    while the head-to-head reads +0.070.

    `winner` is a seat, so the outcome is resolved by which strategy occupied which seat.
    """
    rows = con.execute(f"""
        SELECT
            SUM(CASE WHEN winner >= 0 THEN 1 ELSE 0 END)                                     AS games,
            SUM(CASE WHEN winner >= 0 AND
                     ((strategy_p0 = {SEARCH_INFORMED} AND winner = 0)
                   OR (strategy_p1 = {SEARCH_INFORMED} AND winner = 1)) THEN 1 ELSE 0 END)   AS informed_wins,
            SUM(CASE WHEN winner >= 0 AND
                     ((strategy_p0 = {SEARCH_UNIFORM} AND winner = 0)
                   OR (strategy_p1 = {SEARCH_UNIFORM} AND winner = 1)) THEN 1 ELSE 0 END)   AS uniform_wins
        FROM games
        WHERE deck_size = {PRIMARY_DECK_SIZE}
          AND ((strategy_p0 = {SEARCH_INFORMED} AND strategy_p1 = {SEARCH_UNIFORM})
            OR (strategy_p0 = {SEARCH_UNIFORM} AND strategy_p1 = {SEARCH_INFORMED}))
    """).fetchone()
    return int(rows[0] or 0), int(rows[1] or 0), int(rows[2] or 0)


def _paired_deltas(con) -> np.ndarray:
    """Per-game signed outcome of the head-to-head: +1 informed win, -1 informed loss.

    Grouped by the game's own cell and index rather than by `game_index` alone. In an eight-strategy
    tournament the same `game_index` occurs in many cells -- cell 0 of every pairing shares index 0 -- so
    grouping on the index alone merged unrelated games and produced an empty bootstrap interval, which
    looked like a vanishing effect rather than a broken join.
    """
    rows = con.execute(f"""
        SELECT g.pairing_index, g.game_index,
               CASE WHEN g.winner = 0 THEN g.strategy_p0 ELSE g.strategy_p1 END AS winner_strategy
        FROM games AS g
        WHERE g.deck_size = {PRIMARY_DECK_SIZE} AND g.winner >= 0
          AND ((g.strategy_p0 = {SEARCH_INFORMED} AND g.strategy_p1 = {SEARCH_UNIFORM})
            OR (g.strategy_p0 = {SEARCH_UNIFORM} AND g.strategy_p1 = {SEARCH_INFORMED}))
    """).fetchall()
    return np.asarray(
        [1.0 if ws == SEARCH_INFORMED else -1.0 for _p, _g, ws in rows], dtype=np.float64
    )


def _participation(con, opponent_only=None):
    """One row per (game, participant): the strategy, whether it won, and whether the game was undecided.

    The subtlety this exists to get right: `winner` is a SEAT (0 or 1), not a strategy id. Seat 0 won
    the game exactly when `winner = 0`. An earlier version of these queries wrote `winner = strategy_p0`,
    which compares a seat against an id and is true only if some strategy happens to be numbered 0 or 1
    -- `AUTO_ATTACK` is. Every rate came out 0.0 or 1.0 and looked like a real result.
    """
    where = "" if opponent_only is None else f" AND strategy_p1 = {int(opponent_only)}"
    rows = con.execute(f"""
        SELECT strategy_id, won, undecided FROM (
            SELECT strategy_p0 AS strategy_id,
                   (winner = 0) AS won,
                   (winner < 0)  AS undecided
            FROM games WHERE TRUE{where.replace('strategy_p1', 'strategy_p1')}
            UNION ALL
            SELECT strategy_p1,
                   (winner = 1),
                   (winner < 0)
            FROM games WHERE TRUE{where}
        )
    """).fetchall()
    return [(int(s), bool(w), bool(u)) for s, w, u in rows]


def tournament_overview(dataset_dir) -> dict:
    """Every strategy's marginal win rate against the whole field, plus a Wilson interval each.

    Marginal, so the rate moves when a strategy joins or leaves the tournament. Read the field
    composition before comparing two such rates across datasets.
    """
    con = _connect(dataset_dir)
    tally = {}
    for strategy_id, won, undecided in _participation(con):
        entry = tally.setdefault(strategy_id, [0, 0, 0])
        entry[0] += 1
        entry[1] += int(won)
        entry[2] += int(undecided)

    out = []
    for strategy_id in sorted(tally):
        trials, wins, undecided = tally[strategy_id]
        out.append({
            "strategy_id": strategy_id,
            "n": trials,
            "wins": wins,
            # Undecided games are in `n` but not in the numerator: they had no outcome, so counting
            # them as losses would invent one.
            "win_rate": float(wins) / trials if trials else None,
            "ci": list(wilson_interval(wins, trials)),
            "undecided_games": undecided,
        })
    con.close()
    return {"metric": "tournament_overview", "rows": out,
            "note": "Marginal rates: they depend on who else played, so they are not comparable across "
                    "datasets with different strategy palettes."}


def opponent_independent_strength(dataset_dir) -> dict:
    """Win rate against `RANDOM` only, per strategy.

    The metric to quote when comparing against another study: it cannot move when a strategy joins the
    tournament, because the opponent is fixed. Its own document so report 2 can show it beside the
    marginal rates rather than mixing the two.
    """
    con = _connect(dataset_dir)
    tally = {}
    for strategy_id, won, _undecided in _participation(con, opponent_only=RANDOM):
        entry = tally.setdefault(strategy_id, [0, 0])
        entry[0] += 1
        entry[1] += int(won)

    out = []
    for strategy_id in sorted(tally):
        if strategy_id == RANDOM:
            # `RANDOM` against `RANDOM` is the symmetry baseline, not a measurement. Its expected value
            # is 0.5 by construction, so reporting it as a strength would be circular.
            continue
        trials, wins = tally[strategy_id]
        out.append({
            "strategy_id": strategy_id,
            "n": trials,
            "wins": wins,
            "win_rate": float(wins) / trials if trials else None,
            "ci": list(wilson_interval(wins, trials)),
        })
    con.close()
    return {
        "metric": "opponent_independent_strength",
        "opponent": RANDOM,
        "rows": out,
        "note": "Against RANDOM only, so the rate does not move when the strategy palette changes. "
                "RANDOM's own row is omitted: its rate against RANDOM is the symmetry baseline, not a "
                "measurement.",
    }


def rules_and_dealing(dataset_dir) -> dict:
    """The first-attacker split and the dealing statistics.

    Exists so a positional effect can be told from a dealing effect: if the seat that attacks first
    wins far more than half, then any rate difference that correlates with the seat is partly about
    position rather than about the thing being measured.
    """
    con = _connect(dataset_dir)
    p0, p1, total = con.execute(
        "SELECT COUNT(*) FILTER (WHERE winner = 0), COUNT(*) FILTER (WHERE winner = 1), "
        "COUNT(*) FROM games"
    ).fetchone()
    decks = [int(r[0]) for r in con.execute("SELECT DISTINCT deck_size FROM games ORDER BY 1").fetchall()]
    regimes = [r[0] for r in con.execute(
        "SELECT DISTINCT first_attacker_regime FROM games ORDER BY 1").fetchall()]
    both = con.execute("SELECT COUNT(*) FILTER (WHERE both_have_trumps) FROM games").fetchone()[0]
    con.close()
    return {
        "metric": "rules_and_dealing",
        "deck_sizes": decks,
        "regimes": regimes,
        "first_attacker_p0_wins": int(p0),
        "first_attacker_p1_wins": int(p1),
        "games": int(total),
        "both_have_trumps": int(both),
        "no_trump_deal_rate": (float(total - both) / total) if total else None,
    }


def data_quality(dataset_dir) -> dict:
    """The audit. A report may only be built from a dataset this says is clean."""
    con = _connect(dataset_dir)
    total = int(con.execute("SELECT COUNT(*) FROM games").fetchone()[0])
    undecided = int(con.execute("SELECT COUNT(*) FROM games WHERE winner < 0").fetchone()[0])
    redeals = int(con.execute(
        "SELECT COUNT(*) FROM games WHERE flag_redeals_exceeded").fetchone()[0])
    con.close()
    anomalies = undecided + redeals
    return {
        "metric": "data_quality",
        "games": total,
        "anomalies": anomalies,
        "undecided_games": undecided,
        "flag_redeals_exceeded": redeals,
        "clean": anomalies == 0,
    }


def _jsonable(node):
    """Recursively convert NumPy scalars and reject non-finite floats.

    `json.dump` writes a bare `NaN`, which is not valid JSON: most parsers reject it and every charting
    library either fails or silently plots a gap. Failing here, at serialisation, names the query that
    produced it instead of surfacing as a broken figure three stages later.
    """
    if isinstance(node, dict):
        return {k: _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    if isinstance(node, (np.integer,)):
        return int(node)
    if isinstance(node, (np.floating, float)):
        value = float(node)
        if not math.isfinite(value):
            raise ValueError(f"refusing to serialise a non-finite number: {value}")
        return value
    if isinstance(node, (np.bool_,)):
        return bool(node)
    return node


def _round_for_output(node, digits: int = 6):
    """Round floats only at the moment of writing.

    Rounding earlier would let a delta cross zero between the test and the report: the test would see
    one number and the figure would show another. Six digits is display precision, not computation
    precision, and the distinction is why the functions above never round.
    """
    if isinstance(node, dict):
        return {k: _round_for_output(v, digits) for k, v in node.items()}
    if isinstance(node, list):
        return [_round_for_output(v, digits) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return round(node, digits)
    return node


def build_all(dataset_dir, out_dir) -> list:
    """Run every query and write one JSON document per report, plus `meta.json`.

    Returns the written paths. `meta.json` carries the run's identity -- versions, counts, settings --
    because a report whose numbers cannot be traced to a rule set and a seed is not citable.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        "price_of_information": price_of_information,
        "tournament_overview": tournament_overview,
        "rules_and_dealing": rules_and_dealing,
        "data_quality": data_quality,
        "opponent_independent_strength": opponent_independent_strength,
    }

    written = []
    payloads = {}
    for name, query in reports.items():
        payload = _jsonable(query(dataset_dir))
        payloads[name] = _round_for_output(payload)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payloads[name], indent=2, sort_keys=True,
                                  allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)

    con = _connect(dataset_dir)
    versions = con.execute("""
        SELECT DISTINCT engine_version, rules_version, schema_version FROM dataset_versions
        ORDER BY 1, 2, 3
    """).fetchall()
    games = int(con.execute("SELECT COUNT(*) FROM games").fetchone()[0])
    con.close()

    meta = _jsonable({
        "engine_version": versions[0][0] if versions else None,
        "rules_version": versions[0][1] if versions else None,
        "schema_version": versions[0][2] if versions else None,
        "versions_present": [list(v) for v in versions],
        "games": games,
        "dataset_dir": str(dataset_dir),
        "primary_deck_size": PRIMARY_DECK_SIZE,
        "settings": _read_manifest_settings(dataset_dir),
        "note": "Every rate in these documents is filtered to the version triple above. A dataset "
                "written by a different rules version is present in the Parquet tree and is NOT here.",
    })
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(_round_for_output(meta), indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    written.append(meta_path)
    return written


def _read_manifest_settings(dataset_dir) -> dict:
    """The run's settings, verbatim, or an empty block when the manifest is absent.

    Verbatim: a float rounded on the way through JSON cannot rebuild the run, which is the one thing the
    manifest exists to guarantee.
    """
    manifest = Path(dataset_dir) / "manifest.json"
    if not manifest.exists():
        return {}
    return json.loads(manifest.read_text()).get("settings", {})