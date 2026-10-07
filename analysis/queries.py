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


# --- reports 2 and 3 ----------------------------------------------------------------------
#
# Everything below reads LEVEL 1 only.
#
# That is a constraint, not a preference. The level-2 (`rounds`) and level-3 (`moves`) records carry
# `(run_id, game_index)` and nothing else, while `game_index` is CELL-LOCAL: in the 57-cell primary
# run each of the 20,000 indices occurs in all 57 cells, so 1,140,000 games share 20,000 indices.
# `analysis/schema.sql` joins levels 2 and 3 to level 1 on `(run_id, game_index)`, which fans that
# join out by 57 -- 39,073,046 round rows become 2,227,163,622. Every level-2 number is therefore an
# average over 57 unrelated games, and no level-2 number can be attributed to a strategy at all.
#
# The level-1 columns are unaffected: they live on the game row, so a strategy's throw-ins and takes
# are already per-game and need no join. So the rule these queries follow is: measure from level 1,
# and if a quantity is only available at level 2, report that it is unavailable rather than report a
# fan-out average as if it were a measurement. TRACKED IN IMPROVEMENTS.md.


def pairwise_matchups(dataset_dir) -> dict:
    """Every unordered pair of distinct strategies, both seat orders folded in, with a rate each.

    This is the metric that does not depend on the tournament's composition: two strategies played
    each other, so neither rate moved when a third strategy joined the field. That is what
    `tournament_overview`'s marginal rates cannot say, and the reason this table exists beside them.

    Self-play cells are held out of the rows and returned as `self_play`. `RANDOM` vs `RANDOM` is a
    symmetry check on the engine -- its expected value is 0.5 by construction -- and printing it as a
    matchup row would present the check as a measurement of `RANDOM`.
    """
    con = _connect(dataset_dir)
    rows = con.execute("""
        SELECT strategy_a, strategy_b, COUNT(*) AS games,
               COUNT(*) FILTER (WHERE a_won) AS wins_a
        FROM (
            -- ONE ROW PER GAME, not per participation. Resolving the outcome to strategy `a` needs
            -- to know which seat `a` occupied, and that is only knowable per game: unpivoting to
            -- participants first and folding afterwards would count every matchup twice and halve
            -- every rate's denominator along with it.
            SELECT LEAST(strategy_p0, strategy_p1) AS strategy_a,
                   GREATEST(strategy_p0, strategy_p1) AS strategy_b,
                   CASE WHEN strategy_p0 < strategy_p1 THEN winner = 0 ELSE winner = 1 END AS a_won
            FROM games WHERE winner >= 0 AND strategy_p0 <> strategy_p1
        )
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()

    baselines = con.execute("""
        SELECT strategy_id, COUNT(*) AS games, COUNT(*) FILTER (WHERE won) AS wins
        FROM (
            SELECT strategy_p0 AS strategy_id, winner = 0 AS won FROM games WHERE winner >= 0
            UNION ALL
            SELECT strategy_p1, winner = 1 FROM games WHERE winner >= 0
        )
        WHERE strategy_id IN (
            SELECT strategy_p0 FROM games WHERE winner >= 0 AND strategy_p0 = strategy_p1
        )
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    out = []
    for strategy_a, strategy_b, games, wins_a in rows:
        games, wins_a = int(games), int(wins_a)
        wins_b = games - wins_a
        out.append({
            "strategy_a": int(strategy_a),
            "strategy_b": int(strategy_b),
            "games": games,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "win_rate_a": wins_a / games if games else None,
            "ci_a": list(wilson_interval(wins_a, games)),
            "win_rate_b": wins_b / games if games else None,
            "ci_b": list(wilson_interval(wins_b, games)),
        })

    # A cell where BOTH seats held the same strategy is the only place "does seat 0 win more often"
    # can be asked: with two different strategies the answer mixes the seat with the strength.
    # Reported per strategy rather than pooled, because pooled it would not be comparable between
    # strategies, and a reader checks symmetry on the RANDOM row.
    return {
        "metric": "pairwise_matchups",
        "rows": out,
        "self_play": [{
            "strategy_id": int(sid),
            "games": int(n),
            "win_rate": int(w) / int(n) if n else None,
            "ci": list(wilson_interval(int(w), int(n))) if n else None,
        } for sid, n, w in baselines],
        "note": "One row per unordered pair, both seat orders folded in, so neither rate moves when "
                "the strategy palette changes. This is the metric to quote across studies; the "
                "marginal rates in tournament_overview are not. The self_play rows are a seat check, "
                "not matchups: a strategy against itself has no strength to measure.",
    }


def game_length(dataset_dir) -> dict:
    """How long games run, over every participation.

    Every game contributes two rows -- one per participant -- because a game has two players and a
    length measured over `strategy_p0` alone would halve the denominator while describing half the
    field. `participations` is therefore twice `games`, and is carried so a reader can check that.
    """
    con = _connect(dataset_dir)
    total, mean_rounds, max_rounds = con.execute(
        "SELECT COUNT(*), AVG(rounds), MAX(rounds) FROM games").fetchone()
    percentiles = con.execute(
        "SELECT quantile_cont(rounds, 0.50), quantile_cont(rounds, 0.95) FROM games").fetchone()

    histogram = con.execute("""
        SELECT (rounds / 5) * 5 AS bin_low, COUNT(*) AS games
        FROM games GROUP BY 1 ORDER BY 1
    """).fetchall()

    per_strategy = con.execute("""
        WITH participation AS (
            SELECT strategy_p0 AS strategy_id, rounds FROM games
            UNION ALL
            SELECT strategy_p1 AS strategy_id, rounds FROM games
        )
        SELECT strategy_id, COUNT(*), AVG(rounds), quantile_cont(rounds, 0.95)
        FROM participation GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    return {
        "metric": "game_length",
        "games": int(total),
        "participations": int(total) * 2,
        "mean_rounds": float(mean_rounds) if mean_rounds is not None else None,
        "median_rounds": float(percentiles[0]) if percentiles[0] is not None else None,
        "p95_rounds": float(percentiles[1]) if percentiles[1] is not None else None,
        "max_rounds": int(max_rounds) if max_rounds is not None else None,
        "bin_width_rounds": 5,
        "histogram": [{"bin_low": int(lo), "games": int(n)} for lo, n in histogram],
        "per_strategy": [{
            "strategy_id": int(sid), "games": int(n),
            "mean_rounds": float(mean), "p95_rounds": float(p95),
        } for sid, n, mean, p95 in per_strategy],
    }


def throw_in_efficiency(dataset_dir) -> dict:
    """Throw-ins made and cards beaten, per game, per strategy.

    Both are divided by the games played. A raw count would be comparable only while every strategy
    played the same number of games, and it stops being comparable the moment one of them sits out a
    cell -- at which point the count reports the schedule rather than the strategy.
    """
    con = _connect(dataset_dir)
    rows = con.execute("""
        SELECT strategy_id, COUNT(*) AS games,
               SUM(throw_ins) AS throw_ins, SUM(cards_beaten) AS cards_beaten,
               SUM(cards_collected) AS cards_collected
        FROM (
            SELECT strategy_p0 AS strategy_id, throw_ins_p0 AS throw_ins,
                   cards_beaten_p0 AS cards_beaten, cards_collected_p0 AS cards_collected
            FROM games
            UNION ALL
            SELECT strategy_p1, throw_ins_p1, cards_beaten_p1, cards_collected_p1 FROM games
        )
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    out = []
    for sid, games, throw_ins, beaten, collected in rows:
        games = int(games)
        throw_ins, beaten, collected = int(throw_ins), int(beaten), int(collected)
        out.append({
            "strategy_id": int(sid),
            "games": games,
            "throw_ins": throw_ins,
            "cards_beaten": beaten,
            "cards_collected": collected,
            "throw_ins_per_game": throw_ins / games if games else None,
            "cards_beaten_per_game": beaten / games if games else None,
            "cards_collected_per_game": collected / games if games else None,
        })
    return {
        "metric": "throw_in_efficiency",
        "rows": out,
        "note": "Per game, not a raw count: a count tracks the number of games a strategy was "
                "scheduled for, which is a property of the tournament and not of the strategy.",
    }


def take_behaviour(dataset_dir) -> dict:
    """Voluntary and forced takes per strategy, kept apart.

    A voluntary take is a choice the policy made; a forced take is the rules refusing to let a
    defender continue. They are different facts about a strategy and are not additive into a single
    "take rate": the average of the two is neither the rate of choosing to take nor the rate of being
    forced to, and reporting it as one would make a strategy that takes compulsively look
    indistinguishable from one that is cornered.
    """
    con = _connect(dataset_dir)
    rows = con.execute("""
        SELECT strategy_id, COUNT(*) AS games,
               SUM(voluntary) AS voluntary, SUM(forced) AS forced,
               SUM(cards_collected) AS cards_collected
        FROM (
            SELECT strategy_p0 AS strategy_id, voluntary_takes_p0 AS voluntary,
                   forced_takes_p0 AS forced, cards_collected_p0 AS cards_collected
            FROM games
            UNION ALL
            SELECT strategy_p1, voluntary_takes_p1, forced_takes_p1, cards_collected_p1 FROM games
        )
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    out = []
    for sid, games, voluntary, forced, collected in rows:
        games = int(games)
        voluntary, forced, collected = int(voluntary), int(forced), int(collected)
        takes = voluntary + forced
        out.append({
            "strategy_id": int(sid),
            "games": games,
            "voluntary_takes": voluntary,
            "forced_takes": forced,
            "cards_collected": collected,
            "takes_per_game": takes / games if games else None,
            "voluntary_take_rate": voluntary / takes if takes else None,
            "forced_take_rate": forced / takes if takes else None,
        })
    return {
        "metric": "take_behaviour",
        "rows": out,
        "note": "A voluntary take is a policy choice and a forced take is a rules outcome. The two "
                "are reported separately because their sum is not a rate of anything.",
    }


def wins_by_final_hand_size(dataset_dir) -> dict:
    """The distribution of the LOSER's final hand, which is the degeneracy diagnostic.

    `winner_final_hand` is not charted: the winner's hand is empty by the win condition, so its
    distribution is a single spike at zero and carries no information. `winner_final_hand_is_always_
    zero` records that the spike was checked rather than assumed -- if it ever reads False, the win
    rule has changed and the report's framing needs revisiting.
    """
    con = _connect(dataset_dir)
    decided = int(con.execute("SELECT COUNT(*) FROM games WHERE winner >= 0").fetchone()[0])
    non_zero_winners = int(con.execute(
        "SELECT COUNT(*) FROM games WHERE winner >= 0 AND winner_final_hand <> 0").fetchone()[0])

    loser = con.execute("""
        SELECT loser_final_hand AS final_hand, COUNT(*) AS games
        FROM games WHERE winner >= 0 GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    return {
        "metric": "wins_by_final_hand_size",
        "games": decided,
        "winner_final_hand_is_always_zero": non_zero_winners == 0,
        "loser_hand": [{"final_hand": int(h), "games": int(n)} for h, n in loser],
        "note": "The LOSER's final hand, because the winner's is empty by definition. A game that "
                "ends with the opponent on one card was nearly over; one that ends with them on "
                "twelve was not.",
    }


def endgame_degeneracy(dataset_dir) -> dict:
    """How often a game is decided by a last card played before the defender can answer.

    Spec 3.4: the win check runs the moment a card leaves the hand, so attacking with a single card
    ends the game outright and the defender never gets to answer. This is stated by the design
    rather than fixed by it, but a reader of a Durak result has to be able to see how much of the
    dataset it decides.
    """
    con = _connect(dataset_dir)
    total, single = con.execute("""
        SELECT COUNT(*),
               (COUNT(*) FILTER (WHERE win_from_single_card_p0)
                + COUNT(*) FILTER (WHERE win_from_single_card_p1))
        FROM games
    """).fetchone()
    by_strategy = con.execute("""
        SELECT strategy_id, COUNT(*) AS games, SUM(single) AS single_card_wins
        FROM (
            SELECT strategy_p0 AS strategy_id, win_from_single_card_p0 AS single FROM games
            UNION ALL
            SELECT strategy_p1, win_from_single_card_p1 FROM games
        )
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    total, single = int(total), int(single)
    return {
        "metric": "endgame_degeneracy",
        "games": total,
        "single_card_attack_wins": single,
        "single_card_attack_win_rate": single / total if total else None,
        "per_strategy": [{
            "strategy_id": int(sid),
            "games": int(n),
            "single_card_wins": int(w),
            "single_card_win_rate": int(w) / int(n) if n else None,
        } for sid, n, w in by_strategy],
        "note": "The win check runs before the defender responds, so a last-card attack ends the "
                "game unanswered. The rate is a property of the documented rule set (spec 3.4), "
                "not a defect, and it compresses the endgame.",
    }


def throw_in_limits(dataset_dir) -> dict:
    """The empirical distribution of table attacks against the cap the engine enforces.

    The cap is `min(defender's hand, THROWIN_CAP)` -- read from `durak.rules` rather than restated
    here, so the report cannot quote a cap the engine does not use. The empirical side is the
    observed maximum and how often play reaches it. `within_rule` compares the two: if it ever reads
    False the engine is not applying the rule it documents, and the report says so rather than
    drawing a histogram that agrees with the engine whatever the engine did.
    """
    from durak.rules import THROWIN_CAP

    con = _connect(dataset_dir)
    total, observed_max = con.execute(
        "SELECT COUNT(*), MAX(max_attacks_in_a_round) FROM games").fetchone()
    distribution = con.execute("""
        SELECT max_attacks_in_a_round AS attacks, COUNT(*) AS games
        FROM games GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()

    total = int(total)
    observed_max = int(observed_max) if observed_max is not None else 0
    return {
        "metric": "throw_in_limits",
        "games": total,
        "rule_cap": int(THROWIN_CAP),
        "rule_expression": "min(defender's current hand, THROWIN_CAP)",
        "observed_max": observed_max,
        "at_or_above_cap": sum(int(n) for a, n in distribution if int(a) >= THROWIN_CAP),
        "within_rule": observed_max <= THROWIN_CAP,
        "distribution": [{"attacks": int(a), "games": int(n)} for a, n in distribution],
    }


def deck_split(dataset_dir) -> dict:
    """Win rates per deck, for the sensitivity check, and an honest account of what is missing.

    The plan's 36/52 split is a sensitivity analysis and only exists where both decks were played.
    A deck that was not run is listed in `missing_decks` rather than emitted with a rate of zero:
    a zero on a chart reads as "measured, and it is zero", which is a different and false claim.
    """
    con = _connect(dataset_dir)
    present = [int(r[0]) for r in con.execute(
        "SELECT DISTINCT deck_size FROM games ORDER BY 1").fetchall()]

    rows = []
    for deck_size in present:
        strategies = con.execute(f"""
            SELECT strategy_id, COUNT(*) AS games,
                   COUNT(*) FILTER (WHERE won) AS wins
            FROM (
                SELECT strategy_p0 AS strategy_id, winner = 0 AS won
                FROM games WHERE deck_size = {deck_size} AND winner >= 0
                UNION ALL
                SELECT strategy_p1, winner = 1
                FROM games WHERE deck_size = {deck_size} AND winner >= 0
            )
            GROUP BY 1 ORDER BY 1
        """).fetchall()
        # Counted from the game rows, not by summing the per-strategy denominators: a game has two
        # participants, so the sum is twice the number of games and would report a deck as having
        # been played twice as often as it was.
        games = int(con.execute(
            f"SELECT COUNT(*) FROM games WHERE deck_size = {deck_size} AND winner >= 0"
        ).fetchone()[0])
        rows.append({
            "deck_size": deck_size,
            "games": games,
            "participations": sum(int(n) for _sid, n, _w in strategies),
            "strategies": [{
                "strategy_id": int(sid),
                "games": int(n),
                "wins": int(w),
                "win_rate": int(w) / int(n) if n else None,
            } for sid, n, w in strategies],
        })
    con.close()

    return {
        "metric": "deck_split",
        "rows": rows,
        "missing_decks": [size for size in (36, 52) if size not in present],
        "note": "Rates are marginal within a deck, so they depend on that deck's field. Where only "
                "one deck was played, the sensitivity check has not been run and is listed as "
                "missing rather than reported as a zero.",
    }


def rule_variant_comparison(sweep_dir) -> dict:
    """The `throwin_defense_ranks` sweep, read from the two datasets the sweep wrote.

    The throw-in rule is a genuine ambiguity in the source document rather than a tuning knob, and
    every throw-in statistic in the dataset depends on which reading was applied. `durak.
    analysis.sweeps.rule_variant_sweep` writes one dataset per reading; this reads both back
    through the same view machinery as everything else, so the two rates are computed exactly the
    way the rest of the numbers are.

    `available` is False when the sweep has not been run. A missing sweep and a measured zero look
    identical on a chart, so the report needs to be able to tell them apart.
    """
    from durak.strategies import SEARCH_UNIFORM

    root = Path(sweep_dir)
    variants = []
    for flag in (False, True):
        # The sweep names its output `rule_1` for True and `rule_0` for False.
        target = root / f"rule_{int(flag)}"
        if not (target / "games").exists():
            continue
        con = _connect(target)
        trials, wins = con.execute(f"""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE won)
            FROM (
                SELECT strategy_p0 = {SEARCH_UNIFORM} AS is_arm, winner = 0 AS won
                FROM games WHERE winner >= 0
                UNION ALL
                SELECT strategy_p1 = {SEARCH_UNIFORM}, winner = 1 FROM games WHERE winner >= 0
            )
            WHERE is_arm
        """).fetchone()
        con.close()
        trials, wins = int(trials), int(wins)
        variants.append({
            "throwin_defense_ranks": bool(flag),
            "games": trials,
            "wins": wins,
            "win_rate": wins / trials if trials else None,
            "ci": list(wilson_interval(wins, trials)) if trials else None,
        })

    return {
        "metric": "rule_variant_comparison",
        "available": bool(variants),
        "variants": variants,
        "note": "Two readings of the throw-in rule, each measured in its own tournament. The rates "
                "carry their own denominators because a bare pair of numbers invites a reader to "
                "difference them without asking whether the gap exceeds the sampling noise.",
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
    """Round floats only at the moment of writing, to SIGNIFICANT digits.

    Significant digits, not decimal places. Rounding to six decimal places is harmless for a win rate
    and destroys a p-value: `2.302e-07` becomes `0.0`, so the report would print 0.000e+00 for the one
    number that says whether the result is real. That is not a cosmetic bug -- it silently deletes the
    significance of the headline.

    Rounding earlier would let a delta cross zero between the test and the report, which is why the
    functions above never round at all.
    """
    if isinstance(node, dict):
        return {k: _round_for_output(v, digits) for k, v in node.items()}
    if isinstance(node, list):
        return [_round_for_output(v, digits) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        if node == 0.0 or not math.isfinite(node):
            return node
        return float(f"%.{digits}g" % node)
    return node


def build_all(dataset_dir, out_dir, sweeps_dir=None) -> list:
    """Run every query and write one JSON document per report, plus `meta.json`.

    Returns the written paths. `meta.json` carries the run's identity -- versions, counts, settings --
    because a report whose numbers cannot be traced to a rule set and a seed is not citable.

    `sweeps_dir` is where `durak.analysis.sweeps.rule_variant_sweep` wrote its output, if it was
    run. It is optional and its absence is recorded rather than raised: the rule-variant comparison
    is a sensitivity analysis, and a dataset without it should still produce a full set of
    aggregates with the comparison marked unavailable. `build_all` is not the thing that decides
    whether a sweep was run.
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
        "pairwise_matchups": pairwise_matchups,
        "game_length": game_length,
        "throw_in_efficiency": throw_in_efficiency,
        "take_behaviour": take_behaviour,
        "wins_by_final_hand_size": wins_by_final_hand_size,
        "endgame_degeneracy": endgame_degeneracy,
        "throw_in_limits": throw_in_limits,
        "deck_split": deck_split,
    }
    # The rule-variant sweep reads two other datasets rather than this one, so it is the only
    # document whose input is not `dataset_dir`.
    extra = {"rule_variant_comparison": lambda _dataset: rule_variant_comparison(
        sweeps_dir if sweeps_dir is not None else dataset_dir / "__no_sweeps__")}

    written = []
    payloads = {}
    for name, query in {**reports, **extra}.items():
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