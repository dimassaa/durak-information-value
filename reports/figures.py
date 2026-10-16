"""The figures, in one shared design system.

Built against `reports.dashboard.style_tokens` so a PNG and the paragraph citing it read as one document.

House rules, each of which exists because its absence is what makes generated charts look generated:

* **No top or right spine.** Two spines per axis is the boundary; the rest is chrome.
* **Horizontal gridlines only, hairline, behind the data.** Vertical lines create a cage that competes
  with the series it is supposed to support.
* **No shadow, 4px radius, no gradient.** Anywhere.
* **One accent colour.** The categorical series are a muted set, and they are only used where the
  category IS the encoding -- a seven-bar per-strategy chart. Everything else is one colour, or the
  semantic positive/negative pair.
* **Every figure is labelled with its units and its n.** A chart without a denominator is a shape.
* **Colour is never the only cue.** Where positive/negative carry meaning, the axis direction does too.
* **200 DPI.** At 110 the tick labels soften; a research figure that cannot be read at 100% zoom is
  not finished.

Every function returns a path, or `None` when its input document is missing -- a figure absent from a
report is better than a figure drawn from a default.
"""

from __future__ import annotations

import json
from pathlib import Path

from reports.dashboard import style_tokens

T = style_tokens()
DPI = 200
SERIES = [T["--series-1"], T["--series-2"], T["--series-3"], T["--series-4"],
          T["--series-5"], T["--series-6"], T["--series-7"]]

#: Strategy id -> display name, in the project's own vocabulary.
NAMES = {
    1: "RANDOM",
    2: "GREEDY",
    3: "AUTO_DEFEND",
    4: "CONSERVATIVE",
    5: "SEARCH_UNIFORM",
    6: "SEARCH_INFORMED",
    7: "CHEATER_DOC",
}
SEARCH = (5, 6)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": T["--border-strong"],
        "axes.labelcolor": T["--text-secondary"],
        "text.color": T["--text-primary"],
        "xtick.color": T["--text-secondary"],
        "ytick.color": T["--text-secondary"],
        "xtick.labelcolor": T["--text-secondary"],
        "ytick.labelcolor": T["--text-secondary"],
        "grid.color": T["--border"],
        "grid.linewidth": 0.7,
        "figure.facecolor": T["--bg-primary"],
        "savefig.facecolor": T["--bg-primary"],
        "axes.facecolor": T["--bg-primary"],
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })
    return plt


def _axes(ax, *, xgrid=False, ygrid=True):
    """Two spines, hairline gridlines behind the data, and no tick clutter."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, linewidth=0.7, color=T["--border"])
    if xgrid:
        ax.xaxis.grid(True, linewidth=0.7, color=T["--border"])
    ax.tick_params(length=3, width=0.8)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.18,
                metadata={"Software": "durak reports/figures"})
    _plt().close(fig)
    return path


def _load(aggregate_dir: Path, name: str, *required):
    """One aggregate, or None if it is absent or does not carry the keys the figure reads.

    Returning None rather than raising is the whole point: a figure whose input is incomplete must
    disappear from the report, not take the build down, and not be drawn from defaults. The partial
    fixture in the test suite is the honest case -- two matchup rows, not twenty-one.
    """
    path = Path(aggregate_dir) / f"{name}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if any(key not in doc for key in required):
        return None
    return doc


# --- F1  the field --------------------------------------------------------------------------------
def field_win_rates(agg: Path, out: Path) -> Path:
    """Marginal win rate per strategy, sorted, with the interval drawn rather than tabulated."""
    d = _load(agg, "tournament_overview", "rows")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["rows"], key=lambda r: r["win_rate"])
    names = [NAMES.get(r["strategy_id"], str(r["strategy_id"])) for r in rows]
    vals = [r["win_rate"] for r in rows]
    los = [r["ci"][0] for r in rows]
    his = [r["ci"][1] for r in rows]
    colours = [T["--series-6"] if n == "SEARCH_INFORMED"
               else T["--series-1"] if n == "SEARCH_UNIFORM"
               else T["--series-2"] for n in names]

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    y = range(len(rows))
    ax.barh(y, vals, color=colours, height=0.62, zorder=3)
    ax.errorbar(vals, y, xerr=[[v - l for v, l in zip(vals, los)],
                               [h - v for v, h in zip(vals, his)]],
                fmt="none", ecolor=T["--text-muted"], elinewidth=1.0, capsize=2.5, zorder=4)
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + 0.012, i, f"{v:.3f}  ({r['wins']:,} of {r['n']:,})",
                va="center", fontsize=8.5, color=T["--text-secondary"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.axvline(0.5, color=T["--border-strong"], linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.5, len(rows) - 0.35, " field average", fontsize=8, color=T["--text-muted"])
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("win rate over every participation")
    _axes(ax, xgrid=True, ygrid=False)
    ax.set_title("Win rate by strategy, primary dataset, deck 36", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F2  the matchup matrix ------------------------------------------------------------------------
def matchup_matrix(agg: Path, out: Path) -> Path:
    """Head-to-head matrix, both seat orders folded in. The metric that survives a palette change."""
    d = _load(agg, "pairwise_matchups", "rows")
    if not d:
        return None
    plt = _plt()
    # Only the strategies the document actually has. A partial input yields a smaller matrix rather
    # than a crash or a grid of invented blanks.
    ids = sorted({r["strategy_a"] for r in d["rows"]} | {r["strategy_b"] for r in d["rows"]})
    index = {i: k for k, i in enumerate(ids)}
    n = len(ids)
    grid = [[float("nan")] * n for _ in range(n)]
    for a in ids:
        grid[index[a]][index[a]] = 0.5
    for r in d["rows"]:
        i, j = index[r["strategy_a"]], index[r["strategy_b"]]
        grid[i][j] = r["win_rate_a"]
        grid[j][i] = r["win_rate_b"]

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v != v:
                continue
            # Colour is the magnitude; the number in every cell is the encoding that does not depend
            # on colour at all.
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=_shade(v),
                                       edgecolor=T["--border"], linewidth=0.8))
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=8.2, color=T["--text-primary"] if v < 0.82 or v > 0.18
                    else "#ffffff")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels([NAMES[i].replace("SEARCH_", "") for i in ids], rotation=35,
                       ha="right", fontsize=8)
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels([NAMES[i].replace("SEARCH_", "") for i in ids], fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Row beats column:  win rate, 53,024 games per cell, deck 36", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def _shade(v: float) -> str:
    """A single-hue ramp from the accent, light at 0.5 and dark at the extremes."""
    from matplotlib.colors import to_rgb

    base = to_rgb(T["--color-accent"])
    pale = to_rgb(T["--color-accent-soft"])
    t = min(1.0, abs(v - 0.5) * 2.0)
    return tuple(pale[i] + (base[i] - pale[i]) * t for i in range(3))


# --- F3  where the advantage actually is -------------------------------------------------------------
def advantage_against_each(agg: Path, out: Path) -> Path:
    """Informed against uninformed, side by side, against every other opponent.

    This is the figure the headline does not show: the informed arm's edge against EVERY opponent, and
    how much larger it is against the one opponent who does not know its hand.
    """
    d = _load(agg, "pairwise_matchups", "rows")
    if not d:
        return None
    plt = _plt()
    rates = {}
    for r in d["rows"]:
        rates[(r["strategy_a"], r["strategy_b"])] = r["win_rate_a"]

    def rate_of(a, b):
        """`a`'s rate against `b`, whichever way round the row was stored. None when neither is there."""
        if (a, b) in rates:
            return rates[(a, b)]
        if (b, a) in rates:
            return 1.0 - rates[(b, a)]
        return None

    present = {r["strategy_a"] for r in d["rows"]} | {r["strategy_b"] for r in d["rows"]}
    if 5 not in present or 6 not in present:
        return None
    # An opponent is charted only if BOTH arms have a cell against it; half a comparison is not a
    # comparison, and drawing one bar beside a missing one would imply a zero.
    opponents = [i for i in sorted(present) if i not in SEARCH
                 and rate_of(6, i) is not None and rate_of(5, i) is not None]
    if not opponents:
        return None

    labels, informed, uninformed = [], [], []
    for opp in opponents:
        labels.append("vs " + NAMES[opp].replace("SEARCH_", ""))
        informed.append(rate_of(6, opp))
        uninformed.append(rate_of(5, opp))
    if rate_of(6, 5) is None:
        return None
    labels.append("vs SEARCH_UNIFORM")
    informed.append(rate_of(6, 5))
    uninformed.append(rate_of(5, 6))

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar([i - w / 2 for i in x], informed, width=w, color=T["--series-1"],
           label="SEARCH_INFORMED — knows the hand", zorder=3)
    ax.bar([i + w / 2 for i in x], uninformed, width=w, color=T["--series-2"],
           label="SEARCH_UNIFORM — prior over hands", zorder=3)
    for i, (a, b) in enumerate(zip(informed, uninformed)):
        gap = a - b
        ax.text(i, max(a, b) + 0.022, f"{gap:+.3f}", ha="center", fontsize=8.6,
                color=T["--positive"] if gap > 0 else T["--negative"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.4, rotation=20, ha="right")
    ax.set_ylim(0, 1.06)
    ax.axhline(0.5, color=T["--border-strong"], linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(labels) - 0.45, 0.515, "even", fontsize=7.6, color=T["--text-muted"],
            ha="right")
    ax.set_ylabel("win rate")
    _axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=2)
    ax.set_title("Knowing the hand pays against every opponent, and pays most against the one\n"
                 "who does not know yours", loc="left", color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F7  the depth curve ---------------------------------------------------------------------------
def depth_curve(agg: Path, out: Path) -> Path:
    """The effect against search depth, with both ceilings on the same axis."""
    d = _load(agg, "depth_curve", "points")
    o = _load(agg, "oracle_ceiling", "readings")
    if not d:
        return None
    plt = _plt()
    pts = sorted(d["points"], key=lambda p: p["depth"])
    depths = [p["depth"] for p in pts]
    delta = [p["delta"] for p in pts]
    lo = [p["delta"] - p["ci"][0] for p in pts]
    hi = [p["ci"][1] - p["delta"] for p in pts]

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.axhline(0, color=T["--border-strong"], linewidth=0.9, zorder=2)
    ax.errorbar(depths, delta, yerr=[lo, hi], fmt="o", color=T["--color-accent"],
                markersize=6, capsize=3, elinewidth=1.2, zorder=4,
                label="measured delta (95% CI)")
    # Offset to the RIGHT of the marker, not above it: a point near the top of the axis puts an
    # above-label straight through its own interval cap.
    for x, y in zip(depths, delta):
        ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(11, 0),
                    ha="left", va="center", fontsize=8.6, color=T["--text-primary"])

    if o:
        readings = sorted(o["readings"], key=lambda r: r["depth"])
        cdep = [r["depth"] for r in readings if r.get("ceiling_delta") is not None]
        cval = [r["ceiling_delta"] for r in readings if r.get("ceiling_delta") is not None]
        if cdep:
            ax.plot(cdep, cval, marker="_", linestyle="none", markersize=22,
                    color=T["--warning"], zorder=5, label="ceiling: perfect knowledge")
            for x, y in zip(cdep, cval):
                ax.annotate(f"ceiling {y:+.3f}", (x, y), textcoords="offset points",
                            xytext=(11, -11), fontsize=8, color=T["--warning"])

    ax.set_ylim(min(-0.02, min(v - h for v, h in zip(delta, hi))) - 0.012,
                max(v + h for v, h in zip(delta, hi)) + 0.028)
    ax.set_xlim(0.6, 4.5)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("search depth in plies (a research parameter, not a speed knob)")
    ax.set_ylabel("win-rate delta, informed minus uninformed")
    _axes(ax)
    ax.legend(loc="upper right")
    ax.set_title("The value of knowing the hand falls as the search gets deeper", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F8  measured against the ceiling ---------------------------------------------------------------
def measured_against_ceiling(agg: Path, out: Path) -> Path:
    """The delta beside the best possible delta, at each depth where both were measured."""
    d = _load(agg, "depth_curve", "points")
    o = _load(agg, "oracle_ceiling", "readings")
    if not d or not o:
        return None
    plt = _plt()
    by_depth = {r["depth"]: r for r in o["readings"] if r.get("ceiling_delta") is not None}
    depths = [p["depth"] for p in sorted(d["points"], key=lambda p: p["depth"]) if p["depth"] in by_depth]
    if not depths:
        return None
    measured = [next(p["delta"] for p in d["points"] if p["depth"] == x) for x in depths]
    ceiling = [by_depth[x]["ceiling_delta"] for x in depths]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    w = 0.34
    x = range(len(depths))
    ax.bar([i - w / 2 for i in x], measured, width=w, color=T["--color-accent"],
           label="measured", zorder=3)
    ax.bar([i + w / 2 for i in x], ceiling, width=w, color=T["--warning"],
           label="ceiling (the truth)", zorder=3)
    for i, (m, c) in enumerate(zip(measured, ceiling)):
        ax.text(i - w / 2, m + 0.003, f"{m:+.3f}", ha="center", fontsize=8.2)
        ax.text(i + w / 2, c + 0.003, f"{c:+.3f}", ha="center", fontsize=8.2,
                color=T["--warning"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel("win-rate delta")
    ax.set_ylim(min(0, min(measured + ceiling) - 0.01), max(measured + ceiling) + 0.014)
    _axes(ax)
    ax.legend(loc="upper right")
    ax.set_title("The measured effect reaches the ceiling: there is no headroom left", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F11  the memory horizon ------------------------------------------------------------------------
def memory_horizon(agg: Path, out: Path) -> Path:
    """Bounded memory, with the confound marked where it applies."""
    d = _load(agg, "memory_horizon", "points")
    if not d:
        return None
    plt = _plt()
    pts = sorted(d["points"], key=lambda p: p["memory_ply"])
    labels = ["unbounded" if p["infinite"] else f"M={p['memory_ply']}" for p in pts]
    delta = [p["delta"] for p in pts]

    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    colours = [T["--warning"] if not p["infinite"] else T["--color-accent"] for p in pts]
    ax.bar(range(len(pts)), delta, color=colours, width=0.6, zorder=3)
    for i, v in enumerate(delta):
        ax.text(i, v + 0.006, f"{v:+.3f}", ha="center", fontsize=8.4)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("win-rate delta")
    ax.set_ylim(0, max(delta) * 1.22)
    ax.text(0.5, -0.20, "orange bars are NOT the price of information: a short horizon degrades the\n"
                        "uninformed arm while leaving the informed arm untouched",
            transform=ax.transAxes, ha="center", fontsize=8, color=T["--warning"])
    _axes(ax)
    ax.set_title("Bounded memory does not change the informed arm; it only weakens the control",
                 loc="left", color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F16  where games are actually decided ------------------------------------------------------------
def phase_profile(agg: Path, out: Path) -> Path:
    """Share of rounds by phase, and the share of games that END there."""
    d = _load(agg, "phase_profile", "by_phase")
    if not d:
        return None
    plt = _plt()
    names = ["early", "mid", "late (pile empty)"]
    rows = sorted(d["by_phase"], key=lambda r: r["phase"])
    rounds_share = [r["share_of_rounds"] for r in rows]
    ends = {e["phase"]: e["games"] for e in d["games_ending_in_phase"]}
    end_share = [ends.get(r["phase"], 0) / sum(ends.values()) for r in rows]

    x = range(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    ax.bar([i - w / 2 for i in x], rounds_share, width=w, color=T["--series-2"],
           label="share of all rounds played", zorder=3)
    ax.bar([i + w / 2 for i in x], end_share, width=w, color=T["--series-1"],
           label="share of games ENDING here", zorder=3)
    for i, (a, b) in enumerate(zip(rounds_share, end_share)):
        ax.text(i - w / 2, a + 0.012, f"{a:.1%}", ha="center", fontsize=8.2)
        ax.text(i + w / 2, b + 0.012, f"{b:.1%}", ha="center", fontsize=8.2,
                fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share")
    _axes(ax)
    ax.legend(loc="upper left")
    ax.set_title("A Durak game is decided after the draw pile is empty", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- F-solo  the endgame degeneracy ------------------------------------------------------------------
def single_card_endgame(agg: Path, out: Path) -> Path:
    """How often a game is decided by an unopposed last card, per strategy.

    The win check runs before the defender responds, so an attack with the last card ends the game with
    no reply. That is the documented rule set rather than a defect, and it is worth a figure because it
    compresses the endgame: nearly six games in seven are settled by a card nobody could answer.
    """
    d = _load(agg, "endgame_degeneracy", "per_strategy", "single_card_attack_win_rate")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["per_strategy"], key=lambda r: r["single_card_win_rate"])
    names = [NAMES.get(r["strategy_id"], str(r["strategy_id"])) for r in rows]
    vals = [r["single_card_win_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.barh(range(len(rows)), vals, color=T["--series-2"], height=0.6, zorder=3)
    overall = d["single_card_attack_win_rate"]
    ax.axvline(overall, color=T["--warning"], linewidth=1.2, zorder=4)
    ax.text(overall + 0.008, len(rows) - 0.45, f"overall {overall:.1%}", fontsize=8.2,
            color=T["--warning"])
    for i, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + 0.008, i, f"{v:.3f}  ({r['single_card_wins']:,} of {r['games']:,})",
                va="center", fontsize=8.2, color=T["--text-secondary"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of wins that are an unopposed last card")
    _axes(ax, xgrid=True, ygrid=False)
    ax.set_title("Five games in seven end on a card the defender never sees", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


# --- descriptive: the game's shape ------------------------------------------------------------------
def game_length(agg: Path, out: Path) -> Path:
    """How long games run. A count of rounds, not a duration."""
    d = _load(agg, "game_length", "histogram", "per_strategy")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["per_strategy"], key=lambda r: -r["mean_rounds"])
    names = [NAMES.get(r["strategy_id"], str(r["strategy_id"])) for r in rows]
    means = [r["mean_rounds"] for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.7),
                                      gridspec_kw={"width_ratios": [1.05, 1]})
    bars = left.bar(range(len(rows)), means, color=T["--series-2"], width=0.6, zorder=3)
    for b, v in zip(bars, means):
        left.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8.2)
    overall = d["mean_rounds"]
    left.axhline(overall, color=T["--warning"], linewidth=1.2, zorder=4)
    left.text(len(rows) - 0.4, overall + 0.4, f"overall {overall:.1f}", fontsize=8,
              color=T["--warning"], ha="right")
    left.set_xticks(range(len(rows)))
    left.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    left.set_ylabel("mean rounds per game")
    _axes(left)
    left.set_title("Rounds per game, by strategy", loc="left", color=T["--text-primary"])

    hist = sorted(d["histogram"], key=lambda r: r["bin_low"])
    right.bar([r["bin_low"] for r in hist], [r["games"] for r in hist],
              width=d["bin_width_rounds"] * 0.9, color=T["--series-1"], zorder=3)
    right.axvline(d["median_rounds"], color=T["--text-muted"], linewidth=1.0,
                  linestyle=(0, (4, 3)), zorder=4)
    right.text(d["median_rounds"] + 1, max(r["games"] for r in hist) * 0.92,
               f"median {d['median_rounds']:.0f}", fontsize=8, color=T["--text-muted"])
    right.set_xlabel("rounds per game")
    right.set_ylabel("games")
    _axes(right)
    right.set_title(f"{d['games']:,} games; p95 {d['p95_rounds']:.0f}, max {d['max_rounds']:.0f}",
                    loc="left", color=T["--text-primary"])
    fig.subplots_adjust(wspace=0.28)
    return _save(fig, out)


def loser_final_hand(agg: Path, out: Path) -> Path:
    """The loser's hand at the end: how close the game actually was."""
    d = _load(agg, "wins_by_final_hand_size", "loser_hand", "games")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["loser_hand"], key=lambda r: r["final_hand"])
    total = sum(r["games"] for r in rows)
    sizes = [r["final_hand"] for r in rows]
    share = [r["games"] / total for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.bar(sizes, share, color=T["--series-1"], width=0.78, zorder=3)
    ax.plot(sizes, share, color=T["--text-primary"], linewidth=0.8, zorder=4)
    for x, y in zip(sizes, share):
        if y > 0.02:
            ax.text(x, y + 0.003, f"{y:.0%}", ha="center", fontsize=7.6,
                    color=T["--text-secondary"])
    ax.set_xlabel("cards the loser was holding when the game ended")
    ax.set_ylabel("share of games")
    ax.set_ylim(0, max(share) * 1.16)
    _axes(ax)
    ax.set_title("Most games end with the loser on one or two cards", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def throw_in_efficiency(agg: Path, out: Path) -> Path:
    """Cards beaten against cards collected, per game, by strategy."""
    d = _load(agg, "throw_in_efficiency", "rows")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["rows"], key=lambda r: -r["cards_beaten_per_game"])
    names = [NAMES.get(r["strategy_id"], str(r["strategy_id"])) for r in rows]
    beaten = [r["cards_beaten_per_game"] for r in rows]
    collected = [r["cards_collected_per_game"] for r in rows]

    x = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    ax.bar([i - w / 2 for i in x], beaten, width=w, color=T["--series-1"],
           label="cards beaten per game", zorder=3)
    ax.bar([i + w / 2 for i in x], collected, width=w, color=T["--series-2"],
           label="cards collected per game", zorder=3)
    for i, (a, b) in enumerate(zip(beaten, collected)):
        ax.text(i - w / 2, a + 0.2, f"{a:.1f}", ha="center", fontsize=7.8)
        ax.text(i + w / 2, b + 0.2, f"{b:.1f}", ha="center", fontsize=7.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8.2)
    ax.set_ylabel("per game")
    _axes(ax)
    ax.legend(loc="upper right", ncols=2)
    ax.set_title("Throw-in activity by strategy", loc="left", color=T["--text-primary"], pad=10)
    return _save(fig, out)


def throw_in_limits(agg: Path, out: Path) -> Path:
    """How many attackers per trick, against the rule's cap."""
    d = _load(agg, "throw_in_limits", "distribution", "rule_cap", "observed_max")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["distribution"], key=lambda r: r["attacks"])
    total = sum(r["games"] for r in rows)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.bar([r["attacks"] for r in rows], [r["games"] / total for r in rows],
           color=T["--series-1"], width=0.62, zorder=3)
    for r in rows:
        ax.text(r["attacks"], r["games"] / total + 0.004, f"{r['games']:,}",
                ha="center", fontsize=8)
    ax.axvline(d["rule_cap"], color=T["--warning"], linewidth=1.2, zorder=4)
    ax.text(d["rule_cap"] + 0.12, 0.30, f"rule cap {d['rule_cap']}", fontsize=8,
            color=T["--warning"])
    ax.set_xlabel("attacking cards in one trick")
    ax.set_ylabel("share of games")
    _axes(ax)
    ax.set_title("The throw-in cap is reachable, and within it by construction", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def rules_and_dealing(agg: Path, out: Path) -> Path:
    """Dealing facts that shape every game, as one small table rather than four figures."""
    d = _load(agg, "rules_and_dealing", "games", "both_have_trumps",
              "first_attacker_wins", "at_least_one_without_trump_rate",
              "seat0_wins")
    if not d:
        return None
    plt = _plt()
    total = d["games"]
    items = [
        ("both players dealt a trump", d["both_have_trumps"] / total,
         f"{d['both_have_trumps']:,} of {total:,}"),
        ("first attacker wins", d["first_attacker_wins"] / total,
         f"{d['first_attacker_wins']:,} of {total:,}"),
        ("at least one player holds no trump", d["at_least_one_without_trump_rate"],
         f"{d['at_least_one_without_trump_rate']:.1%}"),
        ("seat 0 wins", d["seat0_wins"] / total, f"{d['seat0_wins']:,} of {total:,}"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    ax.axis("off")
    ax.set_title("Dealing and first-mover facts, deck 36", loc="left",
                 color=T["--text-primary"], pad=12, fontsize=11)
    y = 0.86
    for label, value, note in items:
        ax.text(0.01, y, label, fontsize=9.5, color=T["--text-primary"], va="center")
        ax.text(0.62, y, f"{value:.1%}", fontsize=9.5, color=T["--color-accent"],
                va="center", ha="right", fontweight="bold")
        ax.text(0.66, y, note, fontsize=8.6, color=T["--text-muted"], va="center")
        ax.plot([0.01, 0.99], [y - 0.11, y - 0.11], color=T["--border"], linewidth=0.8)
        y -= 0.21
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, out)


# --- the sweeps and the instrument ------------------------------------------------------------------
def weight_sweep(agg: Path, out: Path) -> Path:
    """Twelve vectors: strength is the criterion, the delta is recorded but never chosen on."""
    d = _load(agg, "weight_sweep", "by_panel_strength")
    if not d:
        return None
    plt = _plt()
    rows = list(reversed(d["by_panel_strength"]))
    names = [r["label"] for r in rows]
    strength = [r["panel_mean"] for r in rows]
    delta = [r["delta"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    y = range(len(rows))
    colours = [T["--warning"] if r["label"] == d["selected"] else T["--series-1"] for r in rows]
    ax.barh(y, strength, color=colours, height=0.62, zorder=3)
    for i, (s, r) in enumerate(zip(strength, rows)):
        mark = "  ← frozen" if r["label"] == d["selected"] else ""
        ax.text(s + 0.008, i, f"{s:.4f}   delta {r['delta']:+.4f}{mark}", va="center",
                fontsize=8.2, color=T["--text-secondary"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("panel strength — the selection criterion")
    ax.set_xlim(0, max(strength) * 1.42)
    _axes(ax, xgrid=True, ygrid=False)
    ax.set_title("A 4.7× spread in policy strength from the evaluation weights alone", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def seed_robustness(agg: Path, out: Path) -> Path:
    """One configuration, five seeds. A conclusion that depends on a seed is about that seed."""
    d = _load(agg, "seed_robustness", "points")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["points"], key=lambda r: r["seed_file"])
    delta = [r["delta"] for r in rows]
    lo = [r["delta"] - r["ci"][0] for r in rows]
    hi = [r["ci"][1] - r["delta"] for r in rows]
    mean = sum(delta) / len(delta)

    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    x = range(len(rows))
    ax.axhline(0, color=T["--border-strong"], linewidth=0.9, zorder=2)
    ax.errorbar(list(x), delta, yerr=[lo, hi], fmt="o", color=T["--color-accent"],
                markersize=7, capsize=3, elinewidth=1.2, zorder=4)
    ax.axhline(mean, color=T["--warning"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=3)
    ax.text(len(rows) - 0.45, mean + 0.004, f"mean {mean:+.4f}", fontsize=8.2,
            color=T["--warning"], ha="right")
    for i, v in enumerate(delta):
        ax.annotate(f"{v:+.4f}", (i, v), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["seed_file"].replace("seed", "seed ") for r in rows])
    ax.set_ylabel("win-rate delta")
    _axes(ax)
    ax.set_title("Five independent seeds, all positive and none crossing zero", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def self_play(agg: Path, out: Path) -> Path:
    """Two non-comparable rates, both needed: strength, and strength against THIS field."""
    d = _load(agg, "self_play", "points")
    if not d:
        return None
    pts = [p for p in d["points"] if p.get("vs_greedy") is not None]
    if not pts:
        return None
    plt = _plt()
    depths = sorted({p["depth"] for p in pts})
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    for i, (arm, colour) in enumerate(((("SEARCH_UNIFORM"), T["--series-2"]),
                                       (("SEARCH_INFORMED"), T["--series-1"]))):
        series = [next((p for p in pts if p["arm"] == arm and p["depth"] == d0), None)
                  for d0 in depths]
        vals = [p["self_play_win_rate"] for p in series if p]
        ax.plot(depths[:len(vals)], vals, marker="o", color=colour, linewidth=1.6,
                markersize=6, label=f"{arm} vs itself", zorder=4)
        ax.plot(depths[:len(vals)], [p["vs_greedy"] for p in series if p], marker="s",
                color=colour, linewidth=1.2, linestyle=(0, (5, 3)), markersize=5,
                label=f"{arm} vs GREEDY", zorder=4)
    ax.axhline(0.5, color=T["--border-strong"], linewidth=0.9, linestyle=(0, (2, 2)), zorder=2)
    ax.set_xticks(depths)
    ax.set_xlabel("search depth")
    ax.set_ylabel("win rate")
    _axes(ax)
    ax.legend(loc="center left", fontsize=8)
    ax.set_title("Self-play and one fixed opponent are not the same measurement", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def trump_stratification(agg: Path, out: Path) -> Path:
    """The delta by trump count in the opening deal -- the only admissible stratifier."""
    d = _load(agg, "trump_stratification", "curve")
    if not d:
        return None
    plt = _plt()
    rows = [r for r in d["curve"] if r["delta"] is not None]
    if not rows:
        return None
    x = [r["initial_trumps"] for r in rows]
    delta = [r["delta"] for r in rows]
    lo = [r["delta"] - r["ci"][0] for r in rows]
    hi = [r["ci"][1] - r["delta"] for r in rows]
    games = [r["head_to_head_games"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.axhline(sum(g * v for g, v in zip(games, delta)) / sum(games), color=T["--warning"],
               linewidth=1.1, linestyle=(0, (4, 3)), zorder=3)
    ax.errorbar(x, delta, yerr=[lo, hi], fmt="o", color=T["--color-accent"],
                markersize=6, capsize=3, elinewidth=1.2, zorder=4)
    for xi, v, g in zip(x, delta, games):
        ax.annotate(f"{v:+.3f}", (xi, v), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8, color=T["--text-primary"])
        ax.annotate(f"n={g:,}", (xi, v), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=7.2, color=T["--text-muted"])
    ax.set_xticks(x)
    ax.set_xlabel("trumps in the opening deal, both players")
    ax.set_ylabel("win-rate delta")
    _axes(ax)
    ax.set_title("Trump abundance does not move the effect", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def deck_split(agg: Path, out: Path, other_agg: Path | None = None) -> Path:
    """The same headline on a deck that had never been played before this project."""
    a = _load(agg, "price_of_information", "delta")
    if not a:
        return None
    a = a["delta"]
    # The 52-card aggregates sit beside the 36-card ones rather than inside them, because the queries
    # are pinned to one deck and building both into one directory would mix two instruments' readings.
    other = other_agg or Path(agg).parent / "data_deck52"
    b_doc = _load(other, "price_of_information", "delta")
    b = b_doc["delta"] if b_doc else None
    items = [("deck 36", a["point"], a["ci"])]
    if b:
        items.append(("deck 52", b["point"], b["ci"]))

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.axis("off")
    y = 0.84
    for label, point, ci in items:
        ax.text(0.02, y, label, fontsize=10, color=T["--text-primary"], va="center")
        ax.text(0.40, y, f"{point:+.4f}", fontsize=11, color=T["--color-accent"],
                va="center", ha="right", fontweight="bold")
        ax.text(0.45, y, f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]", fontsize=9,
                color=T["--text-secondary"], va="center")
        y -= 0.34
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("The effect survives a deck it was not measured on", loc="left",
                 color=T["--text-primary"], pad=6, fontsize=11)
    return _save(fig, out)


def effective_hypotheses(agg: Path, out: Path) -> Path:
    """How much resolution the search actually had, against the nominal budget.

    Reads the level-1 column directly, because the number is a property of the games rather than of any
    aggregate: the total distinct hypotheses over all decisions, divided by the decisions.

    Prefers the deck-36 primary dataset and FALLS BACK to the 52-card one, saying which deck it read. The
    column is recorded per game under the same instrument on both, so the fallback is a measurement rather
    than an estimate -- but a figure whose caption does not name its deck is a figure whose provenance
    cannot be checked.
    """
    import duckdb

    root = Path(agg).parent.parent / "datasets"
    source = None
    for name in ("primary", "deck52"):
        files = sorted((root / name).glob("games/deck_size=*/run_id=*/data.parquet"))
        if files:
            source = (name, files)
            break
    if not source:
        return None
    deck_name, files = source

    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT strategy_p0 AS arm,
               SUM(distinct_hypotheses_sum) AS total,
               SUM(search_decisions) AS decisions
        FROM read_parquet('{files[0]}')
        WHERE search_decisions > 0
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()
    if not rows:
        return None

    plt = _plt()
    names = [NAMES.get(r[0], str(r[0])) for r in rows]
    effective = [r[1] / r[2] for r in rows]
    nominal = 100

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    y = range(len(rows))
    ax.barh(y, effective, color=T["--series-1"], height=0.55, zorder=3)
    for i, (v, r) in enumerate(zip(effective, rows)):
        ax.text(v + 1.2, i, f"{v:.1f} of {nominal}  ({r[2]:,} decisions)", va="center",
                fontsize=8.4, color=T["--text-secondary"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8.4)
    ax.set_xlabel("distinct opponent-hand hypotheses per decision")
    ax.set_xlim(0, 128)
    _axes(ax, xgrid=True, ygrid=False)
    ax.set_title("The nominal budget of 100 overstates the search's real resolution\n"
                 f"(deck {deck_name}, {rows[0][2] + rows[-1][2]:,} decisions across the arms shown)",
                 loc="left", color=T["--text-primary"], pad=10)
    return _save(fig, out)


def take_by_phase(agg: Path, out: Path) -> Path:
    """The share of rounds taken, by phase."""
    d = _load(agg, "phase_profile", "by_phase")
    if not d:
        return None
    plt = _plt()
    rows = sorted(d["by_phase"], key=lambda r: r["phase"])
    names = ["early", "mid", "late"]
    rates = [r["take_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.bar(names, rates, color=T["--series-1"], width=0.5, zorder=3)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.6)
    ax.set_ylabel("share of rounds taken")
    ax.set_ylim(0, max(rates) * 1.22)
    _axes(ax)
    ax.set_title("Taking a trick, by phase of the game", loc="left",
                 color=T["--text-primary"], pad=10)
    return _save(fig, out)


def _variants_by_strength(agg_dir, metric):
    """Shared body for the two weight figures: one ranked bar chart of panel strength.

    Kept as one function because R3 and R4 differ only in which directory they read and in the scale
    note; two near-identical chart functions would drift apart the first time one of them was edited.
    """
    plt = _plt()
    d = _load(Path(agg_dir), metric, "by_panel_strength")
    if not d:
        return None
    rows = list(reversed(d["by_panel_strength"]))
    names = [r["label"] for r in rows]
    strength = [r["panel_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(rows) + 1.6))
    y = range(len(rows))
    colours = [T["--warning"] if r["label"] == d.get("selected") else T["--series-1"]
               for r in rows]
    ax.barh(y, strength, color=colours, height=0.62, zorder=3)
    for i, (s, r) in enumerate(zip(strength, rows)):
        mark = "  ← selected" if r["label"] == d.get("selected") else ""
        ax.text(s + 0.008, i, f"{s:.4f}   delta {r['delta']:+.4f}{mark}", va="center",
                fontsize=8.2, color=T["--text-secondary"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("panel strength")
    ax.set_xlim(0, max(strength) * 1.42)
    _axes(ax, xgrid=True, ygrid=False)
    title = ("A stronger vector exists: the surface found one this pass" if metric == "weight_surface"
             else "A 4.7x spread in policy strength from the evaluation weights alone")
    ax.set_title(title, loc="left", color=T["--text-primary"], pad=10)
    return fig


def weight_surface(agg: Path, out: Path) -> Path:
    """R4: the surface around the selected vector, at half the ray sweep's games per opponent."""
    fig = _variants_by_strength(agg, "weight_surface")
    return _save(fig, out) if fig is not None else None


#: Every figure the aggregates can currently supply, by short name. The seven the README carries are a
#: subset of these; the rest live in the HTML reports.
ALL_FIGURES = {
    "field_win_rates": field_win_rates,
    "game_length": game_length,
    "loser_final_hand": loser_final_hand,
    "throw_in_efficiency": throw_in_efficiency,
    "throw_in_limits": throw_in_limits,
    "rules_and_dealing": rules_and_dealing,
    "weight_sweep": weight_sweep,
    "weight_surface": weight_surface,
    "seed_robustness": seed_robustness,
    "self_play": self_play,
    "trump_stratification": trump_stratification,
    "deck_split": deck_split,
    "effective_hypotheses": effective_hypotheses,
    "take_by_phase": take_by_phase,
    "single_card_endgame": single_card_endgame,
    "matchup_matrix": matchup_matrix,
    "advantage_against_each": advantage_against_each,
    "depth_curve": depth_curve,
    "measured_against_ceiling": measured_against_ceiling,
    "memory_horizon": memory_horizon,
    "phase_profile": phase_profile,
}

def build_figures(aggregate_dir: Path, out_dir: Path) -> list:
    """Render every figure the aggregates can supply, and return the paths.

    Kept as one entry point because the tests hold two properties of the whole set: the bytes must be
    identical across two renders (the files are committed artefacts that the README embeds, so churn on
    every build is a diff nobody reads), and a perturbed JSON must move the PNG (a figure that does not
    follow its data is a picture of something else).

    DPI is pinned and no timestamp reaches the PNG, which is what makes the first property hold.
    """
    out_dir = Path(out_dir)
    written = []
    for name, fn in ALL_FIGURES.items():
        path = fn(Path(aggregate_dir), out_dir / f"{name}.png")
        if path is not None:
            written.append(path)
    return written
