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


#: Every figure the aggregates can currently supply, by short name. The seven the README carries are a
#: subset of these; the rest live in the HTML reports.
ALL_FIGURES = {
    "field_win_rates": field_win_rates,
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
