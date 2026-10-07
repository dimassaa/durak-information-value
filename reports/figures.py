"""PNG figures for the README, rendered from the same JSON the HTML reports use.

One source, two renderings
--------------------------
Every figure here reads the aggregate JSON, exactly as the Chart.js blocks in the HTML do. A figure
computed from the Parquet directly would be free to disagree with the chart beside it, and a README
that disagrees with its own reports is worse than no README. Perturbing a JSON document and
requiring the PNG to move is what pins this; `tests/test_reports.py` does exactly that.

Determinism
-----------
The same JSON must produce byte-identical PNGs, for the same reason the HTML must: these files are
committed, and a figure whose bytes change on every build turns every regeneration into a diff
nobody reads. Two things would break it, and both are closed here:

* **A clock.** Matplotlib's PNG writer does not embed a timestamp by default, but a figure
  annotated with "generated at <now>" would. Nothing here reads the time.
* **The environment.** No style sheet, no font cache, no theme file. The colours are literal values
  taken from `reports.dashboard.style_tokens()`, so the figures match the reports visually without
  inheriting a dependency on whatever Matplotlib considers its default style today.

Matplotlib is imported lazily and the backend is forced to `Agg` before `pyplot`, because on a
headless machine the default backend tries to open a window and the import fails.
"""

from __future__ import annotations

import json
from pathlib import Path

from reports.dashboard import style_tokens

#: Bumped only when a figure's content changes shape. The value is not read from anywhere; it exists
#: so a reader of the PNG's metadata can tell which renderer produced it.
FIGURE_RENDERER = "durak-reports/1"

#: Read once, at import, so the colours cannot drift from the HTML reports between figures.
_TOKENS = style_tokens()


def _strategy_names() -> dict:
    from durak.strategies import policy_names

    return policy_names()


def _name(strategy_id, names: dict) -> str:
    return names.get(int(strategy_id), f"id {strategy_id}")


def _load(aggregate_dir: Path, name: str) -> dict:
    path = Path(aggregate_dir) / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(width=9.0, height=4.6):
    """A figure with the report's colours and no Matplotlib chrome.

    The spines and grid are restyled rather than left default because a figure that looks like a
    Matplotlib example next to a styled HTML report reads as two different projects.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(_TOKENS["--bg-card"])
    ax.set_facecolor(_TOKENS["--bg-card"])
    ax.tick_params(colors=_TOKENS["--text-secondary"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_TOKENS["--bg-header"])
    ax.xaxis.label.set_color(_TOKENS["--text-secondary"])
    ax.yaxis.label.set_color(_TOKENS["--text-secondary"])
    ax.title.set_color(_TOKENS["--text-primary"])
    return fig, ax


def _save(fig, path: Path) -> Path:
    """Write the PNG and close the figure.

    `metadata` is passed explicitly so the embedded `Software` string is this project's rather than
    whatever Matplotlib version wrote it. The version is left OUT on purpose: a version string in
    the bytes would change the file when the toolchain is upgraded even with identical data, which
    is precisely the churn the determinism requirement exists to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=110, facecolor=fig.get_facecolor(),
                metadata={"Software": FIGURE_RENDERER})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _colour(name: str) -> str:
    return _TOKENS.get(name, "#4f9cf9")


def figure_matchup_matrix(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """The pairwise win rates, every matchup, both sides.

    The figure the plan's results section leads with, because it is the only rate on the page that
    does not move with the field composition.
    """
    matchups = _load(aggregate_dir, "pairwise_matchups")
    rows = matchups.get("rows") or []
    if not rows:
        return None

    ordered = sorted(rows, key=lambda r: (-max(r["win_rate_a"] or 0.0, r["win_rate_b"] or 0.0),
                                          r["strategy_a"], r["strategy_b"]))
    labels = [f"{_name(r['strategy_a'], names)}\nvs {_name(r['strategy_b'], names)}" for r in ordered]

    fig, ax = _figure(width=10.5, height=max(4.6, 0.34 * len(ordered)))
    positions = range(len(ordered))
    ax.barh([p + 0.2 for p in positions], [r["win_rate_a"] for r in ordered],
            height=0.38, color=_colour("--color-accent1"), label="first named")
    ax.barh([p - 0.2 for p in positions], [r["win_rate_b"] for r in ordered],
            height=0.38, color=_colour("--color-accent3"), label="second named")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("win rate (both seat orders folded in)")
    ax.set_title("Pairwise win rates: the search arms lose to every heuristic",
                 fontsize=11, loc="left")
    ax.grid(axis="x", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(_TOKENS["--text-secondary"])
    return _save(fig, out_path)


def figure_strength_against_random(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """Rates against RANDOM only: the one rate that survives a change of strategy palette."""
    independent = _load(aggregate_dir, "opponent_independent_strength")
    rows = independent.get("rows") or []
    if not rows:
        return None

    ordered = sorted(rows, key=lambda r: -(r["win_rate"] or 0.0))
    fig, ax = _figure()
    ax.bar([_name(r["strategy_id"], names) for r in ordered],
           [r["win_rate"] for r in ordered], color=_colour("--color-accent2"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("win rate vs RANDOM")
    ax.set_title("Opponent-independent strength: measured against a fixed opponent",
                 fontsize=11, loc="left")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def figure_single_card_endgame(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """How often a game is decided by an attack the defender never answered."""
    degeneracy = _load(aggregate_dir, "endgame_degeneracy")
    rows = degeneracy.get("per_strategy") or []
    if not rows:
        return None

    fig, ax = _figure()
    ax.bar([_name(r["strategy_id"], names) for r in rows],
           [r["single_card_win_rate"] for r in rows], color=_colour("--color-accent4"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of wins from a single-card attack")
    ax.set_title("The degenerate endgame: the win check runs before the defender answers",
                 fontsize=11, loc="left")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def figure_loser_final_hand(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """The loser's hand at the end, binned, since a 36-card game runs past 25 cards."""
    hands = _load(aggregate_dir, "wins_by_final_hand_size")
    rows = hands.get("loser_hand") or []
    if not rows:
        return None

    buckets: dict = {}
    for row in rows:
        key = (int(row["final_hand"]) // 2) * 2
        buckets[key] = buckets.get(key, 0) + int(row["games"])

    fig, ax = _figure(width=10.5)
    ax.bar([f"{k}-{k + 1}" for k in sorted(buckets)], [buckets[k] for k in sorted(buckets)],
           color=_colour("--color-accent1"))
    ax.set_xlabel("loser's final hand size (binned into pairs)")
    ax.set_ylabel("games")
    ax.set_title("How close the game was, measured by what the loser was left holding",
                 fontsize=11, loc="left")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def figure_throwin_limits(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """The empirical table-attack distribution beside the cap the engine enforces."""
    limits = _load(aggregate_dir, "throw_in_limits")
    rows = limits.get("distribution") or []
    if not rows:
        return None

    cap = limits.get("rule_cap")
    fig, ax = _figure()
    colours = [_colour("--color-accent3") if row["attacks"] >= cap else _colour("--color-accent1")
               for row in rows]
    ax.bar([str(row["attacks"]) for row in rows], [row["games"] for row in rows], color=colours)
    if cap is not None:
        ax.axvline(cap - 0.5, color=_colour("--color-accent4"), linewidth=1.4, linestyle="--")
        ax.text(cap - 0.5, ax.get_ylim()[1] * 0.95, f" rule cap = {cap}", color=_colour("--color-accent4"),
                fontsize=9, va="top")
    ax.set_xlabel("maximum attacks on the table in one round")
    ax.set_ylabel("games")
    ax.set_title("Throw-in limits: observed play never exceeds the rule", fontsize=11, loc="left")
    ax.grid(axis="y", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def figure_rule_variant(aggregate_dir: Path, out_path: Path, names: dict) -> Path | None:
    """The two readings of the throw-in rule, each with its own denominator.

    Returns None when the sweep was not run, so an unmeasured sweep produces no figure rather than
    a bar at zero that reads as a measurement.
    """
    variants = _load(aggregate_dir, "rule_variant_comparison")
    if not variants.get("available"):
        return None
    rows = variants.get("variants") or []
    if not rows:
        return None

    labels = [("ranks may be thrown in" if r["throwin_defense_ranks"]
               else "ranks may not be thrown in") for r in rows]
    fig, ax = _figure()
    ax.bar(labels, [r["win_rate"] for r in rows],
           color=[_colour("--color-accent1"), _colour("--color-accent2")])
    for index, row in enumerate(rows):
        ax.text(index, row["win_rate"], f"  {row['games']} games", ha="center", va="bottom",
                color=_TOKENS["--text-secondary"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SEARCH_UNIFORM win rate")
    ax.set_title("Rule sensitivity: the throw-in reading moves the result", fontsize=11, loc="left")
    ax.grid(axis="y", color=_TOKENS["--bg-header"], linewidth=0.7)
    ax.set_axisbelow(True)
    return _save(fig, out_path)


def build_figures(aggregate_dir, out_dir) -> list:
    """Render every figure the dataset can support, and return the written paths.

    A figure whose aggregate is missing is SKIPPED, not rendered empty. An empty axes object saved
    as a PNG is a file that looks like a result and is not one, and the README would embed it.
    """
    aggregate_dir = Path(aggregate_dir)
    out_dir = Path(out_dir)
    names = _strategy_names()

    builders = [
        ("matchup_matrix.png", figure_matchup_matrix),
        ("strength_vs_random.png", figure_strength_against_random),
        ("single_card_endgame.png", figure_single_card_endgame),
        ("loser_final_hand.png", figure_loser_final_hand),
        ("throwin_limits.png", figure_throwin_limits),
        ("rule_variant.png", figure_rule_variant),
    ]

    written = []
    for filename, builder in builders:
        path = builder(aggregate_dir, out_dir / filename, names)
        if path is not None:
            written.append(path)
    return written


__all__ = ["build_figures", "FIGURE_RENDERER"]