"""The README's figures, by the short name it cites them under.

The README and the HTML reports must show the SAME image files, or they drift into two truths. This map is
what lets `reports.build` verify that: it renders the set the README references and the build fails if a
name in the README has no file on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Every figure the README cites, in the order it cites them.
README_FIGURES = (
    "advantage_against_each",
    "field_win_rates",
    "matchup_matrix",
    "depth_curve",
    "measured_against_ceiling",
    "single_card_endgame",
    "phase_profile",
    "memory_horizon",
)

_ASSET = re.compile(r"assets/figures/([a-z0-9_]+)\.png")


def cited_by(readme: Path) -> list[str]:
    """The figure short names the README actually embeds, in document order."""
    if not readme.exists():
        return []
    seen: list[str] = []
    for name in _ASSET.findall(readme.read_text(encoding="utf-8")):
        if name not in seen:
            seen.append(name)
    return seen


def missing(readme: Path, figures_dir: Path) -> list[str]:
    """Figures the README embeds but that do not exist. A broken image is a broken claim."""
    return [n for n in cited_by(readme)
            if not (figures_dir / f"{n}.png").exists()]