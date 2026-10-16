"""Verify that every number quoted in the README is the number in the aggregates.

A README is prose, so nothing structurally ties it to the data it claims to summarise. The figure check
(`reports.readme_figures.missing`) covers the images; this covers the digits, and it is the only thing that
catches a stale figure after the data moves.

Each check names the claim, states where the truth lives, and states the TRUTH'S OWN FORMATTING rather than
a literal. The first version of this file compared `"+6.03"` against `f"{point:+.2f}"`, which is `"+0.06"`
-- the checker reported a mismatch against a correct README and would have reported a match against a wrong
one. Comparing formatted text to a different formatted text is not a check.

Run it:  `.venv/bin/python reports/verify_readme.py`
"""

from __future__ import annotations

import json
from pathlib import Path

AGG = Path("analysis/data")
README = Path("README.md")


def _doc(name: str) -> dict:
    return json.loads((AGG / f"{name}.json").read_text(encoding="utf-8"))


def checks() -> list:
    """(claim as the README states it, whether the aggregate agrees)."""
    poi = _doc("price_of_information")
    delta, curve, ceil = poi["delta"], _doc("depth_curve"), _doc("oracle_ceiling")
    by_depth = {p["depth"]: p for p in curve["points"]}
    ceilings = {r["depth"]: r for r in ceil["readings"]}
    endgame, phase = _doc("endgame_degeneracy"), _doc("phase_profile")
    sweep, seeds = _doc("weight_sweep"), _doc("seed_robustness")
    panel = sweep["by_panel_strength"]

    late = next(r for r in phase["by_phase"] if r["phase"] == 2)
    endings = {e["phase"]: e["games"] for e in phase["games_ending_in_phase"]}

    return [
        ("headline +6.03 points", f"{delta['point'] * 100:.2f}" == "6.03"),
        ("headline CI", [round(x, 4) for x in delta["ci"]] == [0.0518, 0.0688]),
        ("paired games 53,024", poi["informed"]["n"] == 53024),
        ("discordant 28,110 vs 24,914",
         (poi["informed"]["wins"], poi["uniform"]["wins"]) == (28110, 24914)),
        ("depth 1 +0.0911", round(by_depth[1]["delta"], 4) == 0.0911),
        ("depth 2 +0.0713", round(by_depth[2]["delta"], 4) == 0.0713),
        ("depth 3 +0.0460", round(by_depth[3]["delta"], 4) == 0.0460),
        ("depth 4 +0.0130", round(by_depth[4]["delta"], 4) == 0.0130),
        ("ceiling depth 2 +0.0557", round(ceilings[2]["ceiling_delta"], 4) == 0.0557),
        ("ceiling depth 4 +0.0303", round(ceilings[4]["ceiling_delta"], 4) == 0.0303),
        ("oracle ties informed arm", round(ceilings[4]["oracle_vs_informed_delta"], 4) == -0.0007),
        ("84.8% unopposed last cards",
         f"{endgame['single_card_attack_win_rate'] * 100:.1f}" == "84.8"),
        ("50.5% of rounds in the late phase", round(late["share_of_rounds"] * 100, 1) == 50.5),
        ("99.95% of games end late",
         round(endings[2] / sum(endings.values()) * 100, 2) == 99.95),
        ("panel spread 0.148-0.688",
         (round(min(p["panel_mean"] for p in panel), 3),
          round(max(p["panel_mean"] for p in panel), 3)) == (0.148, 0.688)),
        ("delta spread -0.036..+0.061",
         (round(min(p["delta"] for p in panel), 3),
          round(max(p["delta"] for p in panel), 3)) == (-0.036, 0.061)),
        ("5 seeds, all positive",
         len(seeds["points"]) == 5 and seeds["all_positive"] is True),
        ("1,140,016 games, 0 anomalies",
         _doc("data_quality")["games"] == 1140016 and _doc("data_quality")["anomalies"] == 0),
    ]


def main() -> int:
    rows = checks()
    bad = [name for name, ok in rows if not ok]
    for name, ok in rows:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    print(f"\n{len(rows)} numbers checked, {len(bad)} disagree with the aggregates")
    for name in bad:
        print(f"  DISAGREES: {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())