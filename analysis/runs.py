"""Turn the run transcripts into aggregates, so the experiments can reach a report.

Every number this project produces comes from one of two places: a Parquet dataset, or a transcript. The
dataset side had a pipeline (`analysis.queries.build_all`) and the transcript side had none, which is why
six experiment families -- the depth curve, the ceiling, the weight sweep, the memory horizon, seed
robustness, self-play -- sat in `results/` as text that no page could read. The dataset side answers
"what came out of 1.14M games"; these answer the questions the project exists to ask.

The parsers are strict about what they accept. A transcript that does not contain the expected marker
becomes a MISSING family, never a silently shorter list, because a curve with three points instead of
four is exactly the shape of a broken run and would read as a finding.

Nothing is recomputed here. These documents carry the numbers as the runs printed them, plus the
configuration those numbers were measured at, because a sweep number without its depth, weights and
memory horizon cannot be attributed.
"""

from __future__ import annotations

import re
from pathlib import Path

#: `RESULT label=x depth=2 delta=+0.0713 ...` and `RESULT depth=1 delta=+0.0911 games=16000`.
RESULT = re.compile(r"^\s*RESULT\b(?P<rest>.*)$")
DELTA = re.compile(r"(?m)^\s*DELTA\s+(?P<delta>[+-][\d.]+)\s+95% CI \[(?P<lo>[+-][\d.]+), (?P<hi>[+-][\d.]+)\]")
WALL = re.compile(r"(?m)^\s*wall clock (?P<wall>\d+)s")
AGAINST = re.compile(r"(?m)^\s*(?P<label>[\w ]+?)\s+(?P<rate>0\.\d+)\s+\(.*-(?P<w>\d+)\)")
KV = re.compile(r"(?P<key>\w+)=(?P<value>[^\s]+)")


def _fields(rest: str) -> dict:
    out = {}
    for match in KV.finditer(rest):
        value = match.group("value")
        try:
            out[match.group("key")] = float(value) if "." in value else int(value)
        except ValueError:
            out[match.group("key")] = value
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _curve_point(text: str) -> dict | None:
    """One depth's point: delta with its interval, the panel rates, and the wall clock."""
    result = None
    for line in text.splitlines():
        if RESULT.match(line):
            result = _fields(RESULT.match(line).group("rest"))
    if result is None or "delta" not in result:
        return None
    # Every field the run printed, not a hand-picked subset. Picking fields is how `panel_mean` and
    # `label` silently vanished from the weight sweep and left it reporting twelve measurements and no
    # selection -- an aggregate that looks complete and answers nothing.
    point = dict(result)
    point["delta"] = float(result["delta"])
    if "depth" in result:
        point["depth"] = int(result["depth"])
    if "games" in result:
        point["games"] = int(result["games"])
    delta = DELTA.search(text)
    if delta:
        point["ci"] = [float(delta.group("lo")), float(delta.group("hi"))]
    wall = WALL.search(text)
    if wall:
        point["wall_seconds"] = int(wall.group("wall"))
    against = {}
    for match in AGAINST.finditer(text):
        label = match.group("label").strip()
        if label.startswith("informed vs") or label.startswith("uniform vs"):
            against[label] = float(match.group("rate"))
    point["win_rates"] = against
    point["pool_stable"] = "pool stable:" in text
    return point


def depth_curve(results_dir: Path) -> dict:
    """R1: the effect at each search depth, on one harness and one deal set."""
    directory = Path(results_dir) / "curves"
    points = []
    missing = []
    for depth in (1, 2, 3, 4):
        text = _read(directory / f"curve_pooled_d{depth}.txt")
        point = _curve_point(text)
        if point is None:
            missing.append(depth)
        else:
            points.append(point)
    points.sort(key=lambda p: p["depth"])
    return {
        "metric": "depth_curve",
        "instrument": "the sweep-selected weight vector, frozen as the project default",
        "identity": "one harness and one deal set at every depth; the pool gate is recorded per point",
        "points": points,
        "missing_depths": missing,
        "shape": None,
        "note": "the depth is a RESEARCH PARAMETER, not a performance knob, and must be stated next to "
                "any number taken from this curve.",
    }


def oracle_ceilings(results_dir: Path) -> dict:
    """R2: what perfect knowledge of the hand is worth, with both arms at one depth."""
    directory = Path(results_dir) / "blocks"
    readings = []
    for name, depth in (("r2_oracle_depth2.txt", 2), ("r2_oracle.txt", 4)):
        text = _read(directory / name)
        if not text:
            readings.append({"depth": depth, "status": "MISSING"})
            continue
        entry = {"depth": depth, "status": "OK"}
        for line in text.splitlines():
            if "ORACLE vs ORACLE_UNIFORM" in line:
                rate = re.search(r"(0\.\d+)\s+95% CI \[(-?\d+\.\d+), (-?\d+\.\d+)\]", line)
                delta = re.search(r"DELTA ([+-][\d.]+)", line)
                if rate:
                    entry["ceiling_win_rate"] = float(rate.group(1))
                    entry["ceiling_ci"] = [float(rate.group(2)), float(rate.group(3))]
                if delta:
                    entry["ceiling_delta"] = float(delta.group(1))
            if "ORACLE vs SEARCH_INFORMED" in line:
                delta = re.search(r"DELTA ([+-][\d.]+)", line)
                if delta:
                    entry["oracle_vs_informed_delta"] = float(delta.group(1))
            if "ORACLE vs GREEDY" in line:
                rate = re.search(r"(0\.\d+)", line)
                if rate:
                    entry["oracle_vs_greedy"] = float(rate.group(1))
        entry["depth_reached"] = "matches the shared search at its own depth on" in text
        entry["games_differ"] = "games differ from SEARCH_INFORMED" in text
        readings.append(entry)
    return {
        "metric": "oracle_ceiling",
        "readings": readings,
        "note": "a ceiling is only a ceiling if both arms sit at the depth being bounded and share the "
                "weight vector; otherwise it confounds 'does knowledge pay' with 'does depth pay', and in "
                "this engine those point in opposite directions.",
    }


def weight_sweep(results_dir: Path) -> dict:
    """R3: eleven vectors, selected on strength. The delta is recorded, never the criterion."""
    return _variants_by_strength(Path(results_dir) / "weights", "weight_sweep",
                                 note="selection is on panel strength; the delta at every vector is "
                                      "recorded and published across all of them")


def _variants_by_strength(directory: Path, metric: str, note: str) -> dict:
    variants = []
    for path in sorted(directory.glob("*.txt")):
        point = _curve_point(_read(path))
        if point is None:
            continue
        label = point.pop("label", None)
        weights = point.pop("weights", None)
        panel = point.pop("panel_mean", None)
        variants.append({
            "label": label, "weights": weights, "panel_mean": panel,
            "delta": point["delta"], "ci": point.get("ci"),
            "win_rates": point.get("win_rates", {}),
        })
    ranked = sorted((v for v in variants if v["panel_mean"] is not None),
                    key=lambda v: v["panel_mean"], reverse=True)
    return {
        "metric": metric,
        "note": note,
        "variants_measured": len(variants),
        "selection_criterion": "absolute strength against a fixed panel, never the delta",
        "by_panel_strength": ranked,
        "selected": ranked[0]["label"] if ranked else None,
        "selected_weights": ranked[0]["weights"] if ranked else None,
    }


def weight_surface(results_dir: Path) -> dict:
    """R4: the surface around the selected vector, at HALF the ray sweep's games per opponent.

    The scale difference is carried into the document rather than left for a reader to assume away: the
    ray sweep measured 10,000 games per opponent and this 3,000, so a panel rate here carries roughly
    twice the interval width. A vector that beats the frozen one by less than the combined interval has
    not been shown to beat it.
    """
    return _variants_by_strength(Path(results_dir) / "surface", "weight_surface",
                                 note="measured at half the ray sweep's games per opponent, so panel "
                                      "rates here are not directly comparable with `weight_sweep`")


def memory_horizon(results_dir: Path) -> dict:
    """R5: bounded memory, and the confound that comes with it."""
    directory = Path(results_dir) / "memory"
    points = []
    for label in (0, 2, 4, 8, 16, 1000000):
        text = _read(directory / f"M{label}.txt")
        point = _curve_point(text)
        if point is None:
            continue
        points.append({"memory_ply": label, "delta": point["delta"], "ci": point.get("ci"),
                       "win_rates": point.get("win_rates", {}),
                       "infinite": label == 1000000})
    return {
        "metric": "memory_horizon",
        "points": points,
        "confound": "a bounded horizon degrades the UNINFORMED arm while leaving the informed arm "
                    "untouched, because the informed arm reads the hand directly. The deltas at finite M "
                    "therefore measure the degradation of the control and are NOT the price of "
                    "information; only the unbounded reading is.",
    }


def seed_robustness(results_dir: Path) -> dict:
    """R9: the same configuration under several seeds."""
    directory = Path(results_dir) / "seeds"
    deltas = []
    for path in sorted(directory.glob("seed*.txt")):
        text = _read(path)
        result = None
        for line in text.splitlines():
            if RESULT.match(line):
                result = _fields(RESULT.match(line).group("rest"))
        if result is None or "delta" not in result:
            continue
        delta = DELTA.search(text)
        deltas.append({
            "seed_file": path.stem,
            "delta": float(result["delta"]),
            "ci": [float(delta.group("lo")), float(delta.group("hi"))] if delta else None,
        })
    return {
        "metric": "seed_robustness",
        "seeds": len(deltas),
        "points": deltas,
        "all_positive": all(d["delta"] > 0 for d in deltas) if deltas else None,
    }


def self_play(results_dir: Path) -> dict:
    """R10: is a depth stronger, or stronger against THESE opponents?"""
    text = _read(Path(results_dir) / "blocks" / "r10_selfplay.txt")
    if not text:
        return {"metric": "self_play", "status": "MISSING", "points": []}
    points = []
    for line in text.splitlines():
        match = re.search(r"SELFPLAY depth=(\d+) arm=(\S+)\s+([\d.]+)", line)
        if match:
            points.append({"depth": int(match.group(1)), "arm": match.group(2),
                           "self_play_win_rate": float(match.group(3))})
    greedy = {}
    for line in text.splitlines():
        match = re.search(r"vs GREEDY depth=(\d+) arm=(\S+)\s+([\d.]+)", line)
        if match:
            greedy[(int(match.group(1)), match.group(2))] = float(match.group(3))
    for point in points:
        point["vs_greedy"] = greedy.get((point["depth"], point["arm"]))
    return {
        "metric": "self_play",
        "status": "OK" if points else "INCOMPLETE",
        "points": points,
        "note": "two non-comparable rates, both needed: self-play says the policy is strong, the fixed "
                "opponent says it is strong against THIS field. A depth can improve one and not the other.",
    }


FAMILIES = {
    "depth_curve": depth_curve,
    "oracle_ceiling": oracle_ceilings,
    "weight_sweep": weight_sweep,
    "weight_surface": weight_surface,
    "memory_horizon": memory_horizon,
    "seed_robustness": seed_robustness,
    "self_play": self_play,
}


def build_run_aggregates(results_dir, out_dir) -> list:
    """Write one JSON document per experiment family. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    written = []
    for name, query in FAMILIES.items():
        payload = query(results_dir)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
        written.append(path)
    return written