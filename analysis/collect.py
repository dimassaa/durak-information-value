"""Regenerate a dataset from a manifest, then emit the report JSON beside it.

The driver the whole pipeline is run through, and deliberately a thin one. It reads a manifest, calls
the runner, and writes the queries' output. It contains no statistics and no aggregation of its own,
because a driver that computed something would make "what produced this number" depend on how it was
invoked.

Run it as a module:

    python -m analysis.collect --manifest path/to/manifest.json --out data/run1

The manifest that was used is copied next to the output. A dataset whose manifest is somewhere else is
a dataset nobody can rebuild, and the manifest is cheap to carry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from analysis.queries import build_all
from durak.runner import RunManifest, run_tournament
from durak.tournament import pairing_grid


def load_manifest(path) -> RunManifest:
    """Rebuild a :class:`RunManifest` from its JSON form.

    Round-trips through the class's own ``from_json`` rather than reading fields by hand, so a field
    added to the manifest cannot be silently dropped here. That drop would be invisible until someone
    tried to reproduce the run and found the setting missing.
    """
    return RunManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(manifest_path, out_dir, workers: int = 0, block_size: int | None = None):
    """Play the manifest's grid and write the dataset plus its report JSON.

    Returns the runner's :class:`~durak.runner.RunSummary`. The report JSON is written afterwards and
    deliberately not fused into the summary: the summary describes the simulation, the JSON describes
    the claims, and a report built from a summary could not be re-derived from the dataset alone.
    """
    manifest = load_manifest(manifest_path)
    if block_size is not None:
        # Rebuild rather than mutate: `RunManifest` is frozen, and silently ignoring the override would
        # make the written manifest disagree with the block size actually used.
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        data["block_size"] = int(block_size)
        manifest = RunManifest.from_dict(data)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_tournament(manifest, out_dir, workers=workers)

    # The manifest beside the data, so the dataset carries its own reproducibility contract.
    shutil.copyfile(manifest_path, out_dir / "manifest.json")

    build_all(out_dir, out_dir / "analysis")
    return summary


def _grid_preview(manifest: RunManifest) -> dict:
    """What the manifest asks for, for the log line before a long run starts."""
    grid = pairing_grid(manifest.strategies, deck_sizes=manifest.deck_sizes,
                        regimes=manifest.regimes)
    return {
        "cells": len(grid),
        "games_per_pairing": manifest.games_per_pairing,
        "total_games": len(grid) * manifest.games_per_pairing,
        "block_size": manifest.block_size,
        "strategies": manifest.strategies,
        "deck_sizes": manifest.deck_sizes,
        "regimes": manifest.regimes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.collect",
        description="Regenerate a dataset from a manifest and emit the report JSON.",
    )
    parser.add_argument("--manifest", required=True, help="path to a RunManifest JSON file")
    parser.add_argument("--out", required=True, help="output directory for the dataset and its JSON")
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 means one per CPU")
    parser.add_argument("--block-size", type=int, default=None,
                        help="override the manifest's block size")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    preview = _grid_preview(manifest)
    print(json.dumps({"running": preview}, sort_keys=True), file=sys.stderr)

    summary = run(args.manifest, args.out, workers=args.workers, block_size=args.block_size)

    print(json.dumps({
        "games": summary.games,
        "anomalies": summary.anomalies,
        "publishable": summary.publishable,
        "elapsed_seconds": round(summary.elapsed_seconds, 3),
        "out": str(Path(args.out) / "analysis"),
    }, sort_keys=True))
    return 0 if summary.publishable else 1


if __name__ == "__main__":
    raise SystemExit(main())