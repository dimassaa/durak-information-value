"""Is every run finished, and did each one actually produce what it was supposed to?

The reason this exists: a run launched with `setsid ... &` is detached, and the shell command that
launched it returns immediately. The harness then reports that command as completed, which says nothing
about the run. Knowing a run finished therefore has to come from the run's own output, not from a
notification -- and "the process is gone" is not evidence either, since a run killed by a reboot leaves
exactly the same trace as a run that succeeded.

So completion is defined per run by an ARTEFACT with content, not by a process being absent. A missing
artefact is MISSING. An artefact present but without the line that proves the measurement completed is
INCOMPLETE -- a crashed run leaves a transcript, and a transcript is not a result.

Run it any time; it reads only files and starts nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: (run, artefact, substrings that must ALL appear). A file's existence is not completion.
EXPECTED = [
    ("R1 curve (incumbent weights)", "results/curves/depth_curve_deck_hidden_4x.txt",
     ["RESULT depth=1", "RESULT depth=2", "RESULT depth=3", "RESULT depth=4"]),
    ("R1 curve (frozen instrument)", "results/curves/depth_curve_selected_weights_4x.txt",
     ["RESULT depth=1", "RESULT depth=2", "RESULT depth=3", "RESULT depth=4"]),
    ("R2 oracle ceiling (measured)", "results/blocks/r2_oracle.txt",
     ["ORACLE vs ORACLE_UNIFORM"]),
    ("R11 pilot", "results/blocks/r11_pilot.txt", ["games "]),
]

#: The sweep has twelve vectors. Counted from transcripts rather than from the log's exit lines, which
#: the shell appends and which can be lost if the loop's trailing marker never ran.
SWEEP_EXPECTED_VECTORS = 12


def check_sweep() -> tuple[str, str]:
    """The sweep is finished when all twelve PER-VARIANT transcripts carry a RESULT line.

    Read from the per-variant files rather than from `sweep.log`, for two reasons. The RESULT line is
    indented like every other measurement line, so an anchored `^RESULT` found nothing on a sweep that
    had in fact completed. And the sweep's closing marker is a top-level command AFTER the loop, which
    means its presence depends on a shell reading a file at a byte offset -- something this project has
    already been bitten by. The per-variant transcripts are written by `timeout ... > file` and cannot be
    affected by either.
    """
    files = sorted(Path("results/weights").glob("*.txt"))
    if not files:
        return "MISSING", "no per-variant transcripts"
    with_result = [f for f in files
                   if any(line.lstrip().startswith("RESULT")
                          for line in f.read_text(errors="replace").splitlines())]
    if len(with_result) >= SWEEP_EXPECTED_VECTORS:
        return "OK", f"{len(with_result)} vectors measured"
    return "INCOMPLETE", f"{len(with_result)}/{SWEEP_EXPECTED_VECTORS} vectors measured"


def check_dataset() -> tuple[str, str]:
    """The pilot must exist on disk AND report zero anomalies. A dataset with anomalies is not a result."""
    parquet = list(Path("datasets/pilot/games").rglob("*.parquet"))
    transcript = Path("results/blocks/r11_pilot.txt")
    if not transcript.exists():
        return "MISSING", "no transcript"
    text = transcript.read_text()
    if not parquet:
        return "INCOMPLETE", "transcript present but datasets/pilot has no parquet"
    if "anomalies 0" not in text:
        return "INCOMPLETE", "the pilot's game count printed but not a clean anomaly count"
    return "OK", f"{len(parquet)} parquet file(s), anomalies 0"


#: Chain scripts are DISCOVERED from the repository, not listed here. A hardcoded list reported
#: "ALL RUNS COMPLETE" while a fifth chain was still running, because nobody had remembered to add it --
#: and a completion signal that can be wrong by omission is worse than no signal at all.
def _chain_script_names() -> list[str]:
    return sorted(p.name for p in Path(".").glob("_run_*.sh"))


def _live_chain_scripts() -> list[str]:
    """Which chain scripts still have a live process.

    Read from /proc rather than by running `ps`, so the answer cannot itself spawn a process that races
    the check. A process that exits between listing and reading is simply not counted: the artefact
    checks above are what decide completion, and this only says whether to keep waiting.
    """
    scripts = _chain_script_names()
    live = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        live.extend(s for s in scripts if s in cmdline)
    return sorted(set(live))


def main() -> int:
    rows = []
    for name, path, needles in EXPECTED:
        target = Path(path)
        if not target.exists():
            rows.append((name, "MISSING", f"{path} does not exist"))
            continue
        text = target.read_text(errors="replace")
        absent = [n for n in needles if n not in text]
        if absent:
            rows.append((name, "INCOMPLETE", f"missing: {', '.join(absent)}"))
        else:
            rows.append((name, "OK", path))

    sweep_status, sweep_detail = check_sweep()
    rows.append(("R3 weight sweep (gated)", sweep_status, sweep_detail))
    rows.append(("R11 pilot (dataset on disk)", *check_dataset()))

    width = max(len(r[0]) for r in rows)
    all_ok = True
    for name, status, detail in rows:
        all_ok &= status == "OK"
        print(f"  {name:<{width}}  {status:<11} {detail}")

    # A refusal or a traceback is a distinct state from "not finished": the run ended immediately and
    # will never produce a result, so waiting on it is pointless. EITHER marker counts.
    oracle = Path("results/blocks/r2_oracle.txt")
    if oracle.exists():
        text = oracle.read_text(errors="replace")
        if "refusing to run" in text or "Traceback" in text:
            rows.append(("R2 oracle ceiling", "FAILED",
                         "the harness refused or crashed; see the transcript"))

    running = _live_chain_scripts()
    complete = all_ok and not running
    print(f"\n  chain scripts alive: {', '.join(running) if running else 'none'}")
    print("  ALL RUNS COMPLETE" if complete else "  NOT COMPLETE")
    # The exit code follows the printed verdict. They disagreed once already, which makes a check that
    # cannot be trusted for automation worse than no check: `NOT COMPLETE` with status 0 is indistinguishable
    # from success to anything that reads the code.
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())