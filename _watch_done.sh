#!/bin/bash
# Waits for every chain script to exit, then writes results/RUNS_COMPLETE.md from _check_runs.py.
#
# Exists because nothing else will tell me. A run launched with `setsid ... &` is detached, and the shell
# command that launched it returns at once, so the harness reports THAT command as completed and the
# notification says nothing about the run. Polling for it costs a turn every time and depends on me
# asking. This watcher does the waiting, so the only thing anyone has to do afterwards is read one file.
#
# A chain that dies leaves a transcript but no sentinel, and the sentinel is written only when the check
# passes -- so the sentinel's presence is evidence, not an assumption. Absence means read the checker's
# table, which distinguishes MISSING from INCOMPLETE.
set -u
cd /home/dimass/opencode/durak
mkdir -p results

while true; do
  alive=$(ps -eo args --no-headers | grep -cE "_run_(block1|block2|memory|weights)\.sh" || true)
  [ "$alive" -eq 0 ] && break
  sleep 60
done

echo "=== all chain scripts exited $(date -Is) ===" > results/RUNS_COMPLETE.md
.venv/bin/python _check_runs.py >> results/RUNS_COMPLETE.md 2>&1
echo "" >> results/RUNS_COMPLETE.md
echo "Written $(date -Is). Re-check at any time with: .venv/bin/python _check_runs.py" \
  >> results/RUNS_COMPLETE.md