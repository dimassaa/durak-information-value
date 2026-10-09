#!/bin/bash
# Waits for every measurement chain to exit, then writes results/RUNS_COMPLETE.md.
#
# Waiting is by PID file, never by `pgrep -f` on a script name. A chain launched from a shell wrapper
# whose command line contains its own script text matches ITSELF, and then waits forever for a process
# that cannot exit -- which is exactly what happened to block6, for 36 minutes on an idle machine.
#
# A chain that dies leaves a transcript but no sentinel, and the sentinel is written only when the check
# passes. So the sentinel's presence is evidence; its absence means read the checker's table, which
# separates MISSING from INCOMPLETE and FAILED.
set -u
cd /home/dimass/opencode/durak
mkdir -p results

while true; do
  alive=0
  for pidfile in results/blocks/*.pid; do
    [ -e "$pidfile" ] || continue
    pid=$(cat "$pidfile" 2>/dev/null || true)
    [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 60
done

echo "=== all chains exited $(date -Is) ===" > results/RUNS_COMPLETE.md
.venv/bin/python _check_runs.py >> results/RUNS_COMPLETE.md 2>&1
echo "" >> results/RUNS_COMPLETE.md
echo "Written $(date -Is). Re-check any time with: .venv/bin/python _check_runs.py" \
  >> results/RUNS_COMPLETE.md