#!/bin/bash
# Block F: the weight surface (R4), after block E.
#
# Waiting is by PID file, never by `pgrep -f` on a script name -- a chain launched from a wrapper whose
# command line contains the script text matches ITSELF and waits forever, which is what cost block6
# 36 minutes of an idle machine.
#
# Eight workers. Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks

PIDFILE="results/blocks/blockF.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

while [ -e results/blocks/blockE.pid ]; do
  pid=$(cat results/blocks/blockE.pid 2>/dev/null || true)
  if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 30
done

echo "=== blockF (weight surface) started $(date -Is) ===" | tee -a results/blocks/blockF.log
bash _run_surface.sh
echo "R4 surface exit=$? at $(date -Is)" | tee -a results/blocks/blockF.log
echo "=== blockF done $(date -Is) ===" | tee -a results/blocks/blockF.log