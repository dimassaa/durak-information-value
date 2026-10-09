#!/bin/bash
# Block B1, in the order docs/RUNS.md gives it: R2 (oracle ceiling) then R11 (pilot at 10%) then the
# Step-1 audit over the pilot. R1 (the depth curve) is already running and is waited for rather than
# run alongside.
#
# The wait is not politeness. _run_curve.sh clears durak/__pycache__ before every depth, and a second
# multi-process run compiling against a cache deleted underneath it is the numba cache race this project
# has already paid for twice. run_tournament warms the cache under a lock; _oracle_ceiling.py drives its
# own pool and does not, so the two together are exactly the failure the warm-up was written to remove.
#
# Depth is stated, not inherited. Both oracle arms run at depth 4 because the ceiling is only a ceiling
# if it bounds the depth the curve measured; the script also hard-fails if the oracle does not answer as
# the shared search at its own depth, or if the two arms produce byte-identical games.
#
# Nothing is written to /tmp. datasets/ holds the pilot, results/blocks/ holds every transcript.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks
export DURAK_WORKERS=8

while pgrep -f "python _depth_curve.py" > /dev/null; do sleep 20; done
echo "=== block B1 resumed $(date -Is), curve finished ===" | tee -a results/blocks/block1.log

# --- R2, oracle ceiling ---------------------------------------------------------------------
rm -rf durak/__pycache__ durak/strategies/__pycache__
DURAK_SEARCH_DEPTH=4 DURAK_ORACLE_DEPTH=4 ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
  timeout 21600 .venv/bin/python _oracle_ceiling.py > results/blocks/r2_oracle.txt 2>&1
echo "R2 oracle exit=$? at $(date -Is)" | tee -a results/blocks/block1.log

# --- R11, the 10% pilot, then the audit over it ---------------------------------------------
rm -rf datasets/pilot durak/__pycache__ durak/strategies/__pycache__
PILOT_GAMES_PER_PAIRING=2650 timeout 21600 .venv/bin/python _pilot.py > results/blocks/r11_pilot.txt 2>&1
echo "R11 pilot exit=$? at $(date -Is)" | tee -a results/blocks/block1.log
timeout 3600 .venv/bin/python _audit.py >> results/blocks/r11_pilot.txt 2>&1
echo "R11 audit exit=$? at $(date -Is)" | tee -a results/blocks/block1.log