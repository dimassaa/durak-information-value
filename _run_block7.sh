#!/bin/bash
# Everything measurement-side that remains, in one chain, with a worker budget that fits the machine.
#
# WAITING IS BY PID FILE, NEVER BY `pgrep -f` ON A SCRIPT NAME. That is not a style preference: a chain
# launched from a shell wrapper whose command line contains the wrapper's own heredoc will match ITSELF,
# and block6 sat waiting for a process that could never exit -- 36 minutes of an idle machine, caused by
# a pattern that had worked four times before. `kill -0` on a recorded PID cannot match the waiter.
#
# The budget is sequential at eight workers on twelve cores. Running two eight-worker jobs at once would
# put sixteen workers on twelve cores, which makes every wall clock a measurement of the scheduler. Four
# cores stay free, as the machine is also the thing the user is reading this on.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks results/curves

PIDFILE="results/blocks/block7.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

step() { echo "=== $1 at $(date -Is) ===" | tee -a results/blocks/block7.log; }

# --- R1, the depth curve on ONE harness, all four depths on one deal set -------------------------
# The pool rewrite changed the block decomposition (pairing_index 0 -> 700/701) and a game's seed derives
# from `(run_seed, pairing_index, game_index)`, so depths 1-3 of the previous curve shared a deal set and
# depth 4 did not. A consistency repair, not a correction: the re-run at depth 3 returned +0.0460 where
# the old harness said +0.0590, matching the independent memory and seed sweeps exactly.
step "R1 pooled curve, depth 1"
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=1 timeout 7200 .venv/bin/python _depth_curve.py 1 8000 1 \
  > results/curves/curve_pooled_d1.txt 2>&1
echo "d1 exit=$?" | tee -a results/blocks/block7.log
grep -E "^[[:space:]]*RESULT|pool stable" results/curves/curve_pooled_d1.txt | tail -2 | tee -a results/blocks/block7.log

step "R1 pooled curve, depth 2"
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=2 timeout 7200 .venv/bin/python _depth_curve.py 2 4000 1 \
  > results/curves/curve_pooled_d2.txt 2>&1
echo "d2 exit=$?" | tee -a results/blocks/block7.log
grep -E "^[[:space:]]*RESULT|pool stable" results/curves/curve_pooled_d2.txt | tail -2 | tee -a results/blocks/block7.log

step "R1 pooled curve, depth 3"
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=3 timeout 7200 .venv/bin/python _depth_curve.py 3 2000 1 \
  > results/curves/curve_pooled_d3.txt 2>&1
echo "d3 exit=$?" | tee -a results/blocks/block7.log
grep -E "^[[:space:]]*RESULT|pool stable" results/curves/curve_pooled_d3.txt | tail -2 | tee -a results/blocks/block7.log

step "R1 pooled curve, depth 4"
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=4 timeout 7200 .venv/bin/python _depth_curve.py 4 1000 1 \
  > results/curves/curve_pooled_d4.txt 2>&1
echo "d4 exit=$?" | tee -a results/blocks/block7.log
grep -E "^[[:space:]]*RESULT|pool stable" results/curves/curve_pooled_d4.txt | tail -2 | tee -a results/blocks/block7.log

# --- R2b, the oracle at DEPTH 2, which is where the headline is measured --------------------------
# The depth-4 ceiling is +0.0303. Bounding a depth-2 delta with a depth-4 ceiling compares across
# depths, and depth moves the delta more than belief does. Both arms at depth 2, same weights, belief
# the only difference -- otherwise the number confounds "does perfect knowledge pay" with "does depth
# pay", and in this engine those point in opposite directions.
step "R2b oracle at depth 2"
rm -rf .numba_cache/oracle_d2 durak/__pycache__ durak/strategies/__pycache__
DURAK_SEARCH_DEPTH=2 DURAK_ORACLE_DEPTH=2 ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
  NUMBA_CACHE_DIR=.numba_cache/oracle_d2 \
  timeout 21600 .venv/bin/python _oracle_ceiling.py > results/blocks/r2_oracle_depth2.txt 2>&1
echo "R2b exit=$?" | tee -a results/blocks/block7.log
grep -E "ORACLE vs ORACLE_UNIFORM|depth reach" results/blocks/r2_oracle_depth2.txt \
  | tee -a results/blocks/block7.log

# --- R12, the primary dataset at the scale that was chosen -----------------------------------------
# 280,000 games, not 1.14M: the larger figure was ruled out explicitly. At the measured 339 games/s this
# is about fourteen minutes, so the reason for refusing it was the cost, and the cost is not what it was.
# Levels 2 and 3 are recorded, which is what finally makes the phase analysis possible.
step "R12 primary dataset"
rm -rf datasets/primary durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=primary DATASET_GAMES_PER_PAIRING=6500 DURAK_WORKERS=8 \
  timeout 21600 .venv/bin/python _dataset.py > results/blocks/r12_primary.txt 2>&1
echo "R12 exit=$?" | tee -a results/blocks/block7.log
grep -E "^games|SEARCH_DEPTH|weights=" results/blocks/r12_primary.txt | tee -a results/blocks/block7.log

step "block7 done"
echo "ALL RUNS COMPLETE $(date -Is)" | tee -a results/blocks/block7.log