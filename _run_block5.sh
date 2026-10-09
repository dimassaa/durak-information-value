#!/bin/bash
# R2, the oracle ceiling, at full scale -- re-queued after the failure recorded in IMPROVEMENTS 84.
#
# The failure was NOT a measurement result: a stale call site in the depth-reach probe and a weight vector
# that had silently diverged from the arms'. Both are fixed, and the probe now passes both of its guards
# -- 10/10 positions matching the shared search at its own depth, and games that differ from
# SEARCH_INFORMED's under the same seeds. A ceiling that failed either guard would be bounding nothing.
#
# Both arms sit at depth 4 and, now, at the arms' own weight vector: a ceiling measured against a
# different evaluator bounds a different experiment.
#
# Waits for the memory and seed sweeps, because both take eight workers and the machine has twelve cores.
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks .numba_cache

while pgrep -f "_run_memory.sh" > /dev/null; do sleep 30; done
echo "=== R2 (oracle) resumed $(date -Is), memory and seed sweeps finished ===" \
  | tee -a results/blocks/block5.log

rm -rf .numba_cache/oracle durak/__pycache__ durak/strategies/__pycache__
DURAK_SEARCH_DEPTH=4 DURAK_ORACLE_DEPTH=4 ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
  NUMBA_CACHE_DIR=.numba_cache/oracle \
  timeout 21600 .venv/bin/python _oracle_ceiling.py > results/blocks/r2_oracle.txt 2>&1
echo "R2 oracle exit=$? at $(date -Is)" | tee -a results/blocks/block5.log
echo "=== R2 done $(date -Is) ===" | tee -a results/blocks/block5.log