#!/bin/bash
# Block B2 remainder: R2 (the oracle ceiling) then R5 (memory horizon) and R9 (seed robustness).
#
# R2 is here rather than re-run from block 1 because block 1's oracle step REFUSED to start: the harness
# demands a private NUMBA_CACHE_DIR and block 1 did not set one. That refusal is the guard working -- it
# would have run on a shared numba cache otherwise, which is how one experiment silently plays another's
# compiled parameters. Re-running block 1 would have repeated the 114,000-game pilot as well, which had
# already succeeded.
#
# Depth is a decision, not an inherited default, and it comes from the measured curve:
#
#   depth 1  delta +0.0200  CI [+0.0090, +0.0310]   informed vs GREEDY 0.0690
#   depth 2  delta -0.0060  CI [-0.0215, +0.0095]   informed vs GREEDY 0.0739
#   depth 3  delta -0.0290  CI [-0.0509, -0.0071]   informed vs GREEDY 0.1165
#   depth 4  delta -0.0500  CI [-0.0810, -0.0190]   informed vs GREEDY 0.1260
#
# The oracle runs at depth 4, because that is where the effect is largest and furthest from zero, so the
# ceiling is measured where there is something to bound. R5 and R9 run at depth 3: depth 4 would cost
# about five hours for both, and depth 2's interval straddles zero, so a memory sweep there would be
# measuring something there is no signal to move.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks .numba_cache

# Wait for the weight sweep: both use eight workers, and running them together would put sixteen workers
# on twelve cores, which makes every wall clock a measure of the scheduler.
while pgrep -f "_run_weights.sh" > /dev/null; do sleep 30; done
echo "=== R2/R5/R9 resumed $(date -Is), weight sweep finished ===" | tee -a results/blocks/block3.log

# --- R2, oracle ceiling, depth 4, private cache --------------------------------------------
rm -rf .numba_cache/oracle durak/__pycache__ durak/strategies/__pycache__
DURAK_SEARCH_DEPTH=4 DURAK_ORACLE_DEPTH=4 ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
  NUMBA_CACHE_DIR=.numba_cache/oracle \
  timeout 21600 .venv/bin/python _oracle_ceiling.py > results/blocks/r2_oracle.txt 2>&1
echo "R2 oracle exit=$? at $(date -Is)" | tee -a results/blocks/block3.log

# --- R5 and R9, depth 3 -------------------------------------------------------------------
bash _run_memory.sh 3 2000
echo "R5+R9 exit=$? at $(date -Is)" | tee -a results/blocks/block3.log
echo "=== block B2 remainder done $(date -Is) ===" | tee -a results/blocks/block3.log