#!/bin/bash
# Block B2 redone against the FROZEN instrument.
#
# The depth curve and the oracle were both going to run with `EVAL_WEIGHTS`, the original incumbent. The
# R3 sweep ranks that vector sixth of eleven on absolute strength (panel mean 0.259 against 0.688 for
# the selected `cards_3x`, which beats GREEDY 0.526 against 0.075). A curve and a ceiling measured with a
# rejected instrument describe that instrument, so both are re-measured rather than published.
#
# No weight is set here on purpose: `SELECTED_WEIGHTS` is now the default in both `durak.config` and
# `durak.strategies.search_policy`, and passing it explicitly from a shell script is how two literals end
# up disagreeing. The first curve is KEPT, as a real result about the incumbent, under its own name.
#
#   depth 1  delta +0.0200  CI [+0.0090, +0.0310]   informed vs GREEDY 0.0690
#   depth 2  delta -0.0060  CI [-0.0215, +0.0095]   informed vs GREEDY 0.0739
#   depth 3  delta -0.0290  CI [-0.0509, -0.0071]   informed vs GREEDY 0.1165
#   depth 4  delta -0.0500  CI [-0.0810, -0.0190]   informed vs GREEDY 0.1260
#   ...all measured with the incumbent, at depth 2 the selected arm beats GREEDY 0.5208.
#
# Depth choices are inherited from that curve and re-checked once the new curve lands: the oracle at
# depth 4, where the effect was largest and furthest from zero, and R5/R9 at depth 3, where the interval
# excluded zero at a third of the cost of depth 4.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks .numba_cache results/curves

# Wait for the weight sweep, so two eight-worker experiments never share twelve cores.
while pgrep -f "_run_weights.sh" > /dev/null; do sleep 30; done
echo "=== block B2 (frozen instrument) resumed $(date -Is) ===" | tee -a results/blocks/block4.log

# --- R1, re-measured: the depth curve on the instrument the project actually froze -----------------
rm -rf durak/__pycache__ durak/strategies/__pycache__
DURAK_EVAL_WEIGHTS=3.0,2.0,0.5,0.3,1.5 \
  timeout 28800 bash _run_curve.sh > results/curves/depth_curve_selected_weights_4x.txt 2>&1
echo "R1 curve (selected weights) exit=$? at $(date -Is)" | tee -a results/blocks/block4.log

# --- R2, the oracle ceiling, same weights, both arms at the depth being bounded ------------------
rm -rf .numba_cache/oracle durak/__pycache__ durak/strategies/__pycache__
DURAK_SEARCH_DEPTH=4 DURAK_ORACLE_DEPTH=4 ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
  NUMBA_CACHE_DIR=.numba_cache/oracle DURAK_EVAL_WEIGHTS=3.0,2.0,0.5,0.3,1.5 \
  timeout 21600 .venv/bin/python _oracle_ceiling.py > results/blocks/r2_oracle.txt 2>&1
echo "R2 oracle exit=$? at $(date -Is)" | tee -a results/blocks/block4.log

# --- R5 and R9 -------------------------------------------------------------------------------------
bash _run_memory.sh 3 2000
echo "R5+R9 exit=$? at $(date -Is)" | tee -a results/blocks/block4.log
echo "=== block B2 (frozen instrument) done $(date -Is) ===" | tee -a results/blocks/block4.log