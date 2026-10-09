#!/bin/bash
# R1 re-measured on the POOLED harness, so all four depths share one deal set.
#
# The pool rewrite changed the block decomposition: one `play_block` with `pairing_index=0` became blocks
# with `pairing_index=700/701`, and a game's seed derives from `(run_seed, pairing_index, game_index)`.
# Depths 1-3 of the previous curve therefore shared deal set 0 and depth 4 used a different one, at
# exactly the point where the interval is widest. The pool's own gate compares one worker against eight
# WITHIN the block decomposition, so it cannot see this -- the change is invisible to it by construction.
#
# It is a consistency repair, not a correction of a result: a re-run at depth 3 returned +0.0460 where the
# old harness returned +0.0590, and the re-run matches the independent memory sweep's unbounded arm
# exactly. Both are valid estimates of the same quantity and differ by less than the interval, but a
# headline whose points come from different deals is not one number per depth.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/curves

while pgrep -f "_run_block5.sh" > /dev/null; do sleep 30; done
echo "=== R1 pooled re-measure $(date -Is) ===" | tee -a results/blocks/block6.log

rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=1 timeout 7200 .venv/bin/python _depth_curve.py 1 8000 1 \
  > results/curves/curve_pooled_d1.txt 2>&1
echo "d1 exit=$?" | tee -a results/blocks/block6.log
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=2 timeout 7200 .venv/bin/python _depth_curve.py 2 4000 1 \
  > results/curves/curve_pooled_d2.txt 2>&1
echo "d2 exit=$?" | tee -a results/blocks/block6.log
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=3 timeout 7200 .venv/bin/python _depth_curve.py 3 2000 1 \
  > results/curves/curve_pooled_d3.txt 2>&1
echo "d3 exit=$?" | tee -a results/blocks/block6.log
rm -rf durak/__pycache__ durak/strategies/__pycache__
CURVE_WORKERS=8 DURAK_SEARCH_DEPTH=4 timeout 7200 .venv/bin/python _depth_curve.py 4 1000 1 \
  > results/curves/curve_pooled_d4.txt 2>&1
echo "d4 exit=$? at $(date -Is)" | tee -a results/blocks/block6.log
echo "=== R1 pooled re-measure done $(date -Is) ===" | tee -a results/blocks/block6.log
