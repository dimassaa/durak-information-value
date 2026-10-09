#!/bin/bash
# Block B2, step 1: the weight sweep, after block B1 has finished.
#
# A separate chain script rather than an append to _run_block1.sh, because bash reads a script lazily by
# byte offset: editing a running script makes it resume at a shifted offset and execute garbage. The
# wait is what keeps the two sweeps apart -- R1 clears the SHARED durak/__pycache__ before every depth,
# and the sweep deliberately uses PRIVATE cache directories, so nothing here can be compiled against a
# cache another experiment is deleting.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks

while pgrep -f "_run_block1.sh" > /dev/null; do sleep 30; done
echo "=== block B2 (weights) resumed $(date -Is), block B1 finished ===" \
  | tee -a results/blocks/block2.log

SWEEP_GAMES="${SWEEP_GAMES:-10000}" \
SWEEP_DELTA_GAMES="${SWEEP_DELTA_GAMES:-4000}" \
SWEEP_WORKERS="${SWEEP_WORKERS:-8}" \
  bash _run_weights.sh
echo "R3 weight sweep exit=$? at $(date -Is)" | tee -a results/blocks/block2.log