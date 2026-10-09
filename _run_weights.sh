#!/bin/bash
# R3 -- twelve evaluation-weight vectors, one process and one numba cache directory each.
#
# The separation is not tidiness. `@njit` freezes module globals at compile time and numba's on-disk
# cache does not invalidate when a global changes, so two vectors sharing a process or a cache means
# every vector after the first silently plays the FIRST one's weights -- and the sweep returns the same
# number twelve times under twelve names, which reads as a flat result and is a broken measurement.
#
# The script runs every vector and prints them all. It does NOT choose one: selection is on absolute
# strength against the fixed panel, and a script that picked the largest headline would be fitting the
# answer to the question. The delta is printed at every vector and is not the criterion.
#
# Nothing is written to /tmp. Caches go to .numba_cache/<variant> inside the repository, which is
# gitignored, and transcripts to results/weights/.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/weights .numba_cache

export SWEEP_GAMES="${SWEEP_GAMES:-10000}"
export SWEEP_DELTA_GAMES="${SWEEP_DELTA_GAMES:-4000}"
export SWEEP_PROBE="${SWEEP_PROBE:-30}"
export SWEEP_WORKERS="${SWEEP_WORKERS:-8}"
export SWEEP_SEED="${SWEEP_SEED:-1}"

# label:cards,trumps,high,role,deck -- the order `durak.evaluation.evaluate` consumes them.
VARIANTS=(
  "incumbent:1.0,2.0,0.5,0.3,1.5"
  "cards_3x:3.0,2.0,0.5,0.3,1.5"
  "cards_5x:5.0,2.0,0.5,0.3,1.5"
  "trumps_half:1.0,1.0,0.5,0.3,1.5"
  "trumps_off:1.0,0.0,0.5,0.3,1.5"
  "trumps_3x:1.0,6.0,0.5,0.3,1.5"
  "high_4x:1.0,2.0,2.0,0.3,1.5"
  "role_off:1.0,2.0,0.5,0.0,1.5"
  "deck_3x:1.0,2.0,0.5,0.3,4.5"
  "cards_up_trumps_down:3.0,1.0,0.5,0.3,1.5"
  "cards_down_trumps_up:1.0,4.0,0.5,0.3,1.5"
  "cards_only:1.0,0.0,0.0,0.0,0.0"
)

echo "=== weight sweep: ${#VARIANTS[@]} vectors, $SWEEP_GAMES games/opponent, $SWEEP_WORKERS workers ===" \
  | tee results/weights/sweep.log
for spec in "${VARIANTS[@]}"; do
  label="${spec%%:*}"
  weights="${spec#*:}"
  rm -rf ".numba_cache/$label"
  SWEEP_LABEL="$label" NUMBA_CACHE_DIR=".numba_cache/$label" DURAK_EVAL_WEIGHTS="$weights" \
    timeout 7200 .venv/bin/python _weight_sweep.py > "results/weights/$label.txt" 2>&1
  status=$?
  echo "$label exit=$status $(date -Is)" | tee -a results/weights/sweep.log
  grep -E "^RESULT" "results/weights/$label.txt" | tee -a results/weights/sweep.log
done
echo "=== weight sweep done $(date -Is) ===" | tee -a results/weights/sweep.log