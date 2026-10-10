#!/bin/bash
# R4 -- the weight SURFACE, once the ray sweep found a winner worth refining.
#
# R3 measured twelve rays from the incumbent and `cards_3x` led at 0.688, with `cards_5x` at 0.663 behind
# it. That ordering already says the optimum along the card-count ray is near 3 and not above it, so this
# refines there rather than spreading over the five-term space: a full grid over five terms is
# 3^5 = 243 points, which is not a measurement, it is a month. The three terms that moved anything are
# card count, trumps and deck-exhausted; the grid covers those, and four random interior points test
# whether the optimum is on a ridge or off it.
#
# The harness is the sweep's, unchanged -- one process and one private numba cache per vector, the
# no-op-vector gate, and the two-arm differ check. Selection remains on panel strength, never the delta.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/surface .numba_cache

export SWEEP_GAMES="${SWEEP_GAMES:-6000}"
export SWEEP_DELTA_GAMES="${SWEEP_DELTA_GAMES:-3000}"
export SWEEP_PROBE="${SWEEP_PROBE:-30}"
export SWEEP_WORKERS="${SWEEP_WORKERS:-8}"
export SWEEP_SEED="${SWEEP_SEED:-1}"

# label:cards,trumps,high,role,deck
VARIANTS=(
  "ray_cards_2.0:2.0,2.0,0.5,0.3,1.5"
  "ray_cards_2.5:2.5,2.0,0.5,0.3,1.5"
  "ray_cards_3.5:3.5,2.0,0.5,0.3,1.5"
  "ray_cards_4.0:4.0,2.0,0.5,0.3,1.5"
  "grid_trumps_1.0:3.0,1.0,0.5,0.3,1.5"
  "grid_trumps_3.0:3.0,3.0,0.5,0.3,1.5"
  "grid_deck_0.5:3.0,2.0,0.5,0.3,0.5"
  "random_a:2.2,2.7,0.9,0.2,2.3"
  "random_b:3.4,1.4,0.7,0.4,0.9"
)

echo "=== weight surface: ${#VARIANTS[@]} points, $SWEEP_GAMES games/opponent ===" | tee results/surface/surface.log
for spec in "${VARIANTS[@]}"; do
  label="${spec%%:*}"
  weights="${spec#*:}"
  rm -rf ".numba_cache/surf_$label"
  SWEEP_LABEL="$label" NUMBA_CACHE_DIR=".numba_cache/surf_$label" DURAK_EVAL_WEIGHTS="$weights" \
    timeout 7200 .venv/bin/python _weight_sweep.py > "results/surface/$label.txt" 2>&1
  echo "$label exit=$? $(date -Is)" | tee -a results/surface/surface.log
  grep -E "^[[:space:]]*RESULT" "results/surface/$label.txt" | tee -a results/surface/surface.log
done
echo "=== weight surface done $(date -Is) ===" | tee -a results/surface/surface.log