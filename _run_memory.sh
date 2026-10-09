#!/bin/bash
# R5 -- the memory horizon, and R9 -- seed robustness. Both re-run the head-to-head at ONE chosen depth
# with everything else frozen, which is why they wait for the depth curve: a memory result measured at a
# depth the project does not otherwise use answers a question nobody asked.
#
# Each value gets its OWN process and its OWN numba cache directory. `@njit` freezes module globals at
# compile time and numba's cache does not invalidate on a changed global, so sharing either means every
# value after the first silently plays the first one's parameter -- and the sweep comes out perfectly
# flat, which reads as "memory does not matter".
#
# DEPTH is an argument rather than a default, so the depth cannot drift away from the curve without
# being visible in the transcript.
#
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
DEPTH="${1:?usage: _run_memory.sh DEPTH [GAMES_PER_ARM]}"
GAMES="${2:-4000}"
mkdir -p results/memory results/seeds .numba_cache

# R5: M = how many discarded plies the belief remembers. `0` remembers nothing -- the opponent's hand is
# inferred from the cards currently on the table and in the draw pile. `1000000` is the project's
# unbounded default and reproduces the earlier behaviour exactly, which is its own regression gate.
echo "=== R5 memory horizon at depth $DEPTH, $GAMES games per arm ===" | tee results/memory/sweep.log
for m in 0 2 4 8 16 1000000; do
  label="M=$m"
  rm -rf ".numba_cache/mem_$m"
  DURAK_MEMORY_PLY="$m" NUMBA_CACHE_DIR=".numba_cache/mem_$m" DURAK_SEARCH_DEPTH="$DEPTH" \
    timeout 7200 .venv/bin/python _depth_curve.py "$DEPTH" "$GAMES" 1 \
    > "results/memory/M${m}.txt" 2>&1
  echo "memory $label exit=$? $(date -Is)" | tee -a results/memory/sweep.log
  grep -E "^RESULT|^  DELTA|^  informed" "results/memory/M${m}.txt" | tee -a results/memory/sweep.log
done

# R9: the same configuration under several seeds. Cheap, and it is the check most often skipped: a
# conclusion that depends on one seed is a conclusion about that seed.
echo "=== R9 seed robustness at depth $DEPTH, unbounded memory ===" | tee results/seeds/sweep.log
for seed in 1 2 3 4 5; do
  rm -rf ".numba_cache/seed_$seed"
  NUMBA_CACHE_DIR=".numba_cache/seed_$seed" DURAK_SEARCH_DEPTH="$DEPTH" \
    timeout 7200 .venv/bin/python _depth_curve.py "$DEPTH" "$GAMES" "$seed" \
    > "results/seeds/seed${seed}.txt" 2>&1
  echo "seed $seed exit=$? $(date -Is)" | tee -a results/seeds/sweep.log
  grep -E "^RESULT|^  DELTA" "results/seeds/seed${seed}.txt" | tee -a results/seeds/sweep.log
done
echo "=== R5 and R9 done $(date -Is) ===" | tee -a results/memory/sweep.log