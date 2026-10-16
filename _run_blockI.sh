#!/bin/bash
# Re-measure everything that carries the frozen instrument, after the re-freeze onto
# (3.0, 2.0, 0.5, 0.3, 0.5) -- the vector that beat `cards_3x` by 0.0122 at ONE scale, at the cost of a
# SMALLER headline (+0.0460 against +0.0612). The selection rule is absolute strength, and following it
# lowered the number; that is the rule working.
#
# NOTHING measured under `cards_3x` may be published beside a headline from the new vector. The depth
# curve, both ceilings, the memory horizon and the seed sweep all describe the old instrument, so they
# are re-measured rather than reused.
#
# Eight workers, sequential, eight of twelve cores so the machine stays usable.
# Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks

PIDFILE="results/blocks/blockI.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
step() { echo "=== $1 at $(date -Is) ===" | tee -a results/blocks/blockI.log; }

# --- datasets: the headline and everything read off it ---------------------------------------------
step "R12 primary dataset, 1.14M, deck 36, re-frozen instrument"
rm -rf datasets/primary durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=primary DATASET_GAMES_PER_PAIRING=26512 DATASET_DECKS=36 DURAK_WORKERS=8 \
  timeout 43200 .venv/bin/python _dataset.py > results/blocks/r12_primary.txt 2>&1
echo "R12 exit=$?" | tee -a results/blocks/blockI.log
grep -E "^games|weights=" results/blocks/r12_primary.txt | tee -a results/blocks/blockI.log

step "R7 deck 52, same instrument (the deck comparison must share it)"
rm -rf datasets/deck52 durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=deck52 DATASET_GAMES_PER_PAIRING=26512 DATASET_DECKS=52 DURAK_WORKERS=8 \
  timeout 43200 .venv/bin/python _dataset.py > results/blocks/r7_deck52.txt 2>&1
echo "R7 exit=$?" | tee -a results/blocks/blockI.log
grep -E "^games|decks" results/blocks/r7_deck52.txt | tee -a results/blocks/blockI.log

step "R13 sensitivity grid"
rm -rf datasets/sensitivity durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=sensitivity DATASET_GAMES_PER_PAIRING=1000 DATASET_DECKS=36,52 DURAK_WORKERS=8 \
  timeout 21600 .venv/bin/python _dataset.py > results/blocks/r13_sensitivity.txt 2>&1
echo "R13 exit=$?" | tee -a results/blocks/blockI.log
grep -E "^games|decks" results/blocks/r13_sensitivity.txt | tee -a results/blocks/blockI.log

# --- the curve and the ceilings, on the same vector ------------------------------------------------
# One process and one private cache per depth: `@njit` freezes module globals at compile time, so two
# depths sharing a process means one of them silently plays the other's depth.
step "R1 depth curve, one harness and one deal set at every depth"
for d in 1 2 3 4; do
  case $d in 1) g=8000;; 2) g=4000;; 3) g=2000;; 4) g=1000;; esac
  rm -rf ".numba_cache/rc_d$d" durak/__pycache__ durak/strategies/__pycache__
  CURVE_WORKERS=8 NUMBA_CACHE_DIR=".numba_cache/rc_d$d" DURAK_SEARCH_DEPTH="$d" \
    timeout 7200 .venv/bin/python _depth_curve.py "$d" "$g" 1 > "results/curves/curve_pooled_d$d.txt" 2>&1
  echo "curve depth $d exit=$?" | tee -a results/blocks/blockI.log
  grep -E "^[[:space:]]*RESULT|pool stable" "results/curves/curve_pooled_d$d.txt" | tail -2 | tee -a results/blocks/blockI.log
done

step "R2 oracle ceilings at depth 2 and depth 4"
for d in 2 4; do
  rm -rf ".numba_cache/rc_oracle_d$d" durak/__pycache__ durak/strategies/__pycache__
  DURAK_SEARCH_DEPTH="$d" DURAK_ORACLE_DEPTH="$d" ORACLE_GAMES=6000 ORACLE_WORKERS=8 \
    NUMBA_CACHE_DIR=".numba_cache/rc_oracle_d$d" \
    timeout 21600 .venv/bin/python _oracle_ceiling.py > "results/blocks/r2_oracle_depth$d.txt" 2>&1
  echo "oracle depth $d exit=$?" | tee -a results/blocks/blockI.log
  grep -E "ORACLE vs ORACLE_UNIFORM|depth reach" "results/blocks/r2_oracle_depth$d.txt" | tee -a results/blocks/blockI.log
done
# Depth 4 is the project's published ceiling file; depth 2 is the one the headline is bounded at.
cp results/blocks/r2_oracle_depth2.txt results/blocks/r2_oracle_depth2.txt.bak
cp results/blocks/r2_oracle_depth4.txt results/blocks/r2_oracle.txt

# --- memory horizon and seeds, on the same vector --------------------------------------------------
step "R5 memory horizon and R9 seed robustness"
bash _run_memory.sh 3 2000
echo "R5+R9 exit=$?" | tee -a results/blocks/blockI.log

step "blockI done"
echo "ALL RUNS COMPLETE $(date -Is)" | tee -a results/blocks/blockI.log