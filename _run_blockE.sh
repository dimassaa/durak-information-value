#!/bin/bash
# Block E: everything left in the plan, at the scales chosen.
#
# R12 goes to 1.14M games. The earlier 280k run took twelve minutes at the measured 339 games/s, which
# retired the reason for refusing the larger figure: it was cost, and the cost turned out not to be
# there. 26,512 games per pairing over 43 cells is 1,140,016 games, and it gives a per-cell Wilson
# half-width near 0.006.
#
# Eight workers, sequential steps. Two eight-worker jobs on twelve cores would put sixteen workers on
# twelve cores and make every wall clock a measurement of the scheduler. Nothing is written to /tmp.
set -u
cd /home/dimass/opencode/durak
mkdir -p results/blocks

PIDFILE="results/blocks/blockE.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
step() { echo "=== $1 at $(date -Is) ===" | tee -a results/blocks/blockE.log; }

# --- R12, the primary dataset at 1.14M, deck 36 ---------------------------------------------------
# Levels 2 and 3 are recorded, which is what finally makes R8's phase analysis possible at all.
step "R12 primary dataset, 1.14M games, deck 36"
rm -rf datasets/primary durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=primary DATASET_GAMES_PER_PAIRING=26512 DATASET_DECKS=36 DURAK_WORKERS=8 \
  timeout 43200 .venv/bin/python _dataset.py > results/blocks/r12_primary.txt 2>&1
echo "R12 exit=$?" | tee -a results/blocks/blockE.log
grep -E "^games|weights=|SEARCH_DEPTH" results/blocks/r12_primary.txt | tee -a results/blocks/blockE.log

# --- R7, the same grid on a 52-card deck, never played until now ------------------------------------
step "R7 deck 52"
rm -rf datasets/deck52 durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=deck52 DATASET_GAMES_PER_PAIRING=26512 DATASET_DECKS=52 DURAK_WORKERS=8 \
  timeout 43200 .venv/bin/python _dataset.py > results/blocks/r7_deck52.txt 2>&1
echo "R7 exit=$?" | tee -a results/blocks/blockE.log
grep -E "^games|decks|weights=" results/blocks/r7_deck52.txt | tee -a results/blocks/blockE.log

# --- R10, self-play: is a depth stronger, or stronger against THESE opponents? -----------------------
# One process and one private cache per depth: `@njit` freezes module globals at compile time, so a
# loop that set the depth after compiling would measure the first depth while printing the others.
: > results/blocks/r10_selfplay.txt
for d in 1 2 3 4; do
  rm -rf ".numba_cache/selfplay_$d"
  DURAK_SEARCH_DEPTH="$d" NUMBA_CACHE_DIR=".numba_cache/selfplay_$d" SELFPLAY_WORKERS=8 \
    timeout 7200 .venv/bin/python _selfplay.py "$d" >> results/blocks/r10_selfplay.txt 2>&1
  echo "selfplay depth $d exit=$?" | tee -a results/blocks/blockE.log
done
echo "R10 exit=0" | tee -a results/blocks/blockE.log
grep -E "^SELFPLAY|^  arm" results/blocks/r10_selfplay.txt | tee -a results/blocks/blockE.log

# --- R13, the sensitivity grid across the deck dimension --------------------------------------------
step "R13 sensitivity grid"
rm -rf datasets/sensitivity durak/__pycache__ durak/strategies/__pycache__
DATASET_NAME=sensitivity DATASET_GAMES_PER_PAIRING=2000 DATASET_DECKS=36,52 DURAK_WORKERS=8 \
  timeout 21600 .venv/bin/python _dataset.py > results/blocks/r13_sensitivity.txt 2>&1
echo "R13 exit=$?" | tee -a results/blocks/blockE.log
grep -E "^games|decks" results/blocks/r13_sensitivity.txt | tee -a results/blocks/blockE.log

step "blockE done"
echo "ALL RUNS COMPLETE $(date -Is)" | tee -a results/blocks/blockE.log