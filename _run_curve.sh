#!/bin/bash
# One clean compile per depth -- numba's disk cache does not invalidate when a module global changes, so a
# warm cache makes every depth silently run the first one compiled and the curve comes out flat.
#
# Games PER DEPTH fall with depth, because cost grows by the branching factor (measured mean 4.80) per
# ply: depth 4 costs about 125x depth 1 at equal scale. Equal counts would mean five hours, which is not
# available. Shrunken counts buy the SIGN at every depth and cost precision at the deep ones, and the
# precision actually achieved is printed per depth rather than assumed.
cd /home/dimass/opencode/durak
for spec in "1 8000" "2 4000" "3 2000" "4 1000"; do
  set -- $spec
  d=$1; g=$2
  rm -rf durak/__pycache__ durak/strategies/__pycache__
  DURAK_SEARCH_DEPTH=$d timeout 5400 .venv/bin/python _depth_curve.py $d "$g" 2>&1 \
    | grep -E "^RESULT|^  (informed|DELTA|uniform|wall)"
done

# Nothing this project measures is written under /tmp. It has been wiped twice in one session and took a
# 1.14M-game dataset with it, along with every before/after measurement of the depth-2 work. Output goes
# to results/curves/ inside the repository, which is gitignored because it is large and regenerable from
# a seed, but which survives a reboot.
