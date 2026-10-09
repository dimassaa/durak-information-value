import numpy as np
from durak import rules
from durak.evaluation import EVAL_WEIGHTS
from durak.search import expectimax
from durak.strategies.search_policy import _truth_particles
VARIANTS = [EVAL_WEIGHTS, (5.0,2.0,0.5,0.3,1.5), (1.0,0.0,0.0,0.0,0.0), (1.0,2.0,2.0,0.3,1.5)]
base = {v: [] for v in VARIANTS}
for seed in range(30):
    st = rules.deal_state(36, np.uint64(seed), np.uint64(seed+7), np.uint64(seed+13),
                          rules.DEFAULT_SETTINGS, False, -1)
    if int(st.winner) != rules.EMPTY: continue
    p = int(st.attacker)
    pa = _truth_particles(st, 1-p)
    hypo = np.repeat(st.deck[:st.deck_len][::-1][None,:], pa.shape[0], axis=0)
    for v in VARIANTS:
        base[v].append(int(expectimax(st, p, pa, hypo, v, 2)))
inc = base[VARIANTS[0]]
for v in VARIANTS:
    diff = sum(1 for a,b in zip(base[v], inc) if a != b)
    print(f"weights {str(v):<28} differs from the incumbent in {diff:>3} of {len(inc)} positions")
