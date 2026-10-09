"""Fingerprint every level-1 column EXCEPT the ones an additive change is allowed to touch.

Used to prove that adding a column did not change what any game did. The comparison is against another
checkout of an earlier commit, so a change to the engine that silently altered outcomes would show up
as a differing fingerprint instead of passing as harmless telemetry.

Two things it deliberately does NOT hash, each of which has already cost a false conclusion here:

  * **Version stamps.** `schema_version` is bumped ON PURPOSE by an additive schema change. Hashing it
    reported a correct additive change as a behaviour change on the first run of this gate.
  * **The new columns themselves**, named on the command line, since their whole point is to differ.

Everything else must match bit for bit.

    .venv/bin/python _additivity_gate.py                     # fingerprint the working tree
    .venv/bin/python _additivity_gate.py distinct_hypotheses_sum search_decisions

A single fingerprint is only evidence if it is reproducible, so check it twice before believing a
difference: a run that is not deterministic is a worse finding than the one being investigated.
"""

import hashlib
import sys

import numpy as np

from durak import rules
from durak.engine import play_block
from durak.strategies import GREEDY, RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM, InfoMode

#: The search arms, both information modes, and two non-search players. Only a mixture exercises the
#: paths a new column could disturb: the search writes into the state it is handed, and a column
#: added to the dtype must not change what a non-search game records.
MATCHUPS = [
    (SEARCH_UNIFORM, GREEDY, InfoMode.PARTICLES, InfoMode.NONE, 11),
    (SEARCH_INFORMED, RANDOM, InfoMode.TRUTH, InfoMode.NONE, 12),
    (GREEDY, RANDOM, InfoMode.NONE, InfoMode.NONE, 13),
]
GAMES_PER_MATCHUP = 12
VERSION_STAMPS = frozenset({"schema_version", "engine_version", "rules_version"})


def fingerprint(excluded) -> str:
    """SHA-256 over the bytes of every level-1 column that is neither excluded nor a version stamp."""
    digest = hashlib.sha256()
    for p0, p1, m0, m1, seed in MATCHUPS:
        outcomes, _, _ = play_block(
            36, p0, p1, m0, m1, seed, 0, 0, GAMES_PER_MATCHUP, rules.DEFAULT_SETTINGS,
            record_rounds=False, record_moves=False,
        )
        for name in outcomes.dtype.names:
            if name in excluded or name in VERSION_STAMPS:
                continue
            digest.update(np.ascontiguousarray(outcomes[name]).tobytes())
    return digest.hexdigest()[:16]


def main() -> None:
    print(fingerprint(frozenset(sys.argv[1:])))


if __name__ == "__main__":
    main()