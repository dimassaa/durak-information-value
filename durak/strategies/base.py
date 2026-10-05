"""Strategy identifiers, information modes, and the policy signature.

Every strategy is an integer id and an ``njit`` function, never an object. Two reasons,
both required by the execution mode:

* Polymorphic dispatch over Python objects is not supported inside a Numba ``prange``
  loop, and the tournament runs its games in one.
* A strategy is a pure function of ``(state, player, info_mode, rng_state)``. An object
  would carry hidden state, and hidden state is what makes a seeded run irreproducible.

The ids are written into Parquet rows and into the run manifest, so they are pinned by
``tests/test_legality.py`` and must never be renumbered: a renumbering would silently
re-label every dataset ever produced.

Ids are reserved up front for the strategies the plan schedules, so a runner can be written
against the final table. :func:`durak.strategies.get_policy` raises for a reserved but
unimplemented id rather than substituting a placeholder -- a stub returning a random move
for ``SEARCH_INFORMED`` would produce a plausible, wrong dataset for the experiment's
headline comparison.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import numpy as np


class StrategyId(IntEnum):
    """Stable integer identity of each strategy."""

    #: Attacks with the lowest non-trump, takes when defending. A throwaway driver kept
    #: because the engine needs *something* to generate games with before the real
    #: policies exist.
    AUTO_ATTACK = 0
    #: Beats with the lowest beater, takes when it cannot, attacks with the lowest
    #: non-trump. A throwaway driver, as above.
    AUTO_DEFEND = 1
    #: Uniform over legal actions. The exact symmetry baseline: the only exact property
    #: asserted anywhere in the project is ``RANDOM vs RANDOM = 0.5``.
    RANDOM = 2
    #: Strong, simple, information-free reference. See spec 5.5.
    GREEDY = 3
    #: Reserves trumps and high cards. Implemented after the particle filter (Task 13).
    CONSERVATIVE = 4
    #: Expectimax over a particle filter. One half of the primary comparison.
    SEARCH_UNIFORM = 5
    #: The same function as ``SEARCH_UNIFORM`` with the true hand as the source. The other
    #: half. Any behavioural difference between the two is a difference in information.
    SEARCH_INFORMED = 6
    #: The source document's Cheater: perfect information, no sampling, deterministic
    #: search. A deliberately *different* code path, kept so the document's specification
    #: can be measured rather than argued about.
    CHEATER_DOC = 7


class InfoMode(IntEnum):
    """Where a strategy gets the opponent's hand from.

    This is the project's experimental variable. ``SEARCH_UNIFORM`` and
    ``SEARCH_INFORMED`` are the same function; only this value differs between them, so
    their win-rate difference is attributable to information and to nothing else.
    """

    #: No opponent-hand information at all. ``RANDOM``, ``GREEDY``, and the auto drivers.
    NONE = 0
    #: A particle set drawn from the posterior over hands consistent with the observations.
    PARTICLES = 1
    #: The opponent's real hand, one particle.
    TRUTH = 2


# A policy takes the acting player -- not the attacker. Deriving the role from `player`
# keeps every policy from having to re-derive it, and there is one place (`acting_player`)
# where the derivation lives.
Policy = Callable[[object, int, int, int, np.uint64], np.int32]
