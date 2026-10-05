"""Strategies: how a player chooses a move.

Five strategies span the spectrum from no knowledge to perfect knowledge, and the two search
strategies are deliberately *the same function* differing only in where they read the
opponent's hand from. That is the experiment: any win-rate difference between
``SEARCH_UNIFORM`` and ``SEARCH_INFORMED`` is a difference in information, because depth,
particle budget, evaluation weights, tie-breaking, and action ordering are all shared
structurally rather than by convention.

Each strategy is an integer id plus an ``njit`` function of
``(state, player, strategy_id, info_mode, rng_state)``. No classes, no dispatch on Python
objects -- the tournament runs its games inside a Numba ``prange`` loop where neither is
supported, and a policy carrying hidden state would make a seeded run irreproducible.

Implemented now: the two throwaway auto drivers, which exist so the engine can generate games
before the real policies exist, plus ``RANDOM`` and ``GREEDY``. ``CONSERVATIVE``,
``SEARCH_UNIFORM``, ``SEARCH_INFORMED``, and ``CHEATER_DOC`` are reserved ids that
``get_policy`` refuses rather than stubs.
"""

from durak.strategies.base import InfoMode, Policy, StrategyId
from durak.strategies.policies import (
    policy_auto_attack,
    policy_auto_defend,
    policy_greedy,
    policy_random,
)
from durak.strategies.registry import get_policy, policy_names, registered_ids

# The pinned spellings, re-exported so callers write `from durak.strategies import GREEDY`
# rather than reaching through two modules for an enum member.
AUTO_ATTACK = int(StrategyId.AUTO_ATTACK)
AUTO_DEFEND = int(StrategyId.AUTO_DEFEND)
RANDOM = int(StrategyId.RANDOM)
GREEDY = int(StrategyId.GREEDY)

__all__ = [
    "AUTO_ATTACK",
    "AUTO_DEFEND",
    "GREEDY",
    "RANDOM",
    "InfoMode",
    "Policy",
    "StrategyId",
    "get_policy",
    "policy_auto_attack",
    "policy_auto_defend",
    "policy_greedy",
    "policy_random",
    "policy_names",
    "registered_ids",
]
