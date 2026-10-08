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
before the real policies exist, ``RANDOM``, ``GREEDY``, the two search arms, and the
document's Cheater, and ``CONSERVATIVE``, which reads the particle filter's aggregate counts as
hand rules rather than running a search.
"""

from durak.strategies.base import InfoMode, Policy, StrategyId
from durak.strategies.policies import (
    policy_auto_attack,
    policy_auto_defend,
    policy_greedy,
    policy_random,
)
from durak.strategies.registry import get_policy, policy_names, registered_ids
from durak.strategies.cheater_doc import policy_cheater_doc
from durak.strategies.conservative import policy_conservative
from durak.strategies.oracle import ORACLE_DEPTH, policy_oracle, policy_oracle_uniform
from durak.strategies.search_policy import INFO_PARTICLES, INFO_TRUTH, policy_search

# The pinned spellings, re-exported so callers write `from durak.strategies import GREEDY`
# rather than reaching through two modules for an enum member.
AUTO_ATTACK = int(StrategyId.AUTO_ATTACK)
AUTO_DEFEND = int(StrategyId.AUTO_DEFEND)
RANDOM = int(StrategyId.RANDOM)
GREEDY = int(StrategyId.GREEDY)
CONSERVATIVE = int(StrategyId.CONSERVATIVE)
SEARCH_UNIFORM = int(StrategyId.SEARCH_UNIFORM)
SEARCH_INFORMED = int(StrategyId.SEARCH_INFORMED)
CHEATER_DOC = int(StrategyId.CHEATER_DOC)
ORACLE = int(StrategyId.ORACLE)
ORACLE_UNIFORM = int(StrategyId.ORACLE_UNIFORM)

#: The two information modes, pinned. `INFO_TRUTH` is the informed arm: the opponent's real hand as a
#: single particle. Everything else is sampled from the prior. These are re-exported from the search policy
#: rather than spelled out there as literals, so "the arm with perfect knowledge" appears in exactly one
#: place in the source.
INFO_TRUTH = INFO_TRUTH
INFO_PARTICLES = INFO_PARTICLES

#: The depth the oracle pair searches to, re-exported for the same reason: the depth is a research
#: parameter and a published oracle number is meaningless without it, so it has to be readable from
#: the one place a caller already imports strategy constants from. `@njit` freezes it at compile time,
#: so a run that reports it must also have set `DURAK_ORACLE_DEPTH` before the first compile.
ORACLE_DEPTH = ORACLE_DEPTH

__all__ = [
    "AUTO_ATTACK",
    "AUTO_DEFEND",
    "GREEDY",
    "RANDOM",
    "CHEATER_DOC",
    "CONSERVATIVE",
    "ORACLE",
    "ORACLE_UNIFORM",
    "ORACLE_DEPTH",
    "SEARCH_INFORMED",
    "SEARCH_UNIFORM",
    "INFO_PARTICLES",
    "INFO_TRUTH",
    "InfoMode",
    "Policy",
    "StrategyId",
    "get_policy",
    "policy_cheater_doc",
    "policy_conservative",
    "policy_oracle",
    "policy_oracle_uniform",
    "policy_search",
    "policy_auto_attack",
    "policy_auto_defend",
    "policy_greedy",
    "policy_random",
    "policy_names",
    "registered_ids",
]
