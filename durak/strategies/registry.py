"""Strategy registry: an integer id to a function lookup.

Not a dict of callables chosen by name, and not subclassing. Both are wrong for this
project for the same underlying reason: the tournament dispatches inside a Numba ``prange``
loop, where the callee must be a compile-time constant. A table of ``njit`` functions keyed
by ``IntEnum`` gives Numba a switch it can resolve, and it is the same shape the engine will
use when a strategy is chosen per game from the run configuration.

The table is built by name in one place, so adding a strategy is one line here plus its
function. It is deliberately *not* built by scanning the module namespace: an accidental
import would then silently register a strategy, and a strategy that appears in a dataset
without appearing in the registry is how a result stops being reproducible.
"""

from __future__ import annotations

from durak.strategies.base import InfoMode, Policy, StrategyId
from durak.strategies.policies import (
    policy_auto_attack,
    policy_auto_defend,
    policy_greedy,
    policy_random,
)
from durak.strategies.cheater_doc import policy_cheater_doc
from durak.strategies.conservative import policy_conservative
from durak.strategies.search_policy import policy_search

# Keyed by the pinned integer, not the enum member, so the dispatch inside compiled code is a
# plain integer switch. The enum stays the readable spelling.
_POLICIES = {
    int(StrategyId.AUTO_ATTACK): policy_auto_attack,
    int(StrategyId.AUTO_DEFEND): policy_auto_defend,
    int(StrategyId.RANDOM): policy_random,
    int(StrategyId.GREEDY): policy_greedy,
    # Reads the particle filter's aggregate counts and applies hand rules, with no leaf arithmetic and no
    # reply enumeration. A different point on the strength axis, not a cheaper copy of the search.
    int(StrategyId.CONSERVATIVE): policy_conservative,
    # Both search arms are the *same* function. `InfoMode` is the only input that differs between them,
    # which is what makes their win-rate difference attributable to information and to nothing else.
    int(StrategyId.SEARCH_UNIFORM): policy_search,
    int(StrategyId.SEARCH_INFORMED): policy_search,
    # Deliberately a separate implementation from `policy_search` -- see durak/strategies/cheater_doc.py.
    # Sharing the code would make the document's specification unmeasurable: the two would agree only
    # because they are one function.
    int(StrategyId.CHEATER_DOC): policy_cheater_doc,
}

_NAMES = {
    int(StrategyId.AUTO_ATTACK): "AUTO_ATTACK",
    int(StrategyId.AUTO_DEFEND): "AUTO_DEFEND",
    int(StrategyId.RANDOM): "RANDOM",
    int(StrategyId.GREEDY): "GREEDY",
    int(StrategyId.CONSERVATIVE): "CONSERVATIVE",
    int(StrategyId.SEARCH_UNIFORM): "SEARCH_UNIFORM",
    int(StrategyId.SEARCH_INFORMED): "SEARCH_INFORMED",
    int(StrategyId.CHEATER_DOC): "CHEATER_DOC",
}


def get_policy(strategy_id: int) -> Policy:
    """The policy function for `strategy_id`.

    Raises for an unknown id and for an id reserved by the plan but not yet implemented. The
    second case matters more: a stub returning a random move for ``SEARCH_INFORMED`` would
    produce a plausible, wrong dataset for the experiment's headline comparison, and the
    failure would show up as a suspiciously small effect size rather than as a crash.
    """
    try:
        key = int(strategy_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown strategy id: {strategy_id!r}") from exc

    policy = _POLICIES.get(key)
    if policy is None:
        name = _NAMES.get(key, "unregistered")
        raise ValueError(
            f"unknown strategy {key} ({name}): not implemented. A placeholder is "
            f"deliberately not provided -- it would enter the dataset as a result."
        )
    return policy


def policy_names() -> dict:
    """Strategy id to display name, for manifests and report tables.

    Includes the reserved-but-unimplemented ids, so a report can name a strategy the registry
    cannot dispatch without inventing a second table.
    """
    return dict(_NAMES)


def registered_ids() -> tuple:
    """The ids that can currently be dispatched, in ascending order."""
    return tuple(sorted(_POLICIES))


__all__ = [
    "InfoMode",
    "Policy",
    "StrategyId",
    "get_policy",
    "policy_names",
    "registered_ids",
]
