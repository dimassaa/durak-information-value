"""Depth as a parameter: the gates that make it safe to change.

The refactor replaced one written-out leaf and two fixed-ply enumerators with a loop. Existing tests
still passed, which is necessary and not sufficient -- they only ever covered depths 1 and 2, the depths
that happened to exist. These tests cover the ones nothing covered before:

  * the four-ply leaf agrees with the ENGINE after four `apply_action` calls, so the loop's refill and
    role handling are right at a ply count no written-out body ever reached;
  * `plies_after` equals `rules.legal_actions` after three plies, which is the enumeration the depth-2
    path never used -- and after a refill, where the table is empty and the ATTACK branch is the only
    legal shape, the case a blind generalisation would silently omit;
  * the search returns a legal action and is deterministic at every depth from 1 to 4.

The bar is EXACT equality against the engine. Approximate equality would pass for a leaf that is close
but wrong, and the whole point of the leaf is that it is a faithful reproduction.
"""
import numpy as np
import pytest

import durak.cards
from durak import rules
from durak.evaluation import HIGH_RANK
from durak.actions import ATTACK, BEAT, STOP, TAKE, THROW, pack_action  # noqa: F401
from durak.evaluation import EVAL_WEIGHTS
from durak.leaf import leaf_profile_after_n, plies_after
from durak.search import expectimax

W = EVAL_WEIGHTS
CAP = 70


def _card(rank, suit):
    return rank * 4 + suit


#: The seed the current position was dealt from, so `_twin` can redeal it. Module-level because the
#: fixture functions below build states in a loop and a `_twin` needs to know which one.
_SEED_OF = [0]


def _fixture(seed=11):
    """A dealt 36-card position with both players holding trumps, so all four ply shapes exist."""
    _SEED_OF[0] = seed
    return _deal(seed)


def _engine_value(state, player):
    """`evaluate`'s value read straight off a live state, in the leaf's own order."""
    def stats(seat):
        size = int(state.hand_sizes[seat])
        trumps = sum(1 for i in range(size) if int(state.hands[seat, i]) % 4 == state.trump)
        high = sum(1 for i in range(size)
                   if durak.cards.rank_of(int(state.hands[seat, i])) >= HIGH_RANK)
        return size, trumps, high

    cards_p, trumps_p, highs_p = stats(player)
    cards_o, trumps_o, highs_o = stats(1 - player)
    v = W[0] * (cards_o - cards_p) + W[1] * (trumps_p - trumps_o) + W[2] * (highs_p - highs_o)
    v += W[3] * (int(state.attacker) == player) + W[4] * (int(state.deck_len) == 0)
    return v


def _deal(seed):
    return rules.deal_state(36, np.uint64(seed), np.uint64(seed + 101), np.uint64(seed + 202),
                            rules.DEFAULT_SETTINGS, False, -1)


def _twin(state):
    """A second, independent deal from the SAME seed.

    `rules.apply_action` mutates in place, so the engine needs its own copy. Copying the state field by
    field is the usual approach and it is a trap here: a field added later is silently forgotten and the
    twin diverges in exactly the way the test cannot see. Dealing the same seed twice cannot diverge.
    """
    return _deal(_SEED_OF[0])


def _hands(state):
    return state.hands[0].copy(), state.hands[1].copy()


def _first_matching(state, player, kinds_wanted):
    """The acting player's first legal move of the wanted kinds, or None."""
    for a in rules.legal_actions(state, player):
        k = rules.action_kind(int(a))
        if k in kinds_wanted:
            return k, rules.action_card(int(a))
    return None


# The differential tests for a ply count above three -- the four-ply leaf against the engine, and
# `plies_after` at three plies including the `table_size == 0` branch -- live in
# `tests/test_depth_plies.py`. They were written after this module and this module's earlier draft of them
# did not work; keeping them apart means a failure there is unambiguously about the ply-count coverage and
# not about the legality and determinism checks here.
