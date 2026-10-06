"""`CHEATER_DOC`: the reference document's Cheater, implemented literally and separately.

The project's source document specifies a "Cheater" strategy: perfect information, no sampling,
deterministic search. This is that specification, in its own module, and it deliberately **does not share
code** with :mod:`durak.search`.

Why the duplication is the point
--------------------------------
If the Cheater reused the search, then agreement between the two would say only that a function agrees with
itself, and the document's specification would go unmeasured. Implementing it separately means the
specification can be *checked* rather than argued about: a disagreement on any position is then information
about the document, and it surfaces as a concrete example instead of a debate.

The two share exactly one thing, the evaluation weights, and that is deliberate. The weights are a
pre-registered research parameter rather than part of either strategy's definition, so sharing them makes the
comparison about the *search* rather than about two different notions of what a hand is worth.

What "no sampling" means here
-----------------------------
Every hypothesis the Cheater holds is the opponent's actual hand. There is no particle set to draw, so the
branching factor is exactly the opponent's real options -- not an average over beliefs. That is the whole
difference from :func:`durak.search.expectimax` under ``InfoMode.TRUTH``, which reads the same hand but
still routes it through the particle machinery and therefore still pays for ``k`` and still carries the
prior's code path.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from durak.actions import (
    ATTACK,
    BEAT,
    STOP,
    TAKE,
    THROW,
    action_card,
    action_kind,
)
from durak.evaluation import EVAL_WEIGHTS
from durak.leaf import leaf_profile_after, reply_cards_in
from durak.rules import EMPTY, GameState, legal_actions


@njit(cache=True)
def _cheater_action_value(state, action: int, player: int, weights) -> float:
    """The mean leaf score of `action`, over the opponent's *actual* replies.

    One hypothesis, so the average is over replies only. No particles are allocated and no stream is read.
    """
    opponent = 1 - player
    kind = action_kind(action)
    card = action_card(action)
    trump = state.trump
    hand = state.hands[opponent]
    size = int(state.hand_sizes[opponent])

    total = 0.0
    counted = 0

    if kind == STOP or kind == TAKE:
        total += leaf_profile_after(state, player, kind, 0, STOP, 0, state.hands[player], hand, weights)
        return total

    if kind == BEAT:
        total += leaf_profile_after(state, player, BEAT, card, STOP, 0, state.hands[player], hand,
                                    weights)
        return total

    replies = reply_cards_in(hand, size, trump, card)
    for r in range(replies.shape[0]):
        total += leaf_profile_after(state, player, kind, card, BEAT, int(replies[r]),
                                    state.hands[player], hand, weights)
        counted += 1
    total += leaf_profile_after(state, player, kind, card, TAKE, 0, state.hands[player], hand, weights)
    counted += 1

    return total / counted if counted else 0.0


@njit(cache=True)
def policy_cheater_doc(state: GameState, player: int, strategy_id: int, info_mode: int,
                       rng_state: np.uint64) -> np.int32:
    """Perfect information, no sampling, deterministic. As the document specifies it.

    Deterministic in the strong sense: no RNG is read, no particle is drawn, and the tie-break is the first
    candidate in ``legal_actions`` order to reach the maximum -- the same rule the search uses, so that a
    disagreement between the two is about the *search* and not about who breaks ties first.

    ``rng_state`` is accepted and ignored, like every other policy's: the signature is uniform across the
    registry and a policy that returned a stream instead of an action would change it.
    """
    if state.winner != EMPTY:
        return np.int32(-1)

    options = legal_actions(state, player)
    if options.shape[0] == 0:
        return np.int32(-1)

    best_action = int(options[0])
    best_score = -1e308

    for i in range(options.shape[0]):
        action = int(options[i])
        score = _cheater_action_value(state, action, player, EVAL_WEIGHTS)
        if score > best_score:
            best_score = score
            best_action = action

    return np.int32(best_action)
