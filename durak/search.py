"""Expectimax over particle hypotheses: the informed strategy's decision rule.

Depth 2, and two modelling choices that are load-bearing
------------------------------------------------------
**The opponent's replies are averaged, not maximised.** Every legal reply is enumerated and the leaf values
averaged. That makes the search *optimistic* about moves with bad worst cases -- it asks "what happens on
average" rather than "what happens if they play well". Maximisation would model an opponent who always
finds the best reply, which is stronger than any policy in this study and would make the informed arm look
worse than it is.

This is a modelling decision, not a bug, and it is stated here because **both arms must average**. The
uninformed arm runs this same average over its own particle set, so the only difference between the arms is
where the hypothesis came from. Changing one arm's reply model without the other's would measure the reply
model instead of the information.

**The search's own next move is maximised.** Ply three is the acting player again, so it is their own node:
the search takes the best of their legal moves, with the same strict `>` tie-break as the outer scan.
Averaging there instead would model the player choosing at random at depth 2, which no policy in this study
does -- and the outer loop maximises over actions, so averaging the same player's own choices here would
also mean maximising a mean over candidates that are not equally weighted by the outer argmax.

The shape of a depth-2 evaluation
---------------------------------
``player`` moves, the opponent answers, and then -- only if the round has not ended and no hand has emptied
-- ``player`` moves again. The third ply is reachable in both role directions, and reaching it in the
defending direction is why the opponent's *throw-ins* are enumerated too (:func:`durak.leaf.second_plies`).
Depth 1 hard-coded "the attacker stops" in that branch, which made the defender's third ply -- beat, or
take -- unreachable, and the take that ends the game is precisely a third-ply move.

A line with no third ply is scored by :func:`durak.leaf.leaf_profile_after`, the depth-1 leaf, and gets
exactly the depth-1 number. So the outer tie-break is unchanged where depth 1 already applied: a strict `>`
scan in ``legal_actions`` order keeps the first candidate reaching the maximum.

Particles enter by substitution, not by weighting
-------------------------------------------------
The opponent's replies are enumerated against one particle's cards at a time. That is what makes the
hypothesis set matter: a card the search believes is in their hand is a card it believes can be played, so
the branching factor itself changes with the belief. That now holds for throw-ins as well as for beats.

The alternative -- enumerate only the moves visible to the observer and reweight outcomes -- cannot work
here. A card not in the observer's hand is a card the observer cannot name, so the set of replies is only
enumerable *given* a hypothesis.

No states, no working memory
----------------------------
An earlier version of this search needed three scratch ``GameState`` buffers, because an njit entry point
called from Python cannot construct one. :mod:`durak.leaf` removes the need instead of working around it:
the leaf profile is arithmetic over two card arrays, and the ply enumerations are arithmetic over the same
arrays, so no derived state is ever materialised. The caller's state is read-only throughout, which is what
makes ``test_the_search_does_not_modify_the_state`` mean anything.

Determinism
-----------
Ties go to the earlier action in ``legal_actions`` order: the scan uses a strict ``>``, so the first
candidate reaching the maximum is kept. Fixed and documented rather than randomised, because a tie-break by
RNG would still be legal, still be "informed", and would quietly destroy the dataset's reproducibility.
"""

import numpy as np
from numba import njit

from durak.actions import STOP, action_card, action_kind
from durak.leaf import (
    leaf_profile_after,
    leaf_profile_after_ply3,
    second_plies,
    third_plies,
)
from durak.rules import EMPTY, HAND_CAPACITY, legal_actions

#: Replies enumerated per particle before the search refuses rather than truncating.
#
#: One action's answers are the opponent's legal replies to it: every beater plus taking, or every throw-in
#: plus stopping. The count is bounded by the opponent's hand plus one. The cap guards against a pathological
#: position, and exceeding it *raises* rather than truncating: a truncated enumeration makes the opponent
#: look weaker than the rules allow, which biases the search toward attacking and looks like strength.
RESPONSE_CAP = 64


@njit(cache=True)
def action_value(state, action: int, player: int, particles: np.ndarray, weights) -> float:
    """The expected leaf score of playing `action`, averaged over replies and particles.

    The reply list is recomputed inside each particle's hypothesis rather than once against the real hand.
    Enumerating once would hand the informed arm the opponent's actual cards for free -- the single worst
    bug available in this module, and one that would drive the measured price of information to roughly zero
    and look like a finding.

    Depth 2: each reply is followed by the acting player's own next moves, of which the best is taken. A
    reply that ended the round has no third ply and is scored by the depth-1 leaf.
    """
    kind = action_kind(action)
    card = action_card(action)

    k = particles.shape[0]
    total = 0.0
    counted = 0

    # One pair of buffers for the whole search rather than one per call: the enumerations write into them
    # and return a count, and an allocation inside the innermost loop would dominate the cost. Sized for
    # the longest legal action list, which is a full hand plus a cardless kind.
    reply_kinds = np.empty(RESPONSE_CAP + 2, dtype=np.int64)
    reply_card = np.empty(RESPONSE_CAP + 2, dtype=np.int16)
    third_kinds = np.empty(HAND_CAPACITY + 2, dtype=np.int64)
    third_cards = np.empty(HAND_CAPACITY + 2, dtype=np.int16)

    for i in range(k):
        replies = second_plies(state, player, kind, card, state.hands[player], particles[i],
                               reply_kinds, reply_card)

        if replies > RESPONSE_CAP:
            raise ValueError(
                f"the opponent had more than {RESPONSE_CAP} legal replies; raise RESPONSE_CAP. "
                f"Truncating would make the opponent look weaker than the rules allow."
            )

        if replies == 0:
            # `player`'s own action ended the round, so there is no second ply -- the depth-1 leaf is
            # already the right answer, and gives exactly the number depth 1 gave.
            total += leaf_profile_after(state, player, kind, card, STOP, 0, state.hands[player],
                                        particles[i], weights)
            counted += 1
            continue

        for r in range(replies):
            kind_b = int(reply_kinds[r])
            card_b = int(reply_card[r])

            thirds = third_plies(state, player, kind, card, kind_b, card_b, state.hands[player],
                                 particles[i], third_kinds, third_cards)
            if thirds == 0:
                # The reply ended the round or emptied a hand. Depth 1 applies.
                total += leaf_profile_after(state, player, kind, card, kind_b, card_b,
                                            state.hands[player], particles[i], weights)
            else:
                best = -1e308
                for t in range(thirds):
                    value = leaf_profile_after_ply3(state, player, kind, card, kind_b, card_b,
                                                    int(third_kinds[t]), int(third_cards[t]),
                                                    state.hands[player], particles[i], weights)
                    # Strict `>`, the same tie-break as the outer scan: the first third ply reaching the
                    # maximum wins, in enumeration order.
                    if value > best:
                        best = value
                total += best
            counted += 1

    if counted == 0:
        return 0.0
    return total / counted


@njit(cache=True)
def expectimax(state, player: int, particles: np.ndarray, weights) -> np.int32:
    """The action `player` should take as a packed int32, or ``-1`` if none is legal.

    Depth 2: every legal action is played, the opponent's every legal reply is enumerated under every
    particle hypothesis, the acting player's every next move is enumerated after each reply, and each
    resulting position is scored. The opponent's replies are averaged; the acting player's own next move is
    maximised. The best-scoring action wins.

    Reads `state` and `particles` and writes nothing at all: no working buffer, no derived state. A search
    that applied its move to the caller's state would make every recorded game depend on whether the search
    ran, and the cross-mode equivalence probe would stop comparing like with like.
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
        score = action_value(state, action, player, particles, weights)
        # Strict `>`, so the first candidate reaching the maximum wins: a documented tie-break, not an
        # accident of iteration order.
        if score > best_score:
            best_score = score
            best_action = action

    return np.int32(best_action)