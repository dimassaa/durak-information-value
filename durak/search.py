"""Expectimax over particle hypotheses: the informed strategy's decision rule.

Depth 1, and two modelling choices that are load-bearing
--------------------------------------------------------
**The opponent's reply is averaged, not maximised.** Every legal reply is enumerated and the leaf values
averaged. That makes the search *optimistic* about moves with bad worst cases -- it asks "what happens on
average" rather than "what happens if they play well". Maximisation would model an opponent who always
finds the best reply, which is stronger than any policy in this study and would make the informed arm look
worse than it is.

This is a modelling decision, not a bug, and it is stated here because **both arms must average**. The
uninformed arm runs this same average over its own particle set, so the only difference between the arms is
where the hypothesis came from. Changing one arm's reply model without the other's would measure the reply
model instead of the information.

**The leaf is scored without a recursive continuation.** One ply deep, then the leaf profile. A deeper
search could only find moves this one misses, so the informed arm's strength is bounded from *below* --
which makes the measured price of information a lower bound too.

Particles enter by substitution, not by weighting
-------------------------------------------------
The opponent's replies are enumerated against one particle's cards at a time. That is what makes the
hypothesis set matter: a card the search believes is in their hand is a card it believes can be played, so
the branching factor itself changes with the belief.

The alternative -- enumerate only the moves visible to the observer and reweight outcomes -- cannot work
here. A card not in the observer's hand is a card the observer cannot name, so the set of replies is only
enumerable *given* a hypothesis.

No states, no working memory
----------------------------
An earlier version of this search needed three scratch ``GameState`` buffers, because an njit entry point
called from Python cannot construct one. :mod:`durak.leaf` removes the need instead of working around it:
the leaf profile is arithmetic over two card arrays, so no derived state is ever materialised. The
caller's state is read-only throughout, which is what makes
``test_the_search_does_not_modify_the_state`` mean anything.

Determinism
-----------
Ties go to the earlier action in ``legal_actions`` order: the scan uses a strict ``>``, so the first
candidate reaching the maximum is kept. Fixed and documented rather than randomised, because a tie-break by
RNG would still be legal, still be "informed", and would quietly destroy the dataset's reproducibility.
"""

import numpy as np
from numba import njit

from durak.actions import ATTACK, BEAT, STOP, TAKE, THROW, action_card, action_kind
from durak.leaf import leaf_profile_after, reply_cards_in
from durak.rules import EMPTY, legal_actions

#: Replies enumerated per particle before the search refuses rather than truncating.
#
# The defender's replies are "beat with any card that beats this" plus "take", so the count is bounded by
# hand size -- eight for a six-card hand. The cap guards against a pathological post-take defender, and
# exceeding it *raises* rather than truncating: a truncated enumeration makes the opponent look weaker than
# the rules allow, which biases the search toward attacking and looks like strength.
RESPONSE_CAP = 64


@njit(cache=True)
def action_value(state, action: int, player: int, particles: np.ndarray, weights) -> float:
    """The expected leaf score of playing `action`, averaged over replies and particles.

    The reply list is recomputed inside each particle's hypothesis rather than once against the real hand.
    Enumerating once would hand the informed arm the opponent's actual cards for free -- the single worst
    bug available in this module, and one that would drive the measured price of information to roughly zero
    and look like a finding.
    """
    opponent = 1 - player
    kind = action_kind(action)
    card = action_card(action)
    trump = state.trump

    k, width = particles.shape
    total = 0.0
    counted = 0

    for i in range(k):
        if kind == STOP or kind == TAKE:
            # Both end the round; there is no reply.
            total += leaf_profile_after(state, player, kind, 0, STOP, 0, state.hands[player],
                                        particles[i], weights)
            counted += 1
            continue

        if kind == BEAT:
            # The actor is defending. The attacker then throws in or stops; which throw-ins are legal is a
            # rule of the table rather than of either hand, so the branching does not depend on the
            # hypothesis here. A difference between the arms in this position therefore has to come from
            # the leaf score, never from the branching factor -- which is the correct division of labour,
            # since it is the defender who does not know what the attacker still holds.
            total += leaf_profile_after(state, player, BEAT, card, STOP, 0, state.hands[player],
                                        particles[i], weights)
            counted += 1
            continue

        # ATTACK or THROW: the opponent is the defender and answers this card.
        replies = reply_cards_in(particles[i], width, trump, card)
        reply_count = replies.shape[0] + 1  # the beats, plus taking

        if reply_count > RESPONSE_CAP:
            raise ValueError(
                f"the opponent had more than {RESPONSE_CAP} legal replies; raise RESPONSE_CAP. "
                f"Truncating would make the opponent look weaker than the rules allow."
            )

        for r in range(reply_count):
            if r < replies.shape[0]:
                total += leaf_profile_after(state, player, kind, card, BEAT, int(replies[r]),
                                            state.hands[player], particles[i], weights)
            else:
                total += leaf_profile_after(state, player, kind, card, TAKE, 0, state.hands[player],
                                            particles[i], weights)
            counted += 1

    if counted == 0:
        return 0.0
    return total / counted


@njit(cache=True)
def expectimax(state, player: int, particles: np.ndarray, weights) -> np.int32:
    """The action `player` should take as a packed int32, or ``-1`` if none is legal.

    Depth 1: every legal action is played, the opponent's every legal reply is enumerated under every
    particle hypothesis, the resulting positions are scored, and the total is averaged over replies and over
    particles. The best-scoring action wins.

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
