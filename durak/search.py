"""Expectimax over particle hypotheses: the informed strategy's decision rule.

Depth 1, and two modelling choices that are load-bearing
--------------------------------------------------------
**The opponent's reply is averaged, not maximised.** Every legal reply is enumerated and the leaf values
averaged. That makes the search *optimistic* about moves with bad worst cases -- it asks "what happens
on average" rather than "what happens if they play well". Maximisation would model an opponent who always
finds the best reply, which is stronger than any policy in this study and would make the informed arm
look worse than it is.

This is a modelling decision, not a bug, and it is stated here because **both arms must average**. The
uninformed arm runs this same average over its own single hypothesis, so the only difference between the
arms is where the hypothesis came from. Changing one arm's response model without the other's would
measure the response model instead of the information.

**The leaf is scored without a recursive continuation.** One ply deep, then
:func:`durak.evaluation.evaluate`. A deeper search could only find moves this one misses, so the informed
arm's strength is bounded from *below* -- which makes the measured price of information a lower bound
too. The true cost is at least what this reports.

Particles enter by substitution, not by weighting
-------------------------------------------------
The opponent's reply is enumerated in a position where their hand is *replaced* by one particle's cards.
That is what makes the hypothesis set matter: a card the search believes is in their hand is a card it
believes can be played, so the branching factor itself changes with the belief.

The alternative -- enumerate only the moves visible to the observer and reweight outcomes -- cannot work
here. A card not in the observer's hand is a card the observer cannot name, so the set of replies is only
enumerable *given* a hypothesis.

Determinism
-----------
Ties go to the earlier action in ``legal_actions`` order: the scan uses a strict ``>``, so the first
candidate reaching the maximum is kept. Fixed and documented rather than randomised, because a tie-break
by RNG would still be legal, still be "informed", and would quietly destroy the dataset's
reproducibility.

State handling
--------------
No function here constructs a `GameState`; :mod:`durak.scratch` explains why that is a hard constraint at
this boundary and why the scratch states are caller-owned. The search mutates them and copies back before
each use, and never touches the caller's `state`.
"""

import numpy as np
from numba import njit

from durak.evaluation import evaluate
from durak.rules import DEFAULT_SETTINGS, apply_action, legal_actions
from durak.scratch import ScratchStates, _copy_into

#: Replies enumerated per particle before the search refuses rather than truncating.
#
# The defender's legal replies are "beat with any card that beats this" plus "take", so the count is
# bounded by hand size -- eight for a six-card hand. The cap guards against a pathological post-take
# defender, and exceeding it *raises* rather than truncating: a truncated enumeration makes the opponent
# look weaker than the rules allow, which biases the search toward attacking and looks like strength.
RESPONSE_CAP = 64


@njit(cache=True)
def _substitute_hand(state, player: int, hand: np.ndarray, width: int) -> None:
    """Replace `player`'s hand in place with one particle's cards.

    In place, because the caller has already copied the pristine state into the scratch. The vacated
    slots are cleared rather than left stale: a leftover card past the new hand size would be counted by
    nothing and would still be a lie about the position.

    *Populates* the buffer it is given. It never copies a source state, so unlike
    :func:`durak.scratch._copy_into` it does not overwrite anything else -- but a caller must still
    clear the destination first, because this only writes one player's hand.
    """
    size = width
    if size > state.hands.shape[1]:
        size = state.hands.shape[1]
    for i in range(size):
        state.hands[player, i] = hand[i]
    for i in range(size, state.hands.shape[1]):
        state.hands[player, i] = -1
    state.hand_sizes[player] = size


@njit(cache=True)
def leaf_value(state, player: int, particles: np.ndarray, weights, scratch) -> float:
    """The mean leaf score of `state` from `player`'s view, over the particle set.

    Each particle is a hypothesis about the *opponent's* hand, substituted in turn. This is the informed
    arm's whole advantage and the only place the hypotheses are used: replace this loop with a single pass
    over one hypothesis and :func:`expectimax` still returns a legal, deterministic, sensible-looking
    move -- it simply stops being informed, and nothing would say so.
    """
    k, width = particles.shape
    if k == 0:
        return 0.0

    opponent = 1 - player
    total = 0.0
    for i in range(k):
        _copy_into(scratch, state)
        _substitute_hand(scratch, opponent, particles[i], width)
        total += evaluate(scratch, player, weights)

    return total / k


@njit(cache=True)
def action_value(state, action: int, player: int, particles: np.ndarray, weights,
                 after, hypothesis, reply) -> float:
    """The expected leaf score of playing `action`, averaged over replies and particles.

    The reply list is recomputed inside each particle's hypothesis rather than once on the post-action
    state, because which cards can be played *is* the hypothesis. Enumerating once against a state that
    still holds the opponent's real hand would hand the informed arm their actual cards for free -- the
    single worst bug available in this module, and one that would drive the measured price of information
    to roughly zero and look like a finding.

    `after`, `hypothesis` and `reply` are the caller's three scratch buffers; all three are overwritten
    before use, and `state` is only read.
    """
    _copy_into(after, state)
    apply_action(after, action, DEFAULT_SETTINGS)

    opponent = 1 - player
    k, width = particles.shape
    total = 0.0
    replies = 0

    for i in range(k):
        _copy_into(hypothesis, after)
        _substitute_hand(hypothesis, opponent, particles[i], width)

        options = legal_actions(hypothesis, opponent)
        if options.shape[0] == 0:
            # The opponent cannot act: the round is over or the hand is empty. Score it as it stands.
            total += evaluate(hypothesis, player, weights)
            replies += 1
            continue

        for r in range(options.shape[0]):
            if r >= RESPONSE_CAP:
                raise ValueError(
                    f"the opponent had more than {RESPONSE_CAP} legal replies; raise RESPONSE_CAP. "
                    f"Truncating would make the opponent look weaker than the rules allow."
                )
            _copy_into(reply, hypothesis)
            apply_action(reply, int(options[r]), DEFAULT_SETTINGS)
            total += evaluate(reply, player, weights)
            replies += 1

    if replies == 0:
        return evaluate(after, player, weights)
    return total / replies


@njit(cache=True)
def _expectimax(state, player: int, particles: np.ndarray, weights, after, hypothesis, reply) -> int:
    """The compiled search. See :func:`expectimax`."""
    options = legal_actions(state, player)
    if options.shape[0] == 0:
        return -1

    best_action = int(options[0])
    best_score = -1e308

    for i in range(options.shape[0]):
        action = int(options[i])
        score = action_value(state, action, player, particles, weights, after, hypothesis, reply)
        # Strict `>`, so the first candidate reaching the maximum wins: a documented tie-break, not an
        # accident of iteration order.
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def _scratch_or_new(scratch, deck_size: int):
    """The caller's scratch states, or a fresh set sized for `deck_size`.

    Allocating here costs about 190 us per state, which is why the engine's policies pass their own. The
    fallback exists so the function is usable from a test or a one-off query without the caller having to
    know anything about buffers.
    """
    return ScratchStates(deck_size) if scratch is None else scratch


def enumerate_responses(state, action: int, player: int, scratch=None):
    """The opponent's legal replies to `action`, as packed int32.

    Exactly ``legal_actions`` on the resulting position: no truncation, no reordering. It exists as a named
    function because bounding the branching factor is the reason this search is affordable, and a bound
    that returned too few replies would make the opponent look weaker than the rules allow.

    *Overwrites* `scratch.after`, so a caller who pre-loaded that buffer with a hypothesis loses it. That
    is the trap that made one draft test compare two particles and see the same reply set twice.
    """
    buffers = _scratch_or_new(scratch, int(state.initial_deck))
    _copy_into(buffers.after, state)
    apply_action(buffers.after, action, DEFAULT_SETTINGS)
    return legal_actions(buffers.after, player)


def responses_in(state, action: int, player: int, buffer) -> np.ndarray:
    """The opponent's legal replies to `action` in a state the caller has already prepared.

    For a hypothesis-based position: use this, not :func:`enumerate_responses`, because that one
    overwrites the buffer with `state` and would discard the substituted hand.
    """
    apply_action(buffer, action, DEFAULT_SETTINGS)
    return legal_actions(buffer, player)


def expectimax(state, player: int, particles: np.ndarray, weights, scratch=None) -> int:
    """The action `player` should take as a packed int32, or ``-1`` if none is legal.

    Depth 1: every legal action is played, the opponent's every legal reply is enumerated under every
    particle hypothesis, the resulting positions are scored, and the total is averaged over replies and
    over particles. The best-scoring action wins.

    Mutates nothing the caller owns: `state` and `particles` are only read, and `scratch` is fully
    overwritten before each use. A search that applied its move to the caller's state would make every
    recorded game depend on whether the search ran, and the cross-mode equivalence probe would stop
    comparing like with like.
    """
    buffers = _scratch_or_new(scratch, int(state.initial_deck))
    return _expectimax(state, player, particles, weights, buffers.after, buffers.hypothesis,
                       buffers.reply)


def mean_leaf_value(state, player: int, particles: np.ndarray, weights, scratch=None) -> float:
    """Mean leaf score over the particle set, with the scratch supplied for you.

    Suffixed rather than shadowing the compiled :func:`leaf_value`, because ``@njit`` silently ignores a
    Python-level default argument: the dispatcher would raise ``TypeError: not enough arguments`` from
    inside itself rather than applying the default.
    """
    buffers = _scratch_or_new(scratch, int(state.initial_deck))
    return leaf_value(state, player, particles, weights, buffers.hypothesis)
