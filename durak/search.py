"""Expectimax over particle hypotheses: the informed strategy's decision rule.

Depth is a PARAMETER
--------------------
One implementation, not one per depth, and the walk is ITERATIVE rather than recursive. Both are forced.
The leaf plays any ply count in a loop (`durak.leaf.leaf_profile_after_n`) and the ply enumeration answers
for any ply count (`durak.leaf.plies_after`), so depth reaches the model without a code path per depth.
And numba compiles a recursive `@njit` function in the pytest process and then kills every spawned worker
with `BrokenProcessPool` -- ruled out by measurement, not reasoning: not xdist (`-n0` fails identically),
not a cold cache (a single-process warm-up does not help), not the runtime recursion (`SEARCH_DEPTH = 1`
fails exactly like depth 2). See IMPROVEMENTS.md 60.

Depth 2 is the default and is not privileged in the code. It was privileged in the RESULT, which is a
different thing: measured on one harness, the information effect was +0.0176 at depth 1 and -0.0527 at
depth 2. **Depth is a research parameter, not a performance knob**, so it arrives from the run
configuration, is recorded in the manifest, and must appear next to every published number -- depth 4
against depth 1 compares two different policies.

Modelling choices that are load-bearing
--------------------------------------
**The opponent's replies are averaged, not maximised.** Every legal reply is enumerated and the leaf values
averaged. That makes the search *optimistic* about moves with bad worst cases -- it asks "what happens on
average" rather than "what happens if they play well". Maximisation would model an opponent who always
finds the best reply, which is stronger than any policy in this study and would make the informed arm look
worse than it is.

This is a modelling decision, not a bug, and it is stated here because **both arms must average**. The
uninformed arm runs this same average over its own particle set, so the only difference between the arms is
where the hypothesis came from. Changing one arm's reply model without the other's would measure the reply
model instead of the information.

**The acting player's own next move is maximised.** Ply two is the acting player again, so it is their own
node. Averaging there would model the player choosing at random, which no policy in this study does.

Ply parity carries that distinction once the recursion is gone: ply 0 is the acting player, so an EVEN ply
index is theirs and is maximised, an ODD one is the opponent's and is averaged.

**The line stops when the rules stop it.** A ply that empties a hand ends the game; a ply that ends the
round ends the sequence. Both surface as an empty enumeration, so the leaf scores the plies that actually
happened rather than padding to `depth`.

Particles enter by substitution, not by weighting
------------------------------------------------
The opponent's replies are enumerated against one particle's cards at a time. That is what makes the
hypothesis set matter: a card the search believes is in their hand is a card it believes can be played, so
the branching factor itself changes with the belief.

The alternative -- enumerate only the moves visible to the observer and reweight outcomes -- cannot work
here. A card not in the observer's hand is a card the observer cannot name, so the set of replies is only
enumerable *given* a hypothesis.

No states, no working memory
----------------------------
The leaf profile is arithmetic over two card arrays and the ply enumerations are arithmetic over the same
arrays, so no derived state is ever materialised. `GameState` cannot be constructed inside compiled code
reached from Python, which is why the leaf is arithmetic rather than a wrapper around one. The caller's
state is read-only throughout, which is what makes `test_the_search_does_not_modify_the_state` mean
anything.

Determinism
-----------
Ties go to the earlier action in `legal_actions` order: the scan uses a strict `>`, so the first candidate
reaching the maximum is kept. Fixed and documented rather than randomised, because a tie-break by RNG would
still be legal, still be "informed", and would quietly destroy the dataset's reproducibility.
"""

import numpy as np
from numba import njit

from durak.actions import action_card, action_kind
from durak.leaf import leaf_profile_after_n, plies_after
from durak.rules import EMPTY, legal_actions

#: Replies enumerated per particle before the search refuses rather than truncating.
#:
#: One action's answers are the opponent's legal replies to it: every beater plus taking, or every throw-in
#: plus stopping. The count is bounded by the opponent's hand plus one. The cap guards against a pathological
#: position, and exceeding it *raises* rather than truncating: a truncated enumeration makes the opponent
#: look weaker than the rules allow, which biases the search toward attacking and looks like strength.
RESPONSE_CAP = 64

#: The depth used when a caller does not state one. NOT a default in the nopython sense: `njit` silently
#: ignores Python-level default arguments, so `depth` is always passed explicitly by every entry point.
DEFAULT_DEPTH = 2


@njit(cache=True)
def _line_value(state, player: int, nplies: int, ply_kinds, ply_cards, depth: int,
                hand_p, hand_o, weights, opt_kinds, opt_cards) -> float:
    """The value of the line `nplies` plies in, choosing `depth` further plies.

    `depth` counts the plies still to be CHOSEN, so the leaf scores `nplies + depth` plies in total: the
    caller's own action is one of them. Depth 1 therefore chooses the opponent's reply and scores two plies,
    which is the depth-1 leaf; depth 2 additionally chooses the acting player's own reply.

    An ordinary explicit-stack depth-first walk. Level bookkeeping, all indexed by level so no two live
    frames share storage:

      `opt_*[level]`  the options enumerated there. A deeper level writes its own row, so a row an outer
                      loop is still walking is never overwritten.
      `cursor[level]` which option is next.
      `acc[level]`    the running value: a maximum, so seeded at -inf, or a sum, so seeded at 0.
      `count[level]`  how many options that level had.
    """
    if depth <= 0:
        return leaf_profile_after_n(state, player, nplies, ply_kinds, ply_cards, hand_p, hand_o, weights)

    top = nplies + depth - 1
    cursor = np.zeros(top + 1, dtype=np.int64)
    count = np.zeros(top + 1, dtype=np.int64)
    acc = np.zeros(top + 1, dtype=np.float64)

    level = nplies
    is_max = (level % 2) == 0
    first = plies_after(state, player, nplies, ply_kinds, ply_cards, hand_p, hand_o,
                        opt_kinds[level], opt_cards[level])
    if first > RESPONSE_CAP:
        raise ValueError(
            f"a player had more than {RESPONSE_CAP} legal moves at ply {nplies}; raise RESPONSE_CAP. "
            f"Truncating would make that player look weaker than the rules allow."
        )
    if first == 0:
        return leaf_profile_after_n(state, player, nplies, ply_kinds, ply_cards, hand_p, hand_o, weights)
    count[level] = first
    acc[level] = -1e308 if is_max else 0.0

    while True:
        c = cursor[level]
        if c >= count[level]:
            # Exhausted: fold this level and hand the value to the parent.
            value = acc[level] if is_max else acc[level] / count[level]
            level -= 1
            if level < nplies:
                return value
            is_max = (level % 2) == 0
            if is_max:
                if value > acc[level]:
                    acc[level] = value
            else:
                acc[level] += value
            cursor[level] += 1
            continue

        ply_kinds[level] = opt_kinds[level][c]
        ply_cards[level] = opt_cards[level][c]

        if level == top:
            value = leaf_profile_after_n(state, player, level + 1, ply_kinds, ply_cards,
                                         hand_p, hand_o, weights)
            if is_max:
                if value > acc[level]:
                    acc[level] = value
            else:
                acc[level] += value
            cursor[level] += 1
            continue

        below = plies_after(state, player, level + 1, ply_kinds, ply_cards, hand_p, hand_o,
                            opt_kinds[level + 1], opt_cards[level + 1])
        if below > RESPONSE_CAP:
            raise ValueError(
                f"a player had more than {RESPONSE_CAP} legal moves at ply {level + 1}; raise RESPONSE_CAP."
            )
        if below == 0:
            # The rules ended the line here. Score what happened; do not pad to `depth`.
            value = leaf_profile_after_n(state, player, level + 1, ply_kinds, ply_cards,
                                         hand_p, hand_o, weights)
            if is_max:
                if value > acc[level]:
                    acc[level] = value
            else:
                acc[level] += value
            cursor[level] += 1
            continue

        level += 1
        is_max = (level % 2) == 0
        count[level] = below
        cursor[level] = 0
        acc[level] = -1e308 if is_max else 0.0

    return 0.0


@njit(cache=True)
def action_value(state, action: int, player: int, particles: np.ndarray, weights,
                 depth: int) -> float:
    """The expected leaf score of playing `action`, averaged over replies and particles.

    The reply list is recomputed inside each particle's hypothesis rather than once against the real hand.
    Enumerating once would hand the informed arm the opponent's actual cards for free -- the single worst
    bug available in this module, and one that would drive the measured price of information to roughly zero
    and look like a finding.
    """
    kind = action_kind(action)
    card = action_card(action)

    k = particles.shape[0]
    total = 0.0
    counted = 0

    # One set of buffers per search rather than one per call: the enumerations write into them and return
    # a count, and an allocation inside the innermost loop would dominate the cost.
    ply_kinds = np.empty(depth + 2, dtype=np.int64)
    ply_cards = np.empty(depth + 2, dtype=np.int16)
    rows_kinds = np.empty((depth + 2, RESPONSE_CAP + 2), dtype=np.int64)
    rows_cards = np.empty((depth + 2, RESPONSE_CAP + 2), dtype=np.int16)

    for i in range(k):
        ply_kinds[0] = kind
        ply_cards[0] = card
        total += _line_value(state, player, 1, ply_kinds, ply_cards, depth,
                             state.hands[player], particles[i], weights, rows_kinds, rows_cards)
        counted += 1

    if counted == 0:
        return 0.0
    return total / counted


@njit(cache=True)
def _expectimax(state, player: int, particles: np.ndarray, weights, depth: int) -> np.int32:
    """The action `player` should take as a packed int32, or ``-1`` if none is legal.

    Every legal action is played, the opponent's every legal reply is enumerated under every particle
    hypothesis, the acting player's next move is maximised and the opponent's averaged, `depth - 1` times,
    and each resulting position is scored.

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
        score = action_value(state, action, player, particles, weights, depth)
        # Strict `>`, so the first candidate reaching the maximum wins: a documented tie-break, not an
        # accident of iteration order.
        if score > best_score:
            best_score = score
            best_action = action

    return np.int32(best_action)


def expectimax(state, player: int, particles: np.ndarray, weights, depth: int = DEFAULT_DEPTH) -> np.int32:
    """The depth-parameterised entry point. A thin Python wrapper, deliberately.

    `@njit` **silently ignores Python-level default arguments**, so a default declared on the compiled
    function would be quietly dropped and every caller would have to pass a depth. That trap is recorded in
    IMPROVEMENTS.md; the wrapper is where the default lives, and the compiled implementation takes the depth
    explicitly with no default to ignore.
    """
    return _expectimax(state, player, particles, weights, depth)