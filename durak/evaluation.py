"""The leaf evaluation: five weighted differentials, and nothing else.

Every term is a difference
--------------------------
All five terms are *differentials* between the two seats, never absolute counts. That is what makes
:func:`evaluate` exactly antisymmetric under a seat swap, which
``test_swapping_the_seat_exactly_negates_the_value`` pins. It matters more than it looks: a search that
preferred one seat for reasons unrelated to the position would bias every win rate in the dataset, and
the bias would be invisible because the tournament runs both directions.

Signs, which are the entire content of this function
-----------------------------------------------------
Three terms read "opponent minus player" and two read "player minus opponent", and the split is not
arbitrary:

* **card count** reads opponent minus player, because *fewer cards is good* -- you win by emptying your
  hand.
* **trumps** and **high cards** read player minus opponent, because *more is good*.

Getting that wrong is invisible until a test asserts each direction separately. An early draft used
opponent-minus-player for all three, and the search shed its trumps first: a wrong move that looks
entirely plausible in play.

The five terms and why they carry the weights they do
-----------------------------------------------------
``card_count`` (1.0) -- the baseline. Cards are the currency the game is played in.

``trump`` (2.0) -- twice a card, because a trump is a card *and* a tempo weapon: it beats anything and
cannot be stopped, whereas an ordinary card is only something you no longer have to defend. Six trumps
in a 36-card deck are 17% of the deck and a much larger share of the wins.

``high_card`` (0.5) -- half a card, for rank >= 10 printed. A high card wins the tricks it touches and is
otherwise ordinary. The threshold is a step, not a gradient, because rank is ordinal in Durak rather
than a bidding scale, and a gradient would invent precision the rules do not contain.

``role`` (0.3) -- the smallest weight and the most debatable. Attacking first is worth something, but it
is already partly expressed through hand size (the attacker has committed cards to the table), so a
larger weight would double-count. The defensive player also gets to pick up the pile, which this term
does not model because that is the card-count term's job.

``deck_exhausted`` (1.5) -- the second-largest weight, and it is exactly zero until the draw pile runs
out. Then it dominates *and inverts*: taking means keeping the entire remainder of the game, so a player
holding one card against six can be better off than when the same position had a full deck. The
ordinary card-count differential points the wrong way in that regime, and this term is what corrects it.

Not tuned
---------
These values are a hand justification, not a fit. Tuning them against an opponent would confound algorithm
quality with tuning budget and would risk giving the informed and uninformed arms different budgets,
corrupting the primary metric before it was measured. :data:`EVAL_SWEEP_GRID` perturbs them so a report
can state how much the result depends on them -- a sensitivity analysis, not an optimisation.
"""

import numpy as np
from numba import njit

from durak.cards import rank_of

#: Rank at or above which a card is "high", from spec 5.4. Ranks are 0-based offsets, so this is the
#: sixth-highest rank in a 36-card deck.
HIGH_RANK = 5

#: Order matches the order :func:`evaluate` consumes them:
#: (card count, trumps, high cards, role, deck exhausted).
EVAL_WEIGHTS = (1.0, 2.0, 0.5, 0.3, 1.5)


def _sweep_grid() -> tuple:
    """The weight perturbations Task 12 sweeps, one weight at a time.

    Each row multiplies a single weight by ``factor`` and leaves the rest alone, plus the all-ones row
    that every swept result is compared against. Two weights moved together would make the change
unattributable, which is the entire purpose of the sweep.
    """
    factors = (0.5, 2.0)
    grid = [(1.0,) * 5]
    for index in range(5):
        for factor in factors:
            row = [1.0] * 5
            row[index] = factor
            grid.append(tuple(row))

    return tuple(grid)


EVAL_SWEEP_GRID = _sweep_grid()


@njit(cache=True)
def evaluate(state, player: int, weights) -> float:
    """How well `player` is placed, in the weighted units of `weights`. Positive is good.

    Reads a state and returns a float, so it needs none of the scratch machinery
    :mod:`durak.scratch` exists for: no construction, no copy, and therefore no exposure to the Numba
    constraint that a `GameState` cannot be constructed inline across the Python boundary.

    Both hands are profiled in one loop with plain scalar accumulators, and nothing is allocated. Three
    earlier shapes were tried and two of them failed *only* once this function was compiled into a larger
    graph -- calling it straight from Python worked in both cases, which is what made them hard to
    localise. A helper returning a three-element tuple, and a helper taking `np.zeros` out-parameters,
    each produced ``TypeError: cannot unpack non-iterable GameState object`` at the call site with
    nothing in the message pointing here. One function with scalar accumulators leaves nothing for Numba
    to mis-resolve, and the search calls this in its innermost loop, so it is also the cheapest shape.

    Verified by hand on hands (2 vs 3) with trump suit 0: attacker 1.8, defender 1.2, empty pile 0.3.

    `weights` is a plain sequence of five floats because no Python object crosses into ``nopython``;
    ``Settings.eval_weights`` is unpacked to it at the boundary.
    """
    cards = [0, 0]
    trumps = [0, 0]
    highs = [0, 0]

    for seat in range(2):
        card_count = trump_count = high_count = 0
        for i in range(state.hand_sizes[seat]):
            card = state.hands[seat, i]
            card_count += 1
            if np.uint64(np.int64(card)) % np.uint64(4) == np.uint64(np.int64(state.trump)):
                trump_count += 1
            if rank_of(card) >= HIGH_RANK:
                high_count += 1
        cards[seat] = card_count
        trumps[seat] = trump_count
        highs[seat] = high_count

    opponent = 1 - player

    # Three terms read opponent-minus-player and two read player-minus-opponent. Fewer cards is good;
    # more trumps and more high cards are good. See the module docstring for why that split is the
    # thing most worth getting right.
    value = weights[0] * (cards[opponent] - cards[player])
    value += weights[1] * (trumps[player] - trumps[opponent])
    value += weights[2] * (highs[player] - highs[opponent])
    value += weights[3] * (1.0 if state.attacker == player else -1.0)

    if state.deck_len == 0:
        # Inverted: with an empty pile, more cards in hand is good, because taking them is now fatal.
        value += weights[4] * (cards[player] - cards[opponent])

    return value
