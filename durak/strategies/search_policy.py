"""`SEARCH_UNIFORM` and `SEARCH_INFORMED`: one function, two hand sources.

This is the primary comparison of the project. Everything it reports follows from one claim: the two arms
differ **only** in where the opponent's hand comes from. Depth, evaluation weights, tie-break, reply model,
action ordering and particle count are read from the same place in both, and there is exactly one call to
:func:`durak.search.expectimax` in the function -- so the claim is structural, not a convention that a
future edit could quietly break.

The experiment made honest
--------------------------
``SEARCH_INFORMED`` reads the opponent's real hand as a single particle. ``SEARCH_UNIFORM`` builds ``k``
particles from the uniform prior over hands consistent with what is visible. Those are the only two lines
that differ.

The particle filter is **rebuilt from scratch on every call**. Not cached, and that is a correctness
decision rather than a performance one. A particle set is only valid relative to an observation history,
and a policy receives the current state with no history attached -- so a cached filter could not be
validated against the position it is being asked about, and a stale one would produce a confident answer to
a question nobody asked. Rebuilding costs ``k * U`` work, which is negligible next to the search itself.

What this buys, and what it does not
-----------------------------------
Depth 1, replies averaged, leaf scored without a continuation: the informed arm's strength is bounded from
below, so the measured price of information is a lower bound. And per the recorded limitation, a depth-1
search throws even when every throw-in can be beaten, because a beaten throw leaves both hands unchanged
and `STOP` makes the attacker pick up the table. That is pinned by a test in `tests/test_search.py` so it
stays a fact rather than a surprise when Task 12 measures win rates.
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
from durak.opponent import _draw_particles, _unseen_into
from durak.strategies.base import InfoMode
from durak.rules import GameState
from durak.search import expectimax

#: Particles drawn per decision when the hand is *not* known. The informed arm overrides this with 1, so
#: the two arms differ in the source of the hypothesis and never in how many hypotheses are averaged over a
#: hand they can see exactly.
PARTICLE_COUNT = 100

#: Pinned spellings of the two information modes, so callers need not import the enum to say which arm
#: they mean. Named here rather than at the call site because "the arm with the real hand" is the single
#: most consequential constant in the project and it should appear in exactly one place.
INFO_TRUTH = int(InfoMode.TRUTH)
INFO_PARTICLES = int(InfoMode.PARTICLES)


@njit(cache=True)
def _truth_particles(state, opponent: int) -> np.ndarray:
    """The opponent's real hand as a single particle.

    Read straight out of the state, sorted, because the filter's own particles are sorted and a mixed
    representation would make the two arms' inputs differ in a way that has nothing to do with information.
    """
    width = int(state.hand_sizes[opponent])
    out = np.empty((1, width), dtype=np.int16)
    for i in range(width):
        out[0, i] = state.hands[opponent, i]
    for j in range(1, width):
        value = out[0, j]
        back = j - 1
        while back >= 0 and out[0, back] > value:
            out[0, back + 1] = out[0, back]
            back -= 1
        out[0, back + 1] = value
    return out


@njit(cache=True)
def _prior_particles(state, opponent: int, k: int, rng_state: np.uint64) -> np.ndarray:
    """`k` hypotheses drawn from the uniform prior, rebuilt from the current state.

    The stream advances so two players never draw from the same position, which is what makes
    ``RANDOM vs RANDOM = 0.5`` a symmetry test rather than a coincidence.
    """
    unseen = np.empty(state.initial_deck, dtype=np.int16)
    u_len = _unseen_into(state, 1 - int(opponent), unseen)
    particles, _ = _draw_particles(unseen, u_len, int(state.hand_sizes[opponent]), k, rng_state)
    return particles


@njit(cache=True)
def policy_search(state: GameState, player: int, strategy_id: int, info_mode: int,
                  rng_state: np.uint64) -> np.int32:
    """Expectimax over the opponent's hand. One function for both search arms.

    `info_mode` is the whole experiment. `durak.strategies.InfoMode.TRUTH` reads the opponent's real hand
    as one particle; anything else draws `PARTICLE_COUNT` hypotheses from the uniform prior. Every other
    input -- depth, weights, tie-break, reply model, action ordering -- is below, read once, outside the
    branch. That is the point of the module: a future edit cannot give one arm a different search without
    passing through these lines.

    `rng_state` is threaded but unused, like the other policies' -- the stream lives in the state so the
    advance is recorded where the next call will read it.
    """
    opponent = 1 - player

    # --- the only thing that differs between the arms ---------------------------
    if info_mode == INFO_TRUTH:
        particles = _truth_particles(state, opponent)
    else:
        particles = _prior_particles(state, opponent, PARTICLE_COUNT, np.uint64(rng_state))
    # -----------------------------------------------------------------------------

    # One search, one set of parameters, both arms.
    return expectimax(state, player, particles, EVAL_WEIGHTS)
