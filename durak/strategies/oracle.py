"""``ORACLE``: the ceiling model. Perfect opponent hand, deeper search.

What this arm is supposed to bound
----------------------------------
The project's headline is a difference between two policies that are demonstrably weak, so it can
only be reported as a lower bound. ``ORACLE`` supplies the upper one: the same expectimax, the same
leaf, the same weights, with the opponent's hand read exactly and the search taken
:data:`ORACLE_DEPTH` plies deep. Depth is the axis here, and depth is read from the environment
before the first compile -- see "Depth is frozen at compile time" below.

``ORACLE_UNIFORM`` is the other half of the bound, and it is the half that makes the number mean
something. The headline is a *difference* -- informed against uninformed -- so bounding it requires
the uninformed arm of the same policy class, not the uninformed arm at depth 2. This module provides
both at :data:`ORACLE_DEPTH`, differing only in where the belief comes from, which is the same
structural claim ``SEARCH_UNIFORM`` and ``SEARCH_INFORMED`` make at depth 2. Without the pair,
"the oracle beats the informed arm" says only that depth matters; with it, the difference IS the
ceiling on the value of perfect knowledge in this class.

The deck order is not an axis, and that is the finding
------------------------------------------------------
The brief for this arm was "the opponent's hand exactly, AND the order of the deck". Measured
before building rather than assumed: **this engine does not hide the draw pile from anybody.**

``GameState.deck`` is a public array of the remaining cards in draw order, every policy receives the
state, and ``durak.leaf._prepare`` copies that array verbatim into the leaf's working copy, so
``_refill_one`` inside the leaf draws the *true* next cards. Verified, not inferred: over 720
positions from real games, ``deck_len`` ran 2 to 24, and a discriminating differential test
(``tests/test_oracle.py::test_the_leaf_draws_the_real_deck_order``) shows two positions differing
only in the order of their draw pile scoring differently.

So ``SEARCH_UNIFORM`` already knows the order of the deck. Adding it to ``ORACLE`` would add nothing,
and an arm that only re-asserted it would be a copy of ``SEARCH_INFORMED`` -- which is a bug report,
not an oracle. The deck-order half of the specification is therefore already satisfied by every
search arm in the project, and the value of perfect knowledge of the draw pile is exactly zero here
*by construction*, not by measurement.

The same measurement found the stronger version of the same fact, and it is recorded in
``IMPROVEMENTS.md`` because it bears on the headline rather than on this arm: the opponent's hand is
also determined. Cards are conserved, the draw pile is visible, and cards on the table and in the
discard are visible, so ``complement(visible)`` *is* the opponent's hand -- 720/720 positions. The
particle filter's candidate set is ``unseen = everything but the observer's own hand, the table and
the discard``, which at the opening is a 30-card set of which 6 are in the opponent's hand and 24 are
sitting face up in the draw pile. ``SEARCH_UNIFORM`` samples a size-6 subset of that 30 and calls it
inference. ``SEARCH_INFORMED`` is handed the answer. The measured delta between them is therefore
closer to the value of computing a set difference than to the value of epistemic knowledge, and no
oracle can change that -- it is a property of what the engine hides, not of how deep a search is.

Why this is the existing expectimax and not a rollout
-----------------------------------------------------
A brief suggested rolling the line out to a real win/loss under each particle instead of scoring a
static leaf. Rejected, for reasons that are about the model rather than about convenience:

* The rollout needs a rollout policy for the opponent. Both seats here are search policies, so a
  "free" rollout is only free if it scores the opponent with something weaker than they are -- and a
  ceiling measured against a weak opponent model is not a ceiling, it is a number for a different
  game. Giving the opponent the same search inside the rollout reproduces the expectimax at greater
  depth, which is what this module does instead.
* Rollouts to a terminal win/loss need an opponent sample per node. At the measured branching factor
  of 4.80 a 50-ply rollout is 4.8**50 leaf continuations; sampling it back to a usable variance
  means the win/loss estimate is dominated by rollout noise rather than by the decision.
* The leaf is differential-tested against ``rules.apply_action`` at depths 1 to 4, exactly, over
  random positions. A state-free 50-ply rollout engine would be new code on the hot path with none
  of that, and a rules divergence inside it would move the reported ceiling rather than crash. A
  wrong ceiling is worse than no ceiling, because the ceiling is what every other number in the
  project gets interpreted against.

Depth is frozen at compile time
-------------------------------
``@njit`` freezes module globals when it compiles, and Numba's on-disk cache does **not** invalidate
when a global's value changes. A process that compiled at depth 2 keeps depth 2 whatever this
variable is reassigned to. Every depth therefore needs its own process with its own
``NUMBA_CACHE_DIR``, which is strictly better than clearing ``__pycache__``: two experiments sharing
a cache directory silently play each other's depth. Measured, on this arm -- a fingerprint taken at
depth 2 read depth 1 for a whole run because a concurrently running experiment had written the cache
first. The default is 4 because that is the deepest depth the plan calls usable; a published oracle
number carries the depth beside it, since depth 4 against depth 2 compares two different policies.

No information mode, on purpose
------------------------------
:func:`policy_oracle` ignores ``info_mode`` and reads the hand from the state. Perfect information is
the arm's definition rather than a setting of it, so there is no configuration under which it
degrades into the uninformed one -- and a runner that passes ``NONE`` (``runner.info_modes_for`` does,
because this is not an arm it recognises) measures the oracle rather than silently measuring
something else.
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit

from durak.evaluation import EVAL_WEIGHTS
from durak.rules import GameState
from durak.search import _expectimax
from durak.strategies.search_policy import PARTICLE_COUNT, _prior_particles, _truth_particles

#: Ply depth the oracle searches to. Read from the environment rather than assigned literally,
#: because `@njit` freezes module globals at compile time: a process that compiled at depth 2 keeps
#: depth 2 whatever the variable is reassigned to afterwards, and the on-disk cache does not
#: invalidate. Measured at depths 1-4 in this project; see `docs/PLAN.md` item 2 for why depth is a
#: research parameter and not a performance knob.
ORACLE_DEPTH = int(os.environ.get("DURAK_ORACLE_DEPTH", "4"))


@njit(cache=True)
def policy_oracle(state: GameState, player: int, strategy_id: int, info_mode: int,
                  rng_state: np.uint64) -> np.int32:
    """The ceiling arm's decision: the exact opponent hand, searched :data:`ORACLE_DEPTH` plies.

    One call, and every input to it is shared with `policy_search` on purpose -- the same weights,
    the same leaf, the same tie-break, the same iterative walk, and the same `_truth_particles`
    belief rather than a second reader of the opponent's hand that could drift from it. The only
    difference between this arm and ``SEARCH_INFORMED`` is the depth argument, which is what makes
    "the oracle against ``SEARCH_INFORMED``" a measurement of search depth rather than of two
    hand-rolled searches that happen to disagree.

    `info_mode` and `rng_state` are accepted and ignored, for the reasons in the module docstring:
    the hand is read from the state because that is what the arm is, and the search draws no
    randomness. The signature matches the registry's so the engine can dispatch it unchanged.
    """
    return _expectimax(state, player, _truth_particles(state, 1 - player), EVAL_WEIGHTS,
                       ORACLE_DEPTH)


@njit(cache=True)
def policy_oracle_uniform(state: GameState, player: int, strategy_id: int, info_mode: int,
                          rng_state: np.uint64) -> np.int32:
    """The uninformed arm of the same class: :data:`ORACLE_DEPTH` plies over the uniform prior.

    Structurally the mirror of :func:`policy_oracle`, and built the way the project's primary pair is:
    one depth, one weight vector, one leaf, one tie-break, and exactly one line differing -- where the
    belief about the opponent's hand comes from. Any difference in their win rates is therefore
    attributable to the belief and to nothing else, which is the only reason the difference between
    them can be called a ceiling on the value of perfect knowledge rather than a comparison of two
    differently-tuned searches.

    The arm exists because the headline is a *difference*. Bounding a difference needs both of its
    arms at the depth being bounded: ``ORACLE`` against the depth-2 ``SEARCH_UNIFORM`` confounds
    "does perfect knowledge pay" with "does depth pay", and in this engine those two point in
    opposite directions.
    """
    return _expectimax(state, player,
                       _prior_particles(state, 1 - player, PARTICLE_COUNT, np.uint64(rng_state)),
                       EVAL_WEIGHTS, ORACLE_DEPTH)