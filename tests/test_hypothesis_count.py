"""How much resolution the particle-based search actually had, per game.

The search draws `PARTICLE_COUNT` hypotheses about the opponent's hand and averages its action values
over them, so `PARTICLE_COUNT` is a claim about the search's resolution rather than a measurement of
it. Two things make the drawn set smaller than the claim. Drawn hands collide once the candidate set
shrinks -- late in a game the opponent may hold two cards out of a handful of unseen -- and the
candidate set itself is a *memory-bounded* approximation, because `opponent._unseen_into` sees only
the last `DURAK_MEMORY_PLY` discarded cards. Neither is visible from outside, so both are recorded.

**The statistic is a sum over decisions and a count of them, not a minimum.** A minimum was tried first
and measured: it came out as exactly 1 in 100 games out of 100. That is not depletion -- it is the
endgame. When the opponent holds one card and the other cards are all accounted for, there is
genuinely only one hand to believe, the hundred draws coincide by construction, and the search is
certain for the right reason. A per-game minimum is therefore dominated by the one decision where
certainty is correct, and it reported 1 while the rest of the game went unmeasured. The mean is not
available from the accumulator alone, which is why there are two columns.

The sentinel is zero, not negative, and it is unambiguous only because there are two of them: a game
with no particle strategy has zero counted decisions, whereas a game that counted decisions and still
summed to zero would mean every single decision collapsed to one hypothesis, which is the defect these
columns exist to expose.
"""

import numpy as np
import pytest

from durak.engine import GAME_DTYPE, play_block
from durak.strategies import (
    GREEDY,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
    InfoMode,
)
from durak.strategies.search_policy import PARTICLE_COUNT
from durak.telemetry import level1_schema

SUM = "distinct_hypotheses_sum"
DECISIONS = "search_decisions"


def _block(p0, p1, m0, m1, seed, games):
    from durak import rules

    return play_block(36, p0, p1, m0, m1, seed, 0, 0, games, rules.DEFAULT_SETTINGS,
                      record_rounds=False, record_moves=False)[0]


@pytest.mark.parametrize("column", [SUM, DECISIONS])
def test_a_game_without_a_particle_strategy_counts_nothing(column):
    outcomes = _block(RANDOM, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 20)

    assert np.all(outcomes[column] == 0)


def test_the_truth_arm_counts_one_hypothesis_at_every_decision():
    outcomes = _block(SEARCH_INFORMED, SEARCH_INFORMED, InfoMode.TRUTH, InfoMode.TRUTH, 3, 3)

    assert np.all(outcomes[DECISIONS] > 0)
    assert np.all(outcomes[SUM] == outcomes[DECISIONS])


def test_the_particle_arm_counts_more_than_one_hypothesis_per_decision_on_average():
    outcomes = _block(SEARCH_UNIFORM, GREEDY, InfoMode.PARTICLES, InfoMode.NONE, 4, 20)
    counted = outcomes[outcomes[DECISIONS] > 0]

    assert np.all(counted[DECISIONS] > 0)
    # Every decision has at least one hypothesis by construction, and a mean of exactly one would mean
    # the set collapsed every single time -- which is the failure these columns exist to detect.
    assert np.all(counted[SUM] > counted[DECISIONS])


def test_no_decision_can_exceed_the_particle_budget():
    outcomes = _block(SEARCH_UNIFORM, GREEDY, InfoMode.PARTICLES, InfoMode.NONE, 4, 20)
    counted = outcomes[outcomes[DECISIONS] > 0]

    assert np.all(counted[SUM] <= PARTICLE_COUNT * counted[DECISIONS])


@pytest.mark.parametrize("column", [SUM, DECISIONS])
def test_the_columns_reach_the_level_one_schema(column):
    assert column in GAME_DTYPE.names
    assert column in level1_schema().names