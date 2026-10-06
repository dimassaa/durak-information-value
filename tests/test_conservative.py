"""Tests for `CONSERVATIVE`: a strength reference that is not the search.

This strategy earns its place in the dataset by being a *different point on the strength axis*, not a
stronger version of an existing one. `SEARCH_UNIFORM` and `SEARCH_INFORMED` are the same expectimax over a
particle filter; `CONSERVATIVE` reads only the filter's aggregate counts and applies hand rules, so it is
slower to write and much cheaper to run, and it fails in different ways.

The thresholds are pinned by these tests. A conservative policy is a set of hand-tuned thresholds wearing a
strategy's clothes, and the moment those numbers move the strategy is a different strategy: the constant's
own tests assert the values, so a change here cannot slip in as a "tuning" commit.
"""

from pathlib import Path

import numpy as np
import pytest

from durak import rules
from durak.actions import BEAT, STOP, TAKE, THROW, action_card, action_kind, pack_action
from durak.cards import beats, rank_of, suit_of
from durak.evaluation import HIGH_RANK
from durak.strategies import CONSERVATIVE
from durak.strategies.conservative import (
    CHEAP_TAKE_CARDS,
    HIGH_CARD_RANK_FLOOR,
    TRUMP_RESERVE_FLOOR,
    policy_conservative,
)
from durak.strategies.search_policy import PARTICLE_COUNT
from durak.rng import RngStream
from tests.support import Pool, card_at, legal_state, random_position

T = rules.DEFAULT_SETTINGS


def test_the_thresholds_are_pinned_and_documented():
    """Each threshold is a decision, so both its value and its written justification are asserted.

    Module-level ints rather than a class instance: `njit` cannot read a Python class, and a first version
    carried them on a `_Thresholds` object so that every call failed to compile. The consequence worth
    guarding is that a number could be changed without its reason being rewritten, so the docstring is
    checked for a mention of each name as well as the values themselves.
    """
    assert TRUMP_RESERVE_FLOOR == 2
    assert CHEAP_TAKE_CARDS == 2
    assert HIGH_CARD_RANK_FLOOR == 7

    from durak.strategies import conservative
    doc = conservative.__doc__ or ""
    module_doc = (Path(conservative.__file__).read_text()
                  if Path(conservative.__file__).exists() else "")
    for name, value in (("TRUMP_RESERVE_FLOOR", TRUMP_RESERVE_FLOOR),
                        ("CHEAP_TAKE_CARDS", CHEAP_TAKE_CARDS),
                        ("HIGH_CARD_RANK_FLOOR", HIGH_CARD_RANK_FLOOR)):
        assert f"{name}: int = {value}" in module_doc, (
            f"{name} is not defined at module level with its value"
        )
        assert name in doc or name in module_doc, f"{name} has no written justification"


def _defending(trump: int, beater_suit: int, beater_rank: int, my_other_trumps: int):
    """A defending position with exactly one beater available.

    The attack is a suit-3 card of rank 0, so any trump beats it and so does any later suit-3 card. The
    filler deliberately avoids trumps and higher suit-3 cards, because a fixture that accidentally offers a
    second beater makes the "it declined to beat" assertion pass for the wrong reason -- the policy would
    have skipped a card it was willing to play, and the test would be measuring the filler rather than the
    rule. Asserting the reply set has exactly one entry keeps that from happening silently.
    """
    pool = Pool()
    # The attack shares its suit with the beater when the beater is a plain non-trump, so that a beat is
    # legal at all; a trump beater can answer anything. A fixed suit-3 attack made the non-trump case
    # unbuildable -- a suit-1 card cannot beat a suit-3 card when the trump is suit 0.
    attack_suit = 3 if beater_suit == 0 else beater_suit
    attack = pool.reserve(card_at(0, attack_suit))

    # Every reserve happens BEFORE any `take`. `Pool.take` hands out the lowest available card, and these
    # cards are all low ids, so drawing filler first consumes the very cards the fixture is about to
    # reserve and `reserve` then raises "already reserved".
    beater = pool.reserve(card_at(beater_rank, beater_suit))
    # The extra trumps start above the beater's rank. Ranging from 1 collided with the beater whenever it
    # was itself a trump of rank 1, and `Pool.reserve` raised on the duplicate -- a fixture that could not
    # express the case it was written for.
    extras = [pool.reserve(card_at(beater_rank + 1 + r, 0)) for r in range(my_other_trumps)]
    hand = [beater] + extras
    # Filler must not beat the attack and must not be a trump, or "it declined to beat" passes because it
    # had three other trumps it could have played instead. Drawn by scanning card ids rather than through
    # `Pool.take`, which deals the lowest ids -- for a suit-0 trump those are almost all trumps.
    filler = []
    for card in range(rules.DEFAULT_SETTINGS.deck_size if hasattr(rules.DEFAULT_SETTINGS, "deck_size")
                      else 36):
        if len(filler) == 5 - my_other_trumps:
            break
        if card in hand or card == attack:
            continue
        if suit_of(card) != trump and not beats(card, attack, trump):
            filler.append(pool.reserve(card))
    hand.extend(filler)
    # The opponent of the hand must be the ATTACKER, so seat 0 holds the table card and seat 1 defends.
    # An earlier version passed `attacker=1`, which made `acting_player` return seat 0 -- the hand was
    # asked to defend while seat 0 held the lead, and the fixture offered no legal action at all. The
    # assertion below is what caught it: an empty reply set looked like "the trump cannot be spared" and
    # the test passed for entirely the wrong reason.
    state = legal_state(pool.take(6), hand, trump=trump, attacker=0, table_attacks=[attack])

    assert int(rules.acting_player(state)) == 1, "seat 1 must be the defender on this position"
    beaters = [int(a) for a in rules.legal_actions(state, 1) if action_kind(int(a)) == BEAT]
    # Every beater is a trump: the filler is screened so it cannot beat, and the beater is a trump, so the
    # expected count is exactly "the beater plus the reserve". Asserting `== 1` was wrong whenever
    # `my_other_trumps > 0` -- the extra trumps are beaters too, which is the whole point of the fixture.
    assert len(beaters) == my_other_trumps + 1, (
        f"expected {my_other_trumps + 1} trump beaters, got {beaters}; hand was {hand}"
    )
    return state, beater


def test_it_declines_a_trump_beat_when_the_trump_reserve_is_healthy():
    """Two other trumps in hand and one card at stake: keep the trump, let the cheap card go.

    The rule the threshold encodes: spending a trump to avoid taking a single card is a bad trade when the
    trump has a job later.
    """
    state, beater = _defending(trump=0, beater_suit=0, beater_rank=1, my_other_trumps=2)

    action = policy_conservative(state, 1, int(CONSERVATIVE), 0, np.uint64(0))

    assert action_kind(action) == TAKE, (
        "with two trumps in reserve and one card at stake it must take rather than spend the trump"
    )


def test_it_spends_the_trump_when_the_reserve_is_already_gone():
    """With no other trump held, the same trump is the right thing to play.

    The paired case. A test that only checks the refusal would pass a policy that never beats with a trump,
    which is a different and much weaker strategy.
    """
    state, beater = _defending(trump=0, beater_suit=0, beater_rank=1, my_other_trumps=0)

    action = policy_conservative(state, 1, int(CONSERVATIVE), 0, np.uint64(0))

    assert action_kind(action) == BEAT
    assert action_card(action) == beater


def test_it_always_beats_with_a_low_non_trump():
    """A cheap beater is never worth refusing: no reserve is spent and the card leaves play."""
    state, beater = _defending(trump=0, beater_suit=1, beater_rank=2, my_other_trumps=2)

    action = policy_conservative(state, 1, int(CONSERVATIVE), 0, np.uint64(0))

    assert action_kind(action) == BEAT
    assert action_card(action) == beater


def test_it_never_throws_in_a_trump_even_on_a_matching_rank():
    """Throwing a trump gives the opponent a trump and hands them the lead. Never, whatever the rank."""
    pool = Pool()
    throw_trump = pool.reserve(card_at(2, 0))
    filler = [pool.take(1)[0] for _ in range(5)]
    state = legal_state([throw_trump] + filler, pool.take(6), trump=0, attacker=0)

    action = policy_conservative(state, 0, int(CONSERVATIVE), 0, np.uint64(0))

    assert not (action_kind(action) == THROW and suit_of(action_card(action)) == 0), (
        "a trump must never be thrown in"
    )


def test_it_throws_in_a_matching_non_trump_when_it_believes_the_opponent_cannot_beat():
    """The throw-in is the strategy's only source of aggression, so it must fire in its one good case.

    Asserted as legality and as the absence of a take, not as a specific card: the point is that the
    policy attacks rather than always taking, which a "never throws" policy would also satisfy.
    """
    for seed in range(12):
        state = random_position(deck_size=36, seed=seed, max_actions=3 + seed % 5)
        if state.winner != rules.EMPTY:
            continue
        player = int(state.attacker)
        if int(state.hand_sizes[player]) < 2:
            continue
        offered = rules.legal_actions(state, player)
        if offered.shape[0] == 0:
            continue
        if action_kind(int(offered[0])) != THROW:
            continue
        action = policy_conservative(state, player, int(CONSERVATIVE), 0, np.uint64(seed))
        if action_kind(action) == THROW:
            assert action_card(action) != 0
            assert state.hands[player, : state.hand_sizes[player]].tolist().count(
                action_card(action)) == 1, "a thrown card must be held exactly once"
            assert suit_of(action_card(action)) != int(state.trump), "a trump must never be thrown in"
            return
    pytest.skip("no seeded position offered a throw-in")


def test_every_returned_action_is_legal_across_random_positions():
    """The legality invariant. A policy returning an illegal action is a crash three layers down."""
    for seed in range(80):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 20)
        if state.winner != rules.EMPTY:
            continue
        player = int(rules.acting_player(state))
        action = policy_conservative(state, player, int(CONSERVATIVE), 0, np.uint64(seed))
        assert action in rules.legal_actions(state, player).tolist(), (
            f"seed {seed} produced an illegal action"
        )


def test_it_plays_a_whole_game_without_raising():
    """End to end through the engine, so the policy is exercised in both roles across round boundaries."""
    from durak.engine import play_block
    from durak.strategies import RANDOM

    outcomes, _, _ = play_block(36, CONSERVATIVE, RANDOM, 0, 0, 4, 0, 0, 12, T, False, False)
    assert len(outcomes) == 12
    assert int((outcomes["winner"] < 0).sum()) == 0, "CONSERVATIVE must not stall the game"