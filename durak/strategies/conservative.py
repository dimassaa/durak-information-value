"""`CONSERVATIVE`: resource discipline from the opponent model's aggregate counts.

Where it sits on the strength axis
----------------------------------
``SEARCH_UNIFORM`` and ``SEARCH_INFORMED`` are one expectimax over a particle filter: they enumerate actions,
enumerate the opponent's replies, and score the resulting position. ``CONSERVATIVE`` reads only the
filter's *aggregate* -- how many trumps and high cards the opponent probably holds -- and applies hand
rules to it. No leaf arithmetic, no reply enumeration, no depth.

That makes it a genuinely different point on the axis rather than a cheaper copy of the search: it is the
strategy that plays correctly without enumerating, so it isolates how much of the search's strength comes
from the evaluation and how much from the enumeration. It is also the project's check that the particle
filter's aggregate is usable on its own -- if the filter were only meaningful as a set of full hands, these
rules could not be written at all.

Why it is not the same as ``GREEDY``: ``GREEDY`` is information-free by construction, and this policy is
built on the filter's output. They should land in different places on the win-rate axis, and if they do not,
one of them is not doing what its name says.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from durak.actions import BEAT, STOP, TAKE, THROW, action_card, action_kind, pack_action
from durak.cards import beats, rank_of, suit_of
from durak.opponent import marginals_and_entropy, _unseen_into
from durak.rules import EMPTY, GameState, legal_actions
from durak.strategies.search_policy import _prior_particles
from durak.strategies.search_policy import PARTICLE_COUNT


# The numbers the policy is built on. Each is a decision, not a tuning constant.
#
# Module-level ints rather than a frozen object because `njit` cannot read a Python class: an earlier version
# carried them on a `_Thresholds` instance and every call failed to compile with "Untyped global name". The
# per-value reasoning that lived in that class's docstring is here instead, so no justification was lost to
# satisfy the compiler.
#
# TRUMP_RESERVE_FLOOR: how many trumps beyond the one being spent must remain for that trump to count as
#   reserve. Two is the point where a single remaining trump can still be relied on to win one more trick;
#   with one left, spending the second leaves none.
# CHEAP_TAKE_CARDS: how many cards may be at stake before a take stops being cheap. At two or fewer the cards
#   are worth less than the trump they would cost to defend, so a healthy reserve takes rather than beats.
# HIGH_CARD_RANK_FLOOR: the rank at which a non-trump stops being cheap. Ranks at or above this are held for
#   later, because a high card is the only thing that can win a trick once the trumps are gone.
TRUMP_RESERVE_FLOOR: int = 2
CHEAP_TAKE_CARDS: int = 2
HIGH_CARD_RANK_FLOOR: int = 7


@njit(cache=True)
def _counts(hand, size: int, trump: int):
    """(trumps, high cards) in a hand prefix. The two resources this policy reasons about."""
    trumps = 0
    high = 0
    for i in range(size):
        card = int(hand[i])
        if suit_of(card) == trump:
            trumps += 1
        elif rank_of(card) >= HIGH_CARD_RANK_FLOOR:
            high += 1
    return trumps, high


@njit(cache=True)
def _estimated_opponent_resources(state: GameState, player: int, k: int, rng: np.uint64):
    """Expected (trumps, high cards) in the opponent's hand, from the particle filter's marginal.

    Deliberately the marginal rather than any single particle: a strategy that read one hypothesis would be
    sampling rather than modelling, and would swing on the draw. The expectation over the marginal is the
    part of the filter that survives even when the individual hypotheses are indistinguishable.
    """
    opponent = 1 - player
    # `_prior_particles`, not `opponent.initial_particles`: the latter is a plain Python function that takes
    # and mutates an `RngStream` object, so `njit` cannot call it ("Untyped global name"). The wrapper takes
    # the stream's integer snapshot instead, which is why the search policies use it too.
    particles = _prior_particles(state, opponent, k, rng)
    # `_unseen_into`, not `opponent.unseen_cards`: same reason as `_prior_particles` above -- the public
    # helper is a Python wrapper around this one.
    unseen = np.empty(state.initial_deck, dtype=np.int16)
    u_len = _unseen_into(state, player, unseen)
    marginals, _ = marginals_and_entropy(particles, unseen)

    trumps = 0.0
    high = 0.0
    for i in range(u_len):
        p = float(marginals[i])
        if p <= 0.0:
            continue
        card = int(unseen[i])
        if suit_of(card) == state.trump:
            trumps += p
        elif rank_of(card) >= HIGH_CARD_RANK_FLOOR:
            high += p
    return trumps, high


@njit(cache=True)
def policy_conservative(state: GameState, player: int, strategy_id: int, info_mode: int,
                        rng_state: np.uint64) -> np.int32:
    """Defend by spending nothing that is needed later; attack only with cards worth throwing.

    Returns the chosen packed action. Deliberately rule-based rather than scored: there is no leaf evaluation
    here, so a position where the rules have nothing to say falls back to the first legal candidate in
    ``legal_actions`` order, and the engine's tie-break is what makes that deterministic.

    ``rng_state`` is threaded into the particle filter and so is genuinely used here -- unlike the
    information-free policies, this one reads the opponent model. The signature stays uniform across the
    registry regardless.
    """
    if state.winner != EMPTY:
        return np.int32(-1)

    options = legal_actions(state, player)
    if options.shape[0] == 0:
        return np.int32(-1)

    opponent = 1 - player
    trump = state.trump
    my_size = int(state.hand_sizes[player])
    my_trumps, my_high = _counts(state.hands[player], my_size, trump)
    attacking = int(state.attacker) == player

    # --- defending -------------------------------------------------------------
    if not attacking:
        target = int(state.table_attacks[state.table_defense_count])
        cards_at_stake = int(state.table_attack_count - state.table_defense_count)
        cheap_take = cards_at_stake <= CHEAP_TAKE_CARDS
        est_trumps, _ = _estimated_opponent_resources(state, player, PARTICLE_COUNT, rng_state)

        for i in range(options.shape[0]):
            action = int(options[i])
            if action_kind(action) != BEAT:
                continue
            card = action_card(action)
            if suit_of(card) == trump:
                # Spending a trump defends one trick. Worth it only if the reserve is already thin, or the
                # take is expensive enough to matter more than the trump.
                if my_trumps - 1 >= TRUMP_RESERVE_FLOOR and cheap_take:
                    continue
                return np.int32(action)
            # A cheap non-trump is always worth playing: no reserve is spent and the card leaves play.
            if rank_of(card) >= HIGH_CARD_RANK_FLOOR and my_high <= 1 \
                    and cheap_take:
                continue
            return np.int32(action)
        return np.int32(pack_action(TAKE, 0))

    # --- attacking -------------------------------------------------------------
    best = np.int32(-1)
    for i in range(options.shape[0]):
        action = int(options[i])
        kind = action_kind(action)

        if kind == STOP:
            continue
        if kind == TAKE:
            continue

        card = action_card(action)
        if kind == THROW:
            # Never a trump, and never a rank the defending player could match cheaply.
            if suit_of(card) == trump:
                continue
            target_rank = rank_of(int(state.table_attacks[state.table_attack_count - 1]))
            if rank_of(card) != target_rank:
                continue
            return np.int32(action)

        # Opening attack: the lowest card that is not a trump. Attacking with a trump invites a beat for
        # nothing, and the reserve rule applies here exactly as it does on defense.
        if suit_of(card) != trump:
            best = np.int32(action)
            break

    if best >= 0:
        return best

    # Nothing the rules want to attack with: fall back to the cheapest legal action.
    for i in range(options.shape[0]):
        action = int(options[i])
        if action_kind(action) == STOP:
            return np.int32(action)
    return np.int32(int(options[0]))