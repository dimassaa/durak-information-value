"""The first four policies.

All four share one shape: enumerate the legal actions, score each by a fixed preference, and
take the best. They never construct an action and never check legality afterwards, because a
policy that builds an action from scratch has to re-derive the legality rules -- and a
re-derivation is a second implementation of them, free to disagree with
``durak.rules``. Scoring the list the rules already produced makes legality structural: a
policy here *cannot* return an illegal move, and no test is needed to prove it does not.

Determinism
-----------
Ties are broken by position in the legal-action list, taking the first strictly better
candidate. ``legal_actions`` enumerates hand slots in order, so the tie-break follows hand
order rather than depending on how a set or dict happened to iterate -- a property that
would differ between the two execution modes the equivalence test compares.

Card preference
---------------
``GREEDY`` and both auto drivers prefer a non-trump over a trump, and within a class prefer
the lower rank. The rank comparison uses the *printed* rank rather than the id, so "lowest"
means lowest in the usual sense; ``beats`` compares ids, which orders identically inside one
deck, so the two never disagree about which card is higher.

Randomness
----------
Only ``RANDOM`` draws from a stream. It draws through the player's own stream in the state
(``rng_p0`` / ``rng_p1``), never the deal's, so a run that differs in one strategy differs
only in that strategy's decisions. The draw also advances the stream in the state, because a
policy returns only its action and has no other channel to report the consumed stream.
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
    pack_action,
)
from durak.cards import beats, rank_of
from durak.rng import next_below
from durak.rules import GameState, legal_actions
from durak.strategies.base import InfoMode, StrategyId

# Sentinels for "no card of this preference class exists". Larger than any real card id, so
# a comparison against them never accidentally matches one.
_NO_CARD = 1 << 20


@njit(cache=True, inline="always")
def _preferred_attack_card(state: GameState, player: int) -> int:
    """The lowest non-trump card in `player`'s hand, else the lowest trump.

    An attack has no rank constraint -- any card in hand opens the round -- so this is a
    straight two-class preference. Returns ``_NO_CARD`` for an empty hand, which the caller
    cannot encounter: a player with an empty hand has already won.
    """
    best_non_trump = _NO_CARD
    best_trump = _NO_CARD
    size = state.hand_sizes[player]

    for i in range(size):
        card = state.hands[player, i]
        if card % 4 == state.trump:
            if best_trump == _NO_CARD or rank_of(card) < rank_of(best_trump):
                best_trump = card
        else:
            if best_non_trump == _NO_CARD or rank_of(card) < rank_of(best_non_trump):
                best_non_trump = card

    if best_non_trump != _NO_CARD:
        return best_non_trump
    return best_trump


@njit(cache=True, inline="always")
def _preferred_throwin(state: GameState, player: int) -> int:
    """The best rank-matching card in hand, or ``_NO_CARD`` when none may be thrown.

    Scans the hand against the rank set already on the table rather than filtering the legal
    list, for one reason: the cap check lives in `durak.rules`, and duplicating it here would
    be a second implementation of the throw-in limit free to disagree with the first. The
    `_is_legal` assertion in the caller is what guarantees this returns a legal action, and
    the legality suite is what guarantees it never does not.
    """
    trump_suit = state.trump
    best_non_trump = _NO_CARD
    best_trump = _NO_CARD
    size = state.hand_sizes[player]

    for i in range(size):
        card = state.hands[player, i]
        if not _rank_on_table(state, card):
            continue
        if card % 4 == trump_suit:
            if best_trump == _NO_CARD or rank_of(card) < rank_of(best_trump):
                best_trump = card
        else:
            if best_non_trump == _NO_CARD or rank_of(card) < rank_of(best_non_trump):
                best_non_trump = card

    if best_non_trump != _NO_CARD:
        return best_non_trump
    return best_trump


@njit(cache=True, inline="always")
def _rank_on_table(state: GameState, card: int) -> bool:
    """Whether `card`'s rank already appears among the table's attack or defense cards.

    Mirrors spec rule R7 with ``throwin_defense_ranks`` in force: any card on the table
    opens its rank. Reading the flag here rather than hardcoding one reading is what makes the
    variant a recorded setting rather than a hardcoded assumption.
    """
    target_rank = rank_of(card)
    for i in range(state.table_attack_count):
        if rank_of(state.table_attacks[i]) == target_rank:
            return True
    if state.throwin_defense_ranks:
        for i in range(state.table_defense_count):
            if rank_of(state.table_defenses[i]) == target_rank:
                return True
    return False


@njit(cache=True, inline="always")
def _lowest_beater(state: GameState, player: int, target: int) -> int:
    """The lowest card in `player`'s hand that beats `target`, or ``_NO_CARD``."""
    best = _NO_CARD
    size = state.hand_sizes[player]

    for i in range(size):
        card = state.hands[player, i]
        if beats(card, target, state.trump):
            if best == _NO_CARD or rank_of(card) < rank_of(best):
                best = card
    return best


@njit(cache=True, inline="always")
def _first_legal(state: GameState, player: int, kind: int) -> int:
    """The first legal action of `kind` in the rules' own enumeration order, else -1.

    The fallback path for every preference in this module. It exists so a policy never has to
    re-derive a rule: the throw-in cap lives in `durak.rules`, and a policy that recomputed it
    would be a second implementation free to disagree with the first. Instead the preference
    proposes, the rules dispose, and this is what the policy falls back to.
    """
    actions = legal_actions(state, player)
    for i in range(len(actions)):
        if action_kind(int(actions[i])) == kind:
            return int(actions[i])
    return -1


@njit(cache=True, inline="always")
def _any_legal(state: GameState, player: int) -> int:
    """The first legal action of any kind, else -1.

    Reached only if every preferred kind turned out to be illegal, which would be a bug in the
    policy rather than a state the rules produce. Returning the first legal action keeps a game
    playable and the anomaly visible in the telemetry, instead of aborting a million-game run
    at the first occurrence.
    """
    actions = legal_actions(state, player)
    if len(actions) == 0:
        return -1
    return int(actions[0])


@njit(cache=True, inline="always")
def _preferred_or_fallback(state: GameState, player: int, preferred: int, kind: int) -> int:
    """`preferred` if the rules list it as legal, else the first legal action of `kind`.

    The single point where a preference becomes an action. Legality is decided by
    ``legal_actions`` and nowhere else, so no policy in this file can be wrong about what is
    allowed -- at worst it picks a different legal move than intended, which is visible in the
    win rates rather than silent.
    """
    if preferred >= 0 and _is_legal(state, player, preferred):
        return preferred
    return _first_legal(state, player, kind)


@njit(cache=True, inline="always")
def _is_legal(state: GameState, player: int, action: int) -> bool:
    """Whether `action` appears in the acting player's legal list."""
    actions = legal_actions(state, player)
    for i in range(len(actions)):
        if int(actions[i]) == action:
            return True
    return False


@njit(cache=True)
def policy_auto_attack(state: GameState, player: int, strategy_id: int, info_mode: int,
                       rng_state: np.uint64) -> np.int32:
    """Attacks with the lowest non-trump; always takes when defending.

    A throwaway driver: it generates games for the engine and the invariant tests before the
    real policies exist. It never defends, so a game against it ends by running out of
    attacks -- which is why it is a driver and not a reported strategy.
    """
    if player == state.attacker and state.table_attack_count == 0:
        action = _preferred_or_fallback(
            state, player, pack_action(ATTACK, _preferred_attack_card(state, player)), ATTACK
        )
    elif player == state.attacker:
        action = _preferred_or_fallback(state, player, pack_action(STOP, 0), STOP)
    else:
        action = _preferred_or_fallback(state, player, pack_action(TAKE, 0), TAKE)

    fallback = _any_legal(state, player)
    return np.int32(action if action >= 0 else fallback)


@njit(cache=True)
def policy_auto_defend(state: GameState, player: int, strategy_id: int, info_mode: int,
                       rng_state: np.uint64) -> np.int32:
    """Beats with the lowest beater when defending; attacks with the lowest non-trump.

    The mirror of ``AUTO_ATTACK``, and the pair is the plan's "both auto policies" so the
    legality suite covers a driver from each side.
    """
    if player == state.attacker and state.table_attack_count == 0:
        action = _preferred_or_fallback(
            state, player, pack_action(ATTACK, _preferred_attack_card(state, player)), ATTACK
        )
    elif player == state.attacker:
        action = _preferred_or_fallback(state, player, pack_action(STOP, 0), STOP)
    elif state.table_attack_count > state.table_defense_count:
        target = state.table_attacks[state.table_defense_count]
        beater = _lowest_beater(state, player, target)
        if beater == _NO_CARD:
            action = _preferred_or_fallback(state, player, pack_action(TAKE, 0), TAKE)
        else:
            action = _preferred_or_fallback(state, player, pack_action(BEAT, beater), BEAT)
    else:
        action = _preferred_or_fallback(state, player, pack_action(TAKE, 0), TAKE)

    fallback = _any_legal(state, player)
    return np.int32(action if action >= 0 else fallback)


@njit(cache=True)
def policy_random(state: GameState, player: int, strategy_id: int, info_mode: int,
                  rng_state: np.uint64) -> np.int32:
    """Uniform over the legal actions, including `TAKE` and `STOP`.

    The one exact assertion in the project is that this against itself gives 0.5. That
    symmetry fails for any reason other than a genuine bias -- a shared stream between the
    two players, a tie-break that favours a seat, a deal that is not uniform -- so the
    policy draws from the *acting player's* stream and nothing else.

    `rng_state` is ignored: the stream lives in the state, so that the advance is recorded
    where the next call will read it. A policy that returned a stream instead of an action
    would need a different signature, and the engine would have to thread both.
    """
    actions = legal_actions(state, player)
    count = len(actions)
    if count == 0:
        raise ValueError("no legal actions available")

    if player == 0:
        index, next_state = next_below(state.rng_p0, count)
        state.rng_p0 = np.uint64(next_state)
    else:
        index, next_state = next_below(state.rng_p1, count)
        state.rng_p1 = np.uint64(next_state)

    return np.int32(actions[int(index)])


@njit(cache=True)
def policy_greedy(state: GameState, player: int, strategy_id: int, info_mode: int,
                  rng_state: np.uint64) -> np.int32:
    """The plan's strong, simple, information-free reference (spec 5.5).

    * beats with the lowest card that can beat, and never declines an available beat;
    * attacks with the lowest non-trump, falling back to the lowest trump;
    * throws in a rank-matching card while the throw-in limit allows, otherwise stops.

    Two of those are the whole strategy, and both are worth stating as consequences rather
    than as code:

    Never declining a beat makes the strategy give its opponent a free round every time a
    cheap card would have done. That is a real weakness, accepted on purpose: this is a
    *reference*, and a reference that spent trumps to keep its ratio would be a second
    strategy with a second set of judgement calls in it.

    Throwing in whenever the limit allows means it hands cards over for free, because the
    defender may take and keep them. Also deliberate. The strategy's job is to be strong
    and obvious, so that when the search arms beat it, the reason is search quality rather
    than a hand-tuned heuristic.
    """
    if player != state.attacker:
        # Defending. A beat is preferred unconditionally: the plan's "never declines an
        # available beat" is what makes this a reference rather than a second strategy with
        # its own judgement calls in it.
        if state.table_attack_count > state.table_defense_count:
            target = state.table_attacks[state.table_defense_count]
            beater = _lowest_beater(state, player, target)
            if beater != _NO_CARD:
                action = _preferred_or_fallback(state, player, pack_action(BEAT, beater), BEAT)
                if action >= 0:
                    return np.int32(action)
        action = _preferred_or_fallback(state, player, pack_action(TAKE, 0), TAKE)
        return np.int32(action if action >= 0 else _any_legal(state, player))

    # Attacking. An empty table takes an attack; a fully answered table takes a throw-in while
    # the cap allows one, and otherwise the round stops.
    #
    # The cap is *not* recomputed here. `_preferred_throwin` only knows about rank matching, so
    # a card it prefers can still be refused by the cap -- and `_preferred_or_fallback` then
    # takes the first throw-in the rules actually allow, or falls back to stopping. That keeps
    # the limit in one place instead of two implementations that can disagree.
    if state.table_attack_count == 0:
        action = _preferred_or_fallback(
            state, player, pack_action(ATTACK, _preferred_attack_card(state, player)), ATTACK
        )
    else:
        throwin = _preferred_throwin(state, player)
        if throwin == _NO_CARD:
            action = -1
        else:
            action = _preferred_or_fallback(state, player, pack_action(THROW, throwin), THROW)
        if action < 0:
            action = _preferred_or_fallback(state, player, pack_action(STOP, 0), STOP)

    return np.int32(action if action >= 0 else _any_legal(state, player))
