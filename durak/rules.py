"""The normative rules engine.

This module is the single source of truth for how a game of Durak is played. Everything
else -- strategies, the runner, the reports -- takes the rules from here. An error in this
file does not crash the simulation; it produces a plausible, wrong distribution of win
rates. That is why each rule is a discrete named step, and why ``check_invariants`` is
invoked after every state transition rather than only at game end.

State representation
--------------------
``GameState`` is a Numba ``jitclass``: mutable fields with declared dtypes, constructible
from Python and callable from ``nopython`` code. That is the only representation tried
that satisfies all three requirements at once -- in-place mutation of array *and* scalar
fields, rebinding an array when the table outgrows its capacity, and passing across the
compiled boundary. A plain Python class cannot cross into ``nopython``; a ``NamedTuple``
can, but its ``_replace`` method cannot, so every scalar change would need a full
twenty-two-field reconstruction. Both alternatives were measured and rejected.

Fields carry explicit Numba types rather than inferred ones: a jitclass rejects any
attribute missing from its spec, and declaring them makes that a construction-time error
instead of a silent misread mid-game.

``check_invariants`` returns a *bitmask*, not a list of strings, for the same reason: a
Python list cannot cross the compiled boundary. ``INVARIANT_NAMES`` decodes it.

The rules, in the order they apply
----------------------------------
1. Shuffle, deal six cards each, turn the **bottom** draw-pile card face up as trump.
2. Redeal while a player holds all six trumps, capped at 100 attempts.
3. Choose the first attacker: lowest trump wins; sole trump holder wins; neither player
   holding a trump means a uniform random choice.
4. Per round: attack, defend (beat or take), optional throw-ins, resolve.
5. The win check fires the moment a card leaves a hand, **before** any refill, and after
   every beat as well as after every attack.
6. Refill only while a hand is below six, never discarding down; the round's attacker
   draws first.

Rule switches
--------------
``RuleTunables`` carries configurable rule flags as plain scalars rather than holding a
config object, because no Python object crosses ``nopython``. The one switch that changes
measured behaviour is ``throwin_defense_ranks`` (spec rule R7): whether a defense card's
rank on the table enables a throw-in. The default follows the literal rule text; the
variant is reported as a sensitivity because it materially changes throw-in rates.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
from numba import njit, types
from numba.experimental import jitclass

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
from durak.cards import DECK_SIZES, beats, build_deck, rank_of
from durak.rng import next_below, shuffle_inplace_with_state

# A legal hand can never exceed the deck size, and 52 < 64, so 64 bounds both variants.
HAND_CAPACITY = 64
HAND_TARGET = 6
EMPTY = -1

MAX_REDEALS = 100
MAX_ROUNDS = 10_000

# Sentinel for "choose the first attacker by the rule". Distinct from 0 and 1, which are seats --
# a sentinel of 0 would silently disable the rule for half of every rotation-balanced run.
USE_RULE = -1

# Throw-in ceiling. Counts attack cards only, never defense cards, and binds even when the
# defender holds more than six cards -- which happens only after a take.
THROWIN_CAP = 6

# Phase cut, fixed before any data existed so it cannot be tuned to flatter a result.
LATE_FRACTION = 0.15
EARLY_FRACTION = 0.50
PHASE_EARLY = 0
PHASE_MID = 1
PHASE_LATE = 2

# Invariant bit positions, decoded by INVARIANT_NAMES.
INV_CONSERVATION = 0
INV_NO_DUPLICATE = 1
INV_WINNER_EMPTY = 2
INV_LOSER_NOT_EMPTY = 3
INV_HAND_CAPACITY = 4
INV_DECK_LEN = 5
INV_ROUND_BOUND = 6
INV_TRUMP_IS_SUIT = 7
INV_TRUMP_CARD_IN_PLAY = 8

INVARIANT_NAMES: Tuple[str, ...] = (
    "card_conservation",
    "no_duplicate_card",
    "winner_has_empty_hand",
    "loser_hand_not_empty",
    "hand_within_capacity",
    "deck_len_within_array",
    "round_index_within_cap",
    # `trump` must be a suit index, not the trump card id. This bit exists because getting it
    # wrong is silent: every off-suit beat is simply refused, no test fails, and the whole
    # dataset shifts. It is the cheapest possible guard against that specific regression.
    "trump_is_a_suit_index",
    "trump_card_is_in_play",
)


# `rank_of`, `pack_action`, `action_kind`, `action_card`, and `beats` are imported from
# their own modules rather than reimplemented here. Those modules are decorated with
# `@njit`, so Numba can call them from `nopython` code. An earlier draft carried
# "compiled shims" duplicating that arithmetic inside this file; the duplication was
# removed because two implementations of the beat relation would be exactly the class of
# defect this project exists to prevent.


class RuleTunables(NamedTuple):
    """Rule switches as scalars, because no config object crosses ``nopython``."""

    throwin_defense_ranks: bool
    redeal_limit: int
    round_cap: int


DEFAULT_SETTINGS = RuleTunables(
    throwin_defense_ranks=True,
    redeal_limit=MAX_REDEALS,
    round_cap=MAX_ROUNDS,
)


_GAME_STATE_SPEC = [
    ("deck", types.int16[::1]),
    ("deck_len", types.int64),
    ("initial_deck", types.int64),
    # `trump` is the trump SUIT (0..3), not the trump card. `beats()` compares
    # `suit_a == trump`, so storing the card id here silently disabled the entire off-suit
    # rule: a trump could never beat a non-trump in any real game. The card itself is kept
    # separately in `trump_card`, because which card was turned over is worth recording and
    # because the card stays in the draw pile to be drawn last.
    ("trump", types.int64),
    ("trump_card", types.int64),
    ("hands", types.int16[:, :]),
    ("hand_sizes", types.int8[::1]),
    ("discard", types.int16[::1]),
    ("discard_len", types.int64),
    ("table_attacks", types.int16[::1]),
    ("table_attack_count", types.int64),
    ("table_defenses", types.int16[::1]),
    ("table_defense_count", types.int64),
    ("attacker", types.int64),
    ("round_idx", types.int64),
    ("winner", types.int64),
    ("redeals", types.int64),
    ("both_have_trumps", types.boolean),
    ("first_attacker_random_fallback", types.boolean),
    ("flag_redeals_exceeded", types.boolean),
    ("last_take_voluntary", types.boolean),
    ("throwin_defense_ranks", types.boolean),
    # Three separate streams, for the reasons in `durak.rng`. `rng_state` carries the
    # deal's continuation and is consumed by setup (choosing a first attacker when neither
    # player holds a trump); `rng_p0` and `rng_p1` belong to the players.
    #
    # The players' streams live in the state rather than in the caller because a policy
    # returns only its action: there is no channel through which an advanced stream could
    # travel back out. Keeping them per player is what stops one policy's draw count from
    # shifting the opponent's randomness, which is the property cross-strategy
    # comparability depends on.
    ("rng_state", types.uint64),
    ("rng_p0", types.uint64),
    ("rng_p1", types.uint64),
]


@jitclass(_GAME_STATE_SPEC)
class GameState:
    """One game's complete state.

    Mutable throughout: the rules are a state machine, and copying on every action would
    dominate the runtime.

    ``rng_state`` is threaded so a game can be replayed from a snapshot without
    re-deriving anything, and so the two execution modes can be compared on identical
    random draws.
    """

    def __init__(self, deck, hands, hand_sizes, initial_deck, trump, throwin_defense_ranks,
                 trump_card=-1):
        self.deck = deck
        self.deck_len = len(deck)
        self.initial_deck = initial_deck
        self.trump = trump
        self.trump_card = trump_card
        self.hands = hands
        self.hand_sizes = hand_sizes
        self.discard = np.empty(0, dtype=np.int16)
        self.discard_len = 0
        # Capacity 4 is enough for the common case; these grow on demand.
        self.table_attacks = np.empty(4, dtype=np.int16)
        self.table_attack_count = 0
        self.table_defenses = np.empty(4, dtype=np.int16)
        self.table_defense_count = 0
        self.attacker = 0
        self.round_idx = 0
        self.winner = EMPTY
        self.redeals = 0
        self.both_have_trumps = True
        self.first_attacker_random_fallback = False
        self.flag_redeals_exceeded = False
        self.last_take_voluntary = False
        self.throwin_defense_ranks = throwin_defense_ranks
        self.rng_state = np.uint64(1)
        self.rng_p0 = np.uint64(1)
        self.rng_p1 = np.uint64(1)


def settings_from_config(settings) -> RuleTunables:
    """Unpack a ``durak.config.Settings`` into the scalars the rules take."""
    return RuleTunables(
        throwin_defense_ranks=bool(settings.throwin_defense_ranks),
        redeal_limit=MAX_REDEALS,
        round_cap=MAX_ROUNDS,
    )


# --- construction helpers (Python side) --------------------------------------


def empty_state(deck_size: int) -> GameState:
    """A blank state for constructing a specific position in tests.

    Deliberately not a legal game: hands and deck are empty, so ``check_invariants`` will
    report violations until a test fills it in. Tests build positions through this rather
    than by playing forward from ``setup_game``, so each failure names the rule under test.
    """
    if deck_size not in DECK_SIZES:
        raise ValueError(f"deck_size must be one of {DECK_SIZES}, got {deck_size}")
    return GameState(
        np.empty(0, dtype=np.int16),
        np.full((2, HAND_CAPACITY), EMPTY, dtype=np.int16),
        np.zeros(2, dtype=np.int8),
        deck_size,
        0,
        True,
    )


def with_defense_ranks(state: GameState, cards) -> GameState:
    """Attach defense cards to a state, for constructing a throw-in position."""
    state.table_defenses = np.array(list(cards), dtype=np.int16)
    state.table_defense_count = len(state.table_defenses)
    return state


def with_throwin_setting(state: GameState, throwin_defense_ranks: bool) -> GameState:
    """Toggle the defense-rank throw-in rule on a state (spec rule R7)."""
    state.throwin_defense_ranks = throwin_defense_ranks
    return state


# --- setup -------------------------------------------------------------------


def setup_game(
    deck_size: int,
    streams,
    force_redeal: bool = False,
    redeal_limit: int = MAX_REDEALS,
    throwin_defense_ranks: bool = True,
) -> GameState:
    """Deal a game and choose the first attacker.

    A thin wrapper. The deal itself is :func:`deal_state`, which is compiled, and the split
    exists because the engine runs inside ``nopython`` and cannot construct a ``GameStreams``
    -- a Python class does not cross that boundary. Rather than write the deal twice, the
    compiled function *is* the deal and this unwraps the streams into the three ``uint64``
    states it wants.

    Draws only from the ``deal`` stream. Consuming a player's stream during the deal would make
    a run that differs in one strategy differ in the deal too, which would destroy the paired
    comparison the primary metric depends on.

    ``force_redeal`` forces the redeal condition on the first deal so the cap can be exercised
    without waiting for a 1-in-1.9M event. It forces the *condition*, not the result, so the
    state it returns is a genuinely re-dealt one. It is a test hook; the engine never sets it.
    """
    if deck_size not in DECK_SIZES:
        raise ValueError(f"deck_size must be one of {DECK_SIZES}, got {deck_size}")

    tunables = RuleTunables(
        throwin_defense_ranks=throwin_defense_ranks,
        redeal_limit=redeal_limit,
        round_cap=MAX_ROUNDS,
    )
    return deal_state(
        deck_size,
        np.uint64(streams.deal.snapshot()),
        np.uint64(streams.player0.snapshot()),
        np.uint64(streams.player1.snapshot()),
        tunables,
        force_redeal,
        USE_RULE,
    )


@njit(cache=True)
def deal_state(deck_size: int, deal_stream: np.uint64, stream_p0: np.uint64,
               stream_p1: np.uint64, tunables: RuleTunables, force_redeal: bool,
               first_attacker_override: int) -> GameState:
    """Shuffle, deal, apply the redeal rule, and choose the first attacker.

    The single implementation of the deal, called from the Python side via
    :func:`setup_game` and from the engine directly. It takes three stream *states* rather than
    a ``GameStreams`` bundle, because a Python object cannot cross into compiled code.

    The redeal rule: while a player holds all six trumps, reshuffle, up to
    ``tunables.redeal_limit`` attempts. On reaching the cap the deal falls through with
    ``flag_redeals_exceeded`` set rather than looping -- the cap exists to make the failure loud,
    not to prevent it.
    """
    if deck_size != 36 and deck_size != 52:
        raise ValueError("deck_size must be 36 or 52")

    redeals = 0
    exceeded = False
    force_first = force_redeal

    while True:
        deck = np.arange(deck_size, dtype=np.int16)
        deal_stream = shuffle_inplace_with_state(deck, deal_stream)

        hands = np.full((2, HAND_CAPACITY), EMPTY, dtype=np.int16)
        hand_sizes = np.array([HAND_TARGET, HAND_TARGET], dtype=np.int8)
        # Dealt one at a time, alternately, starting with player 0. The alternation is a
        # convention, not a statistical property: a uniform shuffle makes the assignment uniform
        # either way.
        for i in range(2 * HAND_TARGET):
            hands[i % 2, i // 2] = deck[i]

        # The trump card is the bottom of the draw pile: face up, but still *in* the pile, so it
        # is drawn last. It stays in the array rather than being lifted out, because it is still
        # a card in play -- removing it would break card conservation by exactly one and quietly
        # invalidate the strongest invariant in the project.
        draw_pile = deck[2 * HAND_TARGET:].copy()
        trump_card = draw_pile[draw_pile.shape[0] - 1]
        # The *suit* is what the rules operate on. Storing the card id here instead was the
        # defect recorded in the Stage 05 report: `beats` compares a suit index against this
        # field, and a card id never equals a suit, so every off-suit beat was refused.
        trump = trump_card % 4

        if not force_first and not _holds_all_six_trumps(hands, hand_sizes, trump):
            break

        force_first = False
        redeals += 1
        if redeals >= tunables.redeal_limit:
            exceeded = True
            break

    state = GameState(
        draw_pile, hands, hand_sizes, deck_size, trump, tunables.throwin_defense_ranks,
        trump_card,
    )
    state.redeals = redeals
    state.flag_redeals_exceeded = exceeded
    state.rng_state = deal_stream
    # The players' streams are seeded here, once, and are never touched by the deal. A strategy
    # that drew from the deal stream would make a run differing in one strategy differ in the
    # deal too, which would destroy the paired comparison.
    state.rng_p0 = stream_p0
    state.rng_p1 = stream_p1
    state = assign_first_attacker(state)

    if first_attacker_override == 0 or first_attacker_override == 1:
        state.attacker = first_attacker_override
    return state


@njit(cache=True)
def _holds_all_six_trumps(hands, hand_sizes, trump_suit: int) -> bool:
    for player in (0, 1):
        size = int(hand_sizes[player])
        count = 0
        for i in range(size):
            if int(hands[player, i]) % 4 == trump_suit % 4:
                count += 1
        if count == 6:
            return True
    return False


@njit(cache=True)
def assign_first_attacker(state: GameState) -> GameState:
    """Choose who attacks first, per the three documented cases.

    Neither-player-holds-a-trump is not an error: the trump card stays in the draw pile,
    so all six trumps land undealt in roughly 2.8% of 36-card deals. The attacker is then
    chosen uniformly at random and the fallback recorded, because that field is the only
    way a later analysis can distinguish this distribution from a dealt one.
    """
    trump_suit = state.trump
    first = _lowest_trump(state.hands, state.hand_sizes, 0, trump_suit)
    second = _lowest_trump(state.hands, state.hand_sizes, 1, trump_suit)

    if first != EMPTY and second != EMPTY:
        state.attacker = 0 if rank_of(first) < rank_of(second) else 1
        state.both_have_trumps = True
        return state

    if first != EMPTY:
        state.attacker = 0
        state.both_have_trumps = False
        return state

    if second != EMPTY:
        state.attacker = 1
        state.both_have_trumps = False
        return state

    choice, rng_state = next_below(state.rng_state, 2)
    state.attacker = int(choice)
    state.both_have_trumps = False
    state.first_attacker_random_fallback = True
    state.rng_state = np.uint64(rng_state)
    return state


@njit(cache=True)
def _lowest_trump(hands, hand_sizes, player: int, trump_suit: int) -> int:
    size = int(hand_sizes[player])
    best = EMPTY
    for i in range(size):
        card = int(hands[player, i])
        if card % 4 == trump_suit:
            if best == EMPTY or rank_of(card) < rank_of(best):
                best = card
    return best


# --- hand access -------------------------------------------------------------


@njit(cache=True)
def remove_from_hand(state: GameState, player: int, card: int) -> GameState:
    """Take a card out of a player's hand, closing the gap.

    Keeping every live card in slots ``0..size-1`` is what lets refill and the legality
    checks index by position instead of scanning for gaps.
    """
    size = int(state.hand_sizes[player])
    for i in range(size):
        if state.hands[player, i] == card:
            for j in range(i, size - 1):
                state.hands[player, j] = state.hands[player, j + 1]
            state.hands[player, size - 1] = EMPTY
            state.hand_sizes[player] = np.int8(size - 1)
            return state
    raise ValueError(f"card {card} is not in player {player}'s hand")


@njit(cache=True)
def add_to_hand(state: GameState, player: int, card: int) -> GameState:
    """Append a card to a player's hand."""
    size = int(state.hand_sizes[player])
    if size >= HAND_CAPACITY:
        raise ValueError(f"player {player}'s hand is at capacity {HAND_CAPACITY}")
    state.hands[player, size] = card
    state.hand_sizes[player] = np.int8(size + 1)
    return state


@njit(cache=True)
def hand_contains(state: GameState, player: int, card: int) -> bool:
    size = int(state.hand_sizes[player])
    for i in range(size):
        if state.hands[player, i] == card:
            return True
    return False


@njit(cache=True)
def trumps_in_hand(state: GameState, player: int) -> int:
    """How many trumps a player holds. Used by strategies and by deal statistics."""
    size = int(state.hand_sizes[player])
    count = 0
    for i in range(size):
        if state.hands[player, i] % 4 == state.trump:
            count += 1
    return count


@njit(cache=True)
def lowest_trump(state: GameState, player: int) -> int:
    """The lowest-ranked trump in a player's hand, or ``EMPTY`` if they hold none."""
    return _lowest_trump(state.hands, state.hand_sizes, player, state.trump)


# --- whose turn --------------------------------------------------------------


@njit(cache=True)
def acting_player(state: GameState) -> int:
    """The player who may act right now.

    Derived rather than stored: an attack is answered by the defender, and control returns
    to the attacker only once every card on the table has been beaten. Deriving it from
    the table counts keeps the rule in one place -- a stored turn flag would have to be
    updated in step with every transition, and one missed update would let both players
    act for the same card.

    Note this is *not* always the attacker: after an attack the defender responds.
    """
    if state.table_attack_count == state.table_defense_count:
        return state.attacker
    return 1 - state.attacker


# --- legality ----------------------------------------------------------------


@njit(cache=True)
def legal_actions(state: GameState, player: int) -> np.ndarray:
    """Every action available to ``player``, as a packed ``int32`` array.

    Every action listed here must be applicable by :func:`apply_action`. The search relies on that:
    it enumerates the opponent's replies from this function and then applies each one, so any action
    offered here but rejected there makes the search fail rather than merely misbehave. An empty array
    means the player has no legal move, which happens for a defender facing an empty table -- there is
    nothing to beat and nothing to take.
    """
    out = np.empty(HAND_CAPACITY + 2, dtype=np.int32)
    count = 0
    size = int(state.hand_sizes[player])

    if state.winner != EMPTY:
        return out[:0]

    if player == state.attacker:
        if state.table_attack_count == 0:
            for i in range(size):
                out[count] = pack_action(ATTACK, int(state.hands[player, i]))
                count += 1
        else:
            for i in range(size):
                card = int(state.hands[player, i])
                if _throwin_legal(state, card):
                    out[count] = pack_action(THROW, card)
                    count += 1
            # Stopping is only offered once every card on the table has been answered.
            # Offering it while a card is unbeaten would let a round end without the
            # defender responding, which the rules do not allow.
            if state.table_attack_count == state.table_defense_count:
                out[count] = pack_action(STOP, 0)
                count += 1
    else:
        # Taking is only a thing you can do when there is something on the table. Offering it with an
        # empty table contradicts `apply_action`, which rejects it -- and the search enumerates every
        # action `legal_actions` returns and applies each one, so the two have to agree exactly.
        if state.table_attack_count > 0:
            if state.table_attack_count > state.table_defense_count:
                target = int(state.table_attacks[state.table_defense_count])
                for i in range(size):
                    card = int(state.hands[player, i])
                    if beats(card, target, state.trump):
                        out[count] = pack_action(BEAT, card)
                        count += 1
            out[count] = pack_action(TAKE, 0)
            count += 1

    return out[:count]


@njit(cache=True)
def _throwin_legal(state: GameState, card: int) -> bool:
    """Whether ``card`` may be added to the table.

    Two conditions, both easy to get wrong. The cap counts *attack* cards only, against
    ``min(defender's current hand, 6)`` -- not the defender's hand alone, because a
    post-take defender can hold more than six and a naive reading would let the table grow
    to nine attacks, changing every throw-in statistic in the dataset. And the rank match
    must consider ranks already on the table, which grows as the table does.
    """
    # The previous card is still unbeaten; throwing in again before it is answered is not
    # allowed.
    if state.table_defense_count < state.table_attack_count:
        return False

    limit = int(state.hand_sizes[1 - state.attacker])
    if limit > THROWIN_CAP:
        limit = THROWIN_CAP
    if state.table_attack_count + 1 > limit:
        return False

    card_rank = rank_of(card)
    for i in range(state.table_attack_count):
        if rank_of(int(state.table_attacks[i])) == card_rank:
            return True
    if state.throwin_defense_ranks:
        for i in range(state.table_defense_count):
            if rank_of(int(state.table_defenses[i])) == card_rank:
                return True
    return False


# --- applying actions --------------------------------------------------------


@njit(cache=True)
def apply_action(state: GameState, action: int, tunables: RuleTunables) -> GameState:
    """Apply one action, returning the resulting state.

    The win check lives here and nowhere else, so there is exactly one place where it can
    be wrong. It fires after every card leaving a hand -- including a defender's beat --
    and always before any refill.
    """
    kind = action_kind(action)

    if kind == ATTACK:
        return _do_attack(state, action_card(action))
    if kind == THROW:
        return _do_throw(state, action_card(action))
    if kind == STOP:
        return _do_stop(state)
    if kind == BEAT:
        return _do_beat(state, action_card(action))
    return _do_take(state)


@njit(cache=True)
def _do_attack(state: GameState, card: int) -> GameState:
    if not hand_contains(state, state.attacker, card):
        raise ValueError(f"attack with illegal card {card}")
    state = remove_from_hand(state, state.attacker, card)
    state = _append_attack(state, card)

    # The win check precedes everything else: an empty hand ends the game before the
    # defender can respond, and before a refill could hand back six cards.
    if state.hand_sizes[state.attacker] == 0:
        state.winner = state.attacker
    return state


@njit(cache=True)
def _do_throw(state: GameState, card: int) -> GameState:
    if not hand_contains(state, state.attacker, card):
        raise ValueError(f"throw with illegal card {card}")
    state = remove_from_hand(state, state.attacker, card)
    state = _append_attack(state, card)

    if state.hand_sizes[state.attacker] == 0:
        state.winner = state.attacker
    return state


@njit(cache=True)
def _do_beat(state: GameState, card: int) -> GameState:
    defender = 1 - state.attacker
    if not hand_contains(state, defender, card):
        raise ValueError(f"beat with illegal card {card}")
    state = remove_from_hand(state, defender, card)
    state = _append_defense(state, card)

    # A defender who beats with their last card ends the game at that moment. Without
    # this check the throw-in phase would continue from a hand that no longer exists.
    if state.hand_sizes[defender] == 0:
        state.winner = defender
    return state


@njit(cache=True)
def _do_take(state: GameState) -> GameState:
    defender = 1 - state.attacker
    if state.table_attack_count == 0:
        raise ValueError("take is not a legal action with an empty table")

    # Voluntary means the defender had a beating card available and declined. The
    # distinction matters: a forced take says nothing about the defender's skill.
    voluntary = _any_legal_beat(state, defender)

    for i in range(state.table_attack_count):
        state = add_to_hand(state, defender, int(state.table_attacks[i]))
    for i in range(state.table_defense_count):
        state = add_to_hand(state, defender, int(state.table_defenses[i]))
    state.table_attack_count = 0
    state.table_defense_count = 0
    state.last_take_voluntary = voluntary

    # After a take the roles do not swap: the defender, who now holds the cards, attacks.
    return _begin_next_round(state, defender)


@njit(cache=True)
def _any_legal_beat(state: GameState, defender: int) -> bool:
    if state.table_attack_count <= state.table_defense_count:
        return False
    target = int(state.table_attacks[state.table_defense_count])
    size = int(state.hand_sizes[defender])
    for i in range(size):
        if beats(int(state.hands[defender, i]), target, state.trump):
            return True
    return False


@njit(cache=True)
def _do_stop(state: GameState) -> GameState:
    """The attacker stopped; every attack was beaten, so the table is discarded."""
    if state.table_attack_count != state.table_defense_count:
        raise ValueError("cannot stop while a card on the table is unbeaten")

    for i in range(state.table_attack_count):
        state = _append_discard(state, int(state.table_attacks[i]))
    for i in range(state.table_defense_count):
        state = _append_discard(state, int(state.table_defenses[i]))
    state.table_attack_count = 0
    state.table_defense_count = 0

    # Successful defense: the roles swap.
    return _begin_next_round(state, 1 - state.attacker)


@njit(cache=True)
def _begin_next_round(state: GameState, next_attacker: int) -> GameState:
    """Refill in the round's draw order, then hand the turn to ``next_attacker``.

    The draw order follows the round that just *ended*, not the next attacker. On a
    successful defense the roles swap, so drawing "the next attacker first" would invert
    the spec's rule and silently change who sees fresh cards.
    """
    previous_attacker = state.attacker
    state.round_idx = state.round_idx + 1
    state = _refill_for(state, previous_attacker)
    state.attacker = next_attacker
    return state


# --- refill ------------------------------------------------------------------


@njit(cache=True)
def refill(state: GameState, tunables: RuleTunables) -> GameState:
    """Top both hands up to six, attacker first.

    "Never discard down" is the load-bearing rule: a player who took twelve cards draws
    nothing. Reading "until they have six cards" literally would shrink a post-take hand
    and change the whole endgame.
    """
    return _refill_for(state, state.attacker)


@njit(cache=True)
def _refill_for(state: GameState, first_drawer: int) -> GameState:
    # A finished game is never refilled. Without this, a defender who emptied their hand
    # would be handed six fresh cards immediately after winning.
    if state.winner != EMPTY:
        return state
    state = _refill_one(state, first_drawer)
    return _refill_one(state, 1 - first_drawer)


@njit(cache=True)
def _refill_one(state: GameState, player: int) -> GameState:
    while state.hand_sizes[player] < HAND_TARGET and state.deck_len > 0:
        card = int(state.deck[0])
        for i in range(state.deck_len - 1):
            state.deck[i] = state.deck[i + 1]
        state.deck_len = state.deck_len - 1
        state = add_to_hand(state, player, card)
    return state


@njit(cache=True)
def round_cap_reached(state: GameState, tunables: RuleTunables) -> bool:
    """Whether the round cap has been hit.

    The cap condition in one place, so the two things that react to it cannot disagree: the
    engine asks this and records a flagged anomaly, while :func:`begin_round` asks it and raises.
    Numba's ``nopython`` mode has no usable exception handling around a mutated state, so a
    predicate plus a raise is the only way to keep one condition and two responses.
    """
    return state.round_idx >= tunables.round_cap


@njit(cache=True)
def begin_round(state: GameState, tunables: RuleTunables) -> GameState:
    """Guard the round cap before a round begins.

    A loud failure for callers that have no anomaly channel. The engine does not use this to stop
    a game -- it uses :func:`round_cap_reached` and records the game as a flagged anomaly instead
    of aborting a million-game run.
    """
    if round_cap_reached(state, tunables):
        raise RuntimeError(
            f"round cap {tunables.round_cap} exceeded at round {state.round_idx}"
        )
    return state


# --- phases ------------------------------------------------------------------


@njit(cache=True)
def phase_of(state: GameState, initial_deck: int) -> int:
    """Game phase from the draw pile, on the pre-registered cut."""
    ratio = state.deck_len / initial_deck
    if ratio <= LATE_FRACTION:
        return PHASE_LATE
    if ratio > EARLY_FRACTION:
        return PHASE_EARLY
    return PHASE_MID


# --- invariants --------------------------------------------------------------


@njit(cache=True)
def check_invariants(state: GameState) -> int:
    """Bitmask of violated invariants; ``0`` means clean.

    Checked after every transition. Conservation and duplication are the two that earn
    the dataset its credibility: a card in two hands leaves the game perfectly playable
    while corrupting every metric computed from it.
    """
    mask = 0

    accounted = (
        state.deck_len
        + state.hand_sizes[0]
        + state.hand_sizes[1]
        + state.discard_len
        + state.table_attack_count
        + state.table_defense_count
    )
    if accounted != state.initial_deck:
        mask |= 1 << INV_CONSERVATION

    if _has_duplicate(state):
        mask |= 1 << INV_NO_DUPLICATE

    if state.winner != EMPTY:
        if state.hand_sizes[state.winner] != 0:
            mask |= 1 << INV_WINNER_EMPTY
        if state.hand_sizes[1 - state.winner] <= 0:
            mask |= 1 << INV_LOSER_NOT_EMPTY

    for player in (0, 1):
        if state.hand_sizes[player] > HAND_CAPACITY:
            mask |= 1 << INV_HAND_CAPACITY

    if state.deck_len > state.deck.shape[0]:
        mask |= 1 << INV_DECK_LEN

    if state.round_idx > MAX_ROUNDS:
        mask |= 1 << INV_ROUND_BOUND

    if state.trump < 0 or state.trump > 3:
        mask |= 1 << INV_TRUMP_IS_SUIT

    if state.trump_card >= 0 and state.trump_card != state.trump:
        # The trump card must belong to the trump suit. Cheap, and it catches the two fields
        # being swapped or one of them being derived from the other by mistake.
        if state.trump_card % 4 != state.trump:
            mask |= 1 << INV_TRUMP_CARD_IN_PLAY

    return mask


@njit(cache=True)
def _has_duplicate(state: GameState) -> bool:
    """Whether any card sits in two places at once.

    A bitmask rather than a boolean array, and the reason is the spec's own requirement: this
    runs after *every* state transition, which makes it part of the tournament's hot path rather
    than a debug aid. Profiling put 200,000 transitions at 104 seconds -- nearly 260 microseconds
    each -- and the `np.zeros` allocation on entry was the bulk of it. A 52-card deck fits in one
    64-bit mask, so the whole scratch space becomes a single register and there is nothing to
    allocate per call.

    A set would be faster still in theory and is not available at all: a Python object cannot
    cross into ``nopython`` mode.
    """
    seen = np.int64(0)

    for i in range(state.deck_len):
        card = int(state.deck[i])
        bit = np.int64(1) << card
        if seen & bit:
            return True
        seen |= bit
    for player in (0, 1):
        for i in range(state.hand_sizes[player]):
            card = int(state.hands[player, i])
            bit = np.int64(1) << card
            if seen & bit:
                return True
            seen |= bit
    for i in range(state.discard_len):
        card = int(state.discard[i])
        bit = np.int64(1) << card
        if seen & bit:
            return True
        seen |= bit
    for i in range(state.table_attack_count):
        card = int(state.table_attacks[i])
        bit = np.int64(1) << card
        if seen & bit:
            return True
        seen |= bit
    for i in range(state.table_defense_count):
        card = int(state.table_defenses[i])
        bit = np.int64(1) << card
        if seen & bit:
            return True
        seen |= bit
    return False


# --- array helpers -----------------------------------------------------------


@njit(cache=True)
def _append_attack(state: GameState, card: int) -> GameState:
    if state.table_attack_count >= state.table_attacks.shape[0]:
        bigger = np.empty(max(4, state.table_attacks.shape[0] * 2), dtype=np.int16)
        for i in range(state.table_attacks.shape[0]):
            bigger[i] = state.table_attacks[i]
        state.table_attacks = bigger
    state.table_attacks[state.table_attack_count] = card
    state.table_attack_count = state.table_attack_count + 1
    return state


@njit(cache=True)
def _append_defense(state: GameState, card: int) -> GameState:
    if state.table_defense_count >= state.table_defenses.shape[0]:
        bigger = np.empty(max(4, state.table_defenses.shape[0] * 2), dtype=np.int16)
        for i in range(state.table_defenses.shape[0]):
            bigger[i] = state.table_defenses[i]
        state.table_defenses = bigger
    state.table_defenses[state.table_defense_count] = card
    state.table_defense_count = state.table_defense_count + 1
    return state


@njit(cache=True)
def _append_discard(state: GameState, card: int) -> GameState:
    if state.discard_len >= state.discard.shape[0]:
        bigger = np.empty(max(8, state.discard.shape[0] * 2), dtype=np.int16)
        for i in range(state.discard.shape[0]):
            bigger[i] = state.discard[i]
        state.discard = bigger
    state.discard[state.discard_len] = card
    state.discard_len = state.discard_len + 1
    return state


# --- helpers used by the engine and the tests -------------------------------


@njit(cache=True)
def draw_action(actions: np.ndarray, rng_state) -> tuple:
    """Pick uniformly among ``actions``, returning ``(action, new_state_as_python_int)``.

    The invariant tests and the benchmark need a policy that explores the legal action
    space. A uniformly random policy reaches rare positions; a greedy one never would, and
    the "every transition conserves cards" claim would rest on a biased sample.

    Callers in Python must re-wrap the returned state as ``np.uint64`` before passing it
    back: Numba types an argument by its Python-level value, and a plain ``int`` above
    2**63-1 does not fit the ``int64`` it would infer. Inside compiled code the state
    stays ``uint64`` throughout and no wrapping is needed.
    """
    if len(actions) == 0:
        raise ValueError("no legal actions available")
    index, next_state = next_below(np.uint64(rng_state), len(actions))
    return np.int32(actions[int(index)]), next_state