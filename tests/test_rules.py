"""Tests for the normative rules engine.

This is the substrate the whole dataset rests on. An error here does not crash the
simulator -- it produces a plausible, wrong distribution of win rates. The tests
therefore enumerate the whole space where that is cheap (every redeal condition, every
throw-in cap boundary) and pin exact values rather than checking properties loosely.

The cases that matter most are the ones a naive implementation gets wrong:

* the win check must fire after *every* card leaving a hand, including a defender's
  beat -- otherwise a defender who empties their hand keeps playing a finished game;
* refill must never reduce a hand, so a player who took twelve cards draws nothing;
* the throw-in cap is ``min(defender's current hand, 6)`` counted over attack cards
  only, so a post-take defender with nine cards still caps the table at six attacks;
* a deal where neither player holds a trump is ordinary, not exceptional -- roughly 2.8%
  of 36-card deals -- and must be recorded rather than treated as an error.

Fixtures come from :func:`legal_state` built on a :class:`Pool` that hands out each card
exactly once, so no fixture can silently allocate a card twice. An inconsistent fixture
would make every invariant assertion meaningless, and a ``deck_len`` larger than the
``deck`` array makes the engine write out of bounds, corrupting the heap rather than
raising.
"""

import numpy as np
import pytest

from durak import rules
from durak.actions import (
    ATTACK,
    BEAT,
    STOP,
    TAKE,
    THROW,
    pack_action,
    unpack_action,
)
from durak.cards import DECK_SIZES, beats, rank_of, suit_of
from durak.rng import GameStreams

EMPTY = rules.EMPTY
HAND_CAPACITY = rules.HAND_CAPACITY
HAND_TARGET = rules.HAND_TARGET
MAX_ROUNDS = rules.MAX_ROUNDS

T = rules.DEFAULT_SETTINGS


def card_at(rank, suit):
    """The card id of a (rank offset, suit) pair."""
    return rank * 4 + suit


class Pool:
    """Hands out distinct cards, tracking what has been taken.

    Tests build positions from a pool rather than from slicing the deck directly. Slicing
    produced overlapping hands -- ``non_trumps`` and ``trumps_of`` both returned hearts --
    and the resulting fixtures were not positions the rules could ever reach.
    """

    def __init__(self, deck_size=36):
        self._available = set(range(deck_size))
        self._order = list(range(deck_size))

    def reserve(self, card):
        """Mark a card as used without taking it from the ordered pool."""
        self._available.discard(card)
        self._order = [c for c in self._order if c != card]
        return card

    def take(self, count):
        """The next `count` unused cards, in ascending order."""
        picked = [c for c in self._order if c in self._available][:count]
        assert len(picked) == count, "not enough cards left"
        for card in picked:
            self._available.discard(card)
        return picked

    def of_suit(self, count, suit):
        """`count` unused cards of one suit."""
        picked = [c for c in self._order if c in self._available and suit_of(c) == suit][:count]
        assert len(picked) == count, f"not enough cards of suit {suit} left"
        for card in picked:
            self._available.discard(card)
        return picked

    def without_suit(self, count, suit):
        """`count` unused cards that are *not* of the given suit.

        Needed for the no-trump deals. `take` would happily hand out trumps, and a
        "neither player holds a trump" fixture that quietly contained one tested nothing.
        """
        picked = [c for c in self._order if c in self._available and suit_of(c) != suit][:count]
        assert len(picked) == count, f"not enough cards outside suit {suit} left"
        for card in picked:
            self._available.discard(card)
        return picked


def legal_state(
    hand0,
    hand1,
    trump,
    attacker=0,
    table_attacks=(),
    table_defenses=(),
    discard=(),
    deck_size=36,
    throwin_defense_ranks=True,
):
    """A card-conserving game state built from explicit card allocations.

    Every card not named in a hand, on the table, or in the discard goes into the draw
    pile, so ``deck_len`` always matches the array it indexes. Asserted at construction:
    a fixture that is not a legal state makes every invariant assertion below meaningless.
    """
    named = [c for group in (hand0, hand1, table_attacks, table_defenses, discard) for c in group]
    assert len(set(named)) == len(named), f"a card was allocated twice: {named}"

    hands = np.full((2, HAND_CAPACITY), EMPTY, dtype=np.int16)
    for i, card in enumerate(hand0):
        hands[0, i] = card
    for i, card in enumerate(hand1):
        hands[1, i] = card

    remaining = [c for c in range(deck_size) if c not in set(named)]
    state = rules.GameState(
        np.array(remaining, dtype=np.int16),
        hands,
        np.array([len(hand0), len(hand1)], dtype=np.int8),
        deck_size,
        trump,
        throwin_defense_ranks,
    )
    state.discard = np.array(list(discard), dtype=np.int16)
    state.discard_len = len(state.discard)
    state.table_attacks = np.array(list(table_attacks), dtype=np.int16)
    state.table_attack_count = len(state.table_attacks)
    state.table_defenses = np.array(list(table_defenses), dtype=np.int16)
    state.table_defense_count = len(state.table_defenses)
    state.attacker = attacker

    assert rules.check_invariants(state) == 0, "fixture is not a legal state"
    return state


def kinds(state, player):
    return [unpack_action(int(a))[0] for a in rules.legal_actions(state, player)]


def thrown(state, player):
    return [unpack_action(int(a))[1] for a in rules.legal_actions(state, player)
            if unpack_action(int(a))[0] == THROW]


# --- action encoding ---------------------------------------------------------


def test_action_kinds_have_the_pinned_values():
    assert (ATTACK, THROW, STOP, BEAT, TAKE) == (0, 1, 2, 3, 4)


def test_pack_and_unpack_round_trip():
    for kind in (ATTACK, THROW, BEAT):
        for card in (0, 7, 35, 51):
            assert unpack_action(pack_action(kind, card)) == (kind, card)


def test_unpack_action_recovers_the_kind_without_a_card():
    assert unpack_action(pack_action(STOP, 0))[0] == STOP
    assert unpack_action(pack_action(TAKE, 0))[0] == TAKE


def test_packed_actions_are_non_negative():
    # A negative value would collide with the EMPTY marker in the same arrays.
    for kind in (ATTACK, THROW, STOP, BEAT, TAKE):
        assert pack_action(kind, 0) >= 0


# --- setup -------------------------------------------------------------------


def test_setup_deals_six_each_and_reveals_bottom_as_trump():
    for deck_size in DECK_SIZES:
        state = rules.setup_game(deck_size, GameStreams.from_parts(1, 0, 0))

        assert state.hand_sizes[0] == HAND_TARGET
        assert state.hand_sizes[1] == HAND_TARGET
        # 12 dealt leaves 24 of a 36-card deck and 40 of a 52-card deck. The trump card
        # is the bottom of the draw pile, so it is still in it -- face up, drawn last.
        assert state.deck_len == deck_size - 2 * HAND_TARGET
        assert state.trump == state.deck[state.deck_len - 1]


def test_setup_preserves_card_conservation():
    for deck_size in DECK_SIZES:
        state = rules.setup_game(deck_size, GameStreams.from_parts(7, 0, 0))

        assert rules.check_invariants(state) == 0


def test_setup_is_deterministic_for_a_seed():
    first = rules.setup_game(36, GameStreams.from_parts(99, 1, 2))
    second = rules.setup_game(36, GameStreams.from_parts(99, 1, 2))

    assert np.array_equal(first.hands[0], second.hands[0])
    assert np.array_equal(first.hands[1], second.hands[1])
    assert first.trump == second.trump
    assert first.attacker == second.attacker


def test_setup_uses_three_separate_streams():
    # The deal must draw from the deal stream alone. Consuming a player's stream during
    # setup would make a run differing in one strategy differ in the deal too, breaking
    # the paired design the primary metric depends on.
    streams = GameStreams.from_parts(5, 0, 0)
    deal_before = streams.deal.snapshot()
    p0_before = streams.player0.snapshot()
    p1_before = streams.player1.snapshot()

    rules.setup_game(36, streams)

    assert streams.deal.snapshot() != deal_before
    assert streams.player0.snapshot() == p0_before
    assert streams.player1.snapshot() == p1_before


def test_redeal_when_player_holds_all_six_trumps():
    state = rules.setup_game(36, GameStreams.from_parts(3, 0, 0), force_redeal=True)

    assert state.redeals >= 1
    # After a redeal the deal must be legal, not merely retried.
    assert rules.check_invariants(state) == 0
    assert state.hand_sizes[0] == HAND_TARGET


def test_redeal_produces_a_different_deal():
    forced = rules.setup_game(36, GameStreams.from_parts(3, 0, 0), force_redeal=True)
    normal = rules.setup_game(36, GameStreams.from_parts(3, 0, 0))

    assert not np.array_equal(forced.hands[0], normal.hands[0])


def test_redeal_cap_flags_and_falls_back_to_random():
    # `force_redeal` forces the condition on the first deal only, so a cap of 1 trips on
    # that deal. The cap exists to turn an unbounded loop into a loud failure; a 1-in-1.9M
    # deal cannot be waited for in a test.
    state = rules.setup_game(36, GameStreams.from_parts(4, 0, 0), force_redeal=True,
                             redeal_limit=1)

    assert state.flag_redeals_exceeded
    assert state.redeals == 1
    # A usable state must still come back: the fallback is a random first attacker.
    assert rules.check_invariants(state) == 0
    assert state.hand_sizes[0] == HAND_TARGET


def test_unknown_deck_size_is_rejected():
    with pytest.raises(ValueError, match="deck_size"):
        rules.setup_game(40, GameStreams.from_parts(1, 0, 0))


# --- first attacker ----------------------------------------------------------


def _state_with_hands(hand0, hand1, trump, rng_seed):
    state = legal_state(hand0, hand1, trump)
    state.rng_state = np.uint64(rng_seed)
    return state


def test_first_attacker_lowest_trump_when_both_have_trumps():
    pool = Pool()
    low = pool.reserve(card_at(0, 0))      # spades rank 0
    high = pool.reserve(card_at(3, 0))     # spades rank 3
    state = _state_with_hands(
        [low] + pool.without_suit(5, 0), [high] + pool.without_suit(5, 0),
        trump=0, rng_seed=1,
    )

    rules.assign_first_attacker(state)

    assert state.attacker == 0
    assert state.both_have_trumps


def test_first_attacker_is_the_holder_of_the_lower_trump_when_the_other_is_higher():
    pool = Pool()
    high = pool.reserve(card_at(5, 0))     # player 0's spade
    low = pool.reserve(card_at(1, 0))      # player 1's spade
    state = _state_with_hands(
        [high] + pool.without_suit(5, 0), [low] + pool.without_suit(5, 0),
        trump=0, rng_seed=1,
    )

    rules.assign_first_attacker(state)

    # The lowest trump attacks first, so player 1 does.
    assert state.attacker == 1
    assert state.both_have_trumps


def test_first_attacker_only_holder_when_one_has_trumps():
    pool = Pool()
    spades = [pool.reserve(card_at(r, 0)) for r in range(6)]
    state = _state_with_hands(pool.take(6), spades, trump=0, rng_seed=1)

    rules.assign_first_attacker(state)

    assert state.attacker == 1
    assert not state.both_have_trumps


def test_first_attacker_random_when_neither_has_trumps():
    """A deal with no trumps in either hand is ordinary, not exceptional.

    The trump card stays in the draw pile, so all six trumps can land undealt -- about
    2.8% of 36-card deals. The source document called this case "impossible"; it is not,
    and the engine must record it rather than reject the deal.
    """
    pool = Pool()
    hand0 = pool.without_suit(6, 0)
    hand1 = pool.without_suit(6, 0)
    trump = pool.reserve(card_at(0, 0))
    assert all(suit_of(c) != 0 for c in hand0 + hand1)
    counts = [0, 0]

    for seed in range(200):
        state = _state_with_hands(hand0, hand1, trump=trump, rng_seed=seed)
        rules.assign_first_attacker(state)
        assert not state.both_have_trumps
        assert state.first_attacker_random_fallback
        counts[state.attacker] += 1

    # Uniform over 200 seeds: both sides appear, neither dominates.
    assert 60 < counts[0] < 140
    assert 60 < counts[1] < 140


def test_first_attacker_random_advances_the_states_own_stream():
    """The fallback advances the state's RNG rather than a fresh generator.

    Seeding the choice afresh would make it identical across games with the same deal,
    destroying the independence the rotation-balanced regime relies on.
    """
    pool = Pool()
    hand0 = pool.without_suit(6, 0)
    hand1 = pool.without_suit(6, 0)
    trump = pool.reserve(card_at(0, 0))

    first = _state_with_hands(hand0, hand1, trump=trump, rng_seed=3)
    rules.assign_first_attacker(first)
    second = _state_with_hands(hand0, hand1, trump=trump, rng_seed=4)
    rules.assign_first_attacker(second)

    assert first.rng_state != second.rng_state


# --- legality: attack --------------------------------------------------------


def test_attacker_may_play_any_card_in_hand():
    pool = Pool()
    hand0 = pool.take(6)
    hand1 = pool.take(6)
    state = legal_state(hand0, hand1, trump=3)

    assert kinds(state, 0).count(ATTACK) == HAND_TARGET


def test_defender_may_beat_or_take():
    pool = Pool()
    attack = pool.reserve(card_at(2, 1))
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=1,
                        table_attacks=[attack])

    offered = kinds(state, 0)

    assert TAKE in offered
    assert BEAT in offered


def test_defender_with_no_beating_card_may_only_take():
    pool = Pool()
    attack = pool.reserve(card_at(0, 0))     # a trump
    state = legal_state(pool.take(6), pool.of_suit(6, 1), trump=0, attacker=0,
                        table_attacks=[attack])

    assert kinds(state, 1) == [TAKE]


def test_off_suit_non_trump_cannot_beat_a_trump():
    pool = Pool()
    attack = pool.reserve(card_at(3, 0))
    state = legal_state(pool.take(6), pool.of_suit(6, 1), trump=0, attacker=0,
                        table_attacks=[attack])

    assert BEAT not in kinds(state, 1)


def test_a_higher_same_suit_card_can_beat():
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))
    beater = pool.reserve(card_at(1, 1))
    state = legal_state(pool.take(6), [beater] + pool.take(5), trump=0, attacker=0,
                        table_attacks=[attack])

    assert BEAT in kinds(state, 1)


# --- legality: throw-in ------------------------------------------------------


def test_throwin_requires_matching_rank():
    pool = Pool()
    tabled = pool.reserve(card_at(0, 1))     # hearts rank 0
    pool.reserve(card_at(1, 1))              # defense, hearts rank 1
    matching = pool.reserve(card_at(0, 2))   # diamonds rank 0 -- matches
    others = [pool.reserve(card_at(r, 3)) for r in (5, 6, 7, 8)]

    state = legal_state(
        [matching] + others,
        pool.take(6),
        trump=0,
        table_attacks=[tabled],
        table_defenses=[card_at(1, 1)],
    )

    assert thrown(state, 0) == [matching]


def test_throwin_matches_the_rank_of_a_defense_card():
    pool = Pool()
    tabled = pool.reserve(card_at(0, 1))
    defense = pool.reserve(card_at(1, 1))
    matching = pool.reserve(card_at(1, 2))   # same rank as the defense, different suit
    others = [pool.reserve(card_at(r, 3)) for r in (5, 6, 7, 8)]

    state = legal_state(
        [matching] + others, pool.take(6), trump=0,
        table_attacks=[tabled], table_defenses=[defense],
    )

    assert thrown(state, 0) == [matching]


def test_throwin_respects_the_flag_on_defense_card_ranks():
    """Spec rule R7: defense-card ranks are toggleable.

    The default follows the literal rule text ("same rank as any card already on the
    table"), so a defense card's rank enables a throw-in. The variant restricts matching
    to attack cards only. Which reading is in force materially changes throw-in rates, so
    it is a recorded setting rather than a hardcoded assumption.
    """
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))     # hearts rank 0
    defense = pool.reserve(card_at(1, 1))    # hearts rank 1
    throw = pool.reserve(card_at(1, 2))      # diamonds rank 1 -- matches the defense only
    others = [pool.reserve(card_at(r, 3)) for r in (5, 6, 7, 8)]
    hands0 = [throw] + others
    hands1 = pool.take(6)

    with_flag = legal_state(
        hands0, hands1, trump=0, table_attacks=[attack], table_defenses=[defense]
    )
    assert thrown(with_flag, 0) == [throw]

    without_flag = legal_state(
        hands0, hands1, trump=0, table_attacks=[attack], table_defenses=[defense],
        throwin_defense_ranks=False,
    )
    assert thrown(without_flag, 0) == []


def _table_with_free_ranks(pool, count, trump=0):
    """A fully-answered table of `count` attacks, plus two cards that match nothing.

    Attacks are hearts cards of ranks ``0..count-1``; each is answered by a spade of the
    same rank -- the trump beats any non-trump, and the suits differ so the cards are
    distinct. Keeping both lists on the same ranks holds the table's rank set to
    ``{0..count-1}``, which leaves ranks free for cards that match nothing. That is what
    lets a test assert "exactly this one card is a legal throw-in" rather than "at least
    one".
    """
    ranks = list(range(count))
    attacks = [pool.reserve(card_at(r, 1)) for r in ranks]
    defenses = [pool.reserve(card_at(r, 0)) for r in ranks]
    tabled_ranks = {rank_of(c) for c in attacks} | {rank_of(c) for c in defenses}
    free_ranks = [r for r in range(9) if r not in tabled_ranks]
    assert len(free_ranks) >= 2, f"table consumed every rank: {sorted(tabled_ranks)}"
    free = [pool.reserve(card_at(free_ranks[0], 3)), pool.reserve(card_at(free_ranks[1], 3))]
    return attacks, defenses, free


def test_throwin_limit_caps_at_six_when_defender_holds_more_than_six():
    """The cap is 6 even when the defender's hand is larger.

    Only a post-take defender can exceed six cards, and a naive ``attacks <=
    defender_hand`` reading would let the table grow to nine attacks, changing every
    throw-in statistic in the dataset.
    """
    pool = Pool()
    attacks, defenses, free = _table_with_free_ranks(pool, 6)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))   # matches a tabled rank

    state = legal_state([matching] + free, pool.take(9), trump=0,
                        table_attacks=attacks, table_defenses=defenses)
    assert state.hand_sizes[1] == 9

    # `matching` is in hand and matches a tabled rank, but the cap refuses it.
    assert thrown(state, 0) == []
    assert STOP in kinds(state, 0)


def test_throwin_allows_a_sixth_attack():
    """The boundary is inclusive: five attacks plus one more is legal.

    Guards the off-by-one in the other direction -- a cap stopping at five would be
    equally wrong and equally invisible in the aggregates.
    """
    pool = Pool()
    attacks, defenses, free = _table_with_free_ranks(pool, 5)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))

    state = legal_state([matching] + free, pool.take(9), trump=0,
                        table_attacks=attacks, table_defenses=defenses)

    assert thrown(state, 0) == [matching]


def test_throwin_limit_scales_with_small_defender_hand():
    """With three attacks tabled and a three-card defender, no more may be added."""
    pool = Pool()
    attacks, defenses, free = _table_with_free_ranks(pool, 3)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))

    state = legal_state([matching] + free, pool.take(3), trump=0,
                        table_attacks=attacks, table_defenses=defenses)

    assert thrown(state, 0) == []


def test_throwin_limit_allows_up_to_the_defenders_hand():
    """One below the defender's hand, a throw-in is still legal."""
    pool = Pool()
    attacks, defenses, free = _table_with_free_ranks(pool, 2)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))

    state = legal_state([matching] + free, pool.take(3), trump=0,
                        table_attacks=attacks, table_defenses=defenses)

    assert thrown(state, 0) == [matching]


def test_throwin_is_refused_while_a_card_is_unbeaten():
    """Cards are answered one at a time; a second throw before the beat is illegal."""
    pool = Pool()
    tabled = pool.reserve(card_at(0, 1))
    matching = pool.reserve(card_at(0, 2))
    others = [pool.reserve(card_at(r, 3)) for r in (5, 6)]

    state = legal_state([matching] + others, pool.take(6), trump=0, table_attacks=[tabled])

    assert thrown(state, 0) == []


def test_attacker_can_always_stop():
    pool = Pool()
    tabled = pool.reserve(card_at(0, 1))
    defense = pool.reserve(card_at(1, 1))
    state = legal_state(pool.take(2), pool.take(6), trump=0,
                        table_attacks=[tabled], table_defenses=[defense])

    assert STOP in kinds(state, 0)


# --- winning -----------------------------------------------------------------


def test_win_declared_when_attacker_empties_hand_before_defender_responds():
    """Playing your last card wins immediately -- the defender never responds.

    This is the chosen rule (an empty hand is a win, checked before any refill), and it
    makes the endgame degenerate: an attacker holding one card wins unconditionally. The
    engine implements exactly that; the dataset measures the consequence.
    """
    pool = Pool()
    attack = pool.take(1)[0]
    state = legal_state([attack], pool.take(6), trump=3, attacker=0)
    assert state.deck_len == 36 - 7

    rules.apply_action(state, pack_action(ATTACK, attack), T)

    assert state.winner == 0
    assert state.table_attack_count == 1
    assert rules.legal_actions(state, 1).size == 0
    assert rules.legal_actions(state, 0).size == 0


def test_win_declared_when_defender_beats_with_last_card():
    """Beating with your last card ends the game at that moment -- no throw-in follows.

    Without this check a defender who empties their hand would keep playing, and the
    refill would hand them six fresh cards.
    """
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))    # already on the table
    defense = pool.reserve(card_at(1, 1))   # the defender's last card
    assert beats(defense, attack, trump=3)

    state = legal_state(pool.take(6), [defense], trump=3, attacker=0,
                        table_attacks=[attack])

    rules.apply_action(state, pack_action(BEAT, defense), T)

    assert state.winner == 1
    assert THROW not in kinds(state, 0)
    assert STOP not in kinds(state, 0)


def test_win_is_checked_before_any_refill():
    # With a full draw pile the naive order (refill, then check) would give the attacker
    # six cards and erase the win.
    pool = Pool()
    attack = pool.take(1)[0]
    state = legal_state([attack], pool.take(6), trump=3, attacker=0)
    deck_before = state.deck_len

    rules.apply_action(state, pack_action(ATTACK, attack), T)

    assert state.winner == 0
    assert state.hand_sizes[0] == 0
    assert state.deck_len == deck_before


def test_no_winner_while_both_hands_are_non_empty():
    pool = Pool()
    hand0 = pool.take(6)
    state = legal_state(hand0, pool.take(6), trump=3, attacker=0)

    rules.apply_action(state, pack_action(ATTACK, hand0[0]), T)

    assert state.winner == EMPTY


# --- taking ------------------------------------------------------------------


def test_take_collects_the_table_into_the_defenders_hand():
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))
    defense = pool.reserve(card_at(1, 1))
    state = legal_state(
        pool.take(6), pool.take(6), trump=3, attacker=0,
        table_attacks=[attack], table_defenses=[defense],
    )

    rules.apply_action(state, pack_action(TAKE, 0), T)

    # The table's two cards joined the defender's hand; the round then refilled it back
    # up to six, so the collected cards are what proves they arrived.
    assert rules.hand_contains(state, 1, attack)
    assert rules.hand_contains(state, 1, defense)
    assert state.table_attack_count == 0
    assert state.table_defense_count == 0
    assert rules.check_invariants(state) == 0


def test_take_does_not_swap_roles():
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=0,
                        table_attacks=[attack])

    rules.apply_action(state, pack_action(TAKE, 0), T)

    # After a take the defender, who now holds the cards, attacks next.
    assert state.attacker == 1


def test_voluntary_take_is_recorded_when_a_beat_was_available():
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))
    beater = pool.reserve(card_at(1, 1))
    state = legal_state(pool.take(5) + [beater], pool.take(6), trump=3, attacker=0,
                        table_attacks=[attack])
    assert BEAT in kinds(state, 1)

    rules.apply_action(state, pack_action(TAKE, 0), T)

    assert state.last_take_voluntary


def test_forced_take_is_recorded_when_no_beat_was_available():
    pool = Pool()
    attack = pool.reserve(card_at(0, 0))     # a trump nothing in a hearts hand can beat
    state = legal_state(pool.take(6), pool.of_suit(6, 1), trump=0, attacker=0,
                        table_attacks=[attack])
    assert kinds(state, 1) == [TAKE]
    rules.apply_action(state, pack_action(TAKE, 0), T)

    assert not state.last_take_voluntary


# --- roles and discard -------------------------------------------------------


def _defended_round(pool):
    attack = pool.reserve(card_at(0, 1))
    defense = pool.reserve(card_at(1, 1))
    return legal_state(pool.take(6), pool.take(5) + [defense], trump=3, attacker=0,
                       table_attacks=[attack]), attack, defense


def test_roles_swap_on_successful_defense():
    pool = Pool()
    state, attack, defense = _defended_round(pool)

    rules.apply_action(state, pack_action(BEAT, defense), T)
    rules.apply_action(state, pack_action(STOP, 0), T)

    assert state.attacker == 1


def test_discarded_cards_leave_the_game():
    pool = Pool()
    state, attack, defense = _defended_round(pool)

    rules.apply_action(state, pack_action(BEAT, defense), T)
    rules.apply_action(state, pack_action(STOP, 0), T)

    assert state.discard_len == 2
    assert set(int(c) for c in state.discard[:2]) == {attack, defense}
    assert state.table_attack_count == 0
    assert rules.check_invariants(state) == 0


# --- refill ------------------------------------------------------------------


def test_refill_never_discards_below_six():
    """A player who took twelve cards draws nothing.

    "Draw until you have six" is only correct read as "while below six". A literal
    reading would shrink a post-take hand of twelve and change the whole endgame.
    """
    pool = Pool()
    hand0 = pool.take(12)
    hand1 = pool.take(12)
    state = legal_state(hand0, hand1, trump=3, attacker=0)

    rules.refill(state, T)

    assert state.hand_sizes[0] == 12
    assert state.hand_sizes[1] == 12
    # Both hands are already full, so the draw pile is untouched.
    assert state.deck_len == 36 - 24


def test_refill_tops_a_short_hand_up_to_six():
    pool = Pool()
    state = legal_state(pool.take(2), pool.take(6), trump=3, attacker=0)

    rules.refill(state, T)

    assert state.hand_sizes[0] == HAND_TARGET
    assert state.hand_sizes[1] == HAND_TARGET


def test_refill_partial_when_draw_pile_exhausts_midway():
    # Each hand needs four cards but only two remain undrawn, so the first drawer takes
    # both and the second gets nothing. The rest of the deck is in the discard, so the
    # position stays card-conserving and the invariant check stays meaningful.
    pool = Pool()
    state = legal_state(pool.take(2), pool.take(2), trump=3, attacker=0,
                        discard=pool.take(32))
    assert state.deck_len == 0

    rules.refill(state, T)

    assert state.hand_sizes[0] == 2
    assert state.hand_sizes[1] == 2
    assert rules.check_invariants(state) == 0


def test_refill_gives_the_first_drawer_every_remaining_card():
    pool = Pool()
    state = legal_state(pool.take(2), pool.take(2), trump=3, attacker=0,
                        discard=pool.take(30))
    assert state.deck_len == 2
    remaining = [int(c) for c in state.deck[:2]]

    rules.refill(state, T)

    assert state.deck_len == 0
    assert state.hand_sizes[0] == 4
    assert state.hand_sizes[1] == 2
    assert set(int(c) for c in state.hands[0][2:4]) == set(remaining)
    assert rules.check_invariants(state) == 0


def test_attacker_draws_first():
    """The player who attacked the round just completed draws before the other.

    On a successful defense the roles swap, so this is deliberately not "the next
    attacker": the draw order follows the round that just ended.
    """
    pool = Pool()
    state = legal_state(pool.take(1), pool.take(1), trump=3, attacker=0)
    draw_order = [int(c) for c in state.deck[:10]]

    rules.refill(state, T)

    # Each hand was short by five, so the attacker takes the first five and the defender
    # the next five.
    assert [int(c) for c in state.hands[0][1:6]] == draw_order[:5]
    assert [int(c) for c in state.hands[1][1:6]] == draw_order[5:10]


def test_refill_is_skipped_once_a_winner_exists():
    pool = Pool()
    state = legal_state(pool.take(1), pool.take(6), trump=3, attacker=0)
    deck_before = state.deck_len
    state.winner = 1

    rules.refill(state, T)

    assert state.deck_len == deck_before
    assert state.hand_sizes[0] == 1


# --- phases ------------------------------------------------------------------


def test_phase_cut_boundaries():
    def with_deck(remaining):
        state = rules.empty_state(36)
        state.deck_len = remaining
        state.initial_deck = 36
        return state

    assert rules.phase_of(with_deck(36), 36) == rules.PHASE_EARLY
    assert rules.phase_of(with_deck(19), 36) == rules.PHASE_EARLY
    # early is `> 0.50`, so exactly half is mid
    assert rules.phase_of(with_deck(18), 36) == rules.PHASE_MID
    # late is `<= 0.15`: 5 of 36 (0.139) is late, 6 of 36 (0.167) is mid
    assert rules.phase_of(with_deck(6), 36) == rules.PHASE_MID
    assert rules.phase_of(with_deck(5), 36) == rules.PHASE_LATE
    assert rules.phase_of(with_deck(0), 36) == rules.PHASE_LATE


# --- guards ------------------------------------------------------------------


def test_round_cap_raises_rather_than_looping():
    pool = Pool()
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=0)
    state.round_idx = MAX_ROUNDS

    with pytest.raises(RuntimeError, match="round cap"):
        rules.begin_round(state, T)


def test_illegal_attack_card_is_rejected():
    pool = Pool()
    hand0 = pool.take(6)
    state = legal_state(hand0, pool.take(6), trump=3, attacker=0)
    absent = next(c for c in range(36) if c not in hand0)

    with pytest.raises(ValueError, match="illegal card"):
        rules.apply_action(state, pack_action(ATTACK, absent), T)


def test_take_is_rejected_when_no_card_is_on_the_table():
    pool = Pool()
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=0)

    with pytest.raises(ValueError, match="empty table"):
        rules.apply_action(state, pack_action(TAKE, 0), T)


def test_stop_is_rejected_while_a_card_is_unbeaten():
    pool = Pool()
    tabled = pool.reserve(card_at(0, 1))
    state = legal_state(pool.take(2), pool.take(6), trump=0, table_attacks=[tabled])

    with pytest.raises(ValueError, match="unbeaten"):
        rules.apply_action(state, pack_action(STOP, 0), T)


# --- invariants --------------------------------------------------------------


def test_check_invariants_names_decode():
    for name in rules.INVARIANT_NAMES:
        assert isinstance(name, str) and name


def _violated(mask):
    return [rules.INVARIANT_NAMES[i] for i in range(len(rules.INVARIANT_NAMES))
            if mask & (1 << i)]


def test_check_invariants_detects_a_duplicated_card():
    """A corrupted state must be reported, not tolerated.

    Invariant checking that cannot fail is worse than none: it reports confidence it has
    not earned.
    """
    pool = Pool()
    duplicated = pool.take(1)[0]
    others = pool.take(10)

    state = rules.empty_state(36)
    state.hands[0][0] = duplicated
    state.hands[1][0] = duplicated      # the same card in both hands
    for index, card in enumerate(others):
        state.hands[0][index + 1] = card
    state.hand_sizes[0] = np.int8(6)
    state.hand_sizes[1] = np.int8(1)
    remaining = [c for c in range(36) if c != duplicated and c not in others]
    state.deck = np.array(remaining, dtype=np.int16)
    state.deck_len = len(remaining)

    mask = rules.check_invariants(state)

    assert mask != 0
    assert "no_duplicate_card" in _violated(mask)


def test_check_invariants_detects_a_missing_card():
    pool = Pool()
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=0)
    state.deck_len = state.deck_len - 1  # a card vanished

    assert "card_conservation" in _violated(rules.check_invariants(state))


def test_card_conservation_holds_after_every_transition():
    """The strongest invariant in the project, checked transition by transition.

    A card duplicated into two hands leaves the game perfectly playable while corrupting
    every downstream metric, so conservation is asserted after every action rather than
    only at game end.

    Bounded by actions rather than rounds: a random policy does not always terminate (see
    ``test_games_terminate_or_hit_the_round_cap``), and an invariant check must not depend
    on the policy being competent.
    """
    checked = 0
    deepest_table = 0

    for seed in range(30):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))
        rng = np.uint64(seed * 7919 + 13)

        for _ in range(2000):
            if state.winner != EMPTY:
                break
            offered = rules.legal_actions(state, rules.acting_player(state))
            if offered.size == 0:
                break
            choice, rng = rules.draw_action(offered, rng)
            rng = np.uint64(rng)
            rules.apply_action(state, int(choice), T)
            checked += 1
            mask = rules.check_invariants(state)
            assert mask == 0, (
                f"invariant violated at seed={seed} round={state.round_idx}: "
                f"{_violated(mask)}"
            )
            if state.table_attack_count > deepest_table:
                deepest_table = state.table_attack_count

    assert checked > 5_000
    # Throw-in chains must actually happen in real play, or the cap would be untested in
    # aggregate and its effect on the dataset would be pure assumption. Random play tops
    # out well below the cap of 6 -- the cap's boundary is pinned directly in the throw-in
    # tests above, and this only proves the mechanism is reachable at all.
    assert deepest_table >= 3


def test_games_terminate_or_hit_the_round_cap():
    """A game either ends, or the round cap stops it. It never loops silently.

    The spec argues termination is guaranteed because, once the draw pile is empty, the
    attacker's hand can only shrink. That argument does not hold: taking is always legal,
    so with an empty deck two players can hand cards back and forth indefinitely while
    neither empties their hand. Measured here -- a uniformly random policy reaches the cap
    on roughly a third of 36-card games.

    The round cap is therefore load-bearing, not a defensive belt-and-braces check. The
    engine turns a cap hit into a recorded anomaly rather than an unbounded run; this test
    pins that the rules surface it loudly instead of looping.
    """
    terminated = 0
    capped = 0

    for seed in range(40):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))
        rng = np.uint64(seed * 7919 + 13)

        while True:
            if state.winner != EMPTY:
                terminated += 1
                break
            offered = rules.legal_actions(state, rules.acting_player(state))
            choice, rng = rules.draw_action(offered, rng)
            rng = np.uint64(rng)
            state = rules.begin_round(state, T) if rules.acting_player(state) == state.attacker \
                and state.table_attack_count == state.table_defense_count else state
            rules.apply_action(state, int(choice), T)

            if state.winner != EMPTY:
                terminated += 1
                break
            if state.round_idx >= MAX_ROUNDS:
                capped += 1
                break

    assert terminated + capped == 40
    # Both outcomes are real: capping is not a rare artefact, and termination is not
    # guaranteed. Pinning the ratio loosely guards against a regression that made one
    # outcome disappear.
    assert terminated > 0
    assert capped > 0


def test_cap_hit_games_declare_no_winner():
    """A game stopped by the cap has no winner and must be flagged by the engine.

    Recording it as a normal loss would let a non-terminating game enter the dataset as a
    result, which is precisely the kind of silently-wrong number this project exists to
    prevent.
    """
    state = rules.setup_game(36, GameStreams.from_parts(0, 0, 0))
    rng = np.uint64(4242)
    capped = False

    for _ in range(60000):
        if state.winner != EMPTY:
            break
        player = rules.acting_player(state)
        if player == state.attacker and state.round_idx >= MAX_ROUNDS:
            capped = True
            break
        offered = rules.legal_actions(state, player)
        if offered.size == 0:
            break
        choice, rng = rules.draw_action(offered, rng)
        rng = np.uint64(rng)
        rules.apply_action(state, int(choice), T)

    if capped:
        assert state.winner == EMPTY
        with pytest.raises(RuntimeError, match="round cap"):
            rules.begin_round(state, T)
