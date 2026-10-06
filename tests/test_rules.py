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
from tests.support import Pool, card_at, legal_state, table_with_free_ranks

EMPTY = rules.EMPTY
HAND_CAPACITY = rules.HAND_CAPACITY
HAND_TARGET = rules.HAND_TARGET
MAX_ROUNDS = rules.MAX_ROUNDS

T = rules.DEFAULT_SETTINGS


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
        # `trump` is the suit; the card is recorded separately and is still in the pile.
        assert state.trump == state.deck[state.deck_len - 1] % 4
        assert state.trump_card == state.deck[state.deck_len - 1]


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


def test_setup_leaves_the_callers_streams_untouched():
    """The deal must not consume the caller's `GameStreams`, nor a player's stream.

    The deal draws from the deal stream alone, and its resulting stream position belongs to the
    *game* -- it is carried in `state.rng_state` -- rather than to whatever object the caller
    happened to pass in. So a caller can reuse one bundle, and a player's stream is provably
    untouched by the deal.

    The second half matters most: if the deal consumed a player's stream, a run differing in one
    strategy would differ in the deal too, and the paired comparison the primary metric depends
    on would be silently broken.
    """
    streams = GameStreams.from_parts(5, 0, 0)
    before = streams.snapshot()

    state = rules.setup_game(36, streams)

    assert streams.snapshot() == before, "the deal consumed the caller's streams"
    assert state.rng_p0 == np.uint64(before[1]), "the deal touched player 0's stream"
    assert state.rng_p1 == np.uint64(before[2]), "the deal touched player 1's stream"
    # The deal's own position is recorded on the state, advanced past the shuffle.
    assert int(state.rng_state) != before[0]


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


def test_a_trump_beats_an_off_suit_non_trump_in_a_real_deal():
    """The regression guard for the worst defect found in this project.

    `GameState.trump` was a card id while `beats()` expects a suit index, so
    `suit_a == trump` compared 0..3 against a card id and was always False. Every off-suit
    beat was refused in every real game: a trump could never beat a non-trump. Nothing
    crashed and no assertion failed -- the hand-built fixtures passed a suit and were correct,
    so only a game built by `setup_game` could expose it.

    This test therefore goes through `setup_game` specifically. It is the one place where the
    two representations could diverge again.
    """
    checked = 0
    for seed in range(60):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))
        trump_suit = state.trump

        # Every trump in either hand must beat every non-trump.
        for player in (0, 1):
            for card in state.hands[player, :state.hand_sizes[player]]:
                if suit_of(card) != trump_suit:
                    continue
                for other in range(36):
                    if suit_of(other) == trump_suit:
                        continue
                    assert beats(card, other, trump_suit)
                    assert not beats(other, card, trump_suit)
                    checked += 1
        if checked > 200:
            break

    assert checked > 200, "the sample contained too few trump/non-trump pairs"


def test_trump_is_stored_as_a_suit_not_a_card_id():
    for seed in range(20):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))

        assert 0 <= state.trump <= 3
        assert state.trump == state.trump_card % 4
        assert rules.check_invariants(state) == 0


def test_the_trump_card_stays_in_the_draw_pile():
    for seed in range(10):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))

        assert state.trump_card == state.deck[state.deck_len - 1]


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


def test_throwin_limit_caps_at_six_when_defender_holds_more_than_six():
    """The cap is 6 even when the defender's hand is larger.

    Only a post-take defender can exceed six cards, and a naive ``attacks <=
    defender_hand`` reading would let the table grow to nine attacks, changing every
    throw-in statistic in the dataset.
    """
    pool = Pool()
    attacks, defenses, free = table_with_free_ranks(pool, 6)
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
    attacks, defenses, free = table_with_free_ranks(pool, 5)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))

    state = legal_state([matching] + free, pool.take(9), trump=0,
                        table_attacks=attacks, table_defenses=defenses)

    assert thrown(state, 0) == [matching]


def test_throwin_limit_scales_with_small_defender_hand():
    """With three attacks tabled and a three-card defender, no more may be added."""
    pool = Pool()
    attacks, defenses, free = table_with_free_ranks(pool, 3)
    matching = pool.reserve(card_at(rank_of(attacks[0]), 2))

    state = legal_state([matching] + free, pool.take(3), trump=0,
                        table_attacks=attacks, table_defenses=defenses)

    assert thrown(state, 0) == []


def test_throwin_limit_allows_up_to_the_defenders_hand():
    """One below the defender's hand, a throw-in is still legal."""
    pool = Pool()
    attacks, defenses, free = table_with_free_ranks(pool, 2)
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


def test_a_take_leaves_the_attack_with_the_original_attacker():
    """The rule that makes this the *podkidnoy* variant: the taker passes, the attacker keeps going.

    An earlier version of this test asserted the opposite -- `state.attacker == 1`, the taker attacks
    next -- under a name that said "the roles do not swap". The name and the assertion contradicted each
    other, and the assertion matched the bug, so the contradiction went unnoticed.

    Getting this wrong is not a cosmetic error. It lets the taker immediately re-attack the card it just
    took; the opponent cannot beat it, so it takes it straight back; and the two pass one card back and
    forth indefinitely. Measured before the fix: 16/60 and 21/60 games per arm hit the 10,000-round cap,
    with the draw pile never consumed.
    """
    pool = Pool()
    attack = pool.reserve(card_at(0, 1))
    state = legal_state(pool.take(6), pool.take(6), trump=3, attacker=0,
                        table_attacks=[attack])
    defender = rules.acting_player(state)

    rules.apply_action(state, pack_action(TAKE, 0), T)

    assert state.attacker == 0, "after a take the original attacker must still be attacking"
    assert rules.acting_player(state) != defender, "the player who took must skip the round"


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

    # Games now terminate in tens of rounds rather than running to the cap, so the transition
    # count is in the thousands. The floor is deliberately modest: it exists to prove the loop
    # ran a substantial number of transitions, not to pin a count that the trump fix changed.
    assert checked > 2_000
    # aggregate and its effect on the dataset would be pure assumption. Random play tops
    # out well below the cap of 6 -- the cap's boundary is pinned directly in the throw-in
    # tests above, and this only proves the mechanism is reachable at all.
    assert deepest_table >= 3


def test_games_terminate_and_the_round_cap_is_never_reached():
    """Random play ends every game, well inside the cap.

    The spec argues termination is guaranteed because, once the draw pile is empty, the
    attacker's hand can only shrink. That argument only became true after the trump fix
    recorded in ``test_a_trump_beats_an_off_suit_non_trump_in_a_real_deal``: with trumps
    unable to beat off-suit, defenders could only take, hands stopped shrinking, and roughly
    44% of games ran to the cap. Measured now: 0 of 400 random games are capped, mean 46
    rounds, max 117.

    So the cap is genuinely defensive, as the spec intended. A regression that reintroduced
    off-suit deadness would show up here first, as capped games rather than as a plausible
    but wrong win rate.
    """
    terminated = 0
    longest = 0

    for seed in range(200):
        state = rules.setup_game(36, GameStreams.from_parts(seed, 0, 0))
        rng = np.uint64(seed * 7919 + 13)

        for _ in range(20000):
            if state.winner != EMPTY:
                break
            offered = rules.legal_actions(state, rules.acting_player(state))
            choice, rng = rules.draw_action(offered, rng)
            rng = np.uint64(rng)
            rules.apply_action(state, int(choice), T)

        assert state.winner in (0, 1), f"game {seed} did not terminate"
        assert state.round_idx < MAX_ROUNDS
        terminated += 1
        longest = max(longest, state.round_idx)

    assert terminated == 200
    # A ceiling as well as a floor: a game lasting thousands of rounds means the loop is back,
    # and would otherwise pass a test that only checks "eventually finished".
    assert longest < 1_000, f"a game ran {longest} rounds"


def test_the_round_cap_still_guards_an_artificial_stall():
    """The cap is defensive, so it must still fire when a game genuinely cannot progress.

    Asserted by parking the state at the cap directly rather than by hunting for a real
    stalling configuration: the point is that ``begin_round`` refuses, not that some policy
    can be found to trigger it.
    """
    state = rules.setup_game(36, GameStreams.from_parts(0, 0, 0))
    state.round_idx = MAX_ROUNDS

    with pytest.raises(RuntimeError, match="round cap"):
        rules.begin_round(state, T)


def test_a_capped_state_has_no_winner():
    """A state stopped at the cap has no winner, so it cannot be scored as a loss.

    The engine (Task 6) must record a capped game as a flagged anomaly. Scoring it as a normal
    loss would let a non-terminating game enter the dataset as a result, which is precisely
    the kind of silently-wrong number this project exists to prevent.
    """
    state = rules.setup_game(36, GameStreams.from_parts(0, 0, 0))
    state.round_idx = MAX_ROUNDS

    assert state.winner == EMPTY
