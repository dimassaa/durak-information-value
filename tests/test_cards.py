"""Tests for card encoding, deck construction, and the beat relation.

The beat relation is the single most correctness-critical function in the
project: every legality decision in the engine routes through it, and a subtle
error there silently biases every win rate in the dataset. The tests therefore
enumerate the entire cross product rather than spot-checking examples.
"""

import numpy as np
import pytest

from durak.cards import (
    DECK_SIZES,
    beats,
    build_deck,
    card_label,
    parse_card_label,
    printed_rank,
    rank_min,
    rank_of,
    suit_of,
)


def test_build_deck_returns_strictly_ascending_unique_ids():
    for deck_size in DECK_SIZES:
        deck = build_deck(deck_size)

        assert len(deck) == deck_size
        assert deck.dtype == np.int16
        assert np.all(np.diff(deck) > 0), f"{deck_size}: ids must be strictly ascending"


def test_build_deck_ids_are_contiguous_from_zero():
    # Contiguity is what makes `deck_remaining + hands + discard + table` checks
    # simple and makes unseen-set bookkeeping a difference of ranges.
    for deck_size in DECK_SIZES:
        deck = build_deck(deck_size)

        assert deck[0] == 0
        assert deck[-1] == deck_size - 1


def test_rank_min_matches_deck_definition():
    assert rank_min(36) == 6
    assert rank_min(52) == 2


def test_rank_and_suit_decode_round_trip():
    # `rank_of` returns a 0-based rank offset, so it takes no deck_size and
    # carries no deck knowledge: id = offset * 4 + suit.
    for deck_size in DECK_SIZES:
        for card in range(deck_size):
            offset = rank_of(card)
            suit = suit_of(card)

            # 36 cards carry 9 ranks (offsets 0..8), 52 cards carry 13 (0..12).
            assert 0 <= offset < deck_size // 4
            assert 0 <= suit <= 3
            assert offset * 4 + suit == card


def test_printed_rank_adds_the_deck_floor():
    assert printed_rank(0, 36) == 6
    assert printed_rank(0, 52) == 2
    for deck_size in DECK_SIZES:
        for card in range(deck_size):
            assert printed_rank(card, deck_size) == rank_of(card) + rank_min(deck_size)


def test_printed_rank_spans_the_full_range():
    for deck_size in DECK_SIZES:
        printed = {printed_rank(c, deck_size) for c in range(deck_size)}

        assert min(printed) == rank_min(deck_size)
        assert max(printed) == 14
        assert len(printed) == deck_size // 4


def test_no_card_beats_itself():
    for deck_size in DECK_SIZES:
        for trump in range(4):
            for card in range(deck_size):
                assert not beats(card, card, trump)


def test_trump_of_any_rank_beats_every_non_trump():
    for deck_size in DECK_SIZES:
        for trump in range(4):
            for trump_card in range(deck_size):
                if suit_of(trump_card) != trump:
                    continue
                for other in range(deck_size):
                    if suit_of(other) == trump:
                        continue
                    assert beats(trump_card, other, trump), (
                        f"trump {trump_card} must beat non-trump {other}"
                    )


def test_non_trump_never_beats_a_trump():
    for deck_size in DECK_SIZES:
        for trump in range(4):
            for trump_card in range(deck_size):
                if suit_of(trump_card) != trump:
                    continue
                for other in range(deck_size):
                    if suit_of(other) == trump:
                        continue
                    assert not beats(other, trump_card, trump)


def test_within_a_suit_only_higher_ranks_beat():
    for deck_size in DECK_SIZES:
        for trump in range(4):
            for a in range(deck_size):
                for b in range(deck_size):
                    if suit_of(a) != suit_of(b):
                        continue
                    expected = rank_of(a) > rank_of(b)
                    assert beats(a, b, trump) is expected


def test_beats_is_asymmetric_across_the_whole_36_card_deck():
    # Enumerate every ordered pair under a fixed trump and confirm that
    # `beats(a, b)` and `beats(b, a)` are never both true.
    deck_size = 36
    trump = 0

    for a in range(deck_size):
        for b in range(deck_size):
            assert not (beats(a, b, trump) and beats(b, a, trump))


def test_within_trump_suit_higher_rank_wins():
    deck_size = 36
    trump = 0

    trump_cards = [c for c in range(deck_size) if suit_of(c) == trump]
    for a in trump_cards:
        for b in trump_cards:
            assert beats(a, b, trump) is (rank_of(a) > rank_of(b))


def test_card_label_round_trips():
    for deck_size in DECK_SIZES:
        for card in range(deck_size):
            label = card_label(card, deck_size)

            assert parse_card_label(label, deck_size) == card


def test_card_labels_are_unique():
    for deck_size in DECK_SIZES:
        labels = [card_label(c, deck_size) for c in range(deck_size)]

        assert len(set(labels)) == deck_size


def test_card_label_uses_suit_symbols():
    deck = build_deck(36)

    labels = [card_label(int(c), 36) for c in deck]
    joined = " ".join(labels)

    for symbol in ("♠", "♥", "♦", "♣"):
        assert symbol in joined


def test_unknown_deck_size_is_rejected():
    with pytest.raises(ValueError, match="deck_size"):
        build_deck(40)

    with pytest.raises(ValueError, match="deck_size"):
        rank_min(48)


def test_card_out_of_range_is_rejected_by_label_helpers():
    with pytest.raises(ValueError, match="card"):
        card_label(36, 36)

    with pytest.raises(ValueError, match="card"):
        card_label(-1, 36)

    # The 52-card deck runs ids 0..51; 36 is the last valid id there.
    assert card_label(36, 52) is not None

    with pytest.raises(ValueError, match="card"):
        card_label(52, 52)

    assert card_label(51, 52) == "A♣"


def test_rank_absent_from_the_smaller_deck_is_rejected():
    # "2♠" is a real card in a 52-card deck but not in a 36-card deck, so parsing
    # it against 36 must fail rather than silently alias onto another card.
    assert parse_card_label("2♠", 52) == 0

    with pytest.raises(ValueError, match="card"):
        parse_card_label("2♠", 36)


def test_malformed_label_is_rejected():
    with pytest.raises(ValueError, match="suit"):
        parse_card_label("AQ", 36)

    with pytest.raises(ValueError, match="rank"):
        parse_card_label("Z♠", 36)

    with pytest.raises(ValueError, match="short"):
        parse_card_label("♠", 36)