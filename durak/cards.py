"""Card encoding, deck construction, and the beat relation.

Encoding
--------
A card is a single integer: ``card_id = rank_offset * 4 + suit``, where ``suit``
is 0..3 and ``rank_offset`` is the card's rank indexed from zero *within its deck*
(0 means 6 in the 36-card deck, 0 means 2 in the 52-card deck).

Ids are therefore dense and contiguous from zero, which makes two things cheap
downstream: the unseen-card bookkeeping in the opponent model becomes a set
difference over integer ranges, and the card-conservation invariant can be
asserted by counting ids rather than by comparing sets.

``rank_of`` returns the *offset*, not the printed rank, because it takes no
``deck_size`` and therefore cannot know which deck it is in. That is the right
split: every ordering decision the engine makes (the beat relation, "lowest
trump") is preserved by the offset, so those comparisons need no deck knowledge
at all. Only the places that compare against a printed rank -- "lowest trump
attacks first", "high card means rank >= 10" -- need :func:`printed_rank`, which
takes the deck size.

The beat relation
-----------------
``beats(a, b, trump)`` is the whole legal-move substrate: every attack,
throw-in, and defense decision in the game routes through it. An error here does
not crash -- it biases every win rate in the dataset silently. The rules are:

* a card beats another of the same suit when its rank is higher;
* any trump beats any non-trump, regardless of rank;
* within the trump suit, the higher rank wins (which is the same-suit rule).

``beats`` is a pure integer comparison with no allocation, and every function in
this module is safe to call from Numba's ``nopython`` mode.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Deck sizes this project supports, and their lowest rank.
DECK_SIZES = (36, 52)
_RANK_MIN = {36: 6, 52: 2}
_RANK_MAX = 14
_NUM_SUITS = 4

# Suit order is fixed and indexed 0..3 throughout the project. The order is
# arbitrary but never sorted at runtime, so encoding stays deterministic.
SUIT_SYMBOLS = ("♠", "♥", "♦", "♣")

_RANK_LABELS = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
_LABEL_RANKS = {label: rank for rank, label in _RANK_LABELS.items()}


def _validate_deck_size(deck_size: int) -> None:
    if deck_size not in _RANK_MIN:
        raise ValueError(
            f"deck_size must be one of {DECK_SIZES}, got {deck_size}"
        )


def _validate_card(card: int, deck_size: int) -> None:
    if not 0 <= card < deck_size:
        raise ValueError(
            f"card id {card} is out of range for a {deck_size}-card deck"
        )


def rank_min(deck_size: int) -> int:
    """Lowest printed rank in the deck (6 for 36 cards, 2 for 52)."""
    _validate_deck_size(deck_size)
    return _RANK_MIN[deck_size]


# The accessors below are decorated with `@njit` because the rules engine calls them from
# inside its hot loop, and Numba cannot infer an untyped plain-Python global there.
# `@njit` keeps a single implementation: the same function serves Python callers and
# compiled callers, so the beat relation cannot drift between the readable reference and
# the engine.


@njit(cache=False, inline="always")
def rank_of(card: int) -> int:
    """Rank index of a card id, counted from zero within its deck.

    0-based on purpose: the beat relation and "lowest trump" only need ordering,
    and offsets preserve ordering in every deck. Callers that need a printed
    rank (6..14) use :func:`printed_rank`.
    """
    return card // _NUM_SUITS


@njit(cache=False, inline="always")
def suit_of(card: int) -> int:
    """Suit index of a card id, in 0..3."""
    return card % _NUM_SUITS


@njit(cache=False, inline="always")
def printed_rank(card: int, deck_size: int) -> int:
    """Printed rank of a card id: 6..14 for 36 cards, 2..14 for 52.

    Needed only where the rules name a printed rank: "the lowest trump attacks
    first" and "high card means rank >= 10".

    The deck floor is taken from ``_RANK_MIN`` as a lookup rather than through
    :func:`rank_min`, because a dict lookup cannot be typed by Numba and the validation in
    ``rank_min`` is only reachable from Python anyway.
    """
    floor = 6 if deck_size == 36 else 2
    return rank_of(card) + floor


@njit(cache=False, inline="always")
def card_id(rank_offset: int, suit: int) -> int:
    """Encode a (rank offset, suit) pair into a card id."""
    return rank_offset * _NUM_SUITS + suit


def build_deck(deck_size: int) -> np.ndarray:
    """Every card in the deck as contiguous ids from 0, in ascending order.

    Ascending order is not cosmetic: the shuffle in ``durak.rng`` permutes this
    array, and a sorted starting order means any determinism bug in the shuffle
    shows up immediately as an unchanged deck.
    """
    _validate_deck_size(deck_size)
    return np.arange(deck_size, dtype=np.int16)


@njit(cache=False, inline="always")
def beats(a: int, b: int, trump: int) -> bool:
    """Whether card ``a`` beats card ``b`` given trump suit ``trump``.

    Deliberately not short-circuited into a single expression: the two rules are
    conceptually separate and keeping them separate keeps the trump-vs-trump case
    on the same-suit path, where the rank comparison already holds.
    """
    suit_a = a % _NUM_SUITS
    suit_b = b % _NUM_SUITS

    if suit_a == suit_b:
        return a // _NUM_SUITS > b // _NUM_SUITS

    return suit_a == trump


def card_label(card: int, deck_size: int) -> str:
    """Human-readable label such as ``"A♠"``, for logs and report annotations."""
    _validate_deck_size(deck_size)
    _validate_card(card, deck_size)
    rank = printed_rank(card, deck_size)
    return f"{_RANK_LABELS[rank]}{SUIT_SYMBOLS[suit_of(card)]}"


def parse_card_label(label: str, deck_size: int) -> int:
    """Inverse of :func:`card_label`, so labels survive a round trip."""
    _validate_deck_size(deck_size)
    if len(label) < 2:
        raise ValueError(f"card label {label!r} is too short")

    suit_symbol = label[-1]
    if suit_symbol not in SUIT_SYMBOLS:
        raise ValueError(f"card label {label!r} has no valid suit symbol")
    rank_label = label[:-1]
    if rank_label not in _LABEL_RANKS:
        raise ValueError(f"card label {label!r} has no valid rank")

    printed = _LABEL_RANKS[rank_label]
    offset = printed - rank_min(deck_size)
    if offset < 0:
        raise ValueError(
            f"card label {label!r} names a rank that does not exist in a "
            f"{deck_size}-card deck"
        )

    card = card_id(offset, SUIT_SYMBOLS.index(suit_symbol))
    _validate_card(card, deck_size)
    return card