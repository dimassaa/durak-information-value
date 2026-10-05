"""Action encoding.

An action is packed into a single ``int32`` so the innermost loops of the engine can
pass it around as a scalar: ``(card << 6) | kind``. The card field is 6 bits wide, which
covers the 52-card deck with room to spare, and the kind occupies the low three bits so a
packed action is never negative -- which matters because ``-1`` is the empty-hand marker in
the same arrays.

The engine dispatches *strategies* by integer id for the same reason: polymorphic dispatch
over objects is not supported inside a Numba ``prange`` loop.

The functions below are decorated with ``@njit`` so the rules engine can call them from
compiled code. A Numba-decorated function is still callable from ordinary Python, so one
implementation serves both callers and there is no second encoding to drift.
"""

from __future__ import annotations

import numpy as np
from numba import njit

ATTACK = 0
THROW = 1
STOP = 2
BEAT = 3
TAKE = 4

# Card field width. 6 bits holds 0..63, above the largest card id (51), so the packed
# action stays non-negative -- which matters because -1 is the empty-hand marker in the
# same arrays. The card occupies the *high* bits so the kind stays a cheap mask.
CARD_BITS = 6
CARD_MASK = (1 << CARD_BITS) - 1
KIND_MASK = 0b111

# Kinds carrying no card (STOP, TAKE) store 0 in the card field.
NO_CARD = 0


@njit(cache=False, inline="always")
def pack_action(kind: int, card: int = 0) -> int:
    """Pack an action kind and card id into one non-negative integer.

    No range check: Numba supports raising a constant ``ValueError`` from nopython code,
    and the check would otherwise sit in the innermost loop of the engine.
    """
    if kind == STOP or kind == TAKE:
        return kind
    return (card << CARD_BITS) | kind


def unpack_action(action: int) -> tuple:
    """Split a packed action back into ``(kind, card)``.

    Cardless kinds report card ``0``; callers must key off the kind rather than
    inspecting the card for them. Not compiled: it builds a Python tuple and is only
    used by tests, telemetry, and report labels.
    """
    kind = action & KIND_MASK
    if kind == STOP or kind == TAKE:
        return kind, NO_CARD
    return kind, int(action >> CARD_BITS)


@njit(cache=False, inline="always")
def action_kind(action: int) -> int:
    """Just the kind, without materialising the card."""
    return action & KIND_MASK


@njit(cache=False, inline="always")
def action_card(action: int) -> int:
    """Just the card id, or ``0`` for a cardless kind."""
    kind = action & KIND_MASK
    if kind == STOP or kind == TAKE:
        return 0
    return action >> CARD_BITS


def has_card(action: int) -> bool:
    """Whether this action carries a card."""
    return (action & KIND_MASK) not in (STOP, TAKE)


__all__ = [
    "ATTACK",
    "THROW",
    "STOP",
    "BEAT",
    "TAKE",
    "CARD_BITS",
    "CARD_MASK",
    "KIND_MASK",
    "NO_CARD",
    "pack_action",
    "unpack_action",
    "action_kind",
    "action_card",
    "has_card",
]