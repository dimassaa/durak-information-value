"""Scratch states for the search.

Why this module exists
----------------------
Task 10 was blocked on a Numba constraint, isolated by probe rather than guessed:

* an ``@njit`` function called from Python may **not** call ``GameState(...)`` inline. It raises
  ``TypeError: invalid number of args: expected 8, got 7`` from Numba's ``ArgPacker`` -- a mismatch in
  how many ABI values the jitclass expands to.
* it **may** construct one by calling another compiled function: ``rules.deal_state`` does exactly
  that and is reached from Python through ``setup_game``, which works.

So construction is available; only the inline form is not. This module takes the remaining option:
Python allocates the scratch states, and the compiled search copies into them by field assignment.
Assignment is what ``apply_action`` already does successfully, so it crosses the boundary cleanly.

Why the scratch is caller-owned rather than allocated per search
----------------------------------------------------------------
A jitclass argument costs about 190 us per call from Python against about 0.3 us inside compiled code.
Allocating inside the search -- once per candidate action per particle per response -- would cost more
than the search itself. Three states allocated once per game and reused is the difference between a
search that can run inside a tournament and one that cannot.

The cost of reuse is that a stale field leaks between candidates unless the copy covers everything. It
covers everything: :func:`_copy_into` assigns all twenty-four mutable fields, and the probe that
verified this ran fifty copy/apply cycles and confirmed the scratch restored the source exactly. A
search comparing positions built on a half-copied scratch would produce confident nonsense, so the copy
is written out in full rather than delegated to anything clever.

A note on what the scratch is *not*
-----------------------------------
It is a working buffer, not a cache. :func:`durak.search.expectimax` leaves it holding the last
candidate's post-action state, and that is by design -- an earlier test asserted the opposite and was
wrong. The invariant worth testing is that :func:`_copy_into` restores the source, which is a property
of the copy alone.

Overwriting, not merging
------------------------
Every helper that takes a scratch states which it does. :func:`_copy_into` *overwrites* every mutable
field. A helper that merely *populates* one -- as the search does when substituting a particle's hand --
must be documented as doing so, because a caller who pre-loaded a buffer and passed it to a function
that silently re-copies will get a plausible answer to a different question. That mistake cost a test
its entire reason for existing.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from durak import rules


@njit(cache=True)
def _copy_into(dst, src) -> None:
    """Overwrite every mutable field of `dst` with `src`.

    Assignment, never construction -- see the module docstring for why that distinction is the whole
    trick. Arrays are copied element-wise rather than rebound: rebinding a jitclass field is not
    something Numba allows, and a rebind would alias the source, so a later mutation of the scratch
    would reach back into the caller's state.
    """
    for i in range(src.deck.shape[0]):
        dst.deck[i] = src.deck[i]
    dst.deck_len = src.deck_len
    dst.initial_deck = src.initial_deck
    dst.trump = src.trump
    dst.trump_card = src.trump_card

    for player in range(2):
        for j in range(src.hands.shape[1]):
            dst.hands[player, j] = src.hands[player, j]
        dst.hand_sizes[player] = src.hand_sizes[player]

    for i in range(src.discard.shape[0]):
        dst.discard[i] = src.discard[i]
    dst.discard_len = src.discard_len

    for i in range(src.table_attacks.shape[0]):
        dst.table_attacks[i] = src.table_attacks[i]
    dst.table_attack_count = src.table_attack_count

    for i in range(src.table_defenses.shape[0]):
        dst.table_defenses[i] = src.table_defenses[i]
    dst.table_defense_count = src.table_defense_count

    dst.attacker = src.attacker
    dst.round_idx = src.round_idx
    dst.winner = src.winner
    dst.redeals = src.redeals
    dst.both_have_trumps = src.both_have_trumps
    dst.first_attacker_random_fallback = src.first_attacker_random_fallback
    dst.flag_redeals_exceeded = src.flag_redeals_exceeded
    dst.last_take_voluntary = src.last_take_voluntary
    dst.rng_state = src.rng_state
    dst.rng_p0 = src.rng_p0
    dst.rng_p1 = src.rng_p1


class ScratchStates:
    """Three reusable `GameState` buffers for one search loop.

    Owned by the caller -- in practice by the policy inside the engine, allocated once per game.
    Constructing them costs about 190 us each, which is why they are not allocated per call.

    The arrays are sized independently of any particular position, because the scratch is allocated
    before one is known: a discard that outgrew the scratch would write past the end of the buffer and
    take the game down with no diagnostic.
    """

    __slots__ = ("after", "hypothesis", "reply", "deck_size")

    def __init__(self, deck_size: int = 36) -> None:
        self.deck_size = deck_size
        self.after = self._blank()
        self.hypothesis = self._blank()
        self.reply = self._blank()

    def _blank(self):
        state = rules.GameState(
            np.zeros(self.deck_size, dtype=np.int16),
            np.full((2, rules.HAND_CAPACITY), rules.EMPTY, dtype=np.int16),
            np.zeros(2, dtype=np.int8),
            self.deck_size, 0, True,
        )
        # Sized past anything a 36- or 52-card game produces: 64 discard slots covers a 52-card deck,
        # and the table cannot exceed the throw-in cap plus the attack that opened the round.
        state.discard = np.zeros(64, dtype=np.int16)
        state.table_attacks = np.zeros(8, dtype=np.int16)
        state.table_defenses = np.zeros(8, dtype=np.int16)
        return state
