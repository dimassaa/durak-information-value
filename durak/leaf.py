"""The leaf transition: depth-1 search arithmetic with no second `GameState`.

Stage 11 spent three stages failing to give the search working memory. Four designs were tried; the one
chosen failed for a reason the ledger already recorded from the stage before. The framing was wrong. The
search does not need a second state -- it needs the *arithmetic* a state would have supplied.

At depth 1 a leaf is :func:`durak.evaluation.evaluate` on the position after one attacker action and one
defender reply, and ``evaluate`` reads only five things: each hand's size, trump count and high-card count,
the attacker, and whether the draw pile is empty. That is arithmetic over two card arrays.
``rules.legal_actions`` is not needed on the defender's side either -- their replies are "which cards beat
the pending attack", computable from their hand, the table and the trump.

So :func:`leaf_profile_after` reproduces the two-ply transition directly and :func:`reply_cards` replaces
the reply enumeration. Nothing here constructs a state, and no policy needs a buffer, which is why the
registry's pinned signature survives untouched.

The rules, and where each came from
-----------------------------------
Read out of ``durak/rules.py``, not out of the spec and not from memory. The two that are easiest to get
backwards:

* ``STOP`` sends the table to the *discard* and grows **neither** hand. ``TAKE`` fills the defender's hand.
  Both end the round and both refill, and both leave ``attacker = 1 - attacker`` -- so they differ *only*
  in which hand grew. An earlier reading of ``_do_take``'s "the roles do not swap" comment took it to mean
  the attacker field is unchanged; it is not, the comment is saying the defender rather than the original
  attacker ends up attacking.
* Refill happens on **round end only**. ``ATTACK``, ``THROW`` and ``BEAT`` do not refill, because they do
  not end the round. Refilling on those would double-count the deck, which is the sort of error that shows
  up as a plausible endgame rather than a crash.

Also load-bearing: draw order is **previous attacker first**, and "never discard down" -- a player holding
twelve cards draws nothing. And a hand that empties ends the game *before* the opponent responds.

Nothing here is verified against memory. ``tests/test_leaf.py`` is a differential test: it plays the same
transitions through the engine and through this arithmetic and demands identical profiles.
"""

import numpy as np
from numba import njit

from durak.actions import ATTACK, BEAT, STOP, TAKE, THROW, action_card
from durak.cards import beats, rank_of
from durak.evaluation import HIGH_RANK

#: Hands top up to six. Mirrors ``rules.HAND_TARGET``; imported rather than duplicated so the two cannot
#: disagree, which would make the endgame profile wrong in a way no unit test would notice.
from durak.rules import HAND_TARGET, EMPTY


@njit(cache=True)
def _remove_card(hand, size: int, card: int) -> int:
    """Drop `card` from a hand, compacting the rest. Returns the new size.

    Compacting because a later ``add`` appends at ``size``; a removed card left in place would be
    duplicated by the append and the profile would count it twice.
    """
    for i in range(size):
        if hand[i] == card:
            for j in range(i, size - 1):
                hand[j] = hand[j + 1]
            hand[size - 1] = EMPTY
            return size - 1
    return size


@njit(cache=True)
def _stats(hand, size: int, trump: int, out) -> None:
    """`out` = (cards, trumps, high cards) for a hand prefix. The only things `evaluate` reads."""
    trumps = 0
    high = 0
    for i in range(size):
        card = hand[i]
        if np.uint64(np.int64(card)) % np.uint64(4) == np.uint64(trump):
            trumps += 1
        if rank_of(card) >= HIGH_RANK:
            high += 1
    out[0] = size
    out[1] = trumps
    out[2] = high


@njit(cache=True)
def _acc(card: int, trump: int, out) -> None:
    """Add one card's (count, trumps, high) contribution to `out`.

    The per-card form of `_stats`. A card in flight between hands still belongs to whoever played it,
    so the transfer accounting needs a card's stats without rebuilding a whole hand array.
    """
    out[0] += 1
    if np.uint64(np.int64(card)) % np.uint64(4) == np.uint64(trump):
        out[1] += 1
    if rank_of(card) >= HIGH_RANK:
        out[2] += 1


@njit(cache=True)
def _table_stats(table, table_size: int, trump: int, out) -> None:
    """Accumulate the stats of every card on the table -- the attacker's cards.

    On a TAKE these are exactly the cards that change hands. The defenses are not: the defender plays
    them and picks them straight back up, so they leave their hand and return, and their hand size nets
    to zero on its own. Only the attacker's cards genuinely migrate, and every card on the table belongs
    to the attacker, because the attack does not change until the round ends.
    """
    for i in range(table_size):
        _acc(table[i], trump, out)


@njit(cache=True)
def reply_cards(state, defender: int, target: int) -> np.ndarray:
    """Cards in `defender`'s hand that beat `target`. The replacement for `legal_actions` here.

    Every legal beater, never truncated. Dropping one would make the opponent look weaker than the rules
    allow, which biases the search toward attacking and looks like strength.
    """
    out = np.empty(state.hands.shape[1], dtype=np.int16)
    size = int(state.hand_sizes[defender])
    count = 0
    for i in range(size):
        card = int(state.hands[defender, i])
        if beats(card, target, state.trump):
            out[count] = card
            count += 1
    return out[:count].copy()


@njit(cache=True)
def _refill_one(hand, size: int, deck, deck_len: int):
    """Top `hand` up to `HAND_TARGET` from the top of `deck`. Never discards down.

    Returns the new size and the new pile length.
    """
    while size < HAND_TARGET and deck_len > 0 and size < hand.shape[0]:
        hand[size] = deck[0]
        size += 1
        for i in range(deck_len - 1):
            deck[i] = deck[i + 1]
        deck_len -= 1
    return size, deck_len


@njit(cache=True)
def _score(mine, size_p: int, theirs, size_o: int, trump: int, player: int, deck_len: int,
           weights) -> float:
    """The five terms of `evaluate`, from scalars rather than from a state."""
    stats_p = np.zeros(3, dtype=np.int64)
    stats_o = np.zeros(3, dtype=np.int64)
    _stats(mine, size_p, trump, stats_p)
    _stats(theirs, size_o, trump, stats_o)

    # `attacker` is resolved by the caller's role bookkeeping, so the role term is folded in there; this
    # scores everything else. Keeping the split explicit is why there is no `attacker` parameter here.
    value = weights[0] * (stats_o[0] - stats_p[0])
    value += weights[1] * (stats_p[1] - stats_o[1])
    value += weights[2] * (stats_p[2] - stats_o[2])
    if deck_len == 0:
        value += weights[4] * (stats_p[0] - stats_o[0])
    return value


@njit(cache=True)
def leaf_profile_after(state, player: int, kind_a: int, card_a: int, kind_b: int, card_b: int,
                       hand_p, hand_o, weights) -> float:
    """`evaluate`'s value for the position after two plies: one by `player`, then one by the opponent.

    Role-general on purpose. `player` is the *acting* player, not the attacker, so the first ply is an
    attack or a throw only when they are attacking and a beat when they are defending, and the second ply
    is the mirror image. An earlier version assumed the actor was always the attacker, which meant the
    search raised on every position where the defender was to act -- half of all positions.

    `hand_p` and `hand_o` are the two hands the search *knows*: the acting player's own, and its hypothesis
    about the opponent's. Neither is read from `state` beyond the table, the deck and the trump, so the
    same call serves a sampled particle set and the opponent's real hand.

    A ply that empties a hand ends the game: the remaining ply does not happen, and no refill occurs.
    """
    opponent = 1 - player
    trump = state.trump

    mine = np.empty(state.hands.shape[1], dtype=np.int16)
    size_p = int(state.hand_sizes[player])
    for i in range(size_p):
        mine[i] = hand_p[i]

    theirs = np.empty(state.hands.shape[1], dtype=np.int16)
    size_o = int(state.hand_sizes[opponent])
    for i in range(size_o):
        theirs[i] = hand_o[i]

    deck = np.empty(state.deck.shape[0] + 2 * HAND_TARGET + 8, dtype=np.int16)
    deck_len = int(state.deck_len)
    for i in range(deck_len):
        deck[i] = state.deck[i]

    # The table, mutable: cards leave it for a hand or for the discard, and new attacks land on it.
    table = np.empty(16, dtype=np.int16)
    table_size = int(state.table_attack_count)
    for i in range(table_size):
        table[i] = state.table_attacks[i]
    defenses = np.empty(16, dtype=np.int16)
    defense_count = int(state.table_defense_count)
    for i in range(defense_count):
        defenses[i] = state.table_defenses[i]

    # Which seat is attacking *now*. Flipped by both round-ending actions and by nothing else.
    player_attacks = (int(state.attacker) == player)
    round_ends = False

    # Stats of the cards that change hands during these plies, split by original owner. Fed to `_finish`,
    # which neutralises them: a transfer is not progress. See `_finish` for why all three stats travel.
    moved_p = np.zeros(3, dtype=np.int64)
    moved_o = np.zeros(3, dtype=np.int64)

    # --- ply 1: `player` acts ---------------------------------------------------
    if player_attacks:
        if kind_a == STOP:
            # Table to the discard, no hand grows, round ends. Nothing migrated: the cards left play.
            round_ends = True
        elif kind_a == ATTACK or kind_a == THROW:
            size_p = _remove_card(mine, size_p, card_a)
            table[table_size] = card_a
            table_size += 1
            if size_p == 0:
                return _finish(mine, size_p, theirs, size_o, trump, player, deck_len, player_attacks,
                               weights, moved_p, moved_o)
        else:
            raise ValueError(f"an attacking player cannot play kind {kind_a}")
    else:
        if kind_a == TAKE:
            # `player` is the defender, so the table holds the opponent's attacks: they migrate.
            _table_stats(table, table_size, trump, moved_o)
            _collect(theirs, size_o, table, table_size, defenses, defense_count)
            size_o += table_size + defense_count
            round_ends = True
        elif kind_a == BEAT:
            size_p = _remove_card(mine, size_p, card_a)
            defenses[defense_count] = card_a
            defense_count += 1
            if size_p == 0:
                return _finish(mine, size_p, theirs, size_o, trump, player, deck_len, player_attacks,
                               weights, moved_p, moved_o)
        else:
            raise ValueError(f"a defending player cannot play kind {kind_a}")

    # --- ply 2: the other seat acts --------------------------------------------
    # The opponent is attacking exactly when the actor was not.
    if not round_ends:
        # Ply 1 only changes the attacker on a round end, and those cases set `round_ends`, so the
        # attacker is still whoever it was. The actor keeps their own role; the other seat plays the mirror.
        if not player_attacks:
            # The other seat is the attacker: it throws in or stops.
            if kind_b == STOP:
                round_ends = True
            elif kind_b == THROW or kind_b == ATTACK:
                size_o = _remove_card(theirs, size_o, card_b)
                table[table_size] = card_b
                table_size += 1
                if size_o == 0:
                    return _finish(mine, size_p, theirs, size_o, trump, player, deck_len,
                                   not player_attacks, weights, moved_p, moved_o)
            else:
                raise ValueError(f"an attacking opponent cannot play kind {kind_b}")
        else:
            # The other seat is the defender: it beats the newest attack or takes.
            if kind_b == BEAT:
                size_o = _remove_card(theirs, size_o, card_b)
            elif kind_b == TAKE:
                # `player` is the attacker here, so the table holds `player`'s own attacks and those are
                # the cards that migrate to the opponent.
                _table_stats(table, table_size, trump, moved_p)
                _collect(theirs, size_o, table, table_size, defenses, defense_count)
                size_o += table_size + defense_count
                round_ends = True
            else:
                raise ValueError(f"a defending opponent cannot play kind {kind_b}")

    # --- refill, if a round ended ----------------------------------------------
    if round_ends:
        # Draw order is the PREVIOUS attacker first -- the round that just ended, not the one starting.
        # "Never discard down": a player holding twelve cards draws nothing.
        if player_attacks:
            size_p, deck_len = _refill_one(mine, size_p, deck, deck_len)
            size_o, deck_len = _refill_one(theirs, size_o, deck, deck_len)
        else:
            size_o, deck_len = _refill_one(theirs, size_o, deck, deck_len)
            size_p, deck_len = _refill_one(mine, size_p, deck, deck_len)

        # The roles swap on a *successful defense* only. After a STOP the attacker hands over the lead;
        # after a TAKE the original attacker keeps attacking and the taker skips the round. Flipping
        # unconditionally was the same rules bug as in `rules._do_take`, reproduced in the leaf's role
        # bookkeeping, and it made every post-take leaf score the wrong seat for the role term.
        if kind_b != TAKE and kind_a != TAKE:
            player_attacks = not player_attacks

    return _finish(mine, size_p, theirs, size_o, trump, player, deck_len, player_attacks, weights,
                   moved_p, moved_o)


@njit(cache=True)
def _collect(hand, size: int, table, table_size: int, defenses, defense_count: int) -> None:
    """Take the table: every attack, then every defense, in that order."""
    for i in range(table_size):
        hand[size] = table[i]
        size += 1
    for i in range(defense_count):
        hand[size] = defenses[i]
        size += 1


@njit(cache=True)
def _finish(mine, size_p: int, theirs, size_o: int, trump: int, player: int, deck_len: int,
            player_attacks: bool, weights, moved_p, moved_o) -> float:
    """The five terms of `evaluate`, from two hand arrays and four scalars.

    `moved_p` and `moved_o` are the (count, trumps, high) stats of the cards that changed hands during
    the plies, split by their original owner: `moved_p` is what left the acting player's hand, `moved_o`
    what left the opponent's. A card handed to the opponent and picked up again has not left play, so it
    must not read as progress -- otherwise a depth-1 leaf scores *feeding the opponent a card they cannot
    beat* identically to winning a trick, and both arms settle into an unbreakable cycle that never
    terminates (measured: 27-35% of games capped at the round limit, the draw pile never consumed).

    Neutralising the transfer means undoing it in the stats: subtract what left each hand and add it back
    to the player it came from. All three hand-reading terms are corrected together on purpose. Correcting
    only the count leaves the trump term swinging the other way and the pair still sums to a gain, which
    leaves the cycle intact.

    A card that reaches the *discard* is not in `moved_p`/`moved_o` and keeps its credit. That asymmetry is
    the whole point: discarding is progress, transferring is not.
    """
    stats_p = np.zeros(3, dtype=np.int64)
    stats_o = np.zeros(3, dtype=np.int64)
    _stats(mine, size_p, trump, stats_p)
    _stats(theirs, size_o, trump, stats_o)

    # Undo the transfer, per original owner, before any term reads the stats. A card that left `player`'s
    # hand is credited back to `player` and debited from the opponent, and vice versa: the card is scored
    # as if it had never changed hands.
    for k in range(3):
        stats_p[k] += moved_p[k]
        stats_o[k] -= moved_p[k]
        stats_o[k] += moved_o[k]
        stats_p[k] -= moved_o[k]

    value = weights[0] * (stats_o[0] - stats_p[0])
    value += weights[1] * (stats_p[1] - stats_o[1])
    value += weights[2] * (stats_p[2] - stats_o[2])
    value += weights[3] * (1.0 if player_attacks else -1.0)
    if deck_len == 0:
        value += weights[4] * (stats_p[0] - stats_o[0])
    return value


@njit(cache=True)
def reply_cards_in(hand, size: int, trump: int, target: int) -> np.ndarray:
    """Cards in a *hypothesised* hand that beat `target`.

    The form the search uses: the opponent's hand is not `state.hands[opponent]` but one particle's cards,
    so the enumeration has to accept the hand as an array. :func:`reply_cards` is the wrapper for the
    live-state case and exists so `tests/test_leaf.py` can diff it against `rules.legal_actions`.
    """
    out = np.empty(hand.shape[0], dtype=np.int16)
    count = 0
    for i in range(size):
        card = int(hand[i])
        if beats(card, target, trump):
            out[count] = card
            count += 1
    return out[:count].copy()
