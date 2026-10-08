"""The leaf transition: depth-1 and depth-2 search arithmetic with no second `GameState`.

Stage 11 spent three stages failing to give the search working memory. Four designs were tried; the one
chosen failed for a reason the ledger already recorded from the stage before. The framing was wrong. The
search does not need a second state -- it needs the *arithmetic* a state would have supplied.

At depth 1 a leaf is :func:`durak.evaluation.evaluate` on the position after one attacker action and one
defender reply, and ``evaluate`` reads only five things: each hand's size, trump count and high-card count,
the attacker, and whether the draw pile is empty. That is arithmetic over two card arrays.
``rules.legal_actions`` is not needed on the defender's side either -- their replies are "which cards beat
the pending attack", computable from their hand, the table and the trump.

At depth 2 the leaf is the same arithmetic one ply further on: after the opponent has answered, if the
round has not ended, the acting player's own next moves are enumerated (:func:`second_plies`,
:func:`third_plies`) and each line is scored by :func:`leaf_profile_after_ply3`. The state-free property
is load-bearing and is why that enumeration is written out here rather than delegated to
``rules.legal_actions``: a jitclass cannot be constructed inside an ``njit`` function reached from Python
(``TypeError: invalid number of args``), so a derived position cannot be turned back into a state to ask.

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
transitions through the engine and through this arithmetic and demands identical profiles, at depth 1 and
at depth 2.
"""

import numpy as np
from numba import njit

from durak.actions import ATTACK, BEAT, STOP, TAKE, THROW, action_card
from durak.cards import beats, rank_of
from durak.evaluation import HIGH_RANK

#: Hands top up to six. Mirrors ``rules.HAND_TARGET``; imported rather than duplicated so the two cannot
#: disagree, which would make the endgame profile wrong in a way no unit test would notice.
from durak.rules import HAND_TARGET, THROWIN_CAP, EMPTY

#: Table capacity for the local copy. The engine grows its table on demand; here the bound is the rules'
#: own throw-in ceiling of six attacks plus one, and the seventh slot of slack is never used.
TABLE_CAPACITY = 16


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
    # A read cursor rather than a shift. The old body did `deck[i] = deck[i + 1]` for every remaining card
    # after each draw, so filling a hand cost O(pile) per card and a round cost O(pile^2) -- pure memmove of
    # an array only ever read from the front. Nothing indexes the pile by an absolute position, so a cursor
    # is equivalent and O(1) per card.
    #
    # CONSEQUENCE, and it is a convention callers must respect: the pile is consumed from the END, so the
    # hypothesis array is built with the NEXT card LAST. The real draw pile is consumed from the front, so
    # a caller holding the true order must reverse it.
    while size < HAND_TARGET and deck_len > 0 and size < hand.shape[0]:
        hand[size] = deck[deck_len - 1]
        size += 1
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
def _hand_size(hand, capacity: int) -> int:
    """How many leading slots of `hand` hold cards, counting the `EMPTY` sentinel as the end.

    The engine pads a hand with `EMPTY` from `hand_sizes[player]` onwards, so the padding is where the
    size ends. Scanned rather than taken from `state` because the arrays handed to the leaf are the
    search's *belief* about the hands, and in depth 2 that belief is a particle rather than the state.

    A particle array shorter than the live hand -- which a sample can be, if the filter was built a
    round earlier -- is trusted as given rather than padded, because padding it with `EMPTY` would let the
    arithmetic read a card that is not there. Hence the scan is bounded by `hand.shape[0]` as well as by
    `capacity`: a compact particle array has no padding at all, so without that bound the scan would run
    off the end of it.
    """
    limit = capacity
    if hand.shape[0] < limit:
        limit = hand.shape[0]
    size = 0
    while size < limit and int(hand[size]) != EMPTY:
        size += 1
    return size


@njit(cache=True)
def _prepare(state, player: int, hand_p, hand_o, hypo_deck):
    """The local working copy of a position, as plain arrays. The stand-in for a second `GameState`.

    Returns `(mine, size_p, theirs, size_o, table, table_size, defenses, defense_count, deck, deck_len,
    player_attacks)`. The two hands are the ones the search *knows* -- the acting player's own, and its
    hypothesis about the opponent's -- and nothing is read from `state` beyond the table, the deck, the
    hand sizes and the role, so the same copy serves a sampled particle set and the real hand.

    Every ply below mutates these copies and never the caller's state, which is what makes "the search
    does not modify the state" true at depth 2 as well as at depth 1.
    """
    # The hand sizes come from the *arrays*, not from `state.hand_sizes`. The arrays are the belief, so
    # their length is the believed hand size: a particle set may be wider or narrower than the live hand,
    # and reading the live size instead would silently drop the extra hypothesised cards from the
    # branching factor -- which is precisely the substitution the search depends on. Production builds
    # both arms' particles with the live hand size, so the two agree there and only a deliberately
    # mismatched hypothesis can tell them apart.
    mine = np.empty(state.hands.shape[1], dtype=np.int16)
    size_p = _hand_size(hand_p, mine.shape[0])
    for i in range(size_p):
        mine[i] = hand_p[i]

    theirs = np.empty(state.hands.shape[1], dtype=np.int16)
    size_o = _hand_size(hand_o, theirs.shape[0])
    for i in range(size_o):
        theirs[i] = hand_o[i]

    # The draw pile is a HYPOTHESIS, not the real one. Reading `state.deck` here gave the leaf the exact
    # future draw order, which no real observer has: they know the pile's SIZE and nothing else. The
    # consequence of the old version was larger than a modelling nicety -- with the pile public, the
    # complement of the visible cards is exactly the opponent's hand, so "perfect knowledge of the hand"
    # became derivable rather than learned and the research question had no hidden information in it.
    deck = np.empty(hypo_deck.shape[0] + 2 * HAND_TARGET + 8, dtype=np.int16)
    deck_len = hypo_deck.shape[0]
    for i in range(deck_len):
        deck[i] = hypo_deck[i]

    # The table, mutable: cards leave it for a hand or for the discard, and new attacks land on it.
    table = np.empty(TABLE_CAPACITY, dtype=np.int16)
    table_size = int(state.table_attack_count)
    for i in range(table_size):
        table[i] = state.table_attacks[i]
    defenses = np.empty(TABLE_CAPACITY, dtype=np.int16)
    defense_count = int(state.table_defense_count)
    for i in range(defense_count):
        defenses[i] = state.table_defenses[i]

    # Which seat is attacking *now*. Flipped by both round-ending actions and by nothing else.
    return (mine, size_p, theirs, size_o, table, table_size, defenses, defense_count, deck, deck_len,
            int(state.attacker) == player)


@njit(cache=True)
def _apply_ply(kind: int, card: int, actor_attacks: bool, actor_is_player: bool, mine, size_p: int,
               theirs, size_o: int, trump: int, table, table_size: int, defenses,
               defense_count: int, moved):
    """One ply, as arithmetic. Returns `(round_ends, game_over, size_p, size_o, table_size, defense_count)`.

    Role-general, and it has to be: `player` is the *acting* player rather than the attacker, so the same
    body is the attacker's attack or throw and the defender's beat, in either seat order and at any ply.
    An earlier version assumed the actor was always the attacker, which meant the search raised on every
    position where the defender was to act -- half of all positions.

    `moved` accumulates the (count, trumps, high) stats of the cards a TAKE hands over. It is the *attacker's*
    side: on a take the table holds the attacker's cards, whoever picks them up. See `_finish`.

    `game_over` is the empty-hand check, which fires the instant a card leaves a hand and therefore before
    the other ply, before the round can end and before a refill could hand back six cards.
    """
    round_ends = False
    game_over = False

    if actor_attacks:
        if kind == STOP:
            # Table to the discard, no hand grows, round ends. Nothing migrated: the cards left play.
            round_ends = True
        elif kind == ATTACK or kind == THROW:
            if actor_is_player:
                size_p = _remove_card(mine, size_p, card)
                table[table_size] = card
                table_size += 1
                if size_p == 0:
                    game_over = True
            else:
                size_o = _remove_card(theirs, size_o, card)
                table[table_size] = card
                table_size += 1
                if size_o == 0:
                    game_over = True
        else:
            raise ValueError(f"an attacking player cannot play kind {kind}")
    else:
        if kind == TAKE:
            # The defender picks the table up: attacks first, then defenses, into THEIR hand. Crediting
            # these cards to the opponent instead is the defect Stage 12 fixed -- the evaluation reads
            # "opponent's cards minus mine", so handing the opponent cards RAISES the taker's score and
            # taking became free-or-better than beating.
            _table_stats(table, table_size, trump, moved)
            if actor_is_player:
                _collect(mine, size_p, table, table_size, defenses, defense_count)
                size_p += table_size + defense_count
            else:
                _collect(theirs, size_o, table, table_size, defenses, defense_count)
                size_o += table_size + defense_count
            round_ends = True
        elif kind == BEAT:
            if actor_is_player:
                size_p = _remove_card(mine, size_p, card)
                defenses[defense_count] = card
                defense_count += 1
                if size_p == 0:
                    game_over = True
            else:
                size_o = _remove_card(theirs, size_o, card)
                defenses[defense_count] = card
                defense_count += 1
                if size_o == 0:
                    game_over = True
        else:
            raise ValueError(f"a defending player cannot play kind {kind}")

    return round_ends, game_over, size_p, size_o, table_size, defense_count


@njit(cache=True)
def _refill_and_roles(mine, size_p: int, theirs, size_o: int, deck, deck_len: int, player_attacks: bool,
                      round_end_kind: int):
    """Refill both hands in the previous attacker's draw order, then apply the role rule.

    Returns `(size_p, size_o, deck_len, player_attacks)`. Split out because the depth-1 and depth-2 leaves
    reach the round end by different routes and must not drift apart here.

    Draw order is the PREVIOUS attacker first -- the round that just ended, not the one starting. "Never
    discard down": a player holding twelve cards draws nothing.

    The roles swap on a *successful defense* only. After a STOP the attacker hands over the lead; after a
    TAKE the original attacker keeps attacking and the taker skips the round. Flipping unconditionally was
    the same rules bug as in `rules._do_take`, reproduced in the leaf's role bookkeeping, and it made every
    post-take leaf score the wrong seat for the role term.
    """
    if player_attacks:
        size_p, deck_len = _refill_one(mine, size_p, deck, deck_len)
        size_o, deck_len = _refill_one(theirs, size_o, deck, deck_len)
    else:
        size_o, deck_len = _refill_one(theirs, size_o, deck, deck_len)
        size_p, deck_len = _refill_one(mine, size_p, deck, deck_len)

    if round_end_kind != TAKE:
        player_attacks = not player_attacks
    return size_p, size_o, deck_len, player_attacks


@njit(cache=True)
def leaf_profile_after(state, player: int, kind_a: int, card_a: int, kind_b: int, card_b: int,
                       hand_p, hand_o, weights) -> float:
    """The depth-1 leaf: two plies. A named case of :func:`leaf_profile_after_n`."""
    kinds = np.empty(4, dtype=np.int64)
    cards = np.empty(4, dtype=np.int16)
    kinds[0] = kind_a
    cards[0] = card_a
    kinds[1] = kind_b
    cards[1] = card_b
    # REFERENCE path: the real pile, reversed, because the leaf consumes from the end. The search does NOT
    # come through here -- it passes a hypothesised order to `leaf_profile_after_n` directly.
    return leaf_profile_after_n(state, player, 2, kinds, cards, hand_p, hand_o, state.deck[::-1], weights)


@njit(cache=True)
def leaf_profile_after_ply3(state, player: int, kind_a: int, card_a: int, kind_b: int,
                            card_b: int, kind_c: int, card_c: int, hand_p, hand_o,
                            weights) -> float:
    """The depth-2 leaf: three plies. A named case of :func:`leaf_profile_after_n`."""
    kinds = np.empty(6, dtype=np.int64)
    cards = np.empty(6, dtype=np.int16)
    kinds[0], cards[0] = kind_a, card_a
    kinds[1], cards[1] = kind_b, card_b
    kinds[2], cards[2] = kind_c, card_c
    return leaf_profile_after_n(state, player, 3, kinds, cards, hand_p, hand_o, state.deck[::-1], weights)


# --- depth as a parameter ------------------------------------------------------------------------
#
# The two leaves above are one body with their plies written out, and the two ply enumerations are one
# body with a fixed number of leading plies. Verified EXACTLY equal before being relied on: 360/360
# two-ply and 272/272 three-ply comparisons, no ULP of difference. See IMPROVEMENTS.md 57.
#
# Generalising is only safe because the ply sequence is determined: `player` moves, the opponent answers,
# `player` moves again, alternating. The parity of the ply count says whose turn it is and which hand to
# reason about, so no role has to be threaded through the call.


@njit(cache=True)
def leaf_profile_after_n(state, player: int, nplies: int, kinds: np.ndarray, cards: np.ndarray,
                         hand_p, hand_o, hypo_deck, weights) -> float:
    """`evaluate`'s value after `nplies` plies: `player`, the opponent, `player`, and so on.

    The semantics are those of the written-out bodies, and each was carried over rather than reinvented:

    * a ply that empties a hand ends the game at once, so the rest of the sequence never happens and no
      refill occurs -- the win check fires the moment a card leaves a hand, before any answer and before
      six fresh cards could return;
    * a ply that ends the round stops the sequence;
    * the refill happens once, at the end, and only if the round ended;
    * `moved_p` and `moved_o` accumulate per ACTOR, not per parity of the ply index.

    An empty sequence is not meaningful and returns 0.0: every call site has at least one ply.
    """
    if nplies <= 0:
        return 0.0

    trump = state.trump
    mine, size_p, theirs, size_o, table, table_size, defenses, defense_count, deck, deck_len, \
        player_attacks = _prepare(state, player, hand_p, hand_o, hypo_deck)

    moved_p = np.zeros(3, dtype=np.int64)
    moved_o = np.zeros(3, dtype=np.int64)

    round_ends = False
    round_end_kind = 0
    for ply in range(nplies):
        mine_turn = (ply % 2) == 0
        round_ends, game_over, size_p, size_o, table_size, defense_count = _apply_ply(
            kinds[ply], cards[ply], player_attacks if mine_turn else not player_attacks, mine_turn,
            mine, size_p, theirs, size_o, trump, table, table_size, defenses, defense_count,
            moved_o if mine_turn else moved_p)
        round_end_kind = kinds[ply]
        if game_over:
            return _finish(mine, size_p, theirs, size_o, trump, player, deck_len, player_attacks,
                           weights, moved_p, moved_o)
        if round_ends:
            break

    if round_ends:
        size_p, size_o, deck_len, player_attacks = _refill_and_roles(
            mine, size_p, theirs, size_o, deck, deck_len, player_attacks, round_end_kind)

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

    # No transfer correction is applied here. One used to be, to stop a card changing hands from reading
    # as progress. It was masking a bug: the TAKE branch credited the picked-up cards to the OPPONENT, so
    # the transfer was scored backwards, and the correction was derived from that same wrong attribution.
    # With the attribution fixed, taking is already penalised -- the defender's hand grows and the
    # opponent's shrinks, which lowers `cards_opponent - cards_player` -- and undoing the transfer only
    # made taking FREE again, which is why the search never beat anything.
    #
    # `moved_p` / `moved_o` stay as parameters: they record which cards changed hands and for how long, and
    # they let the differential test assert EXACT equality with the engine again rather than a residual.

    value = weights[0] * (stats_o[0] - stats_p[0])
    value += weights[1] * (stats_p[1] - stats_o[1])
    value += weights[2] * (stats_p[2] - stats_o[2])
    value += weights[3] * (1.0 if player_attacks else -1.0)
    if deck_len == 0:
        value += weights[4] * (stats_p[0] - stats_o[0])
    return value



@njit(cache=True)
def _throwin_legal_in(card: int, table, table_size: int, defenses, defense_count: int,
                      defender_size: int, defense_ranks: bool) -> bool:
    """:func:`rules._throwin_legal` over local arrays.

    Three conditions, all of which the engine applies: every card on the table must already be answered,
    the table may not grow past ``min(defender's hand, 6)`` *attack* cards, and the card's rank must
    match a rank already on the table (or, under ``throwin_defense_ranks``, one on a defense).
    """
    if defense_count < table_size:
        return False

    limit = defender_size
    if limit > THROWIN_CAP:
        limit = THROWIN_CAP
    if table_size + 1 > limit:
        return False

    card_rank = rank_of(card)
    for i in range(table_size):
        if rank_of(int(table[i])) == card_rank:
            return True
    if defense_ranks:
        for i in range(defense_count):
            if rank_of(int(defenses[i])) == card_rank:
                return True
    return False


@njit(cache=True)
def _write_throwins(hand, size: int, table, table_size: int, defenses, defense_count: int,
                    defender_size: int, defense_ranks: bool, out_kinds, out_cards) -> int:
    """Every legal throw-in in `hand`, in hand order. The engine emits them in hand order too."""
    count = 0
    for i in range(size):
        card = int(hand[i])
        if _throwin_legal_in(card, table, table_size, defenses, defense_count, defender_size, defense_ranks):
            out_kinds[count] = THROW
            out_cards[count] = card
            count += 1
    return count


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


# --- enumerating the search's own plies, state-free -----------------------------
#
# `rules.legal_actions` cannot be called here: it needs a `GameState`, and a jitclass cannot be built
# inside an njit function reached from Python. So the two enumerations the depth-2 search needs -- the
# opponent's answers, and the acting player's own next moves -- are written out over the local arrays that
# `_prepare` already built. `tests/test_leaf.py` diffs both against the engine over random positions, which
# is what makes reimplementing the rules here safe rather than reckless.



@njit(cache=True)
def plies_after(state, player: int, nplies: int, in_kinds: np.ndarray, in_cards: np.ndarray,
                hand_p, hand_o, hypo_deck, out_kinds, out_cards) -> int:
    """The legal moves of whoever is to move after `nplies` plies. Returns how many.

    One function rather than a `second_plies` and a `third_plies`, because they differ only in how many
    plies precede them. Parity picks the actor: after an odd count it is the opponent, after an even
    count it is `player`. That is what makes the search's mutual recursion possible at all.

    Covers all four shapes the rules distinguish, which the two written-out enumerations between them did
    NOT: at `nplies == 0` or after a refill the table is empty and the attacker opens with any card. A
    depth of four reaches that case on its fifth ply, so omitting it would have produced a search that
    silently cannot open a round -- a plausible-looking policy with a blind spot.
    """
    trump = int(state.trump)
    defense_ranks = bool(state.throwin_defense_ranks)
    mine, size_p, theirs, size_o, table, table_size, defenses, defense_count, deck, deck_len, \
        player_attacks = _prepare(state, player, hand_p, hand_o, hypo_deck)

    moved_p = np.zeros(3, dtype=np.int64)
    moved_o = np.zeros(3, dtype=np.int64)

    for ply in range(nplies):
        mine_turn = (ply % 2) == 0
        round_ends, game_over, size_p, size_o, table_size, defense_count = _apply_ply(
            in_kinds[ply], in_cards[ply], player_attacks if mine_turn else not player_attacks, mine_turn,
            mine, size_p, theirs, size_o, trump, table, table_size, defenses, defense_count,
            moved_o if mine_turn else moved_p)
        if round_ends or game_over:
            return 0

    # Parity of the count already applied says whose turn it is, and the hand to reason about.
    next_is_player = (nplies % 2) == 0
    next_hand = mine if next_is_player else theirs
    next_size = size_p if next_is_player else size_o
    defender_size = size_o if next_is_player else size_p
    next_attacks = player_attacks if next_is_player else not player_attacks

    count = 0
    if next_attacks:
        if table_size == 0:
            # Opening a fresh round, which is reachable only once a refill has emptied the table. Any
            # card in hand opens it.
            for i in range(next_size):
                out_kinds[count] = ATTACK
                out_cards[count] = int(next_hand[i])
                count += 1
            return count
        count = _write_throwins(next_hand, next_size, table, table_size, defenses, defense_count,
                                defender_size, defense_ranks, out_kinds, out_cards)
        if table_size == defense_count:
            out_kinds[count] = STOP
            out_cards[count] = 0
            count += 1
        return count

    # The defender owes an answer to the newest unanswered attack.
    target = int(table[defense_count])
    for i in range(next_size):
        card = int(next_hand[i])
        if beats(card, target, trump):
            out_kinds[count] = BEAT
            out_cards[count] = card
            count += 1
    out_kinds[count] = TAKE
    out_cards[count] = 0
    return count + 1


@njit(cache=True)
def second_plies(state, player: int, kind_a: int, card_a: int, hand_p, hand_o, out_kinds,
                 out_cards) -> int:
    """The opponent's answers to `player`'s first action. The depth-1 case of :func:`plies_after`.

    Kept as a named entry point rather than folded into the general one because the tests name it, and a
    test that reads as "the opponent's replies equal the rules" is worth more than a call with a literal.
    """
    kinds = np.empty(2, dtype=np.int64)
    cards = np.empty(2, dtype=np.int16)
    kinds[0] = kind_a
    cards[0] = card_a
    return plies_after(state, player, 1, kinds, cards, hand_p, hand_o, state.deck[::-1],
                       out_kinds, out_cards)


@njit(cache=True)
def third_plies(state, player: int, kind_a: int, card_a: int, kind_b: int, kind_b_card: int,
                hand_p, hand_o, out_kinds, out_cards) -> int:
    """`player`'s own next moves. The depth-2 case of :func:`plies_after`."""
    kinds = np.empty(4, dtype=np.int64)
    cards = np.empty(4, dtype=np.int16)
    kinds[0], cards[0] = kind_a, card_a
    kinds[1], cards[1] = kind_b, kind_b_card
    return plies_after(state, player, 2, kinds, cards, hand_p, hand_o, state.deck[::-1],
                       out_kinds, out_cards)
