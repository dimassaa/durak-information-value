"""Shared test fixtures for building exact game positions.

Two problems recur across the rules and strategy suites, and both produce tests that
assert nothing while passing.

**Overlapping card allocations.** Slicing the deck directly (``cards[0:6]``, cards of a
given suit) silently returns cards that another hand also contains. The resulting fixture
is a position the rules can never reach, and a test built on it pins a fiction. :class:`Pool`
hands out each card exactly once.

**Fixtures that are not legal states.** A hand plus a table plus a ``deck_len`` set by hand
does not conserve cards. An invariant assertion against such a fixture is meaningless, and
a ``deck_len`` larger than the ``deck`` array makes the engine write past the end of the
array, which corrupts the heap instead of raising. :func:`legal_state` fills the draw pile
from the unallocated remainder and asserts conservation at construction.

Both helpers live here rather than in one test module because the rules suite and the
strategy suite need identical guarantees; a second copy would eventually drift.
"""

import numpy as np

from durak import rules
from durak.cards import rank_of, suit_of

HAND_CAPACITY = rules.HAND_CAPACITY
EMPTY = rules.EMPTY


def card_at(rank, suit):
    """The card id of a (rank offset, suit) pair."""
    return rank * 4 + suit


class Pool:
    """Hands out distinct cards, tracking what has been taken.

    ``take`` returns the lowest unused cards; ``of_suit`` and ``without_suit`` restrict by
    suit. ``without_suit`` exists because a "neither player holds a trump" fixture built
    with ``take`` quietly contains trumps, which makes the test assert nothing.
    """

    def __init__(self, deck_size=36):
        self._available = set(range(deck_size))
        self._order = list(range(deck_size))

    def reserve(self, card):
        """Mark a card as used without taking it from the ordered pool."""
        assert card in self._available, f"card {card} is already reserved"
        self._available.discard(card)
        self._order = [c for c in self._order if c != card]
        return card

    def take(self, count):
        """The next `count` unused cards, in ascending order."""
        return self._pick(count, lambda card: True)

    def of_suit(self, count, suit):
        """`count` unused cards of one suit."""
        return self._pick(count, lambda card: suit_of(card) == suit)

    def without_suit(self, count, suit):
        """`count` unused cards that are *not* of the given suit."""
        return self._pick(count, lambda card: suit_of(card) != suit)

    def of_rank(self, count, rank):
        """`count` unused cards of one rank."""
        return self._pick(count, lambda card: rank_of(card) == rank)

    def _pick(self, count, predicate):
        picked = [c for c in self._order if c in self._available and predicate(c)][:count]
        assert len(picked) == count, f"not enough cards available for {count} picks"
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
    pile, so ``deck_len`` always matches the array it indexes.
    """
    named = [c for group in (hand0, hand1, table_attacks, table_defenses, discard) for c in group]
    assert len(set(named)) == len(named), f"a card was allocated twice: {sorted(named)}"

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


def table_with_free_ranks(pool, count, trump=0):
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


def random_position(deck_size=36, seed=0, max_actions=40):
    """A realistic mid-game position reached by uniformly random play.

    Constructed positions are precise but narrow: they exercise one rule in isolation and
    miss the interactions. Legality claims about a *policy* are only credible across real
    positions, so this walks forward from a real deal with the uniformly random policy
    already in the rules module.
    """
    from durak.rng import GameStreams

    state = rules.setup_game(deck_size, GameStreams.from_parts(seed, 0, 0))
    rng = np.uint64(seed * 7919 + 13)
    for _ in range(max_actions):
        if state.winner != EMPTY:
            break
        offered = rules.legal_actions(state, rules.acting_player(state))
        if offered.size == 0:
            break
        choice, rng = rules.draw_action(offered, rng)
        rng = np.uint64(rng)
        rules.apply_action(state, int(choice), rules.DEFAULT_SETTINGS)
    return state
