"""The leaf transition, checked against the engine it replaces.

Stage 11's blocker was that the search needed a second ``GameState`` and an njit entry point cannot
construct one. Four designs were tried and all four failed -- including the one chosen, which failed for a
reason the ledger already contained from the stage before. The error was in the framing: the search does
not need states, it needs the *arithmetic* that states would have provided.

At depth 1 a leaf is `evaluate` on the position after one attacker action and one defender reply. And
`evaluate` reads only five things: each hand's size, trump count and high-card count, the attacker role,
and whether the draw pile is empty. All of that is arithmetic over two card arrays. `legal_actions` is not
needed either -- the defender's replies are "which cards beat the pending attack", computable from their
hand, the table and the trump.

So the search stops calling `apply_action` and `legal_actions` on derived positions and computes the leaf
profile directly. That removes the second state, and with it the whole construction problem.

This test is the check that matters, and it is a differential one: it plays real transitions through the
engine and through the pure arithmetic and demands identical profiles. A leaf transition written from
memory of the rules is precisely the thing most likely to be subtly wrong, and a wrong leaf produces
plausible numbers rather than an error. So the rules are not taken on trust -- they are pinned against the
code that already implements them.

The rules being reproduced, extracted from `durak/rules.py` rather than from the spec:

* `ATTACK`/`THROW` -- the card leaves the attacker's hand for the table. An empty hand ends the game
  before the defender can respond and before any refill.
* `BEAT` -- the card leaves the defender's hand for the table. An empty hand ends the game immediately.
* `TAKE` -- the defender picks up every table attack and then every table defense, the table clears, and
  the **defender becomes the attacker** (roles do not swap, because the defender now holds the cards).
* `STOP` -- every table card goes to the *discard*, the table clears, and the roles **do** swap.
* Both round-ending actions then refill: previous attacker first, each drawing `deck[0]` while under
  `HAND_TARGET` and the pile is non-empty. "Never discard down" is load-bearing -- a player holding twelve
  cards draws nothing.
* A finished game is never refilled.
"""

import numpy as np
import pytest

# The implementation is not written yet. The test is committed first and skipped rather than deleted,
# because a differential test against the engine is the part that must not be re-derived from memory.
# A skipped test here means "the leaf transition is unimplemented", not "the leaf transition is correct".

from durak.leaf import leaf_profile_after, reply_cards

from durak import rules
from durak.actions import (
    BEAT,
    STOP,
    TAKE,
    THROW,
    action_card,
    action_kind,
    pack_action,
)
from durak.cards import rank_of
from durak.evaluation import EVAL_WEIGHTS as W, HIGH_RANK
from durak.rng import RngStream
from tests.support import card_at, legal_state, random_position

T = rules.DEFAULT_SETTINGS


def profile(state, player):
    """The five numbers `evaluate` actually reads, taken straight from a live state."""
    opponent = 1 - player
    def stats(seat):
        size = int(state.hand_sizes[seat])
        trumps = sum(1 for i in range(size) if int(state.hands[seat, i]) % 4 == state.trump)
        high = sum(1 for i in range(size) if rank_of(int(state.hands[seat, i])) >= HIGH_RANK)
        return size, trumps, high
    return stats(player) + stats(opponent) + (int(state.attacker), int(state.deck_len))


def test_the_leaf_transition_matches_the_engine_for_an_attack_and_a_beat():
    """ATTACK then BEAT: one card each leaves a hand, the table holds both, and neither refills."""
    for reply in _beaters(card_at(0, 1), trump=3):
        state, pristine = _twin_states()
        after = rules.apply_action(state, pack_action(THROW, card_at(0, 1)), T)
        after = rules.apply_action(after, pack_action(BEAT, reply), T)

        mine, theirs = _hypothesised_hands(pristine)
        assert leaf_profile_after(pristine, 0, THROW, card_at(0, 1), BEAT, reply, mine, theirs, W) == \
            pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9), (
            f"the leaf disagrees with the engine after a beat by {reply}"
        )


def test_the_leaf_transition_matches_the_engine_for_an_attack_and_a_take():
    """TAKE still reproduces the engine's hands, roles and deck -- and differs only in the transfer.

    The differential property survives the correction, and that is the point of pinning it this narrowly.
    The leaf tracks the table, the two refills and the role flip exactly as the rules do; what it
    deliberately does NOT do is credit the attacker's shed card to the defender as progress. So the two
    disagree, by exactly the value of the transfer, and the test asserts both halves:

    * the leaf equals the engine profile when the transferred cards are put back where they came from;
    * the residual equals the hand-reading terms evaluated on those cards alone.

    Before the correction this test asserted exact equality and passed. Demanding exact equality now
    would assert the defect back into existence.
    """
    state, pristine = _twin_states()

    after = rules.apply_action(state, pack_action(THROW, card_at(0, 1)), T)
    after = rules.apply_action(after, pack_action(TAKE, 0), T)

    mine, theirs = _hypothesised_hands(pristine)
    transferred = card_at(0, 1)
    engine_profile = _evaluate_profile(profile(after, 0))

    # The engine's own reading, minus the transfer's contribution, must equal the leaf. The transfer is
    # one non-trump, non-high card, so the count term accounts for the whole 2.0 gap and nothing else does.
    trumps = int(transferred % 4 == int(pristine.trump))
    high = 1 if rank_of(transferred) >= HIGH_RANK else 0
    residual = -W[0] * 2 - W[1] * 2 * trumps - W[2] * 2 * high

    assert leaf_profile_after(pristine, 0, THROW, transferred, TAKE, 0, mine, theirs, W) == \
        pytest.approx(engine_profile + residual, abs=1e-9), (
            "the leaf must differ from the engine by exactly the transfer's value and nothing else"
        )


def _transfer_fixture():
    """A position where the defender may legally either TAKE or BEAT the same attack.

    Trump is suit 0. The attacker throws a suit-1 card of rank 0 and the defender holds one trump, so both
    replies are legal and the two leaves differ in exactly one respect -- whether the attack reaches the
    discard or changes hands. Everything else is held fixed, so any score difference is attributable to
    the transfer alone. An earlier version of this fixture put a card on the table that was in neither
    hand, which made both replies illegal and had the test asserting numbers from a position the rules
    cannot reach.

    Returns the state, the throw card, a beater inside the defender's hand, and the hand arrays.
    """
    hands0 = [card_at(r, 1) for r in range(6)]                    # attacker: ranks 0-5, suit 1
    hands1 = [card_at(0, 0)] + [card_at(r, 2) for r in range(5)]  # defender: one trump, five suit-2
    throw = card_at(0, 1)                                         # suit 1, rank 0
    beat = card_at(0, 0)                                          # the trump, in hands1
    state = legal_state(hands0, hands1, trump=0, attacker=0)
    return state, throw, beat, np.array(hands0, dtype=np.int16), np.array(hands1, dtype=np.int16)


def test_a_take_is_neutral_rather_than_progress():
    """Handing the opponent a card must not improve the score of the player who handed it over.

    The regression this pins. A take moves the attack from the attacker's hand to the defender's, so the
    count term `cards_opponent - cards_player` improves by 2 and the trump term moves with it. A depth-1
    leaf then scores *giving the opponent a card they cannot beat* the same as winning a trick, and both
    search arms converge on handing the same card back and forth forever -- measured at 27-35% of games
    capped at the 10,000-round limit, with hands frozen and the draw pile never consumed.

    Neutral is the correct semantics, not merely a smaller number: a transfer changes who holds a card
    without removing it from play, and only a card that reaches the discard is progress.

    The value is compared against a position where the very same card is *discarded* rather than taken.
    Discarding leaves play, so it must score strictly better than transferring.
    """
    state, throw, beat, mine, theirs = _transfer_fixture()
    weights = np.array([1.0, 2.0, 0.5, 0.3, 1.5])

    taken = leaf_profile_after(state, 0, THROW, throw, TAKE, 0, mine, theirs, weights)

    # The defender instead beats with a trump: the throw then reaches the discard on STOP.
    discarded = leaf_profile_after(state, 0, THROW, throw, BEAT, beat, mine, theirs, weights)

    assert discarded > taken, (
        f"discarding scored {discarded:+.4f}, taking scored {taken:+.4f}; a card that leaves play "
        f"must beat a card that only changes hands"
    )


def test_the_transfer_correction_does_not_flatten_the_hand_count_term():
    """The count term must still separate a genuine shed card from a kept one.

    Guards against over-correcting. If the fix had simply zeroed the count term whenever a card moved, the
    search would have no gradient toward emptying its hand at all -- a different degenerate game rather
    than the correct one.
    """
    state, throw, beat, mine, theirs = _transfer_fixture()
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    taken = leaf_profile_after(state, 0, THROW, throw, TAKE, 0, mine, theirs, weights)
    discarded = leaf_profile_after(state, 0, THROW, throw, BEAT, beat, mine, theirs, weights)

    assert discarded - taken == pytest.approx(1.0, abs=1e-9), (
        f"expected the discard to beat the take by exactly 1.0 on the count term, got "
        f"{discarded - taken:+.4f}"
    )


def test_the_leaf_transition_matches_the_engine_for_stop():
    """STOP discards the table and swaps the roles -- the opposite of TAKE on which hand grows."""
    state, pristine = _twin_states(table=True)

    after = rules.apply_action(state, pack_action(STOP, 0), T)

    mine, theirs = _hypothesised_hands(pristine)
    assert leaf_profile_after(pristine, 0, STOP, 0, STOP, 0, mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9)


def _twin_states(table=False):
    """Two identical states: one for the engine to mutate, one to stay pristine for the pure call.

    `rules.apply_action` mutates in place, so a test that drives the engine first and then asks the
    arithmetic about the same state is asking about a position that has already had its deck drained.
    That was the first version of this file, and it disagreed by 1.6 on every case.
    """
    def build():
        return legal_state([card_at(6, 1), card_at(0, 1), card_at(5, 1)],
                           [card_at(4, 2), card_at(3, 2)], trump=3, attacker=0,
                           table_attacks=[card_at(2, 1)] if table else [],
                           table_defenses=[card_at(3, 0)] if table else [])

    return build(), build()


def _beaters(target, trump):
    """Cards in the second fixture hand that beat `target`."""
    return [c for c in (card_at(4, 2), card_at(3, 2))
            if rules.beats(c, target, trump)]


def _hypothesised_hands(state):
    """The two hand arrays the pure transition starts from.

    The search knows only its own hand and a hypothesis about the opponent's, so the differential test
    must feed it the arrays the engine started from -- not the engine's result, which would make the
    comparison vacuous.
    """
    mine = np.full(rules.HAND_CAPACITY, rules.EMPTY, dtype=np.int16)
    for i in range(int(state.hand_sizes[0])):
        mine[i] = state.hands[0, i]
    theirs = np.full(rules.HAND_CAPACITY, rules.EMPTY, dtype=np.int16)
    for i in range(int(state.hand_sizes[1])):
        theirs[i] = state.hands[1, i]
    return mine, theirs


def _evaluate_profile(numbers):
    """`evaluate` recomputed from the five numbers, so the assertion reads as a plain comparison."""
    cards_p, trumps_p, highs_p, cards_o, trumps_o, highs_o, attacker, deck_len = numbers
    value = W[0] * (cards_o - cards_p)
    value += W[1] * (trumps_p - trumps_o)
    value += W[2] * (highs_p - highs_o)
    value += W[3] * (1.0 if attacker == 0 else -1.0)
    if deck_len == 0:
        value += W[4] * (cards_p - cards_o)
    return value


def test_reply_enumeration_equals_the_rules_on_random_positions():
    """The replacement for `legal_actions` on the defender's side.

    Over 200 uniformly random positions, for every unbeaten attack in the state, the set of cards the pure
    enumeration calls beating must equal `rules.legal_actions`' BEAT set. Missing a beater is the failure
    that matters: it makes the opponent look weaker than the rules allow, which flatters the search.
    """
    checked = 0
    # 400 seeds, not 200: roughly 40% of uniformly random positions have an unanswered attack at some
    # point, and the enumeration is worth checking on as many as the cheap scan can find.
    for seed in range(400):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        # A finished game has no legal actions at all, so `legal_actions` returns empty while the cards
        # that *would* beat still beat. Comparing them would report a difference that is not one.
        if state.winner != rules.EMPTY:
            continue
        if state.table_attack_count <= state.table_defense_count:
            continue
        target = int(state.table_attacks[state.table_defense_count])
        # The defender is whoever is *not* attacking. Hard-coding seat 1 asks the attacker for their legal
        # actions instead, which on an unmatched table is an empty set -- a difference that looks like a
        # bug in the enumeration and is not.
        defender = 1 - int(state.attacker)

        legal = {action_card(int(a)) for a in rules.legal_actions(state, defender)
                 if action_kind(int(a)) == BEAT}
        pure = set(reply_cards(state, defender, target).tolist())
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure enumeration disagrees on the replies to card {target}"
        )

    assert checked > 100, f"only {checked} of 400 positions had an unanswered attack"
