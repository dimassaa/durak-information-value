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

from durak.leaf import (
    leaf_profile_after,
    leaf_profile_after_ply3,
    reply_cards,
    second_plies,
    third_plies,
)

from durak import rules
from durak.actions import (
    ATTACK,
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

    assert leaf_profile_after(pristine, 0, THROW, transferred, TAKE, 0, mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9), (
            "the leaf must match the engine EXACTLY after a take"
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

    A take is worth exactly ONE card against a discard on the count term, and that is CORRECT rather than
    a defect: the opponent picked up a card, so they now have one more to shed before they can win, while
    the thrower has shed one and moved closer to an empty hand.

    This test exists because the gap used to be -2.0 -- taking looked worth two cards -- which is what a
    take scores if the picked-up cards are credited to the player rather than the opponent. It pins the
    sign and the magnitude, and the differential tests against the engine pin the same thing end to end.
    """
    state, throw, beat, mine, theirs = _transfer_fixture()
    weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    taken = leaf_profile_after(state, 0, THROW, throw, TAKE, 0, mine, theirs, weights)
    discarded = leaf_profile_after(state, 0, THROW, throw, BEAT, beat, mine, theirs, weights)

    # The take scores +1 against the discard's 0: the opponent is one card further from empty.
    assert taken - discarded == pytest.approx(1.0, abs=1e-9), (
        f"a take must be worth exactly one more card than a discard, got {taken - discarded:+.4f}"
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


# --- depth 2: the third ply, and the two enumerations that reach it ------------
#
# Depth 2 changes the search's *answers*, not just its cost, so the same standard applies to it as to depth
# 1: play the transition through the engine and through the pure arithmetic, and demand the identical
# number. The rule under test below is that after `player` throws and the opponent beats, the opponent's
# beat does not end the round -- so `player` is at their next turn and their third move is a throw-in or a
# stop. A depth-2 leaf that scored the position one ply early would be wrong by exactly the difference
# between "cards committed to the table" and "cards in hand", which is invisible in a win rate.


def _three_ply_fixture():
    """A fully answered table, so both throw-ins and a stop are legal for the attacker.

    Trump is suit 3. The attacker holds a rank-6 trump and a rank-6 off-suit card, both legal throw-ins
    against the rank-6 attack already on the table, plus a card of a rank nothing matches. The defender
    holds two suit-0 cards that beat the off-suit throw-in and cannot touch the trump.

    Returns the state and the two hand arrays, so the pure call starts from the same position the engine
    does -- `rules.apply_action` mutates in place, so the two must be twin states.
    """
    def build():
        return legal_state(
            [card_at(6, 3), card_at(6, 0), card_at(0, 1)],
            [card_at(7, 0), card_at(8, 0)],
            trump=3, attacker=0,
            table_attacks=[card_at(6, 1)], table_defenses=[card_at(5, 2)],
        )

    pristine = build()
    mine, theirs = _hypothesised_hands(pristine)
    return pristine, mine, theirs


def test_the_third_ply_leaf_matches_the_engine_after_a_throw_a_beat_and_a_stop():
    """THROW, then BEAT, then STOP: the round ends on the third ply, and the table reaches the discard.

    The depth-2 counterpart of the depth-1 differential tests. Exact equality, and it is worth noticing
    what makes it a strong assertion: the third ply is a *stop*, so the profile depends on the discard, the
    refill draw order and the role flip all at once. A leaf that stopped early -- at ply two, or by treating
    the beat as a round end -- would still produce a plausible number here, and a different one.
    """
    pristine, mine, theirs = _three_ply_fixture()
    throw, beat = card_at(6, 0), card_at(7, 0)

    after = rules.apply_action(_twin_of(pristine), pack_action(THROW, throw), T)
    after = rules.apply_action(after, pack_action(BEAT, beat), T)
    after = rules.apply_action(after, pack_action(STOP, 0), T)

    assert leaf_profile_after_ply3(pristine, 0, THROW, throw, BEAT, beat, STOP, 0, mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9), (
            "the depth-2 leaf disagrees with the engine after throw/beat/stop"
        )


def test_the_third_ply_leaf_matches_the_engine_after_a_throw_a_beat_and_a_throw():
    """The same three plies ending in a throw-in rather than a stop.

    Included because the two endings are not symmetric in the arithmetic. A stop discards and refills; a
    second throw-in commits another card to the table and refills *nothing*. If one of the two were
    mishandled -- most plausibly the refill firing on a non-ending ply -- only this pair of tests would
    notice, because either alone looks correct.
    """
    pristine, mine, theirs = _three_ply_fixture()
    throw, beat, again = card_at(6, 0), card_at(7, 0), card_at(6, 3)

    after = rules.apply_action(_twin_of(pristine), pack_action(THROW, throw), T)
    after = rules.apply_action(after, pack_action(BEAT, beat), T)
    after = rules.apply_action(after, pack_action(THROW, again), T)

    assert leaf_profile_after_ply3(pristine, 0, THROW, throw, BEAT, beat, THROW, again, mine, theirs, W) \
        == pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9), (
            "the depth-2 leaf disagrees with the engine after throw/beat/throw"
        )


def test_the_third_ply_leaf_matches_the_engine_for_a_defender_answering_a_throw_in():
    """The other direction: `player` defends, the attacker throws in, and `player` must answer.

    The defending third ply is the one that can end the round in the taker's favour -- a TAKE here is the
    move that empties nothing and grows the taker's hand -- so it is the branch where a depth-2 leaf most
    easily reintroduces the transfer defect Stage 12 fixed. Both third-ply replies are asserted, so the
    beat and the take are pinned separately rather than averaged.

    Built so that every ply is forced to be legal, which is the whole difficulty of a hand-built fixture
    here. Trump is suit 0 and the table holds one *unanswered* rank-0 attack in suit 1:

    * seat 0 defends and beats with a trump, which is the only kind of card that beats it;
    * seat 1's throw-in must match a rank already on the table, so it is another rank-0 card, in suit 2 so
      that it is off-trump and the third ply is a real decision;
    * seat 0 still holds a trump, so the third ply offers BOTH a beat and a take, and neither assertion
      below is vacuous.

    An earlier fixture put an equal number of attacks and defenses on the table, so no beat was legal at
    all and `apply_action` raised -- a fixture failing for a reason that had nothing to do with the leaf.
    """
    def build():
        return legal_state(
            [card_at(0, 0), card_at(1, 0)],   # the defender: two trumps
            [card_at(0, 2), card_at(2, 0)],   # the attacker: one throw-in, one trump
            trump=0, attacker=1,
            table_attacks=[card_at(0, 1)], table_defenses=[],
        )

    state, pristine = build(), build()
    beat, throw_in, answer = card_at(0, 0), card_at(0, 2), card_at(1, 0)

    # The fixture must actually reach a third ply: `player` beats, the attacker throws in, and `player` owes
    # an answer. A fixture that ends earlier would make both assertions below vacuous.
    probe = rules.apply_action(rules.apply_action(_twin_of(state), pack_action(BEAT, beat), T),
                               pack_action(THROW, throw_in), T)
    assert probe.winner == rules.EMPTY
    assert {action_kind(int(a)) for a in rules.legal_actions(probe, 0)} == {BEAT, TAKE}

    mine, theirs = _hypothesised_hands(pristine)

    # Each engine run gets its own twin: `apply_action` mutates in place, so reusing one `state` here would
    # make the second assertion compare against a position that had already taken.
    taken = leaf_profile_after_ply3(pristine, 0, BEAT, beat, THROW, throw_in, TAKE, 0, mine, theirs, W)
    taken_after = rules.apply_action(rules.apply_action(_twin_of(state), pack_action(BEAT, beat), T),
                                     pack_action(THROW, throw_in), T)
    taken_after = rules.apply_action(taken_after, pack_action(TAKE, 0), T)
    assert taken == pytest.approx(_evaluate_profile(profile(taken_after, 0)), abs=1e-9), (
        "the depth-2 leaf disagrees with the engine on a third-ply take"
    )

    beaten = leaf_profile_after_ply3(pristine, 0, BEAT, beat, THROW, throw_in, BEAT, answer,
                                    mine, theirs, W)
    beaten_after = rules.apply_action(rules.apply_action(_twin_of(state), pack_action(BEAT, beat), T),
                                     pack_action(THROW, throw_in), T)
    beaten_after = rules.apply_action(beaten_after, pack_action(BEAT, answer), T)
    assert beaten == pytest.approx(_evaluate_profile(profile(beaten_after, 0)), abs=1e-9), (
        "the depth-2 leaf disagrees with the engine on a third-ply beat"
    )


def test_a_third_ply_that_empties_a_hand_ends_the_game_before_any_refill():
    """The win check fires the moment a card leaves a hand, at ply three as at any other.

    Without this the leaf would refill the winner and the profile would score a won game as a position
    with six fresh cards -- a plausible endgame, and not an error the win-rate numbers would surface.
    """
    def build():
        # The defender holds TWO cards, not one. A one-card defender empties their hand on the beat, which
        # ends the game at ply two by the rules, and a third ply applied to a finished game is not a legal
        # continuation -- the first version of this fixture did exactly that and disagreed by exactly the
        # count term. The deck is empty so that a stray refill would hand the winner six fresh cards and
        # change the profile.
        held = (card_at(0, 1), card_at(6, 0), card_at(7, 0), card_at(8, 0), card_at(6, 1), card_at(5, 2))
        return legal_state(
            [card_at(0, 1), card_at(6, 0)],
            [card_at(7, 0), card_at(8, 0)],
            trump=3, attacker=0,
            table_attacks=[card_at(6, 1)], table_defenses=[card_at(5, 2)],
            discard=[c for c in range(36) if c not in held],
        )

    state, pristine = build(), build()
    throw, beat = card_at(0, 1), card_at(7, 0)

    after = rules.apply_action(_twin_of(pristine), pack_action(THROW, throw), T)
    after = rules.apply_action(after, pack_action(BEAT, beat), T)
    after = rules.apply_action(after, pack_action(THROW, card_at(6, 0)), T)

    assert after.winner == 0, "the fixture must actually empty the attacker's hand"
    assert after.deck_len == 0, (
        "the fixture needs an empty draw pile, or the refill could mask the missing win check"
    )

    mine, theirs = _hypothesised_hands(pristine)
    assert leaf_profile_after_ply3(pristine, 0, THROW, throw, BEAT, beat, THROW, card_at(6, 0),
                                   mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9)


def test_the_third_ply_leaf_is_a_superset_of_the_depth_one_leaf():
    """When the second ply ends the round, the third ply does not happen and the two numbers are equal.

    This is what lets `action_value` call the depth-2 leaf unconditionally and fall back to the depth-1 one
    when the ply enumerations return nothing, with no depth flag to keep in step. Asserted on the pair the
    search actually reaches: a throw that is taken.
    """
    pristine, mine, theirs = _three_ply_fixture()
    throw = card_at(6, 0)

    assert third_plies(pristine, 0, THROW, throw, TAKE, 0, pristine.hands[0], theirs, *_buffers()) == 0
    assert leaf_profile_after_ply3(pristine, 0, THROW, throw, TAKE, 0, STOP, 0, mine, theirs, W) == \
        pytest.approx(leaf_profile_after(pristine, 0, THROW, throw, TAKE, 0, mine, theirs, W), abs=1e-9)


def _buffers():
    """Two empty output buffers, sized as `search.action_value` sizes its own."""
    kinds = np.empty(rules.HAND_CAPACITY + 2, dtype=np.int64)
    cards = np.empty(rules.HAND_CAPACITY + 2, dtype=np.int16)
    return kinds, cards


def test_the_second_ply_enumeration_equals_the_rules_when_attacking():
    """The defender's answers to an attack: every beater, plus taking.

    Depth 1 enumerated this; depth 2 moved it into `second_plies` so the search has one enumeration for both
    plies. Over random positions, the pure answer set must equal `rules.legal_actions`' for the state after
    the same first action.

    Twin states, because `rules.apply_action` mutates in place and the pure call must see the position
    *before* the action. Positions where the first action ends the game are skipped: there is no second ply
    to enumerate, and comparing against a finished state's empty action list would report a difference that
    is not one.
    """
    checked = 0
    for seed in range(400):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        if state.winner != rules.EMPTY or int(state.attacker) != 0:
            continue
        offered = [int(a) for a in rules.legal_actions(state, 0)
                   if action_kind(int(a)) in (ATTACK, THROW)]
        if not offered:
            continue
        action = offered[seed % len(offered)]

        after = rules.apply_action(_twin_of(state), action, T)
        if after.winner != rules.EMPTY:
            continue
        legal = {(action_kind(int(a)), action_card(int(a)))
                 for a in rules.legal_actions(after, 1)}
        kinds, cards = _buffers()
        n = second_plies(state, 0, action_kind(action), action_card(action), state.hands[0],
                         state.hands[1], kinds, cards)
        pure = {(int(kinds[i]), int(cards[i])) for i in range(n)}
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure second-ply enumeration disagrees on the answers to {action}"
        )

    assert checked > 50, f"only {checked} of 400 positions were attacking positions"


def _twin_of(state):
    """A state with the same hands, table, deck and role, for the engine to mutate.

    Built field by field rather than by replaying the game, so the twin is cheap and the two cannot diverge
    on anything except the action under test.
    """
    twin = rules.empty_state(int(state.initial_deck))
    twin.deck = state.deck.copy()
    twin.deck_len = state.deck_len
    twin.initial_deck = state.initial_deck
    twin.trump = state.trump
    twin.trump_card = state.trump_card
    twin.hands = state.hands.copy()
    twin.hand_sizes = state.hand_sizes.copy()
    twin.discard = state.discard.copy()
    twin.discard_len = state.discard_len
    twin.table_attacks = state.table_attacks.copy()
    twin.table_attack_count = state.table_attack_count
    twin.table_defenses = state.table_defenses.copy()
    twin.table_defense_count = state.table_defense_count
    twin.attacker = state.attacker
    twin.round_idx = state.round_idx
    twin.winner = state.winner
    twin.redeals = state.redeals
    twin.both_have_trumps = state.both_have_trumps
    twin.first_attacker_random_fallback = state.first_attacker_random_fallback
    twin.flag_redeals_exceeded = state.flag_redeals_exceeded
    twin.last_take_voluntary = state.last_take_voluntary
    twin.throwin_defense_ranks = state.throwin_defense_ranks
    # The three streams are `uint64` fields, so they must be re-wrapped as `np.uint64` on the way in: a
    # plain Python int above 2**63-1 does not fit the field and raises. Same masking rule as
    # `rng.game_stream_states`, and for the same reason -- a wrapped stream would fold as a negative
    # number and the twin would no longer be a twin.
    twin.rng_state = np.uint64(state.rng_state)
    twin.rng_p0 = np.uint64(state.rng_p0)
    twin.rng_p1 = np.uint64(state.rng_p1)
    return twin


def test_the_second_ply_enumeration_equals_the_rules_when_defending():
    """The attacker's throw-ins after a beat, plus stopping -- the branch depth 1 assumed away.

    Depth 1 hard-coded "the attacker stops" here, so this enumeration did not exist. It has to match the
    rules exactly, because it decides which replies the search averages over, and a missing throw-in makes
    the opponent look passive.
    """
    checked = 0
    for seed in range(400):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        if state.winner != rules.EMPTY or int(state.attacker) != 1:
            continue
        if state.table_attack_count <= state.table_defense_count:
            continue
        target = int(state.table_attacks[state.table_defense_count])
        beats = [a for a in rules.legal_actions(state, 0) if action_kind(int(a)) == BEAT]
        if not beats:
            continue
        beat = beats[seed % len(beats)]

        after = rules.apply_action(_twin_of(state), beat, T)
        if after.winner != rules.EMPTY:
            continue
        legal = {(action_kind(int(a)), action_card(int(a)))
                 for a in rules.legal_actions(after, 1)}
        kinds, cards = _buffers()
        n = second_plies(state, 0, action_kind(int(beat)), action_card(int(beat)), state.hands[0],
                         state.hands[1], kinds, cards)
        pure = {(int(kinds[i]), int(cards[i])) for i in range(n)}
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure second-ply enumeration disagrees on the attacker's answers to a beat"
        )

    assert checked > 50, f"only {checked} of 400 positions were defending positions"


def test_the_third_ply_enumeration_equals_the_rules_after_a_throw_and_a_beat():
    """The attacker's own next move: throw-ins that the rules allow, plus stopping.

    Compared against `rules.legal_actions` on the state after the same two actions. The failure that
    matters is a *missing* throw-in: it would drop a line the search is entitled to consider, and the
    search would look like it had considered it and found it worse.
    """
    checked = 0
    for seed in range(400):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        if state.winner != rules.EMPTY or int(state.attacker) != 0:
            continue
        offered = [int(a) for a in rules.legal_actions(state, 0)
                   if action_kind(int(a)) in (ATTACK, THROW)]
        if not offered:
            continue
        first = offered[seed % len(offered)]
        engine = _twin_of(state)
        after_first = rules.apply_action(engine, first, T)
        if after_first.winner != rules.EMPTY:
            continue
        answers = rules.legal_actions(after_first, 1)
        beats = [int(a) for a in answers if action_kind(int(a)) == BEAT]
        if not beats:
            continue
        reply = beats[seed % len(beats)]

        after = rules.apply_action(after_first, reply, T)
        if after.winner != rules.EMPTY:
            continue
        legal = {(action_kind(int(a)), action_card(int(a)))
                 for a in rules.legal_actions(after, 0)}
        kinds, cards = _buffers()
        n = third_plies(state, 0, action_kind(first), action_card(first), action_kind(reply),
                        action_card(reply), state.hands[0], state.hands[1], kinds, cards)
        pure = {(int(kinds[i]), int(cards[i])) for i in range(n)}
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure third-ply enumeration disagrees after {first} answered by {reply}"
        )

    assert checked > 50, f"only {checked} of 400 positions reached a third ply"


def test_the_third_ply_enumeration_equals_the_rules_after_a_beat_and_a_throw_in():
    """The defender's own next move: every beater of the new attack, plus taking.

    The mirror of the test above, and the branch where taking lives. Taking is the move the search was
    measured taking 21% of the time, so an enumeration that wrongly offered it where the rules forbid it --
    or withheld it where they allow it -- is exactly the defect this stage is looking for.
    """
    checked = 0
    # 800 seeds, not 400: this branch needs a position where the attacker can legally throw in AFTER the
    # defender has answered, which is a minority of the positions a random walk reaches. 400 seeds yielded
    # 46 comparisons -- enough to pass on a fixture that happened to work and too few to catch an
    # enumeration that loses one throw-in in ten.
    for seed in range(800):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        if state.winner != rules.EMPTY or int(state.attacker) != 1:
            continue
        if state.table_attack_count <= state.table_defense_count:
            continue
        beats = [int(a) for a in rules.legal_actions(state, 0)
                 if action_kind(int(a)) == BEAT]
        if not beats:
            continue
        beat = beats[seed % len(beats)]
        engine = _twin_of(state)
        after_beat = rules.apply_action(engine, beat, T)
        if after_beat.winner != rules.EMPTY:
            continue
        throws = [int(a) for a in rules.legal_actions(after_beat, 1)
                  if action_kind(int(a)) in (ATTACK, THROW)]
        if not throws:
            continue
        throw_in = throws[seed % len(throws)]

        after = rules.apply_action(after_beat, throw_in, T)
        if after.winner != rules.EMPTY:
            continue
        legal = {(action_kind(int(a)), action_card(int(a)))
                 for a in rules.legal_actions(after, 0)}
        kinds, cards = _buffers()
        n = third_plies(state, 0, action_kind(beat), action_card(beat), action_kind(throw_in),
                        action_card(throw_in), state.hands[0], state.hands[1], kinds, cards)
        pure = {(int(kinds[i]), int(cards[i])) for i in range(n)}
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure third-ply enumeration disagrees after a beat and a throw-in"
        )

    assert checked > 80, f"only {checked} of 800 positions reached a defending third ply"
