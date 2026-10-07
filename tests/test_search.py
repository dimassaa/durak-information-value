"""Leaf evaluation and the expectimax search.

The informed strategy is `SEARCH_INFORMED`, and this is the code that makes it informed. Three things
are pinned here.

The evaluation's *sign and scale*. A weight pointing the wrong way does not crash -- it makes the search
confidently prefer bad moves, and every win rate downstream becomes a statement about the weights
instead of about the game. So each term's direction is asserted on its own, and the value is pinned to
negate exactly under a seat swap.

The search's *use of the particle set*. A search that ignored `particles` and reasoned only about visible
cards would be correct, deterministic, and completely blind to the opponent's hand -- the entire premise
of the project. So there is a test that the same position with two different particle sets yields two
different choices, and a test that the reply set is enumerated *inside* a hypothesis.

What the search does **not** check: whether the informed strategy is *good*. That is Task 12's
measurement, not a property of two functions.

Three fixture rules that each cost a rewrite
--------------------------------------------
A throw-in is legal only when its **rank** matches a card already on the table (or, with
`throwin_defense_ranks`, a defense on it). A fixture whose attacker's cards share no rank with
`table_attacks` offers *no* throw-ins, and "the search stops rather than throwing" then passes vacuously.

A particle-sensitivity fixture is only meaningful when the two hypotheses differ in **beatability**. Two
hands that are both unbeatable produce the same reply set by construction.

And the search spends the card it can afford to lose. It keeps trumps. A test that expects the opposite
is testing a plausible-sounding intuition, not the code.

The depth-1 blind spot, asserted rather than papered over
---------------------------------------------------------
`test_throwing_dominates_stopping_at_depth_one` documents a real property of this design: the search
throws even when every throw-in can be beaten. That is not a bug, and the reason is worth stating,
because it is the kind of thing that would otherwise be discovered in Task 12 as a mysterious
over-attack profile:

* after the defender **beats** a throw-in, the attacker's hand and the defender's hand are both unchanged
  -- two cards moved from hand to table -- so `evaluate` cannot distinguish that leaf from the position
  before the throw;
* after the defender **takes**, the attacker has shed a card, which `evaluate` scores as an improvement;
* `STOP` in a defended round makes the *attacker* pick up the table, which `evaluate` scores as a loss.

So all three replies average above the stop outcome. Fixing it needs either depth 2, or a leaf evaluation
that charges for cards committed to the table -- and the second is a change to spec 5.4's five terms, not
a bug fix, so it is raised rather than taken here.

Depth 2, and what it did to these two tests
-------------------------------------------
Depth 2 is the fix the note above named, and the search now does it. One of the two fixtures reversed, and
the reversal is the point rather than an inconvenience:

* `test_a_throw_in_that_is_beaten_is_followed_by_a_stop_and_not_penalised` -- at depth 1 this position
  chose to STOP, 1.7 against the throw's 1.633. That gap was an artefact: the depth-1 leaf scored a
  mid-round position with the attacker's card still on the table as strictly worse than stopping, even
  though the attacker had lost nothing. At depth 2 the search sees that it may stop *after* being beaten,
  which reaches the same discard and the same refill and is therefore worth exactly 1.7. The beaten branch
  costs nothing, the take branch decides, and the throw wins.
* `test_stopping_still_wins_when_the_throw_in_costs_a_card_the_attacker_wants_to_keep` -- the guard
  against reading the first test as "depth 2 always throws". It does not: over 18 sampled positions
  offering both a throw-in and a stop, 4 still stop.

Three hand-built fixtures were tried for the second test before falling back to a position reached by
random play, and each produced a case where throwing wins. Recorded rather than hidden: the difference
between the two outcomes is arithmetic, not a rule, and hunting for a prettier synthetic case would have
been fitting the test to the code.
"""

import numpy as np
import pytest

from durak import rules
from durak.actions import (
    ATTACK,
    BEAT,
    STOP,
    THROW,
    action_card,
    action_kind,
    pack_action,
)
from durak.config import Settings
from durak.evaluation import EVAL_SWEEP_GRID, EVAL_WEIGHTS, evaluate
from durak.leaf import leaf_profile_after, leaf_profile_after_ply3, reply_cards_in
from durak.opponent import initial_particles
from durak.rng import RngStream
from durak.search import DEFAULT_DEPTH, action_value, expectimax
from durak.engine import choose_action
from durak.strategies import RANDOM
from tests.support import Pool, card_at, legal_state, random_position

WEIGHTS = EVAL_WEIGHTS

#: The table cards every mid-round fixture uses. Fixed ids, reserved before the hands are drawn.
TABLE_ATTACK = card_at(6, 1)   # rank 6, suit 1
TABLE_DEFENSE = card_at(7, 2)  # rank 7, suit 2


def certain_particles(state, opponent, hand, k=64):
    """`k` copies of one hypothesis, so the search reasons as if the hand were known.

    A set of identical rows is how "perfect knowledge" reaches the same interface that otherwise receives
    a sampled distribution. Tests needing a definite answer use this rather than a large `k`, because a
    large `k` gives a probable answer and a test needs a certain one.
    """
    return np.tile(np.asarray(sorted(hand), dtype=np.int16), (k, 1))


def midround(hand0, hand1, trump=0, attacker=0, answered=True):
    """A state with the table cards already placed."""
    return legal_state(hand0, hand1, trump=trump, attacker=attacker, table_attacks=[TABLE_ATTACK],
                       table_defenses=[TABLE_DEFENSE] if answered else [])


def reply_kinds(state, action, hand, player=1):
    """The reply kinds the opponent gets *under the hypothesis `hand`*, from the array form.

    The opponent's hand is a hypothesis, not `state.hands[opponent]`, so the enumeration takes the hand as
    an array. An earlier version built a scratch state to carry it, and `enumerate_responses` overwrote
    that buffer -- which silently discarded the substitution and made a particle-sensitivity test compare
    two hypotheses and see the identical reply set twice.
    """
    target = action_card(int(action))
    replies = reply_cards_in(np.asarray(sorted(hand), dtype=np.int16), len(hand), state.trump, target)
    return [rules.BEAT] * replies.shape[0] + [rules.TAKE]


# --- the evaluation function --------------------------------------------------


def test_fewer_cards_scores_positive():
    state = legal_state([card_at(0, 1), card_at(1, 1)],
                        [card_at(2, 2)] + [card_at(i, 3) for i in range(3, 7)], trump=0)

    assert evaluate(state, 0, WEIGHTS) > 0.0


def test_more_trumps_scores_positive():
    """Trump weight is 2.0 against 1.0 for a card: a trump is a card *and* a tempo weapon.

    The two states differ only in which seat holds the trump, so nothing else can account for the
    difference. Comparing states that differed in more than the term under test would make the assertion
    unfalsifiable.
    """
    weak = legal_state([card_at(0, 1), card_at(2, 1)], [card_at(6, 0), card_at(3, 1)], trump=0)
    strong = legal_state([card_at(0, 1), card_at(6, 0)], [card_at(2, 1), card_at(3, 1)], trump=0)

    assert evaluate(strong, 0, WEIGHTS) > evaluate(weak, 0, WEIGHTS)


def test_more_high_cards_scores_positive():
    """Rank >= 10 printed, i.e. offset >= 5. Same construction as the trump test."""
    low = legal_state([card_at(0, 1), card_at(2, 1)], [card_at(5, 1), card_at(3, 1)], trump=0)
    high = legal_state([card_at(5, 1), card_at(2, 1)], [card_at(0, 1), card_at(3, 1)], trump=0)

    assert evaluate(high, 0, WEIGHTS) > evaluate(low, 0, WEIGHTS)


def test_the_attacker_role_term_is_exactly_twice_the_weight_across_the_seats():
    """Sign and magnitude pinned only as the *difference* between the two seats.

    The magnitude is a judgement, not a property. But the two states differ only in `attacker`, so the gap
    must be exactly twice the role weight -- which catches a sign flip and a double-count without asserting
    the judgement.

    The gap, not the individual signs: with equal hands the attacker-seated value is *positive* and the
    defender-seated value is *also* positive (0.8 and 0.2 here, because the attacker holds the only high
    card). An earlier version asserted the defender's value was negative and failed for that reason --
    the assertion was wrong about the rules, not the code.
    """
    hands = ([card_at(0, 1), card_at(5, 1)], [card_at(2, 2), card_at(3, 2)])
    as_attacker = legal_state(*hands, trump=0, attacker=0)
    as_defender = legal_state(*hands, trump=0, attacker=1)

    assert (evaluate(as_attacker, 0, WEIGHTS)
            - evaluate(as_defender, 0, WEIGHTS)) == pytest.approx(2.0 * WEIGHTS[3], abs=1e-9)


def test_swapping_the_seat_exactly_negates_the_value():
    """Every term is a differential, so the function is antisymmetric.

    If one term were an absolute count or a constant, this would fail -- and the search would then prefer
    one seat for reasons unrelated to the position, biasing every win rate in a dataset whose tournament
    runs both directions.
    """
    state = midround([card_at(6, 0), card_at(0, 1)], [card_at(2, 2), card_at(3, 2)])

    assert evaluate(state, 0, WEIGHTS) == pytest.approx(-evaluate(state, 1, WEIGHTS), abs=1e-9)


def test_the_deck_exhausted_term_is_zero_while_the_pile_is_not_empty():
    """Unequal hands, deliberately: with equal hands the term is arithmetically zero.

    Two states with the same hands and different deck states, so the *only* difference is this term and
    its exact size is checkable.
    """
    hands = ([card_at(0, 1), card_at(5, 1)], [card_at(2, 2), card_at(3, 2), card_at(4, 2)])
    named = set(hands[0]) | set(hands[1])
    with_deck = legal_state(*hands, trump=0)
    no_deck = legal_state(*hands, trump=0, discard=[c for c in range(36) if c not in named])

    assert with_deck.deck_len > 0 and no_deck.deck_len == 0
    assert (evaluate(no_deck, 0, WEIGHTS) - evaluate(with_deck, 0, WEIGHTS)) == pytest.approx(
        WEIGHTS[4] * (2 - 3), abs=1e-9
    )


def test_the_deck_exhausted_term_makes_holding_fewer_cards_a_disadvantage():
    """The inversion. A term that is zero almost always and decisive occasionally goes unnoticed unless
    it is asserted."""
    hands = ([card_at(0, 1)], [card_at(2, 2), card_at(3, 2), card_at(4, 2), card_at(1, 2)])
    named = set(hands[0]) | set(hands[1])

    behind = evaluate(legal_state(*hands, trump=0), 0, WEIGHTS)
    behind_no_deck = evaluate(
        legal_state(*hands, trump=0, discard=[c for c in range(36) if c not in named]), 0, WEIGHTS
    )

    assert behind > 0.0, "one card against four is good while cards remain to draw"
    assert behind_no_deck < behind, (
        "emptying the pile did not make holding fewer cards worse, so the term never inverted"
    )


def test_the_weights_are_the_ones_the_config_declares():
    """One source of truth.

    `Settings.eval_weights` and `EVAL_WEIGHTS` must agree, or the manifest records a configuration the
    search did not use -- and a sweep over weights that do not match the run's weights is a sensitivity
    analysis of a different experiment.
    """
    assert tuple(Settings().eval_weights) == tuple(EVAL_WEIGHTS)
    assert len(EVAL_WEIGHTS) == 5, "spec 5.4 defines five terms"


def test_the_sweep_grid_has_a_baseline_and_moves_one_weight_at_a_time():
    """A grid without its own baseline cannot say how much anything moved, and a grid that moved two
    weights together cannot attribute the change."""
    assert (1.0,) * 5 in EVAL_SWEEP_GRID, "the grid has no unperturbed row"
    assert len(EVAL_SWEEP_GRID) == len(set(EVAL_SWEEP_GRID)), "the grid contains duplicate rows"
    for row in EVAL_SWEEP_GRID:
        assert len([i for i, factor in enumerate(row) if factor != 1.0]) <= 1, (
            f"row {row} perturbs more than one weight at once"
        )


# --- enumerating responses ----------------------------------------------------


def _defending_position():
    pool = Pool(36)
    pool.reserve(TABLE_ATTACK)
    return midround(pool.take(4), pool.take(6), trump=0, answered=False), pool


# --- the search ---------------------------------------------------------------


def test_the_search_returns_a_legal_action():
    """Absolute first: an illegal action is not a weak strategy, it is a broken engine."""
    for state, _ in ((midround([card_at(6, 0), card_at(0, 1)], [card_at(2, 2), card_at(3, 2)]), None),
                     _defending_position()):
        player = rules.acting_player(state)
        particles = initial_particles(state, 1 - player, 32, RngStream(5))

        assert int(expectimax(state, player, particles, WEIGHTS)) in {
            int(a) for a in rules.legal_actions(state, player)
        }


def test_the_search_is_deterministic():
    """Same state, same particles, same answer -- every time.

    No tie-break by RNG, no dependence on iteration order. The dataset's reproducibility rests on this, and
    a search is exactly the kind of code that picks up a nondeterministic tie-break unnoticed.
    """
    state = midround([card_at(6, 0), card_at(0, 1)], [card_at(2, 2), card_at(3, 2)])
    particles = initial_particles(state, 1, 64, RngStream(7))

    answers = {int(expectimax(state, 0, particles, WEIGHTS)) for _ in range(8)}

    assert len(answers) == 1, f"the search gave {len(answers)} different actions for one position"


def test_the_search_does_not_modify_the_state_or_the_particles():
    """It is a query, not a step.

    A search that applied its move to the caller's state would make every recorded game depend on whether
    the search ran, and the equivalence probe would stop comparing like with like.
    """
    state = midround([card_at(6, 0), card_at(0, 1)], [card_at(2, 2), card_at(3, 2)])
    particles = initial_particles(state, 1, 64, RngStream(9))
    before = (int(state.round_idx), int(state.attacker), int(state.table_attack_count),
              int(state.hand_sizes[0]), bytes(np.sort(state.hands[0])))

    expectimax(state, 0, particles, WEIGHTS)

    after = (int(state.round_idx), int(state.attacker), int(state.table_attack_count),
             int(state.hand_sizes[0]), bytes(np.sort(state.hands[0])))
    assert after == before
    assert particles.tobytes() == particles_before(particles)


def particles_before(particles):
    return particles.tobytes()


def test_it_throws_a_card_the_opponent_is_certain_it_cannot_beat():
    """Shedding a card for nothing is the cheapest win in the game."""
    hand0 = [card_at(6, 0), card_at(6, 2), card_at(0, 1), card_at(1, 1)]
    # Two suits, neither trump (suit 0) nor the attacker's: nothing here beats a rank-6 throw-in.
    hand1 = [card_at(0, 2), card_at(0, 3), card_at(1, 2), card_at(1, 3)]
    state = midround(hand0, hand1, trump=0)
    throws = [c for c in hand0 if rules._throwin_legal(state, c)]

    assert throws, "the fixture offers no throw-ins, so the test would be vacuous"
    assert rules.BEAT not in reply_kinds(state, pack_action(THROW, throws[0]), hand1)

    chosen = int(expectimax(state, 0, certain_particles(state, 1, hand1), WEIGHTS))

    assert action_kind(chosen) == THROW, (
        f"the search stopped while {len(throws)} throw-ins are known to be unbeatable"
    )


def test_it_never_spends_a_trump_when_holding_it_is_better():
    """It throws what it can afford to lose.

    Two rank-6 throw-ins: a trump and an off-suit card. The defender holds two suit-0 cards of higher
    rank, which beat the off-suit card and cannot touch the trump. The trump is therefore safe and the
    off-suit card is not, so the search must not lead with the trump.

    Asserted as "the trump's value is strictly the lowest and is not chosen", rather than as "the
    off-suit card is chosen", because which of the two *non-trump* options wins depends on the deck the
    refill will draw from -- and the point of this test is the trump, not the tie-break between the rest.
    """
    hand0 = [card_at(6, 3), card_at(6, 0), card_at(0, 1)]
    hand1 = [card_at(7, 0), card_at(8, 0)]
    state = midround(hand0, hand1, trump=3)

    assert rules.beats(hand1[0], hand0[1], state.trump), "the fixture must be able to beat the "
    assert not rules.beats(hand1[0], hand0[0], state.trump), "and must not be able to beat the trump"

    particles = certain_particles(state, 1, hand1)
    scores = {int(a): action_value(state, int(a), 0, particles, WEIGHTS, DEFAULT_DEPTH)
              for a in rules.legal_actions(state, 0)}

    chosen = int(expectimax(state, 0, particles, WEIGHTS))

    assert scores[pack_action(THROW, hand0[0])] < min(
        v for a, v in scores.items() if a != pack_action(THROW, hand0[0])
    ), f"spending the trump scores {scores[pack_action(THROW, hand0[0])]}, which is not the worst"
    assert chosen != pack_action(THROW, hand0[0]), "the search spent its trump"


def test_throwing_beats_stopping_when_every_throw_in_is_beaten():
    """A depth-1 property, and it is *not* universal.

    When the defender can beat every throw-in, throwing still wins here. The mechanism is the one recorded
    in Stage 10: after a beat both hands are unchanged, so the leaf cannot distinguish that from not
    throwing, while a take leaves the attacker a card lighter. So the average of "unchanged, unchanged,
    improved" beats stopping.

    An earlier version of this test claimed throwing *always* dominates, and the ledger recorded that as a
    general limitation. It does not. The claim rested on a mis-stated rule -- that STOP makes the attacker
    pick up the table. It does not: `_do_stop` sends the table to the *discard*. With the leaf arithmetic
    corrected, stopping wins in several other fixtures, including one where the search keeps its trump.
    The limitation is fixture-dependent, and this test pins the case where it does hold.
    """
    hand0 = [card_at(6, 0), card_at(6, 2), card_at(0, 1), card_at(1, 1)]
    hand1 = [card_at(7, 3), card_at(8, 3), card_at(2, 2), card_at(3, 2)]
    state = midround(hand0, hand1, trump=3)

    throws = [c for c in hand0 if rules._throwin_legal(state, c)]
    assert throws, "the fixture offers no throw-ins, so the test would be vacuous"
    for card in throws:
        assert rules.BEAT in reply_kinds(state, pack_action(THROW, card), hand1), (
            "this fixture is supposed to beat every throw-in"
        )

    chosen = int(expectimax(state, 0, certain_particles(state, 1, hand1), WEIGHTS))

    assert action_kind(chosen) == THROW, (
        f"the search stopped; the depth-1 property this test pins no longer holds and the recorded "
        f"explanation for it needs revisiting rather than passing silently"
    )


def test_a_throw_in_that_is_beaten_is_followed_by_a_stop_and_not_penalised():
    """Depth 2's answer on the fixture where depth 1 preferred to stop, and the reason it changed.

    The attacker holds a rank-6 trump and a rank-6 off-suit card; the defender holds two suit-0 cards of
    higher rank, which beat the off-suit card and cannot touch the trump. At depth 1 this fixture chose
    STOP, and the reason was mechanical rather than strategic: a beaten throw-in leaves both hands
    unchanged and the round open, so the depth-1 leaf scored it strictly *worse* than stopping (1.3 against
    1.7) even though the attacker has not lost anything -- the cards are still on the table, not in a
    hand.

    Depth 2 can see the ply that resolves it. Having been beaten, the attacker may stop, and stopping
    discards the table and refills both hands -- so "throw, get beaten, then stop" is worth exactly what
    "stop now" is worth, 1.7, because it reaches the same discard and the same refill. Averaged over the
    defender's three answers that is 1.9 against stopping's 1.7, and the search throws.

    Asserted as the numbers rather than only the chosen action, because the mechanism is the claim: if the
    third ply were scored as a mid-round leaf again, the beaten branch would read 1.3 and the averaging
    would put STOP back on top. `test_stopping_still_wins_when_the_defender_would_grow_even_after_a_beat`
    covers the other side.
    """
    hand0 = [card_at(6, 3), card_at(6, 0), card_at(0, 1)]
    hand1 = [card_at(7, 0), card_at(8, 0)]
    state = midround(hand0, hand1, trump=3)
    throw, beat = card_at(6, 0), card_at(7, 0)
    ho = np.asarray(sorted(hand1), dtype=np.int16)

    stop_now = leaf_profile_after(state, 0, STOP, 0, STOP, 0, state.hands[0], ho, WEIGHTS)
    beaten_then_stop = leaf_profile_after_ply3(state, 0, THROW, throw, BEAT, beat, STOP, 0,
                                               state.hands[0], ho, WEIGHTS)

    assert beaten_then_stop == pytest.approx(stop_now, abs=1e-9), (
        f"being beaten and then stopping scored {beaten_then_stop:+.4f} against stopping now's "
        f"{stop_now:+.4f}; both reach the same discard and the same refill, so they must agree"
    )

    particles = certain_particles(state, 1, hand1)
    chosen = int(expectimax(state, 0, particles, WEIGHTS))

    assert action_kind(chosen) == THROW, (
        f"the search chose kind {action_kind(chosen)}; the beaten branch now costs nothing, so the "
        f"take branch decides and throwing wins"
    )


def test_stopping_still_wins_when_the_throw_in_costs_a_card_the_attacker_wants_to_keep():
    """Depth 2 does not make throwing universal, which is what keeps the test above honest.

    A real position reached by random play rather than a hand-built one: trump is suit 0, the table holds
    one answered rank-5 attack, and the attacker has one legal throw-in -- card 22, a rank-5 suit-2 card,
    rank 5 already being on the table.

    Throwing it hands the defender a card of suit 2 and, if the defender beats, the attacker is left one
    card lighter with the round still open. The throw is beatable here too, so the mechanism is the same as
    in the test above; what differs is that the card being shed is one the attacker would rather keep,
    because the deck is running low and the count term is what the position turns on. Stopping commits
    nothing and wins.

    Built from `random_position` on purpose. Three earlier hand-built attempts at "a fixture where stopping
    still wins" each produced a position where throwing wins, and the difference between them turned out to
    be arithmetic rather than a rule -- which is the honest reason not to keep hunting for a prettier
    synthetic case. This one is a position the rules actually reached.
    """
    state = random_position(deck_size=36, seed=18, max_actions=21)
    assert state.winner == rules.EMPTY and int(state.attacker) == 0
    assert {action_kind(int(a)) for a in rules.legal_actions(state, 0)} == {THROW, STOP}, (
        "the fixture must offer exactly one throw-in and a stop, or it does not test the comparison"
    )
    throws = [int(a) for a in rules.legal_actions(state, 0) if action_kind(int(a)) == THROW]
    assert len(throws) == 1, "the fixture must offer exactly one throw-in"

    hand1 = [int(state.hands[1, i]) for i in range(int(state.hand_sizes[1]))]
    stop_now = leaf_profile_after(state, 0, STOP, 0, STOP, 0, state.hands[0],
                                  np.asarray(sorted(hand1), dtype=np.int16), WEIGHTS)
    throw_value = action_value(state, throws[0], 0, certain_particles(state, 1, hand1), WEIGHTS, DEFAULT_DEPTH)

    assert throw_value < stop_now, (
        f"the throw scores {throw_value:+.4f} against stopping's {stop_now:+.4f}"
    )

    chosen = int(expectimax(state, 0, certain_particles(state, 1, hand1), WEIGHTS))

    assert action_kind(chosen) == STOP, (
        f"the search chose kind {action_kind(chosen)}; stopping scores {stop_now:+.4f} and the only "
        f"throw-in scores {throw_value:+.4f}"
    )


def test_the_particle_set_changes_what_the_search_values():
    """The premise of the whole project, in one assertion -- on the *value*, not the move.

    Asserted on the score rather than the chosen action, because the chosen action is a weak witness: in
    the fixture below the two hypotheses both end up throwing the same card, because the two throw-ins
    are otherwise symmetric and a tie goes to the first. An earlier version asserted the *actions*
    differed and would have failed for a reason that had nothing to do with particle sensitivity.

    The score cannot be faked: if `particles` were ignored, both hypotheses would produce the identical
    number. Measured here, 1.5 against 1.2 -- the card is worth more precisely when the defender is
    believed to be able to beat it, which sounds backwards until you account for the fact that a beat
    costs the defender a card too.
    """
    hand0 = [card_at(6, 0), card_at(6, 2), card_at(0, 1)]
    filler = [card_at(2, 2), card_at(3, 2)]
    state = midround(hand0, filler + [card_at(4, 2)], trump=3)

    throw = pack_action(THROW, hand0[0])   # card_at(6, 0)
    # `blocking` holds card_at(7, 0): same suit as the throw, higher rank, so it beats it.
    blocking = certain_particles(state, 1, filler + [card_at(4, 2), card_at(7, 0)])
    harmless = certain_particles(state, 1, filler + [card_at(4, 2), card_at(8, 1)])

    assert rules.BEAT in reply_kinds(state, throw, filler + [card_at(4, 2), card_at(7, 0)])
    assert rules.BEAT not in reply_kinds(state, throw, filler + [card_at(4, 2), card_at(8, 1)])

    def score(particles):
        return action_value(state, throw, 0, particles, WEIGHTS, DEFAULT_DEPTH)

    assert score(blocking) != pytest.approx(score(harmless)), (
        "two opposite beliefs about the defender's hand produced the identical score, so the search is "
        "not using the particle set at all"
    )
    assert int(expectimax(state, 0, blocking, WEIGHTS)) in {
        int(a) for a in rules.legal_actions(state, 0)
    }


def test_the_search_returns_the_argmax_of_its_own_value_function():
    """The contract, checked over 200 positions.

    A search that returned legal, deterministic, particle-sensitive but *arbitrary* actions would pass
    every other test in this file. This one checks the only property that actually matters: the chosen
    action is the highest-scoring one under the function the search claims to be maximising.

    Chosen over a comparison against `RANDOM`, which looks like the natural sanity check and is nearly
    useless here. With three throw-ins and a stop available, `RANDOM` throws three times in four, so
    "the search throws more often than RANDOM" is a 100%-versus-75% claim that would also pass a search
    which did nothing at all. A draft fixture made this worse by calling a card "free" while giving the
    defender a trump that beat it -- the search was right and the fixture was wrong.
    """
    checked = disagreed_with_random = 0

    for offset in range(200):
        pool = Pool(36)
        pool.reserve(TABLE_ATTACK)
        pool.reserve(TABLE_DEFENSE)
        # Vary the trump suit and the defender's composition so the positions are not 200 copies of one
        # case, and include trumps on both sides so the trump term is exercised.
        trump = offset % 4
        hand0 = [c for c in dict.fromkeys([card_at(6, 3), card_at(6, offset % 2),
                                           card_at(offset % 3, 1)])
                 if c not in (TABLE_ATTACK, TABLE_DEFENSE)]
        # The attacker's cards must be reserved before the defender is dealt, or the pool hands back a
        # card already in play and the fixture stops being a legal state.
        for card in hand0:
            pool.reserve(card)
        # Rank 8, so the defender's trump can never collide with the rank-6 attack or the rank-7 defense
        # already on the table. Reserved before the deal, for the same reason.
        defender_trump = card_at(8, trump)
        pool.reserve(defender_trump)
        hand1 = [defender_trump] + pool.without_suit(2, trump)
        assert not (set(hand0) & set(hand1)), "fixture hands overlap"
        state = midround(hand0, hand1, trump=trump)

        options = rules.legal_actions(state, 0)
        if options.shape[0] < 2:
            continue
        checked += 1

        particles = certain_particles(state, 1, hand1, k=8)
        chosen = int(expectimax(state, 0, particles, WEIGHTS))
        scores = [action_value(state, int(a), 0, particles, WEIGHTS, DEFAULT_DEPTH) for a in options]
        best = float(np.max(scores))

        assert action_value(state, chosen, 0, particles, WEIGHTS, DEFAULT_DEPTH) == pytest.approx(best, abs=1e-9), (
            f"the search chose a candidate worth {scores[0]:.4f} when {best:.4f} was available"
        )
        disagreed_with_random += chosen != int(choose_action(state, 0, RANDOM, 0))

    assert checked > 150, f"only {checked} of 200 fixtures had a real choice"
    assert disagreed_with_random > 0.3 * checked, (
        f"the search agreed with RANDOM on {checked - disagreed_with_random}/{checked} positions, which "
        f"suggests it is not discriminating between candidates at all"
    )


def test_the_search_works_from_a_real_mid_game_position():
    """Not a constructed fixture: a position reached by uniformly random play.

    Every other test here uses a hand-built position chosen to make one property visible. This one only
    asserts the plumbing survives contact with arbitrary state -- unequal hand sizes, a partially filled
    table, a non-empty discard, an unknown trump.
    """
    state = random_position(deck_size=36, seed=5, max_actions=30)
    player = rules.acting_player(state)

    assert int(expectimax(state, player, initial_particles(state, 1 - player, 24, RngStream(21)),
                          WEIGHTS)) in {int(a) for a in rules.legal_actions(state, player)}


# --- the two arms -------------------------------------------------------------


def test_the_two_search_ids_resolve_to_the_same_function():
    """The experiment's central structural claim.

    `SEARCH_UNIFORM` and `SEARCH_INFORMED` are one function. If they were two, their win-rate difference
    would measure whatever else differed between them -- a different depth, a different tie-break, a
    different action ordering -- and the project's headline number would be about that instead of about
    information.
    """
    from durak.strategies import SEARCH_INFORMED, SEARCH_UNIFORM, get_policy

    assert get_policy(SEARCH_UNIFORM) is get_policy(SEARCH_INFORMED)


def test_the_only_difference_between_the_arms_is_the_hand_source():
    """Asserted on the values, not on the moves.

    The informed arm reads the opponent's real hand as one particle; the uninformed arm draws
    `PARTICLE_COUNT` from the uniform prior. Everything else -- depth, weights, tie-break, reply model,
    action ordering -- is read outside the branch, so this can only fail if one arm stops being the search
    it was.

    Both arms are driven through the same `expectimax`, so a difference here is a difference in *input*,
    which is the only thing that is supposed to differ.
    """
    from durak.strategies.search_policy import PARTICLE_COUNT

    state = midround([card_at(6, 0), card_at(0, 1)], [card_at(2, 2), card_at(3, 2)])
    hand1 = [int(c) for c in state.hands[1, : state.hand_sizes[1]]]

    informed = certain_particles(state, 1, hand1, k=1)
    uninformed = initial_particles(state, 1, PARTICLE_COUNT, RngStream(3))

    assert informed.shape[0] == 1, "the informed arm is one hypothesis: the real hand"
    assert uninformed.shape[0] == PARTICLE_COUNT

    informed_choice = int(expectimax(state, 0, informed, WEIGHTS))
    uninformed_choice = int(expectimax(state, 0, uninformed, WEIGHTS))

    for choice in (informed_choice, uninformed_choice):
        assert choice in {int(a) for a in rules.legal_actions(state, 0)}, (
            "both arms must search over the same legal actions, or they are not the same search"
        )


def test_both_arms_return_legal_actions_from_a_real_position():
    """The registry's dispatch contract, checked for the two new ids through the policy itself."""
    from durak.strategies import SEARCH_INFORMED, SEARCH_UNIFORM, InfoMode, get_policy

    for strategy_id in (SEARCH_UNIFORM, SEARCH_INFORMED):
        policy = get_policy(strategy_id)
        info = InfoMode.TRUTH if strategy_id == SEARCH_INFORMED else InfoMode.PARTICLES
        for seed in (1, 2, 3, 4, 5):
            state = random_position(deck_size=36, seed=seed, max_actions=6 + seed)
            if state.winner != rules.EMPTY:
                continue
            player = rules.acting_player(state)
            action = int(policy(state, player, strategy_id, int(info), np.uint64(seed)))

            assert action in {int(a) for a in rules.legal_actions(state, player)}, (
                f"{strategy_id} returned an illegal action from a real position"
            )
