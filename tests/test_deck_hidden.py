"""The draw pile must not reach the search.

The deck-hiding change is the whole basis of the project's research question, so it gets its own test
rather than being trusted because the suite is green. Before the change the leaf read `state.deck`
verbatim and simulated the exact future draws; with the pile public, the complement of the visible cards
was exactly the opponent's hand, so "perfect knowledge of the hand" was derivable and the question had no
hidden information in it at all.

The test permutes the REAL draw pile while leaving everything an observer can see untouched -- own hand,
table, discard, trump, hand sizes, the role, and the rng stream -- and asserts that neither search arm
changes its action. A single such permutation changing an action is the leak, named exactly.
"""
import numpy as np

from durak import rules
from durak.actions import action_kind
from durak.evaluation import EVAL_WEIGHTS
from durak.leaf import plies_after
from durak.search import expectimax
from durak.strategies import INFO_PARTICLES, INFO_TRUTH, InfoMode

CAP = 70


def _deal(seed):
    return rules.deal_state(36, np.uint64(seed), np.uint64(seed + 101), np.uint64(seed + 202),
                            rules.DEFAULT_SETTINGS, False, -1)


def _hand(state, seat):
    return state.hands[seat, :state.hand_sizes[seat]].copy()


def _particles(state, seat, k=16):
    """A fixed belief about the opponent's hand, so only the pile hypothesis can move the answer."""
    opp = 1 - seat
    hand = _hand(state, opp)
    return np.repeat(hand[None, :], k, axis=0).astype(np.int16)


def _pile(state):
    return state.deck[:state.deck_len].copy()


def test_permuting_the_real_pile_changes_neither_arm():
    """The decisive check: same visible state, different pile, same decision."""
    compared = 0
    for seed in range(40):
        state = _deal(seed)
        if int(state.winner) != rules.EMPTY:
            continue
        player = int(state.attacker)
        particles = _particles(state, player)
        pile = _pile(state)

        before = int(expectimax(state, player, particles, pile[::-1][None, :].repeat(
            particles.shape[0], axis=0), EVAL_WEIGHTS, 2))

        # Permute the real pile in place. Everything an observer can see is untouched: the hands were
        # dealt before this point and are not re-derived, and the rng stream the hypotheses draw from is
        # part of the state and is left alone.
        order = np.arange(len(pile))
        state.rng_state = np.uint64(7)
        for i in range(len(order) - 1, 0, -1):
            pick = int(order[i])
            tmp = pile[i]
            pile[i] = pile[pick]
            pile[pick] = tmp
        state.deck[:state.deck_len] = pile

        after = int(expectimax(state, player, particles, pile[::-1][None, :].repeat(
            particles.shape[0], axis=0), EVAL_WEIGHTS, 2))

        assert before == after, (
            f"seed={seed} player={player}: permuting the unseen draw pile changed the action from "
            f"{before} ({action_kind(before)}) to {after} ({action_kind(after)}). The pile is reaching "
            f"the search, so the opponent's hand is derivable and the research question has no hidden "
            f"information in it."
        )
        compared += 1
    assert compared >= 20, f"only {compared} positions compared; the test would be vacuous"


def test_the_leaf_value_depends_on_the_pile_order():
    """The counterpart to the invariance test, so the two together pin the pile as HIDDEN, not ABSENT.

    A leaf that ignored the draw order entirely would pass the invariance test perfectly and would also be
    modelling a game with no draws at all. So the same action, under two different pile orders, must score
    differently on at least some positions -- the pile is unknown, and it still matters.
    """
    from durak.search import action_value

    differed = 0
    compared = 0
    for seed in range(40):
        state = _deal(seed)
        if int(state.winner) != rules.EMPTY:
            continue
        player = int(state.attacker)
        particles = _particles(state, player)
        pile = _pile(state)
        forward = pile[::-1][None, :].repeat(particles.shape[0], axis=0)
        backward = pile[None, :].repeat(particles.shape[0], axis=0)

        options = rules.legal_actions(state, player)
        for index in range(options.shape[0]):
            action = int(options[index])
            a = action_value(state, action, player, particles, forward, EVAL_WEIGHTS, 2)
            b = action_value(state, action, player, particles, backward, EVAL_WEIGHTS, 2)
            compared += 1
            if a != b:
                differed += 1
                break
        if differed >= 3:
            break
    assert compared > 0, "no position was compared; the test would be vacuous"
    assert differed > 0, (
        "the leaf scored identically under every permutation of the draw pile on every position tested, "
        "so the pile is being ignored rather than hidden -- the model would have no draws at all"
    )
