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
pytest.importorskip("durak.leaf", reason="durak.leaf is not implemented yet")

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
    """ATTACK then BEAT: one card each leaves a hand, the table holds both, nothing is refilled."""
    state = legal_state([card_at(6, 1), card_at(0, 1), card_at(5, 1)],
                        [card_at(4, 2), card_at(3, 2), card_at(6, 0)], trump=3, attacker=0)
    attack = card_at(0, 1)

    for reply in [c for c in range(36)
                  if rules.beats(c, attack, state.trump)
                  and c in {int(x) for x in state.hands[1, :state.hand_sizes[1]]}]:
        after = rules.apply_action(state, pack_action(THROW, attack), T)
        after = rules.apply_action(after, pack_action(BEAT, reply), T)

        mine, theirs = _hypothesised_hands(state, after)
        assert leaf_profile_after(state, 0, THROW, attack, BEAT, reply, mine, theirs, W) == \
            pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9), (
            f"the leaf disagrees with the engine after a beat by {reply}"
        )


def test_the_leaf_transition_matches_the_engine_for_an_attack_and_a_take():
    """TAKE: the defender picks up the table and becomes the attacker, then both refill."""
    state = legal_state([card_at(6, 1), card_at(0, 1), card_at(5, 1)],
                        [card_at(4, 2), card_at(3, 2)], trump=3, attacker=0)

    after = rules.apply_action(state, pack_action(THROW, card_at(0, 1)), T)
    after = rules.apply_action(after, pack_action(TAKE, 0), T)

    mine, theirs = _hypothesised_hands(state, after)
    assert leaf_profile_after(state, 0, THROW, card_at(0, 1), TAKE, 0, mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9)


def test_the_leaf_transition_matches_the_engine_for_stop():
    """STOP discards the table and swaps the roles -- the opposite of TAKE on both counts."""
    state = legal_state([card_at(6, 1), card_at(0, 1), card_at(5, 1)],
                        [card_at(4, 2), card_at(3, 2), card_at(7, 2)], trump=3, attacker=0,
                        table_attacks=[card_at(2, 1)], table_defenses=[card_at(3, 0)])

    after = rules.apply_action(state, pack_action(STOP, 0), T)

    mine, theirs = _hypothesised_hands(state, after)
    assert leaf_profile_after(state, 0, STOP, 0, STOP, 0, mine, theirs, W) == \
        pytest.approx(_evaluate_profile(profile(after, 0)), abs=1e-9)


def _hypothesised_hands(state, after):
    """The hands the pure transition starts from.

    The search knows only its own hand and a hypothesis about the opponent's, so the differential test
    has to feed it the *same* two arrays the engine started from -- not the engine's result, which would
    make the comparison vacuous.
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
    for seed in range(200):
        state = random_position(deck_size=36, seed=seed, max_actions=2 + seed % 25)
        if state.table_attack_count <= state.table_defense_count:
            continue
        target = int(state.table_attacks[state.table_defense_count])

        legal = {action_card(int(a)) for a in rules.legal_actions(state, 1)
                 if action_kind(int(a)) == BEAT}
        pure = set(reply_cards(state, 1, target).tolist())
        checked += 1

        assert pure == legal, (
            f"seed {seed}: the pure enumeration disagrees on the replies to card {target}"
        )

    assert checked > 100, f"only {checked} of 200 positions had an unanswered attack"
