"""The four-ply leaf against the engine, and `plies_after` at three plies.

Both are differential: the leaf is compared against `rules.apply_action` applied the same number of
times, and the enumerator against `rules.legal_actions` in the resulting state. Nothing had ever
exercised a ply count above three, which is exactly where a loop can be wrong while every depth-1 and
depth-2 test still passes.

The fixture. Picking the attacker's first legal attack does not work: it is often unbeatable, the reply
set is then TAKE alone, the round ends there, and no line of any depth exists. So the attack is chosen to
be BEATABLE. Then BEAT, then STOP -- and STOP is what makes a fourth ply possible at all, because it ends
the round, so the refill runs and the roles swap. The fourth ply is therefore whatever the new attacker
does with an empty table, which is the `table_size == 0` branch that neither `second_plies` nor
`third_plies` ever had.

WHY IT IS A SEPARATE MODULE: the three-ply and four-ply fixtures need their own twin bookkeeping, and the
previous attempt to put them in the existing depth module produced a builder that did not reproduce inside
pytest while working by hand. Kept apart so a failure here is unambiguously about this file.
"""
import numpy as np
import pytest

import durak.rules as rules
from durak.actions import ATTACK, BEAT, STOP, THROW, pack_action
from durak.cards import beats, rank_of
from durak.evaluation import EVAL_WEIGHTS, HIGH_RANK
from durak.leaf import leaf_profile_after_n, plies_after

W = EVAL_WEIGHTS
CAP = 70
DEALS = 200


def _deal(seed):
    return rules.deal_state(36, np.uint64(seed), np.uint64(seed + 101), np.uint64(seed + 202),
                            rules.DEFAULT_SETTINGS, False, -1)


def _engine_value(state, player):
    """`evaluate`'s value read straight off a live state, in the leaf's own term order.

    Recomputed here rather than imported so the assertion reads as a plain comparison of two numbers
    rather than as a comparison of the leaf against itself through another function under test.
    """
    def stats(seat):
        size = int(state.hand_sizes[seat])
        trumps = sum(1 for i in range(size) if int(state.hands[seat, i]) % 4 == state.trump)
        high = sum(1 for i in range(size) if rank_of(int(state.hands[seat, i])) >= HIGH_RANK)
        return size, trumps, high

    cards_p, trumps_p, highs_p = stats(player)
    cards_o, trumps_o, highs_o = stats(1 - player)
    value = W[0] * (cards_o - cards_p) + W[1] * (trumps_p - trumps_o) + W[2] * (highs_p - highs_o)
    return value + W[3] * (int(state.attacker) == player) + W[4] * (int(state.deck_len) == 0)


def _legals(state, player):
    return [(rules.action_kind(int(a)), rules.action_card(int(a)))
            for a in rules.legal_actions(state, player)]


def _first_of(state, player, kinds):
    for k, c in _legals(state, player):
        if k in kinds:
            return k, c
    return None


def _beatable_attack(state, player, opponent):
    """An attack the opponent can beat, or None."""
    for k, c in _legals(state, player):
        if k != ATTACK:
            continue
        if any(beats(int(state.hands[opponent, i]), c, int(state.trump))
               for i in range(int(state.hand_sizes[opponent]))):
            return k, c
    return None


def _four_ply_line(seed):
    """ATTACK, BEAT, THROW, BEAT -- four plies INSIDE ONE ROUND. None if this deal has no such line.

    The round boundary is the whole point. A line is one round: a ply that ends the round or empties a
    hand ends the LINE, and the leaf stops there. That is not a shortcut, it is the model -- it is why the
    written-out depth-2 body could `break` on a round end and still match the engine, and why a draft of
    this test that ran ATTACK, BEAT, STOP, ATTACK found nothing: the STOP ends the line, so the fourth ply
    never happens. (An earlier attempt to make the leaf cross the boundary was reverted; the engine agrees
    with stopping, at `1.3` against `0.7`.)

    So the fourth ply has to be reached WITHOUT ending the round, which means the attacker's second move
    must be a throw-in that the opponent can beat. That is the shape a depth-4 search actually walks.
    """
    first = _deal(seed)
    player = int(first.attacker)
    opponent = 1 - player
    opening = _beatable_attack(first, player, opponent)
    if opening is None:
        return None

    # The opening attack must be APPLIED before the opponent owes an answer: on the untouched deal they
    # owe nothing and `_first_of` returns None for every reply, which reads as "no lines exist".
    engine = rules.apply_action(_deal(seed), pack_action(opening[0], opening[1]),
                                rules.DEFAULT_SETTINGS)
    kinds, cards = [opening[0]], [opening[1]]

    beat = _first_of(engine, opponent, (BEAT,))
    if beat is None:
        return None
    engine = rules.apply_action(engine, pack_action(beat[0], beat[1]), rules.DEFAULT_SETTINGS)
    kinds.append(beat[0])
    cards.append(beat[1])

    # The attacker's next move must be a THROW the opponent can beat. STOP would end the round and end the
    # line; a throw-in the opponent cannot beat ends it too, by forcing a take.
    throw = _first_of(engine, player, (THROW,))
    if throw is None:
        return None
    probe = rules.apply_action(engine, pack_action(throw[0], throw[1]), rules.DEFAULT_SETTINGS)
    if int(probe.winner) != rules.EMPTY:
        return None
    if not any(k == BEAT for k, _ in _legals(probe, opponent)):
        return None

    engine = probe
    kinds.append(throw[0])
    cards.append(throw[1])

    # The fourth ply is actually PLAYED, not merely checked for: the leaf is asked to score four plies,
    # so the comparison state has to have four plies too.
    fourth = _first_of(engine, opponent, (BEAT,))
    if fourth is None:
        return None
    engine = rules.apply_action(engine, pack_action(fourth[0], fourth[1]), rules.DEFAULT_SETTINGS)
    kinds.append(fourth[0])
    cards.append(fourth[1])
    if int(engine.winner) != rules.EMPTY:
        return None
    return first, engine, player, kinds, cards


def test_the_four_ply_leaf_matches_the_engine():
    """ATTACK, BEAT, THROW, BEAT: four plies of one round, the shape a depth-4 search walks.

    Nothing has ever exercised a ply count above three, and this is the count where a loop can be wrong
    while every depth-1 and depth-2 test still passes.
    """
    checked = 0
    for seed in range(DEALS):
        line = _four_ply_line(seed)
        if line is None:
            continue
        leaf_state, engine, player, kinds, cards = line
        hand_p = leaf_state.hands[player].copy()
        hand_o = leaf_state.hands[1 - player].copy()
        got = leaf_profile_after_n(leaf_state, player, 4,
                                   np.array(kinds, dtype=np.int64),
                                   np.array(cards, dtype=np.int16), hand_p, hand_o, leaf_state.deck[::-1], W)
        want = _engine_value(engine, player)
        assert got == pytest.approx(want, abs=1e-9), (
            f"seed={seed} kinds={kinds}: leaf {got} vs engine {want}"
        )
        checked += 1
        if checked >= 25:
            break
    assert checked >= 20, f"only {checked} four-ply lines were exercised; the test would be vacuous"


def test_the_three_ply_enumeration_equals_the_rules():
    """After ATTACK, BEAT, STOP the table is empty and the attacker opens with any card.

    That is the `table_size == 0` branch, absent from both written-out enumerators, so a generalisation
    that omitted it would look entirely correct at depth 2 and be blind at depth 4.
    """
    checked = 0
    for seed in range(DEALS):
        first = _deal(seed)
        player = int(first.attacker)
        opening = _beatable_attack(first, player, 1 - player)
        if opening is None:
            continue
        # The opening attack must be applied before the opponent owes an answer; see `_four_ply_line`.
        engine = rules.apply_action(_deal(seed), pack_action(opening[0], opening[1]),
                                    rules.DEFAULT_SETTINGS)
        kinds, cards = [opening[0]], [opening[1]]
        for actor, want in ((1 - player, (BEAT,)), (player, (STOP,))):
            move = _first_of(engine, actor, want)
            if move is None:
                break
            engine = rules.apply_action(engine, pack_action(move[0], move[1]), rules.DEFAULT_SETTINGS)
            kinds.append(move[0])
            cards.append(move[1])
        if len(kinds) != 3 or int(engine.winner) != rules.EMPTY:
            continue

        hand_p = first.hands[player].copy()
        hand_o = first.hands[1 - player].copy()
        out_k = np.zeros(CAP, dtype=np.int64)
        out_c = np.zeros(CAP, dtype=np.int16)
        n = plies_after(first, player, 3, np.array(kinds, dtype=np.int64),
                        np.array(cards, dtype=np.int16), hand_p, hand_o, first.deck[::-1], out_k, out_c)
        legal = _legals(engine, player)
        assert n == len(legal), (
            f"seed={seed}: enumerator found {n}, the rules allow {len(legal)}"
        )
        for i in range(n):
            assert (int(out_k[i]), int(out_c[i])) == legal[i], (
                f"seed={seed} move {i}: enumerator and rules disagree"
            )
        checked += 1
        if checked >= 20:
            break
    assert checked >= 20, f"only {checked} positions exercised; the test would be vacuous"