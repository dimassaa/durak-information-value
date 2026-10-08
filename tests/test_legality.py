"""Tests for strategy legality and the specific behaviour the plan pins.

A strategy that returns an illegal action does not crash the simulation -- the engine
either silently accepts it and produces a state the rules never allow, or it raises in the
middle of a million-game run. Legality is therefore asserted structurally: every policy is
asked for a move at every real position reachable by random play, and the answer must
appear in ``legal_actions``.

The per-strategy tests below pin the specific choices the plan names, because "always
legal" is not the same claim as "plays the lowest non-trump". A policy that is legal but
wrong would pass the first test and corrupt every win rate.
"""

import numpy as np
import pytest

from durak import rules
from durak.actions import (
    ATTACK,
    BEAT,
    STOP,
    TAKE,
    THROW,
    pack_action,
    unpack_action,
)
from durak.cards import rank_of, suit_of
from durak.strategies import (
    SEARCH_INFORMED,
    CHEATER_DOC,
    ORACLE,
    ORACLE_UNIFORM,
    SEARCH_UNIFORM,
    AUTO_ATTACK,
    AUTO_DEFEND,
    GREEDY,
    RANDOM,
    CONSERVATIVE,
    InfoMode,
    StrategyId,
    get_policy,
    policy_names,
    registered_ids,
)
from tests.support import Pool, card_at, legal_state, random_position

T = rules.DEFAULT_SETTINGS

# The ids that must be dispatchable now. The remaining ids in the plan are registered by
# the tasks that implement them, and `test_unimplemented_ids_are_not_dispatchable` pins
# that they are not silently available in the meantime. The two search arms are implemented as
# ONE function, so they are listed separately here only to keep the ids visible; `test_the_two_search_ids_
# resolve_to_the_same_function` in `tests/test_search.py` pins the sharing.
IMPLEMENTED = (AUTO_ATTACK, AUTO_DEFEND, RANDOM, GREEDY, CONSERVATIVE, SEARCH_UNIFORM,
                SEARCH_INFORMED, CHEATER_DOC, ORACLE, ORACLE_UNIFORM)

#: Ids the plan reserves but the registry deliberately refuses to dispatch. `CONSERVATIVE` sat here until
#: Task 13 implemented it; the list is kept so a newly added strategy has to be added to `IMPLEMENTED`
#: rather than appearing in the registry unannounced.
RESERVED = ()

# Positions a policy is asked about. Real random-play positions, not constructed ones: a
# constructed position exercises one rule and misses the interactions a policy meets.
POSITIONS = [random_position(seed=s) for s in range(40)]


def random_position_stream():
    """Positions that land on the acting player's turn, across a spread of walk lengths.

    A single fixed walk-forward depth lands on a defender's turn only occasionally, so a claim
    about defensive behaviour needs several depths rather than more seeds at one depth. Each
    position is truncated at a different point in the same set of games, so the spread costs
    nothing beyond the walks already run.
    """
    for depth in (1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 33, 41):
        for seed in range(12):
            yield random_position(seed=seed, max_actions=depth)


def legal(state, player):
    return {int(a) for a in rules.legal_actions(state, player)}


def offered_kinds(state, player):
    return [unpack_action(a)[0] for a in rules.legal_actions(state, player)]


# --- registry ----------------------------------------------------------------


def test_strategy_id_values_are_pinned():
    # The ids are written into Parquet rows and into the run manifest. Renumbering one
    # would silently re-label every historical dataset.
    assert AUTO_ATTACK == 0
    assert AUTO_DEFEND == 1
    assert RANDOM == 2
    assert GREEDY == 3
    assert StrategyId.CONSERVATIVE == 4
    assert StrategyId.SEARCH_UNIFORM == 5
    assert StrategyId.SEARCH_INFORMED == 6
    assert StrategyId.CHEATER_DOC == 7
    assert StrategyId.ORACLE == 8
    assert StrategyId.ORACLE_UNIFORM == 9


def test_info_mode_values_are_pinned():
    assert InfoMode.NONE == 0
    assert InfoMode.PARTICLES == 1
    assert InfoMode.TRUTH == 2


def test_get_policy_returns_a_callable_for_every_implemented_id():
    for strategy_id in IMPLEMENTED:
        policy = get_policy(strategy_id)

        assert callable(policy)
        assert policy.__name__.startswith("policy_")


def test_get_policy_raises_for_an_unknown_id():
    with pytest.raises(ValueError, match="unknown strategy"):
        get_policy(99)

    with pytest.raises(ValueError, match="unknown strategy"):
        get_policy(-1)


def test_unimplemented_ids_are_not_dispatchable():
    """Ids reserved by the plan must not resolve until their task lands.

    A stub that returned a random move for an unimplemented id would produce a plausible,
    wrong dataset for the experiment's headline comparison.

    `RESERVED` is empty now that Task 13 has landed `CONSERVATIVE`: every id the plan names is
    implemented, so there is nothing left for this to catch. It is kept as a list so that the next
    reserved strategy has a place to be declared, and so the assertion below continues to mean something
    rather than silently passing over an empty tuple.
    """
    for strategy_id in RESERVED:
        with pytest.raises(ValueError, match="unknown strategy"):
            get_policy(int(strategy_id))

    assert set(registered_ids()) == {int(i) for i in StrategyId}, (
        "every pinned StrategyId must now be dispatchable; if one is not, it belongs in RESERVED"
    )


def test_policy_names_cover_every_implemented_id():
    assert set(registered_ids()) == {int(i) for i in IMPLEMENTED}
    # The names table is a superset: it also names the reserved ids, so a report can refer
    # to a strategy the registry cannot yet dispatch.
    assert set(policy_names()) >= {int(i) for i in IMPLEMENTED}


def test_policy_names_are_human_readable():
    names = policy_names()

    assert names[RANDOM] == "RANDOM"
    assert names[GREEDY] == "GREEDY"
    assert names[AUTO_ATTACK] == "AUTO_ATTACK"


# --- legality across real positions -----------------------------------------


@pytest.mark.parametrize("strategy_id", IMPLEMENTED, ids=lambda i: policy_names()[i])
def test_policy_only_returns_legal_actions(strategy_id):
    policy = get_policy(strategy_id)

    for index, state in enumerate(POSITIONS):
        player = rules.acting_player(state)
        if state.winner != rules.EMPTY:
            continue
        allowed = legal(state, player)
        action = int(policy(state, player, int(strategy_id), int(InfoMode.NONE), np.uint64(index)))

        assert action in allowed, (
            f"{policy_names()[strategy_id]} returned {unpack_action(action)} at position "
            f"{index}, which is not among {sorted(unpack_action(a) for a in allowed)}"
        )


@pytest.mark.parametrize("strategy_id", IMPLEMENTED, ids=lambda i: policy_names()[i])
def test_policy_leaves_the_state_untouched(strategy_id):
    """A policy observes the state; it does not move pieces.

    A policy that mutated the state would make the engine's own transition the second
    writer, and the resulting state would depend on the order in which strategies were
    consulted -- invisible in the win rates, fatal to reproducibility.
    """
    policy = get_policy(strategy_id)
    state = random_position(seed=3)
    player = rules.acting_player(state)

    before = (
        state.hands.copy(), state.hand_sizes.copy(), state.deck.copy(), state.deck_len,
        state.discard_len, state.table_attack_count, state.table_defense_count,
        state.attacker, state.round_idx, state.winner, state.rng_state,
    )
    policy(state, player, int(strategy_id), int(InfoMode.NONE), np.uint64(1))
    after = (
        state.hands, state.hand_sizes, state.deck, state.deck_len,
        state.discard_len, state.table_attack_count, state.table_defense_count,
        state.attacker, state.round_idx, state.winner, state.rng_state,
    )

    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])
    assert before[3:] == after[3:]


def test_random_advances_only_the_acting_players_stream():
    """A player's draws must not shift the opponent's randomness.

    This is the property cross-strategy comparability rests on. If both players shared one
    stream, changing how many draws one strategy consumed would change the opponent's
    behaviour, and a run differing in exactly one strategy would not be comparable to a run
    differing in another. The state is a shared object here precisely so that both streams are
    visible, and a policy that touched the wrong one would be easy to miss.
    """
    policy = get_policy(RANDOM)
    state = random_position(seed=11)
    player = rules.acting_player(state)
    before = (int(state.rng_p0), int(state.rng_p1))

    int(policy(state, player, RANDOM, int(InfoMode.NONE), np.uint64(1)))

    after = (int(state.rng_p0), int(state.rng_p1))
    assert after[player] != before[player], "the acting player's stream did not advance"
    assert after[1 - player] == before[1 - player], "the opponent's stream was disturbed"


def test_random_starts_from_the_state_stream_not_the_argument():
    """The `rng_state` argument is advisory; the state is authoritative.

    The plan's policy signature returns only an action, so a policy has no channel to report a
    consumed stream -- it has to write the advance back into the state. This test pins that the
    written value is what the *next* call reads, rather than the argument the caller passed.
    """
    policy = get_policy(RANDOM)
    state = random_position(seed=13)
    player = rules.acting_player(state)
    offered = rules.legal_actions(state, player)

    int(policy(state, player, RANDOM, int(InfoMode.NONE), np.uint64(1)))
    advanced = int(state.rng_p0) if player == 0 else int(state.rng_p1)

    # A stream left where the first call found it would hand out the same card again, so the
    # second call is asked to produce the draw that `advanced` -- not the original -- implies.
    from durak.rng import next_below

    index, _ = next_below(np.uint64(advanced), len(offered))
    expected = int(offered[int(index)])

    assert advanced != int(state.rng_state), "the deal's stream was used instead"
    assert int(policy(state, player, RANDOM, int(InfoMode.NONE), np.uint64(0))) == expected


@pytest.mark.parametrize("strategy_id", (AUTO_ATTACK, AUTO_DEFEND, GREEDY),
                         ids=lambda i: policy_names()[i])
def test_deterministic_policies_ignore_the_rng_stream(strategy_id):
    """Only `RANDOM` is stochastic.

    The equivalence test that proves the engine correct compares move trajectories between
    two execution modes. A policy that drew from the stream would still be reproducible
    there, but it would make a strategy's win rate depend on how many draws it happened to
    consume -- which the two search arms must share exactly.
    """
    policy = get_policy(strategy_id)
    state = random_position(seed=5)
    player = rules.acting_player(state)

    first = int(policy(state, player, int(strategy_id), int(InfoMode.NONE), np.uint64(1)))
    second = int(policy(state, player, int(strategy_id), int(InfoMode.NONE), np.uint64(10**15)))

    assert first == second


def test_random_consumes_the_stream_and_varies():
    """`RANDOM` must actually vary, over positions where varying is possible.

    Scans for a position with more than one legal action rather than pinning a seed. A fixed
    position can have exactly one legal move -- an answered table where the attacker holds no
    rank-matching card is forced to stop -- and a draw from a one-element list is trivially
    constant. Asserting on such a position would pass for a policy that ignored its stream
    entirely, which is the bug the test exists to catch.
    """
    policy = get_policy(RANDOM)
    chosen = None

    for seed in range(40):
        state = random_position(seed=seed)
        if state.winner != rules.EMPTY:
            continue
        player = rules.acting_player(state)
        if len(rules.legal_actions(state, player)) > 2:
            chosen = (state, player)
            break

    assert chosen is not None, "no position with a real choice was found in the sample"
    state, player = chosen

    draws = {
        int(policy(state, player, RANDOM, int(InfoMode.NONE), np.uint64(seed)))
        for seed in range(300)
    }

    # A policy that ignored its stream would return one action forever, and `RANDOM vs
    # RANDOM = 0.5` would then be an accident of a fixed script rather than a symmetry property.
    assert len(draws) > 1, f"300 draws produced {len(draws)} distinct actions"


# --- AUTO_ATTACK -------------------------------------------------------------


def test_auto_attack_picks_the_lowest_non_trump_when_one_exists():
    pool = Pool()
    # One trump, the lowest trump, and three non-trumps of descending rank. The lowest
    # non-trump must win, and the lowest trump must not be preferred merely by rank.
    hand0 = [pool.reserve(card_at(r, s)) for r, s in
             ((0, 0), (1, 0), (1, 1), (5, 1), (2, 2), (8, 3))]
    state = legal_state(hand0, pool.take(6), trump=0, attacker=0)
    policy = get_policy(AUTO_ATTACK)

    action = int(policy(state, 0, AUTO_ATTACK, int(InfoMode.NONE), np.uint64(1)))
    kind, card = unpack_action(action)

    assert kind == ATTACK
    # rank 1 hearts is the lowest non-trump; the trump is lower still but must be avoided.
    assert card == card_at(1, 1)
    assert suit_of(card) != 0
    assert min(rank_of(c) for c in hand0 if suit_of(c) != 0) == rank_of(card)


def test_auto_attack_falls_back_to_the_lowest_trump():
    pool = Pool()
    hand0 = pool.of_suit(6, 0)
    state = legal_state(hand0, pool.take(6), trump=0, attacker=0)
    policy = get_policy(AUTO_ATTACK)

    kind, card = unpack_action(
        int(policy(state, 0, AUTO_ATTACK, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == ATTACK
    assert card == min(hand0, key=rank_of)


def test_auto_attack_takes_when_defending():
    pool = Pool()
    attack = pool.reserve(card_at(2, 1))
    state = legal_state(pool.take(6), pool.take(6), trump=0, attacker=0,
                        table_attacks=[attack])
    policy = get_policy(AUTO_ATTACK)

    kind, _ = unpack_action(
        int(policy(state, 1, AUTO_ATTACK, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == TAKE


# --- AUTO_DEFEND -------------------------------------------------------------


def test_auto_defend_beats_with_the_lowest_beater():
    pool = Pool()
    attack = pool.reserve(card_at(2, 1))          # hearts rank 2
    # hearts rank 0 and 1 cannot beat. hearts rank 3 is the lowest card that can, and the
    # spade is a beater too but sits at rank 0 -- so this position separates "lowest card
    # that beats" from "lowest card of the same suit that beats", and the spec names the
    # first.
    hand1 = [pool.reserve(card_at(r, s)) for r, s in
             ((0, 1), (1, 1), (3, 1), (6, 1), (0, 0), (1, 2))]
    state = legal_state(pool.take(6), hand1, trump=0, attacker=0, table_attacks=[attack])
    policy = get_policy(AUTO_DEFEND)

    kind, card = unpack_action(
        int(policy(state, 1, AUTO_DEFEND, int(InfoMode.NONE), np.uint64(1)))
    )
    beaters = [c for c in hand1
               if any(unpack_action(int(a)) == (BEAT, c)
                      for a in rules.legal_actions(state, 1))]

    assert kind == BEAT
    assert card == min(beaters, key=rank_of)


def test_auto_defend_takes_when_it_cannot_beat():
    pool = Pool()
    attack = pool.reserve(card_at(0, 0))          # a trump
    state = legal_state(pool.take(6), pool.of_suit(6, 1), trump=0, attacker=0,
                        table_attacks=[attack])
    policy = get_policy(AUTO_DEFEND)

    kinds = offered_kinds(state, 1)

    assert kinds == [TAKE]
    kind, _ = unpack_action(
        int(policy(state, 1, AUTO_DEFEND, int(InfoMode.NONE), np.uint64(1)))
    )
    assert kind == TAKE


# --- GREEDY: defence ---------------------------------------------------------


def _greedy_defence(pool, target, hand1, trump=0):
    """A one-card-on-the-table position with `hand1` as the defender's hand.

    `target` and `hand1` must already be reserved from `pool`, and the attacker's hand is
    filled from the same pool. Using one pool throughout is not tidiness: two pools re-issue
    the same low card ids, and the resulting fixture is not a position the rules can reach.
    """
    state = legal_state(pool.take(6), hand1, trump=trump, attacker=0, table_attacks=[target])
    policy = get_policy(GREEDY)
    return state, int(policy(state, 1, GREEDY, int(InfoMode.NONE), np.uint64(1)))


def test_greedy_defends_with_the_lowest_beating_card():
    pool = Pool()
    target = pool.reserve(card_at(2, 1))          # hearts rank 2
    # hearts rank 0 and 1 cannot beat; hearts rank 3 is the lowest that can, and the trump
    # also beats but ranks higher.
    hand1 = [pool.reserve(card_at(r, s)) for r, s in
             ((0, 1), (1, 1), (3, 1), (7, 1), (0, 0), (1, 3))]
    state, action = _greedy_defence(pool, target, hand1)
    kind, card = unpack_action(action)
    beaters = [c for c in hand1
               if any(unpack_action(int(a)) == (BEAT, c)
                      for a in rules.legal_actions(state, 1))]

    assert kind == BEAT
    assert beaters, "the fixture offered no beating card"
    assert card == min(beaters, key=rank_of)


def test_greedy_never_declines_an_available_beat():
    """Over many positions with a beat on the table, `GREEDY` must never take.

    Declining an available beat is the difference between a reference strategy and a broken
    one. It is invisible in an aggregate win rate unless it is asserted directly.
    """
    policy = get_policy(GREEDY)
    declined = 0
    opportunities = 0

    for index, state in enumerate(random_position_stream()):
        if state.winner != rules.EMPTY:
            continue
        player = rules.acting_player(state)
        if player == state.attacker:
            continue
        kinds = offered_kinds(state, player)
        if BEAT in kinds:
            opportunities += 1
            kind, _ = unpack_action(
                int(policy(state, player, GREEDY, int(InfoMode.NONE), np.uint64(index)))
            )
            if kind == TAKE:
                declined += 1

    # A weak threshold, on purpose. The claim under test is "never", and a larger sample only
    # makes the test slower; what matters is that enough defensive turns were actually
    # observed for "never" to mean something.
    assert opportunities >= 10, f"only {opportunities} defensive turns with a beat available"
    assert declined == 0


def test_greedy_takes_when_no_beat_exists():
    pool = Pool()
    target = pool.reserve(card_at(0, 0))          # a trump nothing can beat
    hand1 = pool.of_suit(6, 1)
    state, action = _greedy_defence(pool, target, hand1)
    kind, _ = unpack_action(action)

    assert offered_kinds(state, 1) == [TAKE]
    assert kind == TAKE


# --- GREEDY: attack ----------------------------------------------------------


def test_greedy_never_opens_with_a_trump_while_holding_a_non_trump():
    pool = Pool()
    # Two trumps of ranks 0 and 4, so a naive lowest-rank rule would open with a trump.
    hand0 = [pool.reserve(card_at(r, s)) for r, s in
             ((0, 0), (4, 0), (7, 1), (5, 2), (6, 3), (8, 1))]
    state = legal_state(hand0, pool.take(6), trump=0, attacker=0)
    policy = get_policy(GREEDY)

    kind, card = unpack_action(
        int(policy(state, 0, GREEDY, int(InfoMode.NONE), np.uint64(1)))
    )
    non_trumps = [c for c in hand0 if suit_of(c) != 0]

    assert kind == ATTACK
    assert suit_of(card) != 0
    assert card == min(non_trumps, key=rank_of)


def test_greedy_opens_with_the_lowest_trump_when_that_is_all_it_has():
    pool = Pool()
    hand0 = pool.of_suit(6, 0)
    state = legal_state(hand0, pool.take(6), trump=0, attacker=0)
    policy = get_policy(GREEDY)

    kind, card = unpack_action(
        int(policy(state, 0, GREEDY, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == ATTACK
    assert card == min(hand0, key=rank_of)


def test_greedy_throws_in_while_the_limit_allows_and_stops_otherwise():
    """The plan says `GREEDY` throws in any rank-matching card while the limit allows.

    Asserted at both sides of the boundary, because a policy that throws in unconditionally
    and one that never does are each a plausible single-line mistake with opposite
    consequences for the throw-in statistics.
    """
    from tests.support import table_with_free_ranks

    policy = get_policy(GREEDY)

    under_cap = Pool()
    attacks, defenses, free = table_with_free_ranks(under_cap, 2)
    matching = under_cap.reserve(card_at(rank_of(attacks[0]), 2))
    state = legal_state([matching] + free, under_cap.take(6), trump=0,
                        table_attacks=attacks, table_defenses=defenses)
    kind, card = unpack_action(
        int(policy(state, 0, GREEDY, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == THROW
    assert card == matching

    at_cap = Pool()
    attacks, defenses, free = table_with_free_ranks(at_cap, 6)
    matching = at_cap.reserve(card_at(rank_of(attacks[0]), 2))
    state = legal_state([matching] + free, at_cap.take(9), trump=0,
                        table_attacks=attacks, table_defenses=defenses)
    kind, _ = unpack_action(
        int(policy(state, 0, GREEDY, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == STOP


def test_greedy_stops_when_holding_no_matching_card():
    from tests.support import table_with_free_ranks

    pool = Pool()
    attacks, defenses, free = table_with_free_ranks(pool, 2)
    state = legal_state(list(free), pool.take(6), trump=0,
                        table_attacks=attacks, table_defenses=defenses)
    policy = get_policy(GREEDY)

    kind, _ = unpack_action(
        int(policy(state, 0, GREEDY, int(InfoMode.NONE), np.uint64(1)))
    )

    assert kind == STOP


def test_greedy_and_auto_attack_agree_on_the_opening_attack():
    """Both attack the same way, so a `GREEDY` vs `AUTO_ATTACK` run isolates defence.

    The two policies exist for different jobs -- one is a reference strategy, the other a
    driver for stage 2 -- and the comparison between them is only interpretable if their
    attacking behaviour is identical.
    """
    greedy = get_policy(GREEDY)
    auto = get_policy(AUTO_ATTACK)

    for seed in range(10):
        pool = Pool()
        hand0 = pool.take(6)
        state = legal_state(hand0, pool.take(6), trump=0, attacker=0)

        assert int(greedy(state, 0, GREEDY, 0, np.uint64(seed))) == \
            int(auto(state, 0, AUTO_ATTACK, 0, np.uint64(seed)))
