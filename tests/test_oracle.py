"""Tests for the ceiling arm, and for the two claims that decide what a ceiling can mean.

Four things are pinned here, and they are not the same kind of thing:

1. The arm is a well-behaved policy -- legal actions, deterministic, state untouched, and *different*
   from ``SEARCH_INFORMED``. The last one is a bug detector, not a nicety: the oracle was specified as
   "the opponent's hand plus the order of the deck", and if it came out byte-identical to the
   informed arm then the depth half of the specification had not reached the policy at all.

2. The arm is structurally the shared search at the configured depth, not a second search. That is
   what makes "the oracle against ``SEARCH_INFORMED``" a measurement of depth rather than of two
   hand-rolled searches that happen to disagree.

3. The draw pile is **public** in this engine and the leaf already draws it in true order, so deck
   order cannot be the oracle's second axis and its marginal value is zero by construction rather
   than by measurement. ``test_the_leaf_draws_the_real_deck_order`` is the differential test: the same
   two-ply line scored against two states that differ only in the deck's order must score
   differently, and must score identically when the order is irrelevant.

4. The stronger consequence, which bears on the headline rather than on the oracle: the opponent's
   hand is the *complement* of everything visible, and the particle filter's candidate set wrongly
   contains the cards face up in the draw pile.

Note that tests 1 and 2 depend on ``ORACLE_DEPTH`` differing from the search arms' ``SEARCH_DEPTH``.
That is not a convenience: at equal depth the oracle *is* ``SEARCH_INFORMED`` by construction, so a
suite that ran under ``DURAK_ORACLE_DEPTH=2`` should fail rather than quietly pass.
"""

import numpy as np

from durak import rules
from durak.actions import ATTACK, TAKE, action_card, action_kind, pack_action
from durak.evaluation import EVAL_WEIGHTS
from durak.leaf import leaf_profile_after
from durak.opponent import _unseen_into
from durak.search import expectimax
from durak.strategies import (
    ORACLE,
    ORACLE_UNIFORM,
    SEARCH_INFORMED,
    InfoMode,
    get_policy,
    policy_names,
    registered_ids,
)
from durak.strategies.oracle import ORACLE_DEPTH
from durak.strategies.search_policy import SEARCH_DEPTH, _truth_particles
from tests.support import Pool, legal_state, random_position

#: Real mid-game positions, reached by random play from a real deal. Constructed fixtures exercise one
#: rule in isolation; a legality or determinism claim about a *policy* is only credible where the rules
#: actually reach, which is what these are.
POSITIONS = [random_position(seed=seed) for seed in range(12)]


def _live(state):
    """Whether the position still admits a move, i.e. whether asking the policy is meaningful."""
    if state.winner != rules.EMPTY:
        return False
    return rules.legal_actions(state, rules.acting_player(state)).size > 0


def _acting(state):
    return rules.acting_player(state)


# --- the arm is a policy -----------------------------------------------------------------------


def test_the_oracle_is_registered_under_its_pinned_id():
    assert ORACLE == 8, (
        "the oracle was appended as id 8; renumbering it would relabel any dataset that already "
        "carries the id"
    )
    assert ORACLE in registered_ids()
    assert policy_names()[ORACLE] == "ORACLE"
    assert get_policy(ORACLE).__name__ == "policy_oracle"


def test_the_deep_pair_is_registered_and_is_one_class():
    """`ORACLE` and `ORACLE_UNIFORM` must differ in the belief and in nothing else.

    The ceiling on the value of perfect knowledge is a *difference* between these two, so the claim
    rests on their sharing a depth, a weight vector, a leaf and a tie-break. Pinned the same way
    `policy_oracle` itself is: by showing `ORACLE_UNIFORM` answers exactly as the shared search does
    when handed the prior particles at the same depth. A second, accidental difference -- its own
    weights, its own depth, its own hand reader -- would fail here rather than quietly turning the
    ceiling into a comparison of two differently-tuned searches.
    """
    from durak.strategies.search_policy import PARTICLE_COUNT, _prior_particles

    assert ORACLE_UNIFORM == 9
    assert ORACLE_UNIFORM in registered_ids()
    assert policy_names()[ORACLE_UNIFORM] == "ORACLE_UNIFORM"
    assert get_policy(ORACLE_UNIFORM).__name__ == "policy_oracle_uniform"

    uniform = get_policy(ORACLE_UNIFORM)
    oracle = get_policy(ORACLE)
    compared = differing = 0
    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        stream = np.uint64(index)
        particles = _prior_particles(state, 1 - player, PARTICLE_COUNT, stream)
        expected = int(expectimax(state, player, particles, EVAL_WEIGHTS, ORACLE_DEPTH))

        chosen = int(uniform(state, player, ORACLE_UNIFORM, int(InfoMode.NONE), stream))
        informed = int(oracle(state, player, ORACLE, int(InfoMode.TRUTH), stream))

        assert chosen == expected, (
            f"ORACLE_UNIFORM chose {chosen} where the shared search at depth {ORACLE_DEPTH} over the "
            f"same prior particles chose {expected}"
        )
        compared += 1
        differing += int(chosen != informed)

    assert compared >= 8, f"only {compared} live positions were compared"
    assert differing > 0, (
        "ORACLE_UNIFORM chose the same action as ORACLE on every position, so the two do not differ "
        "in their belief about the opponent's hand and the difference between them is not a "
        "measurement of anything"
    )


def test_the_oracle_returns_legal_actions():
    policy = get_policy(ORACLE)

    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        allowed = set(int(a) for a in rules.legal_actions(state, player))

        action = int(policy(state, player, ORACLE, int(InfoMode.TRUTH), np.uint64(index)))

        assert action in allowed, (
            f"ORACLE returned {action_kind(action)}/{action_card(action)} at position {index}, "
            f"which is not among the {len(allowed)} legal actions"
        )


def test_the_oracle_is_deterministic():
    """Same state, same answer -- across separate calls, which is where an accidental RNG draw shows."""
    policy = get_policy(ORACLE)

    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        stream = np.uint64(index)
        first = int(policy(state, player, ORACLE, int(InfoMode.TRUTH), stream))
        second = int(policy(state, player, ORACLE, int(InfoMode.TRUTH), stream))

        assert first == second, (
            f"ORACLE answered differently on two calls about the same position ({index})"
        )


def test_the_oracle_ignores_the_information_mode():
    """Perfect information is the arm's definition, not a setting of it.

    `runner.info_modes_for` hands every strategy except ``SEARCH_INFORMED``, ``SEARCH_UNIFORM`` and
    ``CHEATER_DOC`` an ``InfoMode.NONE``, the oracle included. If the arm honoured that, a
    tournament run would report the oracle's strength while measuring the uninformed arm's
    information -- the one failure mode that would make this exercise publish a ceiling nobody
    observed.
    """
    policy = get_policy(ORACLE)

    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        stream = np.uint64(index)
        truth = int(policy(state, player, ORACLE, int(InfoMode.TRUTH), stream))
        none = int(policy(state, player, ORACLE, int(InfoMode.NONE), stream))

        assert truth == none, (
            "ORACLE's answer changed with the information mode, so the mode is reaching the arm "
            "and a mis-wired runner could silently measure a different policy"
        )


def test_the_oracle_is_the_shared_search_at_the_configured_depth():
    """`ORACLE` is `_expectimax` on the truth particle at `ORACLE_DEPTH` -- nothing else.

    Structural rather than behavioural, on purpose: the comparison this arm exists for is against
    `SEARCH_INFORMED`, and that comparison only measures depth while the two arms share the weights,
    the leaf, the tie-break and the belief representation. Pinning it by value means a later edit that
    gave the oracle its own weights or its own hand reader fails here rather than quietly turning the
    oracle into a second, differently-tuned search.
    """
    oracle = get_policy(ORACLE)
    compared = 0
    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        particles = _truth_particles(state, 1 - player)
        expected = int(expectimax(state, player, particles, EVAL_WEIGHTS, ORACLE_DEPTH))

        chosen = int(oracle(state, player, ORACLE, int(InfoMode.TRUTH), np.uint64(index)))

        assert chosen == expected, (
            f"ORACLE chose {chosen} where the shared search at depth {ORACLE_DEPTH} chose {expected}"
        )
        compared += 1

    assert compared >= 8, f"only {compared} live positions were compared"


def test_the_oracle_searches_deeper_than_the_search_arms():
    assert ORACLE_DEPTH > SEARCH_DEPTH, (
        f"ORACLE_DEPTH ({ORACLE_DEPTH}) is not above the search arms' depth ({SEARCH_DEPTH}), so the "
        f"oracle is not a ceiling over them: at equal depth it IS `SEARCH_INFORMED`, and every "
        f"comparison against it would measure a difference of zero"
    )


def test_the_oracle_is_not_search_informed():
    """The discriminating check: deeper search, different answers.

    ``test_the_oracle_is_the_shared_search_at_the_configured_depth`` says what the oracle *is*; this
    says what it must not be. An oracle identical to the informed arm on every position would mean
    the depth is not reaching the policy -- the compile-time-freeze trap, which reads as a null
    result rather than as a failure.
    """
    oracle = get_policy(ORACLE)
    informed = get_policy(SEARCH_INFORMED)

    differing = compared = 0
    for index, state in enumerate(POSITIONS):
        if not _live(state):
            continue
        player = _acting(state)
        stream = np.uint64(index)
        chosen = int(oracle(state, player, ORACLE, int(InfoMode.TRUTH), stream))
        armed = int(informed(state, player, SEARCH_INFORMED, int(InfoMode.TRUTH), stream))
        compared += 1
        differing += int(chosen != armed)

    assert compared >= 8, f"only {compared} live positions were compared"
    assert differing > 0, (
        "ORACLE chose the same action as SEARCH_INFORMED on every position"
    )


# --- the draw pile is public, so deck order cannot be the oracle's second axis -------------------


def _take_and_refill_state():
    """A position whose post-round refill draws exactly one card, and which card depends on the deck.

    Player 0 attacks with one card and player 1 takes: a legal two-ply line that ends the round, so
    the leaf refills. Player 0 held six, plays one, and tops back up to six -- which means exactly
    one card is drawn, and it is ``state.deck[0]``. That is the property the test needs: with one card
    drawn, the deck's *order* decides which card arrives, so the profile can only differ between two
    states if the leaf is genuinely reading the deck in order.
    """
    pool = Pool()
    hand0 = pool.take(6)
    hand1 = pool.take(6)
    trump_card = pool.of_suit(1, 0)[0]   # a trump: changes the trump term
    plain_card = pool.without_suit(1, 0)[0]   # a non-trump below the high-card threshold: does not
    assert trump_card not in hand0 + hand1 and plain_card not in hand0 + hand1

    state = legal_state(hand0, hand1, trump=0, attacker=0, deck_size=36)
    deck = [int(c) for c in state.deck]
    assert len(deck) > 2 and trump_card in deck and plain_card in deck, (
        "the fixture no longer has both cards sitting in the draw pile"
    )

    attack = pack_action(ATTACK, hand0[0])
    return state, deck, trump_card, plain_card, attack


def _with_deck(state, deck):
    """A copy of `state` whose draw pile is `deck`, which must be a permutation of the original.

    Copied rather than mutated because the caller keeps the original for the control comparison, and
    a shared array would make the second measurement a description of the first.
    """
    deck = np.asarray(deck, dtype=np.int16)
    assert sorted(int(c) for c in deck) == sorted(int(c) for c in state.deck), (
        "the reordered pile is not a permutation of the real one, so this would be a different "
        "position rather than a different deck order"
    )
    copied = rules.GameState(
        deck,
        state.hands.copy(),
        state.hand_sizes.copy(),
        int(state.initial_deck),
        int(state.trump),
        bool(state.throwin_defense_ranks),
        int(state.trump_card),
    )
    assert rules.check_invariants(copied) == 0
    return copied


def test_the_leaf_draws_the_real_deck_order():
    """The leaf's simulated refill consumes `state.deck` from the front, in order.

    This is the claim the deck-order half of the oracle's specification rests on. It is a
    discriminating test rather than an inspection: two states that differ *only* in the order of
    their draw pile are indistinguishable to a search that does not read it, so equal profiles would
    prove the leaf ignores the pile. Different profiles prove it follows it.
    """
    state, deck, trump_card, plain_card, attack = _take_and_refill_state()

    # Variant A leads with the trump, variant B with the plain card; the remaining order is shared,
    # so only the order differs.
    rest = [c for c in deck if c not in (trump_card, plain_card)]
    lead_trump = [trump_card, plain_card] + rest
    lead_plain = [plain_card, trump_card] + rest

    def profile(candidate):
        return leaf_profile_after(
            candidate, 0, ATTACK, action_card(attack), TAKE, 0,
            candidate.hands[0], candidate.hands[1], EVAL_WEIGHTS,
        )

    as_lead_trump = profile(_with_deck(state, lead_trump))
    as_lead_plain = profile(_with_deck(state, lead_plain))
    unchanged = profile(_with_deck(state, deck))

    assert as_lead_trump != as_lead_plain, (
        "the leaf scored a position identically whether the draw pile led with a trump or with a "
        "low non-trump. It therefore does not read `state.deck`, which means no policy in this "
        "project has any deck-order information to be given -- and the oracle's deck-order axis "
        "does not exist."
    )
    assert unchanged == profile(state), (
        "rebuilding the state from its own deck changed the score, so this fixture is not being "
        "measured against itself and the comparison above is not the comparison it claims to be"
    )


def test_the_draw_pile_is_visible_to_every_policy():
    """`GameState.deck` is populated and readable, so the draw order is not hidden information.

    The brief for the oracle named deck-order knowledge as something it would have and the other arms
    would not. This test is the reason that half of the brief is void, and it should fail loudly the
    day the draw pile becomes genuinely hidden -- at which point the oracle needs a real second axis
    and this test must be rewritten, not deleted.
    """
    lengths = [int(state.deck_len) for state in POSITIONS]

    assert min(lengths) > 0, "no position has a populated draw pile to read"
    for state in POSITIONS:
        assert state.deck.shape[0] >= int(state.deck_len)
        assert all(0 <= int(state.deck[i]) < int(state.initial_deck)
                   for i in range(int(state.deck_len))), (
            "the draw pile holds sentinels rather than the next cards to be drawn"
        )


def test_the_opponents_hand_is_the_complement_of_the_visible_cards():
    """Perfect knowledge of the opponent's hand is arithmetic here, not inference.

    Cards are conserved and everything except the opponent's hand is visible, so the complement of
    {own hand, table, discard, deck} IS the opponent's hand. This is why the headline delta is a
    statement about whether a policy bothers to compute a set difference, and why the oracle can bound
    a ceiling but cannot make that delta mean what its name says.
    """
    checked = 0
    for index, state in enumerate(POSITIONS):
        for opponent in (0, 1):
            visible = set()
            for i in range(int(state.hand_sizes[1 - opponent])):
                visible.add(int(state.hands[1 - opponent, i]))
            for i in range(int(state.table_attack_count)):
                visible.add(int(state.table_attacks[i]))
            for i in range(int(state.table_defense_count)):
                visible.add(int(state.table_defenses[i]))
            for i in range(int(state.discard_len)):
                visible.add(int(state.discard[i]))
            for i in range(int(state.deck_len)):
                visible.add(int(state.deck[i]))
            complement = set(range(int(state.initial_deck))) - visible
            hand = {int(state.hands[opponent, i]) for i in range(int(state.hand_sizes[opponent]))}

            assert complement == hand, (
                f"position {index}: the opponent's hand is not the complement of the visible cards. "
                f"Only the complement has {sorted(complement)}, the hand has {sorted(hand)}"
            )
            checked += 1

    assert checked > 0


def test_the_unseen_set_contains_the_visible_draw_pile():
    """The particle filter's candidate set includes cards that are face up in the draw pile.

    ``_unseen_into`` subtracts the observer's own hand, the table and the discard -- not the deck. So
    at the opening its candidate set is 30 cards of which 6 are in the opponent's hand, and
    ``SEARCH_UNIFORM`` samples 6 of 30 rather than computing the 6 that are already determined.
    Recorded in ``IMPROVEMENTS.md``; deliberately not fixed here, because fixing it moves
    ``SEARCH_UNIFORM`` underneath an experiment currently running against it.

    The first assertion states the defect and the second pins the identity that causes it. They are
    the same fact read twice on purpose: if the prior is ever corrected, the first fails and says the
    reason has changed.
    """
    checked = 0
    for state in POSITIONS:
        for opponent in (0, 1):
            observer = 1 - opponent
            unseen = np.empty(int(state.initial_deck), dtype=np.int16)
            count = _unseen_into(state, observer, unseen)
            cards = {int(c) for c in unseen[:count]}
            deck = {int(state.deck[i]) for i in range(int(state.deck_len))}

            assert len(cards) == int(state.hand_sizes[opponent]) + int(state.deck_len), (
                f"|unseen| is {len(cards)} but the opponent's hand plus the draw pile is "
                f"{int(state.hand_sizes[opponent]) + int(state.deck_len)}: they were the same set, "
                f"so the prior's candidate set has been changed and this test needs rewriting"
            )
            assert deck <= cards, (
                "the draw pile is no longer inside the unseen set: the prior no longer "
                "mis-attributes visible cards to the opponent, which is the defect being pinned"
            )
            checked += 1

    assert checked > 0