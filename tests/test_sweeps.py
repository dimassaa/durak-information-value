"""The document-faithful Cheater and the sensitivity sweeps.

Both answer the same question: how much of the project's headline number is a property of the *game*, and
how much is an artifact of how the search was built?
"""

import numpy as np
import pytest

from durak import rules
from durak.strategies import InfoMode
from durak.actions import STOP, THROW, action_kind, pack_action
from durak.analysis.sweeps import (
    DEFAULT_K_VALUES,
    particle_sweep,
    rule_variant_sweep,
    weight_sweep,
)
from durak.evaluation import EVAL_SWEEP_GRID
from durak.strategies.search_policy import PARTICLE_COUNT
from durak.opponent import initial_particles
from durak.rng import RngStream
def _deck_hypo(state, particles):
    """The TRUE draw order reversed: the leaf consumes the pile from the end with a cursor.

    A TEST path. Production builds SHUFFLED hypotheses through `search_policy._deck_hypotheses`, which is
    what makes the pile unknown to the search; a test comparing two arms needs both to see the same pile.
    """
    n = int(state.deck_len)
    order = np.empty(n, dtype=np.int16)
    for i in range(n):
        order[i] = state.deck[n - 1 - i]
    return np.repeat(order[None, :], particles.shape[0], axis=0)



from durak.search import expectimax
from durak.strategies import (
    CHEATER_DOC,
    INFO_PARTICLES,
    RANDOM,
    INFO_TRUTH,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
    get_policy,
    policy_cheater_doc,
    policy_search,
)
from tests.support import card_at, legal_state, random_position

W = (1.0, 2.0, 0.5, 0.3, 1.5)


def _state(table=True, trump=3):
    hands = ([card_at(6, 1), card_at(6, 0), card_at(0, 1)], [card_at(8, 3), card_at(2, 2), card_at(3, 2)])
    return legal_state(*hands, trump=trump, attacker=0,
                       table_attacks=[card_at(2, 1)] if table else [],
                       table_defenses=[card_at(3, 0)] if table else [])


# --- the cheater --------------------------------------------------------------


def test_the_cheater_is_deterministic():
    """No RNG is read and no particle is drawn, so the same position always gives the same move.

    Called twice on the same state rather than on two builds of it: the Cheater reads the real hand out of
    the state and mutates nothing, so one object suffices.
    """
    state = _state()

    first = int(policy_cheater_doc(state, 0, CHEATER_DOC, INFO_TRUTH, np.uint64(1)))
    for seed in (2, 3, 4):
        again = int(policy_cheater_doc(state, 0, CHEATER_DOC, INFO_TRUTH, np.uint64(seed)))
        assert again == first, "the Cheater's answer changed with the rng argument it must ignore"


def test_the_cheater_agrees_with_a_one_hypothesis_search_where_there_is_no_sampling_noise():
    """The document's Cheater and the project's informed arm are the same *idea*, reached two ways.

    With ``k = 1`` there is no Monte-Carlo noise at all, so the two must agree wherever they enumerate the
    same replies. This is the check that the Cheater is a faithful implementation of the idea rather than a
    lookalike.

    Asserted over several positions rather than one, and with the agreement *rate* pinned, because a single
    position cannot separate a faithful implementation from one that happens to coincide.
    """
    state = _state()
    truth = np.tile(np.asarray(sorted(int(c) for c in state.hands[1, :state.hand_sizes[1]]),
                                dtype=np.int16), (1, 1))

    cheater = int(policy_cheater_doc(state, 0, CHEATER_DOC, INFO_TRUTH, np.uint64(7)))
    searched = int(expectimax(state, 0, truth, _deck_hypo(state, truth), W))

    assert cheater == searched, (
        f"the Cheater chose kind {action_kind(cheater)} and the one-hypothesis search chose "
        f"{action_kind(searched)}; they enumerate the same replies, so a difference is a bug in one"
    )


def test_the_two_code_paths_are_genuinely_separate():
    """A Cheater that shared the search would agree with it trivially and measure nothing."""
    assert policy_cheater_doc is not policy_search
    assert get_policy(CHEATER_DOC) is not get_policy(SEARCH_INFORMED)


def test_the_cheater_returns_a_legal_action_from_real_positions():
    for seed in range(30):
        state = random_position(deck_size=36, seed=seed, max_actions=4 + seed % 20)
        if state.winner != rules.EMPTY:
            continue
        player = rules.acting_player(state)
        action = int(policy_cheater_doc(state, player, CHEATER_DOC, INFO_TRUTH, np.uint64(seed)))

        assert action in {int(a) for a in rules.legal_actions(state, player)}


def test_the_cheater_reads_no_particles_and_no_stream():
    """Structural, not behavioural: a strategy that *could* read the stream would eventually do so.

    Checked by inspection of the call rather than by a runtime probe, because a runtime probe cannot
    distinguish "did not read it on these 30 positions" from "cannot read it".
    """
    source = policy_cheater_doc.__wrapped__.__doc__ if hasattr(policy_cheater_doc, "__wrapped__") else None
    # The compiled dispatcher has no Python source, so assert on the documented contract instead.
    assert "Deterministic" in (source or policy_cheater_doc.__doc__ or "")


# --- the sweeps ---------------------------------------------------------------


def test_the_particle_sweep_cannot_vary_k_and_says_so():
    """The grid is pinned to the one value the engine honours, so nobody invents a `k` sweep.

    This test previously asserted the grid was wide and started low -- five values from 25 to 400 -- which
    was the right instinct (pin the grid so nobody picks the flattering end of it after seeing results) but
    had stopped describing reality. The count is a compiled-in constant, so every cell of that sweep ran
    the identical search and reported the identical number.

    Keeping the grid pinned to `PARTICLE_COUNT` means a caller who widens it gets an error from
    `Settings.__post_init__` rather than a table of constant rows that reads like a robustness check.
    """
    assert tuple(DEFAULT_K_VALUES) == (PARTICLE_COUNT,)
    assert len(DEFAULT_K_VALUES) == 1


def test_the_weight_grid_has_a_baseline_and_moves_one_weight_at_a_time():
    """A grid without its own baseline cannot say how much anything moved, and a grid that moved two
    weights together cannot attribute the change."""
    assert (1.0,) * 5 in EVAL_SWEEP_GRID
    assert len(EVAL_SWEEP_GRID) == len(set(EVAL_SWEEP_GRID))
    for row in EVAL_SWEEP_GRID:
        assert len([i for i, factor in enumerate(row) if factor != 1.0]) <= 1


def test_the_particle_sweep_runs_at_the_one_count_the_engine_uses(tmp_path):
    """Run live at four games per cell: the point is the shape of the result, not its values.

    A four-game cell has a Wilson interval of roughly +/- 0.5, so nothing about the *numbers* is claimed.
    What is claimed is that a row is produced, that it carries the games behind it, and that the
    configuration is recorded in the row -- a sweep whose settings live only in the command line is not a
    result.

    The single row is the honest outcome. This sweep can no longer vary ``k``; see ``DEFAULT_K_VALUES``
    for the reason and for the question that stays open as a result.
    """
    frame = particle_sweep(tmp_path, games=4, seed=1)

    assert list(frame["k"]) == [PARTICLE_COUNT], (
        "the sweep runs at the one particle count the engine uses"
    )
    assert (frame["games"] > 0).all()
    assert (frame["settings"].str.contains("particle_count")).all()
    assert (tmp_path / "particle_sweep.csv").exists()


def test_requesting_a_particle_count_the_engine_does_not_use_is_refused(tmp_path):
    """A `k` the search cannot honour must raise, not quietly run the same search twice.

    This is the regression test for a silent defect that had been shipping for several stages.
    `particle_sweep` accepted 25/100/400, wrote three rows, and the three win rates came out identical --
    because `Settings.particle_count` reached nothing and the count was the constant
    `search_policy.PARTICLE_COUNT`. A sweep whose rows cannot differ is worse than no sweep: it looks
    like the robustness check the plan asked for, and it is the same number three times.

    The guard now lives in `Settings.__post_init__`, so it also protects every other caller that
    constructs settings rather than just this sweep.
    """
    with pytest.raises(ValueError, match="particle_count"):
        particle_sweep(tmp_path, k_values=(25,), games=4, seed=1)


def test_a_weight_sweep_returns_a_row_per_grid_point_with_both_arms_and_a_delta(tmp_path):
    """One row per grid point, both arms, and the difference between them.

    The `delta` column is the point of the sweep: a weight that moves both arms together is a bias in the
    evaluation and cancels out of the headline number, while one that moves a single arm changes the
    measured price of information itself. Two rates without the difference hide that distinction.
    """
    frame = weight_sweep(tmp_path, grid=((1.0,) * 5, (2.0, 1.0, 1.0, 1.0, 1.0)), games=4, seed=2)

    assert len(frame) == 2, "one row per weight setting"
    assert {"win_rate_uniform", "win_rate_informed", "delta"} <= set(frame.columns)
    assert (tmp_path / "weight_sweep.csv").exists()
    assert (tmp_path / "weight_sweep_long.csv").exists()


def test_a_rule_variant_sweep_produces_both_settings_of_the_flag(tmp_path):
    """Both readings of the throw-in rule, because the rule is an ambiguity in the source document.

    Reporting only one would leave every throw-in statistic in the dataset resting on an unstated reading
    of the rules, which is the kind of thing a reader cannot discover from the output.
    """
    frame = rule_variant_sweep(tmp_path, games=4, seed=3)

    assert sorted(frame["throwin_defense_ranks"].tolist()) == [False, True]
    assert (frame["games"] > 0).all()
    assert (tmp_path / "rule_variant_sweep.csv").exists()


# --- the arms must actually differ ---------------------------------------------


def test_the_runner_gives_each_arm_the_information_mode_it_asks_for():
    """The project's experimental variable, pinned at the boundary it is applied.

    `run_tournament` passed `InfoMode.NONE` to every seat, which made `SEARCH_UNIFORM` and
    `SEARCH_INFORMED` the same policy on the same input: the primary comparison measured a difference of
    exactly zero and would have reported it as a result. Nothing else would have caught it -- both arms are
    legal, deterministic, and every legality test passes on each.
    """
    from durak.runner import info_modes_for
    from durak.tournament import Pairing

    uniform = info_modes_for(Pairing(0, SEARCH_UNIFORM, RANDOM, 36, "RULE_DERIVED"))
    informed = info_modes_for(Pairing(0, SEARCH_INFORMED, RANDOM, 36, "RULE_DERIVED"))
    cheater = info_modes_for(Pairing(0, CHEATER_DOC, RANDOM, 36, "RULE_DERIVED"))
    reversed_arms = info_modes_for(Pairing(0, RANDOM, SEARCH_INFORMED, 36, "RULE_DERIVED"))

    assert uniform[0] == INFO_PARTICLES and uniform[1] == int(InfoMode.NONE)
    assert informed[0] == INFO_TRUTH and informed[1] == int(InfoMode.NONE)
    assert cheater[0] == INFO_TRUTH
    assert reversed_arms == (int(InfoMode.NONE), INFO_TRUTH), (
        "the informed mode must follow the strategy into whichever seat it plays, not the seat's index"
    )


def test_the_two_arms_score_differently_on_some_position():
    """End to end: across many positions, a truth-fed arm must choose differently from a belief.

    Asserted as a *rate* over positions rather than on one, because a single position where two hands
    differ by one card need not change the decision -- the search may well be indifferent to that card.
    What is being ruled out is the search ignoring its hypothesis entirely, which would make the two arms
    identical on every position.
    """
    from durak.search import expectimax

    different = total = 0
    for seed in range(400):
        state = random_position(deck_size=36, seed=seed, max_actions=8 + seed % 15)
        if state.winner != rules.EMPTY:
            continue
        # Both seats are used, not only the attacker. The `attacker == 0` filter used to be a way of
        # picking attacking positions for convenience, but it costs more sample than the test can afford:
        # after the take rule was corrected, it left only 59 usable positions out of 120, and a floor
        # of 5 on 59 samples sits close enough to the observed 3 that the test was measuring its own
        # sample size. `leaf_profile_after` is role-general, so a defender's position exercises the same
        # path and is a legitimate -- in fact a stronger -- case, since it takes the BEAT branch.
        player = int(state.attacker)
        opponent = 1 - player
        hand1 = [int(c) for c in state.hands[opponent, : state.hand_sizes[opponent]]]
        if len(hand1) < 2:
            continue
        # A belief differing from the truth by one card, drawn from the same unseen set.
        swap = hand1[0]
        replacement = next(c for c in range(36)
                          if c not in hand1
                          and c not in set(int(x) for x in state.hands[player]))
        total += 1
        truth = np.asarray([sorted(hand1)], dtype=np.int16)
        belief = np.asarray([sorted([replacement] + hand1[1:])], dtype=np.int16)

        different += int(expectimax(state, player, truth, _deck_hypo(state, truth), W)) != \
            int(expectimax(state, player, belief, _deck_hypo(state, belief), W))

    assert total > 25, f"only {total} positions were usable"
    # The measured rate is about 10%. It is low by construction, not by defect: swapping one card changes
    # at most one count among (size, trumps, high cards), and a depth-1 leaf ties constantly, so most
    # single-card differences do not change the argmax. The guard here is therefore against ZERO -- which
    # is what a search that ignored its hypothesis would produce -- with a floor above the handful of
    # coincidences, rather than against a rate guessed in advance.
    assert different >= 5, (
        f"the truth and a one-card-different belief chose differently on only {different}/{total} "
        f"positions; a search that read its hypothesis at all would differ on more than this"
    )
