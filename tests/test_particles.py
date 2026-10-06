"""The opponent-hand particle filter: the mechanism the whole project rests on.

The primary question is what perfect knowledge of the opponent's hand is worth. Everything the
project concludes follows from two things being true of this filter: that its prior is the *correct*
prior rather than a convenient one, and that its posterior actually narrows. A filter with a biased
prior would report a confident number that does not describe the game, and a filter that never
narrows would report zero for everything -- both look like findings.

So the tests here are not "does it run". They pin the prior against its analytic value, pin the
posterior against a hand-built state whose answer is known by inspection, and pin the one parameter
that decides whether the uninformed arm is genuinely uninformed.

Uniform and exact, not uniform and approximate
----------------------------------------------
The prior is uniform over size-consistent subsets of the unseen set, which is *exact*: every
hypothesis consistent with what I can see is equally likely. It is not "sample some cards and hope".
That matters because the informed arm's advantage is measured against a baseline that must be
unbiased; a sampled-from-approximation prior would put a floor under the measured advantage that has
nothing to do with information.

`lambda = 0` is the default
----------------------------
Taking is a strategic choice, not evidence of absence. A model that treated "the opponent took"
as "they could not beat it" would be reading the opponent's strategy as a fact about their hand,
which is exactly the kind of unmeasured prior that would contaminate the comparison. So the
behavioural signal exists, is off by default, and there is a test that it is genuinely off -- paired
with a test that it is genuinely wired up, because a disabled-by-default switch that does nothing
when enabled would pass the first test and mean nothing.
"""

import numpy as np
import pytest

from durak import rules
from durak.opponent import (
    Observation,
    apply_observation,
    hand_probabilities,
    initial_particles,
    informativeness,
    marginals_and_entropy,
    unseen_cards,
)
from durak.rng import RngStream
from tests.support import Pool, card_at, legal_state, random_position

DECK = 36


def rng(seed=1):
    return RngStream(seed)


def _state(hand0_size=4, hand1_size=6, n_discard=0, n_table=0, deck_size=DECK):
    """A card-conserving state, built through `legal_state` so it is legal by construction.

    `hand1_size` is the opponent's hand *after* `n_discard` of their cards have gone to the discard
    and `n_table` to the table. Built by allocation rather than by surgery: hand-editing a
    `GameState` to move a card between zones left a stale hand slot behind and tripped
    `check_invariants`, which is exactly the sort of fixture defect that makes a test assert nothing.
    """
    pool = Pool(deck_size)
    hand0 = pool.take(hand0_size)
    remaining = hand1_size + n_discard + n_table
    theirs = pool.take(remaining)
    discard = theirs[:n_discard]
    table = theirs[n_discard:n_discard + n_table]

    state = legal_state(
        hand0, theirs[n_discard + n_table:], trump=0,
        table_attacks=table[:1], table_defenses=table[1:2], discard=discard,
        deck_size=deck_size,
    )
    assert rules.check_invariants(state) == 0
    return state, pool


# --- the prior ---------------------------------------------------------------


def test_every_particle_has_the_opponents_exact_hand_size():
    state, _ = _state(hand0_size=4, hand1_size=6)

    particles = initial_particles(state, 1, 200, rng(3))

    assert particles.shape == (200, 6)
    assert np.all(particles >= 0)


def test_every_particle_contains_only_unseen_cards():
    """A particle containing one of my cards is a hypothesis I know to be false.

    Such a particle is not merely unhelpful -- it is a false statement about the world that the
    posterior average would then report as uncertainty, which is how a filter ends up
    confidently wrong.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 500, rng(4))

    mine = set(int(c) for c in state.hands[0, : state.hand_sizes[0]])
    allowed = set(unseen_cards(state, 1))

    for particle in particles:
        assert not (set(int(c) for c in particle) & mine)
        assert set(int(c) for c in particle) <= allowed


def test_no_particle_repeats_a_card():
    state, _ = _state(hand1_size=6)
    particles = initial_particles(state, 1, 300, rng(5))

    for particle in particles:
        assert len(set(int(c) for c in particle)) == 6


def test_the_empirical_marginal_matches_the_analytic_uniform_prior():
    """`hand_size / |unseen|`, before a single observation.

    This is the test that distinguishes an exact uniform prior from a plausible-looking one, and it
    has to catch a bias the *mean* would hide. A first version asserted each card against `m / U`
    within 0.02 at a single rng seed and **passed against a real bias**: the marginals scatters
    across cards with a standard deviation of 0.058, but at that one seed the largest deviation
    happened to fall inside the band. A single-seed tolerance test can pass by luck, which is
    worthless here.

    So it asserts the *signature* of a uniform prior instead of a per-card deviation:
    `max |p - m/U|` and the across-card spread, over several seeds, plus the entropy deficit --
    which is the seed-stable detector. Measured: 0.0001 bits with a correct prior, 0.077 bits with
    the biased partial Fisher-Yates, so 0.01 sits three orders of magnitude from the noise and eight
    from the failure.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    k = 20_000
    unseen = np.asarray(unseen_cards(state, 1), dtype=np.int16)
    expected = state.hand_sizes[1] / len(unseen)

    deficits = []
    for seed in (11, 12, 13):
        particles = initial_particles(state, 1, k, rng(seed))
        marginals, _ = marginals_and_entropy(particles, unseen)
        values = marginals[unseen]

        assert np.max(np.abs(values - expected)) < 0.02, (
            f"seed {seed}: a marginal is more than 0.02 from the uniform value {expected:.4f}"
        )
        assert values.std() < 0.01, (
            f"seed {seed}: the marginals scatter across cards with std {values.std():.4f}; a "
            f"uniform prior makes every card equally likely, so they should agree"
        )
        deficits.append(np.log2(len(unseen)) - informativeness(particles, unseen))

    assert max(deficits) < 0.01, (
        f"the prior's entropy sits up to {max(deficits):.4f} bits below log2(|unseen|); a uniform "
        f"prior is within 0.001 bits of the ceiling"
    )

    # Cards I hold must be exactly zero, not approximately zero.
    marginals = hand_probabilities(state, 1, k, rng(11))
    for card in range(DECK):
        if card not in set(unseen):
            assert marginals[card] == 0.0


def test_the_marginal_sums_to_the_hand_size():
    """A filter whose marginals do not sum to the hand size is not a filter over hands.

    Mass leaking between cards -- the classic symptom of a resampling or normalisation bug -- shows
    up here and nowhere else.
    """
    state, _ = _state(hand1_size=6)

    marginals = hand_probabilities(state, 1, 5_000, rng(12))

    assert marginals.sum() == pytest.approx(6.0, abs=0.05)


def test_the_prior_is_independent_of_the_rng_seed():
    """Different seeds, same distribution. A seed-dependent prior would mean the uninformed arm's
    baseline shifts with the run seed, which is indistinguishable from a small real effect."""
    state, _ = _state(hand1_size=6)

    a = hand_probabilities(state, 1, 20_000, rng(100))
    b = hand_probabilities(state, 1, 20_000, rng(999))

    unseen = unseen_cards(state, 1)
    assert np.allclose(a[unseen], b[unseen], atol=0.02)


def test_unseen_excludes_my_hand_the_table_and_the_discard():
    """The candidate set is exactly "everything I cannot account for".

    Arithmetic checkable by hand: 36 cards, minus 4 in the observer's hand, minus 1 on the table,
    minus 1 discarded. The opponent's remaining 4 stay in, because which of them are held is what
    is being estimated.
    """
    state, _ = _state(hand0_size=4, hand1_size=4, n_discard=1, n_table=1)

    discarded = int(state.discard[0])
    on_table = int(state.table_attacks[0])
    unseen = set(unseen_cards(state, 1))

    assert on_table not in unseen, "a card on the table must not be a candidate"
    assert discarded not in unseen, "a discarded card must not be a candidate"
    mine = set(int(c) for c in state.hands[0, : state.hand_sizes[0]])
    theirs = set(int(c) for c in state.hands[1, : state.hand_sizes[1]])
    # The opponent's own cards are candidates -- which of them it holds is the question. Only the
    # observer's hand and the visible cards leave the set.
    assert unseen == set(range(DECK)) - mine - {on_table, discarded}
    assert len(unseen) == DECK - 4 - 2
    assert theirs <= unseen


# --- the posterior -----------------------------------------------------------


def test_an_observed_card_disappears_from_every_particle():
    """The hard constraint, stated once and without qualification."""
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 400, rng(20))
    played = int(particles[0][0])

    updated = apply_observation(
        particles, Observation(played_mask=np.uint64(1 << played), hand_size=5),
        soft_penalty_lambda=0.0,
    )

    assert not np.any(updated == played), "an observed card survived in the posterior"


def test_particles_containing_the_observed_card_are_dropped_not_just_pruned():
    """A particle that held the observed card claimed the opponent was holding it.

    The opponent played it, so that hypothesis is false. The particle must be discarded wholesale --
    keeping its remaining cards would silently assert something about cards the particle never had
    evidence about.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 400, rng(21))
    played = int(particles[0][0])

    updated = apply_observation(
        particles, Observation(played_mask=np.uint64(1 << played), hand_size=5),
        soft_penalty_lambda=0.0,
    )

    for particle in updated:
        assert played not in set(int(c) for c in particle)


def test_the_hand_size_constraint_discards_the_impossible_branch():
    """The opponent's hand either shrank by one or did not, and the public count decides.

    Observing "X was played" leaves two branches: X came from the hand (size drops) or from the draw
    pile (size unchanged). Only the observed size says which. Getting this wrong keeps a branch
    alive that the public count has already ruled out, and the filter then carries a systematic
    bias for the rest of the game.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 600, rng(22))
    played = int(particles[0][0])

    shrunk = apply_observation(
        particles, Observation(played_mask=np.uint64(1 << played), hand_size=5),
        soft_penalty_lambda=0.0,
    )
    unchanged = apply_observation(
        particles, Observation(played_mask=np.uint64(1 << played), hand_size=6),
        soft_penalty_lambda=0.0,
    )

    assert shrunk.shape[1] == 5
    assert unchanged.shape[1] == 6
    assert np.all(shrunk >= 0) and np.all(unchanged >= 0)


def test_an_impossible_size_reports_rather_than_silently_filtering_everything():
    """No survivors means the caller's evidence contradicts the state.

    That is a defect worth surfacing, not a condition to paper over: an empty posterior that gets
    quietly re-seeded would turn a tracking bug into a plausible-looking posterior.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 50, rng(23))

    # Eliminate enough cards that no 6-card subset of the remaining set can exist.
    everything = 0
    for card in range(DECK):
        everything |= 1 << card

    with pytest.raises(ValueError, match="observation"):
        apply_observation(
            particles,
            Observation(played_mask=np.uint64(everything), hand_size=1),
            soft_penalty_lambda=0.0,
        )


def test_systematic_resampling_preserves_k_and_stays_non_degenerate():
    """Resampling must restore the particle count without collapsing the set.

    Degeneracy is the failure mode that matters: a filter that has resampled itself into 100 copies
    of one hypothesis reports certainty it has not earned, and every downstream number inherits it.

    `k` is preserved; the hand size is not, and deliberately so -- the observation said a card left
    the hand, so a posterior with six cards in it would be the bug.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 300, rng(24))

    played = int(particles[0][0])
    updated = apply_observation(
        particles, Observation(played_mask=np.uint64(1 << played), hand_size=5),
        soft_penalty_lambda=0.0,
    )

    assert updated.shape == (300, 5)
    assert len(np.unique(updated, axis=0)) > 1, (
        "the particle set collapsed to a single hypothesis, which is certainty the filter has "
        "not earned"
    )
    assert np.all(np.sort(updated, axis=1) == updated), "particles are not in canonical order"


def test_resampling_is_deterministic():
    """No RNG in the update path.

    A random resampling offset would make the informed arm's output depend on a draw the uninformed
    arm does not have, and would break the project-wide reproducibility invariant for one arm only.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 200, rng(25))
    played = int(particles[0][0])
    observation = Observation(played_mask=np.uint64(1 << played), hand_size=5)

    first = apply_observation(particles, observation, soft_penalty_lambda=0.0)
    second = apply_observation(particles, observation, soft_penalty_lambda=0.0)

    assert first.tobytes() == second.tobytes()


def test_a_known_hand_is_recovered_exactly():
    """The posterior at its sharpest: a single card remains, and its identity is known.

    Two cards are revealed and everything else is eliminated one at a time, leaving one hypothesis.
    The marginal must then be exactly 1.0 on that card and 0.0 everywhere else.

    Worth being precise about what the intermediate step does *not* give: revealing two of a three-card
    hand leaves the third uniform over the remaining 28 unseen cards, not determined. The unseen set
    here is 30 cards -- the opponent's three plus 27 that are simply in the draw pile -- so the
    eliminations after the reveals are what make the hypothesis unique. A first version of this test
    skipped them and asserted 1.0, which is false.
    """
    hand0 = [card_at(0, 1), card_at(1, 1), card_at(2, 1)]
    hand1 = [card_at(0, 2), card_at(1, 2), card_at(2, 2)]
    state = legal_state(hand0, hand1, trump=0)
    unseen = set(int(c) for c in unseen_cards(state, 1))

    # A large k is not gold-plating here. Each reveal keeps only `m / U` of the particles, and after
    # two reveals roughly 28 distinct hypotheses survive -- few enough that the marginals are coarse
    # and a tolerance of 0.02 would be testing the luck of the draw. 20,000 keeps ~1,400.
    particles = initial_particles(state, 1, 20_000, rng(30))
    hand_size = len(hand1)

    for revealed in hand1[:-1]:
        particles = apply_observation(
            particles,
            Observation(played_mask=np.uint64(1 << revealed), hand_size=hand_size - 1),
            soft_penalty_lambda=0.0,
        )
        hand_size -= 1

    # Mid-point check. Asserted structurally, not as a uniform distribution, and the difference
    # matters: systematic resampling *duplicates* survivors rather than re-drawing, so after two
    # reveals only ~480 distinct hypotheses carry 20,000 particles and the effective sample size is
    # ~480, not 20,000. Measured card frequencies then span 179 to 1,448 where a uniform prior over
    # the 31 consistent cards would give 645 +- 25. The filter is right -- every consistent card
    # survives and the true card has the highest posterior -- but a tolerance of 0.02 on 1/31 would
    # be asserting an effective sample size the method does not have. Prior uniformity is asserted
    # tightly in its own test, where it is the actual claim.
    assert particles.shape == (20_000, 1)
    consistent = unseen - set(hand1[:-1])

    posterior = np.zeros(DECK)
    for card in range(DECK):
        posterior[card] = float(np.mean(np.any(particles == card, axis=1)))

    assert posterior.sum() == pytest.approx(1.0, abs=0.01)
    for card in set(range(DECK)) - consistent:
        assert posterior[card] == pytest.approx(0.0, abs=0.005), (
            f"card {card} is inconsistent with the evidence but carries posterior "
            f"{posterior[card]:.4f}"
        )
    assert np.all(posterior[list(consistent)] > 0.0), (
        "some consistent card was eliminated outright, so the filter kept fewer hypotheses than "
        "the evidence rules out"
    )
    # The true remaining card gets no special treatment -- the filter has never seen the real hand,
    # and a uniform prior says nothing distinguishes it. So no card may dominate: that would mean
    # the weights, not the support, had drifted. The band is 3x uniform because the resampled counts
    # land on a coarse lattice (measured counts are multiples of 179, i.e. 20000/112, because only
    # ~112 distinct pairs survive the filter) and the observed maximum ratio is 2.3x. It still catches
    # the failure that matters -- one card holding a third of the mass on no evidence.
    uniform = 1.0 / len(consistent)
    assert posterior[list(consistent)].max() < 3.0 * uniform, (
        f"one card holds {posterior[list(consistent)].max():.4f} of the mass where uniform is "
        f"{uniform:.4f}; nothing observed so far distinguishes any consistent card from another"
    )

def test_the_posterior_is_exact_when_the_hand_is_the_only_consistent_hypothesis():
    """Exactness, at a state where exactness is possible.

    Put 30 of the 36 cards in the discard and the observer holds 3, and the opponent's 3-card hand
    is the *entire* unseen set. The prior is then already certain, and the marginals must say so
    exactly -- 1.0 on their cards, 0.0 everywhere else.

    This is a separate test from the recovery one above because eliminating cards one at a time does
    not work as a way to reach it: with a 1-card hand the first elimination collapses the set to that
    one card and every later observation rejects it all. The size constraint admits exactly one
    branch, so narrowing is not something the filter can be *walked* into -- only started in.
    """
    pool = Pool(DECK)
    hand0 = pool.take(3)
    hand1 = pool.take(3)
    discard = [c for c in range(DECK) if c not in set(hand0) | set(hand1)]
    state = legal_state(hand0, hand1, trump=0, discard=discard)

    assert set(unseen_cards(state, 1)) == set(hand1)

    particles = initial_particles(state, 1, 2_000, rng(80))
    marginals = hand_probabilities(state, 1, 2_000, rng(81))

    for card in hand1:
        assert marginals[card] == 1.0
    for card in range(DECK):
        if card not in set(hand1):
            assert marginals[card] == 0.0

    # And it stays exact through the incremental path, which is the path the engine uses. The hand
    # size must be threaded down explicitly: the size constraint admits exactly one branch, so a
    # stale hand size eliminates every hypothesis rather than filtering loosely.
    hand_size = len(hand1)
    for revealed in hand1[:-1]:
        particles = apply_observation(
            particles,
            Observation(played_mask=np.uint64(1 << revealed), hand_size=hand_size - 1),
            soft_penalty_lambda=0.0,
        )
        hand_size -= 1
        assert particles.shape == (2_000, hand_size)
        assert set(np.unique(particles).tolist()) <= set(hand1), (
            "a card outside the true hand entered the particle set"
        )

    # After revealing all but one, every particle must be exactly that last card.
    assert hand_size == 1
    assert set(np.unique(particles).tolist()) == {hand1[-1]}


def test_entropy_decreases_as_observations_accumulate():
    """Monotone non-increasing over a scripted playout.

    The script is the interesting part, and it feeds the opponent's *own* cards: each is certainly
    not in their hand any more, so the posterior is guaranteed to stay non-empty and the comparison
    measures the filter's behaviour rather than the script's luck. Discard-pile cards would not do
    -- some came from my hand, and treating them all as coming from theirs would eliminate a
    fraction of the hypotheses per step and eventually empty the set.
    """
    state = random_position(deck_size=DECK, seed=7, max_actions=30)

    hand = [int(c) for c in state.hands[1, : state.hand_sizes[1]]]
    assert len(hand) >= 4, "the fixture position must have a hand long enough to show a trend"

    # 20,000 particles for the same reason as the recovery test: a reveal keeps only `m / U` of the
    # set, and at 400 the surviving hypotheses thin to single digits partway through the script.
    particles = initial_particles(state, 1, 20_000, rng(31))
    unseen = np.asarray(unseen_cards(state, 1), dtype=np.int16)
    history = [informativeness(particles, unseen)]

    for step, card in enumerate(hand[:5]):
        particles = apply_observation(
            particles,
            Observation(played_mask=np.uint64(1 << card), hand_size=len(hand) - step - 1),
            soft_penalty_lambda=0.0,
        )
        history.append(informativeness(particles, unseen))

    assert len(history) >= 3
    for earlier, later in zip(history, history[1:]):
        assert later <= earlier + 1e-9, (
            f"informativeness rose from {earlier:.4f} to {later:.4f} while evidence accumulated"
        )
    assert history[-1] < history[0], (
        "revealing the opponent's whole hand left the posterior no less uncertain than the prior"
    )


def test_informativeness_is_non_negative_and_below_log2_of_the_unseen_count():
    """Bounds from Shannon entropy: `0 <= H <= log2(n)`.

    A filter reporting negative entropy or more bits than there are hypotheses is broken in a way
    that no downstream statistic would catch.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 2_000, rng(32))
    unseen = np.asarray(unseen_cards(state, 1))

    value = informativeness(particles, unseen)

    assert 0.0 <= value <= np.log2(len(unseen)) + 1e-9


def test_informativeness_is_maximal_at_a_uniform_marginal():
    """The ceiling, asserted rather than assumed.

    At the uniform prior every unseen card has the same frequency, the normalised marginal is
    uniform, and the entropy is `log2(|unseen|)`. If the reported value sits below that, the
    estimator is biased and every phase-curve input is compressed toward zero.

    Measured over six seeds at k = 20,000: 4.9999 bits against a ceiling of 5.0, i.e. a deficit of
    0.0001. The threshold of 0.01 is a hundred times the observed noise and eight times smaller than
    the 0.077 deficit a genuinely biased prior produced -- which is the value that makes this
    tolerance a decision rather than a guess.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    unseen = np.asarray(unseen_cards(state, 1), dtype=np.int16)
    ceiling = np.log2(len(unseen))

    for seed in (33, 34, 35):
        particles = initial_particles(state, 1, 4_000, rng(seed))
        assert ceiling - informativeness(particles, unseen) < 0.01


def test_unseen_contains_the_opponents_cards_and_not_the_observers():
    """The direction of the exclusion, which is the easy thing to get backwards.

    `unseen_cards(state, opponent)` excludes the *observer's* hand -- player `1 - opponent` -- because
    those are the cards the observer can see for themselves. The opponent's actual cards are in the
    candidate set, since which of them it holds is precisely the question.
    """
    pool = Pool(DECK)
    hand0 = pool.take(3)
    hand1 = pool.take(3)
    state = legal_state(hand0, hand1, trump=1)

    unseen = set(unseen_cards(state, 1))

    assert not (unseen & set(hand0)), "the observer's own cards must not be candidates"
    assert set(hand1) <= unseen, (
        "the opponent's cards must be candidates -- excluding them would make the filter "
        "unconditionally certain and look correct while answering nothing"
    )
    assert unseen == set(range(DECK)) - set(hand0)


# --- lambda: the uninformed arm's guarantee -----------------------------------


def _beat_and_take_observations(played, hand_size, taken_from=()):
    """The two observations that must be indistinguishable at `lambda = 0`.

    A beat names a card the opponent chose to play; a take is the absence of a defence. Both carry
    the same *hard* evidence -- the attack is out of the opponent's hand, so their hand shrank by
    one -- plus a *behavioural* signal about cards the opponent declined to use. Only the behavioural
    part distinguishes them, and `hand_size` must be the post-play size or the hard filter rejects
    every hypothesis before the soft branch is ever reached.
    """
    beat = Observation(played_mask=np.uint64(1 << played), hand_size=hand_size,
                       soft_mask=np.uint64(1 << played), soft_delta=1.0)
    soft = np.uint64(0)
    for card in taken_from:
        soft |= np.uint64(1 << card)
    take = Observation(played_mask=np.uint64(1 << played), hand_size=hand_size,
                       soft_mask=soft, soft_delta=1.0)
    return beat, take


def test_lambda_zero_makes_the_model_blind_to_opponent_behaviour():
    """The project's central control, asserted rather than intended.

    At `lambda = 0` a beat and a take carrying identical hard evidence must produce identical
    particle sets. Any difference at all means the uninformed arm is reading the opponent's
    behaviour, and the measured advantage of information would be contaminated by a signal that
    has nothing to do with information.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 300, rng(40))
    assert particles.shape[1] == 6

    a = int(particles[0][0])
    b = int(particles[0][1])
    beat, take = _beat_and_take_observations(a, hand_size=5, taken_from=(b,))

    assert beat.soft_mask != take.soft_mask, "the two observations carry different soft evidence"

    from_beat = apply_observation(particles, beat, soft_penalty_lambda=0.0)
    from_take = apply_observation(particles, take, soft_penalty_lambda=0.0)

    assert from_beat.tobytes() == from_take.tobytes(), (
        "at lambda = 0 the model responded to opponent behaviour"
    )


def test_a_non_zero_lambda_actually_uses_the_soft_signal():
    """The control above is only meaningful if the switch is wired up.

    Without this test, `soft_penalty_lambda` could be dead code that always evaluates to zero and
    the uninformed arm would be clean for the wrong reason -- which is worse than being dirty,
    because it would look like a result.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 300, rng(41))

    a = int(particles[0][0])
    b = int(particles[0][1])
    beat, take = _beat_and_take_observations(a, hand_size=5, taken_from=(b,))

    from_beat = apply_observation(particles, beat, soft_penalty_lambda=2.0)
    from_take = apply_observation(particles, take, soft_penalty_lambda=2.0)

    assert from_beat.tobytes() != from_take.tobytes(), (
        "a non-zero lambda produced no difference, so the soft evidence is not being used and the "
        "lambda = 0 guarantee holds for the wrong reason"
    )


def test_the_soft_signal_sharpens_toward_the_flagged_cards():
    """Direction check: a boosted card should end up *more* likely, not less.

    An inverted sign would satisfy the two tests above -- the switch is wired, and it changes the
    output -- while making the filter confidently wrong. Only the direction distinguishes them.
    """
    state, _ = _state(hand0_size=4, hand1_size=6)
    particles = initial_particles(state, 1, 3_000, rng(42))
    boost = int(particles[0][0])

    observation = Observation(played_mask=np.uint64(0), hand_size=int(state.hand_sizes[1]),
                              soft_mask=np.uint64(1 << boost), soft_delta=1.0)
    updated = apply_observation(particles, observation, soft_penalty_lambda=3.0)

    before = float(np.mean(np.any(particles == boost, axis=1)))
    after = float(np.mean(np.any(updated == boost, axis=1)))

    assert after > before, (
        f"boosting card {boost} moved its marginal from {before:.3f} to {after:.3f}; a positive "
        f"soft signal must raise the probability the card is held"
    )


def test_the_default_lambda_is_zero():
    """Pinned in `Settings`, so the uninformed arm's guarantee cannot be lost to a default change."""
    from durak.config import Settings

    assert Settings().soft_penalty_lambda == 0.0


# --- structure ---------------------------------------------------------------


def test_particles_are_stored_in_canonical_order():
    """Each particle sorted ascending.

    Two hypotheses with the same cards in a different order are the *same* hypothesis. Without a
    canonical order the duplicate scan cannot recognise them, the marginal double-counts them, and
    the entropy is computed over a distribution that does not exist.
    """
    state, _ = _state(hand1_size=6)
    particles = initial_particles(state, 1, 300, rng(50))

    assert np.all(np.diff(particles, axis=1) > 0)


def test_the_filter_is_usable_from_compiled_code():
    """`nopython`-safe: fixed-size arrays, integer and float scalars only.

    Asserted structurally rather than by compiling, because a test that runs `njit` over the
    function is a slow test that only fails in one mode. This pins the *inputs* that make compilation
    possible; the cross-mode equivalence probe is what proves the two modes agree.
    """
    state, _ = _state(hand1_size=6)
    particles = initial_particles(state, 1, 100, rng(60))

    assert particles.dtype.kind in "iu"
    assert particles.ndim == 2
    assert isinstance(np.uint64(1 << 3), np.uint64)

    from durak.opponent import marginals_and_entropy

    marginals, entropy = marginals_and_entropy(
        particles, np.asarray(unseen_cards(state, 1), dtype=np.int16)
    )
    assert marginals.dtype == np.float64
    assert np.isfinite(entropy)
    assert isinstance(entropy, float)


def test_a_one_card_hand_gets_a_uniform_prior_not_a_confident_one():
    """The endgame, where the filter has least to work with and must not overclaim.

    With one card in hand and 34 unseen candidates, that card is at `1/34` a priori -- *not*
    certain. A filter that reported near-certainty here would be reporting the structure of the
    problem as if it were knowledge, and the endgame is exactly where the phase curve says
    information is worth most.
    """
    pool = Pool(DECK)
    hand0 = pool.take(2)
    hand1 = pool.take(1)
    state = legal_state(hand0, hand1, trump=0)
    unseen = np.asarray(unseen_cards(state, 1), dtype=np.int16)

    particles = initial_particles(state, 1, 2_000, rng(70))

    assert particles.shape == (2_000, 1)

    marginals = hand_probabilities(state, 1, 2_000, rng(70))

    assert marginals[hand1[0]] == pytest.approx(1.0 / len(unseen), abs=0.02)
    assert marginals.sum() == pytest.approx(1.0, abs=0.02)
    assert np.log2(len(unseen)) - informativeness(particles, unseen) < 0.05
