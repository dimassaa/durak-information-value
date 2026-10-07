"""The statistics module, checked against hand-computed fixtures rather than against itself.

Every expected value here is worked out in the test, not produced by the code under test. A
statistics library tested against its own output is a tautology: it would agree with itself after any
refactor, including one that breaks it. The fixtures are small enough to compute by hand, and where a
value is not obvious the arithmetic is written out.

One test per function, each pinning the property that function exists for:

* Wilson stays inside [0, 1] and is symmetric about the estimate when the estimate is 0.5 -- the
  symmetry check of the whole engine depends on that second one.
* McNemar is exact, so it returns exactly 1.0 for a perfectly balanced discordant table rather than
  something near it.
* Holm-Bonferroni is monotonic, because an adjusted p-value family that is not monotonic is not a
  valid family and rejects more than it declares.
* The paired bootstrap on a constant array collapses to a point, which is the only way to know the
  resampling is not adding noise that was not in the data.
"""

import numpy as np
import pytest

from analysis.stats import (
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap,
    proportion_diff_interval,
    wilson_interval,
)


# --- Wilson -------------------------------------------------------------------


def test_wilson_zero_successes_of_ten():
    """0/10: the lower bound must be exactly 0 and the upper about 0.278.

    By hand with z = 1.96, z^2 = 3.8416, n = 10:
        denominator = 1 + 3.8416/10 = 1.38416
        centre      = (0 + 3.8416/20) / 1.38416 = 0.19208 / 1.38416 = 0.138770
        margin      = 1.96/1.38416 * sqrt(0 + 3.8416/400)
                    = 1.416000 * sqrt(0.0096040) = 1.416000 * 0.0979992 = 0.138766
        interval    = (0.000004, 0.277536), clamped at 0 to give (0.0, 0.2775)

    The lower bound is 0 rather than slightly negative because a zero-count interval must not claim a
    negative probability.
    """
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=5e-4)


def test_wilson_half_of_ten():
    """5/10: by hand, denominator 1.38416, centre 0.5, margin 0.26341 -> (0.2366, 0.7634)."""
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=5e-4)
    assert hi == pytest.approx(0.7634, abs=5e-4)


def test_wilson_half_of_a_hundred():
    """50/100: denominator 1.038416, margin 0.096161 -> (0.4038, 0.5962)."""
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=5e-4)
    assert hi == pytest.approx(0.5962, abs=5e-4)


def test_wilson_is_symmetric_about_a_half():
    """The property `RANDOM vs RANDOM = 0.5` is checked with depends on it.

    Any interval around an estimate of 0.5 must be symmetric about 0.5, so that a rate of 0.5 sitting
    exactly at the centre is visibly consistent rather than accidentally inside.
    """
    for trials in (4, 10, 100, 1000):
        lo, hi = wilson_interval(trials // 2, trials)
        assert lo + hi == pytest.approx(1.0, abs=1e-9), f"asymmetric at n={trials}"


def test_wilson_stays_inside_the_unit_interval():
    """Including at the extremes, where the normal approximation would leave [0, 1]."""
    for successes, trials in ((0, 3), (3, 3), (0, 1000), (1000, 1000)):
        lo, hi = wilson_interval(successes, trials)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_of_no_trials_is_the_whole_unit_interval():
    """No evidence means no constraint, and returning a point would be a fabricated answer."""
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- McNemar, exact ------------------------------------------------------------


def test_mcnemar_with_no_discordant_pairs_is_one():
    """(0, 0): every pair agreed, so there is nothing to test and p = 1 exactly."""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_with_all_discordant_pairs_one_way():
    """(10, 0): n = 10 discordant pairs, all in one direction.

        p = 2 * sum_{k=0}^{0} C(10,k) * 0.5^10 = 2/1024 = 0.001953125
    """
    assert mcnemar_exact(10, 0) == pytest.approx(0.001953125, rel=1e-12)


def test_mcnemar_with_balanced_discordant_pairs_is_exactly_one():
    """(5, 5) must return exactly 1.0, not 0.9999.

        sum_{k=0}^{5} C(10,k) = 1 + 10 + 45 + 120 + 210 + 252 = 638
        2 * 638 / 1024 = 1.246094, which exceeds 1 and is clamped.

    This is the case a chi-squared approximation gets wrong: it returns a p-value near 1 but not 1, and
    a perfectly symmetric result then reads as if there were a hint of asymmetry.
    """
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_is_symmetric_in_its_arguments():
    """Swapping the two discordant counts cannot change a two-sided p-value."""
    for b, c in ((3, 7), (1, 2), (10, 10), (8, 0)):
        assert mcnemar_exact(b, c) == pytest.approx(mcnemar_exact(c, b), rel=1e-12)


# --- Holm-Bonferroni -----------------------------------------------------------


def test_holm_bonferroni_on_a_hand_worked_family():
    """[0.01, 0.04, 0.03] adjusted, in the original order.

    Sort ascending: 0.01 (rank 1), 0.03 (rank 2), 0.04 (rank 3). Multiply by m - rank + 1, so the
    SMALLEST p-value gets the largest multiplier:
        0.01 * 3 = 0.03
        0.03 * 2 = 0.06
        0.04 * 1 = 0.04
    Monotonicity is a running maximum taken from the smallest upwards:
        0.03                     (rank 1)
        max(0.06, 0.03) = 0.06   (rank 2)
        max(0.04, 0.06) = 0.06   (rank 3)
    Mapping back to the input order [0.01, 0.04, 0.03] gives [0.03, 0.06, 0.06].

    The direction matters and is the reason the monotonicity test exists: taking the maximum from the
    LARGEST downwards instead would leave 0.04 with a smaller adjusted value than 0.03's, which is not a
    valid family. An earlier version of both the function and this expectation had it backwards.
    """
    assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_bonferroni_is_monotonic_in_the_p_values():
    """Adjusted values must not decrease as the raw p-values increase."""
    raw = [0.001, 0.008, 0.02, 0.04, 0.3, 0.9]
    adjusted = holm_bonferroni(raw)
    assert adjusted == sorted(adjusted)


def test_holm_bonferroni_never_exceeds_one():
    """A family of large p-values must not produce adjusted values above 1."""
    adjusted = holm_bonferroni([0.5, 0.6, 0.7, 0.8])
    assert all(a <= 1.0 for a in adjusted)
    assert adjusted[-1] == pytest.approx(1.0)


def test_holm_bonferroni_leaves_a_single_test_alone():
    """With one test there is nothing to correct, and inflating it would be wrong."""
    assert holm_bonferroni([0.02]) == pytest.approx([0.02])


def test_holm_bonferroni_of_nothing_is_nothing():
    assert holm_bonferroni([]) == []


# --- paired bootstrap ----------------------------------------------------------


def test_paired_bootstrap_on_a_constant_delta_is_a_point():
    """Every resample of a constant array is that constant, so the interval must collapse onto it.

    This is the test that says the resampling adds no variance of its own. A bootstrap that widens on
    constant input is not measuring the data, and every downstream interval inherits the widening.
    """
    deltas = np.full(200, 0.05)
    lo, hi = paired_bootstrap(deltas, resamples=500, seed=0)
    assert lo == pytest.approx(0.05, abs=1e-12)
    assert hi == pytest.approx(0.05, abs=1e-12)


def test_paired_bootstrap_brackets_the_mean():
    """The interval must contain the observed mean, and be no wider than the data's own spread."""
    rng = np.random.default_rng(7)
    deltas = rng.normal(0.10, 0.05, size=500)
    lo, hi = paired_bootstrap(deltas, resamples=2000, seed=1)
    mean = float(deltas.mean())
    assert lo < mean < hi
    assert (hi - lo) < 6 * float(deltas.std())


def test_paired_bootstrap_is_reproducible_for_a_seed():
    """Two runs with the same seed agree; without that the reported interval is not a result."""
    rng = np.random.default_rng(11)
    deltas = rng.normal(0.02, 0.03, size=300)
    assert paired_bootstrap(deltas, resamples=800, seed=5) == \
        paired_bootstrap(deltas, resamples=800, seed=5)
    assert paired_bootstrap(deltas, resamples=800, seed=5) != \
        paired_bootstrap(deltas, resamples=800, seed=6)


def test_paired_bootstrap_around_zero_contains_zero():
    """A null difference must not read as significant, or every report will over-claim."""
    rng = np.random.default_rng(13)
    deltas = rng.normal(0.0, 0.04, size=400)
    lo, hi = paired_bootstrap(deltas, resamples=2000, seed=2)
    assert lo <= 0.0 <= hi


# --- difference of two independent proportions ----------------------------------


def test_proportion_diff_interval_from_two_wilson_bounds():
    """The bound is the worst case, not a normal approximation.

    If A is somewhere in [a_lo, a_hi] and B in [b_lo, b_hi], then A - B lies in
    [a_lo - b_hi, a_hi - b_lo]. Nothing narrower is justified from two intervals alone, and a narrower
    answer would look like more precision than the data carries.
    """
    assert proportion_diff_interval(0.58, 0.62, 0.54, 0.58) == pytest.approx((-0.0, 0.08))


def test_proportion_diff_interval_excludes_zero_when_the_bounds_do_not_overlap():
    """The use this interval exists for: saying the difference is not zero."""
    lo, hi = proportion_diff_interval(0.60, 0.66, 0.50, 0.55)
    assert lo > 0.0, "non-overlapping bounds must exclude zero"


def test_proportion_diff_interval_includes_zero_when_the_bounds_overlap():
    lo, hi = proportion_diff_interval(0.50, 0.60, 0.52, 0.62)
    assert lo <= 0.0 <= hi