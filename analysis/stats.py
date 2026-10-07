"""Statistical machinery for the reports, implemented directly on NumPy.

No SciPy. Every function here is a closed-form sum or a resampling loop small enough to write out, and
a dependency on a statistics library would be paid for the sake of four formulas the tests already pin
to hand-computed values. It would also make "which version of SciPy produced this interval" part of the
dataset's reproducibility contract, which is a worse trade than forty lines of arithmetic.

What each function is for, and what it deliberately is not
----------------------------------------------------------
* `wilson_interval` -- a binomial proportion. Not the normal approximation, which leaves [0, 1] at the
  extremes and is unreliable at the small counts a 36-card deck produces.
* `mcnemar_exact` -- two paired proportions compared on the discordant pairs only. Exact, not
  chi-squared: the approximation is precisely wrong at small discordant counts, which is the regime the
  endgame tail puts us in.
* `holm_bonferroni` -- correction over a family of tests. Monotonicity is enforced because an adjusted
  family that is not monotonic is not a valid family.
* `paired_bootstrap` -- a confidence interval for a mean of paired differences, which is the shape of
  the primary result: the same games played with and without knowledge of the opponent's hand.
* `proportion_diff_interval` -- the difference of two *independent* proportions, reported from two
  Wilson bounds. This is the weakest interval here and is labelled as such wherever it appears: it is a
  bound, not an estimate.
"""

from __future__ import annotations

import math

import numpy as np

#: Two-sided 95%. Named because the number is also the reason the Wilson interval is used: at 0.5 the
#: normal approximation would need the proportion to be nowhere near 0 or 1, which a win rate rarely is.
WILSON_Z = 1.96


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z) -> tuple:
    """Wilson score interval for a binomial proportion, as `(lo, hi)`.

    Preferred over the normal approximation for two reasons that matter here: it cannot leave [0, 1],
    and for an estimate of exactly 0.5 it is symmetric about 0.5 -- which is what lets the project's
    `RANDOM vs RANDOM = 0.5` symmetry check read "the estimate sits at the centre" rather than "the
    estimate happens to be inside".

    Zero trials returns the whole unit interval. There is no evidence, so there is no constraint, and a
    degenerate point would be a fabricated answer.
    """
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs, as a p-value.

    `b` and `c` are the counts of pairs where exactly one of the two arms won. Pairs where both arms
    agreed carry no information about the difference and are discarded -- that is the whole point of the
    test, and discarding them is what makes it sharp on paired data.

    Exact because the two-sided p-value is a binomial tail:

        p = min(1, 2 * sum_{k=0}^{min(b,c)} C(b+c, k) * 0.5^(b+c))

    Summed directly rather than through a normal approximation, which at small `b + c` is wrong exactly
    where this test gets used: the endgame tail of a 36-card deck.

    `(5, 5)` returns exactly 1.0. The raw tail is 1.246 and is clamped, so a perfectly balanced table
    reads as symmetric rather than as a p-value fractionally below 1.
    """
    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got ({b}, {c})")
    n = b + c
    if n == 0:
        return 1.0

    # Binomial coefficients by recurrence, so no factorial overflows at large n and no table is built.
    tail = 0.0
    coefficient = 1.0
    for k in range(0, min(b, c) + 1):
        if k > 0:
            coefficient = coefficient * (n - k + 1) / k
        tail += coefficient
    tail /= 2.0 ** n
    return min(1.0, 2.0 * tail)


def holm_bonferroni(p_values) -> list:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Bonferroni alone multiplies every p-value by the family size, which is valid but blunt. Holm steps
    down through the sorted values and then enforces monotonicity with a running maximum from the
    largest downwards. That enforcement is not cosmetic: without it the adjusted values can decrease
    as the raw values increase, and a set of thresholds that is not monotone rejects more than it
    declares.
    """
    values = [float(p) for p in p_values]
    m = len(values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: values[i])
    adjusted_sorted = [0.0] * m
    running = 0.0
    # Ascending, because the running maximum is what makes the family monotone: the step-down multiplier
    # shrinks as the p-values grow, so without carrying the maximum forward the adjusted values would
    # DECREASE for larger p-values -- an invalid family. An earlier version walked the order backwards
    # and produced [1.0, 1.0, 1.0, 0.8] for [0.5, 0.6, 0.7, 0.8]: the largest raw p-value came out with
    # the smallest adjusted value.
    for rank, index in enumerate(order):
        scaled = values[index] * (m - rank)
        running = max(running, scaled)
        adjusted_sorted[index] = min(1.0, running)

    return adjusted_sorted


def paired_bootstrap(deltas, resamples: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> tuple:
    """Percentile bootstrap interval for the mean of `deltas`, as `(lo, hi)`.

    Resamples the mean with replacement. On a constant input every resample is that constant and the
    interval collapses onto a point, which is the check that the resampling contributes no variance of
    its own -- a bootstrap that widened on constant input would widen every interval downstream.

    `seed` is required rather than defaulted: a reported interval has to be reproducible, and an
    interval nobody can reproduce is not a result.
    """
    values = np.asarray(deltas, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("paired_bootstrap needs at least one delta")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (lo, hi)


def proportion_diff_interval(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> tuple:
    """The difference of two independent proportions, as a worst-case interval over the two bounds.

    If A lies in [a_lo, a_hi] and B in [b_lo, b_hi], then A - B lies in [a_lo - b_hi, a_hi - b_lo].
    Nothing narrower follows from two intervals alone.

    This is the weakest result in the module and is labelled as a bound wherever it is reported. The
    primary comparison is paired -- the same games with and without information -- and gets
    `paired_bootstrap`; this exists for comparing two *different* matchups, where no pairing exists,
    and reading it as an estimate of the difference would overstate what the data supports.
    """
    return (float(a_lo) - float(b_hi), float(a_hi) - float(b_lo))