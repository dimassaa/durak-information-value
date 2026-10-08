"""Opponent-hand inference: a particle filter over size-consistent hypotheses.

This is the mechanism the project's primary question rests on. Everything reported about the value
of information follows from two properties of this filter: that its prior is the *correct* prior
rather than a convenient one, and that its posterior actually narrows as evidence arrives. A filter
with a biased prior reports a confident number that does not describe the game; a filter that never
narrows reports zero for everything. Both look like findings.

The prior is uniform, and it is exact
-------------------------------------
The observer knows their own hand, every card on the table, and the discard pile. Everything else is
the "unseen" set, and the opponent's hand is one of the ``C(U, m)`` subsets of it of the size the
observer can count. The prior puts equal mass on every one of them.

*Exact* means no approximation is involved: every hypothesis consistent with what is visible is
equally likely, and nothing else is admitted. It does not mean the posterior is exact -- it is a
Monte-Carlo approximation over a combinatorial space, which is why the particle count is a setting
rather than a constant.

The exactness of the prior is not cosmetic. The project's headline number is a difference between an
informed arm and an uninformed one. If the *uninformed* arm's baseline were itself biased, that
difference would have a floor under it that has nothing to do with information, and no amount of
statistical care downstream would remove it. So the empirical marginal is asserted against
``m / U`` -- the analytic value -- rather than against anything this code produces.

``lambda = 0`` is the default, and that is the interesting part
-------------------------------------------------------------
Two kinds of evidence reach the filter. *Hard* evidence is a card leaving the opponent's hand:
unambiguous, and it removes a hypothesis. *Soft* evidence is the opponent's choice -- which card
they beat with, or whether they took at all.

Taking is a strategic choice, not evidence of absence. A model that read "the opponent took" as
"they could not beat it" would be reading the opponent's strategy as a fact about their hand. That
prior is unmeasured, and injecting it into the *uninformed* arm would mean both arms saw opponent
behaviour while only one saw opponent cards -- so the measured gap would no longer be a gap about
information.

Hence ``soft_penalty_lambda`` defaults to zero and the soft branch is skipped entirely rather than
scaled to zero. Multiplying by zero would still resample, and resampling perturbs the particle set.
Two tests pin this from both sides -- one asserts the model is blind at zero, the other asserts it
is *not* blind above zero, because a disabled-by-default switch that does nothing when enabled would
pass the first test and mean nothing.

Deviation from the plan's interface, deliberately
-------------------------------------------------
The plan specifies ``apply_observation(particles, state, observation, settings)``. This module takes
``(particles, observation, soft_penalty_lambda)``. Two reasons, both from the codebase's existing
boundary rule that no Python object crosses into ``nopython``:

* ``settings`` is a frozen dataclass and cannot enter compiled code. The caller unpacks the one field
  it needs into a ``float64``, which is the established pattern for every hot path here.
* ``state`` is not needed. The filter updates from the observation alone; the state is consulted
  once, when the prior is drawn, to enumerate the unseen set.

Observing a card played
-----------------------
The update is subtler than "delete the card". Observing "X was played" leaves two branches: X came
out of the hand, so the hand shrank, or X came off the draw pile, so it did not. Only the publicly
counted hand size says which. Particles holding X are therefore shrunk by one and particles not
holding X are kept, and whichever branch disagrees with the observed size is discarded. Keeping both
branches when only one is possible would leave a systematic bias in the posterior for the rest of
the game.

Determinism
-----------
``apply_observation`` takes no RNG. Resampling uses a fixed offset rather than a uniform random one,
which makes it very slightly biased and exactly reproducible. That is the right trade here: the
informed arm's output must depend on the game and nothing else, and a random offset would give one
arm a draw the other does not have, breaking the project-wide reproducibility invariant
asymmetrically.
"""

from __future__ import annotations

import os

from typing import NamedTuple

import numpy as np
from numba import njit

from durak.rng import RngStream, next_below

# Card sets are bitmasks: a 36- or 52-card deck fits in 64 bits, and the engine already uses the
# same representation for its duplicate check. Using it here keeps the compiled loops free of
# allocation and of any container type Numba cannot handle.
_ONE = np.uint64(1)


class Observation(NamedTuple):
    """One update to the filter.

    ``played_mask`` is *hard* evidence: these cards are known not to be in the opponent's hand.
    ``hand_size`` is the publicly counted hand size after the observation, which selects between the
    "came from the hand" and "came from the draw pile" branches.

    ``soft_mask`` and ``soft_delta`` are the *behavioural* signal: cards whose presence in the hand
    made the opponent's observed choice more likely. They are ignored when
    ``soft_penalty_lambda == 0``; see the module docstring for why that is the default.
    """

    played_mask: np.uint64
    hand_size: int
    soft_mask: np.uint64 = np.uint64(0)
    soft_delta: float = 0.0


#: How many played cards a model remembers, in cards discarded. `int` default is effectively unlimited and
#: reproduces the project's earlier behaviour exactly; set `DURAK_MEMORY_PLY` to bound it. Read at import
#: for the same reason as `SEARCH_DEPTH`: `@njit` freezes module globals at compile time, so a sweep over
#: this value would otherwise measure the same policy at every setting.
MEMORY_PLY = int(os.environ.get("DURAK_MEMORY_PLY", "1000000"))


@njit(cache=True)
def _unseen_into(state, observer: int, out: np.ndarray) -> int:
    """Fill `out` with the cards `observer` cannot account for, returning the count.

    The *observer's own* hand, the table and the discard are all visible to them, so the candidate
    set for the opponent's hand is everything left. Note the direction: `observer` is the player
    doing the inferring, and the cards excluded here are the observer's own -- not the opponent's,
    which is exactly what is being estimated.
    """
    seen = np.uint64(0)
    for i in range(state.hand_sizes[observer]):
        seen |= _ONE << np.uint64(state.hands[observer, i])
    for i in range(state.table_attack_count):
        seen |= _ONE << np.uint64(state.table_attacks[i])
    for i in range(state.table_defense_count):
        seen |= _ONE << np.uint64(state.table_defenses[i])
    # The discard pile, bounded by the memory horizon. Every model previously accounted for EVERY card
    # ever played, so the candidate set shrank monotonically and the beliefs were perfect-memory -- which
    # makes "what is perfect knowledge of the hand worth" a question about bookkeeping at least as much
    # as about the opponent. `MEMORY_PLY` plies back is remembered; anything older goes back into the
    # candidate set, and the filter must then place it somewhere, which is the uncertainty being modelled.
    # `MEMORY_PLY` at its default reproduces the old behaviour EXACTLY, which is the regression gate.
    first_remembered = state.discard_len - MEMORY_PLY
    if first_remembered < 0:
        first_remembered = 0
    for i in range(first_remembered, state.discard_len):
        seen |= _ONE << np.uint64(state.discard[i])

    count = 0
    for card in range(state.initial_deck):
        if (seen & (_ONE << np.uint64(card))) == np.uint64(0):
            out[count] = card
            count += 1
    return count


def unseen_cards(state, opponent: int) -> np.ndarray:
    """Card ids `opponent`'s hand could consist of, sorted ascending.

    `opponent` is the player being *modelled*; the observer is the other seat, and it is the
    observer's own hand that is excluded rather than the opponent's. Getting that backwards is the
    natural mistake here and it is silent -- the filter returns a plausible uniform prior over a
    candidate set that still contains the observer's own cards.

    A thin Python wrapper: the hot path calls the compiled ``_unseen_into`` directly, and this exists
    for tests, analysis, and the places where a list of card ids reads better than a bitmask.
    """
    out = np.empty(state.initial_deck, dtype=np.int16)
    count = _unseen_into(state, 1 - int(opponent), out)
    return out[:count].copy()


@njit(cache=True)
def _draw_particles(unseen: np.ndarray, u_len: int, hand_size: int, k: int,
                    rng_state: np.uint64):
    """`k` hypotheses drawn uniformly from the size-consistent subsets of `unseen`.

    Uniform by construction: a partial Fisher-Yates picks each card with probability proportional to
    how many unfilled positions remain, which makes every size-`hand_size` subset equally likely.
    Sampling *with* replacement from the unseen set would not -- that is the approximation this
    project specifically does not make.
    """
    particles = np.empty((k, hand_size), dtype=np.int16)
    pool = np.empty(u_len, dtype=np.int16)

    for i in range(k):
        for j in range(u_len):
            pool[j] = unseen[j]
        for j in range(hand_size):
            # `pick` must lie in [j, u_len), NOT [0, u_len - j). Swapping with a position already
            # consumed would pull a selected card back out of the row, and the resulting
            # distribution is not uniform: measured across the 32 unseen cards the marginals
            # scattered with a standard deviation of 0.058 against a multinomial value of 0.0025,
            # and the prior's entropy sat 0.077 bits below log2(U) no matter how many particles
            # were drawn. That is exactly the bias an "exact uniform prior" cannot have.
            pick, rng_state = next_below(rng_state, u_len - j)
            pick = j + pick
            tmp = pool[j]
            pool[j] = pool[pick]
            pool[pick] = tmp

        # The drawn cards must land in the row *before* the row is sorted. Sorting first would read
        # uninitialised memory -- `np.empty` hands back whatever was in the buffer, so the bug
        # shows up as negative card ids rather than as an exception.
        for j in range(hand_size):
            particles[i, j] = pool[j]

        # Sorted, so two hypotheses holding the same cards are the same array. Without a canonical
        # order the duplicate scan cannot recognise them, the marginal double-counts them, and the
        # entropy is computed over a distribution that does not exist.
        for j in range(1, hand_size):
            value = particles[i, j]
            back = j - 1
            while back >= 0 and particles[i, back] > value:
                particles[i, back + 1] = particles[i, back]
                back -= 1
            particles[i, back + 1] = value

    return particles, rng_state


def initial_particles(state, opponent: int, k: int, rng: RngStream) -> np.ndarray:
    """`k` hypotheses about the opponent's hand, shape `(k, hand_size)`, sorted per row.

    `rng` is an `RngStream` and is advanced past the draws, so the caller's stream position moves
    exactly as it would for any other consumer of that stream. The compiled code never sees the
    object, only its snapshot and the state it hands back.
    """
    unseen = np.empty(state.initial_deck, dtype=np.int16)
    u_len = _unseen_into(state, 1 - int(opponent), unseen)
    hand_size = int(state.hand_sizes[opponent])

    particles, rng_state = _draw_particles(unseen, u_len, hand_size, k, np.uint64(rng.snapshot()))
    rng.restore(int(rng_state))
    return particles


@njit(cache=True)
def _filter_particles(particles: np.ndarray, played_mask: np.uint64, hand_size: int):
    """Hard filter: shrink the particles that held an observed card, keep the rest, enforce size."""
    k, m = particles.shape
    kept = np.empty((k, m), dtype=np.int16)
    sizes = np.empty(k, dtype=np.int64)

    survivors = 0
    for i in range(k):
        new_len = m
        for j in range(m):
            if (played_mask & (_ONE << np.uint64(particles[i, j]))) != np.uint64(0):
                new_len -= 1
        if new_len != hand_size:
            continue
        position = 0
        for j in range(m):
            if (played_mask & (_ONE << np.uint64(particles[i, j]))) != np.uint64(0):
                continue
            kept[survivors, position] = particles[i, j]
            position += 1
        sizes[survivors] = new_len
        survivors += 1

    return kept[:survivors], sizes[:survivors]


@njit(cache=True)
def _popcount(value: np.uint64) -> int:
    count = 0
    remaining = value
    while remaining != np.uint64(0):
        count += int(remaining & np.uint64(1))
        remaining >>= np.uint64(1)
    return count


@njit(cache=True)
def _soft_weights(kept: np.ndarray, sizes: np.ndarray, soft_mask: np.uint64,
                  soft_delta: float, penalty: float) -> np.ndarray:
    """Log-weights from the behavioural signal.

    Symmetric in how many masked cards a particle holds: each hit adds, each miss subtracts. A
    one-sided boost would make "holds no masked card at all" the most likely hypothesis, which is
    not what "this card made the choice more likely" means.

    Deliberately a heuristic, not a fitted likelihood. Anything stronger would be a behavioural model
    of the opponent, which is the thing the ``lambda = 0`` default exists to keep out of the
    uninformed arm.
    """
    n = kept.shape[0]
    weights = np.empty(n, dtype=np.float64)
    mask_bits = _popcount(soft_mask)

    for i in range(n):
        hits = 0
        for j in range(sizes[i]):
            if (soft_mask & (_ONE << np.uint64(kept[i, j]))) != np.uint64(0):
                hits += 1
        weights[i] = penalty * soft_delta * (2.0 * hits - mask_bits)

    return weights


@njit(cache=True)
def _systematic_resample(kept: np.ndarray, sizes: np.ndarray, log_weights: np.ndarray,
                         k: int) -> np.ndarray:
    """Bootstrap `k` particles from `n` survivors. Deterministic: no RNG.

    Fixed offset at half a step. With equal weights and ``n == k`` this copies every particle exactly
    once, so a filter with nothing to learn keeps its diversity instead of shaking it up for free.

    Subtracting the maximum before exponentiating is not optional: ``penalty * soft_delta`` is
    attacker-controlled in the sweep, and an unguarded ``exp`` overflows to infinity above about
    709, which would silently collapse the filter onto a single particle.
    """
    n = kept.shape[0]
    weights = np.empty(n, dtype=np.float64)

    largest = -1e308
    for i in range(n):
        if log_weights[i] > largest:
            largest = log_weights[i]
    total = 0.0
    for i in range(n):
        weights[i] = float(np.exp(log_weights[i] - largest))
        total += weights[i]
    if not (total > 0.0):
        for i in range(n):
            weights[i] = 1.0 / n
        total = 1.0

    out = np.empty((k, sizes[0]), dtype=np.int16)
    cursor = 0
    cumulative = weights[0]
    for i in range(k):
        target = (i + 0.5) / k * total
        while cumulative < target and cursor < n - 1:
            cursor += 1
            cumulative += weights[cursor]
        for j in range(sizes[cursor]):
            out[i, j] = kept[cursor, j]

    return out


def apply_observation(particles: np.ndarray, observation: Observation,
                      soft_penalty_lambda: float = 0.0) -> np.ndarray:
    """Update the particle set with one observation. Returns exactly `k` particles.

    Three stages: filter on the hard evidence, reweight on the behavioural evidence, resample back
    to ``k``. The resample is unconditional even when nothing was learned, so the particle count is a
    constant and the buffer arithmetic stays trivial.

    Raises ``ValueError`` when no hypothesis survives. That means the caller's evidence contradicts
    the state it came from -- a tracking bug. Quietly re-seeding from the prior would turn the bug
    into a plausible posterior, which is the more expensive failure.
    """
    if particles.ndim != 2:
        raise ValueError(f"particles must be 2-D (k, hand_size), got shape {particles.shape}")

    kept, sizes = _filter_particles(particles, observation.played_mask, observation.hand_size)
    if kept.shape[0] == 0:
        raise ValueError(
            f"the observation eliminated every hypothesis: played_mask="
            f"{int(observation.played_mask):#x}, hand_size={observation.hand_size}, particle hand "
            f"size={particles.shape[1]}. Either the evidence contradicts the state it came from, or "
            f"the hand size passed here is wrong."
        )

    k = particles.shape[0]
    if soft_penalty_lambda != 0.0 and observation.soft_mask != np.uint64(0):
        log_weights = _soft_weights(kept, sizes, observation.soft_mask,
                                    observation.soft_delta, soft_penalty_lambda)
    else:
        # Uniform by decision, not by arithmetic accident. The resample below still runs, and
        # resampling on weights that happen to be equal would perturb the particle set for no
        # reason -- which is exactly what would make the uninformed arm drift.
        log_weights = np.zeros(kept.shape[0], dtype=np.float64)

    return _systematic_resample(kept, sizes, log_weights, k)


@njit(cache=True)
def marginals_and_entropy(particles: np.ndarray, unseen: np.ndarray, deck_size: int = 0):
    """Marginal probability per card id, and the Shannon entropy of that marginal in bits.

    The marginal is the pool average of the per-particle indicator vectors -- the "mean" in *mean
    Shannon entropy over the particle set*. Its entropy is then taken over the unseen cards after
    normalising, so it reads as "how many bits remain to say which unseen cards the opponent holds".

    At the uniform prior every unseen card has frequency ``m / U``, the normalised marginal is
    uniform, and the entropy is exactly ``log2(U)`` -- the ceiling, asserted directly in the tests
    rather than assumed.

    What it does *not* measure: asymmetry. Two filters with the same ``|unseen|`` and the same hand
    size score identically here even if one knows the hand exactly, because a concentrated marginal
    and a diffuse one have the same entropy once normalised. That is a real limitation of entropy
    alone as a phase-curve input, and the curve is plotted against round index and cards-left as
    well for exactly this reason.

    Returns a card-indexed array; cards outside ``unseen`` are exactly zero, because they are known
    not to be in the hand and a small non-zero there would be a straight bug.

    ``deck_size`` sets the array length, and callers that hold a state should always pass it. Left at
    0, the length is derived from the highest unseen card -- which is shorter than the deck whenever
    the visible cards sit at high ids, and a caller indexing by deck position then gets an
    ``IndexError`` where a 0.0 belongs. An out-of-range read here is a far more expensive failure than
    a slightly larger array.
    """
    k, width = particles.shape
    size = int(deck_size)
    for card in unseen:
        if int(card) + 1 > size:
            size = int(card) + 1

    marginals = np.zeros(size, dtype=np.float64)
    for i in range(k):
        for j in range(width):
            marginals[int(particles[i, j])] += 1.0
    for card in range(size):
        marginals[card] /= k

    total = 0.0
    for card in unseen:
        total += marginals[int(card)]

    entropy = 0.0
    if total > 0.0:
        for card in unseen:
            p = marginals[int(card)] / total
            if p > 0.0:
                entropy -= p * np.log2(p)

    return marginals, float(entropy)


def hand_probabilities(state, opponent: int, k: int, rng: RngStream) -> np.ndarray:
    """Marginal probability that each card is in `opponent`'s hand, before any observation.

    A card-indexed ``float64`` array: a card the observer can see, or that is on the table or in the
    discard, gets exactly 0.0. Small non-zero values there would be a straight bug, and returning a
    sparse dict would hide the difference.
    """
    particles = initial_particles(state, opponent, k, rng)
    unseen = np.asarray(unseen_cards(state, opponent), dtype=np.int16)
    marginals, _ = marginals_and_entropy(particles, unseen, state.initial_deck)
    return marginals


def informativeness(particles: np.ndarray, unseen: np.ndarray, deck_size: int = 0) -> float:
    """Mean Shannon entropy in bits of the particle set's marginal.

    Monotone non-increasing as evidence accumulates, because every surviving hypothesis is a subset
    of the previous ones and the unseen set only shrinks. The entropy itself does not depend on the
    array length, so ``deck_size`` is optional here; it is accepted only for symmetry.
    """
    _, entropy = marginals_and_entropy(particles, np.asarray(unseen, dtype=np.int16), deck_size)
    return entropy
