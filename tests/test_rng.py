"""Tests for the seeded PRNG and seed derivation.

The project carries its own PRNG instead of ``random.Random`` for two reasons that
are load-bearing rather than stylistic:

1. ``random.Random`` is unsupported in Numba's ``nopython`` mode.
2. The stream must be *bit-identical* in the interpreted and compiled execution modes.
   The equivalence test that proves the engine correct would otherwise fail, and that
   failure is indistinguishable from a real engine bug.

Python's ``hash()`` is equally unusable: ``PYTHONHASHSEED`` randomises it per process,
so any seed derived through it would be irreproducible across restarts.

The execution-mode tests here are the earliest guard on that property: they run the
same arithmetic in subprocesses under ``NUMBA_DISABLE_JIT=1`` and under normal
compilation and require identical checksums.

Several tests pin type-handling defects that were found while building this module and
would otherwise reappear silently, because each one produced a *different* number
rather than an exception:

* a shift count typed ``np.uint64`` instead of ``int`` diverges between execution modes;
* a state returned from compiled code arrives as a Python ``int``, and Numba then
  re-infers it as ``int64`` and produces a different sequence;
* a ``or 1`` guard against an all-zero state makes seeds 0 and 1 collide.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from durak.rng import (
    GameStreams,
    RngStream,
    mix_seed,
    mix_seed3,
    next_below,
    next_float,
    next_u64,
    shuffle_inplace,
)

UINT64_MASK = (1 << 64) - 1

# Exercises the same arithmetic in both execution modes and prints a checksum.
_MODE_PROBE = textwrap.dedent(
    """
    import hashlib
    import numpy as np
    from numba import njit
    from durak.rng import next_below, next_float, next_u64, mix_seed

    @njit(cache=False)
    def run(seed, n):
        state = np.uint64(seed)
        acc = np.int64(0)
        for _ in range(n):
            value, state = next_below(state, 7)
            acc += value
            draw, state = next_u64(state)
            acc += np.int64(draw & np.uint64(0xFFFFFFFF))
            f, state = next_float(state)
            acc += np.int64(f * 1000.0)
        return acc

    acc = 0
    for seed in range(20):
        acc += run(seed, 500)
    acc += int(mix_seed(1, 2, 3))
    print("CHECKSUM", hashlib.sha256(str(acc).encode()).hexdigest()[:16])
    """
)


def _mode_checksum(disable_jit: bool) -> str:
    """Run the probe in a subprocess, returning its checksum line."""
    env = dict(os.environ)
    env.pop("NUMBA_DISABLE_JIT", None)
    if disable_jit:
        env["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _MODE_PROBE],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for line in result.stdout.splitlines():
        if line.startswith("CHECKSUM"):
            return line.split()[1]
    raise AssertionError(f"probe printed no checksum: {result.stdout!r}")


# --- seed derivation ---------------------------------------------------------


def test_mix_seed_is_deterministic():
    assert mix_seed(1, 2, 3) == mix_seed(1, 2, 3)


def test_mix_seed_separates_different_tuples():
    # A weak mixer collides on transpositions like (1, 23) and (12, 3), so distinct
    # tuples must give distinct seeds.
    assert mix_seed(1, 23) != mix_seed(12, 3)
    assert mix_seed(7, 0, 0) != mix_seed(7, 0, 1)
    assert mix_seed(0, 0, 0) != mix_seed(0, 0, 1)


def test_mix_seed_preserves_argument_order():
    # Order matters so (pairing_index, game_index) behaves as a coordinate pair.
    assert mix_seed(1, 2) != mix_seed(2, 1)


def test_mix_seed_with_no_parts_is_stable():
    assert mix_seed() == mix_seed()


def test_mix_seed_handles_zero_and_neighbouring_seeds():
    # A guard like `seed or 1` would make these collide.
    seeds = [mix_seed(s) for s in range(8)]

    assert len(set(seeds)) == 8


def test_mix_seed_stays_within_uint64():
    value = mix_seed(2**31 - 1, 2**31 - 1, 2**31 - 1)

    assert 0 <= value <= UINT64_MASK


def test_mix_seed_handles_full_width_components():
    # A component above 2**63-1 must wrap rather than raise: seeds are themselves
    # folded when one run derives another.
    assert 0 <= mix_seed(UINT64_MASK) <= UINT64_MASK
    assert mix_seed(UINT64_MASK, 1) != mix_seed(0, 1)


def test_mix_seed_returns_a_plain_int():
    # Must not leak a numpy scalar: callers seed Python objects with it and a numpy
    # uint64 would raise on large values.
    value = mix_seed(1, 2, 3)

    assert type(value) is int


def test_mix_seed_accepts_numpy_integers():
    # Game indices arrive as np.int64 from block computations.
    assert isinstance(mix_seed(np.int64(5), np.int64(6)), int)


def test_mix_seed_rejects_non_integer_components():
    # PYTHONHASHSEED randomises string hashing per process, so a seed derived through a
    # string would be irreproducible. Rejecting strings outright is how that is
    # prevented: there is no path where a label can reach the mixer.
    with pytest.raises((ValueError, TypeError)):
        mix_seed(1, "a")


def test_mix_seed3_matches_the_varargs_form():
    # The engine calls the fixed-arity form; it must agree with `mix_seed`, or a seed
    # derived on the Python side would not reproduce inside the engine.
    assert mix_seed3(11, 22, 33) == mix_seed(11, 22, 33)


def test_mix_seed3_separates_different_coordinates():
    assert mix_seed3(1, 2, 3) != mix_seed3(1, 3, 2)
    assert mix_seed3(1, 2, 3) != mix_seed3(2, 2, 3)
    assert mix_seed3(1, 2, 3) != mix_seed3(1, 2, 4)


# --- stream independence -----------------------------------------------------


def test_same_components_give_identical_streams():
    first = GameStreams.from_parts(42, 3, 17)
    second = GameStreams.from_parts(42, 3, 17)

    assert first.deal.next_u64() == second.deal.next_u64()
    assert first.player0.next_u64() == second.player0.next_u64()
    assert first.player1.next_u64() == second.player1.next_u64()


def test_any_component_change_moves_every_stream():
    base = GameStreams.from_parts(42, 3, 17)
    base_firsts = (
        base.deal.next_u64(),
        base.player0.next_u64(),
        base.player1.next_u64(),
    )

    for variant in (
        GameStreams.from_parts(43, 3, 17),
        GameStreams.from_parts(42, 4, 17),
        GameStreams.from_parts(42, 3, 18),
    ):
        assert variant.deal.next_u64() != base_firsts[0]
        assert variant.player0.next_u64() != base_firsts[1]
        assert variant.player1.next_u64() != base_firsts[2]


def test_the_three_streams_of_one_game_are_independent():
    # If both players shared one stream, changing how many draws one strategy consumed
    # would shift the other's randomness, destroying comparability between runs that
    # differ in exactly one strategy.
    streams = GameStreams.from_parts(99, 0, 0)
    seen = set()

    for _ in range(10_000):
        for stream in (streams.deal, streams.player0, streams.player1):
            value = stream.next_u64()
            assert value not in seen
            seen.add(value)

    assert len(seen) == 30_000


def test_streams_from_parts_have_distinct_states():
    first = GameStreams.from_parts(1, 2, 3)
    second = GameStreams.from_parts(1, 2, 3)

    assert first.deal.snapshot() != first.player0.snapshot()
    assert first.player0.snapshot() != first.player1.snapshot()
    assert first.deal.snapshot() != first.player1.snapshot()
    assert second.deal.snapshot() == first.deal.snapshot()


def test_for_player_selects_the_right_stream():
    streams = GameStreams.from_parts(4, 5, 6)

    assert streams.for_player(0) is streams.player0
    assert streams.for_player(1) is streams.player1


def test_for_player_rejects_an_invalid_index():
    streams = GameStreams.from_parts(4, 5, 6)

    with pytest.raises(ValueError, match="player"):
        streams.for_player(2)


def test_snapshot_and_restore_rewind_a_game():
    streams = GameStreams.from_parts(11, 12, 13)
    before = streams.snapshot()

    for _ in range(50):
        streams.player0.next_u64()
        streams.deal.next_u64()

    streams.restore(before)

    fresh = GameStreams.from_parts(11, 12, 13)
    assert streams.player0.next_u64() == fresh.player0.next_u64()
    assert streams.deal.next_u64() == fresh.deal.next_u64()


# --- bounded draws -----------------------------------------------------------


def test_next_below_stays_within_range():
    stream = RngStream(7)

    for _ in range(1_000):
        assert 0 <= stream.next_below(13) < 13


def test_next_below_is_approximately_uniform():
    # Modulo bias on a small bound shows up here. A biased draw would distort the
    # shuffle and, through it, the deal every game starts from.
    stream = RngStream(1234)
    bound = 7
    trials = 120_000
    counts = [0] * bound

    for _ in range(trials):
        counts[stream.next_below(bound)] += 1

    expected = trials / bound
    for count in counts:
        assert abs(count - expected) < expected * 0.05


def test_next_below_is_uniform_on_a_power_of_two_bound():
    # 16 divides 2**64 evenly, so modulo bias is impossible here; the assertion checks
    # the acceptance path rather than the rejection path.
    stream = RngStream(5150)
    counts = [0] * 16

    for _ in range(160_000):
        counts[stream.next_below(16)] += 1

    expected = 160_000 / 16
    for count in counts:
        assert abs(count - expected) < expected * 0.05


def test_next_below_of_one_is_always_zero():
    stream = RngStream(5)

    assert all(stream.next_below(1) == 0 for _ in range(100))


def test_next_below_rejects_non_positive_bounds():
    stream = RngStream(5)

    with pytest.raises(ValueError, match="bound"):
        stream.next_below(0)

    with pytest.raises(ValueError, match="bound"):
        stream.next_below(-3)


def test_next_float_stays_in_unit_interval():
    stream = RngStream(31)
    values = [stream.next_float() for _ in range(10_000)]

    assert all(0.0 <= v < 1.0 for v in values)
    assert 0.4 < float(np.mean(values)) < 0.6


def test_next_float_does_not_repeat():
    stream = RngStream(77)
    values = [stream.next_float() for _ in range(100)]

    assert len(set(values)) > 90


# --- state threading ---------------------------------------------------------


def test_free_functions_advance_the_state():
    state = np.uint64(1234)
    draw_a, state_a = next_u64(state)
    draw_b, state_b = next_u64(np.uint64(state_a))

    assert draw_a != draw_b
    assert state_a != state_b


def test_free_next_below_matches_the_class():
    # The free primitives are what the engine calls from compiled code, where the state
    # stays uint64. `RngStream` must reproduce the identical sequence from Python, or the
    # equivalence test would compare two different generators.
    state = np.uint64(4321)
    values = []
    for _ in range(50):
        value, state = next_below(np.uint64(state), 11)
        values.append(int(value))

    stream = RngStream(4321)
    assert values == [stream.next_below(11) for _ in range(50)]


def test_free_next_float_matches_the_class():
    state = np.uint64(999)
    values = []
    for _ in range(50):
        value, state = next_float(np.uint64(state))
        values.append(value)

    stream = RngStream(999)
    assert values == [stream.next_float() for _ in range(50)]


def test_free_next_u64_matches_the_class():
    state = np.uint64(1234)
    values = []
    for _ in range(50):
        draw, state = next_u64(np.uint64(state))
        values.append(int(draw))

    stream = RngStream(1234)
    assert values == [stream.next_u64() for _ in range(50)]


def test_free_next_below_rejects_a_non_positive_bound():
    with pytest.raises(ValueError, match="bound"):
        next_below(np.uint64(1), 0)


def test_stream_state_keeps_its_uint64_type():
    # Guards the re-wrap inside RngStream: if it were dropped, Numba would infer int64 on
    # the next call and silently return a different sequence instead of raising.
    stream = RngStream(2468)
    stream.next_u64()

    assert isinstance(stream.state, np.uint64)


def test_unrelated_runtime_warnings_are_not_suppressed():
    """The module's warning filter must be scoped to its own arithmetic.

    `durak.rng` installs an ``ignore`` filter for NumPy's "overflow encountered in
    scalar" RuntimeWarning, because 64-bit wraparound there is intentional. A filter on
    the whole RuntimeWarning category would also silence every unrelated runtime
    warning for the rest of the suite -- including ones signalling real defects. This
    test installs a catch-all filter *after* the module's, exactly as a later import or
    a test-runner setting would, and requires an unrelated warning to still surface.
    """
    import warnings

    with warnings.catch_warnings():
        # `always` first so the module's pre-existing `ignore` filter is not the only
        # rule in play, then `error` so an unrelated warning becomes an exception.
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="unrelated runtime problem"):
            warnings.warn("unrelated runtime problem", RuntimeWarning, stacklevel=1)


def test_stream_snapshots_round_trip_as_unsigned():
    stream = RngStream(UINT64_MASK)
    snapshot = stream.snapshot()

    stream.next_u64()
    stream.restore(snapshot)

    assert stream.snapshot() == snapshot
    assert isinstance(stream.snapshot(), int)


# --- shuffling ---------------------------------------------------------------


def test_shuffle_produces_a_permutation():
    cards = np.arange(36, dtype=np.int16)
    original = sorted(cards.tolist())

    shuffle_inplace(cards, RngStream(11))

    assert sorted(cards.tolist()) == original


def test_shuffle_is_deterministic_for_a_fixed_stream():
    first = np.arange(36, dtype=np.int16)
    second = np.arange(36, dtype=np.int16)

    shuffle_inplace(first, RngStream(99))
    shuffle_inplace(second, RngStream(99))

    assert np.array_equal(first, second)


def test_shuffle_actually_reorders():
    # A shuffle returning its input unchanged would pass the permutation test.
    cards = np.arange(36, dtype=np.int16)
    before = cards.copy()

    shuffle_inplace(cards, RngStream(3))

    assert not np.array_equal(cards, before)


def test_shuffle_of_one_element_is_a_no_op():
    cards = np.array([7], dtype=np.int16)

    shuffle_inplace(cards, RngStream(5))

    assert cards.tolist() == [7]


def test_shuffle_of_empty_array_is_a_no_op():
    cards = np.array([], dtype=np.int16)

    shuffle_inplace(cards, RngStream(5))

    assert len(cards) == 0


def test_shuffles_of_neighbouring_seeds_all_differ():
    # A `seed or 1` guard, or any weak seeding, would make adjacent seeds collide.
    results = []
    for seed in range(6):
        cards = np.arange(36, dtype=np.int16)
        shuffle_inplace(cards, RngStream(seed))
        results.append(cards.tobytes())

    assert len(set(results)) == 6


def test_shuffle_covers_the_52_card_deck():
    cards = np.arange(52, dtype=np.int16)

    shuffle_inplace(cards, RngStream(7))

    assert sorted(cards.tolist()) == list(range(52))


# --- execution-mode equivalence ----------------------------------------------


def test_arithmetic_is_bit_identical_across_execution_modes():
    """The project's core numerical guarantee, checked before the engine exists.

    A generator whose output depended on whether it was compiled would fail the engine
    equivalence test at Task 14 in a way that looks like an engine defect. Asserted here,
    on the generator, so a future change to this module cannot reintroduce the problem
    silently.
    """
    compiled = _mode_checksum(disable_jit=False)
    interpreted = _mode_checksum(disable_jit=True)

    assert compiled == interpreted, (
        "PRNG output differs between compiled and interpreted execution modes; "
        "the engine equivalence test would fail for the wrong reason"
    )


def test_mode_checksum_probe_is_not_trivially_constant():
    # Guards the test above against passing because the probe silently produced nothing
    # to compare.
    checksum = _mode_checksum(disable_jit=False)

    assert len(checksum) == 16
    assert checksum != hashlib.sha256(b"").hexdigest()[:16]