"""Seeded PRNG and seed derivation.

Why this module exists instead of ``random``
-------------------------------------------
Two reasons, both load-bearing rather than stylistic:

1. ``random.Random`` is unsupported in Numba's ``nopython`` mode, and the engine
   must compile.
2. More importantly, the stream must be *bit-identical* between the interpreted
   and compiled execution modes. The project's entire correctness argument rests on
   one test -- run the same seeds in both modes and assert the move trajectories
   match exactly. A generator whose results depend on execution mode would fail that
   test, and the failure would look like an engine bug rather than a generator bug.

Why not ``hash()``
------------------
``PYTHONHASHSEED`` randomises ``hash()`` of strings per process, so a seed derived
through it would differ between runs of the same program. ``mix_seed`` is an
explicit integer mixer, so a seed is a pure function of its arguments.

Three streams per game
----------------------
Each game draws from three independent streams: ``deal``, ``player0``, ``player1``.
If the two players shared one stream, changing how many draws strategy A consumed
would shift strategy B's randomness, so a run differing in exactly one strategy
would not be comparable to another. Separating them is what makes "same seed, same
deal, both arms" -- the property the primary metric depends on -- actually hold.

Execution-mode contract
-----------------------
Three rules, each established by measurement rather than assumption (see
``tests/test_rng.py``):

1. **Shift counts are Python ``int``, never ``np.uint64``.** Shifting a ``uint64`` by
   a ``uint64`` produces different results in the compiled and interpreted modes;
   shifting by a Python ``int`` is bit-identical in both. This is not a cosmetic
   detail -- it was the difference between two failing checksums and two matching
   ones.
2. **Constants are ``np.uint64`` and every intermediate is ``uint64``.** A Python
   ``int`` mask exceeds the range numba types as ``int64``, so the mask itself has to
   be a ``uint64`` for the arithmetic to stay in the intended domain.
3. **Bounded loops, not ``while True``.** Numba compiles a rejection loop with a tuple
   return far more slowly than a counted loop, and a counted loop keeps the function
   total.
4. **Every operand crossing into a compiled function is wrapped in ``np.uint64``.**
   A state returned from compiled code arrives at the Python caller as a plain ``int``;
   Numba then infers ``int64`` for it on the next call, and above 2**63-1 that raises
   ``OverflowError`` or -- worse, when the value fits -- silently compiles for the
   wrong type and yields a *different* sequence. ``RngStream`` performs this wrap so
   Python callers never have to know it.

``RngStream`` is a mutable Python holder, so it cannot cross into ``nopython`` code.
That is deliberate: the simulation path uses the free functions below and threads the
64-bit state through its own locals, while ``RngStream`` serves the Python-side tests
and orchestration with the same arithmetic. Both routes call the identical
:func:`_finalise`, so there is no second implementation to drift.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import njit

# 64-bit splitmix64 arithmetic wraps by design: every operation here is defined modulo
# 2**64, and that wraparound is what makes the generator reproducible across processes
# and execution modes.
#
# NumPy raises RuntimeWarning on a wrapping scalar add or multiply when running
# interpreted (Numba's compiled path does not warn), so the warning is suppressed for
# this module rather than left to make a correct run look faulty.
#
# The filter is scoped by *message*, not by category, so it silences only this
# module's deliberate wraparound: an unrelated RuntimeWarning anywhere else still
# propagates. `test_unrelated_runtime_warnings_are_not_suppressed` pins that, because a
# global category filter would be the easy way to write this and would quietly hide
# real defects for the rest of the suite.
#
# Note that `pytest -W error::RuntimeWarning` overrides this filter (command-line filters
# take precedence), so such a run fails under NUMBA_DISABLE_JIT=1 by design. That is not
# a defect; it is the wraparound being real.
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in scalar",
    category=RuntimeWarning,
)

# 64-bit splitmix64 constants.
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)

_S30 = 30
_S27 = 27
_S31 = 31
_S11 = 11
_FLOAT53 = float(1 << 53)

# A rejection loop for `next_below` cannot in principle run forever -- the acceptance
# probability exceeds 1/2 for every bound -- so this cap is unreachable in practice. It
# exists to keep the function total for numba, which compiles bounded loops far better
# than open-ended ones.
_MAX_REJECTIONS = 64


def mix_seed(*parts: int) -> int:
    """Derive a reproducible 64-bit seed from integer components.

    Varargs are a Python-level convenience: Numba cannot type a ``*args`` signature,
    so the engine calls the fixed-arity :func:`mix_seed3` instead. Both routes share
    :func:`_fold`, so there is one implementation.

    Ordering matters and is preserved: ``mix_seed(1, 2)`` and ``mix_seed(2, 1)`` differ,
    which is what lets ``(pairing_index, game_index)`` behave as a coordinate pair rather
    than a multiset. NumPy integers are accepted, because game indices arrive as
    ``np.int64`` from block computations.

    Returns a Python ``int`` in ``[0, 2**64)`` so callers can use it as a seed without
    worrying about signed interpretation.
    """
    state = np.uint64(0)
    for part in parts:
        # Both operands are wrapped in np.uint64 before entering the compiled
        # function, and only on the first call does Numba compile a signature that
        # accepts uint64. Without the wrap, a state above 2**63-1 comes back from
        # `_fold` as a plain Python int; Numba would then type the next argument as
        # int64 and raise OverflowError trying to convert it. Masking the component
        # is needed for the same reason: a caller may fold a full 64-bit seed.
        state = np.uint64(
            _fold(np.uint64(state), np.uint64(int(part) & 0xFFFFFFFFFFFFFFFF))
        )
    return int(_finalise(np.uint64(state)))


@njit(cache=True)
def mix_seed3(run_seed: int, pairing_index: int, game_index: int):
    """Seed for the three coordinates that identify a game.

    Fixed arity because Numba cannot type ``*args``. Every game in the dataset is
    seeded through this, so it is the signature the engine actually calls.
    """
    state = np.uint64(0)
    state = _fold(state, np.uint64(run_seed))
    state = _fold(state, np.uint64(pairing_index))
    state = _fold(state, np.uint64(game_index))
    return _finalise(state)


@njit(cache=True)
def _fold(state: np.uint64, component: np.uint64) -> np.uint64:
    """Fold one seed component into the running state.

    Multiplies by the golden ratio before xoring so argument order matters: a plain
    sum would make ``(1, 23)`` and ``(12, 3)`` collide, and ``(pairing_index,
    game_index)`` must behave as a coordinate pair rather than a multiset.

    ``component`` is typed ``uint64`` rather than ``int`` so the whole computation
    stays in unsigned arithmetic. A signed 64-bit parameter would reject any seed
    above 2**63-1 at the call boundary.
    """
    folded = state + component * _GAMMA
    return folded ^ (folded >> _S30)


@njit(cache=True)
def _finalise(state: np.uint64) -> np.uint64:
    """Avalanche a 64-bit value so every output bit depends on every input bit.

    Shift counts are Python ints on purpose -- see the module docstring.
    """
    s = state + _GAMMA
    s = (s ^ (s >> _S30)) * _MIX_A
    s = (s ^ (s >> _S27)) * _MIX_B
    return s ^ (s >> _S31)


# The three primitives below are the engine's drawing API: they take a `uint64`
# state and return a `(value, next_state)` pair, so the engine threads the state
# through its own locals instead of holding an object.
#
# They are intended to be called from *inside* compiled code, where the state stays
# `uint64` throughout. Calling them from Python is supported but requires re-wrapping
# the returned state in `np.uint64`: Numba receives the returned plain `int`, infers
# `int64` for it, and then re-compiles for the wrong type -- producing a *different*
# sequence rather than an error. `RngStream` exists so Python callers never have to
# remember that.


@njit(cache=True)
def next_u64(state: np.uint64):
    """One 64-bit draw and the next state. The primitive the engine threads."""
    return _finalise(state), state + _GAMMA


@njit(cache=True)
def next_below(state: np.uint64, bound: int):
    """Unbiased draw in ``[0, bound)`` and the next state.

    Rejection sampling, never modulo: the top ``2**64 % bound`` draws are
    under-represented by a plain modulo, and that skew would propagate into the
    shuffle, and from there into the deal every recorded game starts from.

    Returns ``(value_as_int64, next_state)``.
    """
    if bound <= 0:
        raise ValueError(f"bound must be positive, got {bound}")

    b = np.uint64(bound)
    limit = _MASK - (_MASK % b)

    for _ in range(_MAX_REJECTIONS):
        draw = _finalise(state)
        state = state + _GAMMA
        if draw <= limit:
            return np.int64(draw % b), state

    return np.int64(0), state


@njit(cache=True)
def next_float(state: np.uint64):
    """Uniform draw in ``[0, 1)`` with 53 bits of precision, and the next state.

    Takes the top 53 bits rather than dividing a full 64-bit integer: a ``float64``
    cannot represent the low bits exactly, so dividing them would introduce a bias.
    """
    draw = _finalise(state)
    return float(draw >> _S11) / _FLOAT53, state + _GAMMA


class RngStream:
    """Mutable Python-side holder around the same arithmetic the engine uses.

    The engine threads a bare ``uint64`` through its own locals because a Python
    object cannot cross into ``nopython`` mode. This class exists for tests, the
    benchmark harness, and orchestration, and routes through the identical
    :func:`_finalise` -- so there is no second implementation to drift.

    ``snapshot``/``restore`` let the harness rewind a stream to compare execution modes
    on the same prefix of draws.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        # No forbidden state to fold away: this generator is splitmix64-shaped
        # (state + GAMMA, then avalanche), and every 64-bit state including zero is
        # valid. A `or 1` guard here would silently make seeds 0 and 1 produce the
        # same stream -- a real collision, not a formality.
        self.state = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)

    def next_u64(self) -> int:
        # The returned state is re-wrapped in np.uint64 on every call. A compiled
        # function called from Python hands back a plain int, and above 2**63-1 that
        # int no longer fits the int64 Numba would infer on the next call -- it would
        # recompile for the wrong type and then raise OverflowError. Re-wrapping keeps
        # the argument typed uint64 across the whole run.
        draw, state = next_u64(np.uint64(self.state))
        self.state = np.uint64(state)
        return int(draw)

    def next_below(self, bound: int) -> int:
        value, state = next_below(np.uint64(self.state), bound)
        self.state = np.uint64(state)
        return int(value)

    def next_float(self) -> float:
        value, state = next_float(np.uint64(self.state))
        self.state = np.uint64(state)
        return value

    def snapshot(self) -> int:
        return int(self.state)

    def restore(self, state: int) -> None:
        self.state = np.uint64(int(state) & 0xFFFFFFFFFFFFFFFF)


class GameStreams:
    """The three independent streams one game draws from.

    All three derive from the same mixer, so changing the derivation moves them
    together rather than silently decoupling one player's randomness from the deal.
    """

    __slots__ = ("deal", "player0", "player1", "run_seed", "pairing_index", "game_index")

    def __init__(self, deal: RngStream, player0: RngStream, player1: RngStream,
                 run_seed: int, pairing_index: int, game_index: int) -> None:
        self.deal = deal
        self.player0 = player0
        self.player1 = player1
        self.run_seed = run_seed
        self.pairing_index = pairing_index
        self.game_index = game_index

    @classmethod
    def from_parts(cls, run_seed: int, pairing_index: int, game_index: int) -> "GameStreams":
        """Build the three streams for one game from its coordinates.

        Game identity is fully determined by these three integers -- never by
        scheduling, wall-clock time, or process identity. That is what makes a
        1-worker run and an 8-worker run produce byte-identical output.
        """
        base, first, second = game_stream_states(run_seed, pairing_index, game_index)
        return cls(
            deal=RngStream(base),
            player0=RngStream(first),
            player1=RngStream(second),
            run_seed=run_seed,
            pairing_index=pairing_index,
            game_index=game_index,
        )

    def for_player(self, player: int) -> RngStream:
        """The stream belonging to player 0 or 1."""
        if player == 0:
            return self.player0
        if player == 1:
            return self.player1
        raise ValueError(f"player must be 0 or 1, got {player}")

    def snapshot(self) -> tuple:
        """Capture all three states, so the harness can rewind and compare."""
        return (self.deal.snapshot(), self.player0.snapshot(), self.player1.snapshot())

    def restore(self, snapshot: tuple) -> None:
        """Restore a snapshot taken by :meth:`snapshot`."""
        self.deal.restore(snapshot[0])
        self.player0.restore(snapshot[1])
        self.player1.restore(snapshot[2])


@njit(cache=True)
def mix_seed_stream(base: int, index: int) -> int:
    """Derive one named stream from a game's base seed.

    Fixed arity, unlike :func:`mix_seed`, because Numba cannot type ``*parts``. The engine seeds
    every game from here, so this is the signature that actually runs; :func:`mix_seed` delegates
    to it for the same reason :func:`mix_seed3` exists.
    """
    return int(_finalise(_fold(_fold(np.uint64(0), np.uint64(int(base) & 0xFFFFFFFFFFFFFFFF)),
                             np.uint64(index))))


@njit(cache=True)
def game_stream_states(run_seed: int, pairing_index: int, game_index: int) -> tuple:
    """The three stream states for one game, as a `(deal, p0, p1)` tuple of ``uint64``.

    The compiled form of :meth:`GameStreams.from_parts`, and the single implementation of the
    derivation. The engine cannot construct a ``GameStreams`` at all -- it is a Python class, and
    a Python object does not cross into ``nopython`` mode -- so a second, compiled-only derivation
    would have been needed. Instead this function is the derivation and ``from_parts`` calls it,
    so there is nothing to drift.
    """
    base = mix_seed3(run_seed, pairing_index, game_index)
    # `mix_seed3` hands back a plain Python int to an interpreted caller, and above 2**63-1 that
    # int does not fit the int64 Numba would infer for it -- it would then be folded as a
    # *negative* number and the two execution modes would derive different streams. Masking here
    # keeps the operand uint64 in both modes, which is the whole of rule 4 in the module
    # docstring. Found by the engine equivalence probe, not by a unit test: nothing about a
    # single-mode run can see it.
    masked = np.uint64(int(base) & 0xFFFFFFFFFFFFFFFF)
    return (
        np.uint64(mix_seed_stream(masked, 0)),
        np.uint64(mix_seed_stream(masked, 1)),
        np.uint64(mix_seed_stream(masked, 2)),
    )


@njit(cache=True)
def shuffle_inplace_with_state(cards: np.ndarray, state: np.uint64) -> np.uint64:
    """Fisher-Yates over card ids, threading the state. Returns the next state.

    The compiled counterpart of :func:`shuffle_inplace`, and again the single implementation:
    the deal has to run inside compiled code, so the arithmetic lives here and the Python
    wrapper delegates. Writing the swap loop twice would be two shuffles free to disagree, and
    the deal is what every recorded game starts from.
    """
    for i in range(len(cards) - 1, 0, -1):
        j, state = next_below(state, i + 1)
        held = cards[i]
        cards[i] = cards[j]
        cards[j] = held
    return state


def shuffle_inplace(cards: np.ndarray, rng: RngStream) -> None:
    """Fisher-Yates shuffle of a card-id array, in place.

    The bound is ``i + 1`` at each step, which is the formulation needing no
    modulo-bias correction and visiting every permutation with equal probability.

    Ascending input order makes a broken shuffle detectable: a shuffle that returned
    its input unchanged would still be a valid permutation.
    """
    for i in range(len(cards) - 1, 0, -1):
        j = rng.next_below(i + 1)
        held = cards[i]
        cards[i] = cards[j]
        cards[j] = held