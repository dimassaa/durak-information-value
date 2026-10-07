"""The tournament grid: which matchups run, and under which first-attacker regime.

The grid is the experiment's design, stated once and in one place. A pairing omitted here is a
question the dataset cannot answer, and a pairing added ad hoc per run is a dataset that cannot be
compared with the last one.

Why both directions
--------------------
Running `A` vs `B` but not `B` vs `A` confounds the two strategies' relative strength with the
first-attacker advantage, because `A` attacks first in the first direction and defends in the
second. Both are therefore generated, and the pair is treated as two independent matchups rather
than one matchup with two arms.

Why `RANDOM` vs `RANDOM`
------------------------
It is the only self-pairing, and it is not redundant: `RANDOM` vs `RANDOM` = 0.5 is spec 7.3's only
exact assertion, the symmetry test of the whole engine. It needs a cell in the grid to run in. No
other self-pairing is generated, because no report consumes one.

Why two regimes
---------------
Spec 8.4. Under `RULE_DERIVED` the first attacker follows the low-trump rule, which is the honest
"as played" distribution and what the primary metric uses. Under `ROTATION_BALANCED` the seat is
forced alternately, so each strategy attacks exactly half the time regardless of what the deal
gave it. Without the second, a positional effect and a dealing effect are confounded and cannot be
separated; the difference between the regimes is itself reported as a metric.

Ordering
--------
Cells are emitted in a fixed order and carry a dense `pairing_index`. The index is a seed
coordinate -- `game_seeds(run_seed, pairing_index, ...)` derives from it -- so any reordering
between two runs of the same configuration would produce a different dataset from the same
configuration.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from durak.config import MAX_SEED
from durak.rng import mix_seed3
from durak.rules import USE_RULE
from durak.strategies import StrategyId
from durak.strategies.registry import registered_ids

#: The two first-attacker regimes of spec 8.4, in reporting order.
RULE_DERIVED = "RULE_DERIVED"
ROTATION_BALANCED = "ROTATION_BALANCED"
REGIMES = (RULE_DERIVED, ROTATION_BALANCED)


class Pairing(NamedTuple):
    """One cell of the tournament grid.

    `pairing_index` is the cell's position in :func:`pairing_grid` and is part of every seed
    derived from it, so it must be stable across runs of the same configuration.
    """

    pairing_index: int
    strategy_p0: int
    strategy_p1: int
    deck_size: int
    first_attacker_regime: str

    @property
    def label(self) -> str:
        """A human-readable cell name, for manifests and report tables."""
        from durak.strategies import policy_names

        return (
            f"{policy_names()[self.strategy_p0]}_vs_"
            f"{policy_names()[self.strategy_p1]}_{self.deck_size}_{self.first_attacker_regime}"
        )


def pairing_grid(strategy_ids, deck_sizes=(36,), both_directions: bool = True,
                 regimes=REGIMES) -> list:
    """Every matchup to run, as a flat ordered list of :class:`Pairing`.

    A single strategy that is not `RANDOM` yields an **empty** grid: there is nothing to pair it against,
    and a strategy playing itself is only meaningful as the symmetry baseline the document specifies.
    Callers must therefore always include an opponent -- the runner raises rather than reporting a
    zero-game success, and the sweeps pair against `RANDOM`.

    `strategy_ids` are the strategies under test; each appears against every other, plus
    ``RANDOM`` against itself. `both_directions` emits each unordered pairing twice, once per
    seat assignment.

    Raises on an unregistered strategy id rather than skipping it: a grid that silently dropped an
    unknown id would produce a dataset missing a cell, and nothing downstream would say so.
    """
    known = set(registered_ids())
    for strategy_id in strategy_ids:
        if int(strategy_id) not in known:
            raise ValueError(
                f"unknown strategy id {strategy_id}; registered ids are {sorted(known)}. A grid "
                f"that skipped it would silently omit a matchup from the dataset."
            )

    ordered = [int(s) for s in strategy_ids]
    cells = []
    random_id = int(StrategyId.RANDOM)

    for regime in regimes:
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
        for deck_size in deck_sizes:
            for i, first in enumerate(ordered):
                for second in ordered[i:]:
                    # Only `RANDOM` is paired with itself. Every other self-pairing would be a
                    # matchup no report consumes, and emitting them would inflate the run without
                    # adding information -- and, worse, would make a reader think `GREEDY`'s win
                    # rate against itself is a meaningful number rather than an artefact of the
                    # grid.
                    if first == second and first != random_id:
                        continue
                    cells.append((first, second, int(deck_size), regime))
                    if both_directions and first != second:
                        cells.append((second, first, int(deck_size), regime))

    return [
        Pairing(index, p0, p1, int(deck), regime)
        for index, (p0, p1, deck, regime) in enumerate(cells)
    ]


def game_block_ranges(game_count: int, block_size: int):
    """`(start, stop)` ranges partitioning `[0, game_count)` into blocks of at most `block_size`.

    The unit of parallelism. A block is identified by its `game_start` alone, because a game's seed is
    derived from the absolute game index -- so the same range means the same games no matter which worker
    runs it, in which order, or after a resume. That is what makes a 1-worker run equal an 8-worker run,
    and it is asserted in `tests/test_determinism.py` over several worker counts and block sizes.

    The last block is short rather than padded: padding would play games the manifest did not ask for and
    would have to be trimmed back out, which is the kind of round trip that loses rows.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if game_count < 0:
        raise ValueError(f"game_count must be non-negative, got {game_count}")
    if game_count == 0:
        return []
    return [(start, min(start + block_size, game_count))
            for start in range(0, game_count, block_size)]


def game_seeds(run_seed: int, pairing_index: int, game_count: int) -> np.ndarray:
    """The `int64` seeds for `game_count` games of one pairing.

    Derived from `(run_seed, pairing_index, game_index)` and nothing else, so the same
    configuration reproduces the same dataset on any machine, in any order, at any block size.

    Narrowed to 63 bits so the value fits a signed column. A full uint64 written into an int64
    field wraps silently when compiled and raises when interpreted -- the third site of that
    defect, recorded in the Stage 06 report.
    """
    if run_seed < 0 or run_seed > MAX_SEED:
        raise ValueError(f"run_seed must be in [0, 2**31 - 1], got {run_seed}")
    if game_count < 0:
        raise ValueError(f"game_count must be non-negative, got {game_count}")

    seeds = np.empty(game_count, dtype=np.int64)
    for index in range(game_count):
        value = mix_seed3(run_seed, pairing_index, index)
        seeds[index] = np.int64(np.uint64(value) & np.uint64(0x7FFFFFFFFFFFFFFF))

    return seeds


def first_attacker_override(regime: str, game_index: int) -> int:
    """Which seat starts, or :data:`durak.rules.USE_RULE` to let the rules decide.

    ``ROTATION_BALANCED`` alternates on ``game_index % 2``, which makes the split exactly half and
    half for any even game count. Alternating on the *game index* rather than on a draw matters:
    the rotation then holds under any block partitioning, because a game's seat does not depend on
    which worker ran it.
    """
    if regime == RULE_DERIVED:
        return USE_RULE
    if regime == ROTATION_BALANCED:
        return game_index % 2
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")