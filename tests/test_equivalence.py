"""The equivalence tests: one set of games, one result, whatever the execution mode or worker count.

This is the file the project's claims rest on. A win rate is a number produced by running games, and a
number produced by code that silently differs between a laptop and a datacentre is not a result. Three
properties are asserted, each one able to fail on its own:

1. Compiled and interpreted produce identical records. `NUMBA_DISABLE_JIT=1` runs in a subprocess because the
   variable has to be set before Numba is imported. A uint64 crossing the compiled boundary, or a signed
   column receiving an unsigned value, is invisible to any single-mode test.
2. Splitting a block into pieces reproduces the whole block. This is the property the parallel runner will
   need, and it is testable today even though parallelism itself is not: games are seeded by absolute game
   index, so the unit of work can be cut anywhere.
3. The recorded rows are internally consistent -- grouped by game, densely numbered, referring only to
   games that exist.

What is NOT here, and why
-------------------------
There is no serial-versus-`prange` comparison, because there is no `prange` path. `numba` 0.68's parallel
transform does not lower a write into a structured array, and the engine's telemetry is structured arrays
written from inside the game loop. The minimal reproduction and the two options it leaves open are recorded
in `test_the_prange_path_is_not_reachable_and_the_reason_is_recorded`.

The compared fields are every level-1 and level-2 column plus the full move sequence of sampled games, byte
for byte. A checksum is used only to keep the failure message readable.
"""

import hashlib
import re
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from durak import rules
from durak.engine import play_block
from durak.strategies import (
    AUTO_DEFEND,
    CHEATER_DOC,
    CONSERVATIVE,
    GREEDY,
    INFO_PARTICLES,
    INFO_TRUTH,
    RANDOM,
    SEARCH_INFORMED,
    SEARCH_UNIFORM,
)

T = rules.DEFAULT_SETTINGS

#: A pair per strategy, so the comparison covers the cheap policies, the filter-backed one, the search
#: arms and the separately-implemented Cheater. `SEARCH_UNIFORM` against `SEARCH_INFORMED` is the
#: project's headline pairing and belongs in the equivalence surface, not only in the analysis.
PAIRS = [
    (RANDOM, RANDOM, 0, 0),
    (GREEDY, AUTO_DEFEND, 0, 0),
    (CONSERVATIVE, RANDOM, 0, 0),
    (SEARCH_UNIFORM, SEARCH_INFORMED, INFO_PARTICLES, INFO_TRUTH),
    (CHEATER_DOC, SEARCH_INFORMED, INFO_TRUTH, INFO_TRUTH),
]

DECK_SIZES = (36, 52)


def _run_in_subprocess(disable_jit: bool, games: int) -> list[str]:
    """Run the same bundle in a fresh interpreter and return its digest.

    A subprocess is not ceremony: `NUMBA_DISABLE_JIT` is read when Numba is imported, so setting it in
    this process after the imports have happened would compile everything anyway and the test would pass
    without ever having run interpreted.
    """
    program = textwrap.dedent(f"""
        import hashlib, json, sys
        sys.path.insert(0, {str(_repo_root())!r})
        import numpy as np
        from durak import rules
        from durak.engine import play_block
        from durak.strategies import (
            RANDOM, GREEDY, AUTO_DEFEND, CONSERVATIVE, SEARCH_UNIFORM, SEARCH_INFORMED,
            CHEATER_DOC, INFO_PARTICLES, INFO_TRUTH,
        )
        T = rules.DEFAULT_SETTINGS
        pairs = [
            (RANDOM, RANDOM, 0, 0),
            (GREEDY, AUTO_DEFEND, 0, 0),
            (CONSERVATIVE, RANDOM, 0, 0),
            (SEARCH_UNIFORM, SEARCH_INFORMED, INFO_PARTICLES, INFO_TRUTH),
            (CHEATER_DOC, SEARCH_INFORMED, INFO_TRUTH, INFO_TRUTH),
        ]
        parts = []
        for deck in (36, 52):
            for s0, s1, i0, i1 in pairs:
                for pairing in range(3):
                    o, r, m = play_block(deck, s0, s1, i0, i1, 7, pairing, 0, {games}, T, True, True)
                    parts.append(hashlib.sha256(o.tobytes()).hexdigest())
                    parts.append(hashlib.sha256(r.tobytes()).hexdigest())
                    parts.append(hashlib.sha256(m.tobytes()).hexdigest())
        print(json.dumps(parts))
    """)
    env = dict(os.environ)
    if disable_jit:
        env["NUMBA_DISABLE_JIT"] = "1"
    else:
        env.pop("NUMBA_DISABLE_JIT", None)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                            cwd=str(_repo_root()))
    assert result.returncode == 0, f"probe failed:\n{result.stderr[-4000:]}"
    import json
    return json.loads(result.stdout.strip().splitlines()[-1])


def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


@pytest.mark.perf
def test_compiled_and_interpreted_agree_bit_for_bit():
    """The project's central guarantee, across both deck sizes and every strategy pair."""
    games = 60
    compiled = _run_in_subprocess(disable_jit=False, games=games)
    interpreted = _run_in_subprocess(disable_jit=True, games=games)

    assert compiled == interpreted, (
        f"compiled and interpreted disagree on {sum(1 for a, b in zip(compiled, interpreted) if a != b)} "
        f"of {len(compiled)} arrays. Either the two modes disagree about an integer at the compiled "
        f"boundary, or a value's representation differs between them."
    )


def test_the_prange_path_is_not_reachable_and_the_reason_is_recorded():
    """Why there is no serial-versus-parallel test in this file, asserted rather than assumed.

    Task 14 asked for `play_block` compiled with `prange`, and it cannot be done on this toolchain.
    `numba` 0.68 on Python 3.14 does not lower a write into a *structured* array inside a parallel loop:

        NotImplementedError: ... numba/parfors/parfor.py, _numpy_map_to_parfor

    Reproduced minimally: a `@njit(parallel=True)` function whose prange body assigns `out[i]["a"] = i`
    on a `np.dtype([("a", int64)])` array fails to compile, and so does the `row = out[i]` view form.
    Writing to plain one-dimensional arrays inside prange compiles and runs fine.

    `engine._play_one` makes 21 structured-array writes, so the block loop cannot be parallelised without
    restructuring telemetry to collect into per-column 1-D arrays and assemble the structured array on the
    serial side afterwards. That is a real redesign of the telemetry layout, and the plan forbids papering
    over exactly this class of problem -- `numba.objmode`, a `typed.Dict`, or similar.

    So this test is skipped rather than faked: a `prange` path that does not exist cannot be compared to
    the serial one, and a test that passed by comparing the serial path with itself would be worse than no
    test at all.
    """
    pytest.skip(
        "numba 0.68 parfors cannot lower structured-array writes inside prange; a parallel block path "
        "does not exist on this toolchain. See the module docstring of tests/test_equivalence.py."
    )


def test_a_block_split_into_pieces_reassembles_into_the_whole():
    """Block splitting is the runner's unit of parallelism, and the seam must not change the result.

    One block of 40 games against four blocks of 10 with the right `game_start` must produce the same 40
    outcomes. The seeds are derived from the absolute game index, so this holds by construction -- the test
    exists to prove it, because the alternative is discovering it after a parallel run has mixed two
    datasets.
    """
    whole, whole_rounds, _ = play_block(36, GREEDY, AUTO_DEFEND, 0, 0, 11, 0, 0, 40, T, True, False)

    outcomes_parts = []
    rounds_parts = []
    for start in range(0, 40, 10):
        outcomes, rounds, _ = play_block(36, GREEDY, AUTO_DEFEND, 0, 0, 11, 0, start, 10, T, True, False)
        outcomes_parts.append(outcomes)
        rounds_parts.append(rounds)

    # Level-1 rows are one per game, so they concatenate directly. Level-2 rows are written back to back
    # with no per-game padding, so the block's rows are a concatenation too -- but the *split* is
    # reproducible, not an offset arithmetic: each sub-block starts its own write cursor at zero, so the
    # reassembled stream is each game's rows in game order, which is exactly what the whole block holds.
    assert np.array_equal(np.concatenate(outcomes_parts), whole), (
        "level-1 outcomes must be identical however the block is split"
    )
    reassembled_rounds = np.concatenate(rounds_parts)
    assert len(reassembled_rounds) == len(whole_rounds), (
        f"split blocks wrote {len(reassembled_rounds)} round rows against {len(whole_rounds)} whole"
    )
    assert np.array_equal(
        reassembled_rounds["game_index"], whole_rounds["game_index"]
    ), "the games contributing round rows differ between a split and a whole block"
    assert np.array_equal(
        reassembled_rounds["round_idx"], whole_rounds["round_idx"]
    ), "round indices differ between a split and a whole block"


def test_every_recorded_row_is_internally_consistent():
    """Level 2 and 3 rows must belong to games that actually exist and ended.

    Cheap structural check that does not depend on the digest: a racing writer would leave rows pointing at
    a game index no outcome row claims, or move counts that do not match the sampled games.
    """
    outcomes, rounds, moves = play_block(36, CONSERVATIVE, RANDOM, 0, 0, 5, 0, 0, 30, T, True, True)

    assert len(outcomes) == 30
    assert (outcomes["game_index"] == np.arange(30)).all()
    # Round rows are capped per game by ROUNDS_PER_GAME_LIMIT, so only the count of distinct games is exact.
    assert set(np.unique(rounds["game_index"]).tolist()) <= set(range(30))
    assert set(np.unique(moves["game_index"]).tolist()) <= set(range(30))
    if moves.shape[0]:
        # `seq` counts moves WITHIN a game, from zero -- not across the block. Level-3 rows are written
        # back to back across the block, so each sampled game's rows occupy a contiguous stretch and its
        # `seq` restarts at zero. Uniqueness is therefore per (game_index, seq), and that pair being
        # unique is what a racing writer would break: two games sharing a seq, or a gap inside one game.
        assert (np.diff(moves["game_index"]) >= 0).all(), (
            "level-3 rows must be grouped by game in ascending order"
        )
        for game in np.unique(moves["game_index"]):
            seq = moves["seq"][moves["game_index"] == game]
            assert seq.min() == 0 and seq.max() == len(seq) - 1, (
                f"game {int(game)} has seq {seq.min()}..{seq.max()} over {len(seq)} moves; a game's "
                f"sequence must be dense from zero"
            )

def test_the_published_views_filter_to_the_current_versions():
    """`analysis/schema.sql` names the version triple in three macros, and those literals must equal
    `durak.versions`.

    They have drifted twice, and both times the failure was silent and total: with
    `SCHEMA_VERSION` at 2 and the macro still at 1, every view returned zero rows and the reports showed
    empty figures with no error anywhere. The drift is invisible because "no rows" looks like a dataset
    with no games rather than a filter pointing at the wrong version.
    """
    from durak import versions
    from analysis.queries import _SQL_PATH

    sql = _SQL_PATH.read_text()
    for macro, expected in (("current_engine_version", versions.ENGINE_VERSION),
                            ("current_rules_version", versions.RULES_VERSION),
                            ("current_schema_version", versions.SCHEMA_VERSION)):
        match = re.search(rf"MACRO {macro}\(\) AS (\d+);", sql)
        assert match, f"{macro} is not declared in schema.sql"
        assert int(match.group(1)) == int(expected), (
            f"{macro} says {match.group(1)} but durak.versions says {expected}; every view will "
            f"return the wrong rows or none at all"
        )
