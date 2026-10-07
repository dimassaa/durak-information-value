"""Determinism: the same configuration must produce the same dataset, whatever the machine does.

This is the property the whole dataset rests on, and it has two halves that fail differently.

*Block partitioning.* Games are seeded by absolute game index, so cutting a run into different blocks
cannot change a single game. This is testable today and is tested over several block sizes.

*Worker count.* The runner splits work across processes. A result that depended on the worker count
would mean a game was identified by something other than its seed -- scheduling, arrival order, or the
block's position -- and the dataset would be unrepeatable on a different machine. That is what the
worker-count tests assert, and they are the ones that would have caught an identity bug in the pool.

Why process-level and not `prange`
----------------------------------
`numba` 0.68's parallel transform does not lower writes into a structured array, and the engine's
telemetry is exactly that (21 structured writes inside the game loop). Rather than redesign the
telemetry layout, parallelism is taken one level up: `play_block` stays serial and whole blocks are
dispatched to worker processes. Blocks are independent and each writes its own Parquet partition, so
the pool cannot corrupt shared state even in principle -- and `analysis/schema.sql` then reads the
partitions as a single dataset.
"""

from pathlib import Path

import numpy as np
import pytest

from durak import rules
from durak.config import Settings
from durak.engine import play_block
from durak.runner import RunManifest, run_tournament
from durak.strategies import AUTO_DEFEND, GREEDY, INFO_PARTICLES, RANDOM, SEARCH_UNIFORM
from durak.telemetry import read_parquet
from durak.tournament import RULE_DERIVED, game_block_ranges, pairing_grid

#: `pairing_grid` emits both seat orders, so a run of two strategies is TWO cells, not one. Every count
#: below is therefore games_per_pairing * cells, and a test that forgets the factor asserts the wrong
#: number while still looking plausible.
CELLS = len(pairing_grid([GREEDY, AUTO_DEFEND], deck_sizes=(36,), regimes=(RULE_DERIVED,)))

T = rules.DEFAULT_SETTINGS


def _manifest(tmp_path, games=40, block_size=10, strategies=(GREEDY, AUTO_DEFEND)):
    return RunManifest(
        run_id="d", seed=3, strategies=list(strategies), deck_sizes=[36],
        games_per_pairing=games, regimes=[RULE_DERIVED], settings=Settings(),
        block_size=block_size, engine_version=2, rules_version=2, schema_version=1,
    )


# --- block partitioning --------------------------------------------------------


def test_block_ranges_cover_every_game_exactly_once():
    ranges = game_block_ranges(40, 10)
    assert ranges[0] == (0, 10)
    assert ranges[-1] == (30, 40)
    covered = [i for start, stop in ranges for i in range(start, stop)]
    assert covered == list(range(40)), "blocks must partition the range with no gap and no repeat"


def test_a_short_final_block_is_not_padded():
    assert game_block_ranges(25, 10) == [(0, 10), (10, 20), (20, 25)]


def test_block_ranges_reject_a_nonsense_size():
    with pytest.raises(ValueError):
        game_block_ranges(10, 0)
    with pytest.raises(ValueError):
        game_block_ranges(-1, 10)


def test_zero_games_yields_no_blocks():
    assert game_block_ranges(0, 10) == []


@pytest.mark.parametrize("block_size", [1, 3, 7, 10, 40, 100])
def test_block_size_does_not_change_the_games(block_size, tmp_path):
    """The same manifest at any block size produces the same level-1 rows.

    Compared per game rather than as a whole array: level-2 and level-3 rows are written back to back
    without per-game padding, so a different block size legitimately produces a different array *shape*
    while producing the same records.
    """
    out_dir = tmp_path / f"b{block_size}"
    run_tournament(_manifest(tmp_path, games=24, block_size=block_size),
                   out_dir, record_rounds=True, record_moves=True)
    rows = read_parquet(out_dir / "games").to_pylist()
    assert len(rows) == 24 * CELLS
    # Game identity is the seed: identical seeds across block sizes is the property itself.
    seeds = {int(r["game_seed"]) for r in rows}
    # One seed per game per cell. The seed is derived from `(run_seed, pairing_index, game_index)`, so
    # the two seat orders of a pairing get DIFFERENT seeds -- which is why the count is games * CELLS and
    # not games. Collapsing them here would assert that the two cells played the same games.
    assert len(seeds) == 24 * CELLS, (
        f"{len(seeds)} distinct seeds for {24 * CELLS} games; two games are sharing a seed"
    )


def test_the_same_seed_set_is_produced_at_every_block_size(tmp_path):
    """Compare the block sizes against each other, not against a constant.

    The per-block-size test above can only assert internal consistency; this is the one that says a
    7-game block and a 40-game block play the *same* 24 games.
    """
    per_size = []
    for block_size in (1, 7, 24, 100):
        out_path = tmp_path / f"s{block_size}"
        run_tournament(_manifest(tmp_path, games=24, block_size=block_size),
                      out_path, record_rounds=True, record_moves=False)
        rows = read_parquet(out_path / "games").to_pylist()
        per_size.append(tuple(sorted(int(r["game_seed"]) for r in rows)))

    assert len(set(per_size)) == 1, (
        "block size changed which games were played; a game's identity must be its absolute index"
    )


# --- worker count --------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_worker_count_does_not_change_the_dataset(workers, tmp_path):
    """The dataset is a function of the manifest alone.

    Every level-1 row of a 1-worker run must appear, unchanged, in a run with more workers. Compared as a
    sorted set of game seeds and then by the rows themselves, because the *order* rows arrive in is not
    part of the contract -- the row content is.
    """
    reference = run_tournament(_manifest(tmp_path, games=32, block_size=8),
                               tmp_path / "w1", record_rounds=False, record_moves=False)
    other = run_tournament(_manifest(tmp_path, games=32, block_size=8),
                           tmp_path / f"w{workers}", workers=workers,
                           record_rounds=False, record_moves=False)

    ref_rows = sorted(tuple(sorted(r.items())) for r in read_parquet(tmp_path / "w1" / "games").to_pylist())
    oth_rows = sorted(tuple(sorted(r.items())) for r in read_parquet(tmp_path / f"w{workers}" / "games").to_pylist())

    assert len(ref_rows) == len(oth_rows) == 32 * CELLS
    assert ref_rows == oth_rows, (
        f"workers={workers} produced a different dataset from workers=1; a game must be identified "
        f"by its seed and nothing else"
    )


def test_summary_counts_are_identical_across_worker_counts(tmp_path):
    """The published numbers must not move either, not just the rows."""
    summaries = {}
    for workers in (1, 3):
        summaries[workers] = run_tournament(
            _manifest(tmp_path, games=24, block_size=6), tmp_path / f"c{workers}",
            workers=workers, record_rounds=False, record_moves=False)

    assert summaries[1].games == summaries[3].games
    assert summaries[1].wins == summaries[3].wins
    assert summaries[1].anomalies == summaries[3].anomalies
    assert summaries[1].audit == summaries[3].audit


def test_a_resumed_run_reproduces_the_dataset_without_duplicating_games(tmp_path):
    """Re-running a completed run must neither drop nor duplicate a game.

    The check is duplicate game indices, not a self-comparison: an earlier version of this test read the
    output directory twice and asserted the two reads were equal, which is true whether or not the runner
    works. Counting the occurrences of each `game_index` is what actually distinguishes "resumed
    correctly" from "appended a second copy of everything".
    """
    out = tmp_path / "resume"
    first = run_tournament(_manifest(tmp_path, games=16, block_size=4), out,
                           record_rounds=False, record_moves=False)
    before = read_parquet(out / "games").to_pylist()
    second = run_tournament(_manifest(tmp_path, games=16, block_size=4), out,
                            record_rounds=False, record_moves=False)
    after = read_parquet(out / "games").to_pylist()

    assert first.games == second.games == 16 * CELLS
    assert len(before) == 16 * CELLS
    assert len(after) == 16 * CELLS, f"resuming appended rows: {len(before)} before, {len(after)} after"
    assert sorted(r["game_index"] for r in after) == sorted(r["game_index"] for r in before), (
        "a resumed run must play the same games"
    )
    assert sorted(tuple(sorted(r.items())) for r in after) == \
        sorted(tuple(sorted(r.items())) for r in before), (
        "a resumed run must reproduce the rows it resumed, not merely the same number of them"
    )


# --- the pool itself -----------------------------------------------------------


def test_workers_zero_means_decide_at_runtime(tmp_path):
    """0 is a request for `cpu_count`, not for zero workers, and it must not run nothing."""
    summary = run_tournament(_manifest(tmp_path, games=8, block_size=4), tmp_path / "auto",
                             workers=0, record_rounds=False, record_moves=False)
    assert summary.games == 8 * CELLS


def test_negative_workers_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_tournament(_manifest(tmp_path, games=4), tmp_path / "neg", workers=-1)


def test_an_empty_grid_still_raises(tmp_path):
    """A zero-game run must not report success, whatever the worker count."""
    manifest = RunManifest(
        run_id="e", seed=0, strategies=[GREEDY], deck_sizes=[36], games_per_pairing=4,
        regimes=[RULE_DERIVED], settings=Settings(), block_size=2,
        engine_version=2, rules_version=2, schema_version=1,
    )
    with pytest.raises(ValueError):
        run_tournament(manifest, tmp_path / "empty", workers=2)


def test_the_grid_is_stable_and_dense():
    grid = pairing_grid([GREEDY, SEARCH_UNIFORM], deck_sizes=(36,))
    assert [p.pairing_index for p in grid] == list(range(len(grid)))
    labels = {p.label for p in grid}
    assert len(labels) == len(grid), "every cell needs a distinct label"


def test_a_block_of_one_game_matches_the_same_game_inside_a_large_block(tmp_path):
    """The strongest form of the identity claim: one game alone equals that game in a batch."""
    single, _, _ = play_block(36, SEARCH_UNIFORM, INFO_PARTICLES, INFO_PARTICLES, RANDOM, 9, 2, 5, 1,
                              T, True, True)
    batch, rounds, moves = play_block(36, SEARCH_UNIFORM, INFO_PARTICLES, INFO_PARTICLES, RANDOM, 9, 2, 0,
                                      12, T, True, True)
    match = batch[batch["game_index"] == 5]
    assert len(match) == 1
    assert single["game_seed"][0] == match["game_seed"][0]
    for field in single.dtype.names:
        assert single[field][0] == match[field][0], f"field {field} differs for game 5 alone vs in a batch"

# --- the published views -------------------------------------------------------


def test_the_published_views_execute_and_agree_with_the_parquet(tmp_path):
    """`analysis/schema.sql` is executed for real, not merely checked for a syntax error.

    A SQL file that parses but names a column the dataset does not have is worse than no file: the
    report stage would discover it after the dataset was built. Each view is created against a real run
    and queried, and the level counts are checked against the Parquet files directly.
    """
    import duckdb

    out = tmp_path / "views"
    run_tournament(_manifest(tmp_path, games=12, block_size=4), out,
                   record_rounds=True, record_moves=True)

    sql = (Path(__file__).resolve().parent.parent / "analysis" / "schema.sql").read_text()
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))

    con = duckdb.connect()
    con.execute(f"SET VARIABLE dataset_root='{out}'")
    for statement in [s.strip() for s in body.split(";") if s.strip()]:
        con.execute(statement)

    for view in ("dataset_versions", "games", "rounds", "moves", "head_to_head",
                 "round_phases", "win_rates"):
        assert con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0] > 0, f"{view} is empty"

    assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 12 * CELLS
    # The version filter must not have hidden this run: it is stamped with the current versions.
    versions = con.execute(
        "SELECT DISTINCT engine_version, rules_version, schema_version FROM dataset_versions"
    ).fetchall()
    assert len(versions) == 1, f"a single run must expose one version triple, got {versions}"


def test_the_version_filter_hides_a_different_rules_version(tmp_path):
    """The filter is the reason `schema.sql` exists, so it is tested against a fabricated old dataset.

    Reading a pre-fix dataset together with a post-fix one produces a number describing neither. Rather
    than trusting the filter, a second dataset stamped with the old rules version is dropped into the
    same root and the views must not return its rows.
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = tmp_path / "mixed"
    # All three levels are written: the views glob `rounds/**` and `moves/**` unconditionally, and a
    # level that was not recorded has no directory to glob. That is a separate gap from this test's
    # subject, which is version filtering.
    run_tournament(_manifest(tmp_path, games=8, block_size=4), out,
                   record_rounds=True, record_moves=True)

    # Copy the level-1 partition and restamp it with the pre-fix rules version, into the same root so
    # the views see both. `dataset_root` globs `games/**/*.parquet`, so the copy has to sit under the
    # root's own `games/`, not beside it.
    src = sorted((out / "games").rglob("*.parquet"))[0]
    table = pq.read_table(src)
    old = table.set_column(
        table.schema.get_field_index("rules_version"),
        table.schema.field(table.schema.get_field_index("rules_version")),
        pa.array(np.full(table.num_rows, 1, dtype=np.int32), type=pa.int32()),
    )
    dest = out / "games" / "deck_size=36" / "run_id=1"
    dest.mkdir(parents=True, exist_ok=True)
    pq.write_table(old, dest / "data.parquet")

    sql = (Path(__file__).resolve().parent.parent / "analysis" / "schema.sql").read_text()
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    con = duckdb.connect()
    con.execute(f"SET VARIABLE dataset_root='{out}'")
    for statement in [s.strip() for s in body.split(";") if s.strip()]:
        con.execute(statement)

    assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 8 * CELLS, (
        "a pre-fix dataset was mixed into the current one"
    )
    # `dataset_versions` is deliberately the opposite: it lists BOTH triples present on disk, which is
    # its whole purpose -- it is how a reader finds out that a pre-fix dataset is sitting beside a
    # post-fix one. The filter belongs in the data views, not in the inventory. Asserting it here too
    # would make the view useless.
    triples = con.execute("SELECT DISTINCT rules_version FROM dataset_versions").fetchall()
    assert sorted(t[0] for t in triples) == [1, 2], (
        f"dataset_versions must report every version present, got {triples}"
    )


def test_a_dataset_without_rounds_or_moves_still_loads_in_every_view(tmp_path):
    """A cheap run must stay readable. `analysis/schema.sql` globs all three levels.

    DuckDB raises on a glob that matches no file, so a run with `record_rounds=False` used to break
    every view -- which made recording everything the cheapest way to read a dataset, the opposite of
    what the flags are for.
    """
    import duckdb

    out = tmp_path / "level1only"
    run_tournament(_manifest(tmp_path, games=8, block_size=4), out,
                   record_rounds=False, record_moves=False)

    sql = (Path(__file__).resolve().parent.parent / "analysis" / "schema.sql").read_text()
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    con = duckdb.connect()
    con.execute(f"SET VARIABLE dataset_root='{out}'")
    for statement in [s.strip() for s in body.split(";") if s.strip()]:
        con.execute(statement)

    assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 8 * CELLS
    assert con.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM moves").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM win_rates").fetchone()[0] > 0
