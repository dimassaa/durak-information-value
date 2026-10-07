"""Telemetry schemas and the Parquet writer.

Three levels, written for every game, every game, and a sampled subset respectively. The
division is not arbitrary: level 1 answers "who wins", level 2 answers the project's actual
question -- how the value of information changes over the course of a game, which is a per-round
question, because information is worth little while the draw pile is full and a great deal at the
end -- and level 3 exists to explain mechanism, at roughly two orders of magnitude more rows.

Schemas are hand-written
------------------------
Never inferred from a Python object, and never derived from the engine's NumPy dtypes. An
inferred schema follows a refactor: a field could be renamed, retyped, or dropped and the change
would reach the dataset as a difference nobody diffed. Here it is a failing test.

That means two hand-written descriptions of each table exist -- the Arrow schema here and the
NumPy dtype in `durak.engine` -- and `test_engine_dtypes_match_the_arrow_schemas` asserts they
describe the same columns. Two descriptions is a hazard; two *unchecked* descriptions would be a
defect.

Level 3 records a choice, not a duration
----------------------------------------
`chosen_index_in_legal_order` records *which* legal move was taken. Wall-clock decision latency is
deliberately absent: it is not reproducible, so a reproducibility check would fail on shared
hardware for reasons unrelated to the code, and a column that varies between identical runs cannot
be averaged without lying about its spread. Cost is measured in the benchmark instead.

Partitioning
------------
By `run_id` and `deck_size`. Pooling two deck sizes into one undifferentiated table is the
easiest way to publish an average over two different experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from durak.versions import SCHEMA_VERSION

# Byte-reproducible output requires no embedded timestamps and a fixed row-group size. Parquet
# otherwise records the writer's metadata, which would make two identical runs differ in bytes
# while holding identical data.
_ROW_GROUP_SIZE = 50_000


def level1_schema() -> pa.Schema:
    """Spec 6.1, per game, plus the five end-state fields the spec omits.

    The additions exist because spec 6.1 records the winner's and loser's hand sizes but not
    where the remaining cards are, which makes card conservation impossible to verify from the
    dataset alone. Additive: no spec field is renamed or removed.
    """
    return pa.schema([
        # identity and configuration
        pa.field("run_id", pa.int64()), pa.field("game_index", pa.int64()),
        pa.field("game_seed", pa.int64()), pa.field("deck_size", pa.int32()),
        pa.field("strategy_p0", pa.int32()), pa.field("strategy_p1", pa.int32()),
        pa.field("info_mode_p0", pa.int32()), pa.field("info_mode_p1", pa.int32()),
        pa.field("engine_version", pa.int32()), pa.field("rules_version", pa.int32()),
        pa.field("schema_version", pa.int32()),
        # Which grid cell, and under which first-attacker regime. Added in schema version 2: one
        # manifest can carry several regimes, and without these a dataset holding both could not be
        # read by regime at all -- nothing distinguished the two except the seat sequence.
        pa.field("pairing_index", pa.int32()), pa.field("first_attacker_regime", pa.string()),
        # deal and rules
        pa.field("first_attacker", pa.int32()), pa.field("both_have_trumps", pa.bool_()),
        pa.field("initial_trumps_p0", pa.int32()), pa.field("initial_trumps_p1", pa.int32()),
        pa.field("redeals", pa.int32()),
        pa.field("first_attacker_random_fallback", pa.bool_()),
        # outcome
        pa.field("winner", pa.int32()), pa.field("rounds", pa.int32()),
        pa.field("actions", pa.int32()),
        pa.field("winner_final_hand", pa.int32()), pa.field("loser_final_hand", pa.int32()),
        pa.field("deck_exhausted_at_round", pa.int32()),
        pa.field("winner_rounds_played", pa.int32()),
        # end state, beyond spec 6.1
        pa.field("final_hand_p0", pa.int32()), pa.field("final_hand_p1", pa.int32()),
        pa.field("deck_remaining_at_end", pa.int32()),
        pa.field("discard_len_at_end", pa.int32()), pa.field("table_cards_at_end", pa.int32()),
        # resource usage
        pa.field("cards_collected_p0", pa.int32()), pa.field("cards_collected_p1", pa.int32()),
        pa.field("voluntary_takes_p0", pa.int32()), pa.field("voluntary_takes_p1", pa.int32()),
        pa.field("forced_takes_p0", pa.int32()), pa.field("forced_takes_p1", pa.int32()),
        pa.field("trumps_played_p0", pa.int32()), pa.field("trumps_played_p1", pa.int32()),
        pa.field("throw_ins_p0", pa.int32()), pa.field("throw_ins_p1", pa.int32()),
        pa.field("cards_beaten_p0", pa.int32()), pa.field("cards_beaten_p1", pa.int32()),
        pa.field("max_attacks_in_a_round", pa.int32()),
        pa.field("attack_turns_p0", pa.int32()), pa.field("attack_turns_p1", pa.int32()),
        # degeneracy diagnostics
        pa.field("win_from_single_card_p0", pa.bool_()),
        pa.field("win_from_single_card_p1", pa.bool_()),
        pa.field("cards_left_when_deck_ran_out", pa.int32()),
        # anomaly flags
        pa.field("flag_redeals_exceeded", pa.bool_()), pa.field("flag_round_cap_hit", pa.bool_()),
        pa.field("flag_illegal_move", pa.bool_()),
        pa.field("flag_card_conservation_violation", pa.bool_()),
        pa.field("flag_both_players_empty", pa.bool_()),
        pa.field("flag_anomaly_any", pa.bool_()),
        # Telemetry completeness, beyond spec 6.1. Distinct from the flags above: the game was
        # played correctly, but its level-2 or level-3 record hit the per-game bound. Without these
        # a bounded dataset would be indistinguishable from a complete one.
        pa.field("flag_round_records_truncated", pa.bool_()),
        pa.field("flag_move_records_truncated", pa.bool_()),
    ])


def level2_schema() -> pa.Schema:
    """Spec 6.2, per round.

    Carries `run_id` and `game_index` so a round is joinable to its game. A level that cannot be
    joined to its run is not telemetry, it is a log.
    """
    return pa.schema([
        pa.field("run_id", pa.int64()), pa.field("game_index", pa.int64()),
        pa.field("round_idx", pa.int32()), pa.field("attacker", pa.int32()),
        pa.field("n_attacks", pa.int32()), pa.field("n_throw_ins", pa.int32()),
        pa.field("n_defenses", pa.int32()), pa.field("defender_took", pa.bool_()),
        pa.field("take_voluntary", pa.bool_()),
        pa.field("hand_p0", pa.int32()), pa.field("hand_p1", pa.int32()),
        pa.field("deck_remaining", pa.int32()),
        pa.field("trumps_p0", pa.int32()), pa.field("trumps_p1", pa.int32()),
        pa.field("roles_swapped", pa.bool_()), pa.field("winner_in_round", pa.int32()),
        pa.field("phase", pa.int32()),
    ])


def level3_schema() -> pa.Schema:
    """Spec 6.3, per move, sampled.

    No latency column, deliberately. See the module docstring.
    """
    return pa.schema([
        pa.field("run_id", pa.int64()), pa.field("game_index", pa.int64()),
        pa.field("round_idx", pa.int32()), pa.field("seq", pa.int32()),
        pa.field("actor", pa.int32()), pa.field("action_type", pa.int32()),
        pa.field("card_id", pa.int32()), pa.field("info_mode", pa.int32()),
        pa.field("legal_action_count", pa.int32()),
        pa.field("chosen_index_in_legal_order", pa.int32()),
    ])


#: Which schema belongs to which level, and which NumPy dtype carries it.
LEVEL_SCHEMAS = {1: level1_schema, 2: level2_schema, 3: level3_schema}


#: Level-1 columns the *runner* adds rather than the engine. `njit` cannot write a string into a
#: structured row, so `first_attacker_regime` is stamped from `pairing_index` after the games are played.
#: A raw engine array therefore legitimately lacks it, and these columns are filled with a stated default
#: instead of raising -- but only these, and only when genuinely absent, so a schema naming a column
#: nothing writes is still an error.
RUNNER_STAMPED_COLUMNS = {"first_attacker_regime": "UNSTAMPED"}


def to_table(records: np.ndarray, schema: pa.Schema) -> pa.Table:
    """Convert a NumPy record array into an Arrow table under `schema`.

    Casts explicitly rather than relying on Arrow to infer from the array. The point of a
    hand-written schema is that it is authoritative; inference would quietly substitute Arrow's
    opinion wherever the two disagreed.
    """
    records = np.asarray(records)
    if records.size == 0:
        return schema.empty_table()

    columns = []
    for field in schema:
        if field.name in records.dtype.names:
            values = records[field.name]
        elif field.name in RUNNER_STAMPED_COLUMNS:
            # Written by the runner, not the engine. Filling with a named placeholder keeps the column's
            # presence visible: a dataset of "UNSTAMPED" is obviously not a published one, whereas a
            # silently dropped column would look identical to a regime nobody ran.
            values = np.full(len(records), RUNNER_STAMPED_COLUMNS[field.name], dtype="U32")
        else:
            raise ValueError(f"no field of name {field.name}")
        columns.append(pa.array(values, type=field.type))

    return pa.Table.from_arrays(columns, schema=schema)


def infer_level(records: np.ndarray) -> int:
    """Which telemetry level a NumPy record array belongs to, from its dtype.

    Inferred from the *dtype identity*, never from the field names or values. Naming the level is
    an unavoidable extra argument for the caller to get wrong, and getting it wrong silently
    writes level-2 rows through the level-1 schema -- which is how it was first found, as a
    `ValueError` about a missing `game_seed`.

    Inferring dtypes from the data is the thing this module refuses to do elsewhere; inferring
    *which of three known tables* the data is does not weaken any schema, because the schema is
    still hand-written and still authoritative.
    """
    from durak.engine import GAME_DTYPE, MOVE_DTYPE, ROUND_DTYPE

    dtype = np.asarray(records).dtype
    for level, candidate in ((1, GAME_DTYPE), (2, ROUND_DTYPE), (3, MOVE_DTYPE)):
        if dtype == candidate:
            return level

    raise ValueError(
        f"cannot infer a telemetry level from dtype {dtype}; pass `level=` explicitly"
    )


def _write_single_table(table: pa.Table, path: Path, run_id: int | None = None) -> None:
    """Write one already-built table to `path`, deriving partitions from its own columns.

    Split out of `write_parquet` so the `ensure_empty` path writes through exactly the same code as a
    populated level. An empty level written by a different route would be the place where the partition
    layout quietly diverges -- no `run_id` directory, or no `deck_size` -- and every view globs a path.
    """
    schema = table.schema
    # An explicit `run_id` wins; otherwise take it from the rows. For an empty table there are no rows,
    # so the caller's value is the only thing that can name the partition -- and without a partition
    # directory the level-2 file would come back from `read_parquet` without the `run_id` column the
    # views join on.
    resolved_run = run_id
    if resolved_run is None and "run_id" in schema.names and table.num_rows:
        values = set(int(v) for v in np.unique(table.column("run_id").to_pylist()))
        if len(values) == 1:
            resolved_run = values.pop()

    by_deck = "deck_size" in schema.names
    decks = sorted(set(int(v) for v in table.column("deck_size").to_pylist())) \
        if by_deck and table.num_rows else [None]

    for deck_value in decks:
        if by_deck and table.num_rows:
            subset = table.filter(pa.compute.equal(table.column("deck_size"), deck_value))
            partition = path / f"deck_size={deck_value}"
        else:
            subset = table
            partition = path
        if resolved_run is not None:
            partition = partition / f"run_id={resolved_run}"
        partition.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            _drop_partition_columns(subset, ("deck_size", "run_id")),
            partition / "data.parquet",
            row_group_size=_ROW_GROUP_SIZE,
            compression="zstd",
            use_dictionary=False,
        )


def write_parquet(records: np.ndarray, path, schema_version: int = SCHEMA_VERSION,
                  level: int | None = None, run_id: int | None = None,
                  ensure_empty: bool = False) -> None:
    """Write `records` to `path`, partitioned by `run_id` and `deck_size`.

    `level` is inferred from the record dtype when not given; see :func:`infer_level`.

    `ensure_empty` writes a zero-row file carrying the level's schema when there is nothing to write.
    Its purpose is that a reader can tell "this level was not recorded" from "the directory is missing",
    which matters because `read_parquet` on a glob matching nothing is an error rather than an empty
    table: a dataset recorded with `record_rounds=False` broke every view in `analysis/schema.sql`
    because `rounds/**` matched no file. A zero-row file with the right schema reads back as zero rows,
    which is the truth. It is off by default, so a caller that produced nothing still leaves no
    directory behind.

    Partition columns are written into the directory path and **dropped from the stored file**,
    which is the Hive convention. Keeping them in both places is the obvious thing to do and it
    does not work: `pyarrow` then infers `deck_size` from the path as a dictionary-encoded
    partition column and refuses to merge it with the in-file `int32` of the same name. The
    columns come back on read, so the published schema is unchanged either way.

    Empty input writes nothing and creates no directory: a run that produced no rows must not
    leave an empty dataset behind that a later read would treat as a real one.
    """
    records = np.asarray(records)
    if records.size == 0:
        if not ensure_empty:
            return
        # A zero-row file with the declared schema, so the level's glob matches and reads back empty.
        empty_level = infer_level(records) if records.dtype.names else level
        if empty_level is None:
            raise ValueError("ensure_empty needs a level when the records carry no fields")
        _write_single_table(LEVEL_SCHEMAS[empty_level]().empty_table(), Path(path), run_id=run_id)
        return

    if level is None:
        level = infer_level(records)

    schema = LEVEL_SCHEMAS[level]()
    table = to_table(records, schema)

    if "schema_version" in schema.names:
        index = schema.get_field_index("schema_version")
        table = table.set_column(
            index, schema.field(index),
            pa.array(np.full(len(table), schema_version, dtype=np.int32), type=pa.int32()),
        )

    _write_single_table(table, Path(path), run_id=run_id)


def _drop_partition_columns(table: pa.Table, names) -> pa.Table:
    """Remove columns that live in the directory path instead of in the file."""
    keep = [name for name in table.schema.names if name not in names]
    return table.select(keep)


def read_parquet(path) -> pa.Table:
    """Read every partition under `path` back into one table.

    Deliberately whole-dataset. The reports read a run at a time and DuckDB pushes the
    partitioning down when it can; a reader that silently returned only one partition would be a
    far worse failure than one that reads more than it needs.
    """
    root = Path(path)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pa.table({})

    # `partitioning="hive"` restores the `run_id` and `deck_size` columns that live in the
    # directory path, as the plain typed columns the schemas declare. PyArrow appends recovered
    # partition columns at the front, so the order is restored afterwards: the published schema is
    # `level1_schema()`, and a reader that handed back a differently ordered schema would make
    # every downstream equality check on it fail for no substantive reason.
    table = pq.read_table(files, partitioning="hive")
    # PyArrow infers partition columns as dictionary-encoded, which is the right *encoding* for a
    # directory name and the wrong *type* for the published schema. Cast back to the declared types
    # so a table read from disk is schema-identical to the schema declared in code -- otherwise
    # every downstream equality check on it fails for an encoding reason.
    for name, dtype in (("run_id", pa.int64()), ("deck_size", pa.int32())):
        if name in table.schema.names and not table.schema.field(name).type.equals(dtype):
            index = table.schema.get_field_index(name)
            table = table.set_column(
                index, table.schema.field(index).with_type(dtype),
                table.column(index).cast(dtype),
            )

# PyArrow prepends recovered partition columns, so the order no longer matches any declared
    # schema. Reorder to the declared order when the column *set* matches one, which makes a table
    # read from disk schema-identical to the schema declared in code.
    present = set(table.schema.names)
    for schema_fn in LEVEL_SCHEMAS.values():
        declared = schema_fn()
        if present == set(declared.names):
            return table.select(list(declared.names))

    return table


def dataset_manifest(path) -> dict:
    """A JSON manifest describing what a dataset directory contains.

    Written next to the data so a report can state what it read -- row counts, the version
    stamps, the partition layout -- without re-deriving any of it from the files. Anything that
    cannot be read back from the manifest is a number a reader will have to guess.
    """
    import json

    root = Path(path)
    manifest = {"levels": {}, "partitions": []}

    for file in sorted(root.rglob("*.parquet")):
        table = pq.read_table(file, columns=None)
        relative = file.relative_to(root)
        level_dir = relative.parts[0] if len(relative.parts) > 1 else "games"
        entry = manifest["levels"].setdefault(
            level_dir, {"rows": 0, "files": 0, "schema": str(table.schema)},
        )
        entry["rows"] += table.num_rows
        entry["files"] += 1
        manifest["partitions"].append(
            "/".join(relative.parts[:-1]) or "."
        )

    for name in ("engine_version", "rules_version", "schema_version"):
        games = root / "games"
        if games.exists():
            table = read_parquet(games)
            if name in table.schema.names and table.num_rows:
                values = set(table.column(name).to_pylist())
                manifest[name] = values.pop() if len(values) == 1 else sorted(values)
                break

    manifest["partitions"] = sorted(set(manifest["partitions"]))
    return manifest


def write_manifest(path) -> None:
    """Write the manifest as ``manifest.json`` beside the data."""
    import json

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_manifest(root), handle, indent=2, sort_keys=True)
        handle.write("\n")