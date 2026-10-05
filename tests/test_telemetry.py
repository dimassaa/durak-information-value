"""Telemetry schemas, the Parquet writer, and the data-quality audit.

Two things are being tested here, and they fail in different ways.

A **schema** fails quietly. If a column is renamed, retyped, or dropped and the schema is
inferred from whatever the Python objects happen to hold, the change reaches the dataset as a
difference nobody diffed. So the schemas here are hand-written and asserted field by field
against the spec's telemetry tables -- a rename is a failing test, not a Parquet file.

A **corrupted game** fails quietly too, which is worse. A card in two hands leaves the game
perfectly playable: nothing crashes, the win rate looks plausible, and every metric computed
from it is wrong. So the audit is asserted in both directions -- it must flag a deliberately
corrupted state, and it must *not* flag a clean one. An audit that only ever passes is worse
than no audit, because it reports confidence it has not earned.

The round-trip test writes real games to a temporary directory and reads them back, because a
schema that cannot be written and re-read is a schema that has only been described, never used.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from durak import rules
from durak.audit import audit_game, audit_outcomes
from durak.engine import (
    GAME_DTYPE,
    MOVE_DTYPE,
    ROUND_DTYPE,
    choose_action,
    play_block,
)
from durak.strategies import AUTO_DEFEND, GREEDY, RANDOM, InfoMode
from durak.telemetry import (
    level1_schema,
    level2_schema,
    level3_schema,
    read_parquet,
    write_parquet,
)
from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

T = rules.DEFAULT_SETTINGS


def spec_level1():
    """Spec 6.1's level-1 fields, as (name, dtype) pairs.

    Transcribed here rather than imported, so that changing the engine's schema does not
    automatically change what this test expects. A test that derived its expectation from the
    code under test would assert only self-consistency.
    """
    return [
        ("run_id", pa.int64()), ("game_index", pa.int64()), ("game_seed", pa.int64()),
        ("deck_size", pa.int32()),
        ("strategy_p0", pa.int32()), ("strategy_p1", pa.int32()),
        ("info_mode_p0", pa.int32()), ("info_mode_p1", pa.int32()),
        ("engine_version", pa.int32()), ("rules_version", pa.int32()),
        ("schema_version", pa.int32()),
        ("first_attacker", pa.int32()), ("both_have_trumps", pa.bool_()),
        ("initial_trumps_p0", pa.int32()), ("initial_trumps_p1", pa.int32()),
        ("redeals", pa.int32()), ("first_attacker_random_fallback", pa.bool_()),
        ("winner", pa.int32()), ("rounds", pa.int32()), ("actions", pa.int32()),
        ("winner_final_hand", pa.int32()), ("loser_final_hand", pa.int32()),
        ("deck_exhausted_at_round", pa.int32()), ("winner_rounds_played", pa.int32()),
        ("cards_collected_p0", pa.int32()), ("cards_collected_p1", pa.int32()),
        ("voluntary_takes_p0", pa.int32()), ("voluntary_takes_p1", pa.int32()),
        ("forced_takes_p0", pa.int32()), ("forced_takes_p1", pa.int32()),
        ("trumps_played_p0", pa.int32()), ("trumps_played_p1", pa.int32()),
        ("throw_ins_p0", pa.int32()), ("throw_ins_p1", pa.int32()),
        ("cards_beaten_p0", pa.int32()), ("cards_beaten_p1", pa.int32()),
        ("max_attacks_in_a_round", pa.int32()),
        ("attack_turns_p0", pa.int32()), ("attack_turns_p1", pa.int32()),
        ("win_from_single_card_p0", pa.bool_()), ("win_from_single_card_p1", pa.bool_()),
        ("cards_left_when_deck_ran_out", pa.int32()),
        ("flag_redeals_exceeded", pa.bool_()), ("flag_round_cap_hit", pa.bool_()),
        ("flag_illegal_move", pa.bool_()), ("flag_card_conservation_violation", pa.bool_()),
        ("flag_both_players_empty", pa.bool_()), ("flag_anomaly_any", pa.bool_()),
        # Beyond spec 6.1: telemetry completeness, see the schema's comment.
        ("flag_round_records_truncated", pa.bool_()),
        ("flag_move_records_truncated", pa.bool_()),
    ]


def spec_level2():
    """Spec 6.2's level-2 fields."""
    return [
        ("run_id", pa.int64()), ("game_index", pa.int64()), ("round_idx", pa.int32()),
        ("attacker", pa.int32()), ("n_attacks", pa.int32()), ("n_throw_ins", pa.int32()),
        ("n_defenses", pa.int32()), ("defender_took", pa.bool_()),
        ("take_voluntary", pa.bool_()),
        ("hand_p0", pa.int32()), ("hand_p1", pa.int32()), ("deck_remaining", pa.int32()),
        ("trumps_p0", pa.int32()), ("trumps_p1", pa.int32()), ("roles_swapped", pa.bool_()),
        ("winner_in_round", pa.int32()), ("phase", pa.int32()),
    ]


def spec_level3():
    """Spec 6.3's level-3 fields."""
    return [
        ("run_id", pa.int64()), ("game_index", pa.int64()), ("round_idx", pa.int32()),
        ("seq", pa.int32()), ("actor", pa.int32()), ("action_type", pa.int32()),
        ("card_id", pa.int32()), ("info_mode", pa.int32()),
        ("legal_action_count", pa.int32()), ("chosen_index_in_legal_order", pa.int32()),
    ]


# --- schemas -----------------------------------------------------------------


@pytest.mark.parametrize(
    "schema_fn,expected",
    [(level1_schema, spec_level1()), (level2_schema, spec_level2()),
     (level3_schema, spec_level3())],
    ids=["level1", "level2", "level3"],
)
def test_schema_matches_the_spec_field_for_field(schema_fn, expected):
    schema = schema_fn()

    assert isinstance(schema, pa.Schema)
    for name, dtype in expected:
        assert name in schema.names, f"{name} is missing from the schema"
        assert schema.field(name).type == dtype, (
            f"{name} is {schema.field(name).type}, spec says {dtype}"
        )


def test_engine_dtypes_match_the_arrow_schemas():
    """The engine's NumPy dtype and the Arrow schema must describe the same columns.

    Two hand-written descriptions of the same table is a defect waiting to happen: the engine
    writes NumPy arrays and the writer publishes Arrow, so a divergence here would produce a
    file whose schema does not match the rows in it.
    """
    for numpy_dtype, schema_fn in (
        (GAME_DTYPE, level1_schema), (ROUND_DTYPE, level2_schema), (MOVE_DTYPE, level3_schema),
    ):
        schema = schema_fn()

        assert set(numpy_dtype.names) - set(schema.names) == set(), (
            f"the engine writes columns the schema omits: "
            f"{sorted(set(numpy_dtype.names) - set(schema.names))}"
        )
        assert set(schema.names) - set(numpy_dtype.names) == set(), (
            f"the schema declares columns the engine never writes: "
            f"{sorted(set(schema.names) - set(numpy_dtype.names))}"
        )


def test_every_level_carries_the_three_version_stamps():
    """Levels 2 and 3 inherit the run's stamps rather than storing their own.

    A round or move row with no version stamp cannot be attributed to a rule set, and a dataset
    that mixes two of them silently mixes two experiments into one number.
    """
    for schema_fn in (level2_schema, level3_schema):
        names = schema_fn().names

        assert "run_id" in names, "a level that cannot be joined to its run is not telemetry"

    # And the stamps exist at level 1, populated by the engine.
    outcomes, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 5, T,
                                record_rounds=False, record_moves=False)

    assert np.all(outcomes["engine_version"] == ENGINE_VERSION)
    assert np.all(outcomes["rules_version"] == RULES_VERSION)
    assert np.all(outcomes["schema_version"] == SCHEMA_VERSION)


def test_level3_records_no_latency():
    """Wall-clock decision latency is deliberately absent.

    It is not reproducible, so a reproducibility test would fail on shared hardware for reasons
    unrelated to the code, and a column that varies run to run cannot be averaged without lying
    about its spread. Cost is measured in the benchmark instead.
    """
    names = level3_schema().names

    assert "chosen_index_in_legal_order" in names
    for forbidden in ("latency_ms", "decision_ms", "elapsed", "duration_ms", "wall_ms"):
        assert forbidden not in names, (
            f"{forbidden} is not reproducible and must not be stored"
        )


# --- round trip --------------------------------------------------------------


def test_a_thousand_games_survive_the_round_trip(tmp_path):
    """Written to disk and read back: row count, dtypes, and populated stamps.

    A schema that has never been written to a file has only been described. This is the test
    that makes it a schema.
    """
    outcomes, rounds, moves = play_block(
        36, GREEDY, RANDOM, InfoMode.NONE, InfoMode.NONE, 11, 5, 0, 1_000, T,
        record_rounds=True, record_moves=True,
    )

    write_parquet(outcomes, tmp_path / "games", schema_version=SCHEMA_VERSION)
    assert rounds.size > 0, "no round records were produced"
    write_parquet(rounds, tmp_path / "rounds", schema_version=SCHEMA_VERSION)
    assert moves.size > 0, "no move records were produced"
    write_parquet(moves, tmp_path / "moves", schema_version=SCHEMA_VERSION)

    games_back = read_parquet(tmp_path / "games")
    rounds_back = read_parquet(tmp_path / "rounds")
    moves_back = read_parquet(tmp_path / "moves")

    assert len(games_back) == 1_000
    assert len(rounds_back) == len(rounds)
    assert len(moves_back) == len(moves)

    assert games_back.schema.equals(level1_schema(), check_metadata=False), (
        f"round-tripped schema differs from level1_schema(): "
        f"{games_back.schema != level1_schema()}"
    )
    for name, expected in (("engine_version", ENGINE_VERSION),
                           ("rules_version", RULES_VERSION),
                           ("schema_version", SCHEMA_VERSION)):
        values = games_back.column(name).to_pylist()
        assert all(v == expected for v in values), f"{name} is not populated: {set(values)}"

    # The values must survive, not merely the shape: a telemetry writer that silently
    # reorders or coerces a column would pass a row-count check.
    assert games_back.column("game_index").to_pylist() == list(range(1_000))
    assert games_back.column("rounds").to_pylist() == outcomes["rounds"].tolist()
    assert games_back.column("winner").to_pylist() == outcomes["winner"].tolist()


def test_writing_is_byte_reproducible(tmp_path):
    """The same games written twice produce identical bytes.

    This is what lets a report claim to be rebuilt from the dataset rather than from whatever
    happened to be on disk when it was generated.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"

    for target in (first, second):
        outcomes, rounds, _ = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 3, 1,
                                         0, 200, T, record_rounds=True, record_moves=False)
        write_parquet(outcomes, target / "games", schema_version=SCHEMA_VERSION)
        write_parquet(rounds, target / "rounds", schema_version=SCHEMA_VERSION)

    for name in ("games", "rounds"):
        left = read_parquet(first / name).to_pandas()
        right = read_parquet(second / name).to_pandas()
        assert left.equals(right), f"{name} differ between two identical runs"

    # Byte-for-byte, not merely equal-on-read: Parquet embeds a writer version and, with
    # dictionary encoding, ordering that a DataFrame comparison would hide.
    for name in ("games", "rounds"):
        left_bytes = sorted((first / name).rglob("*.parquet"))
        right_bytes = sorted((second / name).rglob("*.parquet"))
        assert [p.read_bytes() for p in left_bytes] == [p.read_bytes() for p in right_bytes], (
            f"{name} differ byte-for-byte between two identical runs"
        )


def test_output_is_partitioned_by_run_and_deck(tmp_path):
    """Partitioning is what lets a query skip a deck size, and what keeps the two apart.

    Pooling 36-card and 52-card games into one undifferentiated table is the easiest way to
    produce an average over two different experiments.
    """
    outcomes, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 20, T,
                                record_rounds=False, record_moves=False)
    wide, _, _ = play_block(52, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 20, T,
                            record_rounds=False, record_moves=False)

    write_parquet(
        np.concatenate([outcomes, wide]), tmp_path / "games", schema_version=SCHEMA_VERSION,
        run_id=77,
    )

    parts = list((tmp_path / "games").rglob("*.parquet"))
    assert parts, "no parquet files were written"

    # Partition columns live in the path, not in the file: that is the Hive convention, and
    # keeping them in both makes pyarrow infer `deck_size` from the path as a dictionary column
    # that then refuses to merge with the in-file int32 of the same name.
    decks = {int(p.parent.parent.name.split("=")[1])
             for p in parts if p.parent.parent.name.startswith("deck_size=")}

    assert decks == {36, 52}
    assert all("run_id=77" in str(p) for p in parts)
    # The columns come back on read, so the published schema is unchanged.
    combined = read_parquet(tmp_path / "games")
    assert set(combined.column("deck_size").to_pylist()) == {36, 52}
    assert set(combined.column("run_id").to_pylist()) == {77}
    assert len(combined) == 40


# --- level 3 sampling --------------------------------------------------------


def test_level3_sampling_is_deterministic_and_worker_independent():
    """`game_index % 20 == 0`, and identical however the work is divided.

    Deterministic by game index, never by RNG: an RNG-based sample would differ between a
    1-worker and an 8-worker run, and the whole point of the dataset is that it rebuilds.
    """
    one_worker, _, moves_one = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE,
                                          9, 0, 0, 40, T, record_rounds=False, record_moves=True)
    pieces = []
    # Uneven block sizes, clamped to the total. An earlier version stepped by 7 with a fixed count
    # of 7 and ran past 40, so the split run contained a game the single-block run did not -- and
    # the test failed for a reason that had nothing to do with what it was checking.
    for start in range(0, 40, 7):
        count = min(7, 40 - start)
        _, _, moves = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 9, 0, start,
                                 count, T, record_rounds=False, record_moves=True)
        pieces.append(moves)
    moves_split = np.concatenate(pieces)

    assert moves_one.tobytes() == moves_split.tobytes()

    assert np.all(moves_one["game_index"] % 20 == 0)
    assert set(np.unique(moves_one["game_index"]).tolist()) == {0, 20}
    assert len(one_worker) == 40


def test_sample_modulus_is_configurable():
    _, _, every_fourth = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 12,
                                   T, record_rounds=False, record_moves=True, sample_modulus=4)

    assert np.all(every_fourth["game_index"] % 4 == 0)
    assert set(np.unique(every_fourth["game_index"]).tolist()) == {0, 4, 8}


# --- audit -------------------------------------------------------------------


def _finished_game(seed=3, strategy=RANDOM):
    """A real, completed game plus its outcome record."""
    outcomes, _, _ = play_block(36, strategy, strategy, InfoMode.NONE, InfoMode.NONE, seed, 0, 0,
                                1, T, record_rounds=False, record_moves=False)
    state = _replay_to_end(seed)
    return state, outcomes[0]


def _replay_to_end(seed, deck_size=36):
    from durak.rng import GameStreams

    state = rules.setup_game(deck_size, GameStreams.from_parts(seed, 0, 0))
    while state.winner == rules.EMPTY:
        player = rules.acting_player(state)
        action = int(choose_action(state, player, RANDOM, int(InfoMode.NONE)))
        rules.apply_action(state, action, T)
    return state


def test_a_clean_game_is_not_flagged():
    """The negative case, which is the one an audit usually gets wrong.

    An audit that flags everything looks identical to an audit that works, until the day
    everything is flagged and nobody notices.
    """
    state, outcome = _finished_game(seed=3)

    assert audit_game(state, outcome) == []


def test_a_card_in_two_hands_is_flagged():
    """Spec 7.1 invariants 2 and 3 -- the ones that catch the expensive class of bug."""
    state, outcome = _finished_game(seed=5)
    outcome = outcome.copy()

    # Put the loser's first card into the winner's hand as well.
    loser_card = int(state.hands[1 - state.winner][0])
    state.hands[state.winner][int(state.hand_sizes[state.winner])] = loser_card
    state.hand_sizes[state.winner] = np.int8(state.hand_sizes[state.winner] + 1)

    found = audit_game(state, outcome)

    assert found, "a duplicated card was not flagged"
    assert any("duplicate" in name or "conserv" in name for name in found), found


def test_a_missing_card_is_flagged():
    state, outcome = _finished_game(seed=7)
    outcome = outcome.copy()
    state.discard_len = state.discard_len + 1

    found = audit_game(state, outcome)

    assert found
    assert any("conserv" in name for name in found), found


def test_a_winner_with_cards_left_is_flagged():
    """Spec 7.1 invariant 4: the winner's hand is empty and the loser's is not."""
    state, outcome = _finished_game(seed=9)
    outcome = outcome.copy()
    state.hands[state.winner][0] = 20
    state.hand_sizes[state.winner] = np.int8(1)

    found = audit_game(state, outcome)

    assert found
    assert any("winner" in name or "empty" in name for name in found), found


def test_both_players_empty_is_flagged():
    """Spec 7.1 invariant 5."""
    state, outcome = _finished_game(seed=11)
    outcome = outcome.copy()
    state.hands[1 - state.winner][0] = 21
    state.hand_sizes[1 - state.winner] = np.int8(0)

    found = audit_game(state, outcome)

    assert found
    assert any("empty" in name or "loser" in name for name in found), found


def test_a_redeal_count_past_the_cap_is_flagged():
    """Spec 7.1 invariant 7."""
    state, outcome = _finished_game(seed=13)
    outcome = outcome.copy()
    state.redeals = rules.MAX_REDEALS + 1

    found = audit_game(state, outcome)

    assert found
    assert any("redeal" in name for name in found), found


def test_a_non_terminating_game_is_flagged():
    """Spec 7.1 invariant 6."""
    state, outcome = _finished_game(seed=15)
    outcome = outcome.copy()
    state.round_idx = rules.MAX_ROUNDS

    found = audit_game(state, outcome)

    assert found
    assert any("terminat" in name or "round_cap" in name for name in found), found


def test_the_audit_names_are_stable_and_ordered_by_check():
    """Names go into reports and anomaly tables, so they are part of the output contract.

    Ordered by check order, not alphabetically, so a report's anomaly block reads in the same
    sequence as spec 7.1 lists the invariants. Asserted against an explicit list rather than
    `sorted(...)`: the point is that the order is *specified*, and comparing a list to itself
    sorted would only prove the implementation happened to be alphabetical.
    """
    state, outcome = _finished_game(seed=17)
    outcome = outcome.copy()
    # Two independent defects at once, so the relative order of their names is observable.
    state.hands[state.winner][int(state.hand_sizes[state.winner])] = 30
    state.hand_sizes[state.winner] = np.int8(state.hand_sizes[state.winner] + 1)
    state.redeals = rules.MAX_REDEALS + 1

    found = audit_game(state, outcome)

    # The expected list is the specified check order of spec 7.1, and the injected card happens
    # to duplicate one already in play -- which is why `card_in_two_locations` is in the list.
    assert found == [
        "card_conservation_violation",
        "card_in_two_locations",
        "winner_hand_not_empty",
        "redeal_cap_exceeded",
        "record_winner_hand_mismatch",
    ], f"anomaly names or their order changed: {found}"


def test_auditing_a_batch_counts_every_anomaly_kind():
    """`audit_outcomes` is the aggregate the reports use.

    Asserted on a batch rather than on the flags the engine already wrote, so the two
    implementations of "is this game clean" -- the engine's bitmask and the audit's independent
    pass -- are cross-checked instead of one restating the other.
    """
    outcomes, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 21, 0, 0, 500,
                                T, record_rounds=False, record_moves=False)

    report = audit_outcomes(outcomes)

    assert report["games"] == 500
    assert report["anomalous_games"] == 0
    assert report["by_kind"] == {}
    assert report["flagged_by_engine"] == 0
    assert report["clean"] is True


def test_a_single_anomaly_is_never_averaged_away():
    """Spec 7.2: a violation is raised and marked, never swallowed into a mean.

    The audit's own count must agree with the engine's flag for a game the engine flagged, so
    the two cannot drift into a situation where one says clean and the other says broken.
    """
    outcomes, _, _ = play_block(36, AUTO_DEFEND, AUTO_DEFEND, InfoMode.NONE, InfoMode.NONE, 4, 0,
                                0, 300, T, record_rounds=False, record_moves=False)
    report = audit_outcomes(outcomes)
    capped = int(outcomes["flag_round_cap_hit"].sum())

    assert report["games"] == 300
    assert report["flagged_by_engine"] == capped
    assert report["clean"] is (capped == 0)
    # Anomalous games are counted, not dropped: a batch with anomalies is never publishable.
    assert report["anomalous_games"] == capped