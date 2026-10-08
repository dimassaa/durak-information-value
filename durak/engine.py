"""The game loop and the outcome records.

This is the module where correctness stops being a property of individual functions and becomes
a property of a *sequence* of them. Every rule in `durak.rules` can be right while the loop that
chains them leaks a card, skips a refill, or scores a game that never finished.

Four decisions carry the weight here.

The win condition is never evaluated in this file
-----------------------------------------------
`play_game` calls `rules.apply_action` and reads `state.winner`. It does not re-derive "is a hand
empty". A second copy of the win condition would be free to disagree with the first, and the
disagreement would surface as a plausible win-rate shift rather than as a crash. The same
applies to the round cap, which lives in `rules.begin_round`.

One loop, two configurations
----------------------------
`play_block` writes round and move records only when asked, into arrays the caller sized, and one
loop body serves both configurations. A separate recording implementation would be a second
engine: it would drift from the fast path, and the cheap configuration -- the one the benchmark
measures -- would not be the one the dataset came from. With recording off, the arrays are passed
at zero length and never written.

Capped games are anomalies, not losses
--------------------------------------
A game stopped at the round cap gets `winner = -1` and `flag_round_cap_hit = True`. Recording it
as a loss for whoever happened to be attacking would put a non-terminating game into the dataset
as a result. `flag_anomaly_any` aggregates the flags so an aggregation can filter without naming
each one. This is not hypothetical: `RANDOM` never reaches the cap, but `GREEDY` does in about 3%
of games and `AUTO_DEFEND` in about 5%, so the column carries real data.

Determinism
-----------
A game's identity is `(block_seed, pairing_index, game_index)` and nothing else -- not scheduling,
not wall-clock, not the block's position in a run. That is what makes a 1-worker run and an
8-worker run byte-identical, and what makes a failed run resumable without duplicating or losing
games. `play_block` is serial here; `prange` arrives in Task 14 and must preserve these semantics
exactly.

A note on the "resources" columns
---------------------------------
`cards_collected`, `cards_beaten`, `trumps_played`, and `throw_ins` are counted at the moment the
action is chosen, before `apply_action` mutates the state. Counting afterwards would be impossible
for a take: the table is emptied by the very transition being counted.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from durak.actions import (
    ATTACK,
    BEAT,
    STOP,
    TAKE,
    THROW,
    action_card,
    action_kind,
)
from durak.rng import game_stream_states, mix_seed3
from durak.rules import (
    EMPTY,
    USE_RULE,
    MAX_ROUNDS,
    GameState,
    RuleTunables,
    acting_player,
    apply_action,
    check_invariants,
    deal_state,
    legal_actions,
    phase_of,
    round_cap_reached,
    trumps_in_hand,
)
from durak.strategies.cheater_doc import policy_cheater_doc
from durak.strategies.conservative import policy_conservative
from durak.strategies.oracle import policy_oracle, policy_oracle_uniform
from durak.strategies.policies import (
    policy_auto_attack,
    policy_auto_defend,
    policy_greedy,
    policy_random,
)
from durak.strategies.search_policy import policy_search
from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

# A finished game with no winner. Distinct from any seat, so a filter can never mistake a capped
# game for a result.
NO_WINNER = -1

# Per-game action ceiling as a last line of defence. Four times the round cap, because a round is
# at least two actions and a throw-in chain can make one much longer. The round cap fires first in
# any game that is genuinely stalling, so reaching this would itself be a defect.
MAX_ACTIONS_PER_GAME = 4 * MAX_ROUNDS + 64

# Records kept per game for the optional telemetry levels.
#
# A game stopped at the round cap emits 10,000 rounds against roughly 14 for a normal one. That
# is not merely a buffer problem: a single stalled game would contribute 700 times its share of
# the level-2 dataset and wreck any per-round analysis, which is the analysis the project's
# primary question depends on. So the per-game count is bounded, and truncation is *recorded* on
# the level-1 row (`flag_round_records_truncated`) rather than being silent -- silently dropping
# rows would leave a dataset that looks complete and is not.
#
# 300 is comfortably above the longest terminating game measured (`RANDOM`, max 117 rounds), so
# the limit binds only on games that already carry `flag_round_cap_hit`.
ROUNDS_PER_GAME_LIMIT = 300
MOVES_PER_GAME_LIMIT = 2_000

# Default level-3 sampling modulus. Deterministic by `game_index`, never RNG-based, so the sample
# does not depend on how work was scheduled.
DEFAULT_SAMPLE_MODULUS = 20


# --- schemas -----------------------------------------------------------------


def _build_game_dtype():
    """Level 1, plus five end-state fields spec 6.1 omits.

    Hand-written, never inferred from a Python object: an inferred schema follows a refactor, so a
    field could be renamed or dropped and the change would reach the Parquet as a silent
    difference. Here a rename is a schema diff.

    The additions (`final_hand_p0`, `final_hand_p1`, `deck_remaining_at_end`, `discard_len_at_end`,
    `table_cards_at_end`) exist because spec 6.1 records the winner's and loser's hand sizes but
    not where the remaining cards are, which makes card conservation impossible to verify from the
    dataset alone.
    """
    return np.dtype([
        # identity and configuration
        ("run_id", "i8"), ("game_index", "i8"), ("game_seed", "i8"), ("deck_size", "i8"),
        ("strategy_p0", "i4"), ("strategy_p1", "i4"),
        ("info_mode_p0", "i4"), ("info_mode_p1", "i4"),
        ("engine_version", "i4"), ("rules_version", "i4"), ("schema_version", "i4"),
        # deal and rules
        ("first_attacker", "i4"), ("both_have_trumps", "?"),
        # Which grid cell produced this game. This column was MISSING and is the reason schema version 2
        # exists: one manifest can carry several regimes, so a dataset held both and nothing on the row
        # said which cell or which regime it came from -- leaving any regime comparison impossible from
        # the data alone. `first_attacker_regime` is a string and is added by the runner from this
        # column, because `njit` cannot write a string into a row.
        ("pairing_index", "i4"),
        ("initial_trumps_p0", "i4"), ("initial_trumps_p1", "i4"),
        ("redeals", "i4"), ("first_attacker_random_fallback", "?"),
        # outcome
        ("winner", "i4"), ("rounds", "i4"), ("actions", "i4"),
        ("winner_final_hand", "i4"), ("loser_final_hand", "i4"),
        ("deck_exhausted_at_round", "i4"), ("winner_rounds_played", "i4"),
        # end state, beyond spec 6.1
        ("final_hand_p0", "i4"), ("final_hand_p1", "i4"),
        ("deck_remaining_at_end", "i4"), ("discard_len_at_end", "i4"),
        ("table_cards_at_end", "i4"),
        # resource usage
        ("cards_collected_p0", "i4"), ("cards_collected_p1", "i4"),
        ("voluntary_takes_p0", "i4"), ("voluntary_takes_p1", "i4"),
        ("forced_takes_p0", "i4"), ("forced_takes_p1", "i4"),
        ("trumps_played_p0", "i4"), ("trumps_played_p1", "i4"),
        ("throw_ins_p0", "i4"), ("throw_ins_p1", "i4"),
        ("cards_beaten_p0", "i4"), ("cards_beaten_p1", "i4"),
        ("max_attacks_in_a_round", "i4"),
        ("attack_turns_p0", "i4"), ("attack_turns_p1", "i4"),
        # degeneracy diagnostics
        ("win_from_single_card_p0", "?"), ("win_from_single_card_p1", "?"),
        ("cards_left_when_deck_ran_out", "i4"),
        # anomaly flags
        ("flag_redeals_exceeded", "?"), ("flag_round_cap_hit", "?"),
        ("flag_illegal_move", "?"), ("flag_card_conservation_violation", "?"),
        ("flag_both_players_empty", "?"), ("flag_anomaly_any", "?"),
        # Telemetry completeness. Not data-quality anomalies -- the game was played correctly --
        # but a dataset that silently dropped rows would be indistinguishable from a complete
        # one, so truncation of levels 2 and 3 is recorded here.
        ("flag_round_records_truncated", "?"), ("flag_move_records_truncated", "?"),
    ])


def _build_round_dtype():
    """Level 2, per round."""
    return np.dtype([
        ("run_id", "i8"), ("game_index", "i8"), ("pairing_index", "i4"), ("round_idx", "i4"),
        # `pairing_index` is what makes a level-2 row joinable to its game. `game_index` is CELL-LOCAL:
        # in the 57-cell primary run each of the 20,000 indices occurs in all 57 cells, so joining on
        # `(run_id, game_index)` alone fans out by 57 and every level-2 figure becomes an average over 57
        # unrelated games. Level 1 already carried the column for exactly this reason; levels 2 and 3 did
        # not, which is why no round- or move-level quantity could be attributed to a strategy at all.
        ("attacker", "i4"), ("n_attacks", "i4"), ("n_throw_ins", "i4"),
        ("n_defenses", "i4"), ("defender_took", "?"), ("take_voluntary", "?"),
        ("hand_p0", "i4"), ("hand_p1", "i4"), ("deck_remaining", "i4"),
        ("trumps_p0", "i4"), ("trumps_p1", "i4"), ("roles_swapped", "?"),
        ("winner_in_round", "i4"), ("phase", "i4"),
    ])


def _build_move_dtype():
    """Level 3, per move, sampled.

    `chosen_index_in_legal_order` records *which* legal move was taken, not how long deciding took.
    Wall-clock latency is deliberately not recorded: it is not reproducible, so a reproducibility
    test would fail on shared hardware for reasons unrelated to the code. Cost is measured in the
    benchmark instead.
    """
    return np.dtype([
        ("run_id", "i8"), ("game_index", "i8"), ("pairing_index", "i4"), ("round_idx", "i4"),
        ("seq", "i4"),
        ("actor", "i4"), ("action_type", "i4"), ("card_id", "i4"), ("info_mode", "i4"),
        ("legal_action_count", "i4"), ("chosen_index_in_legal_order", "i4"),
    ])


GAME_DTYPE = _build_game_dtype()
ROUND_DTYPE = _build_round_dtype()
MOVE_DTYPE = _build_move_dtype()

# A single level-1 record: a zero-dimensional structured array. Field access is `row["winner"]`, and
# it is what `play_game` returns and what each element of a `play_block` outcome array holds.
GameOutcome = np.ndarray


# --- policy dispatch ---------------------------------------------------------


@njit(cache=True)
def choose_action(state: GameState, player: int, strategy_id: int, info_mode: int) -> int:
    """The action `strategy_id` takes in `state` as `player`, as a packed int.

    The single point where a strategy id becomes a call. An if/elif chain rather than a dict
    lookup, because a Numba ``prange`` loop needs a compile-time constant callee and a dict of
    dispatchers is not one. Every branch has the same call shape, so the return type unifies.

    `rng_state` is the acting player's own stream, read out of the state and passed explicitly.
    That keeps the policies' signatures identical across the registry while making the source of
    each policy's randomness visible here, at the boundary, rather than hidden inside each policy.
    """
    stream = state.rng_p0 if player == 0 else state.rng_p1

    if strategy_id == 0:
        return policy_auto_attack(state, player, strategy_id, info_mode, stream)
    if strategy_id == 1:
        return policy_auto_defend(state, player, strategy_id, info_mode, stream)
    if strategy_id == 2:
        return policy_random(state, player, strategy_id, info_mode, stream)
    if strategy_id == 3:
        return policy_greedy(state, player, strategy_id, info_mode, stream)
    # The particle filter's aggregate counts, applied as hand rules rather than as a search.
    if strategy_id == 4:
        return policy_conservative(state, player, strategy_id, info_mode, stream)
    # Both search arms are the same function; `info_mode` is what tells them apart.
    if strategy_id == 5:
        return policy_search(state, player, strategy_id, info_mode, stream)
    if strategy_id == 6:
        return policy_search(state, player, strategy_id, info_mode, stream)
    # A separate implementation on purpose -- see durak/strategies/cheater_doc.py.
    if strategy_id == 7:
        return policy_cheater_doc(state, player, strategy_id, info_mode, stream)
    # The ceiling pair: exact opponent hand at depth 4, and its uninformed twin over the uniform
    # prior. Appended rather than renumbered, so no id already written into a Parquet row moves.
    if strategy_id == 8:
        return policy_oracle(state, player, strategy_id, info_mode, stream)
    if strategy_id == 9:
        return policy_oracle_uniform(state, player, strategy_id, info_mode, stream)

    raise ValueError("strategy id has no policy")


# --- the loop ----------------------------------------------------------------


@njit(cache=True)
def seed_to_i64(value) -> int:
    """A 64-bit seed narrowed to the low 63 bits, so it fits a signed int64 column.

    `mix_seed3` returns a full ``uint64``, but the record column is signed. Writing the raw value
    is a mode divergence rather than a cosmetic issue: Numba wraps silently when compiled, while
    NumPy raises ``OverflowError`` when interpreted, so the same dataset would build in one mode
    and fail in the other.

    Dropping the top bit is a bijection onto ``[0, 2**63)``, and the top bit of a splitmix64
    finaliser is itself an avalanched function of the inputs, so nothing is lost in practice. The
    birthday bound over a 10-million-game dataset is about 5e-6, and the column is a convenience
    identifier in any case -- a game's true identity is ``(block_seed, pairing_index,
    game_index)``, all of which are recorded separately.
    """
    return np.int64(np.uint64(value) & np.uint64(0x7FFFFFFFFFFFFFFF))


@njit(cache=True)
def _blank_outcome(run_id, game_index, game_seed, deck_size, strategy_p0, strategy_p1, info_p0,
                   info_p1):
    """A zeroed level-1 record, pre-filled with everything known before the game is played."""
    row = np.zeros((), dtype=GAME_DTYPE)
    row["run_id"] = run_id
    row["game_index"] = game_index
    row["game_seed"] = game_seed
    row["deck_size"] = deck_size
    row["strategy_p0"] = strategy_p0
    row["strategy_p1"] = strategy_p1
    row["info_mode_p0"] = info_p0
    row["info_mode_p1"] = info_p1
    row["engine_version"] = ENGINE_VERSION
    row["rules_version"] = RULES_VERSION
    row["schema_version"] = SCHEMA_VERSION
    # -1 means "the draw pile never emptied". A column that was always positive would be useless
    # for the per-phase analysis it exists to support.
    row["deck_exhausted_at_round"] = -1
    row["winner"] = NO_WINNER
    row["winner_rounds_played"] = -1
    return row


@njit(cache=True)
def _play_one(outcomes, row_index, rounds, moves, rounds_start, moves_start, run_id, deck_size,
              strategy_p0, strategy_p1, info_p0, info_p1, block_seed, pairing_index, game_index,
              tunables, record_rounds, record_moves, sample_this_game, first_attacker_override):
    """Play one game, writing its level-1 record into ``outcomes[row_index]``.

    `rounds_start` and `moves_start` are the caller's current write offsets, so records accumulate
    across the whole block instead of restarting at zero per game. Returns the *new* offsets.
    Counts are returned rather than kept on the state so the state stays a game and the telemetry
    stays beside it.
    """
    game_seed = seed_to_i64(mix_seed3(block_seed, pairing_index, game_index))
    deal_stream, stream_p0, stream_p1 = game_stream_states(block_seed, pairing_index, game_index)
    # `deal_state` rather than `setup_game`: the engine is compiled, and a `GameStreams` bundle is
    # a Python class that cannot cross into `nopython`. The deal itself has one implementation;
    # `setup_game` is only a wrapper that unwraps a bundle into these three states.
    state = deal_state(deck_size, deal_stream, stream_p0, stream_p1, tunables, False,
                       first_attacker_override)
    row = outcomes[row_index]

    row["run_id"] = run_id
    row["game_index"] = game_index
    row["game_seed"] = game_seed
    row["deck_size"] = deck_size
    row["strategy_p0"] = strategy_p0
    row["strategy_p1"] = strategy_p1
    row["info_mode_p0"] = info_p0
    row["info_mode_p1"] = info_p1
    row["engine_version"] = ENGINE_VERSION
    row["rules_version"] = RULES_VERSION
    row["schema_version"] = SCHEMA_VERSION
    # -1 means "the draw pile never emptied". A column that was always positive would be useless
    # for the per-phase analysis it exists to support.
    row["deck_exhausted_at_round"] = -1
    row["winner"] = NO_WINNER
    row["winner_rounds_played"] = -1
    row["first_attacker"] = state.attacker
    row["pairing_index"] = pairing_index
    row["both_have_trumps"] = state.both_have_trumps
    row["initial_trumps_p0"] = trumps_in_hand(state, 0)
    row["initial_trumps_p1"] = trumps_in_hand(state, 1)
    row["redeals"] = state.redeals
    row["first_attacker_random_fallback"] = state.first_attacker_random_fallback
    row["flag_redeals_exceeded"] = state.flag_redeals_exceeded

    collected = np.zeros(2, dtype=np.int64)
    voluntary = np.zeros(2, dtype=np.int64)
    forced = np.zeros(2, dtype=np.int64)
    trumps_played = np.zeros(2, dtype=np.int64)
    throw_ins = np.zeros(2, dtype=np.int64)
    beaten = np.zeros(2, dtype=np.int64)
    attack_turns = np.zeros(2, dtype=np.int64)
    win_from_single = np.zeros(2, dtype=np.bool_)

    rounds_written = rounds_start
    moves_written = moves_start
    actions = 0
    max_attacks = 0
    cap_hit = False
    illegal = False
    conservation_broken = False
    both_empty = False
    winner_rounds = -1
    deck_ran_out_at = -1
    cards_left_at_deck_out = 0

    # The round being accumulated, flushed when the next one starts.
    round_attacker = state.attacker
    round_attacks = 0
    round_throw_ins = 0
    round_defenses = 0
    round_took = False
    round_take_voluntary = False
    round_roles_swapped = False
    round_open = False

    for _ in range(MAX_ACTIONS_PER_GAME):
        if state.winner != EMPTY:
            break

        player = acting_player(state)
        at_round_start = player == state.attacker and state.table_attack_count == 0

        if at_round_start:
            # The cap condition lives in `rules.round_cap_reached`; the engine reacts by ending
            # the game as a flagged anomaly rather than raising, so one bad game cannot abort a
            # million-game run.
            if round_cap_reached(state, tunables):
                cap_hit = True
                break

            if round_open:
                rounds_written = _flush_round(
                    rounds, rounds_written, rounds_start, row, state, state.round_idx - 1,
                    round_attacker,
                    round_attacks, round_throw_ins, round_defenses, round_took,
                    round_take_voluntary, round_roles_swapped, record_rounds,
                )
            round_attacker = state.attacker
            round_attacks = 0
            round_throw_ins = 0
            round_defenses = 0
            round_took = False
            round_take_voluntary = False
            round_roles_swapped = False
            round_open = True

            attack_turns[player] += 1
            if deck_ran_out_at < 0 and state.deck_len == 0:
                deck_ran_out_at = state.round_idx
                cards_left_at_deck_out = (
                    state.hand_sizes[0] + state.hand_sizes[1]
                )

        offered = legal_actions(state, player)
        if len(offered) == 0:
            illegal = True
            break

        strategy = strategy_p0 if player == 0 else strategy_p1
        info = info_p0 if player == 0 else info_p1
        action = choose_action(state, player, strategy, info)

        if record_moves and sample_this_game:
            moves_written = _record_move(
                moves, moves_written, moves_start, row, state, actions, player, action, offered,
                info
            )

        kind = action_kind(action)
        card = action_card(action)
        plays_a_card = kind == ATTACK or kind == THROW or kind == BEAT
        is_trump = plays_a_card and (card % 4) == state.trump

        if kind == ATTACK:
            round_attacks += 1
        elif kind == THROW:
            round_throw_ins += 1
            throw_ins[player] += 1
        elif kind == BEAT:
            round_defenses += 1
            beaten[player] += 1
        elif kind == TAKE:
            # Counted before the transition: `apply_action` empties the table, so afterwards the
            # cards this take collected are no longer observable anywhere.
            defender = 1 - state.attacker
            collected[defender] += state.table_attack_count + state.table_defense_count
            if _has_legal_beat(state, defender):
                voluntary[defender] += 1
                round_take_voluntary = True
            else:
                forced[defender] += 1
            round_took = True
            round_roles_swapped = True

        if is_trump:
            trumps_played[player] += 1

        hand_before = state.hand_sizes[player]

        apply_action(state, action, tunables)
        actions += 1

        if state.table_attack_count > max_attacks:
            max_attacks = state.table_attack_count

        if check_invariants(state) != 0:
            conservation_broken = True

        if state.winner != EMPTY:
            winner_rounds = state.round_idx
            if state.hand_sizes[1 - state.winner] == 0:
                both_empty = True
            # The degeneracy worth measuring is *attacking* with a last card: the game ends
            # before the defender can respond, so the defender's entire hand was irrelevant.
            # Merely "the winner held one card" would be vacuous -- the win rule is an empty
            # hand, so the winning card is always the last one.
            if hand_before == 1 and kind == ATTACK:
                win_from_single[state.winner] = True

    if round_open:
        rounds_written = _flush_round(
            rounds, rounds_written, rounds_start, row, state, state.round_idx, round_attacker,
            round_attacks,
            round_throw_ins, round_defenses, round_took, round_take_voluntary, round_roles_swapped,
            record_rounds,
        )

    row["winner"] = state.winner if state.winner == 0 or state.winner == 1 else NO_WINNER
    row["rounds"] = state.round_idx
    row["actions"] = actions
    row["final_hand_p0"] = state.hand_sizes[0]
    row["final_hand_p1"] = state.hand_sizes[1]
    row["deck_remaining_at_end"] = state.deck_len
    row["discard_len_at_end"] = state.discard_len
    row["table_cards_at_end"] = state.table_attack_count + state.table_defense_count
    row["winner_rounds_played"] = winner_rounds
    row["deck_exhausted_at_round"] = deck_ran_out_at
    row["cards_left_when_deck_ran_out"] = cards_left_at_deck_out
    if state.winner == 0 or state.winner == 1:
        row["winner_final_hand"] = state.hand_sizes[state.winner]
        row["loser_final_hand"] = state.hand_sizes[1 - state.winner]
    else:
        row["winner_final_hand"] = 0
        row["loser_final_hand"] = 0

    row["cards_collected_p0"] = collected[0]
    row["cards_collected_p1"] = collected[1]
    row["voluntary_takes_p0"] = voluntary[0]
    row["voluntary_takes_p1"] = voluntary[1]
    row["forced_takes_p0"] = forced[0]
    row["forced_takes_p1"] = forced[1]
    row["trumps_played_p0"] = trumps_played[0]
    row["trumps_played_p1"] = trumps_played[1]
    row["throw_ins_p0"] = throw_ins[0]
    row["throw_ins_p1"] = throw_ins[1]
    row["cards_beaten_p0"] = beaten[0]
    row["cards_beaten_p1"] = beaten[1]
    row["attack_turns_p0"] = attack_turns[0]
    row["attack_turns_p1"] = attack_turns[1]
    # The endgame is degenerate by construction: an attacker holding one card wins outright,
    # because playing the last card ends the game before the defender responds. Recorded because
    # its frequency is worth knowing, not because it is a defect.
    row["win_from_single_card_p0"] = win_from_single[0]
    row["win_from_single_card_p1"] = win_from_single[1]

    row["max_attacks_in_a_round"] = max_attacks
    row["flag_round_cap_hit"] = cap_hit
    row["flag_illegal_move"] = illegal
    row["flag_card_conservation_violation"] = conservation_broken
    row["flag_both_players_empty"] = both_empty
    row["flag_anomaly_any"] = (
        cap_hit or illegal or conservation_broken or both_empty
        or bool(state.flag_redeals_exceeded)
    )
    # Telemetry completeness, recorded separately from the data-quality anomalies above: the game
    # was played correctly, but its per-round or per-move record was bounded. Deliberately kept out
    # of `flag_anomaly_any` -- a truncated run is not an invalid game, and conflating the two would
    # make a perfectly good dataset look broken.
    row["flag_round_records_truncated"] = (
        record_rounds and (rounds_written - rounds_start) >= ROUNDS_PER_GAME_LIMIT
    )
    row["flag_move_records_truncated"] = (
        record_moves and sample_this_game
        and (moves_written - moves_start) >= MOVES_PER_GAME_LIMIT
    )

    return rounds_written, moves_written


@njit(cache=True)
def _has_legal_beat(state: GameState, defender: int) -> bool:
    """Whether `defender` has any beat available against the card it is being asked about.

    Read from the legal-action list rather than re-deriving the beat relation, so a take is
    classified as voluntary or forced using exactly the rules' own notion of a legal beat.
    """
    if state.table_attack_count <= state.table_defense_count:
        return False
    for i in range(len(legal_actions(state, defender))):
        if action_kind(int(legal_actions(state, defender)[i])) == BEAT:
            return True
    return False


@njit(cache=True)
def _flush_round(rounds, written, round_start, row, state, round_idx, attacker, n_attacks,
                 n_throw_ins, n_defenses, defender_took, take_voluntary, roles_swapped, record):
    """Write one level-2 record, returning the new write count.

    `record` is False when level 2 is switched off, in which case nothing is written and the
    caller carries on with the same array at zero length.
    """
    if not record or (n_attacks == 0 and n_defenses == 0):
        return written
    if (written - round_start) >= ROUNDS_PER_GAME_LIMIT:
        # A stalled game emits 10,000 rounds against ~14 for a normal one, and one of those
        # would dominate the level-2 dataset. Truncated here and flagged on the level-1 row, so
        # the dataset stays analysable and nothing disappears without a trace.
        return written

    rec = rounds[written]
    rec["run_id"] = row["run_id"]
    rec["game_index"] = row["game_index"]
    rec["pairing_index"] = row["pairing_index"]
    rec["round_idx"] = round_idx
    rec["attacker"] = attacker
    rec["n_attacks"] = n_attacks
    rec["n_throw_ins"] = n_throw_ins
    rec["n_defenses"] = n_defenses
    rec["defender_took"] = defender_took
    rec["take_voluntary"] = take_voluntary
    rec["hand_p0"] = state.hand_sizes[0]
    rec["hand_p1"] = state.hand_sizes[1]
    rec["deck_remaining"] = state.deck_len
    rec["trumps_p0"] = trumps_in_hand(state, 0)
    rec["trumps_p1"] = trumps_in_hand(state, 1)
    rec["roles_swapped"] = roles_swapped
    rec["winner_in_round"] = state.winner
    rec["phase"] = phase_of(state, state.initial_deck)
    return written + 1


@njit(cache=True)
def _record_move(moves, written, move_start, row, state, seq, player, action, offered, info):
    """Write one level-3 record, returning the new write count."""
    if (written - move_start) >= MOVES_PER_GAME_LIMIT:
        # Same reasoning as the round records: a stalled game would otherwise contribute 10,000
        # moves to level 3 and swamp the sampled mechanism analysis.
        return written

    index = -1
    for i in range(len(offered)):
        if int(offered[i]) == action:
            index = i
            break

    rec = moves[written]
    rec["run_id"] = row["run_id"]
    rec["game_index"] = row["game_index"]
    rec["pairing_index"] = row["pairing_index"]
    rec["round_idx"] = state.round_idx
    rec["seq"] = seq
    rec["actor"] = player
    rec["action_type"] = action_kind(action)
    rec["card_id"] = action_card(action)
    rec["info_mode"] = info
    rec["legal_action_count"] = len(offered)
    # -1 would mean the chosen action was absent from the legal list, which is a bug worth
    # seeing in the data rather than a value to paper over.
    rec["chosen_index_in_legal_order"] = index
    return written + 1


# --- public entry points -----------------------------------------------------


@njit(cache=True)
def play_block(deck_size, strategy_p0, strategy_p1, info_p0, info_p1, block_seed, pairing_index,
               game_start, game_count, settings, record_rounds, record_moves, run_id=0,
               sample_modulus=DEFAULT_SAMPLE_MODULUS,
               first_attacker_override=USE_RULE):
    """Play `game_count` games, with `game_index = game_start + i`.

    Returns ``(outcomes, rounds, moves)``. The round and move arrays are empty when the matching
    flag is False -- not allocated-and-trimmed, but genuinely zero-length, so the cheap
    configuration costs what it looks like it costs. The benchmark that decides the dataset size
    measures this function.
    """
    outcomes = np.zeros(game_count, dtype=GAME_DTYPE)

    rounds = np.zeros(game_count * ROUNDS_PER_GAME_LIMIT, dtype=ROUND_DTYPE) \
        if record_rounds else np.zeros(0, dtype=ROUND_DTYPE)
    moves = np.zeros(game_count * MOVES_PER_GAME_LIMIT, dtype=MOVE_DTYPE) \
        if record_moves else np.zeros(0, dtype=MOVE_DTYPE)

    rounds_written = 0
    moves_written = 0

    for i in range(game_count):
        game_index = game_start + i
        # Deterministic sampling by game index, never by RNG: an RNG-based sample would differ
        # between a 1-worker and an 8-worker run and break reproducibility.
        sample_this_game = record_moves and (game_index % sample_modulus == 0)
        # The write offsets are threaded *into* each game. Starting every game at zero was a real
        # defect: every game overwrote the previous game's records, so only the last game's rows
        # survived while the returned counts implied all of them. Level 2 and level 3 would have
        # been silently wrong for every multi-game block.
        written, move_count = _play_one(
            outcomes, i, rounds, moves, rounds_written, moves_written,
            run_id, deck_size, strategy_p0, strategy_p1, info_p0,
            info_p1, block_seed, pairing_index, game_index, settings, record_rounds,
            record_moves, sample_this_game, first_attacker_override,
        )
        rounds_written = written
        moves_written = move_count

    return outcomes[:game_count], rounds[:rounds_written], moves[:moves_written]


@njit(cache=True)
def play_game(deck_size, strategy_p0, strategy_p1, info_p0, info_p1, seed, settings,
              record_rounds, record_moves, pairing_index=0, game_index=0, run_id=0,
              sample_modulus=DEFAULT_SAMPLE_MODULUS, first_attacker_override=USE_RULE):
    """Play one game and return its level-1 record.

    A one-game block, so that a single game and a game inside a block cannot diverge. The
    `pairing_index` and `game_index` arguments exist because the seed is not enough on its own:
    a game's identity is the triple, and letting a caller pass only a seed would make two
    different games in a block indistinguishable.
    """
    outcomes, _, _ = play_block(
        deck_size, strategy_p0, strategy_p1, info_p0, info_p1, seed, pairing_index, game_index, 1,
        settings, record_rounds, record_moves, run_id, sample_modulus, first_attacker_override,
    )
    return outcomes[0]
