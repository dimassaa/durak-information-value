"""Per-game data-quality audit.

Spec 7.2 sets the failure policy: an invariant violation is **raised and marked**, never
swallowed and averaged over. This module is the independent second opinion. The engine already
evaluates invariants during play and writes a bitmask of flags per game; this module re-derives
the same properties from a finished state and its outcome record, and the two are cross-checked
against each other in `tests/test_telemetry.py`.

Why bother with a second implementation at all? Because a check that shares code with what it
checks reports nothing. The engine's flags are set from `check_invariants`, evaluated on
*intermediate* states during play; the audit looks at the *final* state and the recorded outcome,
which is a different input and a different set of questions. A game could end clean while its
record says otherwise, or the reverse, and only two independent passes will disagree.

What it is not
--------------
It is not a repair mechanism and not a filter. It never fixes a state and never drops a game. A
flagged game stays in the dataset with its flag set, because the anomaly counts appear in every
dashboard regardless -- hiding the offending rows would make a broken run look like a clean one.
Deciding whether a run is publishable is the reports' job, not this module's.

Anomaly names are stable strings
--------------------------------
They go into report tables and anomaly columns, so they are part of the project's output
contract. They are returned in check order, matching the order spec 7.1 lists the invariants in,
so a report's anomaly block reads in the same sequence as the spec.
"""

from __future__ import annotations

import numpy as np

from durak import rules


def audit_game(state, outcome) -> list:
    """Every violated data-quality property of one finished game, in check order.

    `state` is the finished `GameState` and `outcome` its level-1 record. Both are read, never
    written: an audit that repaired a state would make the audit's verdict describe something
    other than what was played.

    Returns an empty list for a clean game, which is the case that matters most -- an audit that
    flags everything looks identical to one that works, right up until the day everything is
    flagged and nobody notices.
    """
    found = []
    winner = int(outcome["winner"])

    # Spec 7.1 invariant 1: no strategy returned an illegal action.
    if bool(outcome["flag_illegal_move"]):
        found.append("illegal_action")

    # Invariants 2 and 3: conservation and uniqueness. The expensive class of bug: a card in
    # two hands leaves the game perfectly playable while corrupting every metric computed from
    # it, so these are checked first and named explicitly.
    if _counters_exceed_arrays(state):
        found.append("length_counter_past_array")
    if _conservation_broken(state, outcome):
        found.append("card_conservation_violation")
    if _has_duplicate(state):
        found.append("card_in_two_locations")

    # Invariant 4: the winner's hand is empty and the loser's is not.
    if winner in (0, 1):
        if state.hand_sizes[winner] != 0:
            found.append("winner_hand_not_empty")
        if state.hand_sizes[1 - winner] <= 0:
            found.append("loser_hand_empty")
    else:
        found.append("no_winner_recorded")

    # Invariant 5: both players are never empty at once.
    if state.hand_sizes[0] == 0 and state.hand_sizes[1] == 0:
        found.append("both_players_empty")

    # Invariant 6: the game terminated within the round cap.
    if bool(outcome["flag_round_cap_hit"]) or state.round_idx >= rules.MAX_ROUNDS:
        found.append("round_cap_hit")

    # Invariant 7: redeals stayed within the cap.
    if state.redeals > rules.MAX_REDEALS or bool(state.flag_redeals_exceeded):
        found.append("redeal_cap_exceeded")

    # The outcome record must agree with the state it claims to describe. A disagreement here
    # would mean the telemetry is wrong even where the game was played correctly, which is a
    # different defect with the same visible symptom.
    found.extend(_record_disagreements(state, outcome, winner))

    return found


def _conservation_broken(state, outcome) -> bool:
    """Every card in a hand, on the table, or in the discard -- plus the draw pile."""
    on_table = state.table_attack_count + state.table_defense_count
    accounted = (
        state.deck_len
        + int(state.hand_sizes[0])
        + int(state.hand_sizes[1])
        + state.discard_len
        + on_table
    )
    return accounted != state.initial_deck


def _has_duplicate(state) -> bool:
    """Whether any card sits in two places at once, as a 64-bit mask.

    Reimplements the engine's `_has_duplicate` rather than calling it: the point of this module
    is to be an independent second opinion, and sharing the implementation would defeat that. The
    duplication is five short loops over the same card arrays, and the tests assert it agrees
    with the engine's own finding.

    Every loop is clamped to its array's length. An audit runs *on* the state it is auditing, so
    a corrupted counter -- a `discard_len` past the end of the array -- must produce a finding,
    not an ``IndexError``. An audit that crashes on corruption reports nothing about the corruption
    it was asked about.
    """
    seen = 0
    for player in (0, 1):
        limit = min(int(state.hand_sizes[player]), state.hands.shape[1])
        for i in range(limit):
            card = int(state.hands[player, i])
            bit = 1 << card
            if seen & bit:
                return True
            seen |= bit
    for i in range(min(state.deck_len, state.deck.shape[0])):
        card = int(state.deck[i])
        bit = 1 << card
        if seen & bit:
            return True
        seen |= bit
    for i in range(min(state.discard_len, state.discard.shape[0])):
        card = int(state.discard[i])
        bit = 1 << card
        if seen & bit:
            return True
        seen |= bit
    for i in range(min(state.table_attack_count, state.table_attacks.shape[0])):
        card = int(state.table_attacks[i])
        bit = 1 << card
        if seen & bit:
            return True
        seen |= bit
    for i in range(min(state.table_defense_count, state.table_defenses.shape[0])):
        card = int(state.table_defenses[i])
        bit = 1 << card
        if seen & bit:
            return True
        seen |= bit
    return False


def _counters_exceed_arrays(state) -> bool:
    """Whether any length counter points past the array it indexes.

    Separate from conservation because it is a different failure: conservation can hold while a
    counter is out of range, and an out-of-range counter is what makes the duplicate scan
    dangerous to run unguarded.
    """
    return (
        state.deck_len > state.deck.shape[0]
        or state.discard_len > state.discard.shape[0]
        or state.table_attack_count > state.table_attacks.shape[0]
        or state.table_defense_count > state.table_defenses.shape[0]
        or int(state.hand_sizes[0]) > state.hands.shape[1]
        or int(state.hand_sizes[1]) > state.hands.shape[1]
    )


def _record_disagreements(state, outcome, winner) -> list:
    """Places where the outcome record contradicts the finished state."""
    problems = []

    if winner in (0, 1):
        if int(outcome["winner_final_hand"]) != state.hand_sizes[winner]:
            problems.append("record_winner_hand_mismatch")
        if int(outcome["loser_final_hand"]) != state.hand_sizes[1 - winner]:
            problems.append("record_loser_hand_mismatch")

    if int(outcome["deck_remaining_at_end"]) != state.deck_len:
        problems.append("record_deck_remaining_mismatch")
    if int(outcome["discard_len_at_end"]) != state.discard_len:
        problems.append("record_discard_len_mismatch")
    if int(outcome["rounds"]) != state.round_idx:
        problems.append("record_round_count_mismatch")
    if int(outcome["actions"]) < 1:
        problems.append("record_zero_actions")

    return problems


def audit_outcomes(outcomes) -> dict:
    """Aggregate an outcome batch into a data-quality report.

    Counts anomalies per kind and never drops a row: spec 7.2 makes a run with any flagged game
    unpublishable until the flags are explained, which requires knowing how many there are. The
    engine's own flags are compared against the audit's verdict for the same batch, so a
    disagreement between the two implementations shows up as a number rather than as a silently
    different report.

    Only the checks that are decidable from the record alone are run here -- the per-state checks
    need a state the record does not carry. `audit_game` is the full pass.
    """
    outcomes = np.asarray(outcomes)
    total = len(outcomes)

    by_kind: dict = {}
    for name, column in (
        ("illegal_action", "flag_illegal_move"),
        ("round_cap_hit", "flag_round_cap_hit"),
        ("card_conservation_violation", "flag_card_conservation_violation"),
        ("both_players_empty", "flag_both_players_empty"),
        ("redeal_cap_exceeded", "flag_redeals_exceeded"),
    ):
        hits = int(outcomes[column].sum())
        if hits:
            by_kind[name] = hits

    # `no_winner_recorded` is the audit's own verdict rather than a copy of a flag, so it can
    # disagree with the engine. That disagreement is the point of the second opinion.
    no_winner = int((outcomes["winner"] < 0).sum())

    engine_flagged = int(outcomes["flag_anomaly_any"].sum())
    if no_winner:
        by_kind["no_winner_recorded"] = no_winner

    return {
        "games": total,
        # Every flagged game is counted, never dropped: spec 7.2 makes a run with any flagged
        # game unpublishable until the flags are explained, and that needs a number.
        "anomalous_games": engine_flagged,
        "by_kind": by_kind,
        "flagged_by_engine": engine_flagged,
        # Publishable only with zero anomalies. Stated here, enforced by the reports: this
        # module's job is to count, not to decide whether a run ships.
        "clean": engine_flagged == 0 and no_winner == 0,
    }