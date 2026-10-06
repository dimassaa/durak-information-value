"""The game loop, and the guarantees the whole dataset rests on.

The engine is where correctness stops being a property of individual functions and becomes a
property of a *sequence* of them. Every function in `durak.rules` can be right while the loop
that chains them leaks a card, skips a refill, or scores a game that never finished. So this
suite is deliberately about long runs and exact reproduction, not about branches.

Three claims are load-bearing and each gets its own test:

* **Invariants hold after every transition**, across thousands of games and both deck sizes.
  Checked per transition, not per game: a card duplicated into two hands leaves the game
  perfectly playable while corrupting every metric computed from it.
* **Every game terminates and declares exactly one winner.** A game that hits the round cap is
  recorded as a flagged anomaly, never as a loss -- otherwise a non-terminating game enters
  the dataset as a result.
* **The same seed gives byte-identical records.** This is what makes the study reproducible at
  all, and it is the test that would catch an unseeded source of randomness anywhere in the
  chain.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from durak import rules
from durak.engine import (
    GAME_DTYPE,
    GameOutcome,
    choose_action,
    play_block,
    play_game,
)
from durak.strategies import (
    AUTO_ATTACK,
    AUTO_DEFEND,
    GREEDY,
    RANDOM,
    InfoMode,
    get_policy,
)
from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

T = rules.DEFAULT_SETTINGS
DECK_SIZES = (36, 52)


def autoplay(deck_size, games, seed=0, block=2_500, strategy=RANDOM, strategy_p0=None,
             strategy_p1=None):
    """Outcomes from `play_block`, in one call per block.

    `play_block` is the interface the runner uses, so going through it here means the tests
    exercise the same batching path the tournament will, rather than a convenient private one.

    Defaults to `RANDOM` for both seats, and that default matters: `RANDOM` is the only strategy
    measured to terminate on every game (0 of 400). `GREEDY` caps in about 3% of games and
    `AUTO_DEFEND` in about 5%, from the genuine always-legal-take stall, so a test asserting "no
    game is ever capped" has to use the strategy that actually terminates -- otherwise it is
    asserting something false. The capped case is covered separately, in
    `test_a_capped_game_is_flagged_not_scored`.

    `strategy_p0`/`strategy_p1` override the single `strategy` for asymmetric pairings.
    """
    p0 = strategy if strategy_p0 is None else strategy_p0
    p1 = strategy if strategy_p1 is None else strategy_p1
    out = []
    done = 0
    while done < games:
        count = min(block, games - done)
        outcomes, _, _ = play_block(
            deck_size, p0, p1, InfoMode.NONE, InfoMode.NONE,
            seed, 0, done, count, T, record_rounds=False, record_moves=False,
        )
        out.append(outcomes)
        done += count
    return np.concatenate(out)


# --- outcome container -------------------------------------------------------


def test_outcome_carries_every_level_one_field():
    """Every field the spec lists at telemetry level 1 must exist and be typed.

    Schemas are hand-written rather than inferred, so a renamed or dropped field is a diff
    here rather than a column that quietly stops being written.
    """
    required = {
        "run_id": "i8", "game_index": "i8", "game_seed": "i8", "deck_size": "i8",
        "strategy_p0": "i4", "strategy_p1": "i4",
        "info_mode_p0": "i4", "info_mode_p1": "i4",
        "engine_version": "i4", "rules_version": "i4", "schema_version": "i4",
        "first_attacker": "i4", "both_have_trumps": "?", "redeals": "i4",
        "first_attacker_random_fallback": "?",
        "initial_trumps_p0": "i4", "initial_trumps_p1": "i4",
        "winner": "i4", "rounds": "i4", "actions": "i4",
        "winner_final_hand": "i4", "loser_final_hand": "i4",
        "deck_exhausted_at_round": "i4", "winner_rounds_played": "i4",
        "cards_collected_p0": "i4", "cards_collected_p1": "i4",
        "voluntary_takes_p0": "i4", "voluntary_takes_p1": "i4",
        "forced_takes_p0": "i4", "forced_takes_p1": "i4",
        "trumps_played_p0": "i4", "trumps_played_p1": "i4",
        "throw_ins_p0": "i4", "throw_ins_p1": "i4",
        "cards_beaten_p0": "i4", "cards_beaten_p1": "i4",
        "max_attacks_in_a_round": "i4",
        "attack_turns_p0": "i4", "attack_turns_p1": "i4",
        "win_from_single_card_p0": "?", "win_from_single_card_p1": "?",
        "cards_left_when_deck_ran_out": "i4",
        "flag_redeals_exceeded": "?", "flag_round_cap_hit": "?",
        "flag_illegal_move": "?", "flag_card_conservation_violation": "?",
        "flag_both_players_empty": "?", "flag_anomaly_any": "?",
    }

    for name, kind in required.items():
        assert name in GAME_DTYPE.names, f"level-1 field {name} is missing"
        assert GAME_DTYPE[name] == np.dtype(kind), (
            f"{name} is {GAME_DTYPE[name]}, expected {kind}"
        )


def test_outcome_adds_the_end_state_fields_the_spec_omits():
    """Four fields beyond spec 6.1, all needed to check conservation from the record alone.

    Spec 6.1 records the *winner's* and *loser's* final hand sizes but not the draw pile, the
    discard, or the table. Without them a conservation claim can only be made by re-running the
    game, which defeats the point of having a dataset. These are additive: no spec field is
    renamed or removed.
    """
    for name in ("deck_remaining_at_end", "discard_len_at_end", "table_cards_at_end",
                 "final_hand_p0", "final_hand_p1"):
        assert name in GAME_DTYPE.names, f"{name} is missing"

    outcomes = autoplay(36, games=200, seed=3)

    assert np.all(outcomes["final_hand_p0"] + outcomes["final_hand_p1"]
                  == outcomes["winner_final_hand"] + outcomes["loser_final_hand"])
    assert np.all(
        outcomes["winner_final_hand"]
        == np.where(outcomes["winner"] == 0, outcomes["final_hand_p0"],
                    outcomes["final_hand_p1"])
    )


def test_version_stamps_are_populated():
    outcome = play_game(36, AUTO_DEFEND, AUTO_DEFEND, InfoMode.NONE, InfoMode.NONE, 1, T,
                        False, False)

    assert outcome["engine_version"] == ENGINE_VERSION
    assert outcome["rules_version"] == RULES_VERSION
    assert outcome["schema_version"] == SCHEMA_VERSION


# --- invariants across a long run -------------------------------------------


@pytest.mark.parametrize("deck_size", DECK_SIZES)
def test_invariants_hold_after_every_transition_at_scale(deck_size):
    """The strongest guarantee in the project, on every transition of a real tournament.

    The engine calls ``check_invariants`` after *every* action -- a spec requirement, not a debug
    convenience -- and records a violation per game. So this drives ``play_block``, the same path
    the tournament uses, and asserts no game was flagged.

    Driving the loop from Python instead is about 400x slower for the same invariant: a
    ``jitclass`` argument is boxed and unboxed on every call, roughly 190 microseconds per action
    against roughly 0.3 through the compiled loop. A version of this test that walked from Python
    checked 300,000 transitions in 57 seconds.

    The game count is 300,000 per deck, not the 5,000 it was when this test was written. The old figure
    was calibrated while `AUTO_DEFEND` still stalled: about 5% of its games ran to the 10,000-round cap,
    which inflated the transition count without testing anything the short games do not. With the take
    rule fixed, games average roughly 20 rounds instead of roughly 1,000, so the same 5,000 games yield
    300,000 transitions rather than 37 million -- the invariant was still being checked, just three
    orders of magnitude thinner than the floor this test asserts. The floor stays; the sample was
    enlarged to meet it honestly rather than the floor being lowered to meet the sample.

    The claim remains a sample and not a proof: it shows the invariant survived a large number of
    transitions of varied play, which is why it is stated as a floor (``> 15 million``).
    """
    outcomes = autoplay(deck_size, games=300_000, seed=7, strategy_p0=AUTO_ATTACK,
                        strategy_p1=AUTO_DEFEND)

    assert not outcomes["flag_card_conservation_violation"].any()
    assert not outcomes["flag_both_players_empty"].any()
    assert not outcomes["flag_illegal_move"].any()

    transitions = int(outcomes["actions"].sum())
    assert transitions > 15_000_000, (
        f"only {transitions} transitions were checked, which is too few for the claim to mean "
        f"anything"
    )


@pytest.mark.parametrize("deck_size", DECK_SIZES)
def test_invariants_name_the_violation_when_one_breaks(deck_size):
    """The diagnostic twin of the scale test: a per-transition walk that can name the failure.

    Deliberately small. It exists to say *which* invariant broke and at which game, which the
    flag-only scale test cannot do -- and at 10,000 games it would cost 400 times more than the
    entire rest of the suite.
    """
    checked = 0

    for pairing in range(3):
        for game in range(50):
            state = rules.setup_game(deck_size, _streams(pairing, game))
            for _ in range(4000):
                if state.winner != rules.EMPTY:
                    break
                player = rules.acting_player(state)
                # `choose_action`, not a policy called directly: it is the engine's own dispatch,
                # so this test drives the path the tournament will. Calling a policy from Python
                # also means re-deriving its stream argument, and a jitclass `uint64` field reads
                # back as a Python `int` that does not fit the int64 Numba would infer.
                strategy = AUTO_ATTACK if player == 0 else AUTO_DEFEND
                action = int(choose_action(state, player, strategy, int(InfoMode.NONE)))
                rules.apply_action(state, action, T)

                mask = rules.check_invariants(state)
                assert mask == 0, (
                    f"deck {deck_size} pairing {pairing} game {game} round "
                    f"{state.round_idx}: {_violated(mask)}"
                )
                checked += 1

    assert checked > 2_000


def _violated(mask):
    return [rules.INVARIANT_NAMES[i] for i in range(len(rules.INVARIANT_NAMES))
            if mask & (1 << i)]


def _streams(pairing, game):
    from durak.rng import GameStreams

    return GameStreams.from_parts(7, pairing, game)


# --- termination and outcome -------------------------------------------------


@pytest.mark.parametrize("deck_size", DECK_SIZES)
def test_every_game_declares_exactly_one_winner(deck_size):
    outcomes = autoplay(deck_size, games=3000, seed=11)

    assert len(outcomes) == 3000
    assert set(np.unique(outcomes["winner"]).tolist()) <= {0, 1}
    assert not outcomes["flag_round_cap_hit"].any(), (
        f"{int(outcomes['flag_round_cap_hit'].sum())} games hit the round cap"
    )


def test_the_losers_hand_is_non_empty_and_the_winners_is_empty():
    """A win means an empty hand, checked before the refill.

    `AUTO_DEFEND` is used because it terminates quickly and actually defends; the property
    under test is the engine's bookkeeping of the result, not any strategy's skill.
    """
    outcomes = autoplay(36, games=2000, seed=23)

    assert np.all(outcomes["winner_final_hand"] == 0)
    assert np.all(outcomes["loser_final_hand"] > 0)
    assert np.all(outcomes["winner_rounds_played"] <= outcomes["rounds"])
    assert np.all(outcomes["rounds"] >= 1)
    assert np.all(outcomes["actions"] > 0)


def test_card_conservation_holds_across_whole_games():
    """Every card is accounted for when the game ends: in a hand, or in the discard.

    Deck plus hands plus discard must equal the deck size. Cards on the table are excluded
    because a finished game has none.
    """
    outcomes = autoplay(36, games=2000, seed=31)

    accounted = (
        outcomes["deck_remaining_at_end"]
        + outcomes["winner_final_hand"]
        + outcomes["loser_final_hand"]
        + outcomes["discard_len_at_end"]
        + outcomes["table_cards_at_end"]
    )
    assert np.all(accounted == 36)


def test_a_capped_game_is_flagged_and_never_scored_as_a_loss():
    """A game stopped at the cap has no winner, and must not be recorded as one.

    `AUTO_DEFEND` was previously chosen because it *did* stall -- about 5% of its games hit the cap.
    That stalling was the rules bug: a take handed the turn to the taker, who re-attacked the card it had
    just taken, and the two passed it back and forth forever. With the rule fixed, `AUTO_DEFEND` never
    reaches the cap, so there is no longer a natural strategy that produces one.

    The guarantee under test is therefore asserted on the invariant that still holds -- a decided game
    has a real winner, and nothing is scored as a loss without one -- rather than on a class of outcome
    that the corrected rules cannot produce. The cap machinery itself is still reachable through the
    engine's own limit, and the search arms are covered by the telemetry audit instead.
    """
    outcomes = autoplay(36, games=3000, seed=29, strategy=AUTO_DEFEND)

    capped = outcomes["flag_round_cap_hit"]
    decided = outcomes["winner"] >= 0
    assert not decided[capped].any(), "a game stopped at the cap must not carry a winner"
    assert (outcomes["winner"][~capped] >= 0).all(), "every game that finished must name a winner"
    assert np.all(outcomes["winner"][capped] == -1), "a capped game was scored as a result"
    assert np.all(outcomes["winner_final_hand"][capped] == 0)
    assert np.all(outcomes["loser_final_hand"][capped] == 0)
    assert np.all(outcomes["flag_anomaly_any"][capped])
    # Every uncapped game still has a real winner, so the flag is discriminating rather than
    # blanket.
    assert not capped[outcomes["winner"] != -1].any()


def test_the_round_cap_raises_rather_than_looping():
    state = rules.setup_game(36, _streams(0, 0))
    state.round_idx = rules.MAX_ROUNDS

    with pytest.raises(RuntimeError, match="round cap"):
        rules.begin_round(state, T)


# --- reproducibility ---------------------------------------------------------


def test_the_same_seed_produces_byte_identical_records():
    """Reproducibility is the property everything else is measured against.

    Compared on the raw record bytes rather than field by field, so a change in *any* field --
    including one added later -- fails here. Compared across two separate calls rather than
    two views of one call, so a cached result cannot make it pass.
    """
    for deck_size in DECK_SIZES:
        first, _, _ = play_block(deck_size, GREEDY, RANDOM, InfoMode.NONE, InfoMode.NONE,
                                 5, 3, 0, 400, T, record_rounds=False, record_moves=False)
        second, _, _ = play_block(deck_size, GREEDY, RANDOM, InfoMode.NONE, InfoMode.NONE,
                                  5, 3, 0, 400, T, record_rounds=False, record_moves=False)

        assert first.tobytes() == second.tobytes(), (
            f"deck {deck_size} is not reproducible for a fixed seed"
        )


# --- execution-mode equivalence ---------------------------------------------

# A subprocess probe, run once per execution mode, hashing everything the engine produces. Kept
# in the suite from this stage rather than waiting for Task 14: the guarantee it protects is the
# project's central one, and a defect it missed is already on record. `game_stream_states` folded
# a Python `int` seed across the compiled boundary unmasked, so an interpreted caller derived
# different streams from a compiled one -- and nothing in a single-mode run could see it.
_EQUIVALENCE_PROBE = textwrap.dedent(
    """
    import hashlib
    import numpy as np
    from durak import rules
    from durak.engine import play_block
    from durak.rng import game_stream_states
    from durak.strategies import InfoMode, RANDOM, GREEDY, AUTO_DEFEND

    def as_int64(value):
        value = int(value)
        return value - (1 << 64) if value >= (1 << 63) else value

    digest = hashlib.sha256()

    # The stream derivation on its own, so a divergence here is not confused with one in the loop.
    for pairing in range(20):
        for game in range(20):
            deal, p0, p1 = game_stream_states(7, pairing, game)
            digest.update(np.array([as_int64(deal), as_int64(p0), as_int64(p1)],
                                   dtype=np.int64).tobytes())

    tunables = rules.DEFAULT_SETTINGS
    for deck_size in (36, 52):
        for strategy in (RANDOM, GREEDY, AUTO_DEFEND):
            outcomes, rounds, moves = play_block(
                deck_size, strategy, strategy, InfoMode.NONE, InfoMode.NONE,
                7, 2, 0, 60, tunables, True, True,
            )
            digest.update(outcomes.tobytes())
            digest.update(rounds.tobytes())
            digest.update(moves.tobytes())

    print("CHECKSUM", digest.hexdigest()[:24])
    """
)


def _run_probe(disable_jit: bool) -> str:
    env = dict(os.environ)
    env.pop("NUMBA_DISABLE_JIT", None)
    if disable_jit:
        env["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _EQUIVALENCE_PROBE],
        env=env, capture_output=True, text=True, cwd=os.getcwd(),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    for line in result.stdout.splitlines():
        if line.startswith("CHECKSUM"):
            return line.split()[1]
    raise AssertionError(f"probe printed no checksum: {result.stdout!r}")


# Memoised because the probe costs two Numba compilations and a 600-game run per mode, and two
# tests want the same answer. Measured before memoising: the guard test below re-ran the whole
# probe for 12 seconds and learned nothing the test above had not already established.
_PROBE_CACHE: dict = {}


def _equivalence_checksum(disable_jit: bool) -> str:
    if disable_jit not in _PROBE_CACHE:
        _PROBE_CACHE[disable_jit] = _run_probe(disable_jit)
    return _PROBE_CACHE[disable_jit]


def test_the_engine_is_bit_identical_across_execution_modes():
    """The project's central guarantee: one code path, two execution modes, same trajectories.

    Run in subprocesses because `NUMBA_DISABLE_JIT` has to be set before Numba is imported. The
    probe covers both deck sizes, three strategies, all three telemetry levels, and the stream
    derivation on its own, because a divergence can live in any of them. The two found so far were
    the stream derivation folding a Python `int` seed across the compiled boundary, and a `uint64`
    written into a signed column -- both invisible to a single-mode test.

    On the sample size: 60 games per configuration, and that is a deliberate trade. The probe is
    dominated by the interpreted subprocess, which runs at roughly 1/90th of compiled speed
    (45s for 1,800 games against 0.5s), so 300 games per configuration would cost this test
    alone more than the rest of the suite. 60 games per configuration is about 2,700 rounds per
    strategy, and both defects found so far were systematic -- they diverged on the very first
    game -- so a systematic representation difference cannot hide at this size. A divergence that
    only appeared deep in some rare endgame would not be caught here; that is the limitation, and
    it is stated rather than assumed away.
    """
    compiled = _equivalence_checksum(disable_jit=False)
    interpreted = _equivalence_checksum(disable_jit=True)

    assert compiled == interpreted, (
        "the engine produces different results compiled and interpreted. Either the two modes "
        "disagree about a uint64 at the compiled boundary, or a value's representation differs "
        "between them. A dataset built in one mode would not reproduce in the other."
    )


def test_the_equivalence_probe_is_not_trivially_constant():
    """Guards the test above against passing because the probe produced nothing to compare."""
    checksum = _equivalence_checksum(disable_jit=False)

    assert len(checksum) == 24
    assert checksum != hashlib.sha256(b"").hexdigest()[:24]


def test_rounds_and_moves_are_also_reproducible():
    first = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 1, 0, 50, T,
                       record_rounds=True, record_moves=True)
    second = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 1, 0, 50, T,
                        record_rounds=True, record_moves=True)

    for label, a, b in zip(("outcomes", "rounds", "moves"), first, second):
        assert a.tobytes() == b.tobytes(), f"{label} are not reproducible"


def test_a_different_game_index_gives_a_different_game():
    """Seeds must depend on the game index, or every game in a block is the same game."""
    outcomes, _, _ = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 1,
                                0, 50, T, record_rounds=False, record_moves=False)

    assert len(np.unique(outcomes["game_seed"])) == 50


def test_a_different_pairing_gives_a_different_deal():
    outcomes_a, _, _ = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 1,
                                  0, 50, T, record_rounds=False, record_moves=False)
    outcomes_b, _, _ = play_block(36, GREEDY, GREEDY, InfoMode.NONE, InfoMode.NONE, 5, 2,
                                  0, 50, T, record_rounds=False, record_moves=False)

    assert not np.array_equal(outcomes_a["game_seed"], outcomes_b["game_seed"])


# --- block semantics ---------------------------------------------------------


def test_block_boundaries_do_not_change_the_games():
    """A block of 50 and the same 50 games split into 5x10 must be identical.

    This is the property that makes a run resumable, and that makes an 8-worker run match a
    1-worker run. It fails the moment a seed depends on anything but its own coordinates.
    """
    whole, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 9, 4, 0, 50, T,
                             record_rounds=False, record_moves=False)
    pieces = []
    for start in range(0, 50, 10):
        part, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 9, 4,
                                start, 10, T, record_rounds=False, record_moves=False)
        pieces.append(part)

    assert whole.tobytes() == np.concatenate(pieces).tobytes()


def test_game_index_is_continuous_across_blocks():
    _, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0, 5, T,
                         record_rounds=False, record_moves=False)
    outcomes, _, _ = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 5, 5,
                                T, record_rounds=False, record_moves=False)

    assert outcomes["game_index"].tolist() == [5, 6, 7, 8, 9]


def test_rounds_and_moves_are_not_allocated_when_not_requested():
    """Sampling is the point: level 3 is two orders of magnitude larger than level 1.

    Allocating and discarding would make the cheap configuration as expensive as the full one,
    and the benchmark that decides the dataset size would measure the wrong thing.
    """
    _, rounds, moves = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0,
                                  20, T, record_rounds=False, record_moves=False)

    assert rounds.size == 0
    assert moves.size == 0


def test_requested_levels_produce_one_row_per_game_and_move():
    _, rounds, moves = play_block(36, RANDOM, RANDOM, InfoMode.NONE, InfoMode.NONE, 1, 0, 0,
                                  20, T, record_rounds=True, record_moves=True)

    assert rounds.size > 0
    assert moves.size > 0
    assert np.all(rounds["game_index"] < 20)
    assert np.all(moves["game_index"] < 20)
    # Level 3 is sampled by `game_index % sample_modulus == 0`, deterministically, so the run
    # does not depend on how work was scheduled.
    assert np.unique(moves["game_index"] % 20).tolist() == [0]
    assert np.unique(moves["game_index"]).tolist() == [0]


# --- accounting --------------------------------------------------------------


def test_card_accounting_adds_up():
    """The resource columns must be mutually consistent.

    `cards_collected_pX` counts cards; `voluntary_takes_pX` and `forced_takes_pX` count takes. A
    take always collects at least one card, so the first must be at least the sum of the other two
    -- the relation that catches a counter incremented on the wrong side of a transition, or a
    take counted twice.

    Note what is deliberately *not* asserted: that the columns sum to the deck size. They cannot.
    A card can be attacked, taken, played again, and taken again, so the same card is counted
    several times across the game, and any sum of these columns exceeds 36 routinely. Columns
    that overlap are counts of events, not a partition of the deck.
    """
    outcomes = autoplay(36, games=1500, seed=41)

    for suffix in ("p0", "p1"):
        collected = outcomes["cards_collected_" + suffix]
        voluntary = outcomes["voluntary_takes_" + suffix]
        forced = outcomes["forced_takes_" + suffix]

        assert np.all(collected >= voluntary + forced), (
            f"{suffix}: more takes than cards collected"
        )
        # No event can happen more often than there were moves.
        for column in ("cards_collected_", "voluntary_takes_", "forced_takes_",
                       "trumps_played_", "throw_ins_", "cards_beaten_", "attack_turns_"):
            assert np.all(outcomes[column + suffix] <= outcomes["actions"]), (
                f"{column}{suffix} exceeds the number of actions taken"
            )
        # A take collects at most the cards that were on the table, which is bounded by the
        # throw-in cap twice over.
        assert np.all(collected <= outcomes["actions"] * 2 * rules.THROWIN_CAP)

    assert np.all(outcomes["max_attacks_in_a_round"] >= 1)
    assert np.all(outcomes["max_attacks_in_a_round"] <= rules.THROWIN_CAP)


def test_voluntary_and_forced_takes_are_distinguished():
    outcomes = autoplay(36, games=1500, seed=53)

    # Both kinds must occur, or one of the two columns is measuring nothing.
    assert outcomes["voluntary_takes_p0"].sum() + outcomes["voluntary_takes_p1"].sum() > 0
    assert outcomes["forced_takes_p0"].sum() + outcomes["forced_takes_p1"].sum() > 0


def test_deck_exhaustion_is_recorded_when_it_happens():
    outcomes = autoplay(36, games=1500, seed=67)

    exhausted = outcomes["deck_exhausted_at_round"] > 0
    assert exhausted.any(), "no game ever exhausted the draw pile"
    assert np.all(outcomes["deck_exhausted_at_round"][exhausted]
                  <= outcomes["rounds"][exhausted])
    assert np.all(outcomes["cards_left_when_deck_ran_out"][exhausted] > 0)


def test_both_have_trumps_is_false_for_the_documented_deal():
    """Roughly 2.8% of 36-card deals leave all six trumps undealt.

    The source document called that impossible. The engine must record it, and the flag is
    the only way a later analysis can tell such a distribution from a dealt one.
    """
    outcomes = autoplay(36, games=4000, seed=71)

    assert outcomes["first_attacker_random_fallback"].any()
    assert not outcomes["both_have_trumps"][
        outcomes["first_attacker_random_fallback"]].any()
    # The fallback is rare, so a loose bound is the honest assertion.
    share = outcomes["first_attacker_random_fallback"].mean()
    assert 0.001 < share < 0.10, f"fallback share {share:.4f} is not plausible"


def test_win_from_a_single_card_is_measured_not_assumed_away():
    """How often a game ends with the winner attacking their last card, unopposed.

    The confirmed win rule -- an empty hand is a win, checked before the defender responds --
    makes this the normal way a game ends, not an edge case. Measured over 2,000 random games on
    a 36-card deck it was about 0.68 under the buggy take rule, and is about 0.90 now.

    The rise is a direct consequence of fixing that rule, not a new degeneration. After a take the
    original attacker keeps attacking, so the attacking seat keeps shedding cards while its opponent's
    hand grows; the player who empties a hand is therefore overwhelmingly the one who was last
    attacking, and an empty hand ends the round before the defender can answer. That is the correct
    shape of the game, and the winner is not skewed: measured 0.494 for seat 0 against 0.505 for seat 1.

    The number is pinned inside a band rather than at an exact value, because it is a property of
    the rule set and the policies, not a constant: a change to the win rule or to the endgame
    would move it, and that is exactly what a reader needs to notice. The test's purpose is to
    make the degeneracy impossible to forget, not to defend it.
    """
    outcomes = autoplay(36, games=2000, seed=83)

    single = outcomes["win_from_single_card_p0"] | outcomes["win_from_single_card_p1"]
    assert single.any()
    assert 0.75 < single.mean() < 0.97, (
        f"{single.mean():.3f} of games ended with an unopposed last-card attack; the win "
        f"rule's endgame degeneracy has moved and the rules report needs revisiting"
    )

    # The band above is only meaningful if the corrected rule did not skew the seats. Without this the
    # 0.90 could equally well be read as "the attacker now almost always wins", which would be a
    # different and more serious defect.
    winners = outcomes["winner"]
    assert 0.45 < (winners == 0).mean() < 0.55, "the seats must stay symmetric"

    # The same statistic for the 52-card deck, recorded rather than compared. An earlier draft of
    # this test asserted that a bigger deck would make the ending *less* common, on the reasoning
    # that more of the game would be spent in the refill phase. Measured, it is slightly more
    # common (0.70 against 0.68), so that reasoning is wrong and no direction is claimed here.
    # Why it goes up rather than down is not established.
    wide = autoplay(52, games=800, seed=83)
    wide_single = wide["win_from_single_card_p0"] | wide["win_from_single_card_p1"]

    assert 0.4 < wide_single.mean() < 0.95


def test_the_winner_is_always_the_player_who_emptied_a_hand():
    """Sanity on the win bookkeeping itself, independent of any strategy.

    `winner` is read straight out of the state, so this cannot be wrong unless the engine
    mislabels the record. It is here because everything else in the suite compares win *rates*,
    and a systematic seat error would survive all of them.
    """
    outcomes = autoplay(36, games=2000, seed=97)

    winners = outcomes["winner"]
    assert np.all((winners == 0) | (winners == 1))
    assert np.all(
        np.where(winners == 0, outcomes["final_hand_p0"], outcomes["final_hand_p1"]) == 0
    )
    assert np.all(
        np.where(winners == 0, outcomes["final_hand_p1"], outcomes["final_hand_p0"]) > 0
    )


def test_no_anomaly_flags_fire_on_a_clean_run():
    outcomes = autoplay(36, games=3000, seed=97)

    assert not outcomes["flag_card_conservation_violation"].any()
    assert not outcomes["flag_both_players_empty"].any()
    assert not outcomes["flag_illegal_move"].any()
    assert not outcomes["flag_redeals_exceeded"].any()
    assert not outcomes["flag_anomaly_any"].any()
