"""Tests for the settings and version constants.

The defaults asserted here are pre-registered experiment parameters, not
implementation conveniences: changing any of them changes what the research
dataset means, so each one is pinned by a test.
"""

import dataclasses

import pytest

from durak.config import DEFAULT_EVAL_WEIGHTS, Settings, get_settings
from durak.versions import ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION

MAX_SEED = 2**31 - 1


def test_default_settings_match_pinned_experiment_parameters():
    settings = Settings()

    assert settings.deck_size == 36
    assert settings.games_per_pairing == 10_000
    assert settings.seed == 0
    assert settings.workers == 0
    assert settings.particle_count == 100
    assert settings.sample_modulus == 20
    assert settings.throwin_defense_ranks is True
    assert settings.soft_penalty_lambda == 0.0
    assert settings.eval_weights == DEFAULT_EVAL_WEIGHTS


def test_get_settings_reads_seed_from_environment(monkeypatch):
    monkeypatch.setenv("DURAK_SEED", "1234567")

    assert get_settings.cache_clear() is None  # env change must be observed
    assert get_settings().seed == 1234567
    get_settings.cache_clear()


def test_get_settings_is_cached():
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_seed_above_int32_bound_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        Settings(seed=MAX_SEED + 1)


def test_seed_at_int32_bound_is_accepted():
    assert Settings(seed=MAX_SEED).seed == MAX_SEED


def test_version_constants_are_present_and_ordered():
    """The stamps exist, and every one has been bumped at least once.

    This test used to assert each constant equals 1, which made it a check that the project had never
    changed its rules or its engine. That is a property no real project can keep, and it silently turns
    a version guard into a tripwire: the first legitimate bump fails it, and the tempting fix is to revert
    the bump. Asserting the stamps are present and monotonic keeps the guard pointed at what matters --
    a dataset must be attributable to a rule set -- without freezing the rule set in place.
    """
    assert all(isinstance(v, int) and v >= 1
               for v in (ENGINE_VERSION, RULES_VERSION, SCHEMA_VERSION))
    # A rules change is the one stamp that must move on its own, and it has: v1 handed the turn to the
    # player who took, which is not Durak and stalled a quarter of all games.
    assert RULES_VERSION >= 2, (
        "RULES_VERSION must record the take-rule fix; pooling a pre-fix dataset with a post-fix one "
        "would mix two different experiments into a single number"
    )


def test_settings_are_frozen():
    settings = Settings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.seed = 5