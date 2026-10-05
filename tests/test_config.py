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


def test_version_constants_start_at_one():
    assert ENGINE_VERSION == 1
    assert RULES_VERSION == 1
    assert SCHEMA_VERSION == 1


def test_settings_are_frozen():
    settings = Settings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.seed = 5