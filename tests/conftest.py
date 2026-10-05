"""Shared test fixtures.

Two failure classes are kept out of the suite's verdicts here: environment
leakage between tests (settings are cached, so an override would otherwise
persist into every later test), and nondeterminism (no test may depend on an
unseeded source of randomness).

The RNG fixtures are intentionally absent until Task 3 creates ``durak.rng``.
Adding them earlier would couple the suite to a module that does not exist yet.
"""

import pytest


@pytest.fixture
def fixed_seed() -> int:
    """A seed that is stable across runs and processes.

    Tests that would otherwise draw from a process-global RNG use this, so a
    failure reproduces exactly.
    """
    return 20261005


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Reset the cached settings around every test.

    Without this, a test that sets ``DURAK_*`` in the environment leaks its
    override into every later test in the session, and the failure surfaces far
    from its cause.
    """
    from durak.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()