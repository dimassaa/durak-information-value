"""Experiment settings, read from ``DURAK_``-prefixed environment variables.

Every default here is a pre-registered research parameter. They are not
conveniences: changing a default changes what the dataset means, so they live in
one frozen dataclass, are pinned by ``tests/test_config.py``, and are stamped
into the run manifest.

``Settings`` is frozen but is still a Python object, and no Python object crosses
into Numba's ``nopython`` mode. Callers on the hot path unpack a ``Settings``
into plain scalars at the boundary and pass those; ``Settings`` exists for the
Python-side orchestration (runner, manifest, analysis) only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Mapping, Tuple

from durak.evaluation import SELECTED_WEIGHTS

# The tournament seed is bounded so every per-game seed derived from it stays
# inside int64. An unbounded seed would make derived seeds overflow silently
# rather than fail at configuration time.
MAX_SEED = 2**31 - 1

# Hand-justified leaf-evaluation weights, in the order the evaluation function
# consumes them: card-count differential, trump differential, high-card
# differential (rank >= 10), attacker-role term, deck-exhausted term.
#
# These are deliberately NOT tuned against opponents. Tuning them would confound
# algorithm quality with tuning budget and would risk giving the informed and
# uninformed arms different budgets, which would corrupt the primary metric. The
# grid in analysis/sweeps.py perturbs these to show how much the result moves.
#: The project's frozen instrument, not the original incumbent. See `evaluation.SELECTED_WEIGHTS` for
#: how it was chosen and why selection on the delta was forbidden. One definition, imported: two
#: literals that must agree is a pair of literals that will disagree.
DEFAULT_EVAL_WEIGHTS: Tuple[float, ...] = SELECTED_WEIGHTS

_ENV_PREFIX = "DURAK_"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the experiment parameters."""

    deck_size: int = 36
    games_per_pairing: int = 10_000
    seed: int = 0
    # 0 means "decide at runtime from cpu_count", mirroring the convention the
    # runner adopts; it is not a request for zero workers.
    workers: int = 0
    # Recorded in the manifest because it is part of the reproducibility contract, NOT because it
    # configures anything. The particle count actually used is the module constant
    # `durak.strategies.search_policy.PARTICLE_COUNT`, which is baked into the compiled policy's
    # signature. `__post_init__` enforces that this field agrees with it, so the two can never drift
    # apart silently.
    #
    # This previously read as a live knob and was not one: `particle_sweep` varied it, the sweep wrote
    # a `particle_sweep.csv`, and the three `k` values produced byte-identical win rates -- because the
    # field reached nothing. Any claim that `delta` is stable across `k` is therefore UNMEASURED. Making
    # the field authoritative would mean threading an int through `choose_action`, which is an njit
    # if/elif chain called from a `prange` loop, and from there through `policy_search` and every other
    # policy signature -- a real change, tracked as an open question rather than assumed done.
    particle_count: int = 100
    sample_modulus: int = 20
    throwin_defense_ranks: bool = True
    soft_penalty_lambda: float = 0.0
    eval_weights: Tuple[float, ...] = field(default=DEFAULT_EVAL_WEIGHTS)

    def __post_init__(self) -> None:
        if self.deck_size not in (36, 52):
            raise ValueError(f"deck_size must be 36 or 52, got {self.deck_size}")
        if self.seed < 0 or self.seed > MAX_SEED:
            raise ValueError(
                f"seed must be in [0, 2**31 - 1] so derived seeds stay in "
                f"int64, got {self.seed}"
            )
        if self.particle_count < 1:
            raise ValueError(f"particle_count must be positive, got {self.particle_count}")
        # The recorded value must equal what the search actually uses. Without this check the two can
        # drift, and the failure is invisible: a manifest would record one particle count while the
        # engine ran another, which is exactly the kind of error the manifest exists to prevent.
        from durak.strategies.search_policy import PARTICLE_COUNT
        if self.particle_count != PARTICLE_COUNT:
            raise ValueError(
                f"particle_count={self.particle_count} but the search uses "
                f"{PARTICLE_COUNT}. This field is recorded for reproducibility and does not configure "
                f"the search; making it authoritative requires threading it through choose_action."
            )
        if self.sample_modulus < 1:
            raise ValueError(f"sample_modulus must be positive, got {self.sample_modulus}")
        if len(self.eval_weights) != 5:
            raise ValueError(
                f"eval_weights must hold exactly 5 terms, got {len(self.eval_weights)}"
            )


def _read_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _read_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


def _read_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes"):
        return True
    if lowered in ("0", "false", "no"):
        return False
    raise ValueError(f"{_ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """Build a ``Settings`` from an environment mapping.

    Separated from :func:`get_settings` so tests can pass a mapping directly
    instead of mutating the process environment.
    """
    source = os.environ if env is None else env
    return Settings(
        deck_size=_read_int(source, "DECK_SIZE", 36),
        games_per_pairing=_read_int(source, "GAMES_PER_PAIRING", 10_000),
        seed=_read_int(source, "SEED", 0),
        workers=_read_int(source, "WORKERS", 0),
        particle_count=_read_int(source, "PARTICLE_COUNT", 100),
        sample_modulus=_read_int(source, "SAMPLE_MODULUS", 20),
        throwin_defense_ranks=_read_bool(source, "THROWIN_DEFENSE_RANKS", True),
        soft_penalty_lambda=_read_float(source, "SOFT_PENALTY_LAMBDA", 0.0),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings from the environment, cached.

    Cached because the runner reads settings once per block and rebuilds them
    otherwise; callers that need a fresh read call ``get_settings.cache_clear()``.
    """
    return settings_from_env()