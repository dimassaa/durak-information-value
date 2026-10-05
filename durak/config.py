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
DEFAULT_EVAL_WEIGHTS: Tuple[float, ...] = (1.0, 2.0, 0.5, 0.3, 1.5)

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