"""Durak simulation engine and strategy research harness.

The package answers one question with statistics: how much win rate is worth
perfect knowledge of the opponent's hand?

Layout:

* ``durak.cards``    -- card encoding and the beat relation
* ``durak.rng``      -- seeded PRNG, three independent streams per game
* ``durak.rules``    -- the normative rule set (the single source of truth)
* ``durak.opponent`` -- opponent-hand particle filter
* ``durak.search``   -- expectimax search over particles
* ``durak.engine``   -- the game loop
* ``durak.runner``   -- deterministic parallel tournament orchestration
* ``durak.versions`` -- ENGINE/RULES/SCHEMA stamps

Code inside the simulation path must stay within Numba's ``nopython`` subset: no
classes, no dicts, no Python objects crossing function boundaries, and no
transcendental math functions (their results are not guaranteed identical between
the interpreted and compiled execution modes, which would break the equivalence
test the project's correctness rests on).
"""

__all__ = ["config", "versions"]