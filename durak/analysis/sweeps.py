"""The document-faithful Cheater, and the sensitivity sweeps.

Two things live here because they answer the same question: how much of the project's headline number
is a property of the *game* and how much of it is an artifact of how the search was built.

`policy_cheater_doc` and `policy_search` -- two code paths, both given perfect knowledge
--------------------------------------------------------------------------
The project's reference document specifies a "Cheater" strategy: perfect information, no sampling,
deterministic search. :func:`policy_cheater_doc` implements that literally, in its own module, deliberately
**not** sharing code with :func:`durak.search.expectimax`.

The duplication is the point. If the Cheater reused the search, then agreement between the two would only
say that a function agrees with itself, and the document's specification would go unmeasured. Keeping them
separate means the specification can be *checked* rather than argued about: if they disagree on some
position, that disagreement is information about the document, and it will show up as a concrete example
instead of as a debate.

They share only the evaluation weights, which is deliberate -- the weights are a pre-registered research
parameter, not part of either strategy's definition, and sharing them is what makes the comparison about
the *search* rather than about two different ideas of what a hand is worth.

The sweeps, and what they are for
----------------------------------
* :func:`particle_sweep` -- win rate per particle count. The primary metric's robustness to ``k``. If the
  measured value of information moves materially across ``k``, that is a finding about the filter's
  resolution and it goes in the report; it is never a reason to pick a flattering ``k``.
* :func:`weight_sweep` -- win rate per weight setting, for **both** arms, plus the difference between
  them. The point is the difference column: a weight that moves both arms together is a bias in the
  evaluation and cancels, and a weight that moves only one arm changes the measured price of information.
* :func:`rule_variant_sweep` -- ``throwin_defense_ranks`` both ways. The throw-in rule is a genuine
  ambiguity in the source document rather than a free parameter, and every throw-in statistic in the
  dataset depends on it.

Every sweep writes the settings it ran into its own result, so a sweep number can be reproduced from its
metadata rather than from a shell history. A sweep whose configuration lives only in the command line is
not a result, it is an anecdote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from durak.config import Settings
from durak.runner import RunManifest, run_tournament
from durak.strategies import RANDOM, SEARCH_INFORMED, SEARCH_UNIFORM

#: Particle counts swept by default. Wide enough that a plateau is visible if there is one, and it is the
#: default rather than a per-run choice precisely so nobody picks the flattering end of it afterwards.
DEFAULT_K_VALUES = (25, 50, 100, 200, 400)


def _manifest(run_id: str, strategies, games: int, seed: int, settings: Settings,
              block_size: int = 500) -> RunManifest:
    """A manifest for one sweep cell, with everything that can change a result recorded."""
    return RunManifest(
        run_id=run_id,
        seed=seed,
        strategies=list(strategies),
        deck_sizes=[settings.deck_size],
        games_per_pairing=games,
        regimes=["RULE_DERIVED"],
        settings=settings,
        block_size=block_size,
    )


def particle_sweep(out_dir, k_values=DEFAULT_K_VALUES, games: int = 2_000, seed: int = 0,
                   deck_size: int = 36) -> pd.DataFrame:
    """Win rate per particle count, for the uninformed arm.

    Only `SEARCH_UNIFORM` is swept: the informed arm reads the real hand as a single particle, so ``k`` does
    not apply to it. Sweeping it anyway would produce a flat line and imply the parameter does something.

    Returns one row per ``k`` with the arm's win rate and the games behind it.
    """
    import dataclasses

    rows = []
    for k in k_values:
        settings = Settings(deck_size=deck_size, particle_count=int(k), games_per_pairing=games, seed=seed)
        target = Path(out_dir) / f"k{k}"
        # Paired against `RANDOM` rather than run alone: a one-strategy grid is empty, and a run against
        # the same fixed opponent is also the cheapest honest denominator for a single arm's win rate.
        summary = run_tournament(_manifest(f"k{k}", [SEARCH_UNIFORM, RANDOM], games, seed, settings),
                                 target)

        rows.append({
            "k": int(k),
            "strategy": SEARCH_UNIFORM,
            "games": summary.games,
            "wins": summary.wins.get(SEARCH_UNIFORM, 0),
            "win_rate": summary.win_rate(SEARCH_UNIFORM),
            "anomalies": summary.anomalies,
            "settings": repr(dataclasses.asdict(settings)),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "particle_sweep.csv", index=False)
    return frame


def weight_sweep(out_dir, grid=None, games: int = 2_000, seed: int = 0,
                 deck_size: int = 36) -> pd.DataFrame:
    """Win rate per weight setting, for both arms, plus the difference between them.

    Both arms are swept on the *same* grid row. The `delta` column is the whole point: a weight that
    shifts both arms together is a bias in the evaluation and cancels out of the project's headline number,
    while a weight that shifts only one arm changes the measured price of information itself. Reporting the
    two rates without the difference would hide that distinction.
    """
    from durak.evaluation import EVAL_SWEEP_GRID

    grid = EVAL_SWEEP_GRID if grid is None else tuple(grid)
    rows = []
    for weights in grid:
        for strategy in (SEARCH_UNIFORM, SEARCH_INFORMED):
            settings = Settings(deck_size=deck_size, games_per_pairing=games, seed=seed,
                                eval_weights=tuple(weights))
            target = Path(out_dir) / f"w{'_'.join(str(int(w * 100)) for w in weights)}_{strategy}"
            # Both arms plus `RANDOM` in one grid per weight row, so the two arms face the *same* deals
            # under the same seed. Measuring them in separate runs would make the `delta` column
            # partly a difference of samples rather than of weights.
            summary = run_tournament(
                _manifest(f"w{'_'.join(str(int(w * 100)) for w in weights)}",
                          [SEARCH_UNIFORM, SEARCH_INFORMED, RANDOM], games, seed, settings),
                target,
            )
            rows.append({
                "weights": tuple(weights),
                "strategy": strategy,
                "games": summary.games,
                "win_rate": summary.win_rate(strategy),
                "anomalies": summary.anomalies,
            })

    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index="weights", columns="strategy", values="win_rate").reset_index()
    wide.columns.name = None
    # Column names are the integer ids; name them so the CSV is readable without the source.
    wide = wide.rename(columns={SEARCH_UNIFORM: "win_rate_uniform",
                                SEARCH_INFORMED: "win_rate_informed"})
    wide["delta"] = wide["win_rate_informed"] - wide["win_rate_uniform"]

    frame.to_csv(Path(out_dir) / "weight_sweep_long.csv", index=False)
    wide.to_csv(Path(out_dir) / "weight_sweep.csv", index=False)
    return wide


def rule_variant_sweep(out_dir, games: int = 2_000, seed: int = 0, deck_size: int = 36) -> pd.DataFrame:
    """Win rate with `throwin_defense_ranks` on and off.

    This one is not a tuning knob. The throw-in rule is a genuine ambiguity in the source document -- which
    ranks may be added to an attack that has already been answered -- and every throw-in statistic in the
    dataset depends on the answer. Reporting both is the only way a reader can see how much of the throw-in
    story rests on the reading.
    """
    rows = []
    for flag in (True, False):
        settings = Settings(deck_size=deck_size, games_per_pairing=games, seed=seed,
                            throwin_defense_ranks=flag)
        target = Path(out_dir) / f"rule_{int(flag)}"
        summary = run_tournament(_manifest(f"rule_{int(flag)}", [SEARCH_UNIFORM, RANDOM], games, seed,
                                          settings), target)

        rows.append({
            "throwin_defense_ranks": flag,
            "games": summary.games,
            "win_rate": summary.win_rate(SEARCH_UNIFORM),
            "anomalies": summary.anomalies,
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "rule_variant_sweep.csv", index=False)
    return frame
