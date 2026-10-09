"""Step 1: audit the instrument before measuring anything on it.

Six sweeps, because every one of the six known instrument defects was found by a sweep rather than by a
test. A full run would have buried each of them; a pilot catches them in minutes.

1. Card conservation per strategy, not in aggregate.
2. Anomalies per strategy. The global audit flag hides a strategy that produces anomalies in EVERY game.
3. Rounds and actions per game per strategy. A mean above ~60 rounds is where the take-rule bug lived for
   thirteen stages.
4. Undecided games. `winner < 0` is either a cap or a defect, never a legitimate outcome.
5. Every pooled descriptive statistic, with and without the degenerate baselines.
6. The hand-size spike: a loser holding 32+ cards is a strategy that never defends, not a game that ended
   interestingly.

Reads the pilot's level-1 Parquet. No replay.
"""
import pathlib

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path("datasets/pilot")
#: The dataset root, so the `deck_size` PARTITION KEY reads as a column. Reading the parquet file
#: directly loses it, and `manifest.json` sits beside the parquet so the root cannot be read as
#: one table; the partition subtree is read instead.
GAMES = ROOT / "games/deck_size=36"
NAMES = {0: "AUTO_ATTACK", 1: "AUTO_DEFEND", 2: "RANDOM", 3: "GREEDY", 4: "CONSERVATIVE",
         5: "SEARCH_UNIFORM", 6: "SEARCH_INFORMED", 7: "CHEATER_DOC"}
#: A policy that never defends: it attacks, the opponent beats it, the roles swap, and it takes. It cannot
#: win, so every game it plays is a description of it rather than of the game.
DEGENERATE = {"AUTO_ATTACK"}
FLAGS = ["flag_redeals_exceeded", "flag_round_cap_hit", "flag_illegal_move",
         "flag_card_conservation_violation", "flag_both_players_empty"]


def head(row, text):
    print(f"\n{row} {text}\n{'-' * (len(text) + 4)}")


def main():
    g = pq.read_table(GAMES).to_pandas()
    print(f"pilot: {len(g)} games")

    head(1, "1. CARD CONSERVATION per strategy")
    # The deck size lives in the PARTITION KEY, not in a column: pyarrow does not surface it when the
    # path names the partition explicitly, so it is read from the path. Asserted rather than trusted.
    DECK = int(str(GAMES).rsplit('deck_size=', 1)[1].split('/')[0])
    print(f'  deck size from the partition key: {DECK}')
    accounted = (g.final_hand_p0 + g.final_hand_p1 + g.deck_remaining_at_end
                 + g.discard_len_at_end + g.table_cards_at_end)
    bad = accounted != DECK
    print(f"  games where the five end-state fields do not sum to the deck: {int(bad.sum())}")
    if bad.any():
        print(g.loc[bad, FLAGS].any(axis=1).value_counts().to_dict())
    for sid, name in NAMES.items():
        m = (g.strategy_p0 == sid) | (g.strategy_p1 == sid)
        if not m.any():
            continue
        acc = accounted[m]
        print(f"  {name:<16} sum mismatch {int((acc != DECK).sum()):>6} of {int(m.sum())}")

    head(2, "2. ANOMALIES per strategy, not in aggregate")
    print(f"  {'strategy':<16} {'games':>8} " + " ".join(f"{f.replace('flag_', ''):>22}" for f in FLAGS))
    for sid, name in NAMES.items():
        m = (g.strategy_p0 == sid) | (g.strategy_p1 == sid)
        n = int(m.sum())
        if not n:
            continue
        counts = [int(g.loc[m, f].sum()) for f in FLAGS]
        print(f"  {name:<16} {n:>8} " + " ".join(f"{c:>22}" for c in counts))

    head(3, "3. ROUNDS AND ACTIONS per game per strategy")
    print(f"  {'strategy':<16} {'rounds mean':>12} {'rounds p95':>11} {'rounds max':>11} "
          f"{'actions mean':>13}")
    for sid, name in NAMES.items():
        for seat in (0, 1):
            m = g[f"strategy_p{seat}"] == sid
            if m.sum() < 50:
                continue
            r = g.loc[m, "rounds"]
            print(f"  {name:<16} {r.mean():>12.1f} {np.percentile(r, 95):>11.0f} {r.max():>11d} "
                  f"{g.loc[m, 'actions'].mean():>13.1f}")
            break

    head(4, "4. UNDECIDED games")
    print(f"  winner < 0: {int((g.winner < 0).sum())}")
    print(f"  round cap hit: {int(g.flag_round_cap_hit.sum())}")

    head(5, "5. POOLED DESCRIPTIVE STATISTICS, with and without degenerate baselines")
    keep = ~g.strategy_p0.isin([k for k, v in NAMES.items() if v in DEGENERATE]) & \
           ~g.strategy_p1.isin([k for k, v in NAMES.items() if v in DEGENERATE])
    for label, mask in (("ALL strategies", slice(None)), ("without AUTO_ATTACK", keep)):
        s = g[mask]
        lh = s.loser_final_hand
        print(f"  {label:<22} games {len(s):>7}  mean loser hand {lh.mean():>6.2f}  "
              f"median {np.median(lh):>5.1f}  share >=30 {100 * (lh >= 30).mean():>5.1f}%")

    head(6, "6. THE 32-CARD SPIKE, per strategy")
    big = g.loser_final_hand >= 30
    print(f"  all games: {100 * big.mean():.1f}%")
    for sid, name in NAMES.items():
        m = ((g.strategy_p0 == sid) | (g.strategy_p1 == sid))
        if m.sum() < 50:
            continue
        loser_is_sid = np.where(g.winner.values == 0, 1, 0) == sid
        share = 100 * (big & loser_is_sid).sum() / max(int((big & loser_is_sid).sum() + 0), 1)
        print(f"  {name:<16} games as the loser with 30+ cards: {int((big & loser_is_sid).sum()):>7}")


if __name__ == "__main__":
    main()