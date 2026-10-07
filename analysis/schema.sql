-- DuckDB views over a Parquet dataset produced by `durak.runner.run_tournament`.
--
-- The point of this file is that a reader cannot accidentally mix two experiments. Every dataset row
-- carries `(engine_version, rules_version, schema_version)`, and the project's rules engine changed
-- under it: version 1 handed the turn to the player who had just taken the cards, which is not Durak and
-- made a quarter of all games run to the round cap. A dataset produced before that fix and one produced
-- after it are different experiments with different win rates, and pooling them produces a number that
-- describes neither.
--
-- So the views below read a SINGLE version triple, declared once at the top. A query that wants a
-- different version edits one line and says so, instead of discovering the mixture in a figure.
--
-- `dataset_versions` lists what is actually present, so the choice is made from evidence rather than
-- from what the reader remembers.
--
-- Usage:
--     SET VARIABLE dataset_root = '/path/to/run';
--     <paste this file, or: .read analysis/schema.sql>
-- Every version filter goes through the three macros above, so the version triple is stated once and
-- cannot drift between the four views.

-- The version triple every view is filtered to. Change here, not in the queries.
--
-- These MUST equal `durak.versions`. A stale value does not fail loudly: every view returns zero rows,
-- a report shows empty figures, and the cause is three lines away from the number that went stale.
-- The drift that bit this file once -- schema version 2 added columns while the filter still said 1 --
-- is why the macros exist as named constants rather than as literals sprinkled through the queries.
CREATE OR REPLACE MACRO current_engine_version() AS 2;
CREATE OR REPLACE MACRO current_rules_version() AS 2;
CREATE OR REPLACE MACRO current_schema_version() AS 2;

-- Where the dataset lives: `SET VARIABLE dataset_root = '/path/to/run';` before reading this file.
-- A macro over the variable rather than a literal, so pointing the views at another run is one command
-- and cannot leave a stale absolute path behind in a committed file.
CREATE OR REPLACE MACRO dataset_root() AS getvariable('dataset_root');

-- Every version triple present in the dataset, with row counts.
--
-- This is the first query anyone should run: it is the answer to "what can I compare?", and a dataset
-- holding two rules versions shows two rows here instead of silently averaging them in a chart.
CREATE OR REPLACE VIEW dataset_versions AS
SELECT
    engine_version,
    rules_version,
    schema_version,
    COUNT(*)                                  AS games,
    COUNT(DISTINCT strategy_p0)               AS strategies_p0,
    COUNT(DISTINCT strategy_p1)               AS strategies_p1,
    MIN(deck_size)                            AS deck_sizes
FROM read_parquet(dataset_root() || '/games/**/*.parquet', hive_partitioning = true)
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- Level 1: one row per game. The filter is the reason this file exists.
CREATE OR REPLACE VIEW games AS
SELECT *
FROM read_parquet(dataset_root() || '/games/**/*.parquet', hive_partitioning = true)
WHERE engine_version = current_engine_version()
  AND rules_version  = current_rules_version()
  AND schema_version = current_schema_version();

-- Level 2: one row per round.
--
-- Levels 2 and 3 carry NO version columns -- only level 1 is stamped -- so their version filter is
-- inherited through a join to the already-filtered `games` view rather than repeated here. That is the
-- safer direction: a filter written twice can be edited in one place and not the other, and a stale copy
-- is how pre-fix rows reappear in a phase chart. Joining also means the two levels cannot disagree about
-- which dataset they belong to.
--
-- The join is on `(run_id, game_index)`, the game's own key, never on a row ordinal.
CREATE OR REPLACE VIEW rounds AS
SELECT r.*
FROM read_parquet(dataset_root() || '/rounds/**/*.parquet', hive_partitioning = true) AS r
WHERE (r.run_id, r.game_index) IN (SELECT run_id, game_index FROM games);

-- Level 3: one row per move, only for the sampled games. The sampling modulus is recorded on level-1
-- rows, so the sampling rate of a dataset is readable from `games` rather than assumed here.
CREATE OR REPLACE VIEW moves AS
SELECT m.*
FROM read_parquet(dataset_root() || '/moves/**/*.parquet', hive_partitioning = true) AS m
WHERE (m.run_id, m.game_index) IN (SELECT run_id, game_index FROM games);

-- Win rate per strategy, with games left out of the denominator spelled out.
--
-- A capped game has `winner = -1` and no outcome. Counting it as a loss would invent a result the rules
-- did not produce, and the decision not to count it as a win makes the rate depend on how many games
-- happened to stall -- which is a property of the engine, not of the strategy. Undecided games are
-- therefore excluded from the numerator and named in `undecided_games` beside the rate.
CREATE OR REPLACE VIEW win_rates AS
WITH participation AS (
    -- Each game has two participants. Seating them as rows is what lets one aggregation serve both
    -- seats; a per-strategy rate computed over only `strategy_p0` would halve every denominator.
    SELECT strategy_p0 AS strategy_id, winner = 0 AS won
    FROM games
    WHERE winner >= 0
    UNION ALL
    SELECT strategy_p1 AS strategy_id, winner = 1 AS won
    FROM games
    WHERE winner >= 0
)
SELECT
    strategy_id,
    COUNT(*)                     AS trials,
    COUNT(*) FILTER (WHERE won) AS wins,
    COUNT(*) FILTER (WHERE NOT won) AS losses,
    ROUND(COUNT(*) FILTER (WHERE won)::DOUBLE / NULLIF(COUNT(*), 0), 6) AS win_rate
FROM participation
GROUP BY strategy_id
ORDER BY strategy_id;

-- Per-pairing head to head, which is where the primary comparison lives: the same two strategies in
-- both seat orders, so the first-attacker advantage is visible instead of averaged away.
CREATE OR REPLACE VIEW head_to_head AS
SELECT
    deck_size,
    first_attacker_regime,
    strategy_p0,
    strategy_p1,
    COUNT(*)                              AS games,
    COUNT(*) FILTER (WHERE winner = 0)    AS p0_wins,
    COUNT(*) FILTER (WHERE winner = 1)    AS p1_wins,
    COUNT(*) FILTER (WHERE winner < 0)    AS undecided
FROM games
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4;

-- Round-level view, for the phase analysis: how hands and the draw pile evolve within a game.
CREATE OR REPLACE VIEW round_phases AS
SELECT
    g.strategy_p0,
    g.strategy_p1,
    g.deck_size,
    g.first_attacker_regime,
    r.round_idx,
    COUNT(*)                       AS rounds,
    AVG(r.hand_p0)                 AS mean_hand_p0,
    AVG(r.hand_p1)                 AS mean_hand_p1,
    AVG(r.deck_remaining)          AS mean_deck_remaining
FROM rounds AS r
JOIN games AS g
  ON g.run_id = r.run_id AND g.game_index = r.game_index
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 2, 3, 4, 5;