# What happens next, in order

From the current moment to the point where only simulations, tournaments and conclusions remain, on a
program that is verified working. Ordered, gated, and covering every outcome — because the branches are
where this project's work has gone wrong before.

Each step names its **gate**: the check that must pass before the next step is worth starting. A step
without a gate is a step whose output can be silently wrong, which has happened here eleven times
(`EXPERIMENTS.md`).

---

## Step 0 — finish the depth curve, honestly

**Status.** Depth 1 done: delta +0.0130, informed vs `GREEDY` 0.0685, uniform vs `GREEDY` 0.0648.

**The finding that changes the plan.** Hiding the draw pile did not merely fix correctness — it made the
search roughly **twelve times stronger** against `GREEDY` (0.0057 → 0.0685). With the pile public the search
saw future draws and committed confidently to lines the opponent could disrupt; with it hidden the beliefs
are calibrated. Every earlier number measured a pathological policy.

**The cost problem, and what was decided.** Depth 1 costs about 2 minutes at 4,000 games and each further
ply multiplies by the branching factor, about 5. Equal game counts would make depths 1-4 roughly **five
hours**, which is not available, and depth 5 would be about a day.

**Depth 5 is dropped from the research entirely.** Not deferred, not "as compute allows" -- removed. The
question the curve answers is whether the effect turns around as the policy gets stronger, and depth 4 is
already far beyond the point where the policy beats even `AUTO_DEFEND`; a fifth ply would answer nothing the
fourth has not, at twenty-five times the cost.

**Scale actually used**, games per direction, falling with depth: **2000 / 1000 / 500 / 250**. This buys the
SIGN at every depth and gives the deep ones a wide interval, and that is the honest trade: a stated width is
useful, a five-hour wait is not. The interval achieved is printed per depth rather than assumed.

**Gate.** Every depth must print an interval, and the depth-4 interval must be reported as the weak point of
the curve rather than glossed. If depth 4's interval is too wide to separate the sign, say so and stop at
depth 3 -- a three-point curve with usable intervals beats a four-point curve whose last point is noise.

---

## Step 1 — anomaly sweep and backlog closure

Nothing new is measured until the instrument is audited. Six defects were found here before, all by a
number that looked wrong.

### 1a. Read every row of `IMPROVEMENTS.md` and sort into three piles

- **FIX NOW** — anything that makes a reported number wrong. Expected: the `queries.py` field renames (done),
  the null opponent contaminating descriptive statistics, `ROUND_DTYPE.n_attacks` reading 1 in every row,
  levels emitting one extra round record per game.
- **FIX BEFORE THE FINAL RUN** — anything that makes a measurement unavailable or misleading.
- **ACCEPT AND RECORD** — things that are true and are not defects.

### 1b. Sweep for anomalies nobody has hit yet

Six specific sweeps, because each of the six known defects was found by a sweep rather than by a test:

1. **Card conservation** at every level, not only in the full-suite invariant: does `hands + table + discard + deck` sum to 36 at every recorded end state, per strategy?
2. **Per-strategy anomalies** — the audit reports one global count. A degenerate strategy can produce anomalies that a global average hides. `AUTO_ATTACK` produced 100% degenerate games and the aggregate did not flag it.
3. **Rounds and actions per game** — any strategy with a mean above ~60 rounds is suspect; the take-rule bug lived there for thirteen stages.
4. **Undecided games** — any game with `winner < 0` is either a cap or a defect, never a legitimate outcome.
5. **Field names against their queries** — every aggregate in `analysis/queries.py` recomputed by hand on a small sample. The seat/attacker misnaming was found this way, twice.
6. **Descriptive statistics that pool a null opponent** — the 24.6% of games with 32+ cards in the loser's hand were entirely `AUTO_ATTACK`. Any pooled statistic must either exclude degenerate strategies or be labelled.

**Gate.** Zero anomalies in the audit at every strategy, and every pooled descriptive statistic either
excluding or labelling the degenerate baselines. If this gate fails, **stop** — the program is not working and
no measurement on it means anything.

---

## Step 2 — model work, so that the measurements can be made at all

Three changes, in this order, each with a regression gate.

### 2a. Memory horizon `M`

Currently the filter excludes the **entire** discard pile, so every model has perfect memory. `M` bounds how
many plies back played cards are remembered; older discarded cards return to the candidate set.

**Gate, and it is the whole safety margin:** with `M = ∞` the numbers must be **bit-identical** to today's.
A parameter that changes every belief in the project and cannot be shown to reduce to the old behaviour is
not shippable.

### 2b. Evaluation weights from configuration

`EVAL_WEIGHTS` is a module constant and `@njit` freezes globals at compile time, so a twelve-variant sweep
as it stands would measure the same policy twelve times — the trap that flattened the depth sweep. Weights
arrive per run, are recorded in the manifest, and appear next to every published number.

**Gate.** Two runs differing only in weights produce **different** actions, and two runs with the same
weights produce byte-identical datasets.

### 2c. `pairing_index` on levels 2 and 3

`game_index` is cell-local, so every level-2 aggregate averages 57 unrelated games and no round- or
move-level quantity can be attributed to a strategy. Stamping `pairing_index` is a schema version bump and
invalidates every existing dataset.

**Gate.** A single-pairing run and a 57-cell run of the same games produce identical level-2 rows for the
same game. Until this passes, **no phase analysis is run at all**.

**Sequencing.** 2c is last because it invalidates datasets. Doing it first means re-running everything twice.

---

## Step 3 — measurement, in four blocks

| block | runs | gate before starting |
|---|---|---|
| B1 | pilot at 10%, then the depth curve and the oracle | Step 2a and 2b green |
| B2 | weight sweep, memory horizon, seed robustness | B1's pilot clean |
| C | primary dataset 1.14M + sensitivity grid 228k | weights frozen on **strength**, never on the delta |
| D | trump stratification, deck 52, phase analysis, self-play, the rule variant | C published |

### The pilot is not optional

Every one of the six instrument defects was caught by a pilot; none would have been caught by a full run.
The pilot runs first in every block.

### Rules that bind every run

- Clear `__pycache__` before any run whose result will be believed.
- One fixed seed per experiment, and the **same** seed across arms being compared.
- Resolve the winner to the arm through the seat the arm occupied, per direction. A flat result across a
  paired comparison is the harness cancelling itself, not a null.
- Record depth, weights, memory horizon and rule reading in the manifest and print them next to every number.

---

## Step 4 — conclusions, and the branches

The outcome determines the report's shape, so all branches are written down now.

### If the curve turns around at depth 3–4

The effect is gated on competence. Then: depth is the headline axis, the oracle becomes decisive rather than
optional, and the report's central claim is that **information pays only for a player good enough to use
it** — which is a real finding and a more interesting one than a number.

### If the curve keeps falling

The honest result is negative: perfect knowledge of the hand is worth approximately nothing to a policy that
reasons two plies ahead. Then before saying it: sweep the weights and the rule reading, because a null that
survives both is a finding and a null that disappears under either is a statement about the weights or the
rules instead.

### If the weight sweep finds a vector that beats `GREEDY` comfortably

**Every earlier number was measured with a broken instrument.** Restart from the pilot. This is the branch
that costs the most and it must not be discovered late.

### If no vector beats `GREEDY`

The policy class is unpublishable as "an informed player", and the report says so in its first paragraph.
The instrument is still worth reporting: six defects, each found by a measurement that refused to look wrong.

### If the memory horizon shows a bounded arm beating a perfect one

The framing is wrong and the report leads with that. A sharper prior beating a perfect one is the most
informative outcome available and it has happened once already in this engine.

### Under every branch, these may not be published

- The withdrawn `+0.0799`.
- "The price of information" as a single number. It is a function of depth, weights, rule reading, memory
  horizon and trump composition. The range is the finding.
- That the search arms are strong.
- Anything about particle count — it is fixed by a compiled constant and cannot be varied.
- A causal explanation of the trump curve.