# Pre-registration of the interaction test

**Status: DRAFT.** It becomes registered when the maintainer creates the tag
`registered-interaction` on the merged commit holding this file, the analysis code and
all registered files below. The pilot's tag `registered` does not move.
No model has been called for this test. No held-out group's play has been analysed.

This file fixes what [the design note](../docs/interaction/design.md) left open.
Where the two differ, this file governs. Rules for analysis code not yet present are
requirements for registration, not descriptions of an implemented run.

## What is tested

- **H2, open loop.** Given the group's real earlier rounds, `zoon` predicts each
  held-out person's next contribution band better than `alone` and `prompted`.
- **H3, closed loop.** Groups simulated from the first round without human play in
  the loop score better against held-out groups, on the composite below, with `zoon`
  than with `alone` and `prompted`.

**H2 and H3 are judged separately.** Each has one verdict: "passed", "failed" or
"no verdict". For each hypothesis, both comparisons must pass its rule below. If
either comparison fails the rule, the hypothesis is failed. The run-ending cases
under Data give no verdict for both hypotheses. The `chat` arm decides nothing.

**What a pass means.** H2 shows better next-move prediction by this law using the
others' history than by the chosen law without it and this prompt to this model.
H3 shows better scores for these summaries of these games, over their first 20
rounds in groups of two and four. Neither shows that people influence each other.
Shared tendencies can also make the others' history predict.

A pass is relative. It does not show an adequate picture of human groups, nor how
the summaries vary together. It tests a fitted statistical law, with no model
inside it, against one prompt to one model. It says nothing about coded persons or
prompted agents in general. The participants were students at one university;
they read Chinese instructions and the models read an English rendering.

**What a fail means.** Failure against `alone` does not show that the others'
history improves prediction with this law. Nothing social is built on it until
something does. If H2 passes and H3 fails, one-step prediction succeeded and better
group simulation was not shown. Failure against `prompted` alone does not show the
coded law better than this prompt. A fail is a failure to show a difference, not
a demonstration of no difference. Every result is published.

The intervals are separate, not joint. A pass shows an estimate at least as large
as the threshold with a lower interval end above zero. It does not show that the
true gain exceeds the threshold.

## Data

The first session, first 20 rounds, of the eight linear public goods games in
[task-selection.md](../docs/interaction/task-selection.md). The source is Wang,
Hilbe and Zhang, under CC BY 4.0, pinned in `interaction/data-manifest.json`.
Second sessions, the threshold game and questionnaire answers are not read.

**Split.** `src/zoon/interaction/split.py` fixes salt
`zoon-interaction-split-v1`. Within each game it ranks whole groups by the salted
hash of their raw identifier and holds out the first two fifths, rounded up.
Groups are never divided. The split table in task-selection.md gives 210 training
groups and 143 held-out groups. The latter comprise 111 pairs and 32 groups of
four, hence 7,000 decisions, as counted in the design note's Cost section.
All pinned held-out groups are analysed. Nothing is fitted on them.

The published averages cover training and held-out groups together. As
task-selection.md records, they constrain the held-out level. No fit, choice or
threshold uses them. Training access is through `zoon.interaction.data.load()`;
counts are through `zoon.interaction.data.held_out_counts()`.

**What ends the run, and what only interrupts it.** Both hypotheses receive
"no verdict" if the decision model is withdrawn or reports another version; the
spending cap is reached; or the held-out groups are not exactly the 143 pinned by
the split, or any one is unreadable. A fault of the chat model, its withdrawal or
its version change makes `chat` unusable, with no scores and no change of verdict.
Everything else interrupts a step and is resumed by repeating its command.
Recorded answers are kept. Only missing answers are requested.

A change that is not a permitted repair ends confirmatory use, as in
[the pilot registration](h1.md#order-of-work). What follows is exploratory and
decides neither hypothesis. This is an operator rule; the guard cannot judge intent.

## Primary measures

### H2

For each held-out decision, score the natural logarithm of the probability the arm
gives the band the person chose, given the group's real earlier rounds. All 7,000
decisions count, including the first round. The difference is the mean over
decisions of `zoon` minus the other arm. A group of four carries twice as many
decisions as a pair.

**Probabilities.** Before the logarithm, every arm's five probabilities are floored
at `person.FLOOR = 1e-9` and divided by their sum. For `prompted`, this operation
is applied before recalibration and again after it. The latter is fixed here for
scoring; the current recalibration method itself applies only the former.

**Interval.** Fixed here: 10,000 resamples of held-out groups, with replacement
within each game, preserving its number of groups. Each comparison uses the same
resampled groups for both arms and recomputes the mean over sampled decisions.
Take percentiles 2.5 and 97.5 with the default linear interpolation of
`numpy.percentile`. Use `numpy.random.default_rng(20261120)` in a separate stream
labelled `H2-interval`. The label states its purpose. Games and groups are ordered
by game name and opaque group index. Use the same replicates for both comparisons.

**Pass rule.** Against `alone` and against `prompted` separately, the mean difference
must be at least 0.0063 nats per decision and the lower interval end must be above
zero. Exactly 0.0063 passes the first condition; exactly zero fails the second.
The threshold is fixed here: a quarter of the training estimate 0.0253 in
[candidates.md](../docs/interaction/candidates.md#coded-arm-gains), rounded to
four decimal places. It is a threshold of convenience, not a claim about what
size matters.

### H3

Each usable arm simulates 100 groups per game from round 1. There is no human play
inside the loop. For each held-out group and each summary below, take the fair
ranked probability score in `src/zoon/interaction/compare.py`, function `crps`,
against that arm's 100 simulated groups of the same game. Lower is closer.
For simulated values `x` and observed value `y`, it is
`mean(abs(x - y)) - sum over i < j of abs(x[i] - x[j]) / (n * (n - 1))`.
The fair score needs at least two simulated values.

All summaries use exact contribution shares, as in `compare.summaries`:

| Summary | One group's value |
|---|---|
| Level | Mean share over all seats and rounds |
| Members apart | Mean absolute share difference over pairs of members and rounds |
| Trend | Mean share in rounds 16 to 20 minus that in rounds 1 to 5 |
| Movement | Mean absolute round-to-round share change over seats and the 19 steps |
| Type gap | Mean share of type 0 minus type 1; undefined for `2-FE` and `4-FE`. Type 0 is seat 1 in pairs and seats 1 and 2 in groups of four |

The composite is `compare.combine`: for each summary, the mean score over held-out
groups where it is defined; then the unweighted mean of those five means. The
difference is the other arm's composite minus `zoon`'s. Positive favours `zoon`.
The point estimate uses all 100 simulated groups per arm and game.

**Interval.** Fixed here: 10,000 replicates. In each, draw held-out groups with
replacement within each game, preserving its count and pairing the arms on that
draw. Independently for each arm and game, draw WITHOUT replacement 50 of its 100
simulated groups. Use that same half-ensemble for all summaries and all sampled
human groups of that game. Recompute the scores, composite and difference.
Use the same `zoon` half-ensemble for its comparisons within a replicate.
Take percentiles 2.5 and 97.5, with the H2 interpolation rule. Use a separate
`numpy.random.default_rng(20261120)` stream labelled `H3-interval`, with games,
human groups and arms in name order and simulated groups in group-number order.

The fair score is unbiased in ensemble size. A half-ensemble carries simulation
error without bias and overstates it, which is the cautious side. Statistical
review of this interval is required before registration. This differs from
`compare.interval`, which resamples human groups only.

**Pass rule.** Against `alone` and against `prompted` separately, the composite
difference must be at least 0.0031 and the lower interval end must be above zero.
Exactly 0.0031 passes the first condition; exactly zero fails the second.
The threshold is fixed here: a quarter of the training estimate 0.0124 in
candidates.md, Coded-arm gains.

### Power

candidates.md, Power, gives half-effect power against the chosen twin of 0.782 for
H2 and 0.737 for the composite. These are normal approximations to detecting a
two-sided 95% interval excluding zero on either side. They scale training
intervals from 210 to 143 groups, treat scoring models as fixed, assume held-out
groups vary as training groups do, and take the true effect as half the estimated
training effect. They assume independent groups; lab sittings are not identified.
They are not power for the threshold condition, for both comparisons together,
or for the half-ensemble interval fixed here. The earlier measuring-stick power
in [measure.md](../docs/interaction/measure.md) concerns different models.
No training estimate exists against `prompted`, because no model has been called,
as [prompted.md](../docs/interaction/prompted.md) states. No power is stated for it.

## The coded arms

| Arm | Form | Fitted numbers |
|---|---|---:|
| `zoon` | Candidate B, own and others' traces | 20 |
| `alone` | B's twin, own trace only, channel closed | 18 |

The selection and every candidate's result remain in candidates.md. The chosen
twin and B's own twin are the same arm here.

`src/zoon/interaction/person.py` fixes the law. B has an intercept and coefficients
for the seat's endowment, return, their product and four seats, with no per-game
or per-seat constant. These numbers and the cut points are shared across games
and seats, separately for the first round and later rounds. Each phase has three
log gaps after `FIRST_CUT = 0`. The later phase adds weights on the own and
others' traces, centred at 0.5, and a fitted update fraction for each trace.
The twin omits the others' weight and fraction. A trace starts at its first
observation, not at zero, and then moves toward each observation by its fraction.
Round 1 has no trace features. For score `s` and cut `t[k]`, the probability of
band at most `k` is `sigmoid(t[k] - s)`. The cuts after the first are cumulative
sums of the exponentiated log gaps. Differences of cumulative probabilities give
the band probabilities. With the channel closed, a seat sees only its own
contribution and the rules, not payoffs or others' contributions.

**Fitting.** Before registration, fit B and its twin once on all 210 training
groups, in loader order, by `person.fit` with its default start. The objective is
summed negative log-likelihood plus `RIDGE = 0.01` times the sum of squares of all
fitted numbers. L-BFGS-B uses an analytic gradient and one start. Constants,
weights and log gaps start at zero; trace fractions at 0.5. Log gaps are bounded
by `LOG_GAP_LIMIT = 8` in either direction, fractions by zero and one, and other
numbers are unbounded. The settings are `MAX_ITERATIONS = 2000`,
`FUNCTION_TOLERANCE = 1e-10`, `TOLERANCE = 1e-7` and `maxls = 50`.
Optimiser success is required; failure raises, with no fallback.
These settings are also printed in candidates.md.

Commit the fitted numbers and the amount table in the files marked as added below.
**The registered run uses those committed numbers as they stand; it does not
refit the primary arms.** A test refits and requires every predicted band
probability on the training decisions to agree with the committed numbers' within
`1e-4`, fixed here. The reported game-left-out fits are separate diagnostics.

**Bands.** `src/zoon/interaction/measure.py`, function `band`, uses exact fractions:
zero is band 0; the whole endowment is band 4; a positive share at most one third
is band 1; above one third and at most two thirds is band 2; above two thirds and
below one is band 3. Nothing is rounded into a band.

**Band to points.** Build the table once from all training groups with
`kernel.Amounts.build`, in `src/zoon/interaction/kernel.py`. For each endowment and
middle band, keep all chosen amounts, sorted, with repetitions. Draw uniformly
from that list. Bands 0 and 4 give zero and the endowment exactly. Every arm that
returns bands uses the same committed table. `chat` returns points directly.

**Simulation streams.** Fixed here: seed 20261120; group label
`registered|<game>|<n>`, with n from 0 to 99. Use `kernel.uniform`: SHA-256 of the
compact JSON list `[seed, game, group, round, seat, purpose]`; the first eight bytes
as a big-endian integer, shifted right 11, divided by `2**53`. Round and seat are
zero-based in this key. Purposes are `band` and `amount`. All arms share these
kernel streams: the same seed, game, group number, round, seat and purpose pairs
the arms. The chat model's own sampling is not made deterministic by this key.
Each round collects every seat's choice before applying any of them. The ledger
uses exact fractions of a point. Logs replay byte for byte.

## The `prompted` arm

### What the model is sent

Model `jev-1.14.0`, one question per call with the five declared `ANSWERS` of
`src/zoon/interaction/prompted.py`, in this order:

1. "Nothing"
2. "Some, up to a third"
3. "More than a third, up to two thirds"
4. "More than two thirds but not all"
5. "All"

Each answer's description is the answer itself. The committed renderer fixes
every character. `prompted.instructions` substitutes the game's numbers into
`PAIR` or `FOUR`; the eight rendered instructions and their digests are registered.
`prompted.request` sends those instructions, then "The player you are asked about
is Player n.", then every earlier round. Each round gives "Round r" followed by
"Player n: contribution X points; payoff Y points." for every seat. Pairs also
show group total contribution, common pool and return per player, as
`prompted.round_text` renders them. With no history it says "Round 1: there is no
earlier history." No worked examples of human choices are supplied.

The question is "What share of their initial points will Player n contribute to
the common pool in round r?" The renderer substitutes the seat and round numbers.
The registered request fixtures use made-up histories, not participant records.

The decision model returns probabilities rounded to two decimal places.
As in the pilot, divide by their sum if it is within 0.05 of one; otherwise retry
the reply as a fault. Retain both returned and used probabilities. Then use the
floor and recalibration rules here. A reply still unreadable after retries
interrupts the step; it does not exclude a decision.

### From the model's answer to a probability

`prompted.subsample` fixes 512 training decisions (`SUBSAMPLE_COUNT`), salt
`SUBSAMPLE_SALT = "interaction-recalibration-1"`. Quotas are proportional to each
game's number of training decisions, rounded down, then allocated by largest
remainder, ties by game name. Within each game take the lowest SHA-256 digests of
compact JSON `[salt, game, group index, seat, round]`. Order the selected decisions
by digest. Seat is zero-based and round starts at one.

`prompted.fit` minimises the mean negative log-likelihood of
`softmax(beta * log(p) + shifts)` plus
`PENALTY = 1e-4` times `((beta - 1)**2 + sum(shifts**2))` over all five shifts.
There are four free shifts; the fifth is minus their sum.
`BETA_LOWER_BOUND = 0` constrains beta. The pre-recalibration floor is defined
under H2. `Recalibration.apply` applies the resulting map.

`FIT_METHOD = "trust-exact"`, `FIT_START = (1, 0, 0, 0, 0)` and
`FIT_OPTIONS = {"gtol": 1e-6, "maxiter": 1000}` fix the fit. It requires optimiser
success and polishes with up to `POLISH_STEPS = 20` Newton steps until the maximum
absolute gradient is at most `GRADIENT_LIMIT = 1e-10`. If the unconstrained beta
is negative, it solves on the boundary and checks the one-sided gradient, as the
committed procedure does. Failure raises and interrupts; there is no fallback.
These settings are printed in prompted.md.

In the closed loop, the simulated history is rendered by the same code. The
transcript answers each distinct request once, as in the pilot: its answer is
used for every identical request. Apply the recalibration fitted on the 512
training decisions and let the kernel draw the band and points. The model has
no memory beyond the supplied history. The coded arms fit all training groups;
`prompted` fits only this recalibration subsample. Supervision is unequal.

## Files that are part of this registration

Paths in backticks are relative to the repository root. Relative links are
relative to this document. Everything under `interaction/` is registered,
including files added there before registration. The data manifest is
`interaction/data-manifest.json`. The table describes today's state.

| File | State | Holds |
|---|---|---|
| `prereg/interaction.md` | Existing today | This registration |
| `interaction/` | Existing today | All interaction materials |
| `interaction/data-manifest.json` | Existing today | Source and training hashes |
| `interaction/instructions/` | Existing today | Eight rendered instructions |
| `interaction/instructions/sha256.json` | Existing today | Instruction digests |
| `interaction/requests/` | Existing today | Rendered request fixtures |
| `interaction/fitted.json` | Added by the analysis code | B and twin numbers, from the committed fitting procedure |
| `interaction/amounts.json` | Added by the analysis code | The training band-to-points table |

### Paths to be added by the analysis code

These paths do not exist today and are exempt from the existing-path check:

- `interaction/fitted.json`
- `interaction/amounts.json`
- `src/zoon/interaction/run.py`

The new run module supplies the steps below. The existing spending script
`src/zoon/interaction/budget.py` needs the chat bound. The analysis code also needs
the second sealing step, scoring and intervals, chat requests and replay,
transcripts, guards, state, report and tests. These are open work, not completed
features. A material-regeneration test must compare registered instructions,
fixtures and the amount table with their committed bytes; the fit test uses the
probability tolerance above.

## Models

| Arm | Model | Settings |
|---|---|---|
| `prompted`, which gates | `jev-1.14.0` | Declared-answer probabilities, as above |
| `chat`, reported | `gemini-4-carbon` on Vertex, location global | Thinking level low; temperature left at the service default; at most 2,048 output tokens |

Record the version each response reports. The decision model must report exactly
its name above. For chat, fix the version to the first response's reported version,
which must begin with its model name, as in the pilot. Withdrawal means the
service says the model does not exist on every bounded attempt at a request.
The outcomes for withdrawal and version change are under Data.

### The chat arm

It plays one seat at a time. Its request is the participant instructions, the
line "You are Player n.", the history as rendered for `prompted`, and then:

> How many of your E points do you contribute to the common pool in round r? Answer with one whole number of points and nothing else.

Substitute that seat's endowment for E, seat for n and round for r. Fixed here:
separate these parts by one blank line, with no system instruction. Use the same
empty-history line as `prompted` when there is no earlier round.

After trimming white space, the reply must be a whole number from zero to E.
Fixed here: accept only a non-empty run of ASCII digits, with no sign, decimal
point or other text; leading zeroes are allowed. Retry an unusable reply, up to
three attempts in all. Three unusable replies for a decision make the arm
unusable, with no scores. Faults of this model have the same arm-only outcome
stated under Data.

Every seat of every round of every simulated group is its own call. Identical
requests are NOT reused across decisions. The transcript identifies the game,
group, round, seat and attempt, and a resumed decision uses its recorded replies.
The answers are not deterministic; replay uses the transcript, not fresh calls.
The accepted points bypass recalibration and the band-to-points step.
Run closed loop only, with the H3 simulation count. Score against `zoon` by the
same H3 composite, difference and interval. It changes no verdict.

## Spending

One cap covers the whole run, both models and recall included. **Cap in US
dollars: CAP.** Count spending from reported tokens in recorded responses at the
fixed prices. Retain and count unusable chat replies before retrying them.
As in the pilot, do not send a call if spent tokens' cost plus its estimate and
reservations for calls in flight would exceed the cap. A cap stop gives no
verdict for both hypotheses, even if chat calls reached it.

**Decision-model bound.** prompted.md, spending table: 52,024 calls, at most
96,878,135 input tokens, at most 4.06888167 US dollars. This covers 7,000 open-loop
calls, 512 recalibration calls, 512 disguise calls and 44,000 closed-loop calls.
The bound counts requests without reuse, so reuse can reduce it. Its token
estimate is characters divided by three, rounded up, plus 400 per decision call,
with the longest possible earlier-round rendering repeated. Its printed output
token column assumes one per call; it is not a measured output-token bound.
Output costs zero at the fixed decision-model price.

The prices in prompted.md and `src/zoon/pilot/prices.py` are 0.042 US dollars per
million input tokens and zero per million output tokens, read on 2026-11-15.
**Chat-model bound in US dollars: CHAT BOUND.** The committed spending script
must produce it before registration, including retries and the recall requests.
Read the published [Vertex pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing)
on the day of registration and record the prices and read date with the bound.
Its listed price changes on 2026-12-01; the bound uses the higher price.

Registration goes ahead only when both placeholders are filled from the script's
output and the combined bound is beneath the cap. The current script's proposed
ceiling and pilot cap do not fix this run's cap. No price lookup or model call is
needed to write this draft.

## Order of work

Merge the registration, analysis code and registered materials, then create the
tag as specified above. These commands name a run module to be added; they are
not runnable today. Run them in this order:

1. `python -m zoon.interaction.run calibrate`: make the recalibration calls and the
   disguise calls; fit and save recalibration. Do not refit the primary coded arms.
2. `python -m zoon.interaction.run closed-loop`: simulate all usable arms. Write
   every game's log and digest. Finish all simulation and record all its digests
   before any held-out group is read.
3. `python -m zoon.interaction.run recall`: ask the chat model the recall questions.
4. `python -m zoon.interaction.run held-out`: the first step to read held-out groups.
   Only this step may call the second sealing step that produces them. Make the
   open-loop calls and score H2 and H3. An unusable chat arm does not block it.
5. `python -m zoon.interaction.run report`: produce the result with every transcript
   digest. Inspect and commit the result; the command does not commit it.

Held-out play is not loaded, summarised or rendered before `held-out`. The
second seal verifies the pinned split and completeness before scoring.

**Guard, repairs and persistence.** The guard, repair rules, append-only
transcripts, atomic bound state files and one-command lock are those of
[h1.md](h1.md#order-of-work), with the operational details in
[running.md](../docs/pilot/running.md). Only these differ: the interaction tag and
registered file set above; the step names here; the repair boundary at the start
of `calibrate`; and chat's per-decision transcript identity instead of reuse by
request. Run state and transcripts belong to this registration and cannot reuse
the pilot's. The held-out prerequisite is completed simulation with recorded
digests, not a judge gate. The report lists repairs and differences as in the
pilot. A permitted code repair cannot change the committed primary fits.

## Reported beside the verdicts

None of the following changes either verdict:

- Each summary's difference and interval, with the H3 resampling rule.
- Coverage for each arm and summary: the share of defined held-out values inside
  the central 90% of that arm's simulated values of the same game, inclusive of
  the fifth and 95th percentiles, as in `src/zoon/interaction/compare.py`.
- Every comparison with the candidate's own twin. It is `alone` here, so these
  repeat the gating twin comparisons; they are not independent evidence.
- A game left out: fit each coded form on training groups of the other seven
  games, then score held-out groups of the omitted game, in rotation. Compare
  their H2 scores with the primary committed fits. These diagnostic fits do not
  replace the primary arms and use no held-out answers for fitting.
- When closeness arises: members apart in the first round and later blocks,
  against the same held-out people re-formed into groups at random, by the method
  in measure.md. This is a description and excludes nothing.
- The two null-interaction checks on the committed numbers: with the others'
  weight set to zero, open-channel play equals closed-channel play under the same
  draws; with the channel closed, changing others' play changes no seat's
  distribution. `alone` is a separately fitted comparator, not this null test.
- Replay of every recorded closed-loop game from its log and transcript, with
  digests and byte equality. Chat replay makes no new calls.
- The chat comparison, or its unusable status; fitted-number counts; calls,
  reported tokens and spending by model; registration and step commits; repairs.

### The recognition check

The decision model cannot write text, so it cannot be asked. The control for it
is the disguise run below. Ask the chat model once for each of the eight games,
using `prompted.recall`: the committed instructions followed by:

> Do you recognise this experiment? If so, name the study and give its average contributions, in points, for each player type. Otherwise answer unknown.

Use its settings under Models. Retain the reply as returned, including a reply
cut off at the token limit. Print every reply in full. List an empty or blocked
reply as no reply. Recognition is attempted even if the arm has no scores.

Lower case and split into runs of letters and digits, as the pilot's check does.
Flag a reply containing the run "wang", "hilbe" or "zhang", or the adjacent runs
"asymmetric public goods". A flag labels the chat arm's comparisons "possibly
recalled" and changes no verdict. It is only a string match. Neither a flag nor
its absence establishes whether behaviour was recalled.

### The disguise run

Ask the same recalibration decisions again, with `prompted.DISGUISE = 10`:
every amount of points multiplied by ten and the exchange rate divided by ten.
Apply the same fitted recalibration to both sets of answers, with the probability
rules under H2. Report the mean change, disguised minus original, in log
probability of the observed band and the mean total variation distance,
`0.5 * sum(abs(p_disguised - p_original))`. Nothing is fitted again. It gates
nothing and is not run closed loop.

## Where this differs from the design note

- Recall goes to the chat model, not the decision model. Registered string keys
  replace judging whether an answer names the study or gives its averages.
- The chat model, settings, point request, reply rule and per-decision calls are
  fixed here. It bypasses the band step.
- H3 includes simulation error through independently drawn half-ensembles, rather
  than treating simulated groups as fixed.
- All simulation finishes and its digests are recorded before any held-out read.
- The primary coded fits and amount table are committed before registration. The
  registered run does not refit them; a refit test compares predicted probabilities.
- B and its own twin are fixed; the own-twin comparison duplicates `alone` here.
- H2 weights decisions, stratifies group resampling by game and floors again
  after recalibration. Thresholds, endpoint rules, simulation labels and seeds
  are fixed here. Coverage is the central 90%, not an unspecified range.
- Disguise uses the recalibration subsample and fitted map, with the exchange
  rate adjusted. Game-left-out scoring uses the omitted game's held-out groups.
- Run-ending cases and the new tag are fixed here. The spending placeholders
  remain open; the earlier proposed ceiling is not adopted.

## Open before registration

- The two placeholders, **CHAT BOUND** and **CAP**, filled from the spending
  script's output with dated chat prices.
- The analysis code and its files, including the paths marked as added above,
  the second seal and run machinery, and the required checks.
- Statistical review of the H3 interval.

## What is not tested

Influence between people; reading and interaction together; beliefs about beliefs;
punishment, communication or changing ties; decision timing, needs or decaying
memory. No adequacy claim for behaviour beyond these summaries, no generalisation
beyond these games, no authors' model fitted to all groups, and no classic decline
of cooperation. These sessions have an uncertain end.

## Deviations

None.
