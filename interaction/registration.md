# Pre-registration of the interaction test

**Status: DRAFT.** It becomes registered when the maintainer creates the tag
`registered-interaction` on the merged commit holding this file, the analysis code and
all registered files below. The pilot's tag `registered` does not move.
Format and model-selection calls have been made, as recorded under Models.
No participant record has been sent to a model and no held-out group has been read.

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
"no verdict". For each hypothesis, both comparisons must pass its rule below.
If both comparisons were computed and either does not meet the rule, the hypothesis
is failed; if a comparison could not be computed, there is no verdict. The
completion and run-ending rules under Data govern whether a hypothesis can be judged.
The `chat` arm decides nothing.

**What a pass means.** H2 shows better next-move prediction by this law using the
others' history than by the chosen law without it and this prompt to this model.
H3 shows better scores for these summaries of these games, over their first 20
rounds in groups of two and four. Neither shows that people influence each other.
Shared tendencies can also make the others' history predict. On training groups,
most of the composite gain against `alone` came from members apart
([candidates.md](../docs/interaction/candidates.md#coded-arm-gains)); each summary's
own result is reported.

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

**Split.** `src/zoon/interaction/split.py` fixes
`split.SALT = "zoon-interaction-split-v1"`. Within each game it ranks whole groups
by the salted hash of their raw identifier and holds out the first two fifths, rounded up.
Groups are never divided. The split table in task-selection.md gives 210 training
groups and 143 held-out groups. The latter comprise 111 pairs and 32 groups of
four, hence 7,000 decisions, as counted in the design note's Cost section.
All pinned held-out groups are analysed. Nothing is fitted on them.

The published averages cover training and held-out groups together. As
task-selection.md records, they constrain the held-out level. No fit, choice or
threshold uses them. Training access is through `zoon.interaction.data.load()`;
counts are through `zoon.interaction.data.held_out_counts()`.

**When a comparison can be judged.** A hypothesis is judged only when both of its
comparisons have been computed on all 143 held-out groups with every registered
decision-model request answered. If both comparisons were computed and either does
not meet the rule, the hypothesis is failed; if a comparison could not be computed,
there is no verdict. If `prompted.fit` raises on the 512 recorded
answers, or a step cannot be completed without a change that is not a permitted
repair, both hypotheses receive no verdict and the report says why.

**What ends the run, and what only interrupts it.** Both hypotheses receive
"no verdict" if the decision model is withdrawn or reports another version; the
decision model's spending cap is reached; or the held-out groups are not exactly
the 143 pinned by the split, or any one is unreadable. "Unreadable" means any
condition for which the sealing code in `src/zoon/interaction/data.py` raises
`RecordFault`: a contribution is not numeric (including text and booleans), is
not a finite whole number from zero to the seat's endowment, a seat does not have
20 rounds, or a four-player group does not have exactly four seats by 20 rounds.

For both models, a request has at most 4 attempts, with pauses of 1, 2 and 4
seconds between failed attempts, as in `src/zoon/pilot/clients.py`, function `retry`.
The analysis code must use the same counts and pauses.

A transport failure does not count toward the three attempts. An unusable
reply is a recorded response whose text fails the reply rule, or is empty or
blocked. Three unusable replies for one decision, the model's withdrawal, a
version change, reaching the chat cap, a failed replay of a chat game, or a first
reported version that does not begin with the model name, or an operator ending,
make `chat` unusable with no scores.
Chat game logs are written from the chat transcript when `closed-loop` completes,
if the arm is still usable then, and are not read back by any run step. A later
step that scores them first rebuilds them from the transcript, checked against
`recall.json` `payload.chat.transcript.digest`, and requires the rebuilt bytes to
match `closed-loop.json` `payload.chat_logs`. A failed rebuild makes `chat`
unusable; that step records REPLAY in its own state if recall records no earlier
cause, and the report prints both the recall record and the later step's record.

The operator may use `python -m zoon.interaction.run closed-loop --end-chat REASON`
or `python -m zoon.interaction.run recall --end-chat REASON`.
A blank reason is refused before any work.
Both flag reasons are refused before any work if they contain the project identifier,
and an ending reason is checked against the client's held bearer before recording.
A chat request failure is a transport interruption, a configuration refusal
(including an unavailable credential), or a secret refusal of a reply from the
metadata check or a generating request, leaving no recorded reply and no counted withdrawal;
the flag applies to all three classes.
For an unfinished closed-loop or recall step, the flag is accepted only after
that step saved an interruption by a chat request failure, or an ending of that
step is already saved; otherwise repeat the step without the flag.
It ends the arm only if a chat request fails in this attempt before any generating
request returns a reply; a returned reply uses up the flag before transcript append,
while replies reused from the transcript do not use it up.
Decision-model and storage failures, three unusable replies, the cap, a version
change and withdrawal never create an operator ending.

The run cannot check the operator's reason or distinguish a service failure from
an operator-side fault such as a broken network or a rejected or missing credential.
The ending records OPERATOR, the class, stage, last HTTP status, reason, step,
this step's count of recorded chat request failures including this one, and answered
chat decisions (and answered recall questions in recall); it removes chat scores
and changes no verdict.
The last status is the final HTTP status of the last attempt after redirects,
or null if that attempt received no final HTTP answer.
An operator ending completes this step; recall retains and judges its answers
and marks the rest "not asked".
On completion the command prints the chat status and ending records, and prints
"flag had no effect" once if this flagged attempt has no ending.
If a flagged attempt raises after a generating request returned in that attempt,
it prints "flag used up: a chat request succeeded in this attempt" before propagating the failure.
An ending already saved in this unfinished step is retained without new chat requests
and printed as "ending already recorded"; a completed or terminal step is still refused.
An ending during closed-loop does not stop recall from asking, so a service still
failing then needs its own interrupted attempt and its own flagged attempt.
Apart from a declared repair, the incomplete-line cut and restoring a guarded output
under Guard, repairs and persistence, this is the only operator decision the run
allows, and it cannot be taken once `held-out` has started.

Each step keeps at most one ending event, written once. Recall carries earlier
ending events forward unchanged and may add its own.
The earliest cause, however detected, governs the arm's reported cause; once a
completed step records a cause, later steps retain it. A withdrawal, an ending already in the current step, the cap,
a version change or an invalid first version prevents further recall questions.
Completed state files remain unchanged; the recall state exposes all ending events.
An arm usable at the end of `closed-loop` can become unusable during `recall`.
`Guard.recall_chat_record()` reads the status at the end of recall from `recall.json`
fields `payload.chat.status` and `payload.chat.cause` and refuses unless recall is
completed; an arm unusable there has no scores. A later failed rebuild is recorded
in that later step's state as described above. The arm is unusable if completed
recall says so or any later step records a failed rebuild; the recall record alone
is not final once a later step has recorded one. If the run ends with no verdict
before recall completes, the report prints the chat arm as "not run" and labels
any earlier chat record with its step; it does not call that record final.

If the run cannot record either model's withdrawal, a later attempt records a
withdrawal only if that model is still withdrawn. A failed ending save is carried
by the handler's one normal save; if that also fails, the last successful state
stands. REPLAY is derived again on retry. A decision version change or cap is detected again
from the decision transcript. Unreadable or incomplete chat transcripts prevent
the handler's save; the next attempt refuses with the cut-back instruction for an
incomplete line.

A reply whose append fails before any byte is written is not recorded, is not
counted toward the cap, and is requested again; actual spending can exceed
recorded spending by such replies. No bound is claimed on
paid but unrecorded spending across repeated storage failures.

If `held-out` has not completed within 30 days of its first start, both
hypotheses receive no verdict and the report publishes every score already
computed. Before `held-out` starts, an interruption has no time limit: no
held-out group has been read.

Recall questions with no recorded answer are sent unless the model is withdrawn,
its first reported version is invalid, its version has changed,
or the chat cap is reached. An arm made unusable by the three-reply rule, or ended
by the operator during `closed-loop`, is still asked. If the operator ends the arm
during `recall`, answered questions are kept and judged and the questions not yet
answered are recorded "not asked". `closed-loop` and `recall` complete with the
chat status recorded and "not asked" for each recall question not sent, and later
steps proceed. This changes no hypothesis verdict. The chat reply rule is under
The chat arm. A paid reply with an invalid version is retained as "not judged: version",
without judging or flagging it. Recall reports separately what stopped it asking.

A non-finite probability, score or summary where one is defined is a fault that
interrupts the step; it is never skipped or averaged around. The only undefined
summary is the type gap in `2-FE` and `4-FE`.

Everything else interrupts a step and is resumed by repeating its command.
Recorded answers are kept. Only missing answers are requested. A change that is
not a permitted repair ends confirmatory use; what follows is exploratory and
decides neither hypothesis. This is an operator rule; the guard cannot judge intent.

## Primary measures

### H2

For each held-out decision, score the natural logarithm of the probability the arm
gives the band the person chose, given the group's real earlier rounds. All 7,000
decisions count, including the first round. The difference is the mean over
decisions of `zoon` minus the other arm. A group of four carries twice as many
decisions as a pair.

**Probabilities.** Before the logarithm, every arm's five probabilities are
floored at `prompted.PROMPTED_FLOOR = 0.005` and divided by their sum. This is half
the decision model's reporting step: a probability it reports as 0.00 lies below
0.005. The floor is applied before the division, so a scored probability can fall
slightly below 0.005, and never below 0.005 / 1.02. The floor does not round
larger probabilities. In addition, `prompted`'s used probabilities, after dividing
the returned probabilities by their sum, are floored at 0.005 and divided by their
sum before recalibration, as the committed code does.

This common floor is a rule of the H2 score only. In the closed loop the coded
arms draw from their probabilities with `person.FLOOR = 1e-9`, and `prompted`
from its recalibrated probabilities, as the committed kernel does. For each arm,
report the number of held-out decisions in which the chosen band's probability
was below 0.005 before this scoring floor. Beside the verdict and changing nothing, report the
two H2 differences, with their intervals, rescored with `person.FLOOR` in place of
the scoring floor for the coded arms only; `prompted` is scored as above.

**Interval.** Games are in `sorted(GAMES)` order; within a game, held-out groups
are in order of their group index. Let `n_g` be the number of held-out groups in
game g and `R = 10_000`. A fresh generator is made for each of H2 and H3 as shown.

```python
rng = numpy.random.default_rng(20261120)
for g in sorted(GAMES):
    idx[g] = rng.integers(0, n_g, size=(R, n_g - 1))
```

For replicate r and a comparison, `m_g` is the mean, over the `n_g - 1` drawn
groups, of the group's mean per-decision difference (`zoon` minus the other
arm). The replicate's value is `sum over g of w_g * m_g`, with the fixed weight
`w_g = n_g * (decisions per group in g) / 7000`. The point estimate is the same
sum with all `n_g` groups, which is the mean over the 7,000 decisions. Both
comparisons use the same `idx`. The interval is
`numpy.percentile(values, [2.5, 97.5])`, with its default linear interpolation.

Drawing one fewer group than the game holds makes the resampled variance of a
game's mean unbiased; drawing `n_g` understates it by the factor
`(n_g - 1) / n_g`, which matters in the games with ten or eleven groups.

**Pass rule.** Against `alone` and against `prompted` separately, the mean difference
must be at least 0.0052 nats per decision and the lower interval end must be above
zero. Exactly 0.0052 passes the first condition; exactly zero fails the second.
The threshold is fixed here: a quarter of the training estimate of `zoon` minus
`alone` under this scoring floor, 0.0209 with interval 0.0133 to 0.0281, rounded
to four decimal places. The estimate uses this file's interval rule on the
out-of-fold training scores. It is not printed in
[candidates.md](../docs/interaction/candidates.md#coded-arm-gains), which scores
with `person.FLOOR` and gives 0.0253. It is a threshold of convenience, not a
claim about what size matters.

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

**Interval.** Use the game and held-out group order, `n_g`, and `R` defined
under H2. Make a fresh generator for H3:

```python
rng = numpy.random.default_rng(20261120)
for g in sorted(GAMES):
    human[g] = rng.integers(0, n_g, size=(R, n_g - 1))
    for r in range(R):
        half[g][r] = rng.permutation(100)[:50]
```

`half[g][r]` is a set of simulated group numbers. Every arm uses the same one in
that game and replicate, including `chat`. For replicate r, arm a, summary s,
game g, score each held-out group drawn by `human[g][r]` by the fair score
against arm a's simulated groups numbered `half[g][r]`. Let `m[a, s, g]` be the mean over the
drawn groups. The per-summary value is `sum over g of v[s, g] * m[a, s, g]`,
with fixed weight
`v[s, g] = n_g / (number of held-out groups where s is defined)`, omitting games
where s is undefined. The composite is the mean over the five summaries. For the
composite and for each summary, the difference is the other arm's value minus
`zoon`'s. The point estimate uses all `n_g` groups and all 100 simulated groups, which equals `compare.combine`.
Each summary's own reported interval uses the same replicates. Each interval is
`numpy.percentile(values, [2.5, 97.5])`, with its default linear interpolation.

The fair score is unbiased in ensemble size, so a half-ensemble has the same
expected score as the full one. Half of an ensemble drawn without replacement
has, for an average over simulated groups, the variance of a fresh ensemble of
the full size. The simulated groups of `zoon`, `alone` and `prompted` are paired
by the shared kernel streams, so the same half is used for every arm, which
keeps that pairing. Drawing halves independently per arm would discard it and
roughly double the simulation variance of a difference. This differs from
`compare.interval`, which resamples human groups only.

**Pass rule.** Against `alone` and against `prompted` separately, the composite
difference must be at least 0.0031 and the lower interval end must be above zero.
Exactly 0.0031 passes the first condition; exactly zero fails the second.
The threshold is fixed here: a quarter of the training estimate 0.0124 in
candidates.md, Coded-arm gains.

### Power

candidates.md, Power, gives half-effect power against the chosen twin of 0.782 for
H2 and 0.737 for the composite. Both quoted figures are for the score with
`person.FLOOR`. These are normal approximations to detecting a
two-sided 95% interval excluding zero on either side. They scale training
intervals from 210 to 143 groups, treat scoring models as fixed, assume held-out
groups vary as training groups do, and take the true effect as half the estimated
training effect. They assume independent groups; lab sittings are not identified.
They come from intervals that resample human groups only and, for H2, without
stratifying by game. They overstate the power of the H3 rule fixed here. The
registered H2 interval is about as wide as the note's, but the H2 figure
overstates too, because the estimate under the common floor is smaller: 0.0209
against 0.0253. At this number of groups
the interval condition is the binding one for the comparisons with `alone`.
Power and coverage under the rules fixed here are to be computed on training
groups by a committed script and quoted here before registration. The quoted
figures are not power for both comparisons together. The earlier measuring-stick power
in [measure.md](../docs/interaction/measure.md) concerns different models.
No training estimate exists against `prompted`: its calls so far used made-up
histories only, as recorded under Models and in
[prompted.md](../docs/interaction/prompted.md). No power is stated for it.

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
summed negative log-likelihood plus `person.RIDGE = 0.01` times the sum of squares
of all fitted numbers. L-BFGS-B uses an analytic gradient and one start. Constants,
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
`1e-3`, in the locked environment of `uv.lock`. This test is not part of the run
guard: its failure blocks registration, not a registered run. A refit with the
groups in another order moves a predicted probability by up to about 3e-4;
candidates.md reports the corresponding movement in the fitted numbers.
The reported game-left-out fits are separate diagnostics.

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
compact JSON list `[seed, game, group, round, seat, purpose]`, where `group` is
that string; the first eight bytes as a big-endian integer, shifted right 11, divided by `2**53`. Round and seat are
zero-based in this key. Purposes are `band` and `amount`. `kernel.play` turns a
uniform draw into a band and `Amounts.draw` turns one into points. The band-returning arms share these
kernel streams: the same seed, game, group number, round, seat and purpose pairs
them. The chat arm returns points directly; its sampling is not made deterministic
by this key. Its 100 simulated groups per game are numbered 0 to 99 with the
same group string `registered|<game>|<n>`, so `half[g][r]` names them too.
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

`prompted.subsample` fixes `SUBSAMPLE_COUNT = 512` training decisions, salt
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
committed procedure does. Failure raises, with no fallback; Data fixes the
no-verdict outcome when this fit raises on the recorded answers.
These settings are printed in prompted.md.
A saved recalibration is bound to a digest of its original 512 request keys;
resuming with different keys is refused without refitting.

In the closed loop, the simulated history is rendered by the same code. Identical
requests are deduplicated by the canonical request before dispatch; the transcript
answers each distinct request once, and its answer is used for every identical
request. Apply the recalibration fitted on the 512 training decisions and let the kernel draw the band and points. The model has
no memory beyond the supplied history. The coded arms fit all training groups;
`prompted` fits only this recalibration subsample. Supervision is unequal.

## Files that are part of this registration

Paths in backticks are relative to the repository root. Relative links are
relative to this document. Everything under `interaction/` is registered,
including files added there before registration. The data manifest is
`interaction/data-manifest.json`. The table describes today's state and also
lists the report's output path outside the registered inputs.

| File | State | Holds |
|---|---|---|
| `interaction/registration.md` | Existing today | This registration |
| `interaction/` | Existing today | All interaction materials |
| `interaction/data-manifest.json` | Existing today | Source and training hashes |
| `interaction/instructions/` | Existing today | Eight rendered instructions |
| `interaction/instructions/sha256.json` | Existing today | Instruction digests |
| `interaction/requests/` | Existing today | Rendered request fixtures |
| `interaction/fitted.json` | Existing today | B and twin numbers, from the committed fitting procedure |
| `interaction/amounts.json` | Existing today | The training band-to-points table |
| `src/zoon/interaction/score.py` | Existing today | Registered scores, intervals and verdicts |
| `src/zoon/interaction/registered.py` | Existing today | Committed material readers and regeneration |
| `src/zoon/interaction/power.py` | Existing today | Training power and coverage calculation |
| `docs/interaction/power.md` | Existing today | Power assumptions and one recorded run |
| `src/zoon/interaction/run.py` | Existing today | Calibration and step command dispatch |
| `src/zoon/interaction/steps.py` | Existing today | Closed-loop, recall and operator chat endings |
| `src/zoon/interaction/simulation.py` | Existing today | Committed-arm simulation and game log persistence |
| `src/zoon/interaction/guard.py` | Existing today | Run guards, repairs, state and lock |
| `src/zoon/interaction/decisions.py` | Existing today | Budgeted decision calls and append-only records |
| `src/zoon/interaction/chat.py` | Existing today | Chat requests, client, transcript, replay and recall |
| `docs/interaction/result.md` | Added by the analysis code | The report, outside the registered directory |
| `interaction/chat-requests/` | Existing today | Committed chat request fixtures |
| `src/zoon/interaction/service_answers.py` | Existing today | Service-answer table regeneration |
| `interaction/service-answers.md` | Existing today | Generated service-answer outcomes |

### Paths to be added by the analysis code

These paths do not exist today and are exempt from the existing-path check:

- `docs/interaction/result.md`

The run module dispatches calibration, closed-loop and recall today. The existing spending script
`src/zoon/interaction/budget.py` produces both bounds. The second sealing step,
scoring and intervals, run guards, step state, calibration, closed-loop and recall are implemented; held-out and report remain
open work. A material-regeneration test must compare registered instructions,
fixtures and the amount table with their committed bytes; the fit test uses the
probability tolerance above.

## Models

| Arm | Model | Settings |
|---|---|---|
| `prompted`, which gates | `jev-1.14.0` | Declared-answer probabilities, as above |
| `chat`, reported | `gemini-4-carbon` on Vertex, location global | Thinking level low; temperature 1.0; at most 2048 output tokens (`chat.MAX_OUTPUT_TOKENS = 2048`), including recall. A reply cut off at the limit follows the same reply rule |

Record the version each response reports. The decision model must report exactly
its name above. For chat, fix the version to the first response's reported version,
which must begin with its model name, as in the pilot. Withdrawal means the
service says the model does not exist on every bounded attempt at a request.
A reply whose text holds a secret is never recorded.
The generated [service-answer table](service-answers.md) records the outcome class,
request count and eligibility of the flag on the next attempt for each answer.
The outcomes for withdrawal and version change are under Data.

### Calls made before registration

The decision model was sent format-check requests twice, four calls each time,
all on made-up histories: on 2026-11-20, before the request renderer was written,
and on 2026-11-21 with the four committed format-check requests. All eight replies
were usable five-band distributions.

On 2026-11-21 four candidate chat models (`gemini-4-hydrogen`, `gemini-4-argon`,
`gemini-4-neon`, `gemini-4-carbon`) were tried, to learn whether they could be
used. Each was sent one toy question: a first attempt with a setting the service
refused, 4 refused calls, then 4 answered calls. Each was then sent two requests
built from the made-up histories, a first round and a twentieth round, three
times each at temperature 0: 24 answered calls. That is 28 answered calls. All
four answered the first round with the whole endowment every time, and their
twentieth-round answers varied between repeats.

`gemini-4-carbon` was chosen as the provider's most capable model among the two
that reported no thinking tokens in those probes. Its replies to the made-up
histories were seen before it was chosen, and the chat arm's settings and reply
rule in this file were written after them: the temperature, because the answers
varied even at temperature 0, and the thinking level, because the service refuses
to turn thinking off. The replies are not kept in the repository. No participant
record was sent to any model, and no held-out group has been read.

Later on 2026-11-21, after the chat renderer was committed, the chat format check
was run once with at most 256 output tokens, the limit first chosen: 12 calls, the
first and twentieth rounds of two games' committed made-up requests, three times
each. All 12 replies were usable whole numbers. Three reported thinking tokens,
from 45 to 111, and the others were printed as zero. The exact output is committed in
[`docs/interaction/format-check-chat.txt`](../docs/interaction/format-check-chat.txt).
On 2026-11-21 two further calls at that limit sent one toy question and one
committed made-up request to record which usage fields a reply carries; the record is
[`docs/interaction/reply-shape.txt`](../docs/interaction/reply-shape.txt).

The output limit was then raised, before registration and after seeing replies.
Thinking tokens count against the limit. A second run of the same 12 calls with at
most 256 output tokens returned 2 empty replies, each with 253 thinking tokens. The
same four requests were then sent 10 times each with at most 4096 output tokens:
all 40 replies were usable, and the most thinking tokens in one reply was 384. The
limit was set to 2048 and the 12 calls were run again at it: all 12 replies were
usable, and the most thinking tokens in one reply was 359. These 64 calls are recorded in
[`docs/interaction/output-limit.txt`](../docs/interaction/output-limit.txt).

### The chat arm

It plays one seat at a time. Its request is the participant instructions, the
line "You are Player n.", the history as rendered for `prompted`, and then:

> How many of your E points do you contribute to the common pool in round r? Answer with one whole number of points and nothing else.

Substitute that seat's endowment for E, seat for n and round for r.
The instructions already end in `\n`: append
one `\n`, then `You are Player n.`, then `\n\n`, then the history, then `\n\n`,
then the question, with no final newline and no system instruction. The history
joins rounds with `\n\n`; each round starts with `Round r\n`.
Use the same empty-history line as `prompted` when there is no earlier round.
Safety settings are the pilot's second-family client's: `HARM_CATEGORY_HARASSMENT`,
`HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT` and
`HARM_CATEGORY_DANGEROUS_CONTENT` each have threshold `OFF`. The reply is the
concatenated text parts of the first candidate, in their returned order.
A candidate with text and no finish reason is read as its text.

After trimming white space (space, tab, line feed and carriage return only), the reply must be a whole number from zero to E.
Fixed here: accept only a non-empty run of ASCII digits, with no sign, decimal
point or other text; leading zeroes are allowed. Retry unusable replies under
the three-reply rule in Data; transport failures and arm-only outcomes follow
that section.

Every seat of every round of every simulated group is its own call. Identical
requests are NOT reused across decisions. The transcript identifies the game,
group, round, seat and attempt, and a resumed decision uses its recorded replies.
The incomplete-line rule under Guard, repairs and persistence applies.
Replay requires the recorded whole-transcript SHA-256 and exact line count, and
uses one verified copy re-read from disk without fresh calls. Rendered decision and recall
requests must match; attempts for each decision must be consecutive from one,
with no attempt after an accepted reply and no more than three attempts.
A later round of a group requires accepted replies for every seat of its earlier rounds.
Changing text or token counts on any line fails replay. A game cannot be scored
when the arm is unusable; its whole transcript can still pass replay.
The arm's status is derived from the transcript and the cap, in file order: the
first invalid or changed version, third unusable reply for one decision, or
the chat cap reached under Spending. At the same entry, that is the order of precedence.
The transcript cannot show a withdrawal or an operator ending; the run
step records them, and that record governs later steps.
Each recall question is labelled "judged", "not judged: version", or "not asked".
The accepted points bypass recalibration and the band-to-points step.
Run closed loop only, with the H3 simulation count. Score against `zoon` by the
same H3 composite, difference and interval. It changes no verdict.

## Spending

Each model has its own cap, so exhausting the chat cap cannot end the run;
a configuration refusal can still prevent completion as described above. **Decision-model cap in US dollars: 17.** **Chat-model cap in US dollars,
recall included: 2544.** Count spending from reported tokens in recorded
responses at the fixed prices. Retain and count unusable chat replies before
retrying them. As in the pilot, do not send a call if that model's spent tokens'
cost plus the estimate and reservations for calls in flight would exceed its cap.
The decision-model cap is also reached when recorded spending is above it;
equality is allowed. The chat cap is reached when cumulative recorded spending
plus the largest estimate of any registered decision or recall request is above
its cap; equality is allowed. Apply this rule in file order, including before
the first entry with spending zero, for both decisions and recall. A chat call
that does not fit because of reservations waits for a call in flight to finish
and checks the status again.
Reaching the decision-model cap gives no verdict for both
hypotheses. Reaching the chat-model cap makes `chat` unusable and changes no
verdict.

**Decision-model bound.** prompted.md, spending table: 52,024 calls, at most
96,878,135 input tokens, at most 4.06888167 US dollars. This covers 7,000 open-loop
calls, 512 recalibration calls, 512 disguise calls and 44,000 closed-loop calls.
The bound counts requests without reuse, so reuse can reduce it. It counts one
attempt for each request. A request that is retried is paid for again, and a
request has at most 4 attempts, so one pass through every step costs at most four
times the bound; repeating an interrupted step can add to that, and the cap is
what stops it. Its token
estimate is characters divided by three, rounded up, plus 400 per decision call,
with the longest possible earlier-round rendering repeated. Its printed output
token column assumes one per call; it is not a measured output-token bound.
Output costs zero at the fixed decision-model price.

The prices in prompted.md and `src/zoon/pilot/prices.py` are 0.042 US dollars per
million input tokens and zero per million output tokens, read on 2026-11-15.
**Chat-model bound in US dollars: 2543.10329600.** The committed spending script
produces this for one pass, including the recall requests, with at most 2048
output tokens per call. Retries are paid from the same chat cap.
Read the published [Vertex pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing)
on the day of registration and record the prices and read date with the bound.
Prices for `gemini-4-carbon` were read on 2026-11-21 from the provider's
[pricing page](https://ai.google.dev/gemini-api/docs/pricing) and are to be confirmed
on the Vertex page on the day of registration. Its listed price changes on
2026-12-01; the bound uses the higher price. The lower price runs through
2026-11-30 (`chat_prices.LOW_THROUGH = "2026-11-30"`).

The decision-model cap is four times its bound, rounded up to a whole dollar
(`budget.DECISION_CAP_MULTIPLE = 4`).
The chat cap is its one-pass bound at the maximum output tokens, rounded up to a
whole dollar; retries are paid from this same cap, and reaching it only makes
chat unusable. The caps and chat bound above are filled from the script's output.
The current script's proposed ceiling and pilot cap do not fix this run's caps.

## Order of work

Merge the registration, analysis code and registered materials, then create the
tag as specified above. Calibration, closed-loop and recall are implemented today; held-out and report remain requirements.
Run them in this order:

1. `python -m zoon.interaction.run calibrate`: make the recalibration calls and the
   disguise calls; fit and save recalibration. Do not refit the primary coded arms.
2. `python -m zoon.interaction.run closed-loop`: simulate all usable arms. Write
   each coded and prompted arm's logs and guarded digests when that arm finishes.
   Write chat logs and their separate digests only on completion, after replay against
   the verified transcript. Finish all simulation before any held-out group is read.
3. `python -m zoon.interaction.run recall`: ask the chat model the recall questions.
4. `python -m zoon.interaction.run held-out`: the first step to read held-out groups.
   Only this step may call the second sealing step that produces them. Make the
   open-loop calls and score H2 and H3. An unusable chat arm does not block it.
5. `python -m zoon.interaction.run report`: check replay and write
   `docs/interaction/result.md` with every transcript digest. It is outside
   `interaction/`, whose registered bytes cannot change. The command may be run
   again and must reproduce the same file. Inspect and commit the result; the
   command does not commit it.

Held-out play is not loaded, summarised or rendered before `held-out`. The
second seal verifies the pinned split and completeness before scoring.

**Guard, repairs and persistence.** The repair boundary is the tag itself.
Input files this document cites are read as they stand at the tagged commit. From that
commit, code may change only to fix a fault that stops it running, leaving every
request, inclusion rule and numerical definition here unchanged; a permitted
repair cannot change the committed primary fits.

Before every step, including report and a resumed step, require the registration
tag to identify HEAD or an ancestor, a clean work tree including untracked files
not ignored, and disk bytes equal to HEAD for every path tracked at the tag or
HEAD. Ignored files under `src/` and `interaction/` are refused,
apart from the interpreter's bytecode cache (`.pyc` files inside `__pycache__`).
Registered files must equal their tagged bytes. Verify source and training
hashes against the manifest and regenerate registered materials for comparison.
Refuse continuation if a registered file differs. Other tracked differences
outside docs and tests are code. A code difference requires a committed repair,
a stated reason and an ordered binding of its full snapshot to the registered
commit, including current digests. Each changed snapshot needs a new declaration,
even a return to the registered code. Every declared repair commit must remain an
ancestor of HEAD. The guard cannot certify repair intent.

Transcripts are append-only except for the incomplete-line cut below, and bound
to the full registered commit and mode; check that identity before replay.
An incomplete final line, left by a write the system cut short, is not a record;
every opening refuses it and changes no bytes. The operator cuts the file back
to the byte after its last line feed, or to empty if it has none, changes nothing
else, and repeats the command, and the decision is asked again. If the cut bytes
held a reply, it was paid for and is not counted. The operator notes each cut and
its size in bytes in the message of the commit that adds the result. This is the
one change to a transcript the run allows; other damage is refused. State is separate for this registration and
cannot reuse the pilot's. Each step writes its output and digest atomically,
bound to its step, mode, registered commit, code or declared repair snapshot,
and exact earlier-step states, including absence. Only one command may hold
the run lock. Do not delete or reset state or transcripts during the run, except for that cut;
restoring a guarded output does not change state or transcripts.
A completed calling step is not repeated; `report` may be run again and must
reproduce the same file. An interrupted step resumes with recorded
answers and requests only those missing. A later calling step requires completed
prerequisites; report records terminal outcomes without requiring later calling
steps. The held-out prerequisite is completed simulation with recorded
digests. A guarded output that is missing, is a directory or has a non-directory parent,
cannot be read for lack of permission, or differs from its recorded digest refuses
the step. Other storage faults, including symlink loops, interrupt the step and
name only the recorded output. The operator may put back a copy of the file, and the
step proceeds only if its bytes match the recorded digest; this is not a decision
about any outcome. The report lists every repair's reason, commit and changed files, all
files differing from the tag, and the registration and step commits.

**Replay and null checks.** Replay is checked by `report`. Replay is
`kernel.replay` on the log's bytes. Every recorded closed-loop game of `zoon`,
`alone` and `prompted` must replay from its log to its recorded digest. For
`prompted`, the `returned` vector of every intent record must also equal, within
`1e-12` in every component, `Recalibration.apply` of the used probabilities
recorded in the transcript for that request. If either check fails, H3 receives
no verdict and H2 is unaffected. The chat arm's games replay from
the transcript and a failure makes the chat arm unusable.

The two null-interaction checks on the committed numbers are tests that must
pass before registration; the report repeats them and prints the result:
with the others' weight set to zero, the sequence of bands and points of every
seat and round of the 100 registered groups of every game is equal under an open
and a closed channel; with the channel closed, replacing the other seats' play
by any other play leaves each seat's five probabilities exactly equal.
`alone` is a separately fitted comparator, not this null test.

## Reported beside the verdicts

The report must contain:

- For each of the four gating comparisons: estimate, both interval ends,
  threshold, whether each condition is met, and the verdict.
- The recalibration's five numbers.
- For each arm, the count of held-out decisions in which the chosen band's
  probability was below 0.005 before the H2 floor, and the two H2 differences
  rescored with
  `person.FLOOR` for the coded arms; the rescoring changes no verdict.
- Every transcript and log digest.
- The statement that supervision of the arms is unequal.

None of the following changes either verdict:

- Each summary's difference and interval, with the H3 resampling rule.
- Coverage for each arm and summary: the share of defined held-out values inside
  the central 90% of that arm's simulated values of the same game, inclusive of
  the fifth and 95th percentiles, as in `src/zoon/interaction/compare.py`.
- Every comparison with the candidate's own twin. It is `alone` here, so these
  repeat the gating twin comparisons; they are not independent evidence.
- A game left out: for each omitted game, the mean per-decision difference
  `zoon` minus `alone` on that game's held-out decisions under the seven-game
  fits, beside each arm's own mean log score on those decisions under the
  seven-game fits and the difference under the committed fits, all with the H2
  scoring floor. These diagnostic fits use training groups only and do not
  replace the primary arms.
- When closeness arises: `measure.against_chance` on held-out groups in game and
  index order with its committed seed. This is a description and excludes nothing.
- The chat comparison, or its unusable status; fitted-number counts; calls,
  reported tokens and spending by model; registration and step commits; repairs.

### The recognition check

The decision model may have seen the paper or the data in training, it cannot
be asked, and no adjustment is made; the disguise run is the only control. Ask
the chat model once for each of the eight games, using `prompted.recall`: the committed instructions followed by:

> Do you recognise this experiment? If so, name the study and give its average contributions, in points, for each player type. Otherwise answer unknown.

Use its settings under Models. Retain the reply as returned, including a reply
cut off at the token limit. Print every reply in full. List an empty or blocked
reply as no reply. If the arm is unusable, Data governs questions not sent.

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

## Changes from the design note as it stood before this file

- Recall goes to the chat model, not the decision model. Registered string keys
  replace judging whether an answer names the study or gives its averages.
- The chat model, settings, point request, reply rule and per-decision calls are
  fixed here. It bypasses the band step.
- H3 includes simulation error through a shared half-ensemble per game and
  replicate, preserving the arms' pairing, rather than treating simulated groups as fixed.
- All simulation finishes and its digests are recorded before any held-out read.
- The primary coded fits and amount table are committed before registration. The
  registered run does not refit them; a refit test compares predicted probabilities
  within `1e-3` in the locked environment of `uv.lock`. It is outside the run guard
  and blocks registration on failure, not a registered run.
- B and its own twin are fixed; the own-twin comparison duplicates `alone` here.
- H2 weights decisions and stratifies group resampling by game. Both intervals
  draw one fewer human group per game. Every arm uses the same 0.005 floor for
  the H2 score; `prompted` also uses it on used probabilities before recalibration.
  Closed-loop draws keep the committed kernel's probability rules. H2's threshold
  is 0.0052 nats per decision under the common floor; H3's stays 0.0031. The report
  counts floor-raised chosen bands by arm and rescores H2 with `person.FLOOR` for
  the coded arms beside the verdict. Thresholds, endpoint rules, simulation labels and seeds
  are fixed here. Coverage is the central 90%, not an unspecified range.
- Disguise uses the recalibration subsample and fitted map, with the exchange
  rate adjusted. Game-left-out scoring uses the omitted game's held-out groups.
- Run-ending cases and the new tag are fixed here. The spending caps
  are filled from the committed bounds; the earlier proposed ceiling is not adopted.

## Open before registration

- **DONE:** the decision-model request format check; see
  [Calls made before registration](registration.md#calls-made-before-registration). This satisfies
  the design note's order of work.

- **DONE:** the caps and chat bound are filled from the spending script's
  output with dated chat prices. Vertex price confirmation remains due at registration.
- The held-out step and report, including the path marked as added above,
  and the required checks.
- **DONE:** the format check of the chat request with the committed renderer;
  see [Calls made before registration](registration.md#calls-made-before-registration).
- Replacement of the path test by tests that bind every constant and number in
  this file to the code.
- Power and coverage under the registered rules, computed on training groups by
  a committed script and quoted here. The script must reproduce the H2 training
  estimate 0.0209 under the common scoring floor before registration.

## What is not tested

Influence between people; reading and interaction together; beliefs about beliefs;
punishment, communication or changing ties; decision timing, needs or decaying
memory. No adequacy claim for behaviour beyond these summaries, no generalisation
beyond these games, no authors' model fitted to all groups, and no classic decline
of cooperation. These sessions have an uncertain end.

## Deviations

None.
