# The first interaction test: design

**Status: draft for review.** Nothing here is registered. The draft
[pre-registration](../../interaction/registration.md) fixes the run rules and governs where it
differs from this note. Its "Changes from the design note as it stood before this file"
section lists the changes.

The pilot tested one simulated person on a task where nobody interacts, and the verdict it
registered was a pass. This note designs the next test: people who play together in a group.
It says what is tested, against what, what a pass and a fail would mean, what the
draft registration fixes, what remains open before registration, and what changes in
`docs/design.md`.

## What is tested

Two hypotheses, on the held-out groups of the public goods games chosen in
`task-selection.md`.

- **H2, open loop.** Given what everyone in a group has done so far, a coded person whose
  state includes what the others did predicts that person's next contribution better than
  two alternatives.
- **H3, closed loop.** Groups of coded persons, played forward from the first round with no
  recorded group's play inside the loop, score better against human groups, on a fixed set of
  summaries, than two alternatives.

| Arm | What it is | What beating it shows |
|---|---|---|
| `zoon` | A coded person: a state that carries the person's own level and what the others did, and a coded rule from state to a distribution over contributions. Every number in it is fitted to training groups of these games | — |
| `alone` | The best coded person that is never told what the others did, chosen and fitted by the same procedure | The chosen person with the others' history predicts better, on these games, than the best candidate without it |
| `prompted` | The decision model asked directly: given the rules and the group's history as participants saw it, what will this player contribute next? | This fitted coded person predicts better than this prompt to this model |

**Each hypothesis passes if the `zoon` arm beats both alternatives** by at least a smallest
difference fixed in the pre-registration, with a 95% interval over held-out groups that
excludes zero. The two hypotheses are reported separately. There is one verdict for each.

A second prompted arm is reported beside H3 and decides nothing: a general-purpose chat
model playing each seat. The decision model was built to predict human choices, so it is the
stronger opponent and the one that gates.

## What is claimed, and what is not

- **Prediction, not influence.** A pass says that a person model which uses what the others
  did predicts held-out people, or scores better against held-out groups, than the
  alternatives. It does not say that people influence each other. `measure.md` shows why
  these data cannot separate that from members who share a tendency.
- **Better, not good.** Every comparison is relative. Beating two alternatives does not show
  that simulated groups are an adequate picture of human ones. How often the human value
  falls inside the central 90% of simulated values is reported for each summary, so that
  a reader can see the absolute fit.
- **These summaries, not behaviour as a whole.** H3 scores the summaries listed under
  "Measures", one at a time. It does not score how they vary together, or anything they
  leave out.
- **A fitted law, not an architecture.** In this test the coded person has no language
  model inside it. It is a small statistical model of this game, fitted to training groups
  of the same eight games. A pass over `prompted` is a pass over one prompt to one model,
  with the fitting and the supervision described below. It is not a result about prompted
  agents in general or about coded persons in general.
- **One lab.** The participants were students at one university, the instructions were in
  Chinese, and the models read an English rendering.

## What is not tested

- **No model inside the coded person.** The pilot tested a model's reading of a situation.
  This test leaves it out, as a choice about the design and not a finding: nothing has
  measured what a reading would add here. The reasons are that the measuring sticks in
  `measure.md` lost little on a game left out of the fit, and that any conclusion about
  rules never seen would generalise from eight games. So the only model calls are the
  prompted arms. What the coded arms lose on a game left out is computed and reported.
- **Not reading and interaction together.** That needs data with far more variety in the
  rules, and coded laws for punishment and reward. The data for it is identified in
  `task-selection.md`. It is the next stage, not this one.
- **No beliefs about beliefs, no punishment, no communication, no forming or cutting of
  ties.** The person responds to what the others did. It has no model of what they think.
- **Not the classic decline of cooperation.** These sessions have an uncertain end and the
  paper reports mostly stable contributions.
- **No decision timing.** The data has no response times.
- **Not the authors' own model.** The paper fits a model of players who react to each other
  and prefer equal shares. Its published weights were fitted to all groups, the held-out
  ones included, so it is not run as a comparison here.

## The person in this game

State, held in code. All quantities are shares of the endowment, computed from the exact
contributions, in both loops.

| Part | Law | Status |
|---|---|---|
| Own level | A trace of the person's own past contributions, moved toward each new contribution by a fixed fraction | The error-correcting update of the main design, applied to the person's own behaviour. The fraction is fitted |
| What the others did | A trace of the mean share the other members contributed, moved toward each round's observation by a fixed fraction | The same update. With the fraction at one it is simply the last round |
| First contribution | A rule of its own for the first round, when there is no history | `measure.md` found that one rule for all rounds fits the first round poorly |
| Response | An ordered logit over five bands of the share contributed: nothing, up to a third, up to two thirds, more but not all, everything. The chance of a band at or below each of four cut points is the logistic function of the cut point minus a score. The score is a constant for the game and seat type, plus weights on the traces | A convention fitted to training groups. That contributions rise with what others give is the published finding of conditional cooperation |

Why five bands: a model that returns two-decimal probabilities for its declared answers
cannot spread them over up to 37 amounts, and every arm must be scored on the same response.

### The candidates, and how one is chosen

The form above is a family. The candidates are fixed here, before any of them is fitted:

| | Candidate | What it adds or changes |
|---|---|---|
| A | The base form above | — |
| B | The game's four numbers in place of a constant per game and seat type | Can be applied to a game not in the fit |
| C | Two traces of the others where A has one: seats of the person's own type, and seats of the other type. Where a game has only one of the two kinds of other seat, which is every game but the two unequal games of four, the missing trace takes the value of the present one | In groups of four with unequal seats, who gave may matter, not only how much |
| D | The others' trace in points: the others' mean contribution in points, divided by the person's own endowment and capped at one | Equal amounts as the reference, where A has equal shares |
| E | The person's own first contribution, as a share, kept as a fixed part of the score in every later round. In the open loop it is the recorded one; in the closed loop it is the one drawn | A lasting difference between persons, where A has only a fading trace |
| F | No trace of the person's own level. Its twin has no state at all after the first round | Tests whether the own level is needed at all |

Each candidate has a twin without any trace of the others. The `zoon` arm is the best
candidate and the `alone` arm is the best twin, each chosen by the same rule, so the
ablation has had the same search. The selection in `candidates.md` chose B and B twin.
The winner's own twin is therefore `alone` here; its separately reported comparisons repeat the gating ones. The
registered primary fits and amount table are committed before registration and are not
refitted by the run. The refit test requires training probabilities to agree within
`1e-3` in the locked environment of `uv.lock`; it is outside the run guard and its
failure blocks registration, not a registered run. Reordering the groups moves a
predicted probability by up to about 3e-4; `candidates.md` reports the corresponding
movement in fitted numbers.

The rule, in full, applied once to the candidates and once to the twins:

1. Score each one by the mean log-likelihood of the band chosen, on training groups, each
   group scored by a fit that left its fold out, with the five folds of `measure.py`.
2. Take the one with the highest score; if several have exactly that score, the earliest
   letter among them. Any other is tied with it if the 95% interval of
   their paired difference, resampling groups as `measure.py` does, includes zero.
3. Among the highest and those tied with it, choose the one with the fewest fitted numbers;
   if several have the fewest, the highest score among them; if scores are equal, the
   earliest letter.

The closed-loop score is computed for every candidate and reported; it does not choose. So
the two arms are the best one-step predictors of their kind, and neither is claimed to be
the best simulator of its kind.

The comparison is run by a committed script whose output is committed with it: every
candidate's equation, fitted numbers, score and count of fitted numbers, the winner and
its twin. Nothing is fitted or chosen by hand.

**The published averages are not used.** The paper's averages cover all groups, so with
the training groups they give the held-out average of each game (`task-selection.md`).
No fit, choice or threshold may use them. The pre-registration says so, and the committed
script reads the training file and nothing else.

## The loop

A minimal kernel, built for this test and no larger:

- **Rounds, not milliseconds.** A round is one batch: every seat's intent is collected,
  applied together, and the outcome is delivered before the next round.
- **A ledger in exact units.** Endowments, contributions, the pool and each share are kept
  as exact fractions of a point. Every round balances.
- **From a band to points.** An agent returns a distribution over bands; the kernel draws a
  band and then an amount within it. Nothing and everything are exact. For the three bands
  between, the amount is drawn from the amounts that training participants with the same
  endowment chose within that band, as a committed table. The same step serves every
  band-returning arm; chat supplies points directly.
  So training participants' amounts shape every simulated group, through this table and
  through the fitted numbers. What stays outside the loop is any recorded group's play:
  no simulated round is told what a human group did.
- **A log.** Append-only, one record per intent, transfer and observation. Replaying a log
  under the same seed reproduces it byte for byte.
- **A channel.** It decides what each seat's decision law is told. With the channel open, a
  seat is told every seat's contribution and payoff, as participants were. With it closed, a
  seat is told its own contribution and the rules, and nothing else: not its payoff, since
  in a pair the payoff reveals the other's contribution exactly.
- **Sampling in code.** Draws use streams derived from the run's seed, the group, the round
  and the seat. Seats decide at the same time and do not see each other's draw. The same
  kernel runs the coded arms and the band-returning `prompted` arm; `chat` returns
  points directly.

## The prompted arms

The pre-registration fixes these rules, with the files committed:

- **What the model is given.** An English rendering of the participants' instructions with
  the game's numbers in place, the seat, and every earlier round as participants saw it:
  each member's contribution and payoff, in points. Only two of the eight instructions are
  printed in the paper; the others are rendered from them by substituting numbers. No
  worked examples of other participants' choices are included.
- **What it is asked.** For the band of the next contribution, as five declared answers.
  The probabilities are the model's distribution over its declared answers, read as the
  pilot read them.
- **Recalibration.** Five ordered bands need more than a temperature. The decision model's
  arm gets a temperature and one shift for each band, the shifts summing to zero, so five
  fitted numbers. They are fitted by maximum likelihood on a subsample of training decisions
  drawn by a committed rule, with a small penalty toward no recalibration, so that the fit
  has one answer whatever the model returns. The temperature enters as an inverse
  temperature that cannot be negative. So the arm cannot lose only because its probabilities
  are too extreme or lean toward a band. This depends on flooring its two-decimal
  probabilities at `prompted.PROMPTED_FLOOR = 0.005`, half the reporting step,
  and dividing by their sum before recalibration.
- **The chat model** is run in the closed loop only. It is asked for a number of points and
  its answer is taken as it stands, with no recalibration and no band step, since that is
  how such agents are run. The registration fixes its model as `gemini-4-carbon`, its
  settings, reply rules and per-decision calls. It is reported and decides nothing.
- **Unequal supervision, stated.** The coded arms fit every number on all training groups.
  The prompted arm is fitted only through its recalibration. That is the comparison being
  made, and the result says so, as the pilot's did.
- **In the closed loop** the simulated group's own history is rendered the same way, in the
  points the kernel drew. The same recalibration is applied before a band is drawn. The
  transcript reuses the answer to each identical decision-model request, as in the pilot.
  The model has no memory of its own between rounds: what it knows of a seat is what the
  rendered history shows.
- **Recall.** The chat model is asked the committed recall wording for each game; the
  decision model cannot write text. The registration fixes string keys that label chat
  comparisons "possibly recalled" and change no verdict. The decision-model control is
  disguise. Full replies are printed; a flag is not proof of recall.

## Measures

- **H2.** The mean log-likelihood of each held-out person's band in each round. Every arm's
  five probabilities are all floored at 0.005 and divided by their sum before the
  logarithm. This is half the decision model's reporting step: no arm is scored more
  finely than the coarsest arm reports. The used decision-model probabilities are
  also floored at 0.005 and divided by their sum before recalibration. This common
  scoring floor applies only to H2; closed-loop draws keep `person.FLOOR` for coded
  arms and the recalibrated probabilities for `prompted`, as the kernel does. The
  report counts decisions whose chosen-band probability the floor raised for each
  arm and gives the two differences rescored with `person.FLOOR` for the coded arms,
  changing no verdict. Differences are paired by decision, averaged over decisions
  and resampled by group within each game.
- **H3.** Each held-out human group is scored against the simulated groups of its game, one
  summary at a time, with the continuous ranked probability score in the form that does not
  depend on how many groups were simulated. The measure is the mean over these summaries,
  all in shares of the endowment and all from exact contributions:

  | Summary | What it is, for one group of 20 rounds |
  |---|---|
  | Level | The mean share over all seats and rounds |
  | Members apart | The mean, over rounds and over every pair of members, of the absolute difference between their shares in that round |
  | Trend | The mean share over all seats in rounds 16 to 20, minus the same in rounds 1 to 5 |
  | Movement | The mean, over seats and over the 19 steps from one round to the next, of the absolute change in the seat's share |
  | Type gap | The mean share of the seats of the first type minus that of the seats of the second type, over all rounds. The first type is seat 1 in a pair and seats 1 and 2 in a group of four. Not defined for the two games with equal seats |

  How they are combined. For each summary, each held-out group in a game where the summary
  is defined gives one difference between two arms' scores, and the summary's result is the
  mean of those differences. The measure is the mean of the five results. The type gap is
  defined in six games: every game but the two with equal seats. In one of the six the
  seats differ in productivity only. The interval resamples groups within each game, so
  every resample has the same games, drawing one fewer human group per game with fixed
  weights from the full held-out counts. The registered interval also samples half of each
  arm's simulated groups without replacement, using the same group numbers for every arm
  in a game and replicate to preserve pairing and include simulation error. Its point
  estimate uses the full ensemble. Equal weight for the five is a choice, and the two signed summaries can range more widely than the other three. Each summary is also
  reported alone.

  **Why this set, chosen now.** On training groups the gain from the others' history was
  mostly in how far apart members are, and the power for that summary alone was higher
  (`measure.md`). Gating on it alone would test one property and call it resemblance. The
  mean of five is the broader claim. Movement and the type gap were added in review,
  because the first three cannot tell a group that settles from one that jumps about, nor
  a group whose richer members give a smaller share from its reverse. `candidates.md` now
  reports training effects and power for all five and their mean; the registration quotes
  the chosen arms' power. The measure is not changed afterward, whatever the estimate.
- **Smallest differences that count.** H2 requires 0.0052 nats per decision, a quarter
  of the training estimate 0.0209 under the common scoring floor, with interval
  0.0133 to 0.0281 under the registered interval rule on out-of-fold training scores.
  This estimate is not printed in `candidates.md`, which uses `person.FLOOR` and
  gives 0.0253. H3's threshold stays 0.0031. The registration fixes the endpoint
  rules. These are thresholds of convenience, not claims about what size matters.

## Controls

- **Null interaction.** Two tests on the committed numbers must pass before registration
  and are repeated with their result in the report. With the others' weight set to zero,
  the sequence of bands and points of every seat and round of the 100 registered groups of
  every game is equal under an open and a closed channel. With the channel closed,
  replacing the other seats' play by any other play leaves each seat's five probabilities
  exactly equal. `alone` is a separately fitted comparator, not this null test.
- **Disguise.** Every amount of points is multiplied by ten and the exchange rate divided
  by ten. The coded laws see only shares of the endowment, so their output cannot change;
  a test asserts it. This is a property of
  the representation and not evidence about models. The prompted arm is run once on the
  disguised rules on the recalibration subsample, with the same fitted recalibration.
  The mean change in observed-band log probability and mean total variation are reported.
  It is not run in the closed loop, and no verdict depends on it.
- **When closeness arises.** How far apart the members of a group are, in the first round
  and after, against the same people re-formed into groups at random, is reported for the
  held-out groups as it was for the training groups. It is a description and excludes
  nothing.
- **Replay.** Every recorded closed-loop game of `zoon`, `alone` and `prompted`
  replays from its log to its recorded digest; failure gives H3 no verdict and leaves H2
  unaffected. Chat games replay from the transcript; failure makes `chat` unusable.
  All simulation finishes and its digests are recorded before any held-out group is read.
- **A game left out.** The coded arms are also fitted on the training groups of seven games
  and scored on held-out groups of the eighth, in rotation. Reported, not gating.

## What a pass and a fail would mean

- **H2 passes:** on these games, a coded law that uses the others' history predicts the
  next move better than the best one that does not, and better than this prompt to this
  model.
- **H3 passes:** on these games, simulated groups of coded persons score better against
  human groups, on the mean of the summaries above, than groups of persons who are told
  nothing of each other and than groups of this prompted model. It is a claim about those
  summaries of 20 rounds in groups of two and four.
- **H2 does not pass against `alone`:** this test did not show that the others' history
  improves prediction with this law. Nothing social is built on the law until something
  does.
- **H2 passes and H3 does not:** the person predicts one step ahead, and this test did not
  show that the groups it forms score better. One-step prediction and simulation are known
  to come apart, and that would be the result to report.
- **Either does not pass against `prompted` only:** this test did not show the coded person
  to be better than this prompt to this model. The comparison is published as it stands.

A comparison that does not pass is a failure to show a difference, not a demonstration that
there is none. Whatever happens is published, as with the pilot.

## What remains before registration

The draft pre-registration fixes the arms, fitting and committed numbers, band table,
measures, thresholds, intervals, streams, prompted and chat rules, controls, stopping rules
and order of work. Its "Open before registration" list is exhaustive: the decision-model
cap, chat bound and cap with dated prices; the analysis code and its files and required
checks; statistical review of the H3 interval; a format check of the chat request with the
committed renderer; replacement of the path test by tests binding every registered constant
and number to the code; and power and coverage under the registered rules, computed on
training groups by a committed script and quoted in the registration. That script must
reproduce the H2 training estimate 0.0209 under the common scoring floor. The
decision-model request format check is DONE; the registration's "Calls made before
this file" records it, satisfying the order below.

## What this changes in the main design

Made in the same reviewed change as the pre-registration. Each line names the section of
`docs/design.md` it touches.

| Section | Change |
|---|---|
| Status, and "Kernel: provisional until the pilot passes" | The pilot's registered verdict was a pass on H1-evaluative; H1-strict failed. This test starts the interaction work on that verdict. It uses no judge, so it does not depend on which of the two readings of the pilot is taken |
| Non-goals | "It says nothing about one-shot bargaining, cooperation or norms" becomes: the first interaction test covers contributions to a repeated public good. Reasoning about what another believes, punishment and ties stay out |
| The zoon, one step | In this world the appraisal step is absent: no model is called by the person. The model boundary itself is unchanged |
| The zoon, decision | The design reserved more than two options for a race of accumulators it had not anchored. This test uses an ordered response rule and makes no claim about timing |
| Kernel | Rounds as batches, where the design has milliseconds. Exact fractions of a point, where the design has integers in a smallest unit, because a share of the pool is not a whole number of points. The decision lifecycle is not needed, since seats decide once per round |
| Controls: recall | Kept, for the chat arm; the decision model cannot write text. Its consequence here is a label, as stated above |
| Controls: disguise | For the coded laws it holds by construction and is a unit test. For the prompted arm it is reported and does not gate. The open finding in the main design, about laws that mix payoffs with an appraisal, does not arise here and stays open |
| Controls: framing sensitivity | Deferred. Not run in this test |
| Controls: variation check | Replaced: the ranked probability score compares the spread of simulated groups with the spread of human groups, summary by summary, and coverage is reported |
| Controls: model swap and paraphrase | Model swap is the reported chat-model arm. Paraphrase of the question is deferred |
| Controls: null interaction | Restated as the absence of information, with the two tests above. "Social measures must be exactly null" does not apply to summaries of independent persons, which are not zero |
| Acceptance | A row for this test is added before the kernel milestone, and the kernel milestone is narrowed to what this test does not already build |

## Cost

- Open loop: one call for each held-out decision. The split gives 111 held-out pairs and 32
  held-out groups of four, each of 20 rounds, so 7,000 calls. More for the recalibration
  subsample and the disguise run.
- Closed loop: one call for each seat and round of each simulated group. With N simulated
  groups per game, five pair games and three games of four, that is 440 times N calls for
  each prompted arm, and the rendered history grows with the round.
- A measured bound on calls and spend is produced by a committed script before
  registration, as it was for the pilot. The spending cap is the maintainer's decision.

## Order of work

1. This note, reviewed.
2. The comparison of the candidates on training groups, and the power of the H3 measure.
3. Rendered instructions, a format check of the prompted requests on made-up histories, the
   table from bands to points, and the spending bound. The decision-model format
   check is DONE, as recorded in the registration's "Calls made before this file".
   The chat request check with the committed renderer remains open.
4. The pre-registration and the analysis code, reviewed, with the amendments to the main
   design.
5. Registration by tag; then the run; then the result, whatever it is.
