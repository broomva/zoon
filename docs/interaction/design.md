# The first interaction test: design

**Status: draft for review.** Nothing here is registered. The pre-registration will fix the
details and will govern where it differs from this note.

The pilot tested one simulated person on a task where nobody interacts, and the verdict it
registered was a pass. This note designs the next test: people who play together in a group. It
says what is tested, against what, what a pass and a fail would mean, and what it changes in
`docs/design.md`.

## What is tested

Two hypotheses, on the held-out groups of the public goods games chosen in
`task-selection.md`.

- **H2, open loop.** Given what everyone in a group has done so far, a coded person whose
  state includes what the others did predicts that person's next contribution better than
  two alternatives.
- **H3, closed loop.** Groups of coded persons, played forward from the first round with no
  human data inside the loop, are closer to human groups than two alternatives.

| Arm | What it is | What beating it shows |
|---|---|---|
| `zoon` | A coded person: a state that carries the person's own level and what the others last did, and a coded rule from state to a distribution over contributions | — |
| `alone` | The same person with the channel removed: it never learns what the others did. Its numbers are fitted afresh | Using the others' history improves prediction. In the closed loop this is the null-interaction control, compared with people |
| `prompted` | The decision model asked directly: given the rules and the group's history as participants saw it, what will this player contribute next? | The coded person does better than asking a model, which is how simulated societies are usually built |

**Each hypothesis passes if the `zoon` arm beats both alternatives** by at least a smallest
difference fixed in the pre-registration, with a 95% interval over held-out groups that
excludes zero. The two hypotheses are reported separately. There is one verdict for each.

A second prompted arm is reported beside H3 and decides nothing: a general-purpose chat
model playing each seat, sampled the way such agents are normally run. The decision model is
the stronger comparison, since it was built to predict human choices, so it is the one that
gates.

## What is claimed, and what is not

Both hypotheses are about prediction. A pass says that a person model which uses what the
others did predicts held-out people, or resembles held-out groups, better than the
alternatives. It does not say that people influence each other. `measure.md` shows why these
data cannot separate that from members who share a tendency: the only argument against a
shared tendency is that the experimenters matched people at random.

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

## The person in this game

State, held in code:

| Part | Law | Status |
|---|---|---|
| Own level | A trace of the person's own past contributions, as a share of the endowment, moved toward each new contribution by a fixed fraction | The error-correcting update of the main design, applied to the person's own behaviour. The fraction is fitted |
| What the others did | The mean share the other members contributed, moved toward each round's observation by a fixed fraction | The same update. With the fraction at one it is simply the last round |
| Response | An ordered rule from a score to five bands of the share contributed: nothing, up to a third, up to two thirds, more but not all, everything. The score is a constant for the game and seat, plus weights on the two traces | A convention fitted to training groups. That a contribution rises with what the others are expected to give is the published finding of conditional cooperation |

Decisions taken in the light of `measure.md`:

- **A persistent own level is proposed.** On training groups a person's own history was the
  stronger predictor in either order of entry. In the open loop the trace is fed the
  person's real contributions. In the closed loop there are none, so the first contribution
  is drawn from the rule and the trace then follows the simulated person's own play. The
  comparison of candidates before registration includes a person without it.
- **Five bands, not 37 amounts.** A model that returns two-decimal probabilities for its
  declared answers cannot spread them over 37 options, and every arm must be scored on the
  same response. The coded arms look back at bands too, so that the open and closed loops
  run the same law.
- **The first round is modelled on its own terms.** The measuring sticks shared cut points
  between the first round and later ones and fitted the first round poorly. The candidates
  include a separate rule for the first contribution.
- **The exact form is chosen on training groups before registration**, by a committed
  comparison of a small number of candidates: with and without each trace, and with the
  game's four numbers in place of a constant per game. The pre-registration names the winner
  and the count of candidates tried.

Each arm fits its own numbers, on training groups only. The pre-registration reports each
arm's count of fitted numbers.

## The loop

A minimal kernel, built for this test and no larger:

- **Rounds, not milliseconds.** A round is one batch: every seat's intent is collected,
  applied together, and the outcome is delivered to every seat before the next round. The
  millisecond clock of the main design is not needed and is not built.
- **A ledger in exact units.** Endowments, contributions, the pool and each share are kept as
  exact fractions of a point. Every round balances.
- **A log.** Append-only, one record per intent, transfer and observation. Replaying a log
  under the same seed reproduces it byte for byte.
- **A channel.** It decides what each seat observes. With the channel removed, a seat
  observes only its own outcome: that is the `alone` arm.
- **Sampling in code.** An agent returns a distribution over bands. The kernel draws from
  it with a stream derived from the run's seed, the group and the round. The same kernel
  runs coded and prompted agents.

## The prompted arms

- **What the model is given.** An English rendering of the participants' instructions with
  the game's numbers in place, the seat, and every earlier round as participants saw it:
  each member's contribution and payoff. Only two of the eight instructions are printed in
  the paper; the others are rendered from them by substituting numbers, and are committed.
- **What it is asked.** For the band of the next contribution, as five declared answers.
- **Fitting.** As in the pilot, the prompted arm gets a small recalibration fitted on
  training groups, so that it cannot lose only because its probabilities are too extreme.
- **In the closed loop** the simulated group's own history is rendered the same way, and the
  kernel draws each seat's band from the returned distribution.
- **Recall.** The model is asked whether it recognises the experiment and what the published
  average contributions were. The answers are reported with the result.

## Measures

- **H2.** The mean log-likelihood of each held-out person's band in each round, with every
  arm's probabilities clamped to one floor. Differences are paired by decision and
  resampled by group.
- **H3.** Each held-out human group is scored against the simulated groups of its game on
  three summaries: the group's level over the session, how far apart its members are within
  a round, and how far the last five rounds are above the first five. The score is the
  continuous ranked probability score, in its form that does not depend on how many groups
  were simulated. Which of these gates is fixed in the pre-registration, before the held-out
  groups are read: the mean of the three, or how far apart members are alone. On training
  groups the gain was mostly in the second and its power was higher. Each summary is
  reported alone either way.
- **Smallest differences that count.** Proposed: about a quarter of the effect found on
  training groups, fixed in the pre-registration. On training groups the held-out sample has
  ample power for H2 and less for H3; `measure.md` gives the figures.

## Controls

- **Null interaction.** In the `alone` arm no observation of another seat reaches a person.
  A test asserts that a coded person given zero weight on the others plays exactly as the
  `alone` person does under the same seed.
- **Disguise.** Every amount of points is multiplied by ten. The coded arms work in shares
  of the endowment, so their output must not change at all; a test asserts it. The prompted
  arm is run once on the disguised rules for the training games and the change is reported.
  This settles, for this world, the open finding in the main design about that control:
  quantities are put in canonical units before any law sees them.
- **When closeness arises.** How far apart the members of a group are, in the first round
  and after, against the same people re-formed into groups at random, is reported for the
  held-out groups as it was for the training groups. It is a description and excludes
  nothing.
- **Replay.** A recorded closed-loop run replays to the same log.
- **A game never seen.** The coded arms are also fitted on seven games and scored on the
  eighth, in rotation. Reported, not gating.

## What a pass and a fail would mean

- **H2 passes:** on these games, a coded law that uses the others' history improves the
  prediction of the next move, and the coded person predicts better than this prompt to this model.
- **H3 passes:** on these games, simulated groups of coded persons resemble human groups
  more than non-interacting persons and more than groups of this prompted model. It is a
  claim about three summaries of 20 rounds in groups of two and four. It is not a claim
  about larger populations, networks, or anything outside these rules.
- **H2 fails against `alone`:** the law that uses the others' history is not supported, and
  nothing social is built on it.
- **H2 passes and H3 fails:** the person predicts one step ahead and the groups it forms do
  not look human. That is the known gap between one-step prediction and simulation, and it
  would be the main finding.
- **Either fails against `prompted` only:** the coded person adds nothing over asking the
  model on this task, and the comparison is published as it stands.

Whatever happens is published, as with the pilot.

## What this changes in the main design

Made in the same reviewed change as the pre-registration:

- **Non-goals.** "It says nothing about one-shot bargaining, cooperation or norms" becomes:
  the first interaction test covers contributions to a repeated public good. Reasoning about
  what another believes, punishment and ties stay out.
- **Acceptance.** A row for this test is added before the kernel milestone. The kernel
  milestone is narrowed to what this test does not already build.
- **Decisions with more than two options.** The design reserved these for a race of
  accumulators that it had not anchored. This test uses an ordered response rule with no
  claim about timing, and says so.
- **The disguise control** is restated as above.

## Cost

Open-loop prompted calls: one per held-out decision and one per training decision used for
recalibration. Closed-loop prompted calls: one per seat and round of each simulated group.
The number of simulated groups per game is fixed in the pre-registration. A measured bound
on calls and spend is produced before registration, by a committed script, as it was for
the pilot. The spending cap is the maintainer's decision.

## Order of work

1. This note, reviewed.
2. The comparison of candidate person models on training groups.
3. Rendered instructions, a format check of the prompted requests on made-up histories, and
   the spending bound.
4. The pre-registration and the analysis code, reviewed, with the amendments to the main
   design.
5. Registration by tag; then the run; then the result, whatever it is.
