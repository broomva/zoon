# What does a law about other people add? A measurement on training groups

Run on 2026-11-20, before any design or registration, on the 210 training groups only. The
143 held-out groups are sealed and were not loaded.

**Answer: enough to test, in both directions.** Seeing what the others did last round
improves the prediction of a person's next contribution by 0.0435 nats per decision, and
simulated groups whose members respond to each other are closer to human groups than
simulated groups whose members do not. The held-out groups can detect both. The measurement
also shows two things the design has to accept: almost nothing here depends on reading the
rules, and a person's own past explains ten times more than the other people do.

What this is and is not. Every model below is an ordered logit over five bands of the share
contributed: nothing, up to a third, up to two thirds, more but not all, and everything.
They are measuring sticks. None is the person model that will be registered, no language
model is involved, and the comparison with prompted model agents is not measured here.
Each number comes from models fitted without the group being scored.

## Open loop: predicting the next contribution

Three models that know which game is being played, scored on groups left out of the fit:

| Model looks at | Nats per decision |
|---|---|
| The game and the seat only | -1.4714 |
| Also the person's own last contribution | -1.0046 |
| Also the others' last contributions | -0.9611 |

- **A person's own history is worth 0.4668 nats.** People repeat themselves. Whatever the
  registered person model is, it needs state that carries a person's own level forward.
- **The others' history adds 0.0435 nats**, with an interval from 0.0343 to 0.0527 over
  groups. Over rounds 2 to 20, where there is a history to see, it is 0.0467. The pilot's
  corresponding state effect was under 0.01 nats per choice.

## A game never seen

Fitting on seven games and scoring the eighth costs 0.0137 nats per decision against a model
that has seen the game. Giving the model the game's four numbers recovers 0.0086 of that.
In the first round, where nothing but the rules can guide a prediction, the cost has an
interval that includes zero.

So there is little here for a language model to read. The eight games differ by four numbers,
people respond to them weakly, and a coded model given the numbers gets most of what there is.
A hypothesis that required a model's reading of the rules to add to the coded laws would be
decided by noise.

## Closed loop: simulated groups against human groups

Each fold's models play 400 groups of each game forward from the first round, with no human
data inside the loop. Each human group is then scored against the simulated groups of its
game on three summaries, all in shares of the endowment: the group's level over the session,
how far apart its members are within a round, and how far the last five rounds are above the
first five. The score is the continuous ranked probability score, a distance between one
observed number and a simulated distribution; lower is closer.

| Summary | Members ignore each other | Members respond to each other | Gain |
|---|---|---|---|
| Level | 0.1841 | 0.1736 | +0.0105 |
| Members apart | 0.1538 | 0.1030 | +0.0508 |
| Trend | 0.1408 | 0.1460 | -0.0053 |

- **Interaction shows where it should.** Members of a human group are closer to each other
  than independent people would be, and only the simulation with interaction reproduces that.
- **The level barely moves and the trend gets slightly worse.** The measuring stick has no
  account of why contributions drift over a session.
- Averaged over the three summaries the gain is 0.0187, with an interval from 0.0125 to
  0.0250. It holds for pairs and, with a wider interval, for groups of four.

## Can the held-out groups detect these effects?

Scaling the training intervals to 143 held-out groups, an effect of the size found in
training is detected with probability 1.000 in the open loop and 0.998 in the closed loop.
At half that size the probabilities are 0.969 and 0.683. Detecting means a 95% interval over
groups that excludes zero. The closed loop is the weaker test: if the registered person model
gains less from interaction than this measuring stick does, the closed-loop comparison could
miss it.

## What this means for the design

1. **Test the law about other people, open and closed loop.** Both effects are present in
   the human data and large enough to detect.
2. **Do not make the verdict depend on a model reading the rules.** Report what a reading
   adds; do not gate on it. The pilot tested reading. A test that needs reading and
   interaction together needs data with far more variety in the rules.
3. **The person needs a persistent level of their own.** In the open loop it can be taken
   from the person's history. In the closed loop there is no history to take it from, so it
   has to come from the person's state, drawn at the start and carried forward.
4. **Score the closed loop on what interaction changes.** A score on the group's level alone
   would hardly distinguish people who interact from people who do not.
5. **The prompted comparison is unmeasured.** Nothing here says how a prompted model
   predicts or plays these games. That needs model calls, and comes with the registration.

## Reproduce

`uv run --extra groups python -m zoon.interaction.measure` prints:

```
zoon interaction: what a law about other people adds, on training groups
[data] training groups = 210; decisions = 10240
[bands] share of decisions = 0.095, 0.180, 0.175, 0.131, 0.420
[open loop, game seen in training] seen, no history = -1.4714 nats per decision
[open loop, game seen in training] seen, own history = -1.0046 nats per decision
[open loop, game seen in training] seen, own and others = -0.9611 nats per decision
[open loop, game seen in training] unseen, pooled = -0.9683 nats per decision
[open loop, game seen in training] unseen, game numbers = -0.9600 nats per decision
[open loop] seen, own history minus seen, no history = +0.4668 [+0.4153, +0.5186]
[open loop] seen, own and others minus seen, own history = +0.0435 [+0.0343, +0.0527]
[open loop, rounds 2 to 20] others' history adds = +0.0467 [+0.0374, +0.0564]
[open loop, game never seen] unseen, pooled = -0.9748 nats per decision
[open loop, game never seen] unseen, game numbers = -0.9661 nats per decision
[game never seen] seen minus unseen pooled = +0.0137 [+0.0068, +0.0212]
[game never seen] game numbers minus pooled = +0.0086 [+0.0029, +0.0148]
[game never seen, round 1 only] seen minus pooled = +0.0137 [-0.0077, +0.0372]
[closed loop, level] own history = 0.1841
[closed loop, level] own and others = 0.1736
[closed loop, level] others' history gains = +0.0105 [+0.0051, +0.0162]
[closed loop, members apart] own history = 0.1538
[closed loop, members apart] own and others = 0.1030
[closed loop, members apart] others' history gains = +0.0508 [+0.0364, +0.0650]
[closed loop, trend] own history = 0.1408
[closed loop, trend] own and others = 0.1460
[closed loop, trend] others' history gains = -0.0053 [-0.0079, -0.0025]
[closed loop, all three] others' history gains = +0.0187 [+0.0125, +0.0250]
[closed loop, pairs] others' history gains = +0.0197 [+0.0127, +0.0271]
[closed loop, groups of four] others' history gains = +0.0152 [+0.0024, +0.0271]
[power] held-out groups = 143
[power] open loop: at the training effect = 1.000; at half of it = 0.969
[power] closed loop: at the training effect = 0.998; at half of it = 0.683
```
