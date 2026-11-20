# What does the others' history add? A measurement on training groups

Run on 2026-11-20, before any design or registration, on the 210 training groups only. The
143 held-out groups are sealed and were not loaded.

**Answer: enough to test, in both directions.** Seeing what the others did last round
improves the prediction of a person's next contribution by 0.0435 nats per decision, and
simulated groups whose members respond to each other are closer to human groups than
simulated groups whose members do not. A check that uses no model agrees: members of a human
group start no closer to each other than strangers would be, are much closer from the second
round, and after the fifth are about twice as close. The measurement also shows two things the design has to accept:
knowing which game is played is worth little here, and a person's own past predicts ten
times more than the other people do.

What this is and is not. Every model below is an ordered logit over five bands of the share
contributed: nothing, up to a third, up to two thirds, more but not all, and everything.
They are measuring sticks. None is the person model that will be registered, no language
model is involved, and the comparison with prompted model agents is not measured here.
Each number comes from models fitted without the group being scored.

## Open loop: predicting the next contribution

Three models with a constant for each game and seat, scored on groups left out of the fit:

| Model looks at | Nats per decision |
|---|---|
| The game and the seat only | -1.4714 |
| Also the person's own last contribution | -1.0046 |
| Also the others' last contributions | -0.9611 |

- **A person's own history adds 0.4668 nats.** People repeat themselves. Whatever the
  registered person model is, it needs state that carries a person's own level forward.
- **The others' history adds 0.0435 nats**, with an interval from 0.0343 to 0.0527 over
  groups. Over rounds 2 to 20, where there is a history to see, it is 0.0467. The pilot's
  corresponding state effect was under 0.01 nats per choice.

## Is that influence, or people who were alike from the start?

A gain in prediction does not by itself show that people influence each other. Members who
never looked at each other but shared a tendency from the start would produce the same gain:
the others' history would then be a second reading of the shared tendency. A test in
`tests/test_interaction_measure.py` builds exactly such groups and shows that the model
comparison is fooled by them.

Two things speak against that here. The experiment formed groups at random. And the data can
be asked directly, with no model: how far apart are two members of a group, as a difference
in the share contributed, compared with the same people dealt into groups at random?

| Rounds | Human groups | Re-formed at random, with its 95% range |
|---|---|---|
| 1 | 0.4004 | 0.4055, from 0.3712 to 0.4410 |
| 2 to 5 | 0.2432 | 0.4033, from 0.3754 to 0.4317 |
| 6 to 15 | 0.1978 | 0.4045, from 0.3745 to 0.4305 |
| 16 to 20 | 0.2083 | 0.4295, from 0.3999 to 0.4632 |

In the first round, before anyone has seen anyone else, members of a group are as far apart
as strangers. A shared tendency would show there, and does not. From the second round they
are much closer, and after the fifth about half as far apart as strangers. People moved
toward the people they were grouped with.

This is evidence that members respond to each other. It does not say what the response is:
reciprocity, imitation and learning what the group will sustain would all look like this.

## A game left out

How much does a model lose when it has never seen the game being played? Two steps:

| Step | Nats per decision | Interval |
|---|---|---|
| A constant for each game and seat, against one constant for all | 0.0072 | 0.0016 to 0.0134 |
| Having groups of the game in the fit at all, with one constant | 0.0064 | 0.0032 to 0.0099 |
| Both together | 0.0137 | 0.0068 to 0.0212 |

Giving the model the game's four numbers in place of a constant recovers 0.0086 of the
0.0137. In the first round, where nothing but the rules can guide a prediction, the loss has
an interval from -0.0077 to 0.0372: it could be nothing and it could be several times the
average.

What this does and does not show:

- It bounds what knowing the game is worth to these models: about a third of what the
  others' history adds, and most of it recovered by four numbers.
- It is not a test of a language model's reading of the rules. No such reading was tried,
  and a reading might use more than these models can.
- The intervals resample groups. They do not account for there being only eight games, and
  a claim about reading rules never seen is a claim about games. Eight is too few to carry
  one.

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

- **The gain is where the check without a model says it should be**: in how far apart the
  members of a group are.
- **The level barely moves and the trend gets slightly worse.** The measuring stick has no
  account of why contributions drift over a session.
- Averaged over the three summaries the gain is 0.0187, with an interval from 0.0125 to
  0.0250. It holds for pairs and, with a wider interval, for groups of four.

## Can the held-out groups detect these effects?

A normal approximation, scaling the training intervals to 143 held-out groups: an effect of
the size found in training is detected with probability 1.000 in the open loop and 0.998 in
the closed loop. At half that size the probabilities are 0.969 and 0.683. Detecting means a
two-sided 95% interval over groups that excludes zero.

It assumes the held-out groups vary as the training groups do, treats the scoring models as
fixed, and takes the true effect to be the stated size. Those figures are for these measuring
sticks. The closed loop is the weaker test: if the registered person model gains less from
the others than the measuring stick does, the closed-loop comparison could miss it.

## What this means for the design

1. **Test what the others' history adds, open and closed loop.** Both effects are present in
   the human data and large enough to detect.
2. **Do not make the verdict depend on a model reading the rules.** With eight games that
   differ by four numbers, such a claim could not be supported either way. Report what
   knowing the game adds; do not gate on it. A test that needs reading and interaction
   together needs data with far more variety in the rules.
3. **The person needs a persistent level of their own.** In the open loop it can be taken
   from the person's history. In the closed loop there is no history to take it from, so it
   has to come from the person's state, drawn at the start and carried forward.
4. **Score the closed loop on what the others change.** A score on the group's level alone
   would hardly distinguish people who respond to each other from people who do not.
5. **Report the check without a model beside any model comparison.** It is what separates
   influence from people who were alike from the start.
6. **The prompted comparison is unmeasured.** Nothing here says how a prompted model
   predicts or plays these games. That needs model calls, and comes with the registration.

## Reproduce

`uv run --extra groups python -m zoon.interaction.measure` prints:

```
zoon interaction: what the others' history adds, on training groups
[data] training groups = 210; decisions = 10240
[bands] share of decisions = 0.095, 0.180, 0.175, 0.131, 0.420
[open loop, groups left out] per game, no history = -1.4714 nats per decision
[open loop, groups left out] per game, own history = -1.0046 nats per decision
[open loop, groups left out] per game, own and others = -0.9611 nats per decision
[open loop, groups left out] pooled, own and others = -0.9683 nats per decision
[open loop, groups left out] game numbers, own and others = -0.9600 nats per decision
[open loop] own history adds = +0.4668 [+0.4153, +0.5186]
[open loop] others' history adds = +0.0435 [+0.0343, +0.0527]
[open loop, rounds 2 to 20] others' history adds = +0.0467 [+0.0374, +0.0564]
[open loop, game left out] pooled, own and others = -0.9748 nats per decision
[open loop, game left out] game numbers, own and others = -0.9661 nats per decision
[game left out] a constant per game adds = +0.0072 [+0.0016, +0.0134]
[game left out] having the game in the fit adds = +0.0064 [+0.0032, +0.0099]
[game left out] both together = +0.0137 [+0.0068, +0.0212]
[game left out] the game's numbers recover = +0.0086 [+0.0029, +0.0148]
[game left out, round 1 only] both together = +0.0137 [-0.0077, +0.0372]
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
[members apart, round 1] human groups = 0.4004
[members apart, round 1] re-formed at random = 0.4055 [0.3712, 0.4410]
[members apart, rounds 2 to 5] human groups = 0.2432
[members apart, rounds 2 to 5] re-formed at random = 0.4033 [0.3754, 0.4317]
[members apart, rounds 6 to 15] human groups = 0.1978
[members apart, rounds 6 to 15] re-formed at random = 0.4045 [0.3745, 0.4305]
[members apart, rounds 16 to 20] human groups = 0.2083
[members apart, rounds 16 to 20] re-formed at random = 0.4295 [0.3999, 0.4632]
[power] held-out groups = 143
[power] open loop: at the training effect = 1.000; at half of it = 0.969
[power] closed loop: at the training effect = 0.998; at half of it = 0.683
```
