# What does the others' history add? A measurement on training groups

Run on 2026-11-20, before any design or registration, on the 210 training groups only. They
are fixed groups from the first session of the experiment. The 143 held-out groups are
absent from the file this analysis reads.

**Answer: enough to test, in both directions.** A model that sees what the others did last
round predicts a person's next contribution better, by 0.0435 nats per decision on top of
the person's own history. Simulated groups whose members use the others' history are closer
to human groups than simulated groups whose members do not. Both are statements about
prediction. Neither shows that people influence each other; the section on that says why.

What this is and is not. Every model below is an ordered logit over five bands of the share
contributed: nothing, up to a third, up to two thirds, more but not all, and everything.
They are measuring sticks. None is the person model that will be registered, no language
model is involved, and the comparison with prompted model agents is not measured here.
Every figure that comes from a model comes from one fitted without the group being scored.

## Open loop: predicting the next contribution

Models with a constant for each game and seat, scored on groups left out of the fit:

| Model looks at | Nats per decision |
|---|---|
| The game and the seat only | -1.4714 |
| Also the person's own last contribution | -1.0046 |
| Instead, the others' last contributions | -1.2088 |
| Both | -0.9611 |

A person's own last contribution and the others' are correlated, so what each adds depends
on which is entered first:

| | Entered first | Entered second |
|---|---|---|
| Own history | 0.4668 | 0.2477 |
| Others' history | 0.2626 | 0.0435 |

- **A person's own history is the stronger predictor in either order**, so whatever the
  registered person model is, it needs state that carries a person's own level forward.
- **The others' history adds 0.0435 nats on top of it**, with an interval from 0.0343 to
  0.0527 over groups. Over rounds 2 to 20, where there is a history to see, it is 0.0467.
  This is the quantity the registered open-loop test would be about. The pilot's
  corresponding state effect was under 0.01 nats per choice.

## Does this show that people influence each other?

No. It shows that the others' history predicts. Three things could produce that, and these
data cannot separate them:

- members respond to each other;
- members share a tendency from the start, so the others' history is a second reading of it;
- members share a tendency that shows only weakly or noisily at first and settles later.

One description of the data is still worth having, with no model and no bands: how far apart
are two members of a group, as a difference in the share of the endowment contributed,
compared with the same people dealt into groups at random?

| Rounds | Human groups | Re-formed at random | Closer than chance by, with its 95% range |
|---|---|---|---|
| 1 | 0.4180 | 0.4170 | -0.0010, from -0.0334 to 0.0344 |
| 2 to 5 | 0.2403 | 0.4033 | 0.1630, from 0.1362 to 0.1896 |
| 6 to 15 | 0.1953 | 0.4011 | 0.2058, from 0.1749 to 0.2320 |
| 16 to 20 | 0.2039 | 0.4280 | 0.2242, from 0.1931 to 0.2598 |

In the first round members of a group are as far apart as strangers. In every later block
they are much closer than the same people re-formed at random.

What that does and does not establish:

- It rules out one alternative: a shared tendency that is strong and already visible in the
  first round. A test builds such groups and the first round shows it.
- It does not rule out the others. A test builds groups that share a fixed target, express
  it noisily in the first round and exactly afterwards, with nobody looking at anybody; the
  table for those groups has the same shape as this one. A weak shared tendency also passes
  the first round unnoticed.
- The re-formed groups in later rounds are made of people who had by then played in real
  groups. They are a reference for "these members versus other people", not a picture of
  people who never interacted.
- The measure is of members contributing equal shares. Members who settled on unequal
  shares would look far apart.

The reason to think members were not alike from the start is the experiment's design: the
paper says participants were matched at random. That is the authors' procedure, not
something this analysis checked. People who sat in the same lab sitting share that sitting,
and the published data do not identify sittings.

None of this matters for what will be tested. The hypotheses are about prediction: whether
a person model that uses the others' history predicts held-out people better, and whether
groups simulated with it resemble held-out human groups more.

## A game left out

How much does a model lose when no group of the game being played was in its fit? Two steps:

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

- It is the value of a constant per game to models that already work in shares of the
  endowment and already see the person's last contribution. Both carry much of what
  differs between games, so this is not the value of knowing the game from nothing.
- It is not a test of a language model's reading of the rules. No such reading was tried.
- The intervals resample groups. They do not account for there being only eight games.

## Closed loop: simulated groups against human groups

Each fold's models play 400 groups of each game forward from the first round, with no human
data inside the loop. Each human group is then scored against the simulated groups of its
game on three summaries: the group's level over the session, how far apart its members are
within a round, and how far the last five rounds are above the first five. The simulated
groups play in bands, so here the human groups are put in the same bands, and every summary
is in the share that stands for a band. The score is the continuous ranked probability score,
a distance between one observed number and a simulated distribution; lower is closer.

| Summary | Own history only | Own and others' history | Gain |
|---|---|---|---|
| Level | 0.1841 | 0.1736 | +0.0105 |
| Members apart | 0.1538 | 0.1030 | +0.0508 |
| Trend | 0.1408 | 0.1460 | -0.0053 |

- **Most of the gain is in how far apart the members of a group are.**
- **The gain in level is small and its interval excludes zero. The trend gets slightly
  worse.** The measuring stick has no account of why contributions drift over a session.
- Averaged over the three summaries the gain is 0.0187, with an interval from 0.0125 to
  0.0250. It holds for pairs and, with a wider interval, for groups of four. The average
  gives the three summaries equal weight, which is a choice.

## Can the held-out groups detect these effects?

A normal approximation, scaling the training intervals to 143 held-out groups. Detecting
means a two-sided 95% interval over groups that excludes zero.

| Effect | At the size found in training | At half that size |
|---|---|---|
| Open loop, others' history entered second | 1.000 | 0.969 |
| Closed loop, mean of the three summaries | 0.998 | 0.683 |
| Closed loop, members apart alone | 1.000 | 0.821 |

A printed 1.000 is the approximation's rounding, not certainty. The approximation assumes
the held-out groups vary as the training groups do, treats the scoring models as fixed, and
takes the true effect to be the stated size, which is itself an estimate. Resampling groups
assumes that groups are independent of each other; groups from one lab sitting may not be
entirely, and sittings are not identified. These figures are for the measuring sticks. If
the registered person model gains less from the others' history than they do, the
closed-loop comparison is the one most likely to miss it.

## What this means for the design

1. **Test what the others' history adds to prediction, open and closed loop.** Both effects
   are present in the training groups and large enough to detect.
2. **Claim prediction, not influence.** The registered hypotheses and the write-up of their
   result should say that a model using the others' history predicts better, and stop there.
3. **Leave a model's reading of the rules out of this test's verdict.** This is a choice
   about the design, not a finding. Nothing here measured what a reading adds, and no power
   was computed for it. The reasons are that these models lose little on a game left out,
   and that any conclusion about rules never seen would generalise from eight games.
4. **The person needs a persistent level of their own.** In the open loop it can be taken
   from the person's history. In the closed loop there is no history to take it from, so it
   has to come from the person's state, drawn at the start and carried forward.
5. **Decide before registration how the closed loop is scored.** Here the gain is mostly in
   how far apart members are. Whether the registered measure is that summary or a mean of
   several changes the power, as the table shows.
6. **The prompted comparison is unmeasured.** Nothing here says how a prompted model
   predicts or plays these games. That needs model calls, and comes with the registration.

## Reproduce

`uv run --extra groups python -m zoon.interaction.data` fetches the raw files and writes the
training file. `uv run --extra groups python -m zoon.interaction.measure` then prints:

```
zoon interaction: what the others' history adds, on training groups
[data] training groups = 210; decisions = 10240
[bands] share of decisions = 0.095, 0.180, 0.175, 0.131, 0.420
[open loop, groups left out] per game, no history = -1.4714 nats per decision
[open loop, groups left out] per game, own history = -1.0046 nats per decision
[open loop, groups left out] per game, others' history = -1.2088 nats per decision
[open loop, groups left out] per game, own and others = -0.9611 nats per decision
[open loop, groups left out] pooled, own and others = -0.9683 nats per decision
[open loop, groups left out] game numbers, own and others = -0.9600 nats per decision
[open loop] own history adds, entered first = +0.4668 [+0.4153, +0.5186]
[open loop] others' history adds, entered second = +0.0435 [+0.0343, +0.0527]
[open loop] others' history adds, entered first = +0.2626 [+0.2154, +0.3117]
[open loop] own history adds, entered second = +0.2477 [+0.2133, +0.2828]
[open loop, rounds 2 to 20] others' history adds, entered second = +0.0467 [+0.0374, +0.0564]
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
[members apart, round 1] human groups = 0.4180
[members apart, round 1] re-formed at random = 0.4170 [0.3846, 0.4524]
[members apart, round 1] closer than chance by = -0.0010 [-0.0334, +0.0344]
[members apart, rounds 2 to 5] human groups = 0.2403
[members apart, rounds 2 to 5] re-formed at random = 0.4033 [0.3765, 0.4299]
[members apart, rounds 2 to 5] closer than chance by = +0.1630 [+0.1362, +0.1896]
[members apart, rounds 6 to 15] human groups = 0.1953
[members apart, rounds 6 to 15] re-formed at random = 0.4011 [0.3702, 0.4273]
[members apart, rounds 6 to 15] closer than chance by = +0.2058 [+0.1749, +0.2320]
[members apart, rounds 16 to 20] human groups = 0.2039
[members apart, rounds 16 to 20] re-formed at random = 0.4280 [0.3969, 0.4637]
[members apart, rounds 16 to 20] closer than chance by = +0.2242 [+0.1931, +0.2598]
[power] held-out groups = 143
[power] open loop: at the training effect = 1.000; at half of it = 0.969
[power] closed loop, all three: at the training effect = 0.998; at half of it = 0.683
[power] closed loop, members apart: at the training effect = 1.000; at half of it = 0.821
```
