# Candidate persons on training groups

This compares the six coded persons in `design.md` and each person's twin without the
others' history. It is a training comparison before registration, on 210 training
groups. The 143 held-out groups are absent from the file this comparison reads. The result
is about prediction and the five summaries, not a cause of human play.
The prompted arms are not run here.

## What was compared

Each person has an ordered logit for the first round, with no history, and another for
rounds 2 to 20. In either rule the cumulative probability through a band is the logistic
function of a cut point minus the score. Four ordered cut points give five bands. The
first-round constants and cut points are separate from the later ones.

The later score is a constant plus a weight for each state value, centred on one half.
A has an own trace and a trace of the mean share of the others. B replaces the constants
with an intercept and the four numbers from `measure.traits`. C has separate traces for
others of the same and the other seat type, each with its own fitted update fraction.
When a kind of other seat is missing, the missing trace copies the updated trace
of the present kind. D observes others' mean points divided by the person's own endowment, capped at one before updating
the trace. E adds the exact first contribution share as a fixed anchor in every later
score. It keeps the own trace too. F drops the own trace. Each twin removes every weight
and update fraction for the others. F's twin has no state in its later rule.

Shares use contributions divided by endowments, including within a band. Each trace starts
at its first observation. After a round it moves towards the new observation by its fitted
fraction; the updated trace enters the next decision. There are no fitted starting values
or person effects. E's anchor is recorded play in the open loop and drawn play in the
closed loop.

## Fitting choices

These choices were fixed on synthetic groups before the training comparison.

- Bands are exactly `measure.band`: nothing, positive shares up to a third, up to two
  thirds, more but not all, and everything.
- Constants differ by game and seat type, separately in each of the two rules. There are
  fourteen such constants in each rule. B instead has one intercept and the four game
  numbers in each rule. State weights, update fractions and cut points are shared across
  games and seats within their rule. There is no history in the first rule.
- B needs a declared point unit. The mean endowment is set to 24 before calling
  `measure.traits`. This leaves its four numbers unchanged for these eight games and
  makes a common multiplication of all points a change of units. Productivities do not
  change. An unseen game-seat constant in the other candidates is zero: the shared cut
  points give its baseline. This fixes the otherwise undefined game-left-out calculation.
- The objective is the negative sum of decision log-likelihoods plus 0.01 times the sum
  of squares of every fitted number, including update fractions and cut-point parameters.
  It uses the small ridge penalty of `measure.fit`, extended to the trace fractions.
- A cut-point origin and three log gaps parameterise each set of cut points. Constants
  and state weights start at zero, origins at -1.5, log gaps at zero, and update fractions
  at one half. There is one start for every fit. The fractions lie in the closed interval
  from zero to one; the log gaps lie from -8 to 8 to avoid numerical overflow. All other
  numbers are unbounded.
- L-BFGS-B uses an analytic gradient, at most 2000 iterations and 50 line-search steps,
  function tolerance 0.0000000001 and projected-gradient tolerance 0.00000010. Either
  successful stopping condition is accepted. Any unsuccessful result raises with the
  optimiser's message. A single start is simple and reproducible; it does not guarantee
  a global optimum for the fitted trace fractions.
- Every probability is floored at 0.0000000010 and divided by the sum. H2 is a mean per
  decision. Its paired interval uses `measure.paired`, resampling whole groups.
- Every fitted scalar counts, including the fractions and both rules' cut points.
  Identical twins A, C and D have the same equation, penalty and start. Their fits and
  simulations are reused exactly.

The pasted output gives every fitted number from a fit on all training groups. These are
not the fits that produce the scores. Each score leaves the group's fold out, using the
five folds of `measure.fold_of`. The selection rule takes the highest mean score, using
the earliest letter for an exact tie; collects the candidates whose paired interval with
that reference includes zero; then chooses the fewest fitted numbers, highest score and
earliest letter, in that order. Closed-loop scores do not choose the person.

The design leaves C's sharing of fractions unstated. Its two fractions are fitted
separately; a missing trace copies the present updated trace. The first-round parameter
sharing, the absent-game baseline and B's point unit are also fixed above. Amounts are
sampled at their frequencies, rather than uniformly over distinct amounts, so the table
keeps the training distribution within each band. Empty support stops the run.

## Closed loop and power

The proposed registered value is 100 simulated groups per game and fold. Each fold fits on
the other folds and builds its amount table from those groups only. Within each endowment
and middle band, amounts are drawn at their observed frequencies. Nothing and everything
are exact. Missing training support for a middle band raises an error rather than adding
an invented amount. The full-training table is printed as points and counts in the paste;
the fold tables exclude the groups being scored.

The kernel collects a whole round's decisions before delivering any outcome. It uses
`Game.payoffs` with exact fractions and checks the balance every round. Its append-only log
records intents, transfers and observations. Replay recomputes the draws and ledger from
the logged decision distributions and rejects changed records. Candidates receive every
seat's contribution and payoff. Twins receive their own contribution and the rules, with
no payoff. This matters in a pair, where a payoff could disclose the partner's contribution.

Streams use a SHA-256 hash of the seed, game, group, round, seat and draw purpose. The seed
is `measure.SEED`; simulation group names include the fold, game and simulation index.
There is a separate draw for the band and the amount. The same streams serve all arms,
and adding another group does not consume any existing group's draws.

All summaries use exact contribution shares: level, members apart, trend, movement and
type gap, as defined in `design.md`. The type gap is absent only in the two equal-seat
games; productivity inequality alone defines two types. For draws x and observation y,
the fair ranked probability score is the mean of |x-y| minus the sum of |x_i-x_j| over
ordered distinct pairs divided by twice n(n-1). Its expectation does not depend on the
number of independent simulated groups. Lower scores are better.

For a comparison, differences are averaged over groups where each summary is defined,
then the five summary means have equal weight. Thus groups without a type gap are not
dropped from the other summaries. Intervals use 2000 resamples of whole groups within each
game, keeping each game's count fixed. Coverage counts human values inside the inclusive
fifth to ninety-fifth percentiles of that fold's simulated values, using linear quantiles.

Power uses `measure.power`: a normal approximation for a two-sided interval excluding
zero, at the training effect and half of it. The error read from the training interval
is scaled by the square root of training groups divided by held-out groups. For the
type gap only, both counts exclude the equal-seat games. The approximation assumes the
held-out groups have the same variability and roughly the same game mixture as training,
treats the fitted models and simulations as fixed, and assumes the stated effect is true.
A negative effect also counts as detectable. These are not probabilities of passing the
registered test: they omit its smallest useful difference and its prompted comparator.

The intervals and power are conditional on the realised simulated groups. They include
neither refitting nor selection uncertainty, and do not add uncertainty for the finite
simulation count. Groups are assumed independent. A shared lab sitting could violate that
assumption. Selection and effect estimation use the same training groups; held-out testing is still needed.

## Results

The candidate winner is B. The twin winner is B twin.

Step 2 ties among candidates: A, B, C, D, E.
Step 2 ties among twins: B twin.
The highest candidate score is D's. The rule chooses B because it has the fewest fitted numbers among the tied candidates. The selected
twin is also the winner's own twin, so the two coded-arm comparisons coincide.

Scores below have their group intervals. Higher is better in the open loop and lower
is better in the closed loop. All twelve closed-loop composites are reported; they do
not enter selection.

| Form | Fitted numbers | Open-loop nats per decision | Closed-loop composite |
|---|---|---|---|
| A | 40 | -0.9330 [-0.9930, -0.8699] | +0.1191 [+0.1125, +0.1261] |
| B | 22 | -0.9300 [-0.9905, -0.8667] | +0.1187 [+0.1120, +0.1255] |
| C | 42 | -0.9329 [-0.9928, -0.8697] | +0.1191 [+0.1125, +0.1260] |
| D | 40 | -0.9292 [-0.9924, -0.8658] | +0.1195 [+0.1126, +0.1266] |
| E | 41 | -0.9329 [-0.9931, -0.8698] | +0.1185 [+0.1119, +0.1255] |
| F | 38 | -1.1904 [-1.2583, -1.1236] | +0.1589 [+0.1523, +0.1660] |
| A twin | 38 | -0.9597 [-1.0204, -0.8962] | +0.1321 [+0.1269, +0.1378] |
| B twin | 20 | -0.9553 [-1.0161, -0.8930] | +0.1311 [+0.1260, +0.1365] |
| C twin | 38 | -0.9597 [-1.0204, -0.8962] | +0.1321 [+0.1269, +0.1378] |
| D twin | 38 | -0.9597 [-1.0204, -0.8962] | +0.1321 [+0.1269, +0.1378] |
| E twin | 39 | -0.9599 [-1.0208, -0.8966] | +0.1321 [+0.1268, +0.1378] |
| F twin | 36 | -1.4687 [-1.5227, -1.4163] | +0.1890 [+0.1823, +0.1964] |

The full-training fit for B has own trace fraction +0.658827
and others' fraction +1.000000. The twin's own fraction is
+0.631901. Every fitted number, including both rules' cut points, is in the paste.

### Coded-arm gains

Positive gains favour the selected candidate: candidate minus twin log-likelihood
in the open loop, and twin minus candidate score in the closed loop.

| Measure | Against selected twin | Against own twin |
|---|---|---|
| H2 | +0.0253 [+0.0179, +0.0328] | +0.0253 [+0.0179, +0.0328] |
| composite | +0.0124 [+0.0085, +0.0163] | +0.0124 [+0.0085, +0.0163] |
| level | +0.0052 [+0.0005, +0.0098] | +0.0052 [+0.0005, +0.0098] |
| members apart | +0.0466 [+0.0355, +0.0571] | +0.0466 [+0.0355, +0.0571] |
| trend | -0.0023 [-0.0045, -0.0001] | -0.0023 [-0.0045, -0.0001] |
| movement | +0.0012 [+0.0004, +0.0020] | +0.0012 [+0.0004, +0.0020] |
| type gap | +0.0115 [+0.0040, +0.0184] | +0.0115 [+0.0040, +0.0184] |

Both the H2 and H3 composite intervals are above zero against the coded twin.
The largest summary gain is in members apart. Trend scores worse. This is a training
comparison, with the selection made on these same groups, not a held-out verdict.

### Coverage

The fraction of human values within the simulated central 90% range:

| Summary | Selected candidate | Selected twin | Own twin |
|---|---|---|---|
| level | 0.7333 | 0.6143 | 0.6143 |
| members apart | 0.6714 | 0.6667 | 0.6667 |
| trend | 0.8952 | 0.8619 | 0.8619 |
| movement | 0.5476 | 0.4429 | 0.4429 |
| type gap | 0.8075 | 0.9938 | 0.9938 |

Coverage for the selected candidate is below 90% on every summary. Relative gains
do not establish adequate absolute fit.

### Power

| Measure | Selected twin, training effect | Selected twin, half effect | Own twin, training effect | Own twin, half effect |
|---|---|---|---|---|
| H2 | 1.000 | 0.782 | 1.000 | 0.782 |
| composite | 0.999 | 0.738 | 0.999 | 0.738 |
| level | 0.437 | 0.147 | 0.437 | 0.147 |
| members apart | 1.000 | 0.939 | 1.000 | 0.939 |
| trend | 0.395 | 0.135 | 0.395 | 0.135 |
| movement | 0.627 | 0.208 | 0.627 | 0.208 |
| type gap | 0.732 | 0.252 | 0.732 | 0.252 |

These are two-sided detection estimates under the assumptions above. A printed
1.000 is not certainty. They do not include a smallest useful difference or a
comparison with a prompted arm.

### A game left out

Each game is scored by a fit to the other seven games. The loss is the ordinary
fold score minus this score, paired by decision. The intervals vary groups, not games.

| Arm | Nats per decision | Loss from leaving the game out |
|---|---|---|
| candidate | -0.9357 [-0.9981, -0.8714] | +0.0058 [+0.0018, +0.0098] |
| twin | -0.9587 [-1.0218, -0.8944] | +0.0034 [-0.0003, +0.0072] |

| Game | Candidate | Twin |
|---|---|---|
| 2-AI | -0.8811 | -0.8915 |
| 2-EI | -0.9194 | -0.9203 |
| 2-FE | -0.7242 | -0.7800 |
| 2-MI | -0.9696 | -0.9748 |
| 2-PI | -0.8604 | -0.9047 |
| 4-AI | -0.9518 | -0.9626 |
| 4-FE | -1.1331 | -1.1582 |
| 4-MI | -1.0728 | -1.1012 |

## What this means for the pre-registration

- Use B as the coded person and B twin as the coded comparator. The winner's own
  twin is also B twin; keep that comparison explicit.
- Fix the first-round rule, exact-share state, fitting choices, bands, probability floor,
  decision weighting, fair score and summary combination described here.
- Take the printed full-training amount table, seed streams and simulation count as the
  proposed registered choices. The treatment of finite simulation error still needs an
  explicit registered decision.
- The smallest differences, prompted arms, rendered instructions, recalibration, recall
  check, spending bound and stopping rules remain open. No result here decides them.

## Reproduce

```sh
uv sync --extra groups
uv run --extra groups python -m zoon.interaction.data
uv run --extra groups python -m zoon.interaction.compare
```

```
zoon interaction: candidates on training groups
[data] training groups = 210
[data] decisions = 10240
[data] held-out groups = 143
[simulation] groups per game and fold = 100
[fit] ridge = 0.01
[fit] probability floor = 0.0000000010
[fit] log gap limit = 8.0
[fit] iteration limit = 2000
[fit] gradient tolerance = 0.00000010
[fit] function tolerance = 0.0000000001
[amounts, endowment 12, band 1] points:count = 1:73, 2:96, 3:77, 4:73
[amounts, endowment 12, band 2] points:count = 5:84, 6:171, 7:62, 8:153
[amounts, endowment 12, band 3] points:count = 9:99, 10:307, 11:86
[amounts, endowment 24, band 1] points:count = 1:71, 2:62, 3:69, 4:64, 5:107, 6:96, 7:37, 8:97
[amounts, endowment 24, band 2] points:count = 9:31, 10:233, 11:39, 12:272, 13:27, 14:38, 15:117, 16:47
[amounts, endowment 24, band 3] points:count = 17:20, 18:44, 19:20, 20:144, 21:18, 22:43, 23:40
[amounts, endowment 36, band 1] points:count = 1:76, 2:77, 3:56, 4:61, 5:66, 6:78, 7:37, 8:64, 9:42, 10:201, 11:19, 12:143
[amounts, endowment 36, band 2] points:count = 13:29, 14:27, 15:44, 16:38, 17:8, 18:70, 19:12, 20:157, 21:20, 22:21, 23:19, 24:71
[amounts, endowment 36, band 3] points:count = 25:39, 26:37, 27:26, 28:35, 29:8, 30:200, 31:9, 32:46, 33:12, 34:68, 35:39
[count] A = 40
[numbers, A] first constant 2-AI type 0 = -0.492010
[numbers, A] first constant 2-AI type 1 = -0.113326
[numbers, A] first constant 2-EI type 0 = -0.439959
[numbers, A] first constant 2-EI type 1 = +0.557407
[numbers, A] first constant 2-FE type 0 = +0.223170
[numbers, A] first constant 2-MI type 0 = -0.815205
[numbers, A] first constant 2-MI type 1 = +0.403409
[numbers, A] first constant 2-PI type 0 = +0.231507
[numbers, A] first constant 2-PI type 1 = -0.210836
[numbers, A] first constant 4-AI type 0 = +0.202098
[numbers, A] first constant 4-AI type 1 = +0.311992
[numbers, A] first constant 4-FE type 0 = +0.489304
[numbers, A] first constant 4-MI type 0 = +0.094044
[numbers, A] first constant 4-MI type 1 = +0.811709
[numbers, A] first cut origin = -2.742254
[numbers, A] first log gap 1 = +0.768556
[numbers, A] first log gap 2 = +0.137556
[numbers, A] first log gap 3 = -1.114979
[numbers, A] later constant 2-AI type 0 = +0.001135
[numbers, A] later constant 2-AI type 1 = +0.536647
[numbers, A] later constant 2-EI type 0 = -0.392263
[numbers, A] later constant 2-EI type 1 = +0.593983
[numbers, A] later constant 2-FE type 0 = +0.403299
[numbers, A] later constant 2-MI type 0 = -0.469235
[numbers, A] later constant 2-MI type 1 = +0.482102
[numbers, A] later constant 2-PI type 0 = +0.137610
[numbers, A] later constant 2-PI type 1 = +0.288871
[numbers, A] later constant 4-AI type 0 = +0.034514
[numbers, A] later constant 4-AI type 1 = +0.179657
[numbers, A] later constant 4-FE type 0 = +0.274700
[numbers, A] later constant 4-MI type 0 = -0.276977
[numbers, A] later constant 4-MI type 1 = +0.186343
[numbers, A] later cut origin = -3.469335
[numbers, A] later log gap 1 = +0.942333
[numbers, A] later log gap 2 = +0.805802
[numbers, A] later log gap 3 = +0.407663
[numbers, A] weight own = +7.022916
[numbers, A] weight others = +2.047300
[numbers, A] trace fraction own = +0.658705
[numbers, A] trace fraction others = +1.000000
[open loop] A = -0.9330 [-0.9930, -0.8699]
[closed loop, composite] A = +0.1191 [+0.1125, +0.1261]
[closed loop, level] A = +0.1724 [+0.1575, +0.1880]
[closed loop, members apart] A = +0.0983 [+0.0895, +0.1081]
[closed loop, trend] A = +0.1510 [+0.1353, +0.1683]
[closed loop, movement] A = +0.0551 [+0.0508, +0.0599]
[closed loop, type gap] A = +0.1187 [+0.1031, +0.1356]
[coverage, level] A = 0.7429
[coverage, members apart] A = 0.6905
[coverage, trend] A = 0.8571
[coverage, movement] A = 0.5476
[coverage, type gap] A = 0.7764
[count] B = 22
[numbers, B] first constant = +1.402043
[numbers, B] first endowment = -0.690580
[numbers, B] first return = +0.986222
[numbers, B] first product = -0.744029
[numbers, B] first four seats = +0.432222
[numbers, B] first cut origin = -1.282415
[numbers, B] first log gap 1 = +0.759424
[numbers, B] first log gap 2 = +0.129673
[numbers, B] first log gap 3 = -1.117355
[numbers, B] later constant = +1.882963
[numbers, B] later endowment = -0.622430
[numbers, B] later return = +0.286744
[numbers, B] later product = +1.167341
[numbers, B] later four seats = -0.062736
[numbers, B] later cut origin = -1.763336
[numbers, B] later log gap 1 = +0.936321
[numbers, B] later log gap 2 = +0.805354
[numbers, B] later log gap 3 = +0.405614
[numbers, B] weight own = +7.079210
[numbers, B] weight others = +1.940905
[numbers, B] trace fraction own = +0.658827
[numbers, B] trace fraction others = +1.000000
[open loop] B = -0.9300 [-0.9905, -0.8667]
[closed loop, composite] B = +0.1187 [+0.1120, +0.1255]
[closed loop, level] B = +0.1679 [+0.1551, +0.1816]
[closed loop, members apart] B = +0.1017 [+0.0926, +0.1115]
[closed loop, trend] B = +0.1488 [+0.1330, +0.1659]
[closed loop, movement] B = +0.0547 [+0.0504, +0.0594]
[closed loop, type gap] B = +0.1205 [+0.1044, +0.1384]
[coverage, level] B = 0.7333
[coverage, members apart] B = 0.6714
[coverage, trend] B = 0.8952
[coverage, movement] B = 0.5476
[coverage, type gap] B = 0.8075
[count] C = 42
[numbers, C] first constant 2-AI type 0 = -0.491810
[numbers, C] first constant 2-AI type 1 = -0.115018
[numbers, C] first constant 2-EI type 0 = -0.438966
[numbers, C] first constant 2-EI type 1 = +0.557831
[numbers, C] first constant 2-FE type 0 = +0.223349
[numbers, C] first constant 2-MI type 0 = -0.814907
[numbers, C] first constant 2-MI type 1 = +0.403564
[numbers, C] first constant 2-PI type 0 = +0.231315
[numbers, C] first constant 2-PI type 1 = -0.210656
[numbers, C] first constant 4-AI type 0 = +0.203113
[numbers, C] first constant 4-AI type 1 = +0.311839
[numbers, C] first constant 4-FE type 0 = +0.491105
[numbers, C] first constant 4-MI type 0 = +0.093912
[numbers, C] first constant 4-MI type 1 = +0.812080
[numbers, C] first cut origin = -2.741875
[numbers, C] first log gap 1 = +0.768541
[numbers, C] first log gap 2 = +0.137576
[numbers, C] first log gap 3 = -1.115119
[numbers, C] later constant 2-AI type 0 = +0.001496
[numbers, C] later constant 2-AI type 1 = +0.537250
[numbers, C] later constant 2-EI type 0 = -0.392242
[numbers, C] later constant 2-EI type 1 = +0.594814
[numbers, C] later constant 2-FE type 0 = +0.403630
[numbers, C] later constant 2-MI type 0 = -0.469257
[numbers, C] later constant 2-MI type 1 = +0.482881
[numbers, C] later constant 2-PI type 0 = +0.137991
[numbers, C] later constant 2-PI type 1 = +0.289228
[numbers, C] later constant 4-AI type 0 = +0.058081
[numbers, C] later constant 4-AI type 1 = +0.152382
[numbers, C] later constant 4-FE type 0 = +0.274995
[numbers, C] later constant 4-MI type 0 = -0.248622
[numbers, C] later constant 4-MI type 1 = +0.162086
[numbers, C] later cut origin = -3.469836
[numbers, C] later log gap 1 = +0.942208
[numbers, C] later log gap 2 = +0.806691
[numbers, C] later log gap 3 = +0.407932
[numbers, C] weight own = +7.023070
[numbers, C] weight same = +0.914466
[numbers, C] weight other = +1.134077
[numbers, C] trace fraction own = +0.658782
[numbers, C] trace fraction same = +1.000000
[numbers, C] trace fraction other = +1.000000
[open loop] C = -0.9329 [-0.9928, -0.8697]
[closed loop, composite] C = +0.1191 [+0.1125, +0.1260]
[closed loop, level] C = +0.1723 [+0.1575, +0.1879]
[closed loop, members apart] C = +0.0982 [+0.0894, +0.1080]
[closed loop, trend] C = +0.1505 [+0.1349, +0.1678]
[closed loop, movement] C = +0.0551 [+0.0508, +0.0598]
[closed loop, type gap] C = +0.1192 [+0.1037, +0.1361]
[coverage, level] C = 0.7429
[coverage, members apart] C = 0.6905
[coverage, trend] C = 0.8571
[coverage, movement] C = 0.5476
[coverage, type gap] C = 0.7764
[count] D = 40
[numbers, D] first constant 2-AI type 0 = -0.492247
[numbers, D] first constant 2-AI type 1 = -0.114882
[numbers, D] first constant 2-EI type 0 = -0.439368
[numbers, D] first constant 2-EI type 1 = +0.558024
[numbers, D] first constant 2-FE type 0 = +0.223121
[numbers, D] first constant 2-MI type 0 = -0.815565
[numbers, D] first constant 2-MI type 1 = +0.403709
[numbers, D] first constant 2-PI type 0 = +0.231399
[numbers, D] first constant 2-PI type 1 = -0.210913
[numbers, D] first constant 4-AI type 0 = +0.202962
[numbers, D] first constant 4-AI type 1 = +0.311970
[numbers, D] first constant 4-FE type 0 = +0.490677
[numbers, D] first constant 4-MI type 0 = +0.093889
[numbers, D] first constant 4-MI type 1 = +0.812438
[numbers, D] first cut origin = -2.741697
[numbers, D] first log gap 1 = +0.768376
[numbers, D] first log gap 2 = +0.137571
[numbers, D] first log gap 3 = -1.115071
[numbers, D] later constant 2-AI type 0 = +0.994485
[numbers, D] later constant 2-AI type 1 = -0.216710
[numbers, D] later constant 2-EI type 0 = +0.659795
[numbers, D] later constant 2-EI type 1 = -0.363459
[numbers, D] later constant 2-FE type 0 = +0.267818
[numbers, D] later constant 2-MI type 0 = +0.538784
[numbers, D] later constant 2-MI type 1 = -0.242536
[numbers, D] later constant 2-PI type 0 = +0.035292
[numbers, D] later constant 2-PI type 1 = +0.187493
[numbers, D] later constant 4-AI type 0 = +0.674491
[numbers, D] later constant 4-AI type 1 = -0.484009
[numbers, D] later constant 4-FE type 0 = +0.147234
[numbers, D] later constant 4-MI type 0 = +0.387701
[numbers, D] later constant 4-MI type 1 = -0.484361
[numbers, D] later cut origin = -3.588501
[numbers, D] later log gap 1 = +0.947654
[numbers, D] later log gap 2 = +0.819406
[numbers, D] later log gap 3 = +0.408345
[numbers, D] weight own = +7.047119
[numbers, D] weight others = +2.554551
[numbers, D] trace fraction own = +0.661839
[numbers, D] trace fraction others = +1.000000
[open loop] D = -0.9292 [-0.9924, -0.8658]
[closed loop, composite] D = +0.1195 [+0.1126, +0.1266]
[closed loop, level] D = +0.1719 [+0.1571, +0.1879]
[closed loop, members apart] D = +0.0977 [+0.0888, +0.1078]
[closed loop, trend] D = +0.1515 [+0.1355, +0.1693]
[closed loop, movement] D = +0.0581 [+0.0536, +0.0630]
[closed loop, type gap] D = +0.1182 [+0.1032, +0.1349]
[coverage, level] D = 0.7238
[coverage, members apart] D = 0.6762
[coverage, trend] D = 0.8762
[coverage, movement] D = 0.5048
[coverage, type gap] D = 0.8012
[count] E = 41
[numbers, E] first constant 2-AI type 0 = -0.492179
[numbers, E] first constant 2-AI type 1 = -0.115018
[numbers, E] first constant 2-EI type 0 = -0.439174
[numbers, E] first constant 2-EI type 1 = +0.558054
[numbers, E] first constant 2-FE type 0 = +0.223202
[numbers, E] first constant 2-MI type 0 = -0.815429
[numbers, E] first constant 2-MI type 1 = +0.403717
[numbers, E] first constant 2-PI type 0 = +0.231368
[numbers, E] first constant 2-PI type 1 = -0.210866
[numbers, E] first constant 4-AI type 0 = +0.203055
[numbers, E] first constant 4-AI type 1 = +0.311945
[numbers, E] first constant 4-FE type 0 = +0.490913
[numbers, E] first constant 4-MI type 0 = +0.093874
[numbers, E] first constant 4-MI type 1 = +0.812444
[numbers, E] first cut origin = -2.742275
[numbers, E] first log gap 1 = +0.768648
[numbers, E] first log gap 2 = +0.137598
[numbers, E] first log gap 3 = -1.114883
[numbers, E] later constant 2-AI type 0 = +0.025688
[numbers, E] later constant 2-AI type 1 = +0.583950
[numbers, E] later constant 2-EI type 0 = -0.397979
[numbers, E] later constant 2-EI type 1 = +0.610424
[numbers, E] later constant 2-FE type 0 = +0.397421
[numbers, E] later constant 2-MI type 0 = -0.468242
[numbers, E] later constant 2-MI type 1 = +0.484904
[numbers, E] later constant 2-PI type 0 = +0.117186
[numbers, E] later constant 2-PI type 1 = +0.300787
[numbers, E] later constant 4-AI type 0 = +0.028034
[numbers, E] later constant 4-AI type 1 = +0.174849
[numbers, E] later constant 4-FE type 0 = +0.250791
[numbers, E] later constant 4-MI type 0 = -0.281201
[numbers, E] later constant 4-MI type 1 = +0.165562
[numbers, E] later cut origin = -3.478544
[numbers, E] later log gap 1 = +0.942193
[numbers, E] later log gap 2 = +0.806073
[numbers, E] later log gap 3 = +0.410341
[numbers, E] weight own = +6.804549
[numbers, E] weight others = +2.108689
[numbers, E] weight anchor = +0.325258
[numbers, E] trace fraction own = +0.685318
[numbers, E] trace fraction others = +1.000000
[open loop] E = -0.9329 [-0.9931, -0.8698]
[closed loop, composite] E = +0.1185 [+0.1119, +0.1255]
[closed loop, level] E = +0.1712 [+0.1568, +0.1867]
[closed loop, members apart] E = +0.0980 [+0.0892, +0.1075]
[closed loop, trend] E = +0.1503 [+0.1344, +0.1680]
[closed loop, movement] E = +0.0546 [+0.0503, +0.0593]
[closed loop, type gap] E = +0.1183 [+0.1031, +0.1349]
[coverage, level] E = 0.7476
[coverage, members apart] E = 0.6905
[coverage, trend] E = 0.8476
[coverage, movement] E = 0.5714
[coverage, type gap] E = 0.7950
[count] F = 38
[numbers, F] first constant 2-AI type 0 = -0.491871
[numbers, F] first constant 2-AI type 1 = -0.115022
[numbers, F] first constant 2-EI type 0 = -0.438846
[numbers, F] first constant 2-EI type 1 = +0.557773
[numbers, F] first constant 2-FE type 0 = +0.223104
[numbers, F] first constant 2-MI type 0 = -0.814970
[numbers, F] first constant 2-MI type 1 = +0.403502
[numbers, F] first constant 2-PI type 0 = +0.231231
[numbers, F] first constant 2-PI type 1 = -0.210714
[numbers, F] first constant 4-AI type 0 = +0.203014
[numbers, F] first constant 4-AI type 1 = +0.311763
[numbers, F] first constant 4-FE type 0 = +0.490865
[numbers, F] first constant 4-MI type 0 = +0.093829
[numbers, F] first constant 4-MI type 1 = +0.812022
[numbers, F] first cut origin = -2.741888
[numbers, F] first log gap 1 = +0.768495
[numbers, F] first log gap 2 = +0.137536
[numbers, F] first log gap 3 = -1.114992
[numbers, F] later constant 2-AI type 0 = -0.472036
[numbers, F] later constant 2-AI type 1 = +0.942187
[numbers, F] later constant 2-EI type 0 = -1.351335
[numbers, F] later constant 2-EI type 1 = +1.541776
[numbers, F] later constant 2-FE type 0 = +0.300959
[numbers, F] later constant 2-MI type 0 = -1.508118
[numbers, F] later constant 2-MI type 1 = +1.268023
[numbers, F] later constant 2-PI type 0 = +0.033112
[numbers, F] later constant 2-PI type 1 = -0.002148
[numbers, F] later constant 4-AI type 0 = -0.238949
[numbers, F] later constant 4-AI type 1 = +0.439045
[numbers, F] later constant 4-FE type 0 = +0.049615
[numbers, F] later constant 4-MI type 0 = -0.475551
[numbers, F] later constant 4-MI type 1 = +0.520345
[numbers, F] later cut origin = -2.533134
[numbers, F] later log gap 1 = +0.628663
[numbers, F] later log gap 2 = +0.246391
[numbers, F] later log gap 3 = -0.115412
[numbers, F] weight others = +5.187141
[numbers, F] trace fraction others = +0.752642
[open loop] F = -1.1904 [-1.2583, -1.1236]
[closed loop, composite] F = +0.1589 [+0.1523, +0.1660]
[closed loop, level] F = +0.1875 [+0.1705, +0.2055]
[closed loop, members apart] F = +0.1412 [+0.1296, +0.1539]
[closed loop, trend] F = +0.1445 [+0.1299, +0.1605]
[closed loop, movement] F = +0.1824 [+0.1728, +0.1922]
[closed loop, type gap] F = +0.1391 [+0.1212, +0.1583]
[coverage, level] F = 0.4524
[coverage, members apart] F = 0.4238
[coverage, trend] F = 0.9000
[coverage, movement] F = 0.1905
[coverage, type gap] F = 0.3292
[count] A twin = 38
[numbers, A twin] first constant 2-AI type 0 = -0.492417
[numbers, A twin] first constant 2-AI type 1 = -0.115041
[numbers, A twin] first constant 2-EI type 0 = -0.439316
[numbers, A twin] first constant 2-EI type 1 = +0.558254
[numbers, A twin] first constant 2-FE type 0 = +0.223122
[numbers, A twin] first constant 2-MI type 0 = -0.815836
[numbers, A twin] first constant 2-MI type 1 = +0.403859
[numbers, A twin] first constant 2-PI type 0 = +0.231449
[numbers, A twin] first constant 2-PI type 1 = -0.210979
[numbers, A twin] first constant 4-AI type 0 = +0.203070
[numbers, A twin] first constant 4-AI type 1 = +0.312058
[numbers, A twin] first constant 4-FE type 0 = +0.490616
[numbers, A twin] first constant 4-MI type 0 = +0.093892
[numbers, A twin] first constant 4-MI type 1 = +0.812760
[numbers, A twin] first cut origin = -2.742170
[numbers, A twin] first log gap 1 = +0.768593
[numbers, A twin] first log gap 2 = +0.137591
[numbers, A twin] first log gap 3 = -1.114585
[numbers, A twin] later constant 2-AI type 0 = +0.216674
[numbers, A twin] later constant 2-AI type 1 = +0.332657
[numbers, A twin] later constant 2-EI type 0 = +0.039996
[numbers, A twin] later constant 2-EI type 1 = +0.113874
[numbers, A twin] later constant 2-FE type 0 = +0.386062
[numbers, A twin] later constant 2-MI type 0 = -0.053337
[numbers, A twin] later constant 2-MI type 1 = -0.030815
[numbers, A twin] later constant 2-PI type 0 = +0.051963
[numbers, A twin] later constant 2-PI type 1 = +0.241131
[numbers, A twin] later constant 4-AI type 0 = +0.213238
[numbers, A twin] later constant 4-AI type 1 = +0.141562
[numbers, A twin] later constant 4-FE type 0 = +0.271626
[numbers, A twin] later constant 4-MI type 0 = -0.105378
[numbers, A twin] later constant 4-MI type 1 = +0.126163
[numbers, A twin] later cut origin = -3.432096
[numbers, A twin] later log gap 1 = +0.886611
[numbers, A twin] later log gap 2 = +0.778847
[numbers, A twin] later log gap 3 = +0.380677
[numbers, A twin] weight own = +8.138924
[numbers, A twin] trace fraction own = +0.631056
[open loop] A twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] A twin = +0.1321 [+0.1269, +0.1378]
[closed loop, level] A twin = +0.1752 [+0.1596, +0.1913]
[closed loop, members apart] A twin = +0.1436 [+0.1335, +0.1542]
[closed loop, trend] A twin = +0.1494 [+0.1330, +0.1676]
[closed loop, movement] A twin = +0.0563 [+0.0518, +0.0614]
[closed loop, type gap] A twin = +0.1363 [+0.1263, +0.1465]
[coverage, level] A twin = 0.6048
[coverage, members apart] A twin = 0.6952
[coverage, trend] A twin = 0.8143
[coverage, movement] A twin = 0.4524
[coverage, type gap] A twin = 1.0000
[count] B twin = 20
[numbers, B twin] first constant = +1.320582
[numbers, B twin] first endowment = -0.690753
[numbers, B twin] first return = +0.982805
[numbers, B twin] first product = -0.749621
[numbers, B twin] first four seats = +0.432253
[numbers, B twin] first cut origin = -1.363813
[numbers, B twin] first log gap 1 = +0.759367
[numbers, B twin] first log gap 2 = +0.129722
[numbers, B twin] first log gap 3 = -1.117116
[numbers, B twin] later constant = +1.773358
[numbers, B twin] later endowment = -0.087880
[numbers, B twin] later return = -0.005008
[numbers, B twin] later product = +1.359962
[numbers, B twin] later four seats = -0.013612
[numbers, B twin] later cut origin = -1.816588
[numbers, B twin] later log gap 1 = +0.881953
[numbers, B twin] later log gap 2 = +0.780928
[numbers, B twin] later log gap 3 = +0.380935
[numbers, B twin] weight own = +8.134397
[numbers, B twin] trace fraction own = +0.631901
[open loop] B twin = -0.9553 [-1.0161, -0.8930]
[closed loop, composite] B twin = +0.1311 [+0.1260, +0.1365]
[closed loop, level] B twin = +0.1731 [+0.1586, +0.1880]
[closed loop, members apart] B twin = +0.1484 [+0.1386, +0.1583]
[closed loop, trend] B twin = +0.1465 [+0.1299, +0.1639]
[closed loop, movement] B twin = +0.0559 [+0.0513, +0.0608]
[closed loop, type gap] B twin = +0.1319 [+0.1220, +0.1428]
[coverage, level] B twin = 0.6143
[coverage, members apart] B twin = 0.6667
[coverage, trend] B twin = 0.8619
[coverage, movement] B twin = 0.4429
[coverage, type gap] B twin = 0.9938
[count] C twin = 38
[numbers, C twin] first constant 2-AI type 0 = -0.492417
[numbers, C twin] first constant 2-AI type 1 = -0.115041
[numbers, C twin] first constant 2-EI type 0 = -0.439316
[numbers, C twin] first constant 2-EI type 1 = +0.558254
[numbers, C twin] first constant 2-FE type 0 = +0.223122
[numbers, C twin] first constant 2-MI type 0 = -0.815836
[numbers, C twin] first constant 2-MI type 1 = +0.403859
[numbers, C twin] first constant 2-PI type 0 = +0.231449
[numbers, C twin] first constant 2-PI type 1 = -0.210979
[numbers, C twin] first constant 4-AI type 0 = +0.203070
[numbers, C twin] first constant 4-AI type 1 = +0.312058
[numbers, C twin] first constant 4-FE type 0 = +0.490616
[numbers, C twin] first constant 4-MI type 0 = +0.093892
[numbers, C twin] first constant 4-MI type 1 = +0.812760
[numbers, C twin] first cut origin = -2.742170
[numbers, C twin] first log gap 1 = +0.768593
[numbers, C twin] first log gap 2 = +0.137591
[numbers, C twin] first log gap 3 = -1.114585
[numbers, C twin] later constant 2-AI type 0 = +0.216674
[numbers, C twin] later constant 2-AI type 1 = +0.332657
[numbers, C twin] later constant 2-EI type 0 = +0.039996
[numbers, C twin] later constant 2-EI type 1 = +0.113874
[numbers, C twin] later constant 2-FE type 0 = +0.386062
[numbers, C twin] later constant 2-MI type 0 = -0.053337
[numbers, C twin] later constant 2-MI type 1 = -0.030815
[numbers, C twin] later constant 2-PI type 0 = +0.051963
[numbers, C twin] later constant 2-PI type 1 = +0.241131
[numbers, C twin] later constant 4-AI type 0 = +0.213238
[numbers, C twin] later constant 4-AI type 1 = +0.141562
[numbers, C twin] later constant 4-FE type 0 = +0.271626
[numbers, C twin] later constant 4-MI type 0 = -0.105378
[numbers, C twin] later constant 4-MI type 1 = +0.126163
[numbers, C twin] later cut origin = -3.432096
[numbers, C twin] later log gap 1 = +0.886611
[numbers, C twin] later log gap 2 = +0.778847
[numbers, C twin] later log gap 3 = +0.380677
[numbers, C twin] weight own = +8.138924
[numbers, C twin] trace fraction own = +0.631056
[open loop] C twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] C twin = +0.1321 [+0.1269, +0.1378]
[closed loop, level] C twin = +0.1752 [+0.1596, +0.1913]
[closed loop, members apart] C twin = +0.1436 [+0.1335, +0.1542]
[closed loop, trend] C twin = +0.1494 [+0.1330, +0.1676]
[closed loop, movement] C twin = +0.0563 [+0.0518, +0.0614]
[closed loop, type gap] C twin = +0.1363 [+0.1263, +0.1465]
[coverage, level] C twin = 0.6048
[coverage, members apart] C twin = 0.6952
[coverage, trend] C twin = 0.8143
[coverage, movement] C twin = 0.4524
[coverage, type gap] C twin = 1.0000
[count] D twin = 38
[numbers, D twin] first constant 2-AI type 0 = -0.492417
[numbers, D twin] first constant 2-AI type 1 = -0.115041
[numbers, D twin] first constant 2-EI type 0 = -0.439316
[numbers, D twin] first constant 2-EI type 1 = +0.558254
[numbers, D twin] first constant 2-FE type 0 = +0.223122
[numbers, D twin] first constant 2-MI type 0 = -0.815836
[numbers, D twin] first constant 2-MI type 1 = +0.403859
[numbers, D twin] first constant 2-PI type 0 = +0.231449
[numbers, D twin] first constant 2-PI type 1 = -0.210979
[numbers, D twin] first constant 4-AI type 0 = +0.203070
[numbers, D twin] first constant 4-AI type 1 = +0.312058
[numbers, D twin] first constant 4-FE type 0 = +0.490616
[numbers, D twin] first constant 4-MI type 0 = +0.093892
[numbers, D twin] first constant 4-MI type 1 = +0.812760
[numbers, D twin] first cut origin = -2.742170
[numbers, D twin] first log gap 1 = +0.768593
[numbers, D twin] first log gap 2 = +0.137591
[numbers, D twin] first log gap 3 = -1.114585
[numbers, D twin] later constant 2-AI type 0 = +0.216674
[numbers, D twin] later constant 2-AI type 1 = +0.332657
[numbers, D twin] later constant 2-EI type 0 = +0.039996
[numbers, D twin] later constant 2-EI type 1 = +0.113874
[numbers, D twin] later constant 2-FE type 0 = +0.386062
[numbers, D twin] later constant 2-MI type 0 = -0.053337
[numbers, D twin] later constant 2-MI type 1 = -0.030815
[numbers, D twin] later constant 2-PI type 0 = +0.051963
[numbers, D twin] later constant 2-PI type 1 = +0.241131
[numbers, D twin] later constant 4-AI type 0 = +0.213238
[numbers, D twin] later constant 4-AI type 1 = +0.141562
[numbers, D twin] later constant 4-FE type 0 = +0.271626
[numbers, D twin] later constant 4-MI type 0 = -0.105378
[numbers, D twin] later constant 4-MI type 1 = +0.126163
[numbers, D twin] later cut origin = -3.432096
[numbers, D twin] later log gap 1 = +0.886611
[numbers, D twin] later log gap 2 = +0.778847
[numbers, D twin] later log gap 3 = +0.380677
[numbers, D twin] weight own = +8.138924
[numbers, D twin] trace fraction own = +0.631056
[open loop] D twin = -0.9597 [-1.0204, -0.8962]
[closed loop, composite] D twin = +0.1321 [+0.1269, +0.1378]
[closed loop, level] D twin = +0.1752 [+0.1596, +0.1913]
[closed loop, members apart] D twin = +0.1436 [+0.1335, +0.1542]
[closed loop, trend] D twin = +0.1494 [+0.1330, +0.1676]
[closed loop, movement] D twin = +0.0563 [+0.0518, +0.0614]
[closed loop, type gap] D twin = +0.1363 [+0.1263, +0.1465]
[coverage, level] D twin = 0.6048
[coverage, members apart] D twin = 0.6952
[coverage, trend] D twin = 0.8143
[coverage, movement] D twin = 0.4524
[coverage, type gap] D twin = 1.0000
[count] E twin = 39
[numbers, E twin] first constant 2-AI type 0 = -0.491764
[numbers, E twin] first constant 2-AI type 1 = -0.114269
[numbers, E twin] first constant 2-EI type 0 = -0.439260
[numbers, E twin] first constant 2-EI type 1 = +0.557441
[numbers, E twin] first constant 2-FE type 0 = +0.223151
[numbers, E twin] first constant 2-MI type 0 = -0.814797
[numbers, E twin] first constant 2-MI type 1 = +0.403335
[numbers, E twin] first constant 2-PI type 0 = +0.231283
[numbers, E twin] first constant 2-PI type 1 = -0.210694
[numbers, E twin] first constant 4-AI type 0 = +0.202584
[numbers, E twin] first constant 4-AI type 1 = +0.311769
[numbers, E twin] first constant 4-FE type 0 = +0.490510
[numbers, E twin] first constant 4-MI type 0 = +0.093901
[numbers, E twin] first constant 4-MI type 1 = +0.811626
[numbers, E twin] first cut origin = -2.742009
[numbers, E twin] first log gap 1 = +0.768493
[numbers, E twin] first log gap 2 = +0.137539
[numbers, E twin] first log gap 3 = -1.114841
[numbers, E twin] later constant 2-AI type 0 = +0.219369
[numbers, E twin] later constant 2-AI type 1 = +0.335584
[numbers, E twin] later constant 2-EI type 0 = +0.040558
[numbers, E twin] later constant 2-EI type 1 = +0.113596
[numbers, E twin] later constant 2-FE type 0 = +0.385451
[numbers, E twin] later constant 2-MI type 0 = -0.052029
[numbers, E twin] later constant 2-MI type 1 = -0.031751
[numbers, E twin] later constant 2-PI type 0 = +0.050067
[numbers, E twin] later constant 2-PI type 1 = +0.242019
[numbers, E twin] later constant 4-AI type 0 = +0.213158
[numbers, E twin] later constant 4-AI type 1 = +0.141250
[numbers, E twin] later constant 4-FE type 0 = +0.269650
[numbers, E twin] later constant 4-MI type 0 = -0.105309
[numbers, E twin] later constant 4-MI type 1 = +0.124018
[numbers, E twin] later cut origin = -3.432824
[numbers, E twin] later log gap 1 = +0.886432
[numbers, E twin] later log gap 2 = +0.778797
[numbers, E twin] later log gap 3 = +0.380884
[numbers, E twin] weight own = +8.122233
[numbers, E twin] weight anchor = +0.028822
[numbers, E twin] trace fraction own = +0.632945
[open loop] E twin = -0.9599 [-1.0208, -0.8966]
[closed loop, composite] E twin = +0.1321 [+0.1268, +0.1378]
[closed loop, level] E twin = +0.1750 [+0.1595, +0.1911]
[closed loop, members apart] E twin = +0.1436 [+0.1335, +0.1541]
[closed loop, trend] E twin = +0.1495 [+0.1331, +0.1679]
[closed loop, movement] E twin = +0.0563 [+0.0518, +0.0614]
[closed loop, type gap] E twin = +0.1361 [+0.1263, +0.1463]
[coverage, level] E twin = 0.6143
[coverage, members apart] E twin = 0.7048
[coverage, trend] E twin = 0.8143
[coverage, movement] E twin = 0.4571
[coverage, type gap] E twin = 1.0000
[count] F twin = 36
[numbers, F twin] first constant 2-AI type 0 = -0.492151
[numbers, F twin] first constant 2-AI type 1 = -0.115191
[numbers, F twin] first constant 2-EI type 0 = -0.439002
[numbers, F twin] first constant 2-EI type 1 = +0.557962
[numbers, F twin] first constant 2-FE type 0 = +0.223047
[numbers, F twin] first constant 2-MI type 0 = -0.815378
[numbers, F twin] first constant 2-MI type 1 = +0.403618
[numbers, F twin] first constant 2-PI type 0 = +0.231251
[numbers, F twin] first constant 2-PI type 1 = -0.210875
[numbers, F twin] first constant 4-AI type 0 = +0.203058
[numbers, F twin] first constant 4-AI type 1 = +0.311821
[numbers, F twin] first constant 4-FE type 0 = +0.490928
[numbers, F twin] first constant 4-MI type 0 = +0.093785
[numbers, F twin] first constant 4-MI type 1 = +0.812313
[numbers, F twin] first cut origin = -2.741735
[numbers, F twin] first log gap 1 = +0.768436
[numbers, F twin] first log gap 2 = +0.137546
[numbers, F twin] first log gap 3 = -1.115037
[numbers, F twin] later constant 2-AI type 0 = -0.070386
[numbers, F twin] later constant 2-AI type 1 = +0.471153
[numbers, F twin] later constant 2-EI type 0 = -0.609881
[numbers, F twin] later constant 2-EI type 1 = +0.493811
[numbers, F twin] later constant 2-FE type 0 = +0.407999
[numbers, F twin] later constant 2-MI type 0 = -0.953736
[numbers, F twin] later constant 2-MI type 1 = +0.068395
[numbers, F twin] later constant 2-PI type 0 = -0.220744
[numbers, F twin] later constant 2-PI type 1 = -0.200646
[numbers, F twin] later constant 4-AI type 0 = +0.183274
[numbers, F twin] later constant 4-AI type 1 = +0.592461
[numbers, F twin] later constant 4-FE type 0 = -0.014209
[numbers, F twin] later constant 4-MI type 0 = +0.030773
[numbers, F twin] later constant 4-MI type 1 = +0.594022
[numbers, F twin] later cut origin = -2.258834
[numbers, F twin] later log gap 1 = +0.253094
[numbers, F twin] later log gap 2 = -0.233314
[numbers, F twin] later log gap 3 = -0.572060
[open loop] F twin = -1.4687 [-1.5227, -1.4163]
[closed loop, composite] F twin = +0.1890 [+0.1823, +0.1964]
[closed loop, level] F twin = +0.2226 [+0.2045, +0.2421]
[closed loop, members apart] F twin = +0.1881 [+0.1733, +0.2035]
[closed loop, trend] F twin = +0.1411 [+0.1226, +0.1603]
[closed loop, movement] F twin = +0.2680 [+0.2574, +0.2787]
[closed loop, type gap] F twin = +0.1252 [+0.1089, +0.1432]
[coverage, level] F twin = 0.1381
[coverage, members apart] F twin = 0.2190
[coverage, trend] F twin = 0.6952
[coverage, movement] F twin = 0.0143
[coverage, type gap] F twin = 0.5963
[selection, candidates] highest = D
[selection, candidates] tied at step 2 = A, B, C, D, E
[selection, candidates] winner = B
[selection, twins] highest = B twin
[selection, twins] tied at step 2 = B twin
[selection, twins] winner = B twin
[gain, selected twin, open loop] candidate minus twin = +0.0253 [+0.0179, +0.0328]
[gain, selected twin, closed loop, composite] twin minus candidate = +0.0124 [+0.0085, +0.0163]
[gain, selected twin, closed loop, level] twin minus candidate = +0.0052 [+0.0005, +0.0098]
[gain, selected twin, closed loop, members apart] twin minus candidate = +0.0466 [+0.0355, +0.0571]
[gain, selected twin, closed loop, trend] twin minus candidate = -0.0023 [-0.0045, -0.0001]
[gain, selected twin, closed loop, movement] twin minus candidate = +0.0012 [+0.0004, +0.0020]
[gain, selected twin, closed loop, type gap] twin minus candidate = +0.0115 [+0.0040, +0.0184]
[power, selected twin, H2] training effect = 1.000
[power, selected twin, H2] half effect = 0.782
[power, selected twin, composite] training effect = 0.999
[power, selected twin, composite] half effect = 0.738
[power, selected twin, level] training effect = 0.437
[power, selected twin, level] half effect = 0.147
[power, selected twin, members apart] training effect = 1.000
[power, selected twin, members apart] half effect = 0.939
[power, selected twin, trend] training effect = 0.395
[power, selected twin, trend] half effect = 0.135
[power, selected twin, movement] training effect = 0.627
[power, selected twin, movement] half effect = 0.208
[power, selected twin, type gap] training effect = 0.732
[power, selected twin, type gap] half effect = 0.252
[gain, own twin, open loop] candidate minus twin = +0.0253 [+0.0179, +0.0328]
[gain, own twin, closed loop, composite] twin minus candidate = +0.0124 [+0.0085, +0.0163]
[gain, own twin, closed loop, level] twin minus candidate = +0.0052 [+0.0005, +0.0098]
[gain, own twin, closed loop, members apart] twin minus candidate = +0.0466 [+0.0355, +0.0571]
[gain, own twin, closed loop, trend] twin minus candidate = -0.0023 [-0.0045, -0.0001]
[gain, own twin, closed loop, movement] twin minus candidate = +0.0012 [+0.0004, +0.0020]
[gain, own twin, closed loop, type gap] twin minus candidate = +0.0115 [+0.0040, +0.0184]
[power, own twin, H2] training effect = 1.000
[power, own twin, H2] half effect = 0.782
[power, own twin, composite] training effect = 0.999
[power, own twin, composite] half effect = 0.738
[power, own twin, level] training effect = 0.437
[power, own twin, level] half effect = 0.147
[power, own twin, members apart] training effect = 1.000
[power, own twin, members apart] half effect = 0.939
[power, own twin, trend] training effect = 0.395
[power, own twin, trend] half effect = 0.135
[power, own twin, movement] training effect = 0.627
[power, own twin, movement] half effect = 0.208
[power, own twin, type gap] training effect = 0.732
[power, own twin, type gap] half effect = 0.252
[game left out, candidate] score = -0.9357 [-0.9981, -0.8714]
[game left out, candidate] loss = +0.0058 [+0.0018, +0.0098]
[game left out, candidate, 2-AI] score = -0.8811
[game left out, candidate, 2-EI] score = -0.9194
[game left out, candidate, 2-FE] score = -0.7242
[game left out, candidate, 2-MI] score = -0.9696
[game left out, candidate, 2-PI] score = -0.8604
[game left out, candidate, 4-AI] score = -0.9518
[game left out, candidate, 4-FE] score = -1.1331
[game left out, candidate, 4-MI] score = -1.0728
[game left out, twin] score = -0.9587 [-1.0218, -0.8944]
[game left out, twin] loss = +0.0034 [-0.0003, +0.0072]
[game left out, twin, 2-AI] score = -0.8915
[game left out, twin, 2-EI] score = -0.9203
[game left out, twin, 2-FE] score = -0.7800
[game left out, twin, 2-MI] score = -0.9748
[game left out, twin, 2-PI] score = -0.9047
[game left out, twin, 4-AI] score = -0.9626
[game left out, twin, 4-FE] score = -1.1582
[game left out, twin, 4-MI] score = -1.1012
```
