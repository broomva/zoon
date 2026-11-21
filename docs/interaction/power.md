# Power under the registered rules

This is a training-only scope and record for the registered comparison of the coded
laws. Training groups stand in for held-out groups. The laws are treated as fixed.
Groups are treated as independent.
The closed-loop figures are in-sample for the laws. The calculation reads training
groups and held-out counts; it supplies no result against the prompted arm.

Each trial resamples whole training groups within game with replacement at the
registered held-out counts. The closed-loop calculation draws a fresh paired
ensemble without replacement from a pool simulated once through the kernel. Pool
strings are distinct from the registered simulation strings, and group numbers and
pool indices are shared across arms.

The finite-pool fair-score expectation defines the closed-loop population value.
Coverage checks the interval against that expectation. For the null, each game's
paired score differences are centred; the alternative effects shift the game means
to half and all of the printed training estimate. These are score-location shifts,
with residual variation and pairing kept. Fixed weights make these shifts identical
in the estimate and every interval replicate. Coverage is invariant under the shifts.
The open-loop calculation uses the same centring rule on the out-of-fold differences.

Both the training intervals and each power-trial interval use the full registered
replicate count, generator call order, weights and shared halves. The empirical rates
have sampling error from the trial count and from the finite simulation pool. They
concern one comparison. They do not establish the joint power of the hypotheses.

The table is copied from the labelled output. Every setting is resolved from its
named constant by the program's settings table. The pasted output is a record of the
actual calculation; tests check its labels and settings and exercise the calculation
on small synthetic inputs, without regenerating the slow record.

| Quantity | Result |
|---|---|
| [settings] seed | 20261120 |
| [settings] registered interval replicates | 10000 |
| [settings] power trials | 200 |
| [settings] power interval replicates | 10000 |
| [settings] pool groups per game and arm | 2000 |
| [settings] pool prefix | power-pool |
| [settings] pool draws with replacement | False |
| [settings] simulated groups per game | 100 |
| [settings] half ensemble | 50 |
| [settings] human groups omitted per game | 1 |
| [settings] interval percentiles | (2.5, 97.5) |
| [settings] H2 threshold | 0.0052 |
| [settings] H3 threshold | 0.0031 |
| [settings] scoring floor | 0.005 |
| [settings] rescoring floor | 1e-09 |
| [settings] folds | 5 |
| [settings] rounds | 20 |
| [settings] null effect multiplier | 0.0 |
| [settings] half effect multiplier | 0.5 |
| [settings] training effect multiplier | 1.0 |
| [settings] registered simulation prefix | registered |
| [data] training groups | 210 |
| [data] held-out groups | 143 |
| [training, H2, out of fold] zoon minus alone | 0.0209 [0.0133, 0.0281] |
| [training, H2, out of fold, person floor] zoon minus alone | 0.0253 [0.0178, 0.0326] |
| [training, H3, in-sample laws] alone minus zoon | 0.0140 [0.0091, 0.0189] |
| [H2] coverage at training effect | 0.9500 |
| [H2] pass at zero | 0.0400 |
| [H2] pass at half effect | 0.6850 |
| [H2] pass at training effect | 0.9950 |
| [H3] coverage at training effect | 0.9650 |
| [H3] pass at zero | 0.0100 |
| [H3] pass at half effect | 0.6500 |
| [H3] pass at training effect | 1.0000 |

Output of `python -m zoon.interaction.power`:

```
zoon interaction: registered-rule power on training groups
[settings] seed = 20261120
[settings] registered interval replicates = 10000
[settings] power trials = 200
[settings] power interval replicates = 10000
[settings] pool groups per game and arm = 2000
[settings] pool prefix = power-pool
[settings] pool draws with replacement = False
[settings] simulated groups per game = 100
[settings] half ensemble = 50
[settings] human groups omitted per game = 1
[settings] interval percentiles = (2.5, 97.5)
[settings] H2 threshold = 0.0052
[settings] H3 threshold = 0.0031
[settings] scoring floor = 0.005
[settings] rescoring floor = 1e-09
[settings] folds = 5
[settings] rounds = 20
[settings] null effect multiplier = 0.0
[settings] half effect multiplier = 0.5
[settings] training effect multiplier = 1.0
[settings] registered simulation prefix = registered
[data] training groups = 210
[data] held-out groups = 143
[training, H2, out of fold] zoon minus alone = 0.0209 [0.0133, 0.0281]
[training, H2, out of fold, person floor] zoon minus alone = 0.0253 [0.0178, 0.0326]
[training, H3, in-sample laws] alone minus zoon = 0.0140 [0.0091, 0.0189]
[H2] coverage at training effect = 0.9500
[H2] pass at zero = 0.0400
[H2] pass at half effect = 0.6850
[H2] pass at training effect = 0.9950
[H3] coverage at training effect = 0.9650
[H3] pass at zero = 0.0100
[H3] pass at half effect = 0.6500
[H3] pass at training effect = 1.0000
```
