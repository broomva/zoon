# Can the recommended task test H1? A power calculation on training participants

Run on 2026-11-15, before any pre-registration. The task-selection note names the largest
risk of the recommended task: the effect of state may be too small to detect. This note
replaces that note's rough estimate with a measurement.

**Answer: yes, but only just, and only with all four experiments pooled.** The learning effect
in the human data is about as large as the smallest effect the held-out participants can
detect. No single experiment can detect it. The two courses this leaves are at the end.

What this is and is not. It measures how much a classical learning model gains over the same
model with learning switched off, in predicting choices of people it was not fitted to. No
language model is involved and no part of zoon exists yet. The number is a proxy for what
state could add in the full system, and it says nothing about whether the full system
achieves it.

## Data

Maier, Cheung and Lieder, "Metacognitive learning from consequences of past choices shapes
moral decision-making", Nature Human Behaviour 2025, https://doi.org/10.1038/s41562-025-02271-w.
Data and code at https://osf.io/4up5z/, licence CC-BY 4.0.

**Design, as read from the paper and the authors' scripts.** Each participant answers the
same 13 moral dilemmas, in an order randomised per participant. In each dilemma one option is
recommended by cost-benefit reasoning and the other by a moral rule. After each choice the
participant reads the outcome and rates how good or bad the consequences were. Participants
are assigned to one of two conditions. In one (`cbr_success`) the cost-benefit option is
always followed by the good outcome and the rule option by the bad one; in the other
(`rule_success`) it is the reverse. So the outcome is a fixed function of the choice and the
condition, and what varies from trial to trial is the participant's rating of it.

**All four experiments share this structure.** The dilemmas and outcomes are the same in all
of them. They differ in what follows the 13 trials (questionnaires, donation decisions, a
second session in experiment 4) and, in experiment 2, in the wording of one comprehension
question. They are analysed one by one and pooled.

**Layout of the raw files** (`data_overall.csv`, `dataExp2.csv`, `dataExp3.csv`,
`dataExp4.csv`), worked out from `analysisStudy1.R` and the `data_processing.R` scripts:

| What | Where |
|---|---|
| Participant | The row label in the first, unnamed column. The platform identifiers are blank in the published files |
| Condition | Which set of response columns is filled: `Ladder_Q_YS` is filled in `cbr_success` |
| Choice | `<Dilemma>_Q_<YS or NS>`, "Yes" or "No" to the action the vignette asks about. The suffix says which answer is followed by the good outcome. Whether "Yes" is the cost-benefit option depends on the dilemma |
| Outcome rating | `<Dilemma>_V_<YS or NS>_1`, from -100 to 100, divided by 100 as in the authors' model code |
| Trial number | The position of the dilemma in a Qualtrics display-order column, after removing the attention check. The column differs by experiment and condition: `FL_30_DO` and `FL_31_DO` in experiment 1, `FL_92_DO` and `FL_77_DO` in 2, `FL_36_DO` and `FL_21_DO` in 3, `FL_38_DO` and `FL_23_DO` in 4 |

The authors' exclusion rules are applied: the attention check in every experiment, a second
attention check in experiment 2, and in experiment 4 completion of both sessions and no
earlier participation in the authors' studies. Experiment 1 also had a rule against very
fast completion, which is not applied because its threshold is computed from every
participant's duration; the paper reports that it excluded nobody. The number of
participants left in each experiment ("participants after exclusions" in the output) equals
the final sample size the paper reports.

**Personal data.** The raw files contain columns for platform identifiers, network
addresses, coordinates, recipient fields, demographics and free-text answers. The loader
copies only a fixed list of columns out of each record (row label, exclusion fields, display
order, choices and ratings) and refuses to run if a personal column is added to that list.
Raw files are downloaded into `data/raw/`, which is ignored by git, and checked against the
hashes in `data-manifest.json`.

**The seal.** Participants are assigned to `train` or `held_out` by a hash of a committed
salt, the experiment number and the row label, aiming for the held-out fraction printed in
the output. The loader returns training participants only. For a held-out record it reads
the row label and the exclusion fields, adds one to a count, and reads nothing else. There
is no option, flag or environment variable that returns held-out rows. Every number below
comes from training participants.

## Models

Both models follow the Stan programs in the authors' `CompModels.zip`. The learning model is
`model_free.stan`, a model with one value for each kind of option, started at zero:

```stan
initV = rep_array(0, 2);
target += categorical_logit_lpmf(util_choice[i]|tau * to_vector(v));
pe = value[i] - v[util_choice[i]];
v[util_choice[i]] = v[util_choice[i]] + lr * pe;
```

`value[i]` is the outcome rating divided by 100, `lr` the learning rate and `tau` the
temperature, with the bounds printed in the output. With two options the first line is
`P(cost-benefit option) = sigmoid(tau * (v[cost-benefit] - v[rule]))`. The baseline is
`constant.stan`, a fixed probability:

```stan
target+=bernoulli_lpmf(util_choice[n]|theta);
```

Each model also has a variant with one fitted intercept per dilemma, added to the logit. A
dilemma that was not among the fitting targets takes the share of choices over all fitting
targets, as a logit; the learning model adds the average shift of its fitted intercepts. The
four models are one likelihood with different parts fixed:

`logit P(cost-benefit option on trial t) = intercept[dilemma] + tau * (v[cost-benefit] - v[rule])`

| Model | Intercept | Learning |
|---|---|---|
| constant ("plain") | one, shared | none |
| learning ("plain") | none, as in the paper | `lr` and `tau` fitted |
| constant, dilemma intercepts | one per dilemma | none |
| learning, dilemma intercepts | one per dilemma | `lr` and `tau` fitted |

**How this differs from the paper.**

- The authors fit each participant separately, with priors, by sampling, and compare models
  by each participant's marginal likelihood. Here each model has one set of parameters
  shared by all participants and is fitted by maximum likelihood. This pooling is a
  simplification. It is used because the question is about predicting people the model was
  not fitted to, and it ignores differences between people in how fast they learn.
- Only the paper's model-free learner over the two kinds of option is used. Its model-based
  learner and its learners over action and omission are not.
- The per-dilemma intercepts are an addition and are not in the paper.
- **The plain pair is not nested.** The paper's learning model has no intercept, so before
  any feedback it predicts one half, while the constant model predicts the overall share. The
  two differ on trial 1, where learning cannot have happened. The pair with dilemma
  intercepts is nested: with `tau` at zero the learning model is the constant model. It is
  the cleaner measure of the effect of state and the one the conclusion rests on.

## The two splits

Both are cross-validation inside the training participants, with the number of folds
printed in the output.

- **By participant.** Fit on all folds but one, score the remaining fold.
- **By participant and by dilemma.** As above, and in addition dilemmas are held out two at
  a time, in rotation: each dilemma with its neighbour in a fixed circular order, so that
  every dilemma is held out twice. The models are fitted on the other dilemmas of the fitting
  participants and scored on the held-out dilemmas of the scoring participants.

**What holding out a dilemma means for a model that learns across trials.** A held-out
dilemma is withheld as something to predict and as a source of fitted parameters. Choices on
it add nothing to the fitted likelihood, and it gets no intercept of its own. It is not
withheld as experience. The participant met it somewhere in their own random order, chose,
and saw the outcome, and their later choices depend on that. So the learning model updates
its values on every trial, in the order the participant saw them, held-out dilemmas
included, both in fitting and in scoring. When it predicts a held-out dilemma on trial t it
has used that participant's choices and ratings on the earlier trials and nothing from
trial t.

The alternative, deleting held-out dilemmas from each sequence, was rejected. It would hand
the model a history the participant did not have, and it would test a different model.

Two consequences follow from this definition. Choices that fitting participants made on a
held-out dilemma still shape the value trajectories that the learning rate and temperature
are fitted to; they carry no information that is specific to the dilemma, since neither
parameter is. And the model is told which option of the held-out dilemma is the cost-benefit
one. That label comes from the experimenters and is not learned.

## The quantity and the power calculation

For each scored participant: the mean log-likelihood per choice under the learning model
minus the same under the constant model, in nats, with every probability clamped to the
range printed in the output. Reported are the mean over participants, a 95% percentile
interval from resampling participants, and the standard deviation across participants.

Power is for a paired comparison across participants (a two-sided t-test on the
per-participant differences), using the standard deviation measured on training participants
and the number of held-out participants. Three numbers are reported: the smallest effect
detectable with the target power, the power to detect the effect observed in training, and
the power to detect half of it.

## Results

All four experiments pooled. Every figure is copied from the output at the end.

| Split | Models | Effect, nats per choice | 95% interval | SD | Detectable | Power, observed | Power, half |
|---|---|---|---|---|---|---|---|
| participant | plain | +0.00593 | [+0.00387, +0.00809] | 0.04337 | 0.00467 | 0.945 | 0.428 |
| participant | dilemma intercepts | +0.00952 | [+0.00720, +0.01193] | 0.04859 | 0.00523 | 0.999 | 0.723 |
| participant and dilemma | plain | +0.01986 | [+0.01775, +0.02201] | 0.04395 | 0.00473 | 1.000 | 1.000 |
| participant and dilemma | dilemma intercepts | +0.00571 | [+0.00330, +0.00820] | 0.05047 | 0.00543 | 0.838 | 0.313 |

**The third row is not an effect of state.** Under the dilemma split the plain pair differs
by +0.01595 on trial 1, before any participant has seen an outcome. The learning model starts
at one half, and the constant model's shared probability is fitted on other dilemmas. That
offset accounts for most of the +0.01986. The nested pair shows -0.00061 on trial 1 under the
same split, as it should.

**The effect grows with trial number.** For the nested pair under the participant split it
is +0.00014 on trial 1, +0.00546 on trials 2-7 and +0.01515 on trials 8-13. Under the
dilemma split it is -0.00061, +0.00285 and +0.00962. Learning can only show after feedback,
so trial 1 adds nothing, and about half the trials carry little.

**It depends on which feedback followed which choice.** With each participant's ratings
permuted across their own trials, the effect under the participant split falls to +0.00039
(interval [-0.00094, +0.00175]) for the plain pair and to +0.00215 (interval [+0.00093,
+0.00340]) for the nested pair. A remainder is expected, because the permutation keeps how
often a person chose each option and how good their ratings were overall.

**No single experiment is enough.** For the nested pair, the power to detect the observed
effect is 0.746, 0.386, 0.792 and 0.727 in experiments 1 to 4 under the participant split,
and 0.338, 0.131, 0.288 and 0.294 under the dilemma split.

**The two conditions agree.** For the nested pair under the dilemma split the effect is
+0.00487 (interval [+0.00140, +0.00844]) in `cbr_success` and +0.00653 (interval [+0.00313,
+0.00986]) in `rule_success`. Under the participant split it is +0.01013 and +0.00893.

**Only the product of learning rate and temperature is pinned down.** Fitted to all
training participants, the nested model gives a learning rate of 0.1489 and a temperature of
0.6627, a product of 0.0987. The plain model gives 0.0029 and 20.0000, with the temperature
at its upper bound, a product of 0.0585. The nested model's temperature is also at the bound
in experiments 1 and 3. Values change so little within 13 trials that the two parameters
trade against each other. The predictions, and so the effect, are not sensitive to this; any
reading of the learning rate itself would be.

## Conclusion

Is the state effect large enough that H1 can be tested on this task with the held-out
participants available? Yes, with little margin.

- With all four experiments pooled there are 680 held-out participants. Under the split the
  task needs, by participant and by dilemma, the nested effect is +0.00571 nats per choice
  and the smallest detectable effect is 0.00543. The power is 0.838 if the held-out effect
  equals the training estimate and 0.313 if it is half of it.
- Under the participant split the same comparison has power 0.999, and 0.723 at half.
- Within one experiment the power of the nested comparison is below the target in every
  case.

**How to read the figure for the dilemma split.** It is neither an upper nor a lower bound
on what state adds in the full system.

- It may be too high. These are figures for a model that does nothing but learn two values.
  The full system has to match it through a language model and several conventions, and H1
  asks for more than a difference from zero: a pre-registered smallest difference, for two
  model families.
- It may be too low. Here an unseen dilemma gets a baseline that knows nothing about it. In
  the full system both the `zoon` and the `no-state` arm read the dilemma's text. If that
  reading carries any signal about how the dilemma leans, the comparison moves toward the
  participant split, where the power is 0.999.

**What this implies.** The task can test the state half of H1 only with all four experiments
pooled. Two courses are defensible, and this note does not choose between them:

1. **Pre-register H1 on this task.** Pool all four experiments, compare arms that share the
   same per-dilemma lean, and state the power figures above in the pre-registration. The
   risk is an inconclusive result: at half the training effect the power is 0.313.
2. **Fall back to H1′**, the two-task version in the design. Each of its two tasks is far
   better powered. But the design itself says H1′ shows that each half works where it should
   and does not show that they work together, which is the question the project exists to
   ask.

The choice belongs in the pre-registration. Either way the present note does not change the
design. It supplies the number the task-selection note asked for.

## Limits

About the proxy:

- It is a proxy and not a bound. It is likely optimistic about the full system, for the
  reasons above. It could also be too low: a learner with parameters per person, or the
  paper's model-based learner, which the authors report fits more participants, might gain
  more than this one.
- Parameters are pooled across participants. The paper fits each participant separately.
- The feedback given to the model is the participant's own rating of the outcome, as in the
  paper. That rating is itself a response. A full system would have to produce it or be
  given it. The model was not run on the designed outcome (good or bad) in place of the
  rating.
- Because the outcome is a fixed function of choice and condition, learning from feedback
  cannot be fully separated from consistency of choice. The permutation control removes most
  of the effect, not all.
- In the `no-state` arm of the design, memory and needs are frozen too and a language model
  reads each dilemma. None of that is represented here.

About the dilemma split:

- The interval resamples participants and treats the 13 dilemmas as fixed. It does not
  cover the uncertainty about new dilemmas, and the per-dilemma figures in the output show
  that this uncertainty is large.
- The baseline for an unseen dilemma is the share of choices over the fitting targets. It
  knows nothing about the dilemma. An earlier version of this analysis used the mean of the
  fitted logits instead, which overshoots when a few dilemmas lean heavily one way; that made
  the two conditions disagree, and the pooled figure barely moved when it was corrected.
- Dilemmas are held out in neighbouring pairs in one fixed order. Other pairings would give
  somewhat different numbers. They were not tried.

About the power calculation:

- The standard deviation comes from cross-validation on training participants and is
  assumed to hold for held-out participants.
- "Power for the observed effect" treats the training estimate as the truth. The lower end
  of each interval would give less.
- The paired t-test stands in for the paired bootstrap interval of the H1 protocol. With
  hundreds of participants the two agree closely; this was not checked by simulation.
- Participants scored by the same fitted model share that model's error. The interval
  treats them as independent, which is right for a comparison of fixed fitted models and
  slightly narrow as a statement about refitting.
- In cross-validation each model is fitted on four fifths of the training participants. The
  held-out test would fit on all of them.
- The held-out count assumes every held-out participant who passes the exclusion rules has
  complete trials. That is true of every training participant and cannot be checked for
  held-out ones without reading them.
- Pooling treats the four experiments as one population. Their instructions and what
  followed the task differ.

About the data handling:

- The participant identifier is the row label of the published file, so the split is tied
  to these file versions. The manifest pins their hashes.
- The opaque participant index is a salted hash that keeps row labels out of tables. It is
  not anonymisation: the data and the salt are public.
- **A departure from the seal during development.** Before the identifier column was
  settled, a first version of the split used a column that turned out to be blank, which
  put every row of experiments 1 to 3 in the training set. In that state the display order
  and the response to one dilemma were printed for one record per experiment, and tallies of
  the attention-check answers were printed for all rows. Under the final split the record
  printed from experiment 2 belongs to a held-out participant. No analysis was run in that
  state and nothing was chosen in the light of it. The attention-check fields are ones the
  loader reads for every record by design.

## Reproducing

```
uv sync
uv run pytest
uv run python -m zoon.pilot.report power
```

The last command downloads the four raw files from OSF on first use, checks them against
`data-manifest.json`, and prints the output below. The tests need neither the network nor
the data: they use simulated choices and synthetic raw files, and include two negative
controls (no learning in the data gives no advantage; permuting feedback removes most of
the advantage), a check of the learning update against a hand computation, and the seal.

## Report output

Verbatim, from `uv run python -m zoon.pilot.report power`.

```
zoon pilot: power calculation, training participants only
effect = mean log-likelihood per choice, learning model minus constant model,
         both fitted on other participants; positive favours learning
note: the plain models are not nested (the learning model has no intercept), so
      their difference on trial 1, before any feedback, is not an effect of learning
seed = 20261115
held-out fraction aimed for = 0.3
participant folds = 5
dilemmas = 13
dilemmas held out at a time = 2
times each dilemma is held out = 2
learning rate bounds = [0.0, 1.0]
temperature bounds = [0.0, 20.0]
bootstrap resamples = 10000
probability clamp = [0.001, 0.999]
test = paired t across participants, two-sided alpha = 0.05
target power = 0.8

[experiment 1] participants after exclusions = 387
[experiment 1] training participants = 291
[experiment 1] held-out participants = 96
[experiment 2] participants after exclusions = 380
[experiment 2] training participants = 272
[experiment 2] held-out participants = 108
[experiment 3] participants after exclusions = 834
[experiment 3] training participants = 571
[experiment 3] held-out participants = 263
[experiment 4] participants after exclusions = 727
[experiment 4] training participants = 514
[experiment 4] held-out participants = 213
[pooled] participants after exclusions = 2328
[pooled] training participants = 1648
[pooled] held-out participants = 680
[pooled] training participants, cbr_success = 816
[pooled] training participants, rule_success = 832
[pooled] training choices = 21424
[pooled] feedback range in training data = [-1.00, +1.00]

[experiment 1] constant | fitted P(cost-benefit option) = 0.5231
[experiment 1] learning, plain | fitted learning rate = 0.0039
[experiment 1] learning, plain | fitted temperature = 20.0000
[experiment 1] learning, plain | fitted learning rate x temperature = 0.0779
[experiment 1] learning, plain | in-sample gain over constant = +0.01191 nats/choice
[experiment 1] learning, dilemma intercepts | fitted learning rate = 0.0051
[experiment 1] learning, dilemma intercepts | fitted temperature = 20.0000
[experiment 1] learning, dilemma intercepts | fitted learning rate x temperature = 0.1011
[experiment 1] learning, dilemma intercepts | in-sample gain over constant = +0.01708 nats/choice
[experiment 1] split = participant | models = plain | effect = +0.01147 nats/choice
[experiment 1] split = participant | models = plain | effect 95% interval = [+0.00566, +0.01718]
[experiment 1] split = participant | models = plain | sd across participants = 0.05017
[experiment 1] split = participant | models = plain | detectable effect at 80% power = 0.01449
[experiment 1] split = participant | models = plain | power for the observed effect = 0.601
[experiment 1] split = participant | models = plain | power for half the observed effect = 0.198
[experiment 1] split = participant | models = plain | effect on trial 1 = -0.00012
[experiment 1] split = participant | models = plain | effect on trials 2-7 = +0.00372
[experiment 1] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00092, +0.00816]
[experiment 1] split = participant | models = plain | effect on trials 8-13 = +0.02114
[experiment 1] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.01025, +0.03189]
[experiment 1] split = participant | models = dilemma intercepts | effect = +0.01662 nats/choice
[experiment 1] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00953, +0.02368]
[experiment 1] split = participant | models = dilemma intercepts | sd across participants = 0.06146
[experiment 1] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.01775
[experiment 1] split = participant | models = dilemma intercepts | power for the observed effect = 0.746
[experiment 1] split = participant | models = dilemma intercepts | power for half the observed effect = 0.259
[experiment 1] split = participant | models = dilemma intercepts | effect on trial 1 = -0.00078
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00689
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00179, +0.01200]
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.02924
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.01619, +0.04229]
[experiment 1] split = participant and dilemma | models = plain | effect = +0.02747 nats/choice
[experiment 1] split = participant and dilemma | models = plain | effect 95% interval = [+0.02161, +0.03327]
[experiment 1] split = participant and dilemma | models = plain | sd across participants = 0.05023
[experiment 1] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.01451
[experiment 1] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 1] split = participant and dilemma | models = plain | power for half the observed effect = 0.755
[experiment 1] split = participant and dilemma | models = plain | effect on trial 1 = +0.02005
[experiment 1] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.02019
[experiment 1] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01518, +0.02510]
[experiment 1] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.03598
[experiment 1] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.02503, +0.04684]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect = +0.01006 nats/choice
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00273, +0.01735]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.06331
[experiment 1] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.01829
[experiment 1] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.338
[experiment 1] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.120
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00056
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00317
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00251, +0.00874]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.01873
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00481, +0.03242]

[experiment 2] constant | fitted P(cost-benefit option) = 0.5311
[experiment 2] learning, plain | fitted learning rate = 0.7284
[experiment 2] learning, plain | fitted temperature = 0.2551
[experiment 2] learning, plain | fitted learning rate x temperature = 0.1858
[experiment 2] learning, plain | in-sample gain over constant = +0.00687 nats/choice
[experiment 2] learning, dilemma intercepts | fitted learning rate = 0.6861
[experiment 2] learning, dilemma intercepts | fitted temperature = 0.3268
[experiment 2] learning, dilemma intercepts | fitted learning rate x temperature = 0.2242
[experiment 2] learning, dilemma intercepts | in-sample gain over constant = +0.01116 nats/choice
[experiment 2] split = participant | models = plain | effect = +0.00524 nats/choice
[experiment 2] split = participant | models = plain | effect 95% interval = [-0.00051, +0.01099]
[experiment 2] split = participant | models = plain | sd across participants = 0.04879
[experiment 2] split = participant | models = plain | detectable effect at 80% power = 0.01327
[experiment 2] split = participant | models = plain | power for the observed effect = 0.197
[experiment 2] split = participant | models = plain | power for half the observed effect = 0.086
[experiment 2] split = participant | models = plain | effect on trial 1 = -0.00359
[experiment 2] split = participant | models = plain | effect on trials 2-7 = +0.00760
[experiment 2] split = participant | models = plain | effect on trials 2-7 95% interval = [+0.00095, +0.01430]
[experiment 2] split = participant | models = plain | effect on trials 8-13 = +0.00435
[experiment 2] split = participant | models = plain | effect on trials 8-13 95% interval = [-0.00512, +0.01382]
[experiment 2] split = participant | models = dilemma intercepts | effect = +0.00909 nats/choice
[experiment 2] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00252, +0.01575]
[experiment 2] split = participant | models = dilemma intercepts | sd across participants = 0.05604
[experiment 2] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.01525
[experiment 2] split = participant | models = dilemma intercepts | power for the observed effect = 0.386
[experiment 2] split = participant | models = dilemma intercepts | power for half the observed effect = 0.133
[experiment 2] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00005
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00736
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00019, +0.01460]
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01233
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00161, +0.02315]
[experiment 2] split = participant and dilemma | models = plain | effect = +0.01858 nats/choice
[experiment 2] split = participant and dilemma | models = plain | effect 95% interval = [+0.01282, +0.02439]
[experiment 2] split = participant and dilemma | models = plain | sd across participants = 0.04872
[experiment 2] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.01325
[experiment 2] split = participant and dilemma | models = plain | power for the observed effect = 0.975
[experiment 2] split = participant and dilemma | models = plain | power for half the observed effect = 0.502
[experiment 2] split = participant and dilemma | models = plain | effect on trial 1 = +0.01497
[experiment 2] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.02133
[experiment 2] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01446, +0.02825]
[experiment 2] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.01644
[experiment 2] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.00700, +0.02582]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect = +0.00470 nats/choice
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [-0.00225, +0.01167]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.05867
[experiment 2] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.01596
[experiment 2] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.131
[experiment 2] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.070
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00101
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00640
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00117, +0.01397]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00394
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [-0.00792, +0.01584]

[experiment 3] constant | fitted P(cost-benefit option) = 0.5381
[experiment 3] learning, plain | fitted learning rate = 0.0026
[experiment 3] learning, plain | fitted temperature = 20.0000
[experiment 3] learning, plain | fitted learning rate x temperature = 0.0518
[experiment 3] learning, plain | in-sample gain over constant = +0.00313 nats/choice
[experiment 3] learning, dilemma intercepts | fitted learning rate = 0.0034
[experiment 3] learning, dilemma intercepts | fitted temperature = 20.0000
[experiment 3] learning, dilemma intercepts | fitted learning rate x temperature = 0.0670
[experiment 3] learning, dilemma intercepts | in-sample gain over constant = +0.00807 nats/choice
[experiment 3] split = participant | models = plain | effect = +0.00358 nats/choice
[experiment 3] split = participant | models = plain | effect 95% interval = [-0.00007, +0.00713]
[experiment 3] split = participant | models = plain | sd across participants = 0.04386
[experiment 3] split = participant | models = plain | detectable effect at 80% power = 0.00760
[experiment 3] split = participant | models = plain | power for the observed effect = 0.261
[experiment 3] split = participant | models = plain | power for half the observed effect = 0.101
[experiment 3] split = participant | models = plain | effect on trial 1 = -0.00450
[experiment 3] split = participant | models = plain | effect on trials 2-7 = -0.00147
[experiment 3] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00499, +0.00199]
[experiment 3] split = participant | models = plain | effect on trials 8-13 = +0.00996
[experiment 3] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00362, +0.01628]
[experiment 3] split = participant | models = dilemma intercepts | effect = +0.00788 nats/choice
[experiment 3] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00413, +0.01161]
[experiment 3] split = participant | models = dilemma intercepts | sd across participants = 0.04588
[experiment 3] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00795
[experiment 3] split = participant | models = dilemma intercepts | power for the observed effect = 0.792
[experiment 3] split = participant | models = dilemma intercepts | power for half the observed effect = 0.284
[experiment 3] split = participant | models = dilemma intercepts | effect on trial 1 = -0.00002
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00379
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00114, +0.00643]
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01328
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00636, +0.02015]
[experiment 3] split = participant and dilemma | models = plain | effect = +0.01795 nats/choice
[experiment 3] split = participant and dilemma | models = plain | effect 95% interval = [+0.01422, +0.02159]
[experiment 3] split = participant and dilemma | models = plain | sd across participants = 0.04517
[experiment 3] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00783
[experiment 3] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 3] split = participant and dilemma | models = plain | power for half the observed effect = 0.894
[experiment 3] split = participant and dilemma | models = plain | effect on trial 1 = +0.01123
[experiment 3] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01352
[experiment 3] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.00986, +0.01716]
[experiment 3] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02350
[experiment 3] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.01706, +0.03004]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect = +0.00412 nats/choice
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00028, +0.00802]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.04763
[experiment 3] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00826
[experiment 3] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.288
[experiment 3] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.108
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00044
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00148
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00143, +0.00437]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00752
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00028, +0.01499]

[experiment 4] constant | fitted P(cost-benefit option) = 0.5210
[experiment 4] learning, plain | fitted learning rate = 0.1820
[experiment 4] learning, plain | fitted temperature = 0.4281
[experiment 4] learning, plain | fitted learning rate x temperature = 0.0779
[experiment 4] learning, plain | in-sample gain over constant = +0.00584 nats/choice
[experiment 4] learning, dilemma intercepts | fitted learning rate = 0.3992
[experiment 4] learning, dilemma intercepts | fitted temperature = 0.3446
[experiment 4] learning, dilemma intercepts | fitted learning rate x temperature = 0.1376
[experiment 4] learning, dilemma intercepts | in-sample gain over constant = +0.00856 nats/choice
[experiment 4] split = participant | models = plain | effect = +0.00536 nats/choice
[experiment 4] split = participant | models = plain | effect 95% interval = [+0.00180, +0.00881]
[experiment 4] split = participant | models = plain | sd across participants = 0.04001
[experiment 4] split = participant | models = plain | detectable effect at 80% power = 0.00771
[experiment 4] split = participant | models = plain | power for the observed effect = 0.494
[experiment 4] split = participant | models = plain | power for half the observed effect = 0.164
[experiment 4] split = participant | models = plain | effect on trial 1 = -0.00100
[experiment 4] split = participant | models = plain | effect on trials 2-7 = +0.00147
[experiment 4] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00198, +0.00492]
[experiment 4] split = participant | models = plain | effect on trials 8-13 = +0.01030
[experiment 4] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00423, +0.01630]
[experiment 4] split = participant | models = dilemma intercepts | effect = +0.00784 nats/choice
[experiment 4] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00393, +0.01170]
[experiment 4] split = participant | models = dilemma intercepts | sd across participants = 0.04444
[experiment 4] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00857
[experiment 4] split = participant | models = dilemma intercepts | power for the observed effect = 0.727
[experiment 4] split = participant | models = dilemma intercepts | power for half the observed effect = 0.250
[experiment 4] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00066
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00474
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00074, +0.00875]
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01214
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00567, +0.01863]
[experiment 4] split = participant and dilemma | models = plain | effect = +0.01813 nats/choice
[experiment 4] split = participant and dilemma | models = plain | effect 95% interval = [+0.01457, +0.02160]
[experiment 4] split = participant and dilemma | models = plain | sd across participants = 0.04032
[experiment 4] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00778
[experiment 4] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 4] split = participant and dilemma | models = plain | power for half the observed effect = 0.904
[experiment 4] split = participant and dilemma | models = plain | effect on trial 1 = +0.01815
[experiment 4] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01466
[experiment 4] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01102, +0.01832]
[experiment 4] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02160
[experiment 4] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.01541, +0.02757]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect = +0.00454 nats/choice
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00044, +0.00857]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.04651
[experiment 4] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00897
[experiment 4] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.294
[experiment 4] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.109
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00065
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00147
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00298, +0.00590]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00847
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00163, +0.01511]

[pooled] constant | fitted P(cost-benefit option) = 0.5289
[pooled] learning, plain | fitted learning rate = 0.0029
[pooled] learning, plain | fitted temperature = 20.0000
[pooled] learning, plain | fitted learning rate x temperature = 0.0585
[pooled] learning, plain | in-sample gain over constant = +0.00591 nats/choice
[pooled] learning, dilemma intercepts | fitted learning rate = 0.1489
[pooled] learning, dilemma intercepts | fitted temperature = 0.6627
[pooled] learning, dilemma intercepts | fitted learning rate x temperature = 0.0987
[pooled] learning, dilemma intercepts | in-sample gain over constant = +0.00971 nats/choice
[pooled] split = participant | models = plain | effect = +0.00593 nats/choice
[pooled] split = participant | models = plain | effect 95% interval = [+0.00387, +0.00809]
[pooled] split = participant | models = plain | sd across participants = 0.04337
[pooled] split = participant | models = plain | detectable effect at 80% power = 0.00467
[pooled] split = participant | models = plain | power for the observed effect = 0.945
[pooled] split = participant | models = plain | power for half the observed effect = 0.428
[pooled] split = participant | models = plain | effect on trial 1 = -0.00205
[pooled] split = participant | models = plain | effect on trials 2-7 = +0.00170
[pooled] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00013, +0.00355]
[pooled] split = participant | models = plain | effect on trials 8-13 = +0.01148
[pooled] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00771, +0.01537]
[pooled] split = participant | models = plain | effect on trial 2 = -0.00197
[pooled] split = participant | models = plain | effect on trial 3 = -0.00029
[pooled] split = participant | models = plain | effect on trial 4 = -0.00076
[pooled] split = participant | models = plain | effect on trial 5 = +0.00339
[pooled] split = participant | models = plain | effect on trial 6 = +0.00292
[pooled] split = participant | models = plain | effect on trial 7 = +0.00691
[pooled] split = participant | models = plain | effect on trial 8 = +0.00996
[pooled] split = participant | models = plain | effect on trial 9 = +0.01084
[pooled] split = participant | models = plain | effect on trial 10 = +0.01368
[pooled] split = participant | models = plain | effect on trial 11 = +0.00674
[pooled] split = participant | models = plain | effect on trial 12 = +0.01265
[pooled] split = participant | models = plain | effect on trial 13 = +0.01502
[pooled] split = participant | models = plain | effect, cbr_success = +0.00537
[pooled] split = participant | models = plain | effect, cbr_success 95% interval = [+0.00358, +0.00716]
[pooled] split = participant | models = plain | effect, cbr_success, lowest dilemma = -0.02524
[pooled] split = participant | models = plain | effect, cbr_success, highest dilemma = +0.03356
[pooled] split = participant | models = plain | effect, rule_success = +0.00648
[pooled] split = participant | models = plain | effect, rule_success 95% interval = [+0.00266, +0.01022]
[pooled] split = participant | models = plain | effect, rule_success, lowest dilemma = -0.11720
[pooled] split = participant | models = plain | effect, rule_success, highest dilemma = +0.08827
[pooled] split = participant | models = plain | effect with feedback shuffled within participant = +0.00039
[pooled] split = participant | models = plain | effect with feedback shuffled within participant 95% interval = [-0.00094, +0.00175]
[pooled] split = participant | models = dilemma intercepts | effect = +0.00952 nats/choice
[pooled] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00720, +0.01193]
[pooled] split = participant | models = dilemma intercepts | sd across participants = 0.04859
[pooled] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00523
[pooled] split = participant | models = dilemma intercepts | power for the observed effect = 0.999
[pooled] split = participant | models = dilemma intercepts | power for half the observed effect = 0.723
[pooled] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00014
[pooled] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00546
[pooled] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00349, +0.00744]
[pooled] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01515
[pooled] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.01105, +0.01942]
[pooled] split = participant | models = dilemma intercepts | effect on trial 2 = +0.00024
[pooled] split = participant | models = dilemma intercepts | effect on trial 3 = +0.00228
[pooled] split = participant | models = dilemma intercepts | effect on trial 4 = +0.00304
[pooled] split = participant | models = dilemma intercepts | effect on trial 5 = +0.00660
[pooled] split = participant | models = dilemma intercepts | effect on trial 6 = +0.00963
[pooled] split = participant | models = dilemma intercepts | effect on trial 7 = +0.01096
[pooled] split = participant | models = dilemma intercepts | effect on trial 8 = +0.01094
[pooled] split = participant | models = dilemma intercepts | effect on trial 9 = +0.01516
[pooled] split = participant | models = dilemma intercepts | effect on trial 10 = +0.01645
[pooled] split = participant | models = dilemma intercepts | effect on trial 11 = +0.01413
[pooled] split = participant | models = dilemma intercepts | effect on trial 12 = +0.01743
[pooled] split = participant | models = dilemma intercepts | effect on trial 13 = +0.01682
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success = +0.01013
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success 95% interval = [+0.00683, +0.01351]
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success, lowest dilemma = -0.00123
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success, highest dilemma = +0.02371
[pooled] split = participant | models = dilemma intercepts | effect, rule_success = +0.00893
[pooled] split = participant | models = dilemma intercepts | effect, rule_success 95% interval = [+0.00562, +0.01215]
[pooled] split = participant | models = dilemma intercepts | effect, rule_success, lowest dilemma = -0.00875
[pooled] split = participant | models = dilemma intercepts | effect, rule_success, highest dilemma = +0.02987
[pooled] split = participant | models = dilemma intercepts | effect with feedback shuffled within participant = +0.00215
[pooled] split = participant | models = dilemma intercepts | effect with feedback shuffled within participant 95% interval = [+0.00093, +0.00340]
[pooled] split = participant and dilemma | models = plain | effect = +0.01986 nats/choice
[pooled] split = participant and dilemma | models = plain | effect 95% interval = [+0.01775, +0.02201]
[pooled] split = participant and dilemma | models = plain | sd across participants = 0.04395
[pooled] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00473
[pooled] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[pooled] split = participant and dilemma | models = plain | power for half the observed effect = 1.000
[pooled] split = participant and dilemma | models = plain | effect on trial 1 = +0.01595
[pooled] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01613
[pooled] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01415, +0.01811]
[pooled] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02425
[pooled] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.02042, +0.02815]
[pooled] split = participant and dilemma | models = plain | effect on trial 2 = +0.01439
[pooled] split = participant and dilemma | models = plain | effect on trial 3 = +0.01453
[pooled] split = participant and dilemma | models = plain | effect on trial 4 = +0.01317
[pooled] split = participant and dilemma | models = plain | effect on trial 5 = +0.01635
[pooled] split = participant and dilemma | models = plain | effect on trial 6 = +0.01604
[pooled] split = participant and dilemma | models = plain | effect on trial 7 = +0.02228
[pooled] split = participant and dilemma | models = plain | effect on trial 8 = +0.02406
[pooled] split = participant and dilemma | models = plain | effect on trial 9 = +0.02409
[pooled] split = participant and dilemma | models = plain | effect on trial 10 = +0.02611
[pooled] split = participant and dilemma | models = plain | effect on trial 11 = +0.01779
[pooled] split = participant and dilemma | models = plain | effect on trial 12 = +0.02601
[pooled] split = participant and dilemma | models = plain | effect on trial 13 = +0.02741
[pooled] split = participant and dilemma | models = plain | effect, cbr_success = +0.01918
[pooled] split = participant and dilemma | models = plain | effect, cbr_success 95% interval = [+0.01733, +0.02098]
[pooled] split = participant and dilemma | models = plain | effect, cbr_success, lowest dilemma = -0.01756
[pooled] split = participant and dilemma | models = plain | effect, cbr_success, highest dilemma = +0.07638
[pooled] split = participant and dilemma | models = plain | effect, rule_success = +0.02053
[pooled] split = participant and dilemma | models = plain | effect, rule_success 95% interval = [+0.01669, +0.02433]
[pooled] split = participant and dilemma | models = plain | effect, rule_success, lowest dilemma = -0.08802
[pooled] split = participant and dilemma | models = plain | effect, rule_success, highest dilemma = +0.13457
[pooled] split = participant and dilemma | models = dilemma intercepts | effect = +0.00571 nats/choice
[pooled] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00330, +0.00820]
[pooled] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.05047
[pooled] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00543
[pooled] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.838
[pooled] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.313
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00061
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00285
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00065, +0.00500]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00962
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00528, +0.01415]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 2 = -0.00098
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 3 = +0.00256
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 4 = +0.00010
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 5 = +0.00429
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 6 = +0.00486
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 7 = +0.00627
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 8 = +0.00599
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 9 = +0.00907
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 10 = +0.01365
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 11 = +0.00773
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 12 = +0.00836
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 13 = +0.01294
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success = +0.00487
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success 95% interval = [+0.00140, +0.00844]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success, lowest dilemma = -0.10329
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success, highest dilemma = +0.09414
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success = +0.00653
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success 95% interval = [+0.00313, +0.00986]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success, lowest dilemma = -0.10032
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success, highest dilemma = +0.07745
```
