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
  shared by all participants and is fitted by maximum likelihood, with the paper's prior
  on the temperature kept as a penalty. This pooling is a
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
| participant | plain | +0.00589 | [+0.00383, +0.00803] | 0.04311 | 0.00464 | 0.945 | 0.428 |
| participant | dilemma intercepts | +0.00955 | [+0.00722, +0.01194] | 0.04841 | 0.00521 | 0.999 | 0.728 |
| participant and dilemma | plain | +0.01981 | [+0.01771, +0.02194] | 0.04368 | 0.00470 | 1.000 | 1.000 |
| participant and dilemma | dilemma intercepts | +0.00572 | [+0.00333, +0.00820] | 0.05022 | 0.00540 | 0.843 | 0.317 |

**The third row is not an effect of state.** Under the dilemma split the plain pair differs
by +0.01595 on trial 1, before any participant has seen an outcome. The learning model starts
at one half, and the constant model's shared probability is fitted on other dilemmas. That
offset accounts for most of the +0.01981. The nested pair shows -0.00059 on trial 1 under the
same split, as it should.

**The effect grows with trial number.** For the nested pair under the participant split it
is +0.00014 on trial 1, +0.00551 on trials 2-7 and +0.01515 on trials 8-13. Under the
dilemma split it is -0.00059, +0.00287 and +0.00963. Learning can only show after feedback,
so trial 1 adds nothing, and about half the trials carry little.

**It depends on which feedback followed which choice.** With each participant's ratings
permuted across their own trials, the effect under the participant split falls to +0.00031
(interval [-0.00098, +0.00163]) for the plain pair and to +0.00207 (interval [+0.00089, +0.00327]) for the nested pair. A remainder is expected, because the permutation keeps how
often a person chose each option and how good their ratings were overall.

**No single experiment is enough.** For the nested pair, the power to detect the observed
effect is 0.746, 0.393, 0.792 and 0.737 in experiments 1 to 4 under the participant split,
and 0.321, 0.140, 0.282 and 0.307 under the dilemma split.

**The two conditions agree.** For the nested pair under the dilemma split the effect is
+0.00492 (interval [+0.00146, +0.00846]) in `cbr_success` and +0.00651 (interval [+0.00315, +0.00982]) in `rule_success`. Under the participant split it is +0.01010 and +0.00900.

**Learning rate and temperature are pinned only by the prior.** Over 13 trials the values
move so little that the two parameters trade off: the data fix their product and little
else. The paper's prior on the temperature is kept as a penalty so that the fit lands on one
point. Fitted to all training participants, the nested model gives a learning rate of 0.1836, a
temperature of 0.5713 and a product of 0.1049; the plain model gives 0.0773, 0.8792 and
0.0679. The predictions, and so the effect, depend on the product. Neither parameter should
be read on its own.

## Conclusion

Is the state effect large enough that H1 can be tested on this task with the held-out
participants available? Yes, with little margin.

- With all four experiments pooled there are 680 held-out participants. Under the split the
  task needs, by participant and by dilemma, the nested effect is +0.00572 nats per choice
  and the smallest detectable effect is 0.00540. The power is 0.843 if the held-out effect
  equals the training estimate and 0.317 if it is half of it.
- Under the participant split the same comparison has power 0.999, and 0.728 at half.
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
   risk is an inconclusive result: at half the training effect the power is 0.317.
2. **Fall back to H1′**, the two-task version in the design. No power has been computed for
   its two tasks. The design itself says H1′ shows that each half works where it should and
   does not show that they work together, which is the question the project exists to ask.

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
temperature bounds = [1e-06, 20.0]
temperature prior = lognormal(0, 1.4), as in the paper's model
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
[experiment 1] learning, plain | fitted learning rate = 0.0765
[experiment 1] learning, plain | fitted temperature = 1.1556
[experiment 1] learning, plain | fitted learning rate x temperature = 0.0884
[experiment 1] learning, plain | in-sample gain over constant = +0.01168 nats/choice
[experiment 1] learning, dilemma intercepts | fitted learning rate = 0.0983
[experiment 1] learning, dilemma intercepts | fitted temperature = 1.2208
[experiment 1] learning, dilemma intercepts | fitted learning rate x temperature = 0.1200
[experiment 1] learning, dilemma intercepts | in-sample gain over constant = +0.01688 nats/choice
[experiment 1] split = participant | models = plain | effect = +0.01116 nats/choice
[experiment 1] split = participant | models = plain | effect 95% interval = [+0.00549, +0.01675]
[experiment 1] split = participant | models = plain | sd across participants = 0.04914
[experiment 1] split = participant | models = plain | detectable effect at 80% power = 0.01419
[experiment 1] split = participant | models = plain | power for the observed effect = 0.596
[experiment 1] split = participant | models = plain | power for half the observed effect = 0.196
[experiment 1] split = participant | models = plain | effect on trial 1 = -0.00012
[experiment 1] split = participant | models = plain | effect on trials 2-7 = +0.00367
[experiment 1] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00124, +0.00836]
[experiment 1] split = participant | models = plain | effect on trials 8-13 = +0.02053
[experiment 1] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.01010, +0.03084]
[experiment 1] split = participant | models = dilemma intercepts | effect = +0.01633 nats/choice
[experiment 1] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00933, +0.02329]
[experiment 1] split = participant | models = dilemma intercepts | sd across participants = 0.06041
[experiment 1] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.01745
[experiment 1] split = participant | models = dilemma intercepts | power for the observed effect = 0.746
[experiment 1] split = participant | models = dilemma intercepts | power for half the observed effect = 0.259
[experiment 1] split = participant | models = dilemma intercepts | effect on trial 1 = -0.00083
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00682
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00115, +0.01244]
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.02871
[experiment 1] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.01621, +0.04110]
[experiment 1] split = participant and dilemma | models = plain | effect = +0.02713 nats/choice
[experiment 1] split = participant and dilemma | models = plain | effect 95% interval = [+0.02140, +0.03283]
[experiment 1] split = participant and dilemma | models = plain | sd across participants = 0.04916
[experiment 1] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.01420
[experiment 1] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 1] split = participant and dilemma | models = plain | power for half the observed effect = 0.763
[experiment 1] split = participant and dilemma | models = plain | effect on trial 1 = +0.02005
[experiment 1] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.02014
[experiment 1] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01480, +0.02531]
[experiment 1] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.03529
[experiment 1] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.02488, +0.04561]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect = +0.00964 nats/choice
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00241, +0.01682]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.06249
[experiment 1] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.01805
[experiment 1] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.321
[experiment 1] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.116
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00053
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00260
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00373, +0.00883]
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.01837
[experiment 1] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00524, +0.03138]

[experiment 2] constant | fitted P(cost-benefit option) = 0.5311
[experiment 2] learning, plain | fitted learning rate = 0.7389
[experiment 2] learning, plain | fitted temperature = 0.2527
[experiment 2] learning, plain | fitted learning rate x temperature = 0.1867
[experiment 2] learning, plain | in-sample gain over constant = +0.00687 nats/choice
[experiment 2] learning, dilemma intercepts | fitted learning rate = 0.6992
[experiment 2] learning, dilemma intercepts | fitted temperature = 0.3231
[experiment 2] learning, dilemma intercepts | fitted learning rate x temperature = 0.2259
[experiment 2] learning, dilemma intercepts | in-sample gain over constant = +0.01116 nats/choice
[experiment 2] split = participant | models = plain | effect = +0.00526 nats/choice
[experiment 2] split = participant | models = plain | effect 95% interval = [-0.00047, +0.01099]
[experiment 2] split = participant | models = plain | sd across participants = 0.04853
[experiment 2] split = participant | models = plain | detectable effect at 80% power = 0.01320
[experiment 2] split = participant | models = plain | power for the observed effect = 0.200
[experiment 2] split = participant | models = plain | power for half the observed effect = 0.086
[experiment 2] split = participant | models = plain | effect on trial 1 = -0.00359
[experiment 2] split = participant | models = plain | effect on trials 2-7 = +0.00760
[experiment 2] split = participant | models = plain | effect on trials 2-7 95% interval = [+0.00094, +0.01430]
[experiment 2] split = participant | models = plain | effect on trials 8-13 = +0.00440
[experiment 2] split = participant | models = plain | effect on trials 8-13 95% interval = [-0.00499, +0.01378]
[experiment 2] split = participant | models = dilemma intercepts | effect = +0.00912 nats/choice
[experiment 2] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00259, +0.01572]
[experiment 2] split = participant | models = dilemma intercepts | sd across participants = 0.05564
[experiment 2] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.01514
[experiment 2] split = participant | models = dilemma intercepts | power for the observed effect = 0.393
[experiment 2] split = participant | models = dilemma intercepts | power for half the observed effect = 0.135
[experiment 2] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00006
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00737
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00020, +0.01464]
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01239
[experiment 2] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00179, +0.02308]
[experiment 2] split = participant and dilemma | models = plain | effect = +0.01868 nats/choice
[experiment 2] split = participant and dilemma | models = plain | effect 95% interval = [+0.01293, +0.02445]
[experiment 2] split = participant and dilemma | models = plain | sd across participants = 0.04840
[experiment 2] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.01317
[experiment 2] split = participant and dilemma | models = plain | power for the observed effect = 0.978
[experiment 2] split = participant and dilemma | models = plain | power for half the observed effect = 0.511
[experiment 2] split = participant and dilemma | models = plain | effect on trial 1 = +0.01497
[experiment 2] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.02140
[experiment 2] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01449, +0.02835]
[experiment 2] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.01658
[experiment 2] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.00729, +0.02583]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect = +0.00489 nats/choice
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [-0.00196, +0.01175]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.05793
[experiment 2] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.01576
[experiment 2] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.140
[experiment 2] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.072
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00099
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00645
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00110, +0.01405]
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00431
[experiment 2] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [-0.00727, +0.01597]

[experiment 3] constant | fitted P(cost-benefit option) = 0.5381
[experiment 3] learning, plain | fitted learning rate = 0.0657
[experiment 3] learning, plain | fitted temperature = 0.8830
[experiment 3] learning, plain | fitted learning rate x temperature = 0.0580
[experiment 3] learning, plain | in-sample gain over constant = +0.00302 nats/choice
[experiment 3] learning, dilemma intercepts | fitted learning rate = 0.0684
[experiment 3] learning, dilemma intercepts | fitted temperature = 1.1051
[experiment 3] learning, dilemma intercepts | fitted learning rate x temperature = 0.0756
[experiment 3] learning, dilemma intercepts | in-sample gain over constant = +0.00796 nats/choice
[experiment 3] split = participant | models = plain | effect = +0.00344 nats/choice
[experiment 3] split = participant | models = plain | effect 95% interval = [-0.00015, +0.00694]
[experiment 3] split = participant | models = plain | sd across participants = 0.04319
[experiment 3] split = participant | models = plain | detectable effect at 80% power = 0.00749
[experiment 3] split = participant | models = plain | power for the observed effect = 0.251
[experiment 3] split = participant | models = plain | power for half the observed effect = 0.099
[experiment 3] split = participant | models = plain | effect on trial 1 = -0.00450
[experiment 3] split = participant | models = plain | effect on trials 2-7 = -0.00141
[experiment 3] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00504, +0.00212]
[experiment 3] split = participant | models = plain | effect on trials 8-13 = +0.00962
[experiment 3] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00350, +0.01576]
[experiment 3] split = participant | models = dilemma intercepts | effect = +0.00772 nats/choice
[experiment 3] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00404, +0.01137]
[experiment 3] split = participant | models = dilemma intercepts | sd across participants = 0.04496
[experiment 3] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00780
[experiment 3] split = participant | models = dilemma intercepts | power for the observed effect = 0.792
[experiment 3] split = participant | models = dilemma intercepts | power for half the observed effect = 0.284
[experiment 3] split = participant | models = dilemma intercepts | effect on trial 1 = -0.00002
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00384
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00102, +0.00664]
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01289
[experiment 3] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00621, +0.01947]
[experiment 3] split = participant and dilemma | models = plain | effect = +0.01768 nats/choice
[experiment 3] split = participant and dilemma | models = plain | effect 95% interval = [+0.01403, +0.02126]
[experiment 3] split = participant and dilemma | models = plain | sd across participants = 0.04447
[experiment 3] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00771
[experiment 3] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 3] split = participant and dilemma | models = plain | power for half the observed effect = 0.895
[experiment 3] split = participant and dilemma | models = plain | effect on trial 1 = +0.01123
[experiment 3] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01355
[experiment 3] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.00976, +0.01728]
[experiment 3] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02288
[experiment 3] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.01671, +0.02916]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect = +0.00399 nats/choice
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00022, +0.00781]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.04663
[experiment 3] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00809
[experiment 3] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.282
[experiment 3] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.106
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00042
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00136
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00177, +0.00447]
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00735
[experiment 3] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00043, +0.01447]

[experiment 4] constant | fitted P(cost-benefit option) = 0.5210
[experiment 4] learning, plain | fitted learning rate = 0.2171
[experiment 4] learning, plain | fitted temperature = 0.3793
[experiment 4] learning, plain | fitted learning rate x temperature = 0.0823
[experiment 4] learning, plain | in-sample gain over constant = +0.00584 nats/choice
[experiment 4] learning, dilemma intercepts | fitted learning rate = 0.4102
[experiment 4] learning, dilemma intercepts | fitted temperature = 0.3392
[experiment 4] learning, dilemma intercepts | fitted learning rate x temperature = 0.1391
[experiment 4] learning, dilemma intercepts | in-sample gain over constant = +0.00856 nats/choice
[experiment 4] split = participant | models = plain | effect = +0.00553 nats/choice
[experiment 4] split = participant | models = plain | effect 95% interval = [+0.00203, +0.00894]
[experiment 4] split = participant | models = plain | sd across participants = 0.03947
[experiment 4] split = participant | models = plain | detectable effect at 80% power = 0.00761
[experiment 4] split = participant | models = plain | power for the observed effect = 0.530
[experiment 4] split = participant | models = plain | power for half the observed effect = 0.175
[experiment 4] split = participant | models = plain | effect on trial 1 = -0.00100
[experiment 4] split = participant | models = plain | effect on trials 2-7 = +0.00150
[experiment 4] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00207, +0.00507]
[experiment 4] split = participant | models = plain | effect on trials 8-13 = +0.01066
[experiment 4] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00473, +0.01642]
[experiment 4] split = participant | models = dilemma intercepts | effect = +0.00789 nats/choice
[experiment 4] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00399, +0.01170]
[experiment 4] split = participant | models = dilemma intercepts | sd across participants = 0.04419
[experiment 4] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00852
[experiment 4] split = participant | models = dilemma intercepts | power for the observed effect = 0.737
[experiment 4] split = participant | models = dilemma intercepts | power for half the observed effect = 0.254
[experiment 4] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00066
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00475
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00073, +0.00878]
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01223
[experiment 4] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00584, +0.01867]
[experiment 4] split = participant and dilemma | models = plain | effect = +0.01831 nats/choice
[experiment 4] split = participant and dilemma | models = plain | effect 95% interval = [+0.01480, +0.02174]
[experiment 4] split = participant and dilemma | models = plain | sd across participants = 0.03981
[experiment 4] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00768
[experiment 4] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[experiment 4] split = participant and dilemma | models = plain | power for half the observed effect = 0.916
[experiment 4] split = participant and dilemma | models = plain | effect on trial 1 = +0.01815
[experiment 4] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01468
[experiment 4] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01094, +0.01846]
[experiment 4] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02196
[experiment 4] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.01600, +0.02778]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect = +0.00462 nats/choice
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00056, +0.00862]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.04617
[experiment 4] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00890
[experiment 4] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.307
[experiment 4] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.112
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00065
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00143
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [-0.00306, +0.00589]
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00869
[experiment 4] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00198, +0.01525]

[pooled] constant | fitted P(cost-benefit option) = 0.5289
[pooled] learning, plain | fitted learning rate = 0.0773
[pooled] learning, plain | fitted temperature = 0.8792
[pooled] learning, plain | fitted learning rate x temperature = 0.0679
[pooled] learning, plain | in-sample gain over constant = +0.00588 nats/choice
[pooled] learning, dilemma intercepts | fitted learning rate = 0.1836
[pooled] learning, dilemma intercepts | fitted temperature = 0.5713
[pooled] learning, dilemma intercepts | fitted learning rate x temperature = 0.1049
[pooled] learning, dilemma intercepts | in-sample gain over constant = +0.00971 nats/choice
[pooled] split = participant | models = plain | effect = +0.00589 nats/choice
[pooled] split = participant | models = plain | effect 95% interval = [+0.00383, +0.00803]
[pooled] split = participant | models = plain | sd across participants = 0.04311
[pooled] split = participant | models = plain | detectable effect at 80% power = 0.00464
[pooled] split = participant | models = plain | power for the observed effect = 0.945
[pooled] split = participant | models = plain | power for half the observed effect = 0.428
[pooled] split = participant | models = plain | effect on trial 1 = -0.00205
[pooled] split = participant | models = plain | effect on trials 2-7 = +0.00179
[pooled] split = participant | models = plain | effect on trials 2-7 95% interval = [-0.00016, +0.00374]
[pooled] split = participant | models = plain | effect on trials 8-13 = +0.01131
[pooled] split = participant | models = plain | effect on trials 8-13 95% interval = [+0.00761, +0.01513]
[pooled] split = participant | models = plain | effect on trial 2 = -0.00199
[pooled] split = participant | models = plain | effect on trial 3 = -0.00004
[pooled] split = participant | models = plain | effect on trial 4 = -0.00072
[pooled] split = participant | models = plain | effect on trial 5 = +0.00352
[pooled] split = participant | models = plain | effect on trial 6 = +0.00303
[pooled] split = participant | models = plain | effect on trial 7 = +0.00694
[pooled] split = participant | models = plain | effect on trial 8 = +0.00981
[pooled] split = participant | models = plain | effect on trial 9 = +0.01075
[pooled] split = participant | models = plain | effect on trial 10 = +0.01324
[pooled] split = participant | models = plain | effect on trial 11 = +0.00693
[pooled] split = participant | models = plain | effect on trial 12 = +0.01252
[pooled] split = participant | models = plain | effect on trial 13 = +0.01463
[pooled] split = participant | models = plain | effect, cbr_success = +0.00527
[pooled] split = participant | models = plain | effect, cbr_success 95% interval = [+0.00353, +0.00702]
[pooled] split = participant | models = plain | effect, cbr_success, lowest dilemma = -0.02607
[pooled] split = participant | models = plain | effect, cbr_success, highest dilemma = +0.03463
[pooled] split = participant | models = plain | effect, rule_success = +0.00650
[pooled] split = participant | models = plain | effect, rule_success 95% interval = [+0.00272, +0.01024]
[pooled] split = participant | models = plain | effect, rule_success, lowest dilemma = -0.11805
[pooled] split = participant | models = plain | effect, rule_success, highest dilemma = +0.08920
[pooled] split = participant | models = plain | effect with feedback shuffled within participant = +0.00031
[pooled] split = participant | models = plain | effect with feedback shuffled within participant 95% interval = [-0.00098, +0.00163]
[pooled] split = participant | models = dilemma intercepts | effect = +0.00955 nats/choice
[pooled] split = participant | models = dilemma intercepts | effect 95% interval = [+0.00722, +0.01194]
[pooled] split = participant | models = dilemma intercepts | sd across participants = 0.04841
[pooled] split = participant | models = dilemma intercepts | detectable effect at 80% power = 0.00521
[pooled] split = participant | models = dilemma intercepts | power for the observed effect = 0.999
[pooled] split = participant | models = dilemma intercepts | power for half the observed effect = 0.728
[pooled] split = participant | models = dilemma intercepts | effect on trial 1 = +0.00014
[pooled] split = participant | models = dilemma intercepts | effect on trials 2-7 = +0.00551
[pooled] split = participant | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00347, +0.00754]
[pooled] split = participant | models = dilemma intercepts | effect on trials 8-13 = +0.01515
[pooled] split = participant | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.01111, +0.01936]
[pooled] split = participant | models = dilemma intercepts | effect on trial 2 = +0.00024
[pooled] split = participant | models = dilemma intercepts | effect on trial 3 = +0.00240
[pooled] split = participant | models = dilemma intercepts | effect on trial 4 = +0.00310
[pooled] split = participant | models = dilemma intercepts | effect on trial 5 = +0.00657
[pooled] split = participant | models = dilemma intercepts | effect on trial 6 = +0.00982
[pooled] split = participant | models = dilemma intercepts | effect on trial 7 = +0.01092
[pooled] split = participant | models = dilemma intercepts | effect on trial 8 = +0.01086
[pooled] split = participant | models = dilemma intercepts | effect on trial 9 = +0.01519
[pooled] split = participant | models = dilemma intercepts | effect on trial 10 = +0.01620
[pooled] split = participant | models = dilemma intercepts | effect on trial 11 = +0.01426
[pooled] split = participant | models = dilemma intercepts | effect on trial 12 = +0.01750
[pooled] split = participant | models = dilemma intercepts | effect on trial 13 = +0.01690
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success = +0.01010
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success 95% interval = [+0.00681, +0.01347]
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success, lowest dilemma = -0.00144
[pooled] split = participant | models = dilemma intercepts | effect, cbr_success, highest dilemma = +0.02354
[pooled] split = participant | models = dilemma intercepts | effect, rule_success = +0.00900
[pooled] split = participant | models = dilemma intercepts | effect, rule_success 95% interval = [+0.00568, +0.01220]
[pooled] split = participant | models = dilemma intercepts | effect, rule_success, lowest dilemma = -0.00881
[pooled] split = participant | models = dilemma intercepts | effect, rule_success, highest dilemma = +0.02992
[pooled] split = participant | models = dilemma intercepts | effect with feedback shuffled within participant = +0.00207
[pooled] split = participant | models = dilemma intercepts | effect with feedback shuffled within participant 95% interval = [+0.00089, +0.00327]
[pooled] split = participant and dilemma | models = plain | effect = +0.01981 nats/choice
[pooled] split = participant and dilemma | models = plain | effect 95% interval = [+0.01771, +0.02194]
[pooled] split = participant and dilemma | models = plain | sd across participants = 0.04368
[pooled] split = participant and dilemma | models = plain | detectable effect at 80% power = 0.00470
[pooled] split = participant and dilemma | models = plain | power for the observed effect = 1.000
[pooled] split = participant and dilemma | models = plain | power for half the observed effect = 1.000
[pooled] split = participant and dilemma | models = plain | effect on trial 1 = +0.01595
[pooled] split = participant and dilemma | models = plain | effect on trials 2-7 = +0.01615
[pooled] split = participant and dilemma | models = plain | effect on trials 2-7 95% interval = [+0.01408, +0.01824]
[pooled] split = participant and dilemma | models = plain | effect on trials 8-13 = +0.02410
[pooled] split = participant and dilemma | models = plain | effect on trials 8-13 95% interval = [+0.02038, +0.02791]
[pooled] split = participant and dilemma | models = plain | effect on trial 2 = +0.01435
[pooled] split = participant and dilemma | models = plain | effect on trial 3 = +0.01473
[pooled] split = participant and dilemma | models = plain | effect on trial 4 = +0.01316
[pooled] split = participant and dilemma | models = plain | effect on trial 5 = +0.01638
[pooled] split = participant and dilemma | models = plain | effect on trial 6 = +0.01611
[pooled] split = participant and dilemma | models = plain | effect on trial 7 = +0.02220
[pooled] split = participant and dilemma | models = plain | effect on trial 8 = +0.02393
[pooled] split = participant and dilemma | models = plain | effect on trial 9 = +0.02403
[pooled] split = participant and dilemma | models = plain | effect on trial 10 = +0.02565
[pooled] split = participant and dilemma | models = plain | effect on trial 11 = +0.01796
[pooled] split = participant and dilemma | models = plain | effect on trial 12 = +0.02598
[pooled] split = participant and dilemma | models = plain | effect on trial 13 = +0.02708
[pooled] split = participant and dilemma | models = plain | effect, cbr_success = +0.01901
[pooled] split = participant and dilemma | models = plain | effect, cbr_success 95% interval = [+0.01720, +0.02077]
[pooled] split = participant and dilemma | models = plain | effect, cbr_success, lowest dilemma = -0.01808
[pooled] split = participant and dilemma | models = plain | effect, cbr_success, highest dilemma = +0.07710
[pooled] split = participant and dilemma | models = plain | effect, rule_success = +0.02059
[pooled] split = participant and dilemma | models = plain | effect, rule_success 95% interval = [+0.01677, +0.02437]
[pooled] split = participant and dilemma | models = plain | effect, rule_success, lowest dilemma = -0.08931
[pooled] split = participant and dilemma | models = plain | effect, rule_success, highest dilemma = +0.13545
[pooled] split = participant and dilemma | models = dilemma intercepts | effect = +0.00572 nats/choice
[pooled] split = participant and dilemma | models = dilemma intercepts | effect 95% interval = [+0.00333, +0.00820]
[pooled] split = participant and dilemma | models = dilemma intercepts | sd across participants = 0.05022
[pooled] split = participant and dilemma | models = dilemma intercepts | detectable effect at 80% power = 0.00540
[pooled] split = participant and dilemma | models = dilemma intercepts | power for the observed effect = 0.843
[pooled] split = participant and dilemma | models = dilemma intercepts | power for half the observed effect = 0.317
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 1 = -0.00059
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 = +0.00287
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 2-7 95% interval = [+0.00059, +0.00510]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 = +0.00963
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trials 8-13 95% interval = [+0.00535, +0.01409]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 2 = -0.00095
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 3 = +0.00276
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 4 = +0.00008
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 5 = +0.00423
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 6 = +0.00485
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 7 = +0.00623
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 8 = +0.00578
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 9 = +0.00899
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 10 = +0.01334
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 11 = +0.00795
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 12 = +0.00859
[pooled] split = participant and dilemma | models = dilemma intercepts | effect on trial 13 = +0.01314
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success = +0.00492
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success 95% interval = [+0.00146, +0.00846]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success, lowest dilemma = -0.10350
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, cbr_success, highest dilemma = +0.09540
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success = +0.00651
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success 95% interval = [+0.00315, +0.00982]
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success, lowest dilemma = -0.10074
[pooled] split = participant and dilemma | models = dilemma intercepts | effect, rule_success, highest dilemma = +0.07761

```
