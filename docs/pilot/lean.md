# What could reading a dilemma add? A ceiling from published human norms and codes

Run on 2026-11-15, before any pre-registration and before any language model is involved.

The pilot's `zoon` arm reads each dilemma's text and turns the reading into a lean: how far
people favour the cost-benefit option on that dilemma before any feedback. This note asks
what the best possible reading could add, using published human data in place of a reading.

**Answers.**

- **A lean from published human judgments helps a great deal.** On a dilemma and a
  participant the model was not fitted to, it adds +0.08044 nats per choice over the pooled
  share (95% interval [+0.07452, +0.08642]). That is 0.709 of what knowing the dilemma's own
  share of choices would add.
- **A lean from published codes does not help.** The original paper's codes do not predict
  its own participants' judgments, even among dilemmas of the same set. The cross-validated
  map therefore ignores the codes, and the lean adds +0.00017 (interval [-0.00150, +0.00182]).
  Forced to use the codes, it makes prediction worse, by -0.01836.
- **Learning adds +0.00874 on top of the human-norm lean** (interval [+0.00641, +0.01113]).
  The 680 held-out participants detect an effect of that size with power 0.997, and half of
  it with power 0.646.
- **Reading can be tested on this task, on 13 dilemmas, and the planned route to it is not
  supported.** Details are under "Conclusion".

What this is and is not. No language model is involved. The human-norm lean uses judgments
that people made of nearly the same text, which a reader of the text does not have. It is a
ceiling for what a reading could achieve by this route, not an estimate of what a model will
achieve. Every figure comes from training participants.

## The question

The power note measured what learning adds when a dilemma the model has not seen gets the
pooled share. The task-selection note says what language could add: "how a particular dilemma
leans, read from its text". Thirteen dilemmas are too few to learn how a reading becomes a
lean, so that step has to be learned on other dilemmas. A set of trolley-style dilemmas will
not do: it has no variation on the features that differ among this task's dilemmas.

The 13 dilemmas of this task were adapted from Körner and Deutsch (2022), whose 38 "factual"
dilemmas come with published feature codes and with human ratings from two samples. That
gives a direct test with no model in it:

1. If the lean came from the published ratings, how much would it help?
2. If the lean came from the published codes, how much would it help?
3. How large is the effect of learning on top of each?

If human ratings and codes do not help, no model's reading will.

## Outside data

Körner and Deutsch, "Deontology and Utilitarianism in Real Life: A Set of Moral Dilemmas Based
on Historic Events", Personality and Social Psychology Bulletin 49(10), 1511-1528,
https://doi.org/10.1177/01461672221103058. The article is CC BY 4.0. Materials and data are at
https://osf.io/cg5tq/.

**What the paper measured.** Participants read a dilemma and answered "How appropriate is it
to [perform suggested action]?" on a scale from 1 to 7. Study 1 had 545 German-speaking
participants in a laboratory. Study 2 had 244 participants in the United States, online, and
English texts. Each participant saw a subset. Both studies also rated trolley-type dilemmas:
15 in Study 1 and 20 in Study 2.

**What is committed here, and what is not.**

| File | Content | Source |
|---|---|---|
| `src/zoon/pilot/korner_deutsch_2022.csv` | For each dilemma: the paper's codes and the mean judgment printed for each study | Transcribed from the paper's Tables 1 to 4, 7 and 8. The header of the file names the table behind each column |
| `src/zoon/pilot/task_dilemmas.csv` | For each of the task's 13 dilemmas: its original, and the task authors' own codes | Transcribed from "dilemmas and outcomes.pdf", https://osf.io/download/sgf5y/, in the task's materials (CC BY 4.0) |
| `data-manifest-outside.json` | URL, size and sha256 of the two participant-level files | Recorded on first download, checked afterwards |

The participant-level files are not committed. The OSF project of the outside data states no
licence, so `zoon.pilot.outside.fetch` downloads `dilAll.csv` (Study 1) and `dilEng.csv`
(Study 2) into `data/outside/`, which git ignores.

**The means are computed from the participant-level files**, one row per participant, and not
copied from the paper. They were checked against the paper's tables. The largest difference is
0.0049 in Study 1 and 0.0050 in Study 2. The paper prints two decimals, so rounding allows
0.005: every mean agrees with the paper. The report refuses to run if one does not. The
analysis reads these two files and not the per-session logs they were built from.

**Personal data.** The two files hold a subject number, date, time, age, gender, native
language and free-text comments. The loader reads a cell only if its column holds the
judgment of a dilemma named in the committed table, and raises on any other column. It keeps
a count and a sum per dilemma and nothing per person.

## The mapping

Each of the task's dilemmas maps to one factual dilemma of the original paper. The numbers are
those of the paper's Tables 1 and 2. `s` is +1 where the suggested action is the cost-benefit
option and -1 where it is the rule option; the repository holds it as `CBR_IS_ACTION`.

| Task | Original | s | What changed in the text | Suggested action |
|---|---|---|---|---|
| Ladder | 2, Rope Ladder | +1 | The protagonist is already on deck and is no longer among those saved | Same: push the man off |
| Tyran | 12, Tyrannicide | +1 | The dictator's name is dropped | Same: kill the dictator |
| Lifeboat | 13, Real Lifeboat | +1 | The protagonist is the captain in a second boat and is no longer among those saved; the boat sinks within days, not hours | Same direction: the original throws ten people overboard, the task tells the other boat to |
| Suicide | 15, Assisted Suicide State | +1 | Nothing of substance | Same: legalise |
| Medicine | 17, Medicine Costs | -1 | The cost to other patients is spelled out, with an agency's estimate of more deaths | Same: pay for the medicine |
| RAF | 18, RAF kidnapping | -1 | The group's name is dropped; "many people could get killed" became "killing several people" | Same: release the prisoners |
| Rwanda | 20, Rwanda | -1 | The mob threatens "all the Tutsi citizens inside", no longer "everybody inside" | Same: protect the employee |
| Ransom | 21, (Sahara) Ransom | -1 | "more lives" became "more than 10 lives" | Same: pay the ransom |
| Tax | 29, Tax Fraud Detection | +1 | Adds that buying the data "might promote cybercrime in the future" | Same: buy the data |
| Operations | 27, Knee Operations | +1 | Knee surgery became lung surgery; Germany became a rural area | Same: perform the surgeries |
| Vet | 35, Veterinarian | -1 | The cost of quitting is spelled out: delayed medicines and "many preventable deaths" | Same: quit the job |
| Endow | 38, Endowment | +1 | Small wording | **Reversed.** The original asks about opposing the foundation. The task asks "Do you approve the foundation?" |
| RobinHood | 26, Robin Hood | +1 | Spelling only | Same: redirect the money |

Three things apply to every dilemma:

- The texts compared are the task's and the original paper's English texts. The German sample
  read German texts.
- The question differs. The original asks how appropriate the action is, on seven points. The
  task asks "Do you …?", yes or no.
- The paper numbers three dilemmas differently in its tables of results. Robin Hood is 26 in
  Tables 1 and 2 and 30 in Tables 4 and 8. The committed table is keyed by name.

**The one mapping problem is Endowment.** Its rating is a rating of the opposite action. The
analysis reflects it about the middle of the scale, 8 minus the rating, which is what the
original paper does when it recodes judgments as "utilitarian judgment". That is an
assumption: nobody rated the action the task asks about.

**Adaptation also changed what one code describes.** The original paper codes the protagonist
as among those saved in Rope Ladder, Real Lifeboat and Rwanda. In the task's Rope Ladder and
Real Lifeboat the protagonist is not, and in Rwanda the threat is narrowed. The task authors'
own table has no column for this, so the original code is used, and it is wrong for at least
two of the adapted texts.

## Definitions

One model, with parts switched on or off:

`logit P(cost-benefit option) = s(d) · (alpha + beta · m(d)) + tau · (v[cost-benefit] − v[rule])`

- `m(d)` says how appropriate the suggested action of dilemma `d` is.
- `alpha` is how far people favour the suggested action whatever the dilemma, and `beta` is
  what a unit of `m(d)` adds.
- **The sign multiplies the whole bracket.** Both `alpha` and `beta · m(d)` are about the
  suggested action, so both turn round when the suggested action is the rule option. Signing
  only `beta · m(d)` would be a different model, in which `alpha` favours the cost-benefit
  option and not the suggested action.
- The last term is the learning model of the power note, unchanged: values start at zero, and
  the chosen option's value moves toward the participant's rating of the outcome. With
  `tau = 0` there is no learning.

| Lean | `m(d)` | Fitted on the task |
|---|---|---|
| `pooled` | None. A dilemma that was not fitted takes the share of cost-benefit choices over the fitted dilemmas. This is the existing baseline | One intercept per fitted dilemma |
| `sign` | 0 for every dilemma. Only `s(d)` is known | `alpha` |
| `norm` | The mean rating of the original dilemma in an outside sample | `alpha`, `beta` |
| `coded` | The rating predicted from the original paper's codes | `alpha`, `beta` |
| `known share` | Not a lean a reader could have. Each dilemma's own share among the fitting participants, with no dilemma held out. No lean without learning can beat it | One intercept per dilemma |

`sign` is there because `norm` and `coded` contain it. Without it, an advantage over `pooled`
could come from `s(d)` alone, with `m(d)` adding nothing. A test on simulated data shows
exactly that case.

**Standardising.** `m(d)` is the rating minus its mean over the 13 dilemmas, divided by its
standard deviation over them. The choice of centre does not change any prediction, because
`alpha` is fitted. A rating that is the same for all 13 gives `m(d) = 0`, which is `sign`.

**`norm`.** The US sample is primary and the German sample is the check, as decided before
the analysis. For Endowment the rating is reflected first.

**`coded`.** A ridge regression from the paper's codes to the mean rating.

- **Codes.** Fourteen numbers per dilemma: whether it involves killing; whether the suggested
  action is the utilitarian option; whether the victims are a subset of those who benefit;
  whether the protagonist is among those who benefit; whether the opposite framing is
  reasonable; and nine indicators for the words of the paper's "context" column. The first two
  are the section headings of the paper's Tables 1 and 2.
- **Graded answers.** The published answers are words. Yes is 1 and no is 0. "Rather yes" is
  0.75. "Maybe", "somewhat", "unclear", "?" and one descriptive answer are 0.5. "Probably no",
  "not likely" and "not really" are 0.25. A qualification after the answer is dropped.
- **Not used.** The paper's "norm transgression" is free text that differs for nearly every
  dilemma; it enters only through the killing code. The paper's "feels outdated" is not used:
  the task's own table omits it, and the adaptation removed some dated detail.
- **Fitted on 25 dilemmas.** The paper has 38 factual dilemmas. The 13 that are originals of
  the task's dilemmas are left out, so the map never sees a dilemma it is applied to.
- **Trolley-type dilemmas are left out.** The paper publishes no context or framing codes for
  them. They are all hypothetical killing dilemmas with the action as the utilitarian option,
  so they add no variation on the codes that differ among the task's 13, and the paper shows
  they are judged as less realistic, which it finds goes with different judgments. Including
  them was not tried.
- **Penalty.** Chosen by leave-one-out within the 25, from the list in the output. The list
  includes an infinite penalty, which is the map that ignores the codes and predicts the mean.
- **Applied to the 13** with the original paper's codes of each original, except one: whether
  the suggested action is the utilitarian option is the task's, because that code has to
  describe the action being predicted. So nothing is reflected for `coded`.

## The comparisons

All under the split by participant and by dilemma of the power note: five folds of
participants, and dilemmas held out two at a time in rotation, so that each dilemma is held
out twice. `alpha` and `beta`, and the learning rate and `tau` where the model learns, are
fitted on the fitting participants and on the dilemmas that are not held out. Each choice is
scored by fits that saw neither its participant nor its dilemma. The paper's prior on `tau`
stays as a penalty. The learning model updates its values on every trial, held-out dilemmas
included, as in the power note.

The effect is the mean log-likelihood per choice of one model minus another, per participant.
Reported for each: the mean, a 95% interval from resampling participants, the standard
deviation across participants, the smallest effect detectable at 80% power with the held-out
participants, and the power for the observed effect and for half of it.

One addition. An interval from resampling participants treats the 13 dilemmas as fixed. A
lean is a property of a dilemma, so the report also gives the number of dilemmas on which the
effect is positive and a 95% interval from resampling the 13 dilemmas.

## Results

All four experiments pooled, 1648 training participants, 680 held out. Every figure is copied
from the output at the end.

### The published ratings track the task's choices; the published codes do not

Across the 13 dilemmas, the correlation between `s(d) · m(d)` and the logit of the share of
training participants choosing the cost-benefit option:

| Lean | Correlation |
|---|---|
| `norm`, US sample | +0.891 |
| `norm`, US sample, without Endowment | +0.884 |
| `norm`, German sample | +0.835 |
| `coded`, as cross-validated | undefined: `m(d)` is zero for every dilemma |
| `coded`, penalty forced finite | +0.288 |
| `coded`, penalty forced finite, task authors' codes | +0.368 |
| `s(d)` alone | -0.148 |

### The codes do not predict the ratings, even on the outside dilemmas

| | US sample | German sample |
|---|---|---|
| Standard deviation of the mean rating over the 25 | 0.515 | 0.860 |
| Left-out error, predicting the mean of the other 24 | 0.537 | 0.896 |
| Left-out error of the map, penalty chosen inside each fit | 0.548 | 0.911 |
| Left-out correlation of the map | -0.686 | -0.617 |
| Penalty chosen | infinite | infinite |
| Left-out error at the best finite penalty | 0.538 | 0.899 |

The map predicts a left-out dilemma worse than the mean of the others does, in both samples.
Leave-one-out therefore picks the infinite penalty: ignore the codes. The predicted rating is
then the same for all 13 dilemmas (standard deviation 0.000), `m(d)` is zero, and `coded` is
the same model as `sign`.

The negative correlations are not evidence of a reversed relation. A prediction that carries
no information is close to the mean of the other dilemmas, and that mean is lower when the
left-out dilemma is high. The error is the figure to read.

This agrees with the original paper. Its own comparison of mean judgment across its classes of
dilemma is not significant in either study (its Table 5: F(4, 48) = 0.71; its Table 9:
F(4, 53) = 1.99).

One code cannot be learned at all. The context "Crime" is present in 4 of the task's 13
dilemmas and in 0 of the 25 the map is fitted on.

### Reading, no learning

| Comparison | Effect | 95% interval | SD | Detectable | Power | Power, half | Dilemmas positive | 95% interval, dilemmas |
|---|---|---|---|---|---|---|---|---|
| `norm` − `pooled` | +0.08044 | [+0.07452, +0.08642] | 0.12442 | 0.01339 | 1.000 | 1.000 | 10 of 13 | [+0.02143, +0.14432] |
| `coded` − `pooled` | +0.00017 | [-0.00150, +0.00182] | 0.03500 | 0.00377 | 0.052 | 0.050 | 8 of 13 | [-0.03236, +0.02996] |
| `norm` − `sign` | +0.08027 | [+0.07502, +0.08562] | 0.11041 | 0.01188 | 1.000 | 1.000 | 11 of 13 | [+0.01342, +0.16281] |
| `known share` − `pooled` | +0.11342 | [+0.10753, +0.11927] | 0.12344 | 0.01328 | 1.000 | 1.000 | 12 of 13 | [+0.05550, +0.17778] |

- `norm` gains +0.08044, which is 0.709 of the +0.11342 that knowing each dilemma's own share
  would give.
- `sign` adds nothing by itself: `sign` − `pooled` is +0.00017, and the fitted `alpha` is
  small and unstable (-0.1305 on all training participants; from -0.3407 to +0.0356 across the
  65 cross-validation fits). So the gain of `norm` comes from `m(d)`: `norm` − `sign` is
  +0.08027.
- `beta` is stable. It is +0.8575 fitted to all training participants, and across the 65
  cross-validation fits its mean is +0.8620, its standard deviation 0.0891, and its range
  +0.7352 to +1.1121.
- The gain is uneven. Three dilemmas carry most of it: Tyran (+0.32124), RAF (+0.21990) and
  RobinHood (+0.21078). On Vet the lean is wrong and costs -0.11100: the outside sample rated
  quitting as appropriate, and about half of the task's participants did not quit. Vet is one
  of the dilemmas whose text was changed in adaptation.
- On trials 2 to 13 alone, `norm` − `pooled` is +0.07850.

Checks, each in the output under "check":

- **German sample.** `norm` − `pooled` is +0.07080 (interval [+0.06527, +0.07645]). Smaller
  than with the US sample, and the same conclusion.
- **Codes forced in.** With the best finite penalty, `coded` − `pooled` is -0.01836 (interval
  [-0.02086, -0.01588]) and `coded` − `sign` is -0.01853. With the task authors' own codes of
  their adapted texts it is -0.01776. A lean that correlates weakly with the shares is worse
  than no lean, because `beta` is fitted on eleven dilemmas and then applied to two it has not
  seen.

### Learning on top of a lean

| Comparison | Effect | 95% interval | SD | Detectable | Power | Power, half | Trials 2-13 |
|---|---|---|---|---|---|---|---|
| (`norm` + learning) − `norm` | +0.00874 | [+0.00641, +0.01113] | 0.04878 | 0.00525 | 0.997 | 0.646 | +0.00946 |
| (`coded` + learning) − `coded` | +0.00697 | [+0.00499, +0.00897] | 0.04090 | 0.00440 | 0.993 | 0.602 | +0.00757 |
| (`pooled` + learning) − `pooled` | +0.00572 | [+0.00333, +0.00820] | 0.05022 | 0.00540 | 0.843 | 0.317 | +0.00625 |

- The last row is the nested pair of the power note, reproduced: +0.00572, power 0.843, and
  0.317 at half.
- With a lean that has two fitted numbers in place of one intercept per dilemma, the effect
  of learning is larger and the comparison less noisy, whether or not the lean carries
  anything. The second row is in effect `sign`.
- On trial 1 the effect is +0.00017 for `norm`, as it should be: nobody has seen an outcome.
- By experiment, the power of (`norm` + learning) − `norm` is 0.666, 0.342, 0.678 and 0.606.
  No single experiment reaches the target, as in the power note.

### The whole thing

| Comparison | Effect | 95% interval | SD | Detectable | Power | Power, half | Dilemmas positive | 95% interval, dilemmas |
|---|---|---|---|---|---|---|---|---|
| (`norm` + learning) − (`pooled` + learning) | +0.08346 | [+0.07742, +0.08965] | 0.12727 | 0.01369 | 1.000 | 1.000 | 10 of 13 | [+0.02301, +0.14896] |
| (`coded` + learning) − (`pooled` + learning) | +0.00141 | [-0.00033, +0.00315] | 0.03622 | 0.00390 | 0.173 | 0.080 | 9 of 13 | [-0.03159, +0.03225] |

By experiment, (`norm` + learning) − (`pooled` + learning) is +0.10320, +0.07827, +0.08169 and
+0.07701, with power 1.000 in each and at least 0.898 at half.

## Conclusion

**(a) Does a lean from published human norms help on a held-out dilemma, and by how much?**
Yes. +0.08044 nats per choice over the pooled share, interval [+0.07452, +0.08642] over
participants and [+0.02143, +0.14432] over dilemmas, positive on 10 of 13 dilemmas. It is
about fourteen times the effect of learning measured in the power note, and 0.709 of the
ceiling set by knowing each dilemma's own share.

**(b) Does a lean from published codes help, and by how much?** No. The codes do not predict
the ratings among the outside dilemmas, the cross-validated map ignores them, and the lean
adds +0.00017, interval [-0.00150, +0.00182]. Forced to use the codes, it costs -0.01836.

**(c) How large is the learning effect on top of each, and is it detectable?** +0.00874 on top
of `norm`, with power 0.997, and 0.646 at half the effect. +0.00697 on top of `coded`, with
power 0.993, and 0.602 at half. Both need all four experiments pooled.

**(d) What this implies for the pilot's design.**

1. **Reading can be tested on this task.** There is a great deal to read: `known share` −
   `pooled` is +0.11342, and human judgments of nearly the same text recover 0.709 of it. With
   680 held-out participants, and the spread of `norm` − `pooled`, the smallest detectable
   reading effect is 0.01339, about a sixth of what `norm` achieves.
2. **It would be a test on 13 dilemmas.** The interval over participants says the effect is
   real for these dilemmas. It says little about others. Over dilemmas the interval for
   `norm` − `pooled` runs from +0.02143 to +0.14432, three dilemmas carry most of the gain,
   and one adapted text is enough to cost -0.11100.
3. **The route from these descriptive codes to a lean is not supported.** A reader that
   answered the paper's descriptive questions perfectly would hold the published codes, and
   those codes carry nothing about the lean, here or in the paper's own data. A lean built
   that way would be expected to tie with `laws-only` or lose to it. This says nothing about
   descriptive questions the paper did not code, such as personal force or consent. No
   published set of dilemmas of this kind has both such codes and ratings to test them on.
4. **What does carry the lean is an evaluative judgment**: how appropriate the action is.
   Whether the judge may be asked that is a design question this note does not settle. The
   design forbids asking a model what a person would do. An appropriateness rating is not a
   choice, and it is not a description of the situation either.
5. **Which comparisons are well powered**, with all four experiments pooled:
   - `zoon` against `laws-only`, which is (lean + learning) − (`pooled` + learning): power
     1.000, and 1.000 at half, if the lean is as good as `norm`. If the lean is like `coded`
     there is nothing to detect (power 0.173).
   - `zoon` against `no-state`, which is (lean + learning) − lean: power 0.997, and 0.646 at
     half. This is better than the 0.843 and 0.317 of the power note. It does not depend on
     the lean being good: with `sign` it is 0.993 and 0.602, and with the forced `coded` lean,
     which is worse than no lean, it is 0.980 and 0.519.
   - Within one experiment only the first is well powered.
6. **Reference rows.** `sign`, `norm` and `known share` cost no model calls and bracket any
   reading: `sign` is a reading that carries nothing, `norm` is human judgment of nearly the
   same text, and `known share` is the most a lean can do.

## Limits

About `norm` as a ceiling. It may be too high or too low for what a reader could reach.

- Too high: it uses judgments people made of nearly the same text. A reader of the text does
  not have them.
- Too low: the outside means are noisy (73 to 91 judgments per factual dilemma in the US
  sample), they are of the original texts and not the adapted ones, they answer a different
  question from the task's, and they enter through one linear term. A reader of the adapted
  text could in principle beat `norm`. Nothing without choices on the dilemma can beat `known
  share`.

About the 13 dilemmas.

- Every interval over participants treats the dilemmas as fixed. It is too narrow as a
  statement about new dilemmas.
- The interval over dilemmas resamples 13 values. It is rough, and could be too narrow or too
  wide.
- The texts were changed in adaptation, as the mapping table shows. A change can make the
  outside rating a worse guide, as for Vet, or leave it as good as before.

About Endowment.

- Its rating is reflected about the middle of the scale. If appropriateness of an action and
  of its opposite do not mirror each other, the reflected value is wrong, in either
  direction.
- Without Endowment the correlation with the shares is +0.884 in place of +0.891. The
  cross-validated comparisons were not rerun without it.

About `coded`.

- One coding of the hedged answers, one set of codes and a linear map were tried. Another
  coding, interactions between codes, or the unused free-text norm might find something. With
  14 codes and 25 dilemmas, anything found would be fragile.
- Trolley-type dilemmas were left out. Including them would give more dilemmas to learn the
  killing-related codes from. It would also mix in dilemmas of a kind none of the 13 is.
- The forced map uses the best finite penalty on the list, which is the strongest one. A
  weaker penalty was not run on the task. Leave-one-out error grows as the penalty weakens.
- The code for whether the protagonist benefits is the original's and is wrong for at least
  two adapted texts.

About the samples.

- The US sample was chosen as primary before the analysis. The German sample gives a smaller
  effect (+0.07080) although its ratings spread more across dilemmas. Neither sample is the
  task's population.

About the power figures. These are the limits of the power note, and they apply here.

- The standard deviation comes from cross-validation on training participants and is assumed
  to hold for held-out participants.
- "Power for the observed effect" treats the training estimate as the truth.
- In cross-validation each model is fitted on four fifths of the training participants. The
  held-out test would fit on all of them.
- Power is for a difference from zero. H1 asks for a pre-registered smallest difference, for
  two model families.
- The two arms of each learning comparison share one lean that is the same for every
  participant. In the pilot they would share a model's reading. That could make the
  comparison noisier than here, or leave it as it is; the three leans tried give similar
  figures.

About what was not done.

- No language model was run. Nothing here says how well a model reads a dilemma.
- Only the split by participant and by dilemma was run. Under a split by participant alone a
  fitted intercept per dilemma is available and a lean has nothing to add.

## Reproducing

```
uv sync
uv run pytest
uv run python -m zoon.pilot.report lean
```

The last command downloads the task's four raw files and the two outside files on first use,
checks each against its manifest, checks the outside means against the paper's tables, and
prints the output below. The tests need neither the network nor the data. They use simulated
choices and synthetic files in the published layouts. Among them:

| Behaviour | Test | Line it protects |
|---|---|---|
| Flipping `s` flips the probability of the cost-benefit option | `test_flipping_the_sign_flips_the_probability` | `return sign * (alpha + beta * measure)` in `models.lean` |
| The fit recovers a known `alpha` and `beta` | `test_fit_recovers_alpha_and_beta` | The two gradient lines for `alpha` and `beta` in `models.fit_lean` |
| The fit uses only the dilemmas that are not held out | `test_fit_uses_the_targets_only` | `residual = np.where(targets, ...)` in `models.fit_lean` |
| A lean with signal beats `pooled` and `sign` on held-out dilemmas | `test_a_lean_with_signal_beats_the_pooled_share_on_held_out_dilemmas` | The call to `fit_lean` in `lean.cross_validate` |
| Negative control: with `beta = 0` in the simulator the measure gives no advantage | `test_a_measure_without_signal_gives_no_advantage` | No single line. It guards the comparison as a whole against an advantage that does not come from the measure |
| Held-out dilemmas and the scoring fold stay out of the fit | `test_held_out_dilemmas_and_scoring_participants_stay_out_of_the_fit` | `targets` and `scored` in `lean.cross_validate` |
| The pooled pair is the existing power calculation | `test_pooled_pair_is_the_existing_power_calculation` | `fold = folds(len(data.choice), seed)` in `lean.cross_validate` |
| The map is never fitted on any of the 13 | `test_the_map_is_never_fitted_on_the_tasks_dilemmas` | `and name not in task` in `lean.outside_names` |
| The penalty is chosen without the dilemma being predicted | `test_penalty_is_chosen_without_the_dilemma_being_predicted` | `rest` in `lean.left_out_predictions` |
| A reversed action's rating is reflected | `test_rating_of_a_reversed_action_is_reflected` | The `np.where` in `lean.suggested_action_rating` |
| A map that ignores the codes gives no lean | `test_a_map_that_ignores_the_codes_gives_no_lean` | The early return in `lean.standardise` |
| The seal holds through the new analysis | `test_the_seal_holds_through_the_lean_analysis`, `test_outside_loader_stays_out_of_the_task_data` | That `report` takes participants from `fetch.load` only, and that `outside` reads nothing of the task's files |
| Only judgment columns can be read from the outside files | `test_only_judgment_columns_can_be_read` | The refusal in `outside.allowed_columns` |
| The outside means are checked against the paper | `test_means_are_checked_against_the_paper` | The threshold in `outside.check_against_paper` |
| The pasted output covers every figure quoted in this note | `test_every_quoted_number_is_in_the_report_output` | The entry for this note in `tests/test_docs.py` |
| The manifest check for the outside files | `test_fetch_records_then_verifies_hashes` in `tests/test_outside.py` | The comparison with the manifest in `outside.fetch` |

Each test in this table except the negative control was run with what it protects broken,
and failed.

## Report output

Verbatim, from `uv run python -m zoon.pilot.report lean`.

```
zoon pilot: a lean from published human norms and codes, training participants only
no language model is involved
model: logit P(cost-benefit option) = s(d) * (alpha + beta * m(d)) + tau * (v[1] - v[0])
  s(d)         +1 where the suggested action is the cost-benefit option, -1 where it
               is the rule option; the sign multiplies the whole bracket
  pooled       no lean: a dilemma that was not fitted takes the share over the fitted
               dilemmas (the existing baseline)
  sign         m(d) = 0: only which option the question asks about
  norm         m(d) = mean appropriateness of the original dilemma in an outside
               sample, standardised over the 13; human judgments of nearly the same
               text, which a reader of the text does not have
  coded        m(d) = that judgment predicted from the original paper's codes by a
               ridge map fitted on other dilemmas, standardised over the 13
  known share  each dilemma's own share among the fitting participants; a ceiling,
               not available for a dilemma nobody has answered
  + learning   tau and the learning rate are fitted together with alpha and beta
effect = mean log-likelihood per choice, first model minus second, each choice scored
         by fits that saw neither the participant nor the dilemma (known share: not
         the participant)
seed = 20261115
held-out fraction aimed for = 0.3
participant folds = 5
dilemmas = 13
dilemmas held out at a time = 2
times each dilemma is held out = 2
cross-validation fits per model = 65
learning rate bounds = [0.0, 1.0]
temperature bounds = [1e-06, 20.0]
temperature prior = lognormal(0, 1.4), as in the paper's model
bootstrap resamples = 10000
probability clamp = [0.001, 0.999]
test = paired t across participants, two-sided alpha = 0.05
target power = 0.8
ridge penalties tried = inf, 1000, 300, 100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01

[outside] source = Körner and Deutsch 2022, https://osf.io/cg5tq/
[outside] study 1, German sample | file = https://osf.io/download/72jta/
[outside] study 1, German sample | dilemmas with a judgment = 53
[outside] study 1, German sample | judgments per dilemma = 85 to 366
[outside] study 1, German sample | judgments per factual dilemma = 85 to 107
[outside] study 1, German sample | largest difference between a mean computed from the file and the paper's Tables 3 and 4 = 0.0049 (Immunization (Smallpox))
[outside] study 2, US sample | file = https://osf.io/download/cye64/
[outside] study 2, US sample | dilemmas with a judgment = 58
[outside] study 2, US sample | judgments per dilemma = 73 to 244
[outside] study 2, US sample | judgments per factual dilemma = 73 to 91
[outside] study 2, US sample | largest difference between a mean computed from the file and the paper's Tables 7 and 8 = 0.0050 (Crying Baby)
[outside] difference that rounding to two decimals allows = 0.005

[mapping] Ladder | original = Rope Ladder | s = +1 | suggested action reversed in adaptation = no
[mapping] Tyran | original = Tyrannicide | s = +1 | suggested action reversed in adaptation = no
[mapping] Lifeboat | original = Real Lifeboat | s = +1 | suggested action reversed in adaptation = no
[mapping] Suicide | original = Assisted Suicide State | s = +1 | suggested action reversed in adaptation = no
[mapping] Medicine | original = Medicine Costs | s = -1 | suggested action reversed in adaptation = no
[mapping] RAF | original = RAF kidnapping | s = -1 | suggested action reversed in adaptation = no
[mapping] Rwanda | original = Rwanda | s = -1 | suggested action reversed in adaptation = no
[mapping] Ransom | original = (Sahara) Ransom | s = -1 | suggested action reversed in adaptation = no
[mapping] Tax | original = Tax Fraud Detection | s = +1 | suggested action reversed in adaptation = no
[mapping] Operations | original = Knee Operations | s = +1 | suggested action reversed in adaptation = no
[mapping] Vet | original = Veterinarian | s = -1 | suggested action reversed in adaptation = no
[mapping] Endow | original = Endowment | s = +1 | suggested action reversed in adaptation = yes
[mapping] RobinHood | original = Robin Hood | s = +1 | suggested action reversed in adaptation = no

[coded map] factual dilemmas, with published codes = 38
[coded map] of those, originals of the task's dilemmas, left out = 13
[coded map] trolley-type dilemmas, without published codes, left out = 20
[coded map] dilemmas fitted on = 25
[coded map] fitted on: Conjoined Twins; Cannibalism on Lifeboat; Doodlebugs; Plague Spread; Euthanasia (in Ghetto)
[coded map] fitted on: Immunization (Smallpox); 911; Coma (Mother); Organ Law; Stillbirths
[coded map] fitted on: Septuplets; Atom Bomb; Mare Nostrum; Aid for Syria; Rescue Torture
[coded map] fitted on: War Torture; Rugby Cannibalism; Test Tube Sibling; Organ Market; Ashley Treatment
[coded map] fitted on: Lion Hunt; Bishops; Trail of Tears; Hogesa (Hooligan Demo); Mass Rape
[coded map] codes = 14
[coded map, US sample] sd of the mean judgments of the dilemmas fitted on = 0.515
[coded map, US sample] penalty chosen by leave-one-out = inf
[coded map, US sample] left-out error, predicting the mean of the others = 0.537
[coded map, US sample] left-out error, penalty chosen inside each fit = 0.548
[coded map, US sample] left-out correlation, penalty chosen inside each fit = -0.686
[coded map, US sample] best finite penalty = 1000
[coded map, US sample] left-out error at the best finite penalty = 0.538
[coded map, US sample] left-out correlation at the best finite penalty = -0.913
[coded map, US sample] predicted rating of the task's 13, penalty chosen: sd = 0.000, min = 4.570, max = 4.570
[coded map, US sample] predicted rating of the task's 13, best finite penalty: sd = 0.011, min = 4.548, max = 4.592
[coded map, US sample] code: involves killing | above zero in 14 of the 25 fitted on and 8 of the task's 13 | weight at the best finite penalty = +0.0027
[coded map, US sample] code: action is the utilitarian option | above zero in 19 of the 25 fitted on and 8 of the task's 13 | weight at the best finite penalty = +0.0052
[coded map, US sample] code: victims a subset of the benefited | above zero in 14 of the 25 fitted on and 4 of the task's 13 | weight at the best finite penalty = -0.0050
[coded map, US sample] code: self in the benefited group | above zero in 8 of the 25 fitted on and 5 of the task's 13 | weight at the best finite penalty = -0.0039
[coded map, US sample] code: opposite framing reasonable | above zero in 15 of the 25 fitted on and 8 of the task's 13 | weight at the best finite penalty = -0.0067
[coded map, US sample] code: context: Medicine | above zero in 9 of the 25 fitted on and 3 of the task's 13 | weight at the best finite penalty = +0.0039
[coded map, US sample] code: context: Accident | above zero in 2 of the 25 fitted on and 2 of the task's 13 | weight at the best finite penalty = -0.0025
[coded map, US sample] code: context: War | above zero in 6 of the 25 fitted on and 1 of the task's 13 | weight at the best finite penalty = -0.0008
[coded map, US sample] code: context: Terrorism | above zero in 2 of the 25 fitted on and 4 of the task's 13 | weight at the best finite penalty = +0.0177
[coded map, US sample] code: context: Crime | above zero in 0 of the 25 fitted on and 4 of the task's 13 | weight at the best finite penalty = +0.0000
[coded map, US sample] code: context: Humanitarian Aid | above zero in 2 of the 25 fitted on and 0 of the task's 13 | weight at the best finite penalty = +0.0013
[coded map, US sample] code: context: State authority | above zero in 2 of the 25 fitted on and 0 of the task's 13 | weight at the best finite penalty = +0.0036
[coded map, US sample] code: context: Justice | above zero in 3 of the 25 fitted on and 2 of the task's 13 | weight at the best finite penalty = -0.0084
[coded map, US sample] code: context: Animal rights | above zero in 1 of the 25 fitted on and 1 of the task's 13 | weight at the best finite penalty = -0.0145
[coded map, German sample] sd of the mean judgments of the dilemmas fitted on = 0.860
[coded map, German sample] penalty chosen by leave-one-out = inf
[coded map, German sample] left-out error, predicting the mean of the others = 0.896
[coded map, German sample] left-out error, penalty chosen inside each fit = 0.911
[coded map, German sample] left-out correlation, penalty chosen inside each fit = -0.617
[coded map, German sample] best finite penalty = 1000
[coded map, German sample] left-out error at the best finite penalty = 0.899
[coded map, German sample] left-out correlation at the best finite penalty = -0.923
[coded map, German sample] predicted rating of the task's 13, penalty chosen: sd = 0.000, min = 3.797, max = 3.797
[coded map, German sample] predicted rating of the task's 13, best finite penalty: sd = 0.017, min = 3.761, max = 3.822

[dilemmas] rating = mean appropriateness of the original's suggested action, 1 to 7,
[dilemmas]          computed from the outside files
[dilemmas] m      = the rating of the task's suggested action, standardised over the
[dilemmas]          13; where the action was reversed the rating is reflected first
[dilemmas] share  = share of training participants choosing the cost-benefit option
[dilemmas] dilemma      s  rating US  rating German   m norm  m norm German  m coded  m coded forced   share
[dilemmas] Ladder      +1       4.96           4.58   +0.998         +0.731   +0.000          -0.184   0.589
[dilemmas] Tyran       +1       5.49           5.01   +1.924         +1.196   +0.000          +1.807   0.899
[dilemmas] Lifeboat    +1       3.96           2.54   -0.710         -1.464   +0.000          -0.518   0.325
[dilemmas] Suicide     +1       4.92           5.16   +0.932         +1.357   +0.000          +0.242   0.826
[dilemmas] Medicine    -1       4.31           3.67   -0.109         -0.252   +0.000          -0.068   0.544
[dilemmas] RAF         -1       3.60           2.84   -1.329         -1.145   +0.000          +1.004   0.834
[dilemmas] Rwanda      -1       4.62           4.80   +0.414         +0.973   +0.000          +0.217   0.334
[dilemmas] Ransom      -1       4.46           3.92   +0.148         +0.018   +0.000          +1.004   0.663
[dilemmas] Tax         +1       4.15           4.72   -0.387         +0.879   +0.000          -1.182   0.413
[dilemmas] Operations  +1       4.27           3.49   -0.173         -0.447   +0.000          +0.595   0.425
[dilemmas] Vet         -1       4.92           4.41   +0.932         +0.552   +0.000          -2.079   0.525
[dilemmas] Endow       +1       4.66           4.71   -1.780         -0.660   +0.000          -1.084   0.257
[dilemmas] RobinHood   +1       3.88           2.29   -0.861         -1.737   +0.000          +0.248   0.242
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | norm = +0.891
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | norm, without the reversed dilemma = +0.884
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | coded = undefined, m(d) is zero for every dilemma
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | norm, German sample = +0.835
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | norm, German sample, without the reversed dilemma = +0.837
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | coded, German sample = undefined, m(d) is zero for every dilemma
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | coded, penalty forced finite = +0.288
[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | coded, penalty forced finite, task authors' codes = +0.368
[dilemmas] correlation over the 13 of s(d) with logit share = -0.148

[pooled] training participants = 1648
[pooled] held-out participants = 680
[experiment 1] training participants = 291
[experiment 1] held-out participants = 96
[experiment 2] training participants = 272
[experiment 2] held-out participants = 108
[experiment 3] training participants = 571
[experiment 3] held-out participants = 263
[experiment 4] training participants = 514
[experiment 4] held-out participants = 213

[pooled] norm | alpha, fitted to all of the group = -0.1503
[pooled] norm | alpha over the 65 cross-validation fits: mean = -0.1520, sd = 0.0489, min = -0.2411, max = -0.0590
[pooled] norm | beta, fitted to all of the group = +0.8575
[pooled] norm | beta over the 65 cross-validation fits: mean = +0.8620, sd = 0.0891, min = +0.7352, max = +1.1121
[pooled] norm + learning | alpha, fitted to all of the group = -0.1550
[pooled] norm + learning | alpha over the 65 cross-validation fits: mean = -0.1568, sd = 0.0496, min = -0.2474, max = -0.0619
[pooled] norm + learning | beta, fitted to all of the group = +0.8771
[pooled] norm + learning | beta over the 65 cross-validation fits: mean = +0.8819, sd = 0.0917, min = +0.7519, max = +1.1403
[pooled] norm + learning | fitted learning rate = 0.0990
[pooled] norm + learning | fitted temperature = 0.8680
[pooled] norm + learning | fitted learning rate x temperature = 0.0860
[pooled] sign | alpha, fitted to all of the group = -0.1305
[pooled] sign | alpha over the 65 cross-validation fits: mean = -0.1309, sd = 0.1002, min = -0.3407, max = +0.0356
[pooled] sign + learning | alpha, fitted to all of the group = -0.1340
[pooled] sign + learning | alpha over the 65 cross-validation fits: mean = -0.1343, sd = 0.1018, min = -0.3469, max = +0.0352
[pooled] sign + learning | fitted learning rate = 0.0776
[pooled] sign + learning | fitted temperature = 0.8822
[pooled] sign + learning | fitted learning rate x temperature = 0.0685
[pooled] coded | m(d) is zero for every dilemma: this is `sign`
[pooled] reading, no learning | norm minus pooled | effect = +0.08044 nats/choice
[pooled] reading, no learning | norm minus pooled | effect 95% interval = [+0.07452, +0.08642]
[pooled] reading, no learning | norm minus pooled | sd across participants = 0.12442
[pooled] reading, no learning | norm minus pooled | detectable effect at 80% power = 0.01339
[pooled] reading, no learning | norm minus pooled | power for the observed effect = 1.000
[pooled] reading, no learning | norm minus pooled | power for half the observed effect = 1.000
[pooled] reading, no learning | norm minus pooled | effect on trial 1 = +0.10376
[pooled] reading, no learning | norm minus pooled | effect on trials 2-13 = +0.07850
[pooled] reading, no learning | norm minus pooled | effect on trials 2-13 95% interval = [+0.07235, +0.08475]
[pooled] reading, no learning | norm minus pooled | dilemmas with a positive effect = 10 of 13
[pooled] reading, no learning | norm minus pooled | effect 95% interval, resampling dilemmas = [+0.02143, +0.14432]
[pooled] reading, no learning | coded minus pooled | effect = +0.00017 nats/choice
[pooled] reading, no learning | coded minus pooled | effect 95% interval = [-0.00150, +0.00182]
[pooled] reading, no learning | coded minus pooled | sd across participants = 0.03500
[pooled] reading, no learning | coded minus pooled | detectable effect at 80% power = 0.00377
[pooled] reading, no learning | coded minus pooled | power for the observed effect = 0.052
[pooled] reading, no learning | coded minus pooled | power for half the observed effect = 0.050
[pooled] reading, no learning | coded minus pooled | effect on trial 1 = +0.00232
[pooled] reading, no learning | coded minus pooled | effect on trials 2-13 = -0.00001
[pooled] reading, no learning | coded minus pooled | effect on trials 2-13 95% interval = [-0.00177, +0.00176]
[pooled] reading, no learning | coded minus pooled | dilemmas with a positive effect = 8 of 13
[pooled] reading, no learning | coded minus pooled | effect 95% interval, resampling dilemmas = [-0.03236, +0.02996]
[pooled] reading, no learning | norm minus sign | effect = +0.08027 nats/choice
[pooled] reading, no learning | norm minus sign | effect 95% interval = [+0.07502, +0.08562]
[pooled] reading, no learning | norm minus sign | sd across participants = 0.11041
[pooled] reading, no learning | norm minus sign | detectable effect at 80% power = 0.01188
[pooled] reading, no learning | norm minus sign | power for the observed effect = 1.000
[pooled] reading, no learning | norm minus sign | power for half the observed effect = 1.000
[pooled] reading, no learning | norm minus sign | effect on trial 1 = +0.10144
[pooled] reading, no learning | norm minus sign | effect on trials 2-13 = +0.07851
[pooled] reading, no learning | norm minus sign | effect on trials 2-13 95% interval = [+0.07302, +0.08420]
[pooled] reading, no learning | norm minus sign | dilemmas with a positive effect = 11 of 13
[pooled] reading, no learning | norm minus sign | effect 95% interval, resampling dilemmas = [+0.01342, +0.16281]
[pooled] reading, no learning | coded minus sign | effect = +0.00000 nats/choice
[pooled] reading, no learning | coded minus sign | the two are the same model, so there is nothing to detect
[pooled] reading, no learning | known share minus pooled | effect = +0.11342 nats/choice
[pooled] reading, no learning | known share minus pooled | effect 95% interval = [+0.10753, +0.11927]
[pooled] reading, no learning | known share minus pooled | sd across participants = 0.12344
[pooled] reading, no learning | known share minus pooled | detectable effect at 80% power = 0.01328
[pooled] reading, no learning | known share minus pooled | power for the observed effect = 1.000
[pooled] reading, no learning | known share minus pooled | power for half the observed effect = 1.000
[pooled] reading, no learning | known share minus pooled | effect on trial 1 = +0.12423
[pooled] reading, no learning | known share minus pooled | effect on trials 2-13 = +0.11252
[pooled] reading, no learning | known share minus pooled | effect on trials 2-13 95% interval = [+0.10637, +0.11875]
[pooled] reading, no learning | known share minus pooled | dilemmas with a positive effect = 12 of 13
[pooled] reading, no learning | known share minus pooled | effect 95% interval, resampling dilemmas = [+0.05550, +0.17778]
[pooled] reading, no learning | sign minus pooled | effect = +0.00017 nats/choice
[pooled] reading, no learning | sign minus pooled | effect 95% interval = [-0.00150, +0.00182]
[pooled] reading, no learning | sign minus pooled | power for the observed effect = 0.052
[pooled] reading, no learning | sign minus pooled | power for half the observed effect = 0.050
[pooled] reading, no learning | sign minus pooled | effect on trials 2-13 = -0.00001
[pooled] learning on top of a lean | norm + learning minus norm | effect = +0.00874 nats/choice
[pooled] learning on top of a lean | norm + learning minus norm | effect 95% interval = [+0.00641, +0.01113]
[pooled] learning on top of a lean | norm + learning minus norm | sd across participants = 0.04878
[pooled] learning on top of a lean | norm + learning minus norm | detectable effect at 80% power = 0.00525
[pooled] learning on top of a lean | norm + learning minus norm | power for the observed effect = 0.997
[pooled] learning on top of a lean | norm + learning minus norm | power for half the observed effect = 0.646
[pooled] learning on top of a lean | norm + learning minus norm | effect on trial 1 = +0.00017
[pooled] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 = +0.00946
[pooled] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 95% interval = [+0.00693, +0.01204]
[pooled] learning on top of a lean | norm + learning minus norm | dilemmas with a positive effect = 12 of 13
[pooled] learning on top of a lean | norm + learning minus norm | effect 95% interval, resampling dilemmas = [+0.00414, +0.01355]
[pooled] learning on top of a lean | coded + learning minus coded | effect = +0.00697 nats/choice
[pooled] learning on top of a lean | coded + learning minus coded | effect 95% interval = [+0.00499, +0.00897]
[pooled] learning on top of a lean | coded + learning minus coded | sd across participants = 0.04090
[pooled] learning on top of a lean | coded + learning minus coded | detectable effect at 80% power = 0.00440
[pooled] learning on top of a lean | coded + learning minus coded | power for the observed effect = 0.993
[pooled] learning on top of a lean | coded + learning minus coded | power for half the observed effect = 0.602
[pooled] learning on top of a lean | coded + learning minus coded | effect on trial 1 = -0.00024
[pooled] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 = +0.00757
[pooled] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 95% interval = [+0.00543, +0.00974]
[pooled] learning on top of a lean | coded + learning minus coded | dilemmas with a positive effect = 9 of 13
[pooled] learning on top of a lean | coded + learning minus coded | effect 95% interval, resampling dilemmas = [+0.00232, +0.01179]
[pooled] learning on top of a lean | pooled + learning minus pooled | effect = +0.00572 nats/choice
[pooled] learning on top of a lean | pooled + learning minus pooled | effect 95% interval = [+0.00333, +0.00820]
[pooled] learning on top of a lean | pooled + learning minus pooled | sd across participants = 0.05022
[pooled] learning on top of a lean | pooled + learning minus pooled | detectable effect at 80% power = 0.00540
[pooled] learning on top of a lean | pooled + learning minus pooled | power for the observed effect = 0.843
[pooled] learning on top of a lean | pooled + learning minus pooled | power for half the observed effect = 0.317
[pooled] learning on top of a lean | pooled + learning minus pooled | effect on trial 1 = -0.00059
[pooled] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 = +0.00625
[pooled] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 95% interval = [+0.00365, +0.00893]
[pooled] learning on top of a lean | pooled + learning minus pooled | dilemmas with a positive effect = 9 of 13
[pooled] learning on top of a lean | pooled + learning minus pooled | effect 95% interval, resampling dilemmas = [-0.00059, +0.01205]
[pooled] learning on top of a lean | sign + learning minus sign | effect = +0.00697 nats/choice
[pooled] learning on top of a lean | sign + learning minus sign | effect 95% interval = [+0.00499, +0.00897]
[pooled] learning on top of a lean | sign + learning minus sign | power for the observed effect = 0.993
[pooled] learning on top of a lean | sign + learning minus sign | power for half the observed effect = 0.602
[pooled] learning on top of a lean | sign + learning minus sign | effect on trials 2-13 = +0.00757
[pooled] the whole thing | norm + learning minus pooled + learning | effect = +0.08346 nats/choice
[pooled] the whole thing | norm + learning minus pooled + learning | effect 95% interval = [+0.07742, +0.08965]
[pooled] the whole thing | norm + learning minus pooled + learning | sd across participants = 0.12727
[pooled] the whole thing | norm + learning minus pooled + learning | detectable effect at 80% power = 0.01369
[pooled] the whole thing | norm + learning minus pooled + learning | power for the observed effect = 1.000
[pooled] the whole thing | norm + learning minus pooled + learning | power for half the observed effect = 1.000
[pooled] the whole thing | norm + learning minus pooled + learning | effect on trial 1 = +0.10452
[pooled] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 = +0.08170
[pooled] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 95% interval = [+0.07534, +0.08820]
[pooled] the whole thing | norm + learning minus pooled + learning | dilemmas with a positive effect = 10 of 13
[pooled] the whole thing | norm + learning minus pooled + learning | effect 95% interval, resampling dilemmas = [+0.02301, +0.14896]
[pooled] the whole thing | coded + learning minus pooled + learning | effect = +0.00141 nats/choice
[pooled] the whole thing | coded + learning minus pooled + learning | effect 95% interval = [-0.00033, +0.00315]
[pooled] the whole thing | coded + learning minus pooled + learning | sd across participants = 0.03622
[pooled] the whole thing | coded + learning minus pooled + learning | detectable effect at 80% power = 0.00390
[pooled] the whole thing | coded + learning minus pooled + learning | power for the observed effect = 0.173
[pooled] the whole thing | coded + learning minus pooled + learning | power for half the observed effect = 0.080
[pooled] the whole thing | coded + learning minus pooled + learning | effect on trial 1 = +0.00267
[pooled] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 = +0.00130
[pooled] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 95% interval = [-0.00053, +0.00313]
[pooled] the whole thing | coded + learning minus pooled + learning | dilemmas with a positive effect = 9 of 13
[pooled] the whole thing | coded + learning minus pooled + learning | effect 95% interval, resampling dilemmas = [-0.03159, +0.03225]
[pooled] the whole thing | norm + learning minus sign + learning | effect = +0.08205 nats/choice
[pooled] the whole thing | norm + learning minus sign + learning | effect 95% interval = [+0.07674, +0.08750]
[pooled] the whole thing | norm + learning minus sign + learning | power for the observed effect = 1.000
[pooled] the whole thing | norm + learning minus sign + learning | power for half the observed effect = 1.000
[pooled] the whole thing | norm + learning minus sign + learning | effect on trials 2-13 = +0.08040

[pooled] (norm minus pooled) as a share of (known share minus pooled) = 0.709

[pooled, by dilemma] dilemma      norm - pooled  norm - sign  known share - pooled  norm + learning - norm
[pooled, by dilemma] Ladder            -0.00433     +0.02646              +0.01057                +0.02399
[pooled, by dilemma] Tyran             +0.32124     +0.44818              +0.36173                +0.00739
[pooled, by dilemma] Lifeboat          +0.07495     +0.02946              +0.07455                +0.02202
[pooled, by dilemma] Suicide           +0.11893     +0.21534              +0.21600                +0.00620
[pooled, by dilemma] Medicine          +0.00252     +0.00062              +0.00309                +0.00919
[pooled, by dilemma] RAF               +0.21990     +0.21420              +0.22969                +0.00004
[pooled, by dilemma] Rwanda            +0.04596     +0.04947              +0.07453                +0.00807
[pooled, by dilemma] Ransom            -0.01987     -0.01818              +0.03452                +0.00476
[pooled, by dilemma] Tax               +0.02987     +0.00741              +0.03092                +0.01179
[pooled, by dilemma] Operations        +0.02802     +0.00434              +0.02702                +0.02179
[pooled, by dilemma] Vet               -0.11100     -0.11159              -0.00012                +0.00348
[pooled, by dilemma] Endow             +0.12872     +0.05085              +0.19691                -0.00661
[pooled, by dilemma] RobinHood         +0.21078     +0.12697              +0.21512                +0.00157

[pooled] check | coded, German sample | m(d) is zero for every dilemma: this is `sign`
[pooled] check | norm, German sample minus pooled | effect = +0.07080 nats/choice
[pooled] check | norm, German sample minus pooled | effect 95% interval = [+0.06527, +0.07645]
[pooled] check | norm, German sample minus pooled | power for the observed effect = 1.000
[pooled] check | norm, German sample minus pooled | power for half the observed effect = 1.000
[pooled] check | norm, German sample minus pooled | effect on trials 2-13 = +0.07065
[pooled] check | norm, German sample minus sign | effect = +0.07063 nats/choice
[pooled] check | norm, German sample minus sign | effect 95% interval = [+0.06586, +0.07544]
[pooled] check | norm, German sample minus sign | power for the observed effect = 1.000
[pooled] check | norm, German sample minus sign | power for half the observed effect = 1.000
[pooled] check | norm, German sample minus sign | effect on trials 2-13 = +0.07066
[pooled] check | norm, German sample + learning minus norm, German sample | effect = +0.00857 nats/choice
[pooled] check | norm, German sample + learning minus norm, German sample | effect 95% interval = [+0.00627, +0.01090]
[pooled] check | norm, German sample + learning minus norm, German sample | power for the observed effect = 0.997
[pooled] check | norm, German sample + learning minus norm, German sample | power for half the observed effect = 0.647
[pooled] check | norm, German sample + learning minus norm, German sample | effect on trials 2-13 = +0.00929
[pooled] check | norm, German sample + learning minus pooled + learning | effect = +0.07365 nats/choice
[pooled] check | norm, German sample + learning minus pooled + learning | effect 95% interval = [+0.06797, +0.07939]
[pooled] check | norm, German sample + learning minus pooled + learning | power for the observed effect = 1.000
[pooled] check | norm, German sample + learning minus pooled + learning | power for half the observed effect = 1.000
[pooled] check | norm, German sample + learning minus pooled + learning | effect on trials 2-13 = +0.07370
[pooled] check | coded, penalty forced finite minus pooled | effect = -0.01836 nats/choice
[pooled] check | coded, penalty forced finite minus pooled | effect 95% interval = [-0.02086, -0.01588]
[pooled] check | coded, penalty forced finite minus pooled | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite minus pooled | power for half the observed effect = 0.997
[pooled] check | coded, penalty forced finite minus pooled | effect on trials 2-13 = -0.01877
[pooled] check | coded, penalty forced finite minus sign | effect = -0.01853 nats/choice
[pooled] check | coded, penalty forced finite minus sign | effect 95% interval = [-0.02025, -0.01683]
[pooled] check | coded, penalty forced finite minus sign | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite minus sign | power for half the observed effect = 1.000
[pooled] check | coded, penalty forced finite minus sign | effect on trials 2-13 = -0.01876
[pooled] check | coded, penalty forced finite + learning minus coded, penalty forced finite | effect = +0.00655 nats/choice
[pooled] check | coded, penalty forced finite + learning minus coded, penalty forced finite | effect 95% interval = [+0.00450, +0.00862]
[pooled] check | coded, penalty forced finite + learning minus coded, penalty forced finite | power for the observed effect = 0.980
[pooled] check | coded, penalty forced finite + learning minus coded, penalty forced finite | power for half the observed effect = 0.519
[pooled] check | coded, penalty forced finite + learning minus coded, penalty forced finite | effect on trials 2-13 = +0.00715
[pooled] check | coded, penalty forced finite + learning minus pooled + learning | effect = -0.01754 nats/choice
[pooled] check | coded, penalty forced finite + learning minus pooled + learning | effect 95% interval = [-0.02007, -0.01505]
[pooled] check | coded, penalty forced finite + learning minus pooled + learning | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite + learning minus pooled + learning | power for half the observed effect = 0.993
[pooled] check | coded, penalty forced finite + learning minus pooled + learning | effect on trials 2-13 = -0.01787
[pooled] check | coded, penalty forced finite, task authors' codes minus pooled | effect = -0.01776 nats/choice
[pooled] check | coded, penalty forced finite, task authors' codes minus pooled | effect 95% interval = [-0.02088, -0.01471]
[pooled] check | coded, penalty forced finite, task authors' codes minus pooled | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite, task authors' codes minus pooled | power for half the observed effect = 0.955
[pooled] check | coded, penalty forced finite, task authors' codes minus pooled | effect on trials 2-13 = -0.01818
[pooled] check | coded, penalty forced finite, task authors' codes minus sign | effect = -0.01792 nats/choice
[pooled] check | coded, penalty forced finite, task authors' codes minus sign | effect 95% interval = [-0.02046, -0.01545]
[pooled] check | coded, penalty forced finite, task authors' codes minus sign | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite, task authors' codes minus sign | power for half the observed effect = 0.994
[pooled] check | coded, penalty forced finite, task authors' codes minus sign | effect on trials 2-13 = -0.01816
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus coded, penalty forced finite, task authors' codes | effect = +0.00649 nats/choice
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus coded, penalty forced finite, task authors' codes | effect 95% interval = [+0.00442, +0.00860]
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus coded, penalty forced finite, task authors' codes | power for the observed effect = 0.975
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus coded, penalty forced finite, task authors' codes | power for half the observed effect = 0.501
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus coded, penalty forced finite, task authors' codes | effect on trials 2-13 = +0.00711
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus pooled + learning | effect = -0.01699 nats/choice
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus pooled + learning | effect 95% interval = [-0.02019, -0.01392]
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus pooled + learning | power for the observed effect = 1.000
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus pooled + learning | power for half the observed effect = 0.927
[pooled] check | coded, penalty forced finite, task authors' codes + learning minus pooled + learning | effect on trials 2-13 = -0.01732

[experiment 1] reading, no learning | norm minus pooled | effect = +0.09816 nats/choice
[experiment 1] reading, no learning | norm minus pooled | effect 95% interval = [+0.08371, +0.11267]
[experiment 1] reading, no learning | norm minus pooled | power for the observed effect = 1.000
[experiment 1] reading, no learning | norm minus pooled | power for half the observed effect = 0.968
[experiment 1] reading, no learning | norm minus pooled | effect on trials 2-13 = +0.09656
[experiment 1] reading, no learning | coded minus pooled | effect = +0.00221 nats/choice
[experiment 1] reading, no learning | coded minus pooled | effect 95% interval = [-0.00123, +0.00575]
[experiment 1] reading, no learning | coded minus pooled | power for the observed effect = 0.110
[experiment 1] reading, no learning | coded minus pooled | power for half the observed effect = 0.065
[experiment 1] reading, no learning | coded minus pooled | effect on trials 2-13 = +0.00210
[experiment 1] reading, no learning | norm minus sign | effect = +0.09594 nats/choice
[experiment 1] reading, no learning | norm minus sign | effect 95% interval = [+0.08273, +0.10894]
[experiment 1] reading, no learning | norm minus sign | power for the observed effect = 1.000
[experiment 1] reading, no learning | norm minus sign | power for half the observed effect = 0.984
[experiment 1] reading, no learning | norm minus sign | effect on trials 2-13 = +0.09446
[experiment 1] reading, no learning | coded minus sign | effect = +0.00000 nats/choice
[experiment 1] reading, no learning | coded minus sign | the two are the same model, so there is nothing to detect
[experiment 1] reading, no learning | known share minus pooled | effect = +0.12396 nats/choice
[experiment 1] reading, no learning | known share minus pooled | effect 95% interval = [+0.10883, +0.13843]
[experiment 1] reading, no learning | known share minus pooled | power for the observed effect = 1.000
[experiment 1] reading, no learning | known share minus pooled | power for half the observed effect = 0.997
[experiment 1] reading, no learning | known share minus pooled | effect on trials 2-13 = +0.12511
[experiment 1] learning on top of a lean | norm + learning minus norm | effect = +0.01468 nats/choice
[experiment 1] learning on top of a lean | norm + learning minus norm | effect 95% interval = [+0.00782, +0.02154]
[experiment 1] learning on top of a lean | norm + learning minus norm | power for the observed effect = 0.666
[experiment 1] learning on top of a lean | norm + learning minus norm | power for half the observed effect = 0.223
[experiment 1] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 = +0.01590
[experiment 1] learning on top of a lean | coded + learning minus coded | effect = +0.01152 nats/choice
[experiment 1] learning on top of a lean | coded + learning minus coded | effect 95% interval = [+0.00593, +0.01718]
[experiment 1] learning on top of a lean | coded + learning minus coded | power for the observed effect = 0.624
[experiment 1] learning on top of a lean | coded + learning minus coded | power for half the observed effect = 0.206
[experiment 1] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 = +0.01251
[experiment 1] learning on top of a lean | pooled + learning minus pooled | effect = +0.00964 nats/choice
[experiment 1] learning on top of a lean | pooled + learning minus pooled | effect 95% interval = [+0.00241, +0.01682]
[experiment 1] learning on top of a lean | pooled + learning minus pooled | power for the observed effect = 0.321
[experiment 1] learning on top of a lean | pooled + learning minus pooled | power for half the observed effect = 0.116
[experiment 1] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 = +0.01048
[experiment 1] the whole thing | norm + learning minus pooled + learning | effect = +0.10320 nats/choice
[experiment 1] the whole thing | norm + learning minus pooled + learning | effect 95% interval = [+0.08806, +0.11831]
[experiment 1] the whole thing | norm + learning minus pooled + learning | power for the observed effect = 1.000
[experiment 1] the whole thing | norm + learning minus pooled + learning | power for half the observed effect = 0.971
[experiment 1] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 = +0.10197
[experiment 1] the whole thing | coded + learning minus pooled + learning | effect = +0.00410 nats/choice
[experiment 1] the whole thing | coded + learning minus pooled + learning | effect 95% interval = [+0.00041, +0.00792]
[experiment 1] the whole thing | coded + learning minus pooled + learning | power for the observed effect = 0.230
[experiment 1] the whole thing | coded + learning minus pooled + learning | power for half the observed effect = 0.094
[experiment 1] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 = +0.00413

[experiment 2] reading, no learning | norm minus pooled | effect = +0.07491 nats/choice
[experiment 2] reading, no learning | norm minus pooled | effect 95% interval = [+0.06032, +0.08924]
[experiment 2] reading, no learning | norm minus pooled | power for the observed effect = 1.000
[experiment 2] reading, no learning | norm minus pooled | power for half the observed effect = 0.886
[experiment 2] reading, no learning | norm minus pooled | effect on trials 2-13 = +0.07222
[experiment 2] reading, no learning | coded minus pooled | effect = -0.00070 nats/choice
[experiment 2] reading, no learning | coded minus pooled | effect 95% interval = [-0.00530, +0.00384]
[experiment 2] reading, no learning | coded minus pooled | power for the observed effect = 0.054
[experiment 2] reading, no learning | coded minus pooled | power for half the observed effect = 0.051
[experiment 2] reading, no learning | coded minus pooled | effect on trials 2-13 = -0.00078
[experiment 2] reading, no learning | norm minus sign | effect = +0.07561 nats/choice
[experiment 2] reading, no learning | norm minus sign | effect 95% interval = [+0.06303, +0.08829]
[experiment 2] reading, no learning | norm minus sign | power for the observed effect = 1.000
[experiment 2] reading, no learning | norm minus sign | power for half the observed effect = 0.951
[experiment 2] reading, no learning | norm minus sign | effect on trials 2-13 = +0.07300
[experiment 2] reading, no learning | coded minus sign | effect = +0.00000 nats/choice
[experiment 2] reading, no learning | coded minus sign | the two are the same model, so there is nothing to detect
[experiment 2] reading, no learning | known share minus pooled | effect = +0.10939 nats/choice
[experiment 2] reading, no learning | known share minus pooled | effect 95% interval = [+0.09447, +0.12399]
[experiment 2] reading, no learning | known share minus pooled | power for the observed effect = 1.000
[experiment 2] reading, no learning | known share minus pooled | power for half the observed effect = 0.995
[experiment 2] reading, no learning | known share minus pooled | effect on trials 2-13 = +0.10619
[experiment 2] learning on top of a lean | norm + learning minus norm | effect = +0.00825 nats/choice
[experiment 2] learning on top of a lean | norm + learning minus norm | effect 95% interval = [+0.00186, +0.01466]
[experiment 2] learning on top of a lean | norm + learning minus norm | power for the observed effect = 0.342
[experiment 2] learning on top of a lean | norm + learning minus norm | power for half the observed effect = 0.121
[experiment 2] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 = +0.00890
[experiment 2] learning on top of a lean | coded + learning minus coded | effect = +0.00671 nats/choice
[experiment 2] learning on top of a lean | coded + learning minus coded | effect 95% interval = [+0.00125, +0.01224]
[experiment 2] learning on top of a lean | coded + learning minus coded | power for the observed effect = 0.312
[experiment 2] learning on top of a lean | coded + learning minus coded | power for half the observed effect = 0.114
[experiment 2] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 = +0.00730
[experiment 2] learning on top of a lean | pooled + learning minus pooled | effect = +0.00489 nats/choice
[experiment 2] learning on top of a lean | pooled + learning minus pooled | effect 95% interval = [-0.00196, +0.01175]
[experiment 2] learning on top of a lean | pooled + learning minus pooled | power for the observed effect = 0.140
[experiment 2] learning on top of a lean | pooled + learning minus pooled | power for half the observed effect = 0.072
[experiment 2] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 = +0.00538
[experiment 2] the whole thing | norm + learning minus pooled + learning | effect = +0.07827 nats/choice
[experiment 2] the whole thing | norm + learning minus pooled + learning | effect 95% interval = [+0.06353, +0.09304]
[experiment 2] the whole thing | norm + learning minus pooled + learning | power for the observed effect = 1.000
[experiment 2] the whole thing | norm + learning minus pooled + learning | power for half the observed effect = 0.898
[experiment 2] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 = +0.07574
[experiment 2] the whole thing | coded + learning minus pooled + learning | effect = +0.00113 nats/choice
[experiment 2] the whole thing | coded + learning minus pooled + learning | effect 95% interval = [-0.00373, +0.00580]
[experiment 2] the whole thing | coded + learning minus pooled + learning | power for the observed effect = 0.060
[experiment 2] the whole thing | coded + learning minus pooled + learning | power for half the observed effect = 0.052
[experiment 2] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 = +0.00114

[experiment 3] reading, no learning | norm minus pooled | effect = +0.07880 nats/choice
[experiment 3] reading, no learning | norm minus pooled | effect 95% interval = [+0.06762, +0.08976]
[experiment 3] reading, no learning | norm minus pooled | power for the observed effect = 1.000
[experiment 3] reading, no learning | norm minus pooled | power for half the observed effect = 0.997
[experiment 3] reading, no learning | norm minus pooled | effect on trials 2-13 = +0.07905
[experiment 3] reading, no learning | coded minus pooled | effect = -0.00086 nats/choice
[experiment 3] reading, no learning | coded minus pooled | effect 95% interval = [-0.00426, +0.00259]
[experiment 3] reading, no learning | coded minus pooled | power for the observed effect = 0.063
[experiment 3] reading, no learning | coded minus pooled | power for half the observed effect = 0.053
[experiment 3] reading, no learning | coded minus pooled | effect on trials 2-13 = -0.00063
[experiment 3] reading, no learning | norm minus sign | effect = +0.07966 nats/choice
[experiment 3] reading, no learning | norm minus sign | effect 95% interval = [+0.07016, +0.08881]
[experiment 3] reading, no learning | norm minus sign | power for the observed effect = 1.000
[experiment 3] reading, no learning | norm minus sign | power for half the observed effect = 1.000
[experiment 3] reading, no learning | norm minus sign | effect on trials 2-13 = +0.07968
[experiment 3] reading, no learning | coded minus sign | effect = +0.00000 nats/choice
[experiment 3] reading, no learning | coded minus sign | the two are the same model, so there is nothing to detect
[experiment 3] reading, no learning | known share minus pooled | effect = +0.11457 nats/choice
[experiment 3] reading, no learning | known share minus pooled | effect 95% interval = [+0.10438, +0.12446]
[experiment 3] reading, no learning | known share minus pooled | power for the observed effect = 1.000
[experiment 3] reading, no learning | known share minus pooled | power for half the observed effect = 1.000
[experiment 3] reading, no learning | known share minus pooled | effect on trials 2-13 = +0.11482
[experiment 3] learning on top of a lean | norm + learning minus norm | effect = +0.00689 nats/choice
[experiment 3] learning on top of a lean | norm + learning minus norm | effect 95% interval = [+0.00314, +0.01066]
[experiment 3] learning on top of a lean | norm + learning minus norm | power for the observed effect = 0.678
[experiment 3] learning on top of a lean | norm + learning minus norm | power for half the observed effect = 0.228
[experiment 3] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 = +0.00749
[experiment 3] learning on top of a lean | coded + learning minus coded | effect = +0.00497 nats/choice
[experiment 3] learning on top of a lean | coded + learning minus coded | effect 95% interval = [+0.00199, +0.00800]
[experiment 3] learning on top of a lean | coded + learning minus coded | power for the observed effect = 0.589
[experiment 3] learning on top of a lean | coded + learning minus coded | power for half the observed effect = 0.194
[experiment 3] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 = +0.00540
[experiment 3] learning on top of a lean | pooled + learning minus pooled | effect = +0.00399 nats/choice
[experiment 3] learning on top of a lean | pooled + learning minus pooled | effect 95% interval = [+0.00022, +0.00781]
[experiment 3] learning on top of a lean | pooled + learning minus pooled | power for the observed effect = 0.282
[experiment 3] learning on top of a lean | pooled + learning minus pooled | power for half the observed effect = 0.106
[experiment 3] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 = +0.00436
[experiment 3] the whole thing | norm + learning minus pooled + learning | effect = +0.08169 nats/choice
[experiment 3] the whole thing | norm + learning minus pooled + learning | effect 95% interval = [+0.07036, +0.09303]
[experiment 3] the whole thing | norm + learning minus pooled + learning | power for the observed effect = 1.000
[experiment 3] the whole thing | norm + learning minus pooled + learning | power for half the observed effect = 0.998
[experiment 3] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 = +0.08219
[experiment 3] the whole thing | coded + learning minus pooled + learning | effect = +0.00012 nats/choice
[experiment 3] the whole thing | coded + learning minus pooled + learning | effect 95% interval = [-0.00334, +0.00361]
[experiment 3] the whole thing | coded + learning minus pooled + learning | power for the observed effect = 0.050
[experiment 3] the whole thing | coded + learning minus pooled + learning | power for half the observed effect = 0.050
[experiment 3] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 = +0.00042

[experiment 4] reading, no learning | norm minus pooled | effect = +0.07491 nats/choice
[experiment 4] reading, no learning | norm minus pooled | effect 95% interval = [+0.06491, +0.08492]
[experiment 4] reading, no learning | norm minus pooled | power for the observed effect = 1.000
[experiment 4] reading, no learning | norm minus pooled | power for half the observed effect = 0.997
[experiment 4] reading, no learning | norm minus pooled | effect on trials 2-13 = +0.07116
[experiment 4] reading, no learning | coded minus pooled | effect = +0.00063 nats/choice
[experiment 4] reading, no learning | coded minus pooled | effect 95% interval = [-0.00188, +0.00321]
[experiment 4] reading, no learning | coded minus pooled | power for the observed effect = 0.061
[experiment 4] reading, no learning | coded minus pooled | power for half the observed effect = 0.053
[experiment 4] reading, no learning | coded minus pooled | effect on trials 2-13 = +0.00014
[experiment 4] reading, no learning | norm minus sign | effect = +0.07428 nats/choice
[experiment 4] reading, no learning | norm minus sign | effect 95% interval = [+0.06492, +0.08340]
[experiment 4] reading, no learning | norm minus sign | power for the observed effect = 1.000
[experiment 4] reading, no learning | norm minus sign | power for half the observed effect = 0.999
[experiment 4] reading, no learning | norm minus sign | effect on trials 2-13 = +0.07102
[experiment 4] reading, no learning | coded minus sign | effect = +0.00000 nats/choice
[experiment 4] reading, no learning | coded minus sign | the two are the same model, so there is nothing to detect
[experiment 4] reading, no learning | known share minus pooled | effect = +0.10626 nats/choice
[experiment 4] reading, no learning | known share minus pooled | effect 95% interval = [+0.09566, +0.11675]
[experiment 4] reading, no learning | known share minus pooled | power for the observed effect = 1.000
[experiment 4] reading, no learning | known share minus pooled | power for half the observed effect = 1.000
[experiment 4] reading, no learning | known share minus pooled | effect on trials 2-13 = +0.10392
[experiment 4] learning on top of a lean | norm + learning minus norm | effect = +0.00671 nats/choice
[experiment 4] learning on top of a lean | norm + learning minus norm | effect 95% interval = [+0.00286, +0.01050]
[experiment 4] learning on top of a lean | norm + learning minus norm | power for the observed effect = 0.606
[experiment 4] learning on top of a lean | norm + learning minus norm | power for half the observed effect = 0.200
[experiment 4] learning on top of a lean | norm + learning minus norm | effect on trials 2-13 = +0.00723
[experiment 4] learning on top of a lean | coded + learning minus coded | effect = +0.00534 nats/choice
[experiment 4] learning on top of a lean | coded + learning minus coded | effect 95% interval = [+0.00205, +0.00858]
[experiment 4] learning on top of a lean | coded + learning minus coded | power for the observed effect = 0.544
[experiment 4] learning on top of a lean | coded + learning minus coded | power for half the observed effect = 0.179
[experiment 4] learning on top of a lean | coded + learning minus coded | effect on trials 2-13 = +0.00581
[experiment 4] learning on top of a lean | pooled + learning minus pooled | effect = +0.00462 nats/choice
[experiment 4] learning on top of a lean | pooled + learning minus pooled | effect 95% interval = [+0.00056, +0.00862]
[experiment 4] learning on top of a lean | pooled + learning minus pooled | power for the observed effect = 0.307
[experiment 4] learning on top of a lean | pooled + learning minus pooled | power for half the observed effect = 0.112
[experiment 4] learning on top of a lean | pooled + learning minus pooled | effect on trials 2-13 = +0.00506
[experiment 4] the whole thing | norm + learning minus pooled + learning | effect = +0.07701 nats/choice
[experiment 4] the whole thing | norm + learning minus pooled + learning | effect 95% interval = [+0.06675, +0.08720]
[experiment 4] the whole thing | norm + learning minus pooled + learning | power for the observed effect = 1.000
[experiment 4] the whole thing | norm + learning minus pooled + learning | power for half the observed effect = 0.997
[experiment 4] the whole thing | norm + learning minus pooled + learning | effect on trials 2-13 = +0.07333
[experiment 4] the whole thing | coded + learning minus pooled + learning | effect = +0.00135 nats/choice
[experiment 4] the whole thing | coded + learning minus pooled + learning | effect 95% interval = [-0.00130, +0.00408]
[experiment 4] the whole thing | coded + learning minus pooled + learning | power for the observed effect = 0.096
[experiment 4] the whole thing | coded + learning minus pooled + learning | power for half the observed effect = 0.061
[experiment 4] the whole thing | coded + learning minus pooled + learning | effect on trials 2-13 = +0.00088
```
