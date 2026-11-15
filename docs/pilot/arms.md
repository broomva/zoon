# The arms of the pilot on the moral-learning task

This note says what each arm of the pilot computes, what a language model is asked in each,
and what is fitted on which data. Nothing here has been run with a model. It rests on two
measurements made on training participants with no model: `power.md` and `lean.md`.

**The short version.** On this task a zoon is a reading of each dilemma's text, which sets
how the dilemma leans, plus two values learned from feedback. `lean.md` showed that a lean
built from descriptive facts about a dilemma predicts nothing, and a lean built from a
judgment of how appropriate the action is predicts a great deal. So the pilot runs two zoons:
one whose judge may make that judgment, and one whose judge may only describe. The verdict
rests on the first. The second is reported, and is expected to read nothing.

An earlier version of this note proposed learning the lean on a different set of dilemmas. A
review showed that could not work, and it is withdrawn.

Verification key: **[me]** fetched and read by the author; **[U]** not confirmed.

## What every arm predicts

Each participant answers 13 moral dilemmas in a random order. A dilemma describes a situation
and suggests one action. One answer is the option recommended by cost-benefit reasoning; the
other follows a moral rule. Which is which is published for each dilemma. After answering,
the participant reads an outcome and rates it.

Every arm outputs, for one trial, the probability that this participant chooses the
cost-benefit option. The primary measure is the mean log of the probability given to what
the participant did, over held-out participants and held-out dilemmas.

All the coded arms are one model with parts switched on or off, as defined in `lean.md`:

`logit P(cost-benefit option) = s(d) · (alpha + beta · m(d)) + tau · (v[cost-benefit] − v[rule])`

`s(d)` is +1 if the suggested action is the cost-benefit option and -1 if it is the rule
option. `m(d)` says how appropriate the suggested action is. The last term is the learning
model: the chosen option's value moves toward the participant's rating of the outcome.

## The arms

| Arm | `m(d)` comes from | Learning | A model is asked |
|---|---|---|---|
| `zoon-evaluative` | The judge's rating of how appropriate the suggested action is | Yes | One evaluative question per dilemma |
| `zoon-strict` | The judge's descriptive answers, through a map fitted on other dilemmas | Yes | Descriptive questions per dilemma |
| `no-state`, one for each zoon | The same `m(d)` as that zoon | No | The same |
| `laws-only` | Nothing: the `pooled` baseline of `lean.md` | Yes | Nothing |
| `prompted` | — | — | Which answer this participant gives, with their history |

Reference rows, which use no model and are already computed in `lean.md`: `sign` (a reading
that carries nothing), `coded` (the published descriptive codes through the same map), `norm`
(human ratings of nearly the same text) and `known share` (the most any lean can do). Each is
reported with learning as well.

**Two hypotheses, and which one decides.** The design's H1 was written for a judge that says
what a situation means and sees no options. Only `zoon-strict` fits that wording. So the
pre-registration names two hypotheses:

- **H1-strict**: `zoon-strict` beats `prompted`, `laws-only` and its own `no-state` arm. This
  is H1 as the design wrote it. `lean.md` gives reason to expect its reading to carry nothing.
- **H1-evaluative**: the same for `zoon-evaluative`. Its judge rates the one action the
  situation suggests, which the design's wording does not allow.

The pilot's verdict, and with it the decision to build further, rests on H1-evaluative with
the decision model as the judge. Results for H1-strict, and for both with the second model
family, are reported beside it and cannot change the verdict. Both choices were made by the
maintainer on 2026-11-15. A pass of H1-evaluative supports the amended boundary described
below. It is not a pass of H1 as first written, and is not to be reported as one.

## What the judge is asked

In both zoons the judge sees the text of one dilemma and nothing else. It never sees a
participant, a history, an outcome or another dilemma. It is never asked what people choose.

**Evaluative.** One question: how appropriate is it to take the suggested action? The
wording and the seven labelled answers are those of the original paper's US study, which is
the sample the `norm` lean uses. The paper's two studies worded the question and its end
points differently, so the pre-registration quotes the US wording from the paper and uses no
other. `m(d)` is the expected value of the judge's answer, standardised over the 13 dilemmas.
Standardising uses no data from this task, and `alpha` and `beta` absorb it.

**Strict.** The descriptive questions the original study coded for each dilemma: the context,
whether it involves killing, whether those harmed are among those who benefit, whether the
person deciding benefits, and whether the opposite framing is reasonable [me]. The judge's
answers go through a penalised linear map to a predicted appropriateness. The map is fitted on
the 25 dilemmas of the original study that are not among the 13, using the judge's own
answers for those dilemmas. The penalty is chosen within those 25, and "no map at all" is one
of the choices. With the published codes the data chose no map. If the judge's answers lead
to the same choice, that confirms `lean.md`; it is not a finding about the judge.

### What this does to the design's rule

The design's rule is that a language model is never asked what a person would do. Three
things should be said plainly.

- The evaluative question is not a description. It is a judgment. It is also not a choice,
  and it is not about any person.
- A judge's rating passed through `alpha` and `beta`, which are fitted to people's choices,
  is a predictor of how people lean. That is what a zoon is for. What the rule protects is
  narrower than its wording: the model holds no state, never sees the person, and never sees
  what happened to them. Everything that makes one participant differ from another is in
  code.
- The strict zoon keeps the narrower reading of the rule, and `lean.md` gives the reason to
  expect it to fail at reading: published codes for these same questions carry nothing about
  the lean (+0.00017 nats per choice).

The amended wording goes into the design with the pre-registration, under review.

## Checking the judge alone, before anything is unsealed

The original study published human ratings and codes for 25 dilemmas that are not in this
task. They allow the judge to be checked against people with no data from this task.

- **Evaluative.** The correlation, over those 25, between the judge's rating and the mean
  human rating. The 20 trolley-style dilemmas rated in the same study are reported separately.
- **Strict.** Agreement between the judge's answers and the published codes, per question,
  over the 25 and over the 13.
- **Stop rule.** One wording of the evaluative question is fixed before any call to the
  judge. If the lower end of a confidence interval for its correlation with human ratings on
  the 25 is below a threshold, the pilot stops before any held-out data is read. The
  interval's level and the threshold are fixed in the pre-registration, also before any call.
  With 25 dilemmas the interval is wide: its standard error is about 0.21 on the scale
  usually used for correlations. Stopping is reported as "the judge could not be calibrated".
  It is not a result about either hypothesis. The rule uses outside data only, so passing it
  does not make a false positive on the 13 more likely.

The judge reads this task's own texts for the 13. Several were changed from the originals, and
one suggests the opposite action; `lean.md` tables the changes.

## What a pass would and would not show

On these 13 dilemmas, human ratings of nearly the same text exist, and a lean built from them
(`norm`) gains +0.08044 nats per choice over no lean, or +0.08346 when both sides also learn.
`norm` is a reference and not a ceiling: it rates the original texts, and a reader of the
adapted texts could do better where the adaptation changed the dilemma. The ceiling is
`known share`, at +0.11342. So:

- Beating `laws-only` shows that reading the dilemma adds to the laws. It does not show that a
  model reads better than people.
- The size of the reading is reported as a share of what `norm` achieves and of what
  `known share` achieves.
- Beating `no-state` shows that state adds to the reading.
- Beating `prompted` shows that the whole package did better than one prompt: state kept in
  code, and parameters fitted on training participants. It does not single out the state.

## Power of these comparisons

From `lean.md`, with all four experiments pooled and 680 held-out participants:

| Comparison | Power | Power at half the effect | Depends on |
|---|---|---|---|
| A zoon against its `no-state` arm | 0.997 | 0.646 | Little. With a lean that carries nothing it is 0.993 and 0.602 |
| A zoon against `laws-only` | 1.000 | 1.000 | The lean being about as good as `norm`. The smallest detectable effect is 0.01369. With a lean like `coded` there is nothing to detect: 0.173 |
| A zoon against `prompted` | Unknown | Unknown | How well the model predicts when asked directly. Not measurable without model calls |

These are for a learning model with a lean taken from human material, and each is the power
to detect a difference from zero, not from the pre-registered smallest difference. They are
not the power of the pilot: the judge's reading will differ from `norm` by an unknown amount.

## What is fitted, and on what

| Parameters | Arms | Fitted on |
|---|---|---|
| The map from descriptive answers to appropriateness | `zoon-strict` and its `no-state` | The 25 outside dilemmas. Nothing from this task |
| `alpha`, `beta` | Both zoons and their `no-state` arms | Training participants, dilemmas not held out |
| Learning rate, `tau` | Both zoons, `laws-only` | Training participants, dilemmas not held out |
| One baseline per seen dilemma | `laws-only` | Training participants, dilemmas not held out |
| `a`, `c` | `prompted` | Training participants, dilemmas not held out |

Each arm fits its own parameters, and the counts are reported. The two feedback conditions
share parameters, as in the power calculation.

## The `prompted` arm

The model is given the instructions participants saw, this participant's earlier dilemmas
with their answers, outcomes and ratings, and the current dilemma. It is asked which answer
this participant gives. Then, in this order:

1. Read the model's probability of "yes".
2. Clamp it to the floor and ceiling used for every arm.
3. Convert it to the probability of the cost-benefit option: unchanged where the suggested
   action is the cost-benefit option, and one minus it where it is the rule option.
4. Recalibrate: `logit = a · logit(p) + c`.
5. Clamp the result again.

The two fitted numbers are there because a model's probabilities can be too extreme; the
first test call to the decision model returned exactly 1 and 0 [me]. Without them the arm
could lose for that reason alone. The prompt also holds a fixed set of complete example
participants from the training set. The prompt text, and the number and choice of examples,
are written once, reviewed, and fixed in the pre-registration.

This arm is given the same facts about the dilemma and the participant as a zoon. It does
not have what a zoon gets from fitting on all training participants; it has its examples and
its two fitted numbers. If it wins, the architecture added nothing on this task.

## Recall, and what voids the result

The original dilemmas and their human ratings are published. The three dilemmas that carry
most of what `norm` gains were changed only on the surface when they were adapted. A judge
that has memorised the published table could therefore return those ratings without reading
anything, and the result would look like a success. Comparing its ratings of the adapted and
original texts would not catch this, and neither would asking it for the published means.

Two controls can catch it, and a failure of either voids the verdict:

- **Fresh paraphrases.** Each of the 13 dilemmas is rewritten with new wording, names and
  surface details and the same facts. The paraphrases are written and committed before any
  call to the judge. The lean computed from the paraphrases must do about as well as the lean
  computed from the task's texts, within a bound fixed in the pre-registration.
- **Reversal.** One of the task's dilemmas suggests the opposite action to its original.
  For that dilemma, and for versions of others rewritten to suggest the opposite action where
  the original paper says that framing is reasonable, the judge's rating has to move the
  right way.

What recall would look like: a lean almost exactly as good as `norm`, the reversed dilemma
rated as its original, and the same loss `norm` shows on the dilemma whose adaptation changed
it most (-0.11100).

## What this pilot does not exercise

Memory that decays, needs, and the timing of decisions are in the design and are not used
here. The task has 13 trials, no response times and no needs. A pass would support two
things only: a judgment of the situation that sets a baseline, and values learned from
outcomes.

## What the pre-registration still has to fix

- The exact wording of each judge question, and how answers become numbers.
- The penalty for the strict map, and the calibration threshold for the stop rule.
- The floor and ceiling for probabilities, and the smallest difference that counts.
- The `prompted` prompt and its examples.
- The exclusion of the one held-out participant whose record was seen during development.
- That the interval over dilemmas is reported beside the interval over participants. With 13
  dilemmas the second says the effect is real for these dilemmas and little about others.
- The paraphrases and reversed versions of the dilemmas, and the bound for the recall
  controls.

## Models and cost

The decision model is a hosted model that returns a probability for each declared answer. The
second family is Gemini 2.5 Flash-Lite, read through token log-probabilities. Of the Gemini
models tried on 2026-11-15, it was the only one that returned log-probabilities [me]. It is
an older, small model, which is one reason the verdict does not rest on it.

The hard cap on live calls is 150 US dollars. A planning figure, from arithmetic and not from
a run:

| Part | Calls, per family | Basis |
|---|---|---|
| Judge | A few hundred | 38 dilemmas, plus paraphrased and reversed versions of the 13; up to six questions each |
| `prompted` | About 12,700 | 680 held-out and 300 training participants, 13 trials each |

The dilemma materials run to 6,184 words for 13 dilemmas with four outcome texts each [me], so
a prompt with half a participant's history is a few thousand tokens. At the listed prices
this is a small fraction of the cap for both families. The runner counts real tokens and
prints the cost before any live run, and stops at the cap.

## Risks

- **The judge's rating may not track people's.** Then the stop rule ends the pilot before the
  held-out data is used.
- **Thirteen dilemmas.** Three of them carry most of what `norm` gains, and one adapted text
  costs it -0.11100. A result here is a result about these dilemmas.
- **The judge may recall the study.** See the next section.
- **The `prompted` arm depends on prompt design.** A poor prompt would hand a zoon the win.
- **`alpha` and `beta` are fitted on eleven dilemmas** in each fold.

## Sources

- Maier, Cheung and Lieder 2025, https://doi.org/10.1038/s41562-025-02271-w; data and
  materials at https://osf.io/4up5z/ (CC-BY 4.0) [me].
- Körner and Deutsch 2022, "Deontology and Utilitarianism in Real Life: A Set of Moral
  Dilemmas Based on Historic Events", https://pmc.ncbi.nlm.nih.gov/articles/PMC10478346/
  (CC BY 4.0) [me]: 38 dilemmas, per-dilemma codes, ratings of how appropriate the suggested
  action is on a seven-point scale from two samples, data at https://osf.io/cg5tq/. That
  project states no licence [U].
- `power.md` and `lean.md` in this directory, for every figure quoted here.
- Model tests and prices: 2026-11-15 [me].
