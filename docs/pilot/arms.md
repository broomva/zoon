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

The judge never reads the dilemmas as participants saw them. It reads rewrites from which
the historic incident has been removed, which makes a published rating harder to recite and
does not rule it out. For that reason a pass is a claim about how well this pipeline
predicts, and not about how the judge arrived at its rating.

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
  keeps the design's boundary. It differs from the design's H1 in one respect: the design
  required two model families, and here the second is reported and does not decide. `lean.md`
  gives reason to expect the strict reading to carry nothing.
- **H1-evaluative**: the same for `zoon-evaluative`. Its judge rates the one action the
  situation suggests, which the design's wording does not allow.

The pilot's verdict, and with it the decision to build further, rests on H1-evaluative with
the decision model as the judge. Results for H1-strict, and for both with the second model
family, are reported beside it and cannot change the verdict. Both choices were made by the
maintainer on 2026-11-15. A pass of H1-evaluative supports the amended boundary described
below. It is not a pass of H1 as first written, and is not to be reported as one.

## What the judge is asked

In both zoons the judge sees a de-identified rewrite of one dilemma and nothing else. It
never sees a participant, a history, an outcome or another dilemma. It is never asked what
people choose. It never reads the task's own text of a dilemma, or the original it was
adapted from.

**De-identified rewrites.** The dilemmas are based on historic events, and human ratings of
the originals are published. A judge that recognised the event could return the published
rating without reading anything. So each dilemma is rewritten:

- Kept: the trade-off, who does what to whom, what is at stake for whom, how certain the
  outcomes are, and the suggested action.
- Removed: the historic incident, names, dates, places, and any detail that identifies the
  case.

A model from a vendor other than the two judge families drafts each rewrite. Each is checked
against the task's text for what must be kept and what must go, read by the maintainer, and
committed before any call to the judge. The same is done for the 25 outside dilemmas used to
check the judge, because recall would inflate that check too.

The [draft rewrites and their validation](rewrites.md) are awaiting these checks and reading.

This changes what is read. Participants read the realistic versions, and the original study
found that how realistic a dilemma seemed went with how people judged it. Whether a judge
predicts better or worse from a rewrite than from what participants read is not known.

**Evaluative.** One question: how appropriate is it to take the suggested action? The
wording and the seven labelled answers are those of the original paper's US study, which is
the sample the `norm` lean uses. The paper's two studies worded the question and its end
points differently, so the pre-registration quotes the US wording from the paper and uses no
other. `m(d)` is the expected value of the judge's answer, standardised. The
pre-registration standardises it within each rotation, on the dilemmas that are not held
out. Standardising uses no participant's data.

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

- **Evaluative.** The correlation, over those 25, between the judge's rating of the rewrite
  and the mean human rating of the original. The 20 trolley-style dilemmas rated in the same
  study are not used: they are too well known to de-identify.
- **Strict.** Agreement between the judge's answers and the published codes, per question,
  over the 25 and over the 13.
- **Stop rule.** One wording of the evaluative question is fixed before any call to the
  judge. If its correlation with human ratings on the 25 is below a minimum fixed in the
  pre-registration, the pilot stops before any held-out data is read. With 25 dilemmas a
  correlation is known only roughly: its standard error is about 0.21 on the scale usually
  used for correlations. Stopping is reported as "the judge could not be calibrated". It is
  not a result about either hypothesis. The rule uses outside data only, so passing it does
  not make a false positive on the 13 more likely.

## What a pass would and would not show

A pass is a statement about prediction: this pipeline, a judge's rating of a rewrite turned
into a lean and combined with learned values, predicted held-out choices better than the
three reduced versions. It is not a statement that the judge read the dilemma afresh. Where
its rating came from is not established by anything in this design.

On these 13 dilemmas, human ratings of nearly the same text exist, and a lean built from them
(`norm`) gains +0.08044 nats per choice over no lean, or +0.08346 when both sides also learn.
`norm` is a reference, not a limit: it rates the original texts. The benchmark `known share`,
at +0.11342 with no learning on either side, is an estimate from training participants of the
most a lean could add. So:

- Beating `laws-only` shows that the judge's rating adds predictive value to the laws. It
  does not show that a model reads better than people, or that the rating was not recalled.
- The size of that gain is reported as a share of what `norm` achieves and of what
  `known share` achieves.
- Beating `no-state` shows that state adds to the judge's rating.
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

The model is given a short description of the study, this participant's earlier dilemmas
with their answers, the outcome texts they were shown and their ratings, and the current
dilemma, all in the task's own texts and not the rewrites. `prereg/h1.md` gives
the prompt word for word. It is asked which answer this participant gives. It may use
anything it recalls. Whether the real texts help it or hurt it is not known. Then, in this
order:

1. Read the model's probability of "yes".
2. Clamp it to the floor and ceiling used for every arm.
3. Convert it to the probability of the cost-benefit option: unchanged where the suggested
   action is the cost-benefit option, and one minus it where it is the rule option.
4. Recalibrate: `logit = a · logit(p) + c`.
5. Clamp the result again.

The two fitted numbers are there because a model's probabilities can be too extreme; the
first test call to the decision model returned exactly 1 and 0 [me]. Without them the arm
could lose for that reason alone. The prompt also holds examples from the training set. Each
example is one training participant's answer on one dilemma, with that participant's history
before it. The prompt text, and the number and choice of examples, are written once,
reviewed, and fixed in the pre-registration.

This arm is given the texts participants read, where a zoon's judge gets rewrites, and the
same facts about the participant. It does not have what a zoon gets from fitting on all training participants; it
has its examples and its two fitted numbers, which are fitted on a subsample of training
participants. If it wins, the architecture added nothing on this task.

**The examples respect the held-out dilemmas.** The coded arms never fit to a participant's
answer on a held-out dilemma. The same has to hold for the prompt. So the examples are built
separately for each rotation of held-out dilemmas, and an example stops before the training
participant first met a held-out dilemma. It is an unbroken run of that participant's first
trials, with no dilemma left out of the middle. The rule for building examples is fixed in
the pre-registration.

## Recall is reduced, not ruled out

Removing the incident removes the cues that name a case. It does not remove the case's
structure, and structure can be enough: a dictator to be killed, passengers on a sinking
ship. A judge may also retrieve a rating without being able to name the event. A drafting
model from another vendor may have learned from the same material. So contamination is
unresolved, and the claim a pass supports is limited accordingly, as stated above.

Three checks are reported with the result. None is a gate, because none can certify that
recall did not happen.

- **Recognition.** A model is asked to name the event behind each rewrite, and its replies
  are matched against the details the rewrite removed. The comparisons are repeated without
  the dilemmas that match. Only a model that writes text can be asked, so this check cannot
  be run on the decision model. The rule is fixed in the pre-registration, which also says
  how little a match shows.
- **Reversal.** For dilemmas where the original paper says the opposite framing is
  reasonable, a second rewrite suggests the opposite action and changes nothing else. The
  judge's rating is compared with its own rating of the first rewrite. The two actions need
  not mirror each other on the scale, so the report gives the difference and does not score
  it.
- **Agreement with `norm`.** How closely the judge's ratings of the rewrites follow the
  published ratings of the originals is reported. Close agreement fits both good judgment and
  recall, and is not read as evidence for either.

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
- The de-identified rewrites, the reversed versions, and the list of what each rewrite keeps
  and removes.

## Models and cost

The decision model is a hosted model that returns a probability for each declared answer. The
second family is Gemini 2.5 Flash-Lite, read through token log-probabilities. Of the Gemini
models tried on 2026-11-15, it was the only one that returned log-probabilities [me]. It is
an older, small model, which is one reason the verdict does not rest on it.

The hard cap on live calls is 150 US dollars. A planning figure, from arithmetic and not from
a run:

| Part | Calls, per family | Basis |
|---|---|---|
| Judge | A few hundred | Rewrites of 38 dilemmas, plus reversed versions; up to six questions each |
| `prompted` | About 60,600 | Evaluation: 680 held-out participants, 13 trials, each once for each of the two rotations in which its dilemma is held out, which is 17,680. Fitting its two numbers: 300 training participants, the 11 dilemmas not held out, in each of 13 rotations with that rotation's examples, which is 42,900 |

The dilemma materials run to 6,184 words for 13 dilemmas with four outcome texts each [me], so
one trial's text is a few hundred tokens. The cost of the `prompted` arm depends mostly on how
many examples each prompt holds. With prompts of a few thousand tokens it fits under the cap
for both families at the listed prices; with long prompts it may not, and the number of
examples is then reduced before anything is run. The runner counts real tokens and prints the
cost before any live run, and stops at the cap.

## Risks

- **The judge's rating may not track people's.** Then the stop rule ends the pilot before the
  held-out data is used.
- **Thirteen dilemmas.** Three of them carry most of what `norm` gains, and `norm` loses
  -0.11100 on the veterinarian dilemma. A result here is a result about these dilemmas.
- **The judge may be recalling.** See "Recall is reduced, not ruled out".
- **A rewrite may change the dilemma.** Removing the incident can remove something people
  responded to, or add something. The list of what is kept and removed, and the maintainer's
  reading, are the guard. There is no human rating of the rewrites to check against.
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
