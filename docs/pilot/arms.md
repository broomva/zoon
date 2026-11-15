# The four arms on the moral-learning task

This note says exactly what each arm of the pilot computes, what a language model is asked in
each, and what is fitted on which data. It is the last design step before the
pre-registration. Nothing here has been run.

**The short version.** On this task a zoon is small: a reading of each dilemma's text that
sets how the dilemma leans, plus two values learned from feedback. The hard part is the
reading. There are only 13 dilemmas, too few to learn how to turn a reading into a lean, and
asking a model how people would answer is what the design forbids. The proposal is to learn
that step on a separate published set of 46 dilemmas with human judgments, and to carry it
over.

Verification key: **[me]** fetched and read by the author; **[R]** read by a research pass;
**[U]** not opened or not confirmed.

## The task, and what every arm predicts

Each participant answers 13 moral dilemmas in a random order. A dilemma describes a situation
and one action, and asks "Do you …?". One answer is the option recommended by cost-benefit
reasoning; the other follows a moral rule. Which is which is published for each dilemma. After
answering, the participant reads an outcome and rates it from -100 to 100.

Every arm outputs one number for a trial: the probability that this participant chooses the
cost-benefit option. The primary measure is the mean log of the probability given to what the
participant did, over held-out participants and held-out dilemmas.

Every arm may use the same information: the text of the dilemma, which answer is the
cost-benefit option, and this participant's earlier dilemmas, choices and ratings.

## The four arms

Write `v` for the learned value of the cost-benefit option minus that of the rule option
before the trial, and `lean(d)` for a number that says how dilemma `d` leans.

| Arm | Logit of choosing the cost-benefit option | Reads the text | Has state |
|---|---|---|---|
| `zoon` | `lean(d) + tau · v` | Yes, through the judge | Yes |
| `no-state` | `lean(d)` | Yes, through the judge | No |
| `laws-only` | `b(d) + tau · v`, where `b(d)` is fitted for a seen dilemma and is the pooled share for an unseen one | No | Yes |
| `prompted` | `a · logit(p) + c`, where `p` is the model's own probability for this participant's answer | Yes, directly | Only what the prompt holds |

- `v` follows the paper's learning rule: after each trial, the chosen option's value moves
  toward the participant's rating by a learning rate. This is the model already implemented
  for the power calculation.
- `zoon` with `tau = 0` is `no-state`. `zoon` and `laws-only` differ only in the baseline:
  read from the text, or not.
- The `prompted` arm gets two fitted numbers, `a` and `c`. Without them it could lose only
  because a model's probabilities are too extreme, which would say nothing about the
  architecture. The first test call to the decision model returned exactly 1 and 0.

Reference rows, reported and not part of H1: the paper's own learning model fitted per
participant, as published, and Centaur if it can be run on this task.

## How the lean is read from the text

This is the part that decides whether the pilot means anything.

### What the judge is asked

The judge reads one dilemma and answers questions about the situation it describes. Each
question has declared answers. The questions come from two published codings of moral
dilemmas:

| Dimension | Question, in substance | Source |
|---|---|---|
| Personal force | Does the person directly and physically bring about the harm? | Christensen et al. 2014 |
| Benefit recipient | Is the person's own life at stake? | Christensen et al. 2014 |
| Evitability | Would those harmed have been harmed anyway? | Christensen et al. 2014 |
| Intentionality | Is the harm a means to the end, or a side effect? | Christensen et al. 2014 |
| Killing | Does the action involve killing a person? | The dataset's own table of dilemmas |
| Consent | Does the action violate someone's consent? | The dataset's own table of dilemmas |

Christensen and co-authors define their four factors in the paper; for example, intentionality
is "whether the harm is produced instrumentally, as something willed, or whether it happens as
an unforeseen side-effect" [me]. The dataset's materials categorise its 13 dilemmas by
context, whether they involve killing, the norm transgressed, and more [me].

The judge is never asked what people would choose, what the right answer is, or how the two
options compare.

**A clarification the design needs.** The design says a dimension that "names or ranks an
action" is refused. Here each situation contains exactly one proposed action, and the
dimensions describe it. The rule is therefore restated: a dimension may describe the situation,
including the action it proposes; it may not compare options, say what to do, or say what
people do. This amendment goes into the design with the pre-registration, under review.

### Checking the judge before it is used

Both sources publish their own coding of each dilemma. So the judge can be checked alone,
against people, before it enters any arm:

- On the 46 dilemmas of Christensen et al., compare the judge's answer on each of the four
  factors with the published coding.
- On the 13 dilemmas of this task, compare its answers on killing and consent with the
  dataset's table.
- A dimension on which a model family agrees with the published coding less often than a
  threshold fixed in the pre-registration is dropped for that family.

This is the calibration that the prior-work note called for. It uses no participant's choices.

### From a reading to a lean

Thirteen dilemmas cannot support fitting six weights, and with two held out there are eleven.
So the weights are not fitted on this task at all.

1. **Fit on the outside set.** Christensen et al. collected, for each of their 46 dilemmas,
   how willing people were to take the proposed action, on a scale from 1 to 7, and publish
   the per-dilemma values [me]. A linear map from the judge's six answers to that willingness
   is fitted on those 46 dilemmas, with a fixed penalty on the size of the weights.
2. **Carry it over.** For each dilemma of this task the map gives one number, `m(d)`: how
   willing people are expected to be to take the proposed action.
3. **Fit two numbers on this task.** `lean(d) = alpha + beta · s(d) · m(d)`, where `s(d)` is
   +1 if the proposed action is the cost-benefit option and -1 if it is the rule option.
   `alpha` and `beta` are fitted on training participants and on dilemmas that are not held
   out.

So a held-out dilemma's lean comes from its text, a map learned on other dilemmas, and two
numbers that never saw it.

**This may not carry over.** The 46 dilemmas are sacrificial ones in the style of the trolley
problem. This task's dilemmas are adapted from historic events, and some involve no death at
all: tax fraud, a veterinarian, an endowment. If the map does not transfer, `beta` is near
zero, `zoon` is no better than `laws-only`, and H1 fails. That is a real way for the pilot to
fail, and it is not softened.

### Alternatives considered for this step

| Alternative | Why not |
|---|---|
| Fit the six weights on this task's training dilemmas | Eleven points for six weights. It would fit noise |
| Ask the model what share of people choose the action | It is the question the design forbids, and it is what the `prompted` arm does |
| Use the published human coding instead of the judge | Then no model is involved and nothing about reading is tested. It is kept as a substitution check, below |
| One dimension only, such as killing | Simpler, and a fair fallback if the outside map fails calibration. It tests very little |

## What is fitted, and on what

| Parameters | Arms | Fitted on |
|---|---|---|
| Six weights and an offset, for the map to willingness | `zoon`, `no-state` | The 46 outside dilemmas and their published willingness. No data from this task |
| `alpha`, `beta` | `zoon`, `no-state` | Training participants, dilemmas not held out |
| Learning rate, `tau` | `zoon`, `laws-only` | Training participants, dilemmas not held out |
| One baseline per seen dilemma | `laws-only` | Training participants, dilemmas not held out |
| `a`, `c` | `prompted` | Training participants, dilemmas not held out |

Each arm fits its own parameters. Counts are reported with the result.

## The `prompted` arm in detail

The model is given the instructions the participants saw, this participant's earlier dilemmas
with their answers, outcomes and ratings, and the current dilemma, and is asked which answer
this participant gives. Its probability for "yes" is read from the decision model's
distribution, or from the token log-probabilities of the general model. The prompt also holds
a fixed set of complete example participants drawn from the training participants; their
number and how they are chosen are fixed in the pre-registration.

This arm sees everything `zoon` sees, in one prompt. If it wins, the architecture added
nothing on this task.

## Checks that run before the pre-registration

All of these use training participants, the outside set, or no human data. None touches a
held-out participant.

- **Calibration of the judge**, as above.
- **Paraphrase.** Each judge question in three wordings. A dimension whose answers change with
  the wording is reported, and the wording is fixed before the test.
- **Substitution.** Replace the judge's answers with the published human coding. If `zoon`
  does as well either way on training data, the model's reading matches people's. If it does
  better with the human coding, the gap is the cost of the model's reading.
- **Contamination.** Ask each model for the paper's headline result and for the outcome texts
  of named dilemmas. The dataset was published in 2025 and its project has been public since
  2023, so recall is possible.

## What this pilot does not exercise

Memory that decays, needs, and the timing of decisions are in the design and are not used
here. The task has 13 trials, no response times and no needs. A pass would support two
things only: a reading of the situation that sets a baseline, and values learned from
outcomes. It would say nothing about the rest of the architecture.

## Models and cost

Two families, chosen on 2026-11-15: a hosted decision model that returns a probability for
each declared answer, and Gemini 2.5 Flash-Lite, read through token log-probabilities. Of the
Gemini models tried on that date, it was the only one that returned log-probabilities; the
current ones refused [me]. It is an older, small model, which makes it a weaker `prompted`
baseline than a current one would be. That limit is stated with the result.

The hard cap on live calls is 150 US dollars. A planning figure, from arithmetic and not from
a run:

| Part | Calls, per family | Basis |
|---|---|---|
| Judge | About 1,100 | 59 dilemmas, 6 questions, 3 wordings |
| `prompted` | About 12,700 | 680 held-out and 300 training participants, 13 trials each |

The dilemma materials run to 6,184 words for 13 dilemmas with four outcome texts each [me], so
a trial's text is a few hundred tokens and a prompt with half a participant's history a few
thousand. At the listed prices this is a small fraction of the cap for both families. The
runner counts real tokens and prints the cost before any live run, and stops at the cap.

## Risks

- **The outside map may not transfer**, as said above.
- **The outside norms are thin.** The published willingness comes from 43 undergraduates, on
  a Spanish version of the dilemmas [me]. Whether the per-dilemma values and the English texts
  can be redistributed is unchecked [U].
- **`alpha` and `beta` are fitted on eleven dilemmas.** Two numbers is few, but so is eleven.
- **The judge may be reading what it remembers.** Trolley-style dilemmas are among the most
  discussed texts there are. Agreement with the published coding may be recall of the paper.
  For calibration this does not matter: the question is only whether the coding is right.
- **A weak second family.** A result that holds for the decision model and fails for the small
  general model fails H1, by the two-family rule.
- **The `prompted` arm depends on prompt design.** A poorly written prompt would hand `zoon`
  a win. The prompt is written once, reviewed, and fixed in the pre-registration.

## Sources

- Maier, Cheung and Lieder 2025, https://doi.org/10.1038/s41562-025-02271-w; data and
  materials at https://osf.io/4up5z/ (CC-BY 4.0). The table of dilemmas and the texts are in
  "dilemmas and outcomes.pdf" [me]. The materials credit Körner and Deutsch 2022 for the
  original vignettes [me]; that paper is unread [U].
- Christensen, Flexas, Calabrese, Gut and Gomila 2014, "Moral judgment reloaded: a moral
  dilemma validation study", Frontiers in Psychology 5,
  https://doi.org/10.3389/fpsyg.2014.00607 [me]: 46 dilemmas, the four factors, 43
  participants in the validation experiment, per-dilemma norms in a supplementary spreadsheet.
  Its licence was not visible in the part read [U].
- The stated prices and the model tests are from 2026-11-15 [me].
