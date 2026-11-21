# zoon: social dynamics from simulated individuals, with every claim checked or labelled

zoon is a proposed open-source project that simulates one person at a time and studies what
happens when simulated people interact. Its one unusual rule: a language model is never asked
what a person would do. The person's memory, learning and decision timing are ordinary code
built from published equations. Language models only answer narrow questions about what a
situation means. This document decides what zoon claims, how it would be built, and which
experiment has to succeed before most of it is built at all.

Status: accepted by the maintainer on 2026-11-15 as the basis for the pilot. The pilot section
is binding. The registered H1-evaluative verdict was a pass; H1-strict failed. The design went
through four passes of independent adversarial review by models from three vendors; it has not
been reviewed by people outside the project. The pilot and the interaction kernel and
training comparisons are built; the registered interaction run is not. The first interaction
test starts on the pilot's registered verdict and uses no judge, so it does not depend on
which reading of the pilot is taken. `interaction/registration.md` fixes its draft protocol.
H2 scores all arms with the same 0.005 floor before the logarithm, followed by division
by the sum, and requires 0.0052 nats per decision; H3's threshold stays 0.0031.
The decision-model arm also uses the 0.005 floor before recalibration. Closed-loop
draws keep the committed kernel's probability rules. The refit check uses a
probability tolerance of `1e-3` in the locked environment of `uv.lock`; its failure
blocks registration, not a registered run, and it is outside the run guard.
Reordering the training groups moves a predicted probability by up to about 3e-4;
`docs/interaction/candidates.md` reports the corresponding fitted-number movement.
Reversal cost: high for three decisions the pilot commits to, which are one-way doors: the
language, the model boundary and the licence. Publishing becomes a one-way door at the first
public commit. The general kernel and log schema remain provisional beyond the first
interaction test's round-based kernel. The rest are two-way doors at medium or low cost. Each is graded in "Decisions and what it costs to
reverse them".

## Objective

Decide what zoon is, what it claims, and in what order it is built, before any code exists.

The project is worth building only if keeping a person's state and laws in code, outside the
model, produces behaviour that a prompted model alone does not. That is hypothesis H1. It is
tested first, in a small pilot, before the kernel or any social simulation is built.

## Context

**The idea is not new as a pattern.** Taillandier and co-authors describe "theory-driven hybrid
architectures embedding LLMs in explicit, mechanistic, and reproducible Agent-Based Modeling
(ABM) frameworks", and their full text names the arrangement used here "LLM-as-Subroutine".
The same paper names the risk: such designs "remain vulnerable to physics washing", where
rigorous deterministic parts lend unearned credibility to the behavioural parts.

**Parts of it are already implemented elsewhere.**

| System | What it does in code | What the model still decides |
|---|---|---|
| Generative Agents (2023) | Memory recency decay, factor 0.995 per game hour | Plans, actions, and what happens to objects |
| Humanoid Agents (2023), D2A (2024) | Needs that decay over time | Whether a need is satisfied; which activity to choose |
| Concordia (2023, active) | A component system; a grounded inventory | "Agents take actions by describing what they want to do in natural language", and a model Game Master resolves them |
| AgentSociety (2025) | A coded urban and economic environment | Agent behaviour, from prompted needs and emotions |

Two bodies of work sit closer to the individual than those systems and are not yet read in
full: frameworks that map language agents onto cognitive architectures (CoALA; work by Wray,
Kirk and Laird on Soar and ACT-R with language models), and multi-option decision models
(the leaky competing accumulator, decision field theory, the attentional drift-diffusion
model). Reading the first group is a precondition of the pilot.

The research pass found no system that combines a deterministic kernel, coded memory and
decision-timing laws, and a claim ledger with mutation tests. The search was narrow: arXiv
abstracts for "drift diffusion" with LLM agents and for "ACT-R" with "generative agents". No
novelty is claimed on that basis.

**Language models as stand-ins for people have measured defects.**

- Less variation than people: "there is less variation in responses than in the real surveys"
  (Bisbee et al. 2024).
- They "both misportray and flatten the representations of demographic groups" (Wang,
  Morgenstern and Dickerson).
- More rational and more cooperative than people in economic games (Chen et al. 2023; Mei et
  al. 2024).
- They may "regurgitate problematic and memorized results" of classic experiments (Horton et
  al.).
- A critical review found validation "poorly addressed, with many studies relying solely on
  subjective assessments of model 'believability'" (Larooij and Törnberg 2025).

**A brain cannot be simulated from physics today.** Complete wiring diagrams exist for a
302-neuron worm and a fruit fly of 139,255 neurons. The worm simulation "can demonstrate basic
locomotion", with the full behavioural repertoire "a future goal". So the laws used here sit at
the level of cognition and behaviour, where they are models with published equations.

**The method comes from the first-principles-sim skill**
(https://github.com/broomva/skills): never write code that
draws the phenomenon; give every stated quantity a claim, a kind and a test; add negative
controls and mutations. Three of its own stated limits apply here:

- It is experimental (version 0.1.1), and its notes say its audit "did not pass independent
  review".
- It asks for "a mechanism with known laws", and says that where the laws are themselves
  models "most claims will be predictions and conventions". So it governs the kernel and the
  coded laws, not the model calls.
- It warns that "a closed-form solution of the equation being solved verifies the solver, not
  the physics both sides assume". Matching an analytic result therefore checks zoon's code and
  says nothing about people.

## Goals

1. Test H1 early and cheaply, with a protocol that can fail.
2. A kernel small enough to check exactly.
3. One individual, called a zoon, whose internal dynamics are coded laws, each tested against
   an anchor or labelled as a convention.
4. A model boundary that can be tested, not merely asserted.
5. A report in which every number shows what supports it, layer by layer.

### Hypothesis H1 and its protocol

**H1.** On one task that needs both state and language, a full zoon predicts held-out human
choices better than each of three reduced versions, for two model families.

| Arm | What it is | What beating it shows |
|---|---|---|
| `zoon` | Coded laws with a model in the loop | — |
| `prompted` | The same model asked directly for the choice, given the same training examples | The architecture adds to a prompt |
| `laws-only` | The zoon with models replaced by coded stand-ins | The model adds to the laws |
| `no-state` | The zoon with a model, and memory, values and needs frozen at neutral. The model sees only the current trial, with no history in the text | The state adds to the model |

Two reference rows are reported and are not part of H1: the best published model for the task,
and Centaur. H1 is an ablation and does not claim to be the best predictor, but readers need
the ceiling. A reference known only from a published score is labelled as not comparable,
since it was measured on a different split.

**Amended for the pilot, 2026-11-15.** Measurements on training participants, and four
decisions by the maintainer, changed how H1 is tested. `prereg/h1.md` fixes the details and
governs where it differs from this section.

- **Dilemmas are held out as well as participants.** With participants alone held out, one
  fitted number per dilemma can stand in for reading it (`docs/pilot/task-selection.md`).
- **Two hypotheses are run.** H1-strict keeps the boundary as this document first wrote it.
  H1-evaluative lets the judge rate how appropriate the one suggested action is, because a
  reading built from descriptive facts predicted nothing (`docs/pilot/lean.md`). The pilot's
  verdict rests on H1-evaluative. A pass of it is not a pass of H1 as written above.
- **One model family decides.** The second is reported as a check. Only an older, small
  model of the second family returns the probabilities the test needs.
- **The judge reads de-identified rewrites**, not the dilemmas participants read. That makes
  a published rating harder to recite and does not rule it out, so a pass is a claim about
  how well the pipeline predicts, and not about how the judge arrived at its rating.
- **The `prompted` arm has two fitted numbers**, so that it cannot lose only because its
  probabilities are too extreme.

Each rule below closes a way the test could pass without showing anything.

| Rule | Why |
|---|---|
| One primary measure: mean log-likelihood of each held-out choice | A choice probability is the one quantity every arm can produce. Latency is secondary and is compared only between arms that produce one |
| Every arm's distribution over the options is clamped to the same floor and ceiling, fixed in the pre-registration. An arm whose probabilities come from repeated calls uses token log-probabilities instead where it can, and otherwise at least as many calls as the reciprocal of the floor. The accumulator's two-option probability is computed analytically | A probability of zero gives a log-likelihood of minus infinity. A probability estimated from few calls cannot get close to one, which would hand the comparison to whichever arm computes its probability exactly |
| Split by participant; resample participants in a paired bootstrap | Trials from one person are correlated, and resampling trials gives intervals that are too narrow |
| Each arm fits its own parameters, on training participants only; report each arm's count of fitted parameters and of configurations tried | The zoon arms have fitted parameters and the prompted arm has none. An arm that reused another's parameters would lose through miscalibration and not through what it lacks |
| The number of training examples given to the prompted arm, and how they are chosen, are fixed in the pre-registration | Examples in a prompt are not equivalent to fitted parameters; the difference is stated with the result |
| The task, the split, the floor and the smallest difference that counts are committed in a pre-registration file before the held-out data is read | Otherwise any difference can be called a pass |
| Secondary measures are reported and never decide H1 | Passing on "at least one of several" inflates false positives |

Two limits remain, and both are stated with the result:

- **Contamination.** The candidate datasets were published before current models were
  trained, so any arm that uses a model may have seen them. Two checks are run. The model is asked
  for the dataset's published aggregate result directly. The task is also presented
  paraphrased and with rescaled payoffs, since a model can have memorised the items without
  being able to state the aggregate. If the model produces the aggregate, or its fit changes
  sharply under paraphrase, the result is reported as possibly recalled.
- **A suitable task may not exist.** On a numeric choice task the model has nothing to read.
  On a language-only task the laws have nothing to do. If no licensed dataset needs both, H1
  cannot be tested as stated. The fallback is a weaker hypothesis, H1′: all four arms on two
  tasks, one with learning over trials and one where wording changes behaviour. H1′ passes if
  the zoon beats `prompted` and `no-state` on the learning task and beats `laws-only` on the
  wording task, each by the pre-registered difference. It is reported under that name. It
  shows each half works where it should and does not show that they work together.

**What ends the project.** If H1 fails, the result is written up as a negative finding and the
kernel and social milestones are not built on this architecture. If only H1′ can be tested,
the pilot is recorded as "H1 untestable", with the H1′ result beside it, and the kernel is
started only on the maintainer's explicit decision.

## Non-goals

- **Not a forecast.** No output is a prediction about a real population, market or election.
- **Not a model of any real person.** No real individual's data parameterises a zoon.
- **No demographic identities.** Zoons differ by state: memories, learned values and
  parameters drawn from distributions. They carry no age, gender, nationality or group label.
  The cited flattening results are about exactly that kind of conditioning.
- **Not a replacement for human participants**, and not a validated model of cognition.
- **Not the best predictor of individual choices.** Fine-tuned models hold that position.
- **No model-written goals.** A model never sets or reorders a zoon's goals.
- **No open-ended world in the first version.** Each world declares a closed set of actions,
  and code decides what each action does.
- **No general account of sociality in the first version.** There is no reasoning about what
  another believes or will accept, no punishment of norm violations and no forming or cutting
  of ties. The first version can study the spread of information and learning in a repeated
  two-party exchange. The first interaction test covers contributions to a repeated public good.
  One-shot bargaining and norms stay out.
- **No tooling for optimising persuasion or targeting groups.**
- **No new data collected from people, no hosted service, no visual viewer** in the first
  version.

## Design

### Vocabulary

| Term | Meaning |
|---|---|
| Kernel | The code that holds what is true whatever anyone believes: time, the event log, the ledger, the channel |
| Zoon | One simulated individual: typed state, coded laws, and narrow model calls |
| World | A scenario: resource types, a closed action set, a channel layout, initial conditions |
| Model | A stateless function behind an interface: embed, judge or generate |
| Measure | A function of the event log that reports a quantity. Social patterns are measures, never stored state |
| Claim | One statement the repository makes, with its kind and what enforces it |
| Coded policy | A decision rule written in code and used in place of a zoon, to check the kernel against a known result |

The name is from Aristotle's "political animal" (Politics I, 1253a; the Greek order is
"politikon zoon"). Directory and module names are plain English.

### Layout and dependencies

```
run/        assembles a kernel, a world and zoons; the only place that imports all three
worlds/     scenario data and rules; implements the kernel's World interface
zoon/       the individual; implements the kernel's Agent interface; imports models/
models/     embed, judge, generate; providers; transcripts
measures/   functions of the log
kernel/     clock, queue, log, ledger, channel; defines the Agent and World interfaces
```

`kernel/` imports nothing above it. It calls zoons and worlds only through the two interfaces
it defines. `measures/` imports only the log schema. The pilot needs `zoon/`, `models/` and a
single-zoon loop, and no kernel.

### The zoon

State is typed and held outside any model.

| Part | Law | Check that the code is right | Human data |
|---|---|---|---|
| Memory | Activation `A = B + w·s`. `B = ln(Σ t^-d)` is ACT-R's base-level term with default `d = 0.5`. `s` is embedding similarity to the current situation | Reproduce the published equations; reproduce values from an existing ACT-R implementation | Open. `w` and the scaling of `s` are conventions |
| Decision, two options | Drift-diffusion: evidence accumulates to a boundary and gives a choice and a latency. Drift is `k·(score₁ − score₂)` and noise is fixed at 1. Where latencies are available, `k` and the boundary are fitted to choices and latencies together. On a choice-only task they cannot be told apart and are fitted as one parameter | A lower boundary gives faster and less accurate choices; simulated latencies match the analytic distribution | Confidence Database (CC0); dataset to be chosen |
| Decision, more than two options | An ordered response rule for the first interaction test; a race of accumulators remains deferred | The interaction rule is checked against its committed equation | Training public goods groups; no claim about timing |
| Values | Error-correcting update of the learned worth of an action | Converges to the mean outcome | Open; set by the H1 task |
| Beliefs | One confidence per fixed proposition, moved toward a source's report in proportion to trust | Under a coded policy, matches DeGroot consensus | None. A `convention` |
| Body | Deferred until a world needs it | — | — |

Notes on this table:

- Reproducing another implementation's values checks the transcription of the equations. It
  is a `reference` check on the code and not an `independent` one, because both sides use the
  same formula.
- The error-correcting update is used here for the worth of actions. Rescorla and Wagner
  formulated it for conditioning, so applying it to choice is a convention. It is not expected
  to fit the CPC18 data.
- The belief law averages. Averaging on a connected network tends to agreement, so this law
  cannot produce polarisation. A result showing consensus is a property of the law.
- Whether values should be a separate table or recalled from memory of past outcomes is open
  and is settled on training data in the pilot.

**The score.** For each available action `a` in situation `s`:

`score(a) = value(a) + need(a) + φ(a) · appraisal(s)`

`value(a)` is learned. `need(a)` is how far the action moves a need toward its setpoint.
`appraisal(s)` is the judge's reading of the situation. `φ(a)` is a vector the world's author
writes for each action, saying how much each appraisal dimension matters to it. This formula
has no published law. `φ` and the weights are convention parameters and are counted against
the zoon arm. The difference in `φ` between two options is scaled to unit length, so that its
size cannot be traded against `k`.

One step:

1. **Update beliefs** from the observation.
2. **Recall.** Traces are ranked by activation. Only what is recalled goes forward.
3. **Appraise.** A judge model scores the situation on the world's declared dimensions. It is
   given the observation and the recalled traces. It is not given the available actions.
4. **List options.** The world's action set, filtered by preconditions, in code.
5. **Score options** in code, by the formula above.
6. **Accumulate to a threshold.** Code, seeded. Gives the choice and its latency.
7. **Emit the intent.** Where the action carries content, such as a message, a generate model
   writes the content after the choice is made.
8. **On the outcome**, update values, write a memory trace, update the body.

In the first interaction world the appraisal step is absent: no model is called by the
coded person. It updates contribution traces and samples an ordered response once per round,
without an accumulator or latency. The model boundary below is unchanged.

The model is never told any of these laws.

### Model boundary

| Role | Input | Output | Used for |
|---|---|---|---|
| `embed` | Text | Vector | Similarity of a memory to a situation |
| `judge` | An observation, recalled traces, one question with declared answers | A probability for each declared answer | Appraisal |
| `generate` | A view of state, a typed schema | One value of that schema | Message text; reading a received message into a typed proposition |

Structural rules:

- **Appraisal dimensions come from one vocabulary file shipped with zoon.** A world chooses
  from it and cannot add its own. Adding a dimension is a reviewed change to that file. The
  first vocabulary is a convention.
- **The judge never sees the options**, and the generate model writes content only after the
  choice. Amended 2026-11-15: where a situation suggests exactly one action, a judge may be
  asked to rate that action. It is still never shown a person, a history or a set of options
  to choose among.
- **No free text reaches a decision.** Generated text is sent as a message or parsed into a
  typed value and validated.
- **Other zoons' words are untrusted input.** A message is read into a proposition by a call
  that sees only the message.
- **Sampling happens in code**, with the run's seed.

Those rules narrow the boundary. They do not prove the model is not steering the choice, since
an appraisal can still carry a preference. Two measurements address that, and both are made
against human data so that they cannot be passed by wiring alone:

- **The `no-state` and `laws-only` arms of H1.** They measure what the state and the model
  each add to predicting people.
- **Law ablation.** Each law is switched off in turn. The pre-registration states the
  direction and the smallest size of the expected drop in the primary measure. These are
  diagnostic and do not gate the pilot. A law whose removal fails to lower the measure by
  that amount, or raises it, is reported as unsupported, and no claim is made about what that
  law contributes.

**Where a probability comes from.** Not every model exposes one.

| Source | Status |
|---|---|
| A model that returns a distribution over declared answers | Accepted |
| Token log-probabilities over the declared answers | Accepted |
| Frequency over a declared number of repeated calls, with add-one smoothing | Accepted; the cost is declared |
| A number the model writes in its answer | Not a probability. Recorded as `stated`, never sampled from |

Both model families used for a claim must supply one of the accepted sources.

**Run modes.**

| Mode | Models | What it can check |
|---|---|---|
| `stub` | Coded stand-ins | The kernel and the coded laws. Nothing about models |
| `replay` | An ordered transcript of a recorded run | That the code reproduces the recorded run. Nothing about a model that has since changed |
| `live` | Real calls, recorded | Model-in-loop behaviour. A gate with a declared cap on calls and spend |

A transcript records every call in order, each with the zoon, step, role and ordinal that made
it, the request hash and the exact response. Two identical requests are two entries.

A manifest binds the transcript to: the code commit, world and initial configuration; prompts
and schemas; provider, model version and sampling settings; the run's seed, the random number
algorithm and how streams are derived from the seed; runtime versions; and a digest of the
transcript. Replay refuses a different manifest, a missing or surplus entry, and any call
whose zoon, step, role, ordinal or request hash differs from the entry at that position.

### Kernel

**First interaction kernel.** The pilot's registered verdict passed. The first interaction
test uses rounds as batches and exact fractions of a point, as `interaction/registration.md` states.
Seats decide once per round, simultaneously, and see the outcome before the next round.
It needs no decision lifecycle. The general event-driven design below remains provisional;
its timing rules do not apply to this test.

The kernel owns the clock, the event queue, the log, the ledger and the channel.

- **Clock.** Time is an integer count of rounds in the first interaction test; milliseconds
  in the general design below.
- **Log.** Append-only. World state and each zoon's state are computed from it.
- **Ledger.** World resources in exact fractions of a point in the first interaction test,
  since a pool share need not be a whole point; integer units in the general design. Every
  transfer balances exactly, counting sources, holdings and sinks.
- **Channel.** Decides which events each zoon can observe, and delivers messages.

**Identity and order.** Every queued item gets an identifier when it is queued: its due time,
the identifier of what queued it, and that source's own counter. The identifier never changes.

**General event-driven design: one millisecond is one batch**, handled in a fixed order:

1. Collect every item due at this time.
2. Apply external events in identifier order.
3. Order intents by identifier, then shuffle the whole list with a random stream derived
   from the run's seed and the timestamp.
4. Apply intents one at a time, in that order, to a working copy of the state. An intent whose
   preconditions fail against the working copy is rejected.
5. Commit the working copy. Log accepted intents, then rejections, each in the order applied.
6. Deliver observations ordered by recipient, then by event. Model calls are issued in that
   order.

**Nothing produced in a batch is due in the same batch.** Every delay is rounded up to a whole
millisecond, with a minimum of one. The scheduled time is written to the log. So a batch
always closes.

**A zoon decides on what it last observed, and the kernel checks the result later.** An intent
that is no longer valid when it comes due is rejected, and the zoon observes the rejection.

**Decision lifecycle.** Each zoon is idle, deciding or acting.

- Each decision has a generation number, carried by its intent.
- While deciding, new observations update beliefs and memory at once. They restart the
  decision only if they show the chosen action is no longer possible. A restart raises the
  generation.
- An intent whose generation is stale is dropped when it comes due. The drop is logged, and
  nothing is learned from it.
- While acting, no new decision starts until the action's duration ends.
- Outcomes are attributed by intent identifier.

These rules are a `convention`.

**Three different times, kept apart:**

| Time | Source |
|---|---|
| Decision latency | The zoon's accumulator. Claimed only within the range of the data that anchors it |
| Action duration | Declared by the world for each action |
| When an idle zoon next acts | The time a need will cross its threshold, computed in closed form and scheduled as an event |

**Numbers.**

- The general ledger uses integers in a declared smallest unit. The first interaction test
  logs ledger amounts as exact fractions of a point.
- A zoon's internal quantities are floating point and are not logged. They are recomputed from
  the log by pure functions.
- Replay is claimed to reproduce the log exactly under the same manifest. Equality across
  platforms is not claimed.
- Body energy is not a ledger quantity. Eating moves world units into a sink account, and code
  converts them to body energy. Energy spent leaves no ledger entry.
- A memory trace contributes from the millisecond after it is written, so its age is never
  zero.

### Measures and what supports them

A measure reads the log and nothing else. No variable holds a "shared worldview".

A single label per number, such as the weakest claim kind it depends on, would read the same
on every model-in-loop result and would be ignored. The report prints three things for every
measure:

1. **Support by layer:** kernel, coded laws, model calls, and comparison with human data, each
   with the kinds of the claims involved.
2. **Whether it was compared with human measurement.** If not, it is a `prediction`,
   whatever else is true. Agreement with an analytic result under a coded policy is printed as
   "solver check" and is never called validation.
3. **Sensitivity:** how far the measure moves when each convention parameter it depends on is
   changed by a declared amount.

A measure that declares no dependencies fails the test suite.

### Controls for models in the loop

- **Recall control.** For the first interaction test, the chat model is asked whether it
  recognises the experiment. Registered string matches label its comparisons "possibly
  recalled" and change no verdict. The decision model cannot write text; its control is
  disguise. For later model-in-loop tests, ask for the outcome without simulation and report
  recall under the pre-registered rule.
- **Disguise control.** Payoffs are rescaled by a positive linear transformation, and the
  scores fed to the laws are normalised by the same scale. Nothing the laws see has changed,
  so the measure, as a share of the payoff scale, must not move. This is a gate for
  model-in-loop tests. In the first interaction test it is a unit test for the coded laws;
  the prompted arm's disguise result is reported and does not gate. The open finding about
  mixed payoffs and appraisal stays open.
- **Framing sensitivity.** The cover story is changed. This can properly change the
  appraisal, so its effect is reported and does not gate. A renaming alone is not a disguise,
  because models recognise classic games by their structure. Deferred in the first
  interaction test.
- **Variation check.** In the first interaction test, the fair ranked probability score
  compares simulated and human groups summary by summary; central 90% coverage is reported.
  This replaces a separate spread check for this test.
- **Model swap and paraphrase.** In the first interaction test, model swap is the reported
  chat arm. Paraphrase is deferred.
- **Null interaction.** In the first interaction test, two tests on the committed numbers
  must pass before registration; the report repeats them and prints the result. With the
  others' weight set to zero, the sequence of bands and points of every seat and round of
  the 100 registered groups of every game is equal under an open and a closed channel.
  With the channel closed, replacing the other seats' play by any other play leaves each
  seat's five probabilities exactly equal. Independent persons' summaries need not be zero.
  Exact null social measures remain a control for worlds, such as the threshold cascade,
  where null implies zero.

### Stack

Python 3.12 or later, with uv, ruff, pyright in strict mode and pytest. The event log is JSON
Lines with a versioned schema. A viewer, if built, is a separate application that reads the
log.

## Decisions and what it costs to reverse them

| # | Decision | Door | Committed when | Cost to reverse after implementation |
|---|---|---|---|---|
| D1 | Python. Decided 2026-11-15 | One-way | Pilot | High. A rewrite of everything except the log schema |
| D2 | Models hold no state, never see the person, and return no action. Amended 2026-11-15: a judge may rate the one action a situation suggests | One-way | Pilot | High. It is the project's identity |
| D3 | Licence: Apache-2.0. Decided 2026-11-15 | One-way once others contribute | First public commit | High. Relicensing needs every contributor's consent |
| D4 | Event-sourced kernel, integer time, millisecond batches | One-way | M1; provisional until then | High. Every law and world is written against it |
| D5 | Log schema and the numeric contract | One-way | M1; provisional until then | High. Every transcript and recorded result depends on them; the schema is versioned to allow migration |
| D6 | Go public at the pre-registration commit. Decided 2026-11-15 | One-way from that commit | Before the held-out data is read | Cannot be undone |
| D7 | H1 pilot before the kernel | Two-way | Now | Low |
| D8 | Closed action sets; no generative world model | Two-way | M1 | Medium. An open world can be added as a new world type |
| D9 | Claim ledger and audit from first-principles-sim, vendored | Two-way | Pilot | Medium. The ledger is a YAML file another checker can read |
| D10 | Three run modes, with `live` as a capped gate | Two-way | Pilot | Low |
| D11 | Vendor only datasets with an explicit licence | Two-way | Pilot | Low |

**D6, argued.** A public pre-registration can be read and criticised before the result exists,
which is what makes the result believable. Three options:

| Option | Gives | Costs |
|---|---|---|
| Public from the first commit | The most scrutiny | Half-formed work is public for good |
| Private until the pilot result, with the pre-registration lodged at a public registry or its hash published | A timestamp without exposure | Nobody outside can criticise the protocol before the result |
| Private while the pilot is set up; public at the pre-registration commit, before the held-out data is read | Scrutiny of the protocol, and nothing half-formed | One-way from that commit |

The third was chosen on 2026-11-15. Whatever the option, the pilot result is published whether H1 passes
or fails. Three conditions hold before the first public commit:

- A licence is chosen.
- No transcript is committed until the provider's terms are read. Transcripts are model
  outputs, and some terms restrict redistributing them. Until then the repository holds
  request hashes and aggregate results only.
- No dataset is committed without an explicit licence.

## Alternatives considered

### Run the H1 experiment alone, with no repository design

The cheapest option: one script, one task, four arms.

Adopted in part. It is the first milestone, and the kernel waits on its result. It is not
adopted as the whole plan because the pilot still needs the model boundary, the transcript
format and the pre-registration discipline, and those are decided here once.

### Build on Concordia, with a coded resolver

Concordia is Apache-2.0, active and built from components. Its model Game Master could be
replaced by a coded resolver, and its model calls could be recorded, so exact replay and exact
conservation are achievable there.

Rejected for a narrower reason. In Concordia an agent acts by "describing what they want to do
in natural language", which is the operation zoon forbids. Using Concordia would mean keeping
its memory and component plumbing while replacing both its agent decision and its world
resolution. That is most of the framework.

What this gives up: a maintained framework, a user community and compatibility with published
Concordia experiments. If H1 passes, contributing a coded resolver and ledger to Concordia is
still worth doing.

### Reuse an existing cognitive architecture implementation

ACT-R implementations exist in Python. One could supply memory and decision timing directly.

Rejected as a dependency, kept as a check. The declarative memory equations are a few lines,
and a full production-system runtime brings machinery zoon does not use. Which implementation
to check against is not yet examined.

What this gives up: decades of validated parameter sets and a community that would recognise
the result.

### A model fine-tuned on human behaviour as the individual

Centaur is fine-tuned on Psych-101: "160 psychological experiments and 60,092 participants,
making 10,681,650 choices". Rejected as the individual because it maps a described situation
to a choice and has no clock or decaying memory. Published criticism calls its behaviour "best
characterised as a form of role-playing". Kept as a reference row for H1.

What this gives up: the best available predictor of choices in the tasks it was trained on.

### A unit other than the individual

Model cohorts or population averages directly. Cheaper, and individual behaviour is noisy.
Rejected because the question is what emerges from individuals. Mean-field results are kept as
solver checks.

### A classical agent-based model with no language model

Fully checkable. Rejected as the whole design because coded rules cannot read a situation
described in language. Kept as `stub` mode and as the `laws-only` arm. If H1 fails, this is
what remains.

### TypeScript on Bun

Typed events are a strength, and a browser viewer could share code.

Rejected for a public research repository. The comparators for H1 (published cognitive models,
Centaur, the fitting tools) are Python. They could be called across a process boundary, so
this is a cost and not an impossibility. The people able to check the work use Python.

What this gives up: a toolchain the maintainer already uses, stronger static typing, and one
language for simulation and viewer.

### Prompted agents throughout

Rejected because it cannot separate what emerged from what the model recalled. Kept as the
`prompted` arm.

## Drawbacks and risks

- **The glue may absorb the result.** The score formula has no published law. It either
  drowns the model's signal or fits noise. The `no-state` arm, the law ablations and the
  parameter counts measure this. They do not prevent it.
- **Physics washing.** Exact kernel checks can make the whole look validated. The per-layer
  report is the mitigation. Its cost is that most social numbers will read `prediction` beside
  competitors that print nothing.
- **H1 may fail, or may not be testable.** The cost is the pilot. That is why the pilot comes
  first.
- **Appraisal is unscored.** Nothing yet checks that the judge reads situations as people do.
- **Authoring `φ` is a hidden channel.** A world's author can encode the expected behaviour in
  the action features. Counting those parameters exposes it and does not stop it.
- **Decision timing is anchored only for two options at the scale of seconds.** Everything
  else about timing is a convention.
- **Live runs cost money.** The pilot's live calls have a hard cap of 150 US dollars, set on
  2026-11-15, and use two model families: a hosted decision model that returns a probability
  for each declared answer, and a general model read through token log-probabilities. A seed
  sweep of a model-in-loop run needs one live run per seed, for two families, and again when
  a model changes. Each live gate declares a cap on calls and spend, and the report prints
  calls, tokens and spend. A measured estimate is produced before the first live run.
- **The audit is experimental.** A clean audit is a floor.
- **Copies of one model are correlated.** The variation check measures it.
- **Dual use.** A working simulator of social influence could be used to tune persuasion. A
  written non-goal is not a control. No stronger control is proposed.

## Acceptance

Thresholds are choices made here, not findings. Command names are proposals.

| # | Milestone | Fails unless |
|---|---|---|
| P0 | H1 pilot. One task, four arms, only the laws that task needs, no kernel | The prior work named in open question 10 is read. The pre-registration is committed before `unseal` is run on the held-out split. `uv run pytest` is green. `uv run python -m zoon.report h1 --check` exits 0 only if the `zoon` arm beats `prompted`, `laws-only` and `no-state` on the primary measure by at least the pre-registered difference, with a paired participant-level bootstrap 95% interval excluding zero. Amended 2026-11-15: the hypothesis that decides is H1-evaluative, for the deciding model family, exactly as `prereg/h1.md` states it |
| I1 | First interaction test, H2 and H3 | `interaction/registration.md` is registered before any held-out play is read. Each hypothesis is judged separately by its registered comparisons, thresholds and intervals. The two null-interaction tests on the committed numbers pass before registration and are repeated in the report. Every recorded closed-loop game of `zoon`, `alone` and `prompted` replays from its log to its recorded digest; failure gives H3 no verdict and leaves H2 unaffected. Chat games replay from the transcript; failure makes `chat` unusable. Ledger checks pass; the result is published whatever the verdicts |
| M1 | General kernel beyond the round-based interaction kernel | `uv run pytest` green. Replaying a log reproduces it byte for byte for each of 8 seeds, a chosen count. Ledger totals are exactly unchanged over a run. Under a coded policy, Granovetter's case holds exactly: thresholds 0 to 99 give 100 acting, and removing threshold 1 gives 1. With the channel removed, the cascade size is exactly zero. `sim_audit.py --strict --run-mutations` exits 0 for kernel claims |
| M2 | Laws for the first world, models stubbed | Each law reproduces its published equation within the tolerance written in its claim. Every checked claim has a mutation that turns its own test red |
| M3 | Two zoons in a repeated exchange, two options per decision | `zoon.report m3 --check` exits 0 only if the measure, as a share of the payoff scale, stays within the pre-registered bound under the disguise control, and the recall control does not reproduce it within that bound. Framing sensitivity is reported |
| M4 | Threshold cascade in a population | `zoon.report m4 --check` exits 0 only if the coded-policy run matches the analytic result exactly, null interaction gives exactly zero, and the recall control does not reproduce the model-in-loop result. That result is reported as a `prediction` |

I1 starts on P0's registered H1-evaluative pass, with H1-strict's failure reported.
M1 to M4 are not started if P0's registered verdict fails. If H1 could not be tested and only
H1′ was, they are started only on the maintainer's explicit decision.

## Open questions

1. **Language.** Decided 2026-11-15: Python.
2. **When to go public.** Decided 2026-11-15: at the pre-registration commit.
3. **The H1 task.** Decided 2026-11-15: H1 is pre-registered on the moral-learning task of
   Maier, Cheung and Lieder, with all four experiments pooled and with dilemmas held out as
   well as participants. The survey and the power calculation behind this are in
   `docs/pilot/task-selection.md` and `docs/pilot/power.md`.
4. **The appraisal vocabulary and an anchor for it.** Settled for the pilot: the questions
   are those of two published codings of the same dilemmas, and the anchor is the published
   human ratings of 25 dilemmas outside the task (`docs/pilot/arms.md`). A vocabulary for
   other worlds is still open.
5. **Licences.** This repository is Apache-2.0, decided 2026-11-15. `broomva/skills`, which
   holds the audit, is MIT-licensed. An earlier version of this line said it had no licence;
   that was wrong.
6. **Provider terms for transcripts.** Next step: read the terms of each provider used before
   any transcript is committed.
7. **Datasets without a licence.** choices13k, the Murre and Dros forgetting data, the Iowa
   Gambling Task pool and others state none. Next step: fetch by script without redistributing,
   and write to the authors where one becomes an anchor.
8. **The package name.** `zoon` is taken on npm and crates.io and appeared free on PyPI on
   2026-11-15. Next step: reserve the PyPI name.
9. **Multi-option decisions.** Next step: read the leaky competing accumulator and decision
   field theory papers before any world offers more than two options.
10. **Unread prior work on the same interface.** Done: `docs/pilot/prior-work.md`. No
    published system was found that matches the boundary as a whole; two come close and are
    cited there.

## Sources

Verification key: **[me]** fetched and read by the author on 2026-11-15; **[R]** quoted by a
research pass on 2026-11-15 from the primary text or abstract; **[V]** also fetched
independently by a reviewer; **[S]** secondary source only; **[U]** named but unread.

- Taillandier et al., https://arxiv.org/abs/2507.19364. Abstract quotes [me][V]; the term
  "LLM-as-Subroutine" [R].
- Vezhnevets et al., Concordia, https://arxiv.org/abs/2312.03664. Quotes [me][V]; component
  design and inventory [R].
- Park et al., Generative Agents, https://arxiv.org/abs/2304.03442. Decay factor [R].
- Humanoid Agents, https://arxiv.org/abs/2310.05418; D2A, https://arxiv.org/abs/2412.06435.
  [R]
- AgentSociety, https://arxiv.org/abs/2502.08691; OASIS, https://arxiv.org/abs/2411.11581;
  AgentTorch, https://arxiv.org/abs/2409.10568. [R]
- Sumers et al., CoALA, https://arxiv.org/abs/2309.02427; Wray, Kirk and Laird,
  https://arxiv.org/abs/2505.07087. [R] for existence; [U] for content.
- Bisbee et al. 2024, https://doi.org/10.1017/pan.2024.5. [R]
- Wang, Morgenstern, Dickerson, https://arxiv.org/abs/2402.01908. [R]
- Chen et al. 2023, https://arxiv.org/abs/2305.12763; Mei et al. 2024,
  https://doi.org/10.1073/pnas.2313925121. [R]
- Horton et al., https://arxiv.org/abs/2301.07543. [R]
- Larooij and Törnberg 2025, https://arxiv.org/abs/2504.03274. [R][V]
- Binz et al. 2025, Centaur, https://doi.org/10.1038/s41586-025-09215-4; Psych-101,
  https://huggingface.co/datasets/marcelbinz/Psych-101 (Apache-2.0). [R]
- Bowers et al., https://doi.org/10.31234/osf.io/v9w37_v3. [R]
- Anderson et al. 2004, ACT-R equations and `d = 0.5`,
  http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/526FSQUERY.pdf. [R]
- Ratcliff and McKoon 2008, https://pmc.ncbi.nlm.nih.gov/articles/PMC2474742/. [R]
- Rescorla and Wagner 1972, via http://www.scholarpedia.org/article/Rescorla-Wagner_model. [S]
- Plonsky, Erev, Ert, CPC18 raw data, https://zenodo.org/records/2571510 (CC BY 4.0). [me]
- Rahnev et al. 2020, Confidence Database, https://osf.io/s46pr/ (CC0). [R]
- Granovetter 1978, threshold model test case. [S]
- White et al. 1986 (302 neurons); Dorkenwald et al. 2024 (139,255 neurons). [R] State of
  Brain Emulation Report 2025, https://arxiv.org/abs/2510.15745. [S]
- Aristotle, Politics I, 1253a (Rackham translation, Perseus). [R]
- Registry checks for the name `zoon`. [R]; `broomva/zoon` absent [me].
- first-principles-sim skill, https://github.com/broomva/skills, version 0.1.1: SKILL.md,
  method reference and changelog. [me]

[U], and not relied on: the leaky competing accumulator, decision field theory and the
attentional drift-diffusion model; the published CPC18 baseline models; any specific ACT-R
implementation in Python; DeGroot's convergence conditions in the primary text; any published
set of appraisal dimensions. Each must be read before it is used.

## Known open findings

The last review pass of this document left two findings open.

1. **The disguise control can reject a correct model.** The text above says that after payoffs
   are rescaled and scores normalised "nothing the laws see has changed". That is wrong when a
   score mixes a payoff-based value with an appraisal term that does not scale with payoffs. A
   value difference of 1 and an appraisal contribution of 1 give 2; with payoffs multiplied by
   10 and the score divided by 10, the result is 1.1. The fix is to put payoffs in canonical
   units before learning and scoring, and to gate only on that representation being unchanged.
   The pilot has no payoffs and does not use this control. It stays open for milestone M3.
2. **Probabilities estimated from repeated calls bias the log-likelihood.** Taking the
   logarithm of a sampled probability is biased, and resampling participants does not remove
   it. Settled for the pilot: no arm estimates a probability from repeated calls. One model
   returns a distribution and the other is read through token log-probabilities
   (`prereg/h1.md`). The rule in the H1 table that allows repeated calls is not used.
