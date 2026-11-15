# Choosing the task that tests H1

Surveyed on 2026-11-15. H1 needs a task where behaviour depends on both state (what happened
on earlier trials) and language (what the described situation means).

**Finding: no dataset that could be verified needs both under a split by participant alone.**
Where wording changes behaviour, a condition flag, a fixed similarity matrix or one fitted
number per item can stand in for the language. Language becomes necessary only if the test
also holds out items the fitted models have never seen.

So the H1 protocol in the design is incomplete. It splits by participant. It must also hold
out items.

Verification key: **[me]** fetched and read by the author; **[R]** read by a research pass and
quoted from it; **[U]** not opened or not confirmed.

## Recommended task

**Maier, Cheung and Lieder, moral learning from the consequences of past choices.**

| Property | Value |
|---|---|
| Data | https://osf.io/4up5z/, "CC-By Attribution 4.0 International" [me]. Project created 2023-03-22, modified 2025-10-31 [me] |
| Paper | Nature Human Behaviour, https://www.nature.com/articles/s41562-025-02271-w, published 2025-08-11 [R] |
| Size | 2,328 participants across 4 experiments; 13 dilemmas each, every one a paragraph of text followed by outcome feedback [R] |
| Published models | Six, fitted by likelihood, including model-free and model-based learning and a no-learning baseline [R] |
| Response times | Page timing only [R] |

Why it fits: each trial is text, and choices drift with feedback. It was published after
mid-2025, later than the other candidates, which lowers the chance that a model was trained on
it. It does not remove that chance, since the project has been public since 2023.

**The split it needs.** Hold out participants and hold out dilemmas. A fitted model can then
learn nothing about a held-out dilemma except through its text.

## Risks, in order of seriousness

1. **The state effect may be too small to detect.** The published learning slopes are about
   0.2 and 0.15 per log trial [R]. A rough conversion puts the cost of freezing state at
   0.006 to 0.02 nats per choice. That conversion is the research pass's own approximation and
   not a published figure. It must be replaced by a power calculation on training participants
   before anything is pre-registered.
2. **Personal data in the raw files.** The raw file has columns named `ProlificID`,
   `IPAddress.x` and `LocationLatitude.x` [R]. Whether they are populated is unchecked. The
   data is fetched by script and never committed, and those columns are dropped on load.
3. **Thirteen dilemmas is few.** Holding out dilemmas leaves little to fit on, and the
   interval for a held-out dilemma will be wide.
4. **No decision latency.** The accumulator's timing cannot be tested here. This pilot tests
   choices only.
5. **Whether the order of dilemmas was randomised is unchecked** [U]. If it was fixed,
   learning and dilemma are confounded.

## Other candidates

| Task | For | Against |
|---|---|---|
| Multiple-cue judgment, Collsiöö, Juslin and Winman 2023. In Psych-101 | Wording is the manipulation: "a verbal format can improve learning in non-additive tasks" [R]. Published negative log-likelihoods: Centaur 1.1236, Llama 1.2818, cognitive model 1.9157 [R] | In Psych-101's training split since 2024. The "language" is the format of a cue, not its meaning |
| Trait learning, Frolichs, Rosenblau and Korn 2022, https://github.com/dnhi-lab/PerLe (MIT) [R] | The published winning models need both learning and semantic similarity | The similarity can be a fixed matrix. German items, small samples |
| Two-step task with a cover story, Feher da Silva and Hare 2020, https://github.com/carolfs/muddled_models (GPL-3.0) [R] | Response times; a classical model; Centaur 0.51, Llama 0.63, cognitive model 0.61 [R] | 24 participants. Wording is constant within the data. GPL |
| Semantic labels in task learning, Habibi et al. 2025, https://osf.io/h6saz/ [R] | Recent; labels manipulated between participants | No licence set. Stimuli are pictures |

The Confidence Database (CC0) is mostly perceptual: of 180 datasets, 119 are perception and 28
memory, and none found is a language-described situation with a learning sequence [R]. It
remains a source for decision timing, not for H1.

Psych-101 itself is Apache-2.0 and holds no response times [R]. Centaur's test split is
published separately under CC-BY-ND-4.0 in a gated repository [R].

## If the recommended task fails the power calculation

Fall back to the weaker hypothesis H1′ on two tasks, as the design allows:

- **Learning only:** the drifting four-armed bandit of Bahrami and Navajas 2020, in Psych-101.
  965 participants; Centaur 0.7029, Llama 0.8810, cognitive model 0.9043 [R].
- **Wording only:** the THINGS odd-one-out task of Hebart et al. 2023, in Psych-101. 12,340
  participants; Centaur 0.8068, Llama 1.1386, cognitive model 0.8253 [R].

The original licences of both are unchecked [U].

## The three reduced versions, on the recommended task

| Arm | Concretely |
|---|---|
| `prompted` | The model is given the dilemma text and the participant's earlier dilemmas, choices and outcomes, and its probability of each choice is read |
| `laws-only` | The paper's learning model over the two kinds of option, with one fitted number per dilemma and no access to the text. It has no number for a held-out dilemma |
| `no-state` | The model reads each dilemma, giving a prior lean, and values are never updated from outcomes |

## Next steps

1. Download the data by script. Confirm the structure, whether dilemma order was randomised,
   and whether the personal-data columns are populated.
2. Run a power calculation on training participants only, for the `no-state` comparison under
   a split by participant and by dilemma.
3. If power is adequate, write the pre-registration for this task. If not, write it for H1′.
