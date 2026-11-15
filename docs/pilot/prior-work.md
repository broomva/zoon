# Prior work on the model boundary

Read on 2026-11-15, before any pilot code, to answer one question: is zoon's rule already
published? The rule, D2 in the design: the individual's state and laws are code; a language
model is a stateless subroutine that embeds text, judges a situation on declared dimensions
without seeing the available actions, and writes or parses message content. Code selects the
action.

**Answer: not as a whole.** No published or implemented system was found that combines a
stateless model, a withheld option set, coded learned values and needs, and an
evidence-accumulation process that yields the choice and its latency. Two pieces of prior work
come close, and both must be cited.

Verification key: **[me]** fetched and read by the author; **[R]** read in full text by a
research pass and quoted from it; **[U]** not opened.

## The two closest works

| Work | What it shares with zoon | Where it differs |
|---|---|---|
| Taillandier et al., "From the Fluency Fallacy to the Micro-to-Macro Validity Gap", §5.3. https://arxiv.org/abs/2507.19364 | Names the category "LLM-as-Subroutine". In its blueprint the model "is restricted to estimating separate numerical scores for each construct" and the engine "computes the behavioral intention score via an explicit, calibrated weighting equation" [R] | It is "a hypothetical hybrid architecture" [R]: not implemented, no learning, no latency. Its constructs concern one named behaviour |
| Malloy, Ferreira, Fang, Gonzalez, "Leveraging a Cognitive Model to Measure Subjective Similarity of Human and GPT-4 Written Content", 2024. https://arxiv.org/abs/2409.00269 | Implemented and tested on human data: "an integration of an Instance-Based Learning (IBL) cognitive model with LLM embeddings" [me]. Memory with decay is code, and code selects the action [R] | The model supplies embeddings only: no appraisal, no decision timing |

## What the rest of the field does

| Work | Who selects the action | Tested against human data |
|---|---|---|
| CoALA (Sumers et al. 2023), https://arxiv.org/abs/2309.02427 | Allows code or model; the agents it surveys mostly use the model: "The LLM's output is then parsed and used to determine an external action" [R] | No; conceptual |
| Wray, Kirk, Laird 2025, https://arxiv.org/abs/2505.07087 | Analyses systems "comprised primarily of LLM components" [R] | No |
| LLM-ACTR (Wu et al. 2024), https://arxiv.org/abs/2408.09176 | A fine-tuned model decides; ACT-R is an offline teacher. The opposite direction to zoon [R] | No; the target is ACT-R's output |
| Shi et al. 2026, https://arxiv.org/abs/2510.21965 | One variant delegates selection to the model; another uses "reliable algorithmic solvers" on a game the model extracted [R] | Not seen |
| Romero et al. 2023, https://arxiv.org/abs/2308.09830; Sun 2024, https://arxiv.org/abs/2401.10444 | Proposals that place the model at perception and language [R] | No |

No paper was found that tests a restriction like D2 against a model-chooses baseline on human
data. That comparison is what the pilot does.

## Arguments that the rule is too strict

These are the strongest objections found, and the design has to live with them.

- "Agent code is interpretable and extensible, but often brittle in face of stochasticity and
  limited to address situations the designer anticipates" (CoALA §6) [R].
- With the model at the edges only, "the other modules of the CA still do not fully leverage
  the power of LLMs" (Romero et al.) [R].
- The split confers no validity: "Compatibility with environmental constraints should not,
  however, be confused with the empirical or construct validity of the generative decision
  mechanism itself". The coded mapping is itself "a potential locus of physics washing"
  (Taillandier §5, §5.3) [R].
- "hybridization does not eliminate the stochasticity or representational limits of language
  models; rather, it bounds them" (Taillandier §5) [R].

## What this changes

Five changes follow. They are made in the pre-registration and the design together, under
review, and are listed here so they are not lost.

1. **Calibrate `judge` alone before it enters the loop.** "Behavioral decision heuristics must
   be calibrated against independent empirical human data prior to integration" (Taillandier
   §5.4) [R].
2. **Add a substitution test.** Replace `judge` with a minimal fitted rule. If results are
   statistically the same, the model is, in the paper's words, "decorative complexity" [R].
3. **Treat the score and accumulator equations as hypotheses.** Compare alternative forms on
   training data, and validate on state logs with generated text removed.
4. **Stress-test `judge`** for sensitivity to paraphrase, seed and model, and for action
   leakage in the wording of questions.
5. **Use the field's terms.** Describe the design as LLM-as-Subroutine with externalised
   state. In CoALA's terms: `embed` implements retrieval, `judge` is a reasoning action that
   writes to working memory, `generate` is a grounding procedure, and proposal, evaluation and
   selection are code.

## Not read

Taillandier §1 to §3.1 and §4.3 to §4.8; Shi et al. from §6; the LLM-ACTR appendix; Sun's
2025 follow-up (arXiv:2509.10972); Yang et al. 2026 (paywalled); two papers on appraisal
ratings from language models (arXiv:2310.04450, arXiv:2310.14389). The search for
drift-diffusion processes fed by a language model returned nothing, which is weak evidence
that none exists.
