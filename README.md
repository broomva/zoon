# zoon

Social dynamics from simulated individuals, with every claim checked or labelled.

zoon simulates one person at a time and studies what happens when simulated people interact.
It has one unusual rule: a language model is never asked what a person would do. The person's
memory, learning and decision timing are ordinary code built from published equations.
Language models only answer narrow questions about what a situation means, and they hold no
state.

## Status

The simulator is not built yet. This repository holds the design, and a pilot that tests the
design's first hypothesis: its notes, its pre-registration, its analysis code and its result.

| What | State |
|---|---|
| Design | Accepted as the basis for the pilot, in [docs/design.md](docs/design.md). Reviewed by models, not yet by people outside the project |
| Pilot (hypothesis H1) | Run as registered. The verdict defined in [prereg/h1.md](prereg/h1.md) is a pass: the evaluative zoon predicted held-out choices better than its three reduced versions. The strict hypothesis, the one the design first wrote, failed, and the second model family was excluded. The full output is in [docs/pilot/result.md](docs/pilot/result.md); how it was run, and what should be read with the verdict, is in [docs/pilot/run-log.md](docs/pilot/run-log.md) |
| Kernel and social simulation | Not started, and not started at all if the pilot fails |
| Licence | [Apache-2.0](LICENSE) |

## The test this project has to pass first

**H1.** On one task that needs both state and language, a full zoon predicts held-out human
choices better than each of three reduced versions: the same model prompted directly, the
coded laws with no model, and the model with the state frozen.

If H1 fails, the result is published as a negative finding and the rest is not built on this
architecture.

## What zoon is not

It is not a forecast of any real population, not a model of any real person, and not a
replacement for human participants. Zoons carry no demographic labels. The full list is in the
design.

## Name

From Aristotle's "political animal" (Politics I, 1253a).
