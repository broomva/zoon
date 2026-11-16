# zoon

Social dynamics from simulated individuals, with every claim checked or labelled.

zoon simulates one person at a time and studies what happens when simulated people interact.
It has one unusual rule: a language model is never asked what a person would do. The person's
memory, learning and decision timing are ordinary code built from published equations.
Language models only answer narrow questions about what a situation means, and they hold no
state.

## Status

The simulator is not built yet. This repository holds the design, and a pilot that tests the
design's first hypothesis: its notes, its pre-registration and its analysis code.

| What | State |
|---|---|
| Design | Accepted as the basis for the pilot, in [docs/design.md](docs/design.md). Reviewed by models, not yet by people outside the project |
| Pilot (hypothesis H1) | Pre-registered in [prereg/h1.md](prereg/h1.md), with the analysis code and the files it fixes. The notes behind it are in [docs/pilot](docs/pilot), and the commands that run it in [docs/pilot/running.md](docs/pilot/running.md). At registration no judge had been called and no held-out participant read. The result is published whether it passes or fails |
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
