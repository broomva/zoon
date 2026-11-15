# zoon

Social dynamics from simulated individuals, with every claim checked or labelled.

zoon simulates one person at a time and studies what happens when simulated people interact.
It has one unusual rule: a language model is never asked what a person would do. The person's
memory, learning and decision timing are ordinary code built from published equations.
Language models only answer narrow questions about what a situation means, and they hold no
state.

## Status

No part of zoon is built yet. This repository holds the design and the preparation for a
pilot, including the code for a power calculation.

| What | State |
|---|---|
| Design | Draft, in [docs/design.md](docs/design.md). Reviewed by models, not yet by people outside the project |
| Pilot (hypothesis H1) | Preparation only: notes on prior work and the choice of task, and a power calculation on training participants, in [docs/pilot](docs/pilot). Its protocol will be fixed in a pre-registration before any held-out data is read |
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
