# The prompted arm: requests and spending bound

This is preparation for the prompted prediction arm of the draft interaction design.
The model's instructions are in [the instruction files](../../interaction/instructions/).
Its question and declared answers, with an empty history and a made-up later history,
are in [the request files](../../interaction/requests/). These are the materials given
and the prediction asked for.

This step includes no general chat-model arm, live run, held-out history reader or rule
for deciding recall. The recall wording is a renderer only. The spending cap remains
the maintainer's decision. The proposed ceiling is the pilot's ceiling under its cap.
No model has been called. The format check is for the maintainer:
`python -m zoon.interaction.prompted smoke`. Its offline form adds `--stub`.

The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),
e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.1073/pnas.2525760123),
supplementary appendix, sections 5B and 5D. The other games use substituted parameters.

The spending bound reads training groups and held-out counts only. Prices and the
character estimate are from the pilot. This is a decision-model bound; the chat-model
arm is outside its scope. Recall has an allowance, but no client for sending it here.
The settings below are proposed choices for pre-registration.

| Call | Calls | Input tokens at most | Output tokens at most | USD at most |
|---|---:|---:|---:|---:|
| open loop | 7000 | 13015060 | 7000 | 0.54663252 |
| recalibration | 512 | 960022 | 512 | 0.04032093 |
| disguise | 512 | 962253 | 512 | 0.04041463 |
| closed loop | 44000 | 81940800 | 44000 | 3.44151360 |
| recall | 8 | 7769 | 320 | 0.00032630 |
| total | 52032 | 96885904 | 52344 | 4.06920797 |

Output of `python -m zoon.interaction.budget`:

```
zoon interaction: prompted spending bound
[settings] rounds = 20
[settings] simulated groups per game = 100
[settings] recalibration count limit = 512
[settings] recalibration salt = interaction-recalibration-1
[settings] subsample reason = bounded training supervision before registration
[settings] simulation count reason = same per-game count as the coded comparison
[settings] hash coordinates = game label; group index; zero-based seat; one-based round
[settings] quota allocation = largest remainder; ties by game label
[settings] disguise points multiplier = 10
[settings] fixture seed = 711
[settings] fixture seed reason = fixed made-up histories for inspection
[settings] salt reason = fixed ordering before model probabilities are obtained
[settings] quota reason = integer allocation within one decision of each proportional quota
[settings] smoke games = 2-AI, 4-AI
[settings] smoke games reason = the two games with printed instructions
[settings] recalibration optimiser reason = analytic likelihood gradient; no penalty
[settings] reproducibility reason = compare all fitted coordinates across starts and row order
[settings] recall allowance reason = one wording per game; free-text renderer only
[settings] recalibration start coordinates = 0.0
[settings] recalibration optimiser = BFGS
[settings] recalibration gradient tolerance = 1e-09
[settings] recalibration iteration limit = 1000
[settings] recalibration objective = mean negative log likelihood; no penalty
[settings] recalibration coordinates = log temperature; four shifts; fifth is minus their sum
[settings] probability floor before recalibration = 1e-09
[settings] identity temperature = 1.0
[settings] identity shifts = 0, 0, 0, 0, -0
[settings] reproducibility start nudge = 0.1
[settings] reproducibility tolerance = 1e-06
[settings] recall requests per game = 1
[settings] recall output token bound = 40
[settings] history bound = longest legal round rendering, repeated; all earlier rounds
[settings] model = jev-1.14.0
[settings] input USD per million = 0.042
[settings] output USD per million = 0
[settings] price read = 2026-11-15
[settings] proposed ceiling USD = 120
[settings] pilot cap USD = 150
[data] training groups = 210
[data] held-out groups = 143
[open loop] calls = 7000 | input tokens <= 13015060 | output tokens <= 7000 | USD <= 0.54663252
[recalibration] calls = 512 | input tokens <= 960022 | output tokens <= 512 | USD <= 0.04032093
[disguise] calls = 512 | input tokens <= 962253 | output tokens <= 512 | USD <= 0.04041463
[closed loop] calls = 44000 | input tokens <= 81940800 | output tokens <= 44000 | USD <= 3.44151360
[recall] calls = 8 | input tokens <= 7769 | output tokens <= 320 | USD <= 0.00032630
[total] calls = 52032 | input tokens <= 96885904 | output tokens <= 52344 | USD <= 4.06920797
[budget] largest component = closed loop | USD <= 3.44151360
[budget] share of proposed ceiling = 0.03391007
[budget] total at most proposed ceiling = True
```
