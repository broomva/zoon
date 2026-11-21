# The prompted arm: requests and spending bound

This is preparation for the prompted prediction arm of the draft interaction design.
The model's instructions are in [the instruction files](../../interaction/instructions/).
Its question and declared answers, with an empty history and a made-up later history,
are in [the request files](../../interaction/requests/). These are the materials given
and the prediction asked for.

This step includes no general chat-model arm, live run, held-out history reader or rule
for deciding recall. The recall wording is a renderer only. The spending cap remains
the maintainer's decision.
No model has been called. The format check is for the maintainer:
`python -m zoon.interaction.prompted smoke --live`.
The format check is outside the bound.

The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),
e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.1073/pnas.2525760123),
supplementary appendix, sections 5B and 5D. The other games use substituted parameters.

The spending bound reads training groups and held-out counts only. Prices and the
character estimate are from the pilot. This is a decision-model bound; the chat-model
arm is outside its scope. The recall question is outside this bound.
The settings below are proposed choices for pre-registration.

| Call | Calls | Input tokens at most | Output tokens at most | USD at most |
|---|---:|---:|---:|---:|
| open loop | 7000 | 13015060 | 7000 | 0.54663252 |
| recalibration | 512 | 960022 | 512 | 0.04032093 |
| disguise | 512 | 962253 | 512 | 0.04041463 |
| closed loop | 44000 | 81940800 | 44000 | 3.44151360 |
| total | 52024 | 96878135 | 52024 | 4.06888167 |

Output of `python -m zoon.interaction.budget`:

```
zoon interaction: prompted spending bound
[settings] rounds = 20
[settings] simulated groups per game = 100
[settings] recalibration count limit = 512
[settings] recalibration salt = interaction-recalibration-1
[settings] disguise points multiplier = 10
[settings] fixture seed = 711
[settings] smoke games = ('2-AI', '4-AI')
[settings] recalibration start coordinates = (1.0, 0.0, 0.0, 0.0, 0.0)
[settings] recalibration optimiser = trust-exact
[settings] recalibration optimiser options = {'gtol': 1e-06, 'maxiter': 1000}
[settings] recalibration gradient limit = 1e-10
[settings] recalibration polish steps = 20
[settings] recalibration beta lower bound = 0.0
[settings] recalibration penalty = 0.0001
[settings] probability floor before recalibration = 1e-09
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
[total] calls = 52024 | input tokens <= 96878135 | output tokens <= 52024 | USD <= 4.06888167
[budget] largest component = closed loop | USD <= 3.44151360
[budget] share of proposed ceiling = 0.03390735
[budget] total at most proposed ceiling = True
```
