# The prompted arm: requests and spending bound

This is preparation for the prompted prediction arm of the draft interaction design.
The model's instructions are in [the instruction files](../../interaction/instructions/).
Its question and declared answers, with an empty history and a made-up later history,
are in [the request files](../../interaction/requests/). These are the materials given
and the prediction asked for.

The chat arm's rules are in [the registration](../../interaction/registration.md),
and its first format check is in [the record](format-check-chat.txt);
[the output-limit record](output-limit.txt) holds the check at the registered limit.

The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),
e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.1073/pnas.2525760123),
supplementary appendix, sections 5B and 5D. The other games use substituted parameters.

The spending program reads training groups and held-out counts only. Its settings,
assumptions and output are recorded below. Vertex price confirmation remains due
on registration day. The format checks are outside the bound.

| Call | Calls | Input tokens at most | Output tokens at most | USD at most |
|---|---:|---:|---:|---:|
| open loop | 7000 | 13015060 | 7000 | 0.54663252 |
| recalibration | 512 | 960022 | 512 | 0.04032093 |
| disguise | 512 | 962253 | 512 | 0.04041463 |
| closed loop | 44000 | 81940800 | 44000 | 3.44151360 |
| total | 52024 | 96878135 | 52024 | 4.06888167 |
| chat closed loop | 44000 | 185117000 | 90112000 | 2542.70800000 |
| chat recall | 8 | 16904 | 16384 | 0.39529600 |
| chat total | 44008 | 185133904 | 90128384 | 2543.10329600 |

Output of `python -m zoon.interaction.budget`:

```
zoon interaction: prompted spending bound
[settings] decision cap multiple = 4
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
[settings] probability floor before recalibration = 0.005
[settings] model = jev-1.14.0
[settings] input USD per million = 0.042
[settings] output USD per million = 0
[settings] price read = 2026-11-15
[settings] proposed ceiling USD = 120
[settings] pilot cap USD = 150
[settings] chat model = gemini-4-carbon
[settings] chat location = global
[settings] chat temperature = 1.0
[settings] chat thinking level = low
[settings] chat maximum output tokens = 2048
[settings] chat transport attempts = 4
[settings] chat transport pauses = (1.0, 2.0, 4.0)
[settings] chat reply attempts = 3
[settings] chat input bytes per token = 1
[settings] chat input overhead tokens = 400
[settings] chat smoke repeats = 3
[settings] chat smoke default = stub
[settings] chat input USD per million = 4.00
[settings] chat output USD per million = 20.00
[settings] chat lower input USD per million = 2.00
[settings] chat lower output USD per million = 10.00
[settings] chat higher price from = 2026-12-01
[settings] chat lower price through = 2026-11-30
[settings] chat price read = 2026-11-21
[settings] chat price page = https://ai.google.dev/gemini-api/docs/pricing
[assumption] chat input uses byte-fallback tokenisation
[assumption] service bounds candidate plus thinking tokens by maximum output tokens
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
[chat closed loop] calls = 44000 | input tokens <= 185117000 | output tokens <= 90112000 | USD <= 2542.70800000
[chat recall] calls = 8 | input tokens <= 16904 | output tokens <= 16384 | USD <= 0.39529600
[chat total] calls = 44008 | input tokens <= 185133904 | output tokens <= 90128384 | USD <= 2543.10329600
[caps] decision USD = 17
[caps] chat USD = 2544
```
