# The prompted arm: requests and spending bound

This is preparation for the prompted prediction arm of the draft interaction design.
The model's instructions are in [the instruction files](../../interaction/instructions/).
Its question and declared answers, with an empty history and a made-up later history,
are in [the request files](../../interaction/requests/). These are the materials given
and the prediction asked for.

The chat arm has [its own request fixtures](../../interaction/chat-requests/),
point-reply client, transcript and replay. Its recall check matches the registered
runs of letters and digits and retains replies in full. No live run or held-out
history reader is supplied here. The registration fixes separate spending caps.
The decision model was sent format-check requests twice on made-up histories, four
calls each time; the registration's section on calls made before registration
records both. The replies were usable, and no other call has been made for this
arm. The format check is for the maintainer:
`python -m zoon.interaction.prompted smoke --live`.
Both format checks are outside the bound. The chat check defaults to a seeded
stub: `python -m zoon.interaction.chat smoke`. Live calls require `--live`.

The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),
e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.1073/pnas.2525760123),
supplementary appendix, sections 5B and 5D. The other games use substituted parameters.

The spending bound reads training groups and held-out counts only. The decision
price and character estimate are from the pilot and its figures are unchanged.
The chat bound counts a call for every seat, round and group, plus recall, with
no request reuse. For each game it exhaustively finds the longest legal row
rendering, then repeats it to bound each history, just as the decision bound does.

The chat input estimate is one token per UTF byte, plus an envelope allowance
printed below. It is deliberately more generous than the decision estimator and
than both measured chat requests reported during model selection. A measured
sample cannot prove a universal tokenizer bound: this estimate assumes byte
fallback tokenisation and bounds the rendered input under that assumption.
Output includes thinking and is bounded by the configured maximum. The listed
later chat price is used for both the bound and admission; the earlier price is
kept beside it. The read date and price page are printed below; Vertex price
confirmation remains due on registration day.

Each bound covers one pass. The decision cap rounds up four times its bound;
the chat cap rounds up its one-pass bound. Unusable replies and retries are
paid from that same chat cap; reaching it makes only that arm unusable.
Reservations for calls in flight count in admission. The settings and caps below
are fixed by the code for the draft registration.

| Call | Calls | Input tokens at most | Output tokens at most | USD at most |
|---|---:|---:|---:|---:|
| open loop | 7000 | 13015060 | 7000 | 0.54663252 |
| recalibration | 512 | 960022 | 512 | 0.04032093 |
| disguise | 512 | 962253 | 512 | 0.04041463 |
| closed loop | 44000 | 81940800 | 44000 | 3.44151360 |
| total | 52024 | 96878135 | 52024 | 4.06888167 |
| chat closed loop | 44000 | 185117000 | 11264000 | 965.74800000 |
| chat recall | 8 | 16904 | 2048 | 0.10857600 |
| chat total | 44008 | 185133904 | 11266048 | 965.85657600 |

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
[settings] chat maximum output tokens = 256
[settings] chat transport attempts = 4
[settings] chat transport pauses = (1.0, 2.0, 4.0)
[settings] chat reply attempts = 3
[settings] chat input bytes per token = 1
[settings] chat input overhead tokens = 400
[settings] chat smoke repeats = 3
[settings] chat input USD per million = 4.00
[settings] chat output USD per million = 20.00
[settings] chat lower input USD per million = 2.00
[settings] chat lower output USD per million = 10.00
[settings] chat higher price from = 2026-12-01
[settings] chat lower price through = 2026-11-30
[settings] chat price read = 2026-11-21
[settings] chat price page = https://ai.google.dev/gemini-api/docs/pricing
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
[chat closed loop] calls = 44000 | input tokens <= 185117000 | output tokens <= 11264000 | USD <= 965.74800000
[chat recall] calls = 8 | input tokens <= 16904 | output tokens <= 2048 | USD <= 0.10857600
[chat total] calls = 44008 | input tokens <= 185133904 | output tokens <= 11266048 | USD <= 965.85657600
[caps] decision USD = 17
[caps] chat USD = 966
```
