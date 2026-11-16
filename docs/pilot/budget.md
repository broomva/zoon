# Spending bound

The bound is 63.122655 US dollars, below the registration ceiling of 120 US dollars.
It uses the committed training cohorts and a materials-only bound for evaluation histories.

```
zoon pilot: spending bound
[budget] examples per rotation = 2
[budget] calibration participants = 300
[budget] held-out participants, fixed by registration = 680
[budget] input estimate = ceil(characters / 3); decision adds 400 per call
[budget] decision characters = canonical request JSON, including descriptions
[budget] reversal questions = evaluative only, all committed reversed rewrites
[decision price] model = jev-1.14.0 | read = 2026-11-15 | page = https://docs.typesafe.ai/models
[decision price] USD per million input = 0.042 | USD per million output = 0
[decision outside judge] calls = 25 | input tokens = 21010 | output tokens = 0 | USD = 0.000882
[decision task and strict judges] calls = 203 | input tokens = 159957 | output tokens = 0 | USD = 0.006718
[decision reversal] calls = 17 | input tokens = 14346 | output tokens = 0 | USD = 0.000603
[decision recognition] calls = 0 | input tokens = 0 | output tokens = 0 | USD = 0.000000
[decision calibration] calls = 42900 | input tokens = 277169262 | output tokens = 0 | USD = 11.641109
[decision held-out bound] calls = 17680 | input tokens = 187535840 | output tokens = 0 | USD = 7.876505
[decision total] calls = 60825 | input tokens = 464900415 | output tokens = 0 | USD = 19.525817
[second price] model = gemini-2.5-flash-lite | read = 2026-11-15 | page = https://cloud.google.com/vertex-ai/generative-ai/pricing
[second price] USD per million input = 0.10 | USD per million output = 0.40
[second outside judge] calls = 25 | input tokens = 7421 | output tokens = 25 | USD = 0.000752
[second task and strict judges] calls = 203 | input tokens = 62116 | output tokens = 203 | USD = 0.006293
[second reversal] calls = 17 | input tokens = 5102 | output tokens = 17 | USD = 0.000517
[second recognition] calls = 13 | input tokens = 3822 | output tokens = 520 | USD = 0.000590
[second calibration] calls = 42900 | input tokens = 256737898 | output tokens = 42900 | USD = 25.690950
[second held-out bound] calls = 17680 | input tokens = 178906640 | output tokens = 17680 | USD = 17.897736
[second total] calls = 60838 | input tokens = 435722999 | output tokens = 61345 | USD = 43.596838
[budget] total USD = 63.122655
[budget] registration ceiling USD = 120
[budget] total at most 120 USD = True
```
