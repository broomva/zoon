"""Standard chat token prices as listed on the recorded read date.

The later price is used for both admission and the spending bound. Thinking
is billed as output. Confirmation on the Vertex page remains a registration task.
"""

from decimal import Decimal

from zoon.pilot.prices import Price

PAGE = "https://ai.google.dev/gemini-api/docs/pricing"
READ = "2026-11-21"
HIGH_FROM = "2026-12-01"
LOW_THROUGH = "2026-11-30"
HIGH = Price("gemini-4-carbon", Decimal("4.00"), Decimal("20.00"), PAGE, READ)
LOW = Price("gemini-4-carbon", Decimal("2.00"), Decimal("10.00"), PAGE, READ)
