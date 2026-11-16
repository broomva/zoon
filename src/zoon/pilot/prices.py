"""Published standard token prices in US dollars, read on the recorded date."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Price:
    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    page: str
    read: str

    def dollars(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            self.input_per_million * input_tokens + self.output_per_million * output_tokens
        ) / 1_000_000


DECISION = Price(
    "jev-1.14.0", Decimal("0.042"), Decimal("0"), "https://docs.typesafe.ai/models", "2026-11-15"
)
SECOND = Price(
    "gemini-2.5-flash-lite",
    Decimal("0.10"),
    Decimal("0.40"),
    "https://cloud.google.com/vertex-ai/generative-ai/pricing",
    "2026-11-15",
)
