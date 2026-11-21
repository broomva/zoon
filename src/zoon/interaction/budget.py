"""Separate decision and chat spending bounds; no held-out records or calls."""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import cache
from itertools import product

from zoon.interaction import chat, chat_prices, compare, data, kernel, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.pilot import budget as pilot_budget
from zoon.pilot import clients, prices, prompts

CEILING = pilot_budget.CEILING
SIMULATED_GROUPS = compare.SIMULATED_GROUPS
DECISION_CAP_MULTIPLE = 4


@cache
def longest_row(game: Game) -> tuple[int, ...]:
    return max(
        product(*(range(e + 1) for e in game.endowments)),
        key=lambda row: len(prompted.round_text(game, row)),
    )


def bound_request(game: Game, seat: int, round_: int) -> prompts.Decision:
    row = longest_row(game)
    return prompted.request(
        kernel.View(
            game,
            seat,
            (kernel.Observation(tuple(enumerate(row)), game.payoffs(row)),) * (round_ - 1),
        )
    )


def estimate(groups: Sequence[Group], held: Mapping[str, int]) -> list[pilot_budget.Cost]:
    if set(held) != set(GAMES):
        raise ValueError("held-out counts must contain exactly the games in the test")
    tallies = {
        step: [0, 0, 0]
        for step in (
            "open loop",
            "recalibration",
            "disguise",
            "closed loop",
        )
    }

    def add(step: str, request: prompts.Decision, count: int = 1) -> None:
        row = tallies[step]
        row[0] += count
        row[1] += pilot_budget.input_tokens(request, "decision") * count
        row[2] += count

    for cell, game in GAMES.items():
        for r in range(1, ROUNDS + 1):
            for seat in range(game.size):
                request = bound_request(game, seat, r)
                add("open loop", request, held[cell])
                add("closed loop", request, SIMULATED_GROUPS)
    for decision in prompted.subsample(groups):
        view = prompted.training_view(decision.group, decision.seat, decision.round)
        add("recalibration", prompted.request(view))
        add("disguise", prompted.request(view, prompted.DISGUISE))
    return [pilot_budget.Cost("decision", step, *values) for step, values in tallies.items()]


@dataclass(frozen=True)
class ChatCost:
    step: str
    calls: int
    input_tokens: int
    output_tokens: int

    @property
    def dollars(self) -> Decimal:
        return chat_prices.HIGH.dollars(self.input_tokens, self.output_tokens)


def chat_bound_request(game: Game, seat: int, round_: int) -> str:
    row = longest_row(game)
    return chat.request(
        kernel.View(
            game,
            seat,
            (kernel.Observation(tuple(enumerate(row)), game.payoffs(row)),) * (round_ - 1),
        )
    )


@cache
def chat_max_estimate() -> Decimal:
    """Largest admission estimate among all registered decision and recall requests."""
    return max(
        *(chat.estimate(prompted.recall(game)) for game in GAMES.values()),
        *(
            chat.estimate(chat_bound_request(game, seat, r))
            for game in GAMES.values()
            for r in range(1, ROUNDS + 1)
            for seat in range(game.size)
        ),
    )


def chat_estimate() -> list[ChatCost]:
    calls = tokens = 0
    for game in GAMES.values():
        for r in range(1, ROUNDS + 1):
            for seat in range(game.size):
                calls += SIMULATED_GROUPS
                tokens += chat.input_tokens(chat_bound_request(game, seat, r)) * SIMULATED_GROUPS
    return [
        ChatCost("chat closed loop", calls, tokens, calls * chat.MAX_OUTPUT_TOKENS),
        ChatCost(
            "chat recall",
            len(GAMES),
            sum(chat.input_tokens(prompted.recall(game)) for game in GAMES.values()),
            len(GAMES) * chat.MAX_OUTPUT_TOKENS,
        ),
    ]


def whole_dollars(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_CEILING)


def decision_cap(rows: Sequence[pilot_budget.Cost]) -> Decimal:
    return whole_dollars(DECISION_CAP_MULTIPLE * sum((r.dollars for r in rows), Decimal()))


def chat_cap() -> Decimal:
    return whole_dollars(sum((r.dollars for r in chat_estimate()), Decimal()))


SETTINGS: tuple[tuple[str, object, str], ...] = (
    ("decision cap multiple", sys.modules[__name__], "DECISION_CAP_MULTIPLE"),
    ("rounds", sys.modules[__name__], "ROUNDS"),
    ("simulated groups per game", sys.modules[__name__], "SIMULATED_GROUPS"),
    ("recalibration count limit", prompted, "SUBSAMPLE_COUNT"),
    ("recalibration salt", prompted, "SUBSAMPLE_SALT"),
    ("disguise points multiplier", prompted, "DISGUISE"),
    ("fixture seed", prompted, "FIXTURE_SEED"),
    ("smoke games", prompted, "SMOKE_GAMES"),
    ("recalibration start coordinates", prompted, "FIT_START"),
    ("recalibration optimiser", prompted, "FIT_METHOD"),
    ("recalibration optimiser options", prompted, "FIT_OPTIONS"),
    ("recalibration gradient limit", prompted, "GRADIENT_LIMIT"),
    ("recalibration polish steps", prompted, "POLISH_STEPS"),
    ("recalibration beta lower bound", prompted, "BETA_LOWER_BOUND"),
    ("recalibration penalty", prompted, "PENALTY"),
    ("probability floor before recalibration", prompted, "PROMPTED_FLOOR"),
    ("model", prices, "DECISION.model"),
    ("input USD per million", prices, "DECISION.input_per_million"),
    ("output USD per million", prices, "DECISION.output_per_million"),
    ("price read", prices, "DECISION.read"),
    ("proposed ceiling USD", sys.modules[__name__], "CEILING"),
    ("pilot cap USD", clients, "CAP"),
    ("chat model", chat, "MODEL"),
    ("chat location", chat, "LOCATION"),
    ("chat temperature", chat, "TEMPERATURE"),
    ("chat thinking level", chat, "THINKING_LEVEL"),
    ("chat maximum output tokens", chat, "MAX_OUTPUT_TOKENS"),
    ("chat transport attempts", chat, "TRANSPORT_ATTEMPTS"),
    ("chat transport pauses", chat, "TRANSPORT_PAUSES"),
    ("chat reply attempts", chat, "REPLY_ATTEMPTS"),
    ("chat input bytes per token", chat, "INPUT_BYTES_PER_TOKEN"),
    ("chat input overhead tokens", chat, "INPUT_OVERHEAD"),
    ("chat smoke repeats", chat, "SMOKE_REPEATS"),
    ("chat smoke default", chat, "SMOKE_DEFAULT"),
    ("chat input USD per million", chat_prices, "HIGH.input_per_million"),
    ("chat output USD per million", chat_prices, "HIGH.output_per_million"),
    ("chat lower input USD per million", chat_prices, "LOW.input_per_million"),
    ("chat lower output USD per million", chat_prices, "LOW.output_per_million"),
    ("chat higher price from", chat_prices, "HIGH_FROM"),
    ("chat lower price through", chat_prices, "LOW_THROUGH"),
    ("chat price read", chat_prices, "READ"),
    ("chat price page", chat_prices, "PAGE"),
)


def settings() -> list[str]:
    lines: list[str] = []
    for label, source, path in SETTINGS:
        value = source
        for attribute in path.split("."):
            value = getattr(value, attribute)
        lines.append(f"[settings] {label} = {value}")
    return lines


def dollars(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.00000001"), rounding=ROUND_CEILING))


def report(groups: Sequence[Group], held: Mapping[str, int]) -> list[str]:
    rows = estimate(groups, held)
    total = sum((row.dollars for row in rows), Decimal())
    largest = max(rows, key=lambda row: row.dollars)
    chat_rows = chat_estimate()
    chat_total = sum((row.dollars for row in chat_rows), Decimal())
    return [
        "zoon interaction: prompted spending bound",
        *settings(),
        "[assumption] chat input uses byte-fallback tokenisation",
        "[assumption] service bounds candidate plus thinking tokens by maximum output tokens",
        f"[data] training groups = {len(groups)}",
        f"[data] held-out groups = {sum(held.values())}",
        *[
            f"[{row.step}] calls = {row.calls} | input tokens <= {row.input_tokens} | "
            f"output tokens <= {row.output_tokens} | USD <= {dollars(row.dollars)}"
            for row in rows
        ],
        f"[total] calls = {sum(row.calls for row in rows)} | "
        f"input tokens <= {sum(row.input_tokens for row in rows)} | "
        f"output tokens <= {sum(row.output_tokens for row in rows)} | USD <= {dollars(total)}",
        f"[budget] largest component = {largest.step} | USD <= {dollars(largest.dollars)}",
        f"[budget] share of proposed ceiling = {total / CEILING:.8f}",
        f"[budget] total at most proposed ceiling = {total <= CEILING}",
        *[
            f"[{row.step}] calls = {row.calls} | input tokens <= {row.input_tokens} | "
            f"output tokens <= {row.output_tokens} | USD <= {dollars(row.dollars)}"
            for row in chat_rows
        ],
        f"[chat total] calls = {sum(row.calls for row in chat_rows)} | "
        f"input tokens <= {sum(row.input_tokens for row in chat_rows)} | "
        f"output tokens <= {sum(row.output_tokens for row in chat_rows)} | "
        f"USD <= {dollars(chat_total)}",
        f"[caps] decision USD = {decision_cap(rows)}",
        f"[caps] chat USD = {whole_dollars(chat_total)}",
    ]


def main() -> None:
    print("\n".join(report(data.load(), data.held_out_counts())))


if __name__ == "__main__":
    main()
