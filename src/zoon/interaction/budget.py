"""Decision-model spending bound from training groups and held-out counts."""

import sys
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from functools import cache
from itertools import product

from zoon.interaction import compare, data, kernel, person, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.pilot import budget as pilot_budget
from zoon.pilot import clients, prices, prompts

CEILING = pilot_budget.CEILING
SIMULATED_GROUPS = compare.SIMULATED_GROUPS


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


SETTINGS: tuple[tuple[str, object, str], ...] = (
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
    ("recalibration bounds", prompted, "FIT_BOUNDS"),
    ("recalibration penalty", prompted, "PENALTY"),
    ("probability floor before recalibration", person, "FLOOR"),
    ("model", prices, "DECISION.model"),
    ("input USD per million", prices, "DECISION.input_per_million"),
    ("output USD per million", prices, "DECISION.output_per_million"),
    ("price read", prices, "DECISION.read"),
    ("proposed ceiling USD", sys.modules[__name__], "CEILING"),
    ("pilot cap USD", clients, "CAP"),
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
    return [
        "zoon interaction: prompted spending bound",
        *settings(),
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
    ]


def main() -> None:
    print("\n".join(report(data.load(), data.held_out_counts())))


if __name__ == "__main__":
    main()
