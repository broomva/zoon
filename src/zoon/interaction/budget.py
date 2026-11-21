"""Decision-model spending bound from training groups and held-out counts."""

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
RECALL_OUTPUT_TOKENS = 40
RECALL_REQUESTS_PER_GAME = 1


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
    tallies = {
        step: [0, 0, 0]
        for step in (
            "open loop",
            "recalibration",
            "disguise",
            "closed loop",
            "recall",
        )
    }

    def add(step: str, request: prompts.Decision | str, count: int = 1) -> None:
        row = tallies[step]
        row[0] += count
        row[1] += pilot_budget.input_tokens(request, "decision") * count
        row[2] += (RECALL_OUTPUT_TOKENS if isinstance(request, str) else 1) * count

    for cell, game in GAMES.items():
        for r in range(1, ROUNDS + 1):
            for seat in range(game.size):
                request = bound_request(game, seat, r)
                add("open loop", request, held.get(cell, 0))
                add("closed loop", request, SIMULATED_GROUPS)
        add("recall", prompted.recall(game), RECALL_REQUESTS_PER_GAME)
    for decision in prompted.subsample(groups):
        view = prompted.training_view(decision.group, decision.seat, decision.round)
        add("recalibration", prompted.request(view))
        add("disguise", prompted.request(view, prompted.DISGUISE))
    return [pilot_budget.Cost("decision", step, *values) for step, values in tallies.items()]


def settings() -> list[str]:
    identity = prompted.Recalibration()
    values: dict[str, object] = {
        "rounds": ROUNDS,
        "simulated groups per game": SIMULATED_GROUPS,
        "recalibration count limit": prompted.SUBSAMPLE_COUNT,
        "recalibration salt": prompted.SUBSAMPLE_SALT,
        "subsample reason": "bounded training supervision before registration",
        "simulation count reason": "same per-game count as the coded comparison",
        "hash coordinates": "game label; group index; zero-based seat; one-based round",
        "quota allocation": "largest remainder; ties by game label",
        "disguise points multiplier": prompted.DISGUISE,
        "fixture seed": prompted.FIXTURE_SEED,
        "fixture seed reason": "fixed made-up histories for inspection",
        "salt reason": "fixed ordering before model probabilities are obtained",
        "quota reason": "integer allocation within one decision of each proportional quota",
        "smoke games": ", ".join(prompted.SMOKE_GAMES),
        "smoke games reason": "the two games with printed instructions",
        "recalibration optimiser reason": "analytic likelihood gradient; no penalty",
        "reproducibility reason": "compare all fitted coordinates across starts and row order",
        "recall allowance reason": "one wording per game; free-text renderer only",
        "recalibration start coordinates": prompted.FIT_START,
        "recalibration optimiser": prompted.FIT_METHOD,
        "recalibration gradient tolerance": prompted.FIT_OPTIONS["gtol"],
        "recalibration iteration limit": prompted.FIT_OPTIONS["maxiter"],
        "recalibration objective": "mean negative log likelihood; no penalty",
        "recalibration coordinates": "log temperature; four shifts; fifth is minus their sum",
        "probability floor before recalibration": person.FLOOR,
        "identity temperature": identity.temperature,
        "identity shifts": ", ".join(f"{v:g}" for v in identity.shifts),
        "reproducibility start nudge": prompted.START_NUDGE,
        "reproducibility tolerance": prompted.REPRODUCIBILITY_TOLERANCE,
        "recall requests per game": RECALL_REQUESTS_PER_GAME,
        "recall output token bound": RECALL_OUTPUT_TOKENS,
        "history bound": "longest legal round rendering, repeated; all earlier rounds",
        "model": prices.DECISION.model,
        "input USD per million": prices.DECISION.input_per_million,
        "output USD per million": prices.DECISION.output_per_million,
        "price read": prices.DECISION.read,
        "proposed ceiling USD": CEILING,
        "pilot cap USD": clients.CAP,
    }
    return [f"[settings] {key} = {value}" for key, value in values.items()]


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
