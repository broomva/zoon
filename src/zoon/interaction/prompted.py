"""Instructions, decision requests and recalibration for the prompted arm."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, TypedDict

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from zoon.interaction import data, kernel, measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats
from zoon.pilot import clients, prompts

DIRECTORY = data.REPO / "interaction"
SUBSAMPLE_COUNT = 512
SUBSAMPLE_SALT = "interaction-recalibration-1"
FIXTURE_SEED = 711
DISGUISE = 10
SMOKE_GAMES = ("2-AI", "4-AI")
FIT_METHOD: Literal["L-BFGS-B"] = "L-BFGS-B"


class FitOptions(TypedDict):
    ftol: float
    gtol: float
    maxiter: int


FIT_OPTIONS: FitOptions = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
FIT_START = (1.0, 0.0, 0.0, 0.0, 0.0)
FIT_BOUNDS = ((0.0, None),) + ((None, None),) * 4
PENALTY = 1e-4
FIXTURE_LABEL = "MADE-UP HISTORY: seeded; no participant record.\n\n"
ANSWERS = (
    "Nothing",
    "Some, up to a third",
    "More than a third, up to two thirds",
    "More than two thirds but not all",
    "All",
)
PAIR = """General rules
You will play 2 sessions of decision-making games.
In each session, the system will randomly match another player for you.
There are two roles in the games. You and the co-player are randomly assigned the role of player 1 and player 2.
Neither of you knows each other's true identities.
Your bonus will be determined by the choices made by you and the co-player.
Your total income = 20 Yuan (show-up fee) + {exchange} Yuan × (the cumulative points from two sessions + the survey points).

Introduction to Session 1
In the first round, you and a co-player are randomly assigned the role of player 1 and player 2.
You will repeat the game for at least 20 rounds. After the 20th round, there will be a 50% chance that the game will continue for another round. Neither of your roles will be changed during the game.
In each round, player 1 has an initial {e0} points and player 2 has an initial {e1} points. You each decide how many points you would like to contribute to the common pool and how many to keep for yourself.
For every point that player 1 contributes, the common pool will increase by {p0} points. For every point that player 2 contributes, the common pool will increase by {p1} points.
In each round, the points in the common pool will be split evenly between the two players. Your payoff per round is the sum of the points you keep for yourself in that round and the points you receive from the common pool.
At the end of each round, both of you can see the results about your contributions and payoffs in the round, the group total contribution and returns from the common pool.
"""  # noqa: E501, RUF001
FOUR = """General rules
You will play 2 sessions of decision-making games. In each session, the system will randomly match you with three other players and assign the roles of Player 1, Player 2, Player 3, and Player 4.
None of you will know each other's true identities.
Your bonus will be determined by the choices made by you and the other three players.
Your total income = 15 Yuan (show-up fee) + {exchange} Yuan × (the cumulative points from two sessions + the survey points).

Introduction to Session 1
You and the other three players will repeat the game for at least 20 rounds. After the 20th round, there will be a 50% chance that the game will continue for another round. Neither of your roles will be changed during the game.
In each round, Player 1 and Player 2 each have an initial {e0} points, and Player 3 and Player 4 each have an initial {e1} points.
Each player decides how many points to contribute to the common pool and how many to keep for themselves.
For every point that Player 1 or Player 2 contributes, the common pool will increase by {p0} points. For every point that Player 3 or Player 4 contributes, the common pool will increase by {p1} points.
In each round, the points in the common pool will be split evenly among all four players. Your payoff per round is the sum of the points you keep for yourself in that round and the points you receive from the common pool.
At the end of each round, all four players can see each other's contributions to the common pool and the payoffs in that round.
"""  # noqa: E501, RUF001


def decimal(value: Fraction) -> str:
    numerator, denominator = value.numerator, value.denominator
    check = denominator
    for factor in (2, 5):
        while check % factor == 0:
            check //= factor
    if check != 1:
        raise ValueError("points need a terminating decimal")
    sign = "-" if numerator < 0 else ""
    whole, remainder = divmod(abs(numerator), denominator)
    digits = ""
    while remainder:
        digit, remainder = divmod(remainder * 10, denominator)
        digits += str(digit)
    return f"{sign}{whole}" + (f".{digits}" if digits else "")


def instructions(game: Game, multiplier: int = 1) -> str:
    other = 1 if game.size == 2 else 2
    return (PAIR if game.size == 2 else FOUR).format(
        e0=game.endowments[0] * multiplier,
        e1=game.endowments[other] * multiplier,
        p0=decimal(game.productivities[0]),
        p1=decimal(game.productivities[other]),
        exchange=decimal(Fraction(3, 100 if game.size == 2 else 200) / multiplier),
    )


def round_text(game: Game, row: tuple[int, ...], multiplier: int = 1) -> str:
    lines = [
        f"Player {s + 1}: contribution {c * multiplier} points; "
        f"payoff {decimal(p * multiplier)} points."
        for s, (c, p) in enumerate(zip(row, game.payoffs(row), strict=True))
    ]
    if game.size == 2:
        pool = sum((p * c for p, c in zip(game.productivities, row, strict=True)), Fraction())
        lines.append(
            f"Group total contribution: {sum(row) * multiplier} points; "
            f"common pool: {decimal(pool * multiplier)} points; "
            f"return per player: {decimal(pool * multiplier / game.size)} points."
        )
    return "\n".join(lines)


def request(view: kernel.View, multiplier: int = 1) -> prompts.Decision:
    if not view.open_channel:
        raise ValueError("the prompted arm needs an open channel")
    if not 0 <= view.seat < view.game.size or not 0 <= len(view.history) < ROUNDS:
        raise ValueError("seat or round outside this test")
    history: list[str] = []
    for r, observation in enumerate(view.history, 1):
        visible = dict(observation.contributions)
        if set(visible) != set(range(view.game.size)) or observation.payoffs is None:
            raise ValueError("the prompted arm needs every contribution and payoff")
        row = tuple(visible[s] for s in range(view.game.size))
        if observation.payoffs != view.game.payoffs(row):
            raise ValueError("history payoffs differ from the game")
        history.append(f"Round {r}\n{round_text(view.game, row, multiplier)}")
    return prompts.Decision(
        instructions(view.game, multiplier)
        + f"\nThe player you are asked about is Player {view.seat + 1}.\n"
        + ("\n\n".join(history) if history else "Round 1: there is no earlier history."),
        f"What share of their initial points will Player {view.seat + 1} contribute "
        f"to the common pool in round {len(view.history) + 1}?",
        prompts.declared(ANSWERS),
    )


def training_view(group: Group, seat: int, round_: int) -> kernel.View:
    game = GAMES[group.cell]
    return kernel.View(
        game,
        seat,
        tuple(
            kernel.Observation(tuple(enumerate(row)), game.payoffs(row))
            for row in group.rounds[: round_ - 1]
        ),
    )


def _probability_logs(probabilities: Floats) -> Floats:
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.ndim == 0
        or probabilities.shape[-1] != len(ANSWERS)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities.sum(axis=-1) <= 0).any()
    ):
        raise ValueError("five finite non-negative probabilities are required")
    floored = np.maximum(probabilities, person.FLOOR)
    return np.log(floored / floored.sum(axis=-1, keepdims=True))


@dataclass(frozen=True)
class Recalibration:
    beta: float = 1.0
    free_shifts: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("inverse temperature must be finite and non-negative")
        if len(self.free_shifts) != 4 or not np.isfinite(self.shifts).all():
            raise ValueError("four finite free shifts are required")

    @property
    def shifts(self) -> Floats:
        return np.r_[self.free_shifts, -sum(self.free_shifts)]

    @property
    def numbers(self) -> Floats:
        return np.r_[self.beta, self.free_shifts]

    def apply(self, probabilities: Floats) -> Floats:
        return np.asarray(
            softmax(self.beta * _probability_logs(probabilities) + self.shifts, axis=-1),
            dtype=float,
        )


def fit(
    pairs: Sequence[tuple[Floats, int]], *, start: Sequence[float] | Floats = FIT_START
) -> Recalibration:
    if not pairs:
        raise ValueError("recalibration needs training decisions")
    logs = _probability_logs(np.array([p for p, _ in pairs]))
    choices = np.array([band for _, band in pairs])
    if ((choices < 0) | (choices >= len(ANSWERS))).any():
        raise ValueError("observed band outside the declared answers")

    def objective(numbers: Floats) -> tuple[float, Floats]:
        shifts: Floats = np.r_[numbers[1:], -numbers[1:].sum()]
        logits: Floats = numbers[0] * logs + shifts
        residual: Floats = softmax(logits, axis=1)
        loss = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(pairs)), choices]))
        loss += PENALTY * ((float(numbers[0]) - 1) ** 2 + float(np.sum(shifts**2)))
        residual[np.arange(len(pairs)), choices] -= 1
        gradient = np.r_[
            np.mean(np.sum(residual * logs, axis=1)),
            (residual[:, :4] - residual[:, 4:]).mean(axis=0),
        ]
        gradient += PENALTY * np.r_[2 * (numbers[0] - 1), 2 * (shifts[:4] - shifts[4])]
        return loss, gradient

    result = minimize(
        objective,
        np.asarray(start, dtype=float),
        jac=True,
        method=FIT_METHOD,
        bounds=FIT_BOUNDS,
        options={**FIT_OPTIONS},
    )
    if not result.success:
        raise RuntimeError(f"recalibration did not converge: {result.message}")
    values = [float(v) for v in result.x]
    return Recalibration(values[0], (values[1], values[2], values[3], values[4]))


@dataclass(frozen=True)
class Prompted:
    client: clients.Client
    recalibration: Recalibration = Recalibration()

    def __call__(self, view: kernel.View) -> Floats:
        decision = request(view)
        response = self.client.ask(decision)
        clients.validate(decision, response)
        return self.recalibration.apply(np.array([response.probabilities[a] for a in ANSWERS]))


@dataclass(frozen=True)
class TrainingDecision:
    group: Group
    seat: int
    round: int

    @property
    def digest(self) -> str:
        key = [SUBSAMPLE_SALT, self.group.cell, self.group.index, self.seat, self.round]
        return hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()


def subsample(groups: Sequence[Group]) -> list[TrainingDecision]:
    by_game: dict[str, list[TrainingDecision]] = {}
    for group in groups:
        for r in range(1, len(group.rounds) + 1):
            for seat in range(GAMES[group.cell].size):
                by_game.setdefault(group.cell, []).append(TrainingDecision(group, seat, r))
    total = sum(map(len, by_game.values()))
    count = min(SUBSAMPLE_COUNT, total)
    if not count:
        return []
    quotas = {cell: count * len(rows) // total for cell, rows in by_game.items()}
    order = sorted(by_game, key=lambda c: (-(count * len(by_game[c]) % total), c))
    for cell in order[: count - sum(quotas.values())]:
        quotas[cell] += 1
    return sorted(
        [
            row
            for cell, rows in by_game.items()
            for row in sorted(rows, key=lambda row: row.digest)[: quotas[cell]]
        ],
        key=lambda row: row.digest,
    )


def recall(game: Game) -> str:
    return (
        instructions(game)
        + "\nDo you recognise this experiment? If so, name the study and give its average "
        "contributions, in points, for each player type. Otherwise answer unknown."
    )


def fixtures() -> dict[str, prompts.Decision]:
    result: dict[str, prompts.Decision] = {}
    rng = np.random.default_rng(FIXTURE_SEED)
    for cell, game in sorted(GAMES.items()):
        group = Group(
            cell,
            "made-up",
            tuple(tuple(int(rng.integers(e + 1)) for e in game.endowments) for _ in range(ROUNDS)),
        )
        seats = {measure.seat_type(game, s): s for s in reversed(range(game.size))}
        for kind, seat in sorted(seats.items()):
            for r in (1, ROUNDS):
                decision = request(training_view(group, seat, r))
                result[f"{cell}-type-{kind}-round-{r}.json"] = replace(
                    decision,
                    state=FIXTURE_LABEL + decision.state,
                )
    return result


def write_fixtures() -> None:
    instruction_dir, request_dir = DIRECTORY / "instructions", DIRECTORY / "requests"
    instruction_dir.mkdir(exist_ok=True)
    request_dir.mkdir(exist_ok=True)
    digests: dict[str, str] = {}
    for cell, game in sorted(GAMES.items()):
        name, content = f"{cell}.txt", instructions(game).encode()
        (instruction_dir / name).write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    (instruction_dir / "sha256.json").write_bytes((json.dumps(digests, indent=2) + "\n").encode())
    for name, decision in fixtures().items():
        (request_dir / name).write_bytes(prompts.decision_bytes(decision))


def smoke(client: clients.Client) -> list[str]:
    lines: list[str] = []
    for cell in SMOKE_GAMES:
        for r in (1, ROUNDS):
            name = f"{cell}-type-0-round-{r}.json"
            value = json.loads((DIRECTORY / "requests" / name).read_text())
            decision = prompts.Decision(
                value["state"].removeprefix(FIXTURE_LABEL),
                value["question"],
                tuple(prompts.Answer(**a) for a in value["answers"]),
            )
            response = client.ask(decision)
            clients.validate(decision, response)
            total = (
                sum(response.returned_probabilities.values())
                if response.returned_probabilities is not None
                else response.share
                if response.share is not None
                else sum(response.probabilities.values())
            )
            lines.append(
                f"[smoke {name}] version = {response.version} | "
                f"probabilities = {[response.probabilities[a] for a in ANSWERS]} | "
                f"sum before division = {total} | input tokens = {response.input_tokens} | "
                f"output tokens = {response.output_tokens} | "
                f"USD = {clients.cost(client.family, response):.8f}"
            )
    return lines


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixtures", "smoke"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stub", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args(arguments)
    if args.command == "fixtures":
        write_fixtures()
    else:
        client = clients.DecisionClient() if args.live else clients.StubClient("decision")
        print("\n".join(smoke(client)))


if __name__ == "__main__":
    main()
