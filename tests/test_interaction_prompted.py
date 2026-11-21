"""Prompted requests on made-up groups and the stub client."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.optimize
from scipy.special import softmax

from tests.test_interaction_kernel import training
from zoon.interaction import kernel, measure, person, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS
from zoon.pilot import clients, prices, prompts


def test_instruction_parameters_and_aligned_worked_figures() -> None:
    for cell, game in GAMES.items():
        text = prompted.instructions(game)
        other = 1 if game.size == 2 else 2
        if game.size == 2:
            assert f"player 1 has an initial {game.endowments[0]} points" in text
            assert f"player 2 has an initial {game.endowments[other]} points" in text
            for seat in range(2):
                assert (
                    f"player {seat + 1} contributes, the common pool will increase by "
                    f"{prompted.decimal(game.productivities[seat])} points"
                ) in text
            assert "20 Yuan (show-up fee) + 0.03 Yuan" in text
            assert "group total contribution and returns from the common pool" in text
        else:
            assert f"Player 1 and Player 2 each have an initial {game.endowments[0]} points" in text
            assert (
                f"Player 3 and Player 4 each have an initial {game.endowments[other]} points"
                in text
            )
            for s, label in ((0, "Player 1 or Player 2"), (2, "Player 3 or Player 4")):
                assert (
                    f"{label} contributes, the common pool will increase by "
                    f"{prompted.decimal(game.productivities[s])} points"
                ) in text
            assert "15 Yuan (show-up fee) + 0.015 Yuan" in text
            assert "all four players can see each other's contributions" in text
        assert "at least 20 rounds" in text
        assert "50% chance that the game will continue for another round" in text
        assert "Neither of your roles will be changed" in text
        assert "Exercise" not in text and "Introduction to Session 2" not in text
        if cell.endswith("AI"):
            # Exercise 2: every member gives the endowment. Figures computed independently.
            pool, share = (84, 42) if game.size == 2 else (336, 84)
            assert (
                sum(p * e for p, e in zip(game.productivities, game.endowments, strict=True))
                == pool
            )
            assert game.payoffs(game.endowments) == (Fraction(share),) * game.size
            assert f"payoff {share} points" in prompted.round_text(game, game.endowments)


def test_instruction_and_request_bytes_and_digests() -> None:
    directory = prompted.DIRECTORY / "instructions"
    digests = json.loads((directory / "sha256.json").read_text())
    assert set(digests) == {f"{cell}.txt" for cell in GAMES}
    for cell, game in GAMES.items():
        content = prompted.instructions(game).encode()
        assert (directory / f"{cell}.txt").read_bytes() == content
        assert digests[f"{cell}.txt"] == hashlib.sha256(content).hexdigest()
    fixture_dir = prompted.DIRECTORY / "requests"
    generated = prompted.fixtures()
    assert set(generated) == {p.name for p in fixture_dir.glob("*.json")}
    for name, decision in generated.items():
        assert (fixture_dir / name).read_bytes() == prompts.decision_bytes(decision)
        assert decision.state.startswith("MADE-UP HISTORY: seeded; no participant record.")
    expected = {
        f"{cell}-type-{kind}-round-{r}.json"
        for cell, game in GAMES.items()
        for kind in {measure.seat_type(game, s) for s in range(game.size)}
        for r in (1, ROUNDS)
    }
    assert set(generated) == expected
    late = generated["4-AI-type-1-round-20.json"]
    assert late.state.count("\nRound ") == 19
    assert "Player 3." in late.state and "round 20?" in late.question


def test_history_exact_points_seat_and_no_target_answer() -> None:
    game = GAMES["4-AI"]
    group = Group(game.cell, "example", ((1, 2, 3, 4), (36, 36, 12, 12)))
    decision = prompted.request(prompted.training_view(group, 2, 2))
    assert "The player you are asked about is Player 3." in decision.state
    # Hand calculation: pool = 29.6; share = 7.4; kept = (35, 34, 9, 8).
    for seat, c in enumerate((1, 2, 3, 4)):
        actual = ("42.4", "41.4", "16.4", "15.4")[seat]
        assert (
            f"Player {seat + 1}: contribution {c} points; payoff {actual} points." in decision.state
        )
    assert "contribution 36 points" not in decision.state
    assert "Round 2\n" not in decision.state
    first = prompted.request(kernel.View(game, 0, ()))
    assert "Round 1: there is no earlier history." in first.state
    assert "round 1?" in first.question
    pair = prompted.round_text(GAMES["2-AI"], (1, 2))
    assert "payoff 37.25 points" in pair and "payoff 12.25 points" in pair
    assert (
        "Group total contribution: 3 points; common pool: 4.5 points; return per player: 2.25"
        in pair
    )
    assert prompted.decimal(Fraction(1, 40)) == "0.025"
    assert prompted.decimal(Fraction(-201, 40)) == "-5.025"
    with pytest.raises(ValueError, match="terminating"):
        prompted.decimal(Fraction(1, 3))


def test_answer_order_and_response_mapping() -> None:
    view = kernel.View(GAMES["2-FE"], 0, ())
    decision = prompted.request(view)
    assert tuple(a.answer for a in decision.answers) == (
        "Nothing",
        "Some, up to a third",
        "More than a third, up to two thirds",
        "More than two thirds but not all",
        "All",
    )
    for points, band in ((0, 0), (1, 1), (8, 1), (9, 2), (16, 2), (17, 3), (23, 3), (24, 4)):
        assert measure.band(points, 24) == band
    p = np.array([0.05, 0.1, 0.15, 0.3, 0.4])
    response = clients.Response(
        dict(reversed(list(zip(prompted.ANSWERS, p, strict=True)))),
        None,
        "",
        prices.DECISION.model,
        5,
        1,
    )
    client = clients.StubClient("decision", {clients.key(decision): response})
    assert prompted.Prompted(client)(view) == pytest.approx(p)
    calibrated = prompted.Recalibration(0.2, (0.1, 0.2, -0.1, 0.3))
    assert prompted.Prompted(client, calibrated)(view) == pytest.approx(calibrated.apply(p))


class RecordingClient(clients.StubClient):
    def __init__(self) -> None:
        super().__init__("decision")
        self.requests: list[clients.Request] = []
        self.recorded: dict[str, clients.Response] = {}

    def ask(self, request: clients.Request) -> clients.Response:
        self.requests.append(request)
        return super().ask(request)


def test_closed_channel_refuses_before_call_and_agent_has_no_seat_memory() -> None:
    client = RecordingClient()
    agent = prompted.Prompted(client)
    game = GAMES["2-AI"]
    with pytest.raises(ValueError, match="open channel"):
        agent(kernel.View(game, 0, (), False))
    assert client.requests == []
    view = prompted.training_view(Group(game.cell, "x", ((1, 2),) * ROUNDS), 1, 3)
    expected = agent(view)
    agent(kernel.View(GAMES["4-MI"], 3, ()))
    assert np.array_equal(agent(view), expected)
    assert client.requests[0] == client.requests[-1] == prompted.request(view)
    missing = replace(view.history[0], contributions=((1, 2),), payoffs=None)
    with pytest.raises(ValueError, match="every contribution and payoff"):
        agent(replace(view, history=(missing,)))


def test_kernel_stub_calls_each_seat_each_round_and_replays() -> None:
    amounts = kernel.Amounts.build(training())
    for game in GAMES.values():
        client = RecordingClient()
        agent = prompted.Prompted(client)
        run = kernel.play(game, [agent] * game.size, amounts, 17, "stub")
        assert len(client.requests) == game.size * ROUNDS
        for r in range(ROUNDS):
            for seat in range(game.size):
                decision = client.requests[r * game.size + seat]
                assert isinstance(decision, prompts.Decision)
                assert decision == prompted.request(
                    prompted.training_view(
                        Group(game.cell, "stub", run.rounds),
                        seat,
                        r + 1,
                    )
                )
        assert kernel.replay(run.bytes).bytes == run.bytes
        assert kernel.play(game, [agent] * game.size, amounts, 17, "stub").bytes == run.bytes
        closed = RecordingClient()
        with pytest.raises(ValueError, match="open channel"):
            kernel.play(
                game,
                [prompted.Prompted(closed)] * game.size,
                amounts,
                17,
                "stub",
                open_channel=False,
            )
        assert closed.requests == []


def test_identity_and_zero_sum_shifts() -> None:
    p = np.array([[0, 0.1, 0.2, 0.3, 0.4], [0.2] * 5])
    assert prompted.Recalibration().apply(p) == pytest.approx(
        np.array([kernel.distribution(row) for row in p]), abs=1e-15
    )
    value = prompted.Recalibration(0.5, (0.2, -0.3, 0.4, -0.1))
    assert value.shifts.sum() == pytest.approx(0, abs=1e-15)
    assert value.beta == pytest.approx(0.5)
    assert value.numbers.shape == (5,)
    assert value.apply(p) == pytest.approx(
        softmax(
            np.log(np.array([kernel.distribution(row) for row in p])) / 2 + value.shifts,
            axis=1,
        ),
        abs=1e-15,
    )


def synthetic_pairs() -> tuple[list[tuple[measure.Floats, int]], prompted.Recalibration]:
    rng = np.random.default_rng(27)
    weights = np.array([1, 2, 3, 4, 5])
    shifts = np.log(weights) - np.log(weights).mean()
    known = prompted.Recalibration(
        2.0, (float(shifts[0]), float(shifts[1]), float(shifts[2]), float(shifts[3]))
    )
    pairs: list[tuple[measure.Floats, int]] = []
    for _ in range(20):
        raw = rng.integers(1, 15, 5)
        p = np.sqrt(raw) / np.sqrt(raw).sum()
        for band, count in enumerate(raw * weights):
            pairs.extend((p, band) for _ in range(int(count)))
    return pairs, known


def test_fit_recovers_known_numbers_and_reproduces_identified_coordinates() -> None:
    pairs, known = synthetic_pairs()
    reference = prompted.fit(pairs)
    assert reference.numbers == pytest.approx(known.numbers, abs=0.01)
    assert reference.shifts.sum() == pytest.approx(0, abs=1e-15)
    for other in (
        prompted.fit(pairs, start=np.array(prompted.FIT_START) + 0.1),
        prompted.fit(list(reversed(pairs))),
    ):
        assert other.numbers.shape == (5,)
        assert np.max(np.abs(other.numbers - reference.numbers)) < 1e-4
    # All five coordinates change the probabilities on this varying input.
    probes = np.array([p for p, _ in pairs[::100]])
    columns: list[measure.Floats] = []
    for j in range(5):
        numbers = reference.numbers.copy()
        numbers[j] += 1e-5
        changed = prompted.Recalibration(
            float(numbers[0]),
            (float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])),
        )
        columns.append(((changed.apply(probes) - reference.apply(probes)) / 1e-5).ravel())
    assert np.linalg.matrix_rank(np.column_stack(columns), tol=1e-6) == 5


def test_subsample_hash_quotas_order_and_limit() -> None:
    groups = [
        Group(cell, str(i), ((0,) * GAMES[cell].size,) * ROUNDS)
        for cell, count in (("2-AI", 6), ("4-AI", 4), ("2-FE", 2))
        for i in range(count)
    ]
    rows = prompted.subsample(groups)
    assert rows == prompted.subsample(list(reversed(groups))) == prompted.subsample(groups)
    assert len(rows) == prompted.SUBSAMPLE_COUNT
    # 512 * (240, 320, 80) / 640; largest remainder gives (192, 256, 64).
    assert {
        cell: sum(row.group.cell == cell for row in rows) for cell in ("2-AI", "4-AI", "2-FE")
    } == {
        "2-AI": 192,
        "4-AI": 256,
        "2-FE": 64,
    }
    assert [r.digest for r in rows] == sorted(r.digest for r in rows)
    for cell in ("2-AI", "4-AI", "2-FE"):
        candidates = [
            prompted.TrainingDecision(g, s, r)
            for g in groups
            if g.cell == cell
            for s in range(GAMES[cell].size)
            for r in range(1, ROUNDS + 1)
        ]
        selected = [r for r in rows if r.group.cell == cell]
        assert selected == sorted(candidates, key=lambda r: r.digest)[: len(selected)]
    first = rows[0]
    key = [prompted.SUBSAMPLE_SALT, first.group.cell, first.group.index, first.seat, first.round]
    assert (
        first.digest == hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
    )
    small = prompted.subsample(groups[-1:])
    assert len(small) == ROUNDS * 2
    assert prompted.subsample([]) == []


def test_disguise_scales_points_and_preserves_productivities() -> None:
    group = Group("2-AI", "x", ((1, 2),) * ROUNDS)
    view = prompted.training_view(group, 1, 2)
    text = prompted.request(view, prompted.DISGUISE).state
    assert "initial 360 points" in text and "initial 120 points" in text
    assert "increase by 1.9 points" in text and "increase by 1.3 points" in text
    assert "contribution 10 points; payoff 372.5 points" in text
    assert "contribution 20 points; payoff 122.5 points" in text
    assert "0.003 Yuan" in text
    assert (
        "Group total contribution: 30 points; common pool: 45 points; "
        "return per player: 22.5 points."
    ) in text


def test_recall_instructions_and_question() -> None:
    for game in GAMES.values():
        text = prompted.recall(game)
        assert text == (
            prompted.instructions(game)
            + "\nDo you recognise this experiment? If so, name the study and give its average "
            "contributions, in points, for each player type. Otherwise answer unknown."
        )


def test_smoke_stub_committed_requests_and_usage(capsys: pytest.CaptureFixture[str]) -> None:
    client = RecordingClient()
    lines = prompted.smoke(client)
    expected = [
        "2-AI-type-0-round-1.json",
        "2-AI-type-0-round-20.json",
        "4-AI-type-0-round-1.json",
        "4-AI-type-0-round-20.json",
    ]
    assert len(lines) == len(expected) == len(client.requests)
    for name, decision, line in zip(expected, client.requests, lines, strict=True):
        assert isinstance(decision, prompts.Decision)
        assert prompts.decision_bytes(decision) == prompts.decision_bytes(
            replace(
                prompted.fixtures()[name],
                state=prompted.fixtures()[name].state.removeprefix(prompted.FIXTURE_LABEL),
            )
        )
        response = clients.StubClient("decision").ask(decision)
        assert f"version = {response.version}" in line
        assert f"probabilities = {[response.probabilities[a] for a in prompted.ANSWERS]}" in line
        assert f"sum before division = {sum(response.probabilities.values())}" in line
        assert f"input tokens = {response.input_tokens}" in line
        assert f"output tokens = {response.output_tokens}" in line
        assert f"USD = {clients.cost('decision', response):.8f}" in line
    prompted.main(["smoke", "--stub"])
    assert capsys.readouterr().out == "\n".join(lines) + "\n"
    assert not list((prompted.DIRECTORY / "requests").glob("*transcript*"))


def test_agent_floors_model_zeros_before_recalibration() -> None:
    view = kernel.View(GAMES["2-FE"], 0, ())
    decision = prompted.request(view)
    values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    response = clients.Response(
        dict(zip(prompted.ANSWERS, values, strict=True)), None, "", prices.DECISION.model, 5, 1
    )
    client = clients.StubClient("decision", {clients.key(decision): response})
    assert prompted.Prompted(client)(view) == pytest.approx(kernel.distribution(values), abs=1e-15)


def test_subsample_largest_remainder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompted, "SUBSAMPLE_COUNT", 511)
    groups = [
        Group(cell, str(i), ((0,) * GAMES[cell].size,) * ROUNDS)
        for cell, count in (("2-AI", 6), ("4-AI", 4), ("2-FE", 2))
        for i in range(count)
    ]
    result = prompted.subsample(groups)
    assert {
        cell: sum(d.group.cell == cell for d in result) for cell in ("2-AI", "4-AI", "2-FE")
    } == {
        "2-AI": 192,
        "4-AI": 255,
        "2-FE": 64,
    }
    assert len(result) == 511


def test_smoke_raw_sum_and_no_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RecordingClient()
    for name, decision in prompted.fixtures().items():
        if name.startswith(("2-AI-type-0-", "4-AI-type-0-")):
            raw: dict[str, float] = dict(
                zip(prompted.ANSWERS, (0.1, 0.2, 0.3, 0.1, 0.29), strict=True)
            )
            total = sum(raw.values())
            decision = replace(decision, state=decision.state.removeprefix(prompted.FIXTURE_LABEL))
            client.recorded[clients.key(decision)] = clients.Response(
                {a: p / total for a, p in raw.items()},
                total,
                "",
                "reported-version",
                37,
                4,
                raw,
            )

    def refuse_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected write")

    monkeypatch.setattr(Path, "write_text", refuse_write)
    monkeypatch.setattr(Path, "write_bytes", refuse_write)
    monkeypatch.setattr(clients.Transcript, "__init__", refuse_write)
    lines = prompted.smoke(client)
    assert len(lines) == 4
    for line in lines:
        assert "version = reported-version" in line
        assert "sum before division = 0.99" in line
        assert "input tokens = 37 | output tokens = 4" in line
        assert "USD = 0.00000155" in line


def test_subsample_quota_ties_use_game_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompted, "SUBSAMPLE_COUNT", 3)
    groups = [Group(cell, "x", ((0, 0),) * ROUNDS) for cell in ("2-FE", "2-AI")]
    result = prompted.subsample(groups)
    assert [sum(d.group.cell == cell for d in result) for cell in ("2-AI", "2-FE")] == [2, 1]
    assert result == prompted.subsample(list(reversed(groups)))


def test_fixture_writer_regenerates_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = prompted.DIRECTORY
    monkeypatch.setattr(prompted, "DIRECTORY", tmp_path)

    def refuse_text_write(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixtures must be written as bytes")

    monkeypatch.setattr(Path, "write_text", refuse_text_write)
    prompted.write_fixtures()
    for directory in ("instructions", "requests"):
        expected = {p.name: p.read_bytes() for p in (original / directory).iterdir()}
        written = {p.name: p.read_bytes() for p in (tmp_path / directory).iterdir()}
        assert written == expected


@pytest.mark.parametrize("seat, rounds", [(-1, 0), (2, 0), (0, ROUNDS)])
def test_request_refuses_seat_or_round_outside_test(seat: int, rounds: int) -> None:
    game = GAMES["2-FE"]
    observation = kernel.Observation(((0, 0), (1, 0)), game.payoffs((0, 0)))
    with pytest.raises(ValueError, match="seat or round"):
        prompted.request(kernel.View(game, seat, (observation,) * rounds))


@pytest.mark.parametrize("damage", ["seat", "payoffs", "inconsistent"])
def test_request_refuses_incomplete_or_inconsistent_history(damage: str) -> None:
    game = GAMES["2-FE"]
    observation = kernel.Observation(((0, 1), (1, 2)), game.payoffs((1, 2)))
    if damage == "seat":
        observation = replace(observation, contributions=((1, 2),))
    elif damage == "payoffs":
        observation = replace(observation, payoffs=None)
    else:
        observation = replace(observation, payoffs=game.payoffs((0, 0)))
    with pytest.raises(
        ValueError, match="history payoffs" if damage == "inconsistent" else "every contribution"
    ):
        prompted.request(kernel.View(game, 0, (observation,)))


class InvalidClient:
    family: clients.Family = "decision"

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def ask(self, request: clients.Request) -> clients.Response:
        return clients.Response(self.probabilities, None, "", "example", 1, 1)


@pytest.mark.parametrize("damage", ["sum", "missing", "extra"])
@pytest.mark.parametrize("smoke", [False, True])
def test_prompted_and_smoke_validate_raw_responses(damage: str, smoke: bool) -> None:
    probabilities: dict[str, float] = dict.fromkeys(prompted.ANSWERS, 0.2)
    if damage == "sum":
        probabilities[prompted.ANSWERS[0]] = 0.1
    elif damage == "missing":
        probabilities.pop(prompted.ANSWERS[0])
        probabilities[prompted.ANSWERS[1]] += 0.2
    else:
        probabilities["unexpected"] = 0.1
        probabilities[prompted.ANSWERS[0]] -= 0.1
    client = InvalidClient(probabilities)
    with pytest.raises(clients.StepFailure):
        if smoke:
            prompted.smoke(client)
        else:
            prompted.Prompted(client)(kernel.View(GAMES["2-FE"], 0, ()))


@pytest.mark.parametrize("arguments", [["smoke"], ["smoke", "--stub"], ["smoke", "--live"]])
def test_smoke_client_selection_without_constructing_live_client(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen: list[str] = []
    stub = clients.StubClient("decision")

    def offline() -> clients.StubClient:
        chosen.append("live selected")
        return stub

    def inspect_client(client: clients.Client) -> list[str]:
        assert isinstance(client, clients.StubClient)
        assert client.family == "decision"
        return []

    monkeypatch.setattr(clients, "DecisionClient", offline)
    monkeypatch.setattr(prompted, "smoke", inspect_client)
    prompted.main(arguments)
    assert chosen == (["live selected"] if "--live" in arguments else [])


def test_smoke_requests_equal_rendered_made_up_groups() -> None:
    rng = np.random.default_rng(prompted.FIXTURE_SEED)
    groups = {
        cell: Group(
            cell,
            "made-up",
            tuple(tuple(int(rng.integers(e + 1)) for e in game.endowments) for _ in range(ROUNDS)),
        )
        for cell, game in sorted(GAMES.items())
    }
    client = RecordingClient()
    prompted.smoke(client)
    expected = [
        prompted.request(prompted.training_view(groups[cell], 0, round_))
        for cell in prompted.SMOKE_GAMES
        for round_ in (1, ROUNDS)
    ]
    assert [
        prompts.decision_bytes(d) for d in client.requests if isinstance(d, prompts.Decision)
    ] == [prompts.decision_bytes(d) for d in expected]


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_recalibration_refuses_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="inverse temperature"):
        prompted.Recalibration(beta)


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), -float("inf")])
def test_recalibration_refuses_nonfinite_shifts(shift: float) -> None:
    with pytest.raises(ValueError, match="finite free shifts"):
        prompted.Recalibration(1, (shift, 0, 0, 0))


@pytest.mark.parametrize(
    "values", [[1, 2, 3, 4], [0] * 5, [-1, 1, 1, 1, 1], [float("nan")] * 5, [float("inf")] * 5]
)
def test_apply_refuses_invalid_probabilities(values: list[float]) -> None:
    with pytest.raises(ValueError, match="five finite non-negative"):
        prompted.Recalibration().apply(np.array(values))


def test_apply_accepts_integers_and_zero_beta() -> None:
    assert prompted.Recalibration().apply(np.array([0, 1, 2, 3, 4])) == pytest.approx(
        np.maximum([0, 1, 2, 3, 4], person.FLOOR) / (10 + person.FLOOR)
    )
    assert prompted.Recalibration(0).apply(np.array([0, 1, 2, 3, 4])) == pytest.approx([0.2] * 5)


def test_fit_refuses_empty_input_and_outside_band() -> None:
    with pytest.raises(ValueError, match="training decisions"):
        prompted.fit([])
    for band in (-1, 5):
        with pytest.raises(ValueError, match="observed band"):
            prompted.fit([(np.full(5, 0.2), band)])


def test_fit_objective_gradient_hessian_start_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = np.array([[0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.1, 0.1, 0.2], [0.2] * 5])
    choices = np.array([0, 3, 4])
    start = np.array([1.1, 0.1, 0.1, 0.1, 0.1])
    called = False
    minimize: Any = prompted.minimize

    def inspect_objective(fun: Any, x0: measure.Floats, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        if len(x0) == 4:
            return minimize(fun, x0, **kwargs)
        assert np.array_equal(x0, start)
        assert set(kwargs) == {"jac", "hess", "method", "options"}
        assert kwargs["jac"] is True
        assert kwargs["method"] == "trust-exact"
        assert kwargs["options"] == {"gtol": 1e-6, "maxiter": 1000}
        for numbers in (
            start,
            np.array([0.3, -0.8, 0.7, 0.2, -0.5]),
            np.array([2.0, 0.1, -0.5, 0.9, 0.3]),
        ):
            shifts = np.r_[numbers[1:], -numbers[1:].sum()]
            logs = np.log(
                np.maximum(probabilities, 1e-9)
                / np.maximum(probabilities, 1e-9).sum(axis=1, keepdims=True)
            )
            expected = -np.log(
                softmax(numbers[0] * logs + shifts, axis=1)[np.arange(3), choices]
            ).mean()
            expected += 1e-4 * ((numbers[0] - 1) ** 2 + (shifts**2).sum())
            loss, gradient = fun(numbers)
            assert loss == pytest.approx(expected, abs=1e-12)
            step = 1e-6
            finite = np.array(
                [
                    (fun(numbers + step * axis)[0] - fun(numbers - step * axis)[0]) / (2 * step)
                    for axis in np.eye(5)
                ]
            )
            assert gradient == pytest.approx(finite, abs=1e-8)
            finite_hessian = np.column_stack(
                [
                    (fun(numbers + step * axis)[1] - fun(numbers - step * axis)[1]) / (2 * step)
                    for axis in np.eye(5)
                ]
            )
            hessian = kwargs["hess"](numbers)
            assert hessian == pytest.approx(finite_hessian, abs=1e-8)
            assert np.linalg.eigvalsh(hessian).min() >= 2 * prompted.PENALTY - 1e-12
        return minimize(fun, x0, **kwargs)

    monkeypatch.setattr(prompted, "minimize", inspect_objective)
    result = prompted.fit(list(zip(probabilities, choices.tolist(), strict=True)), start=start)
    assert_stationary(probabilities, choices, result)
    assert called


def test_fit_raises_unsuccessful_optimizer_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsuccessful(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, message="stopped before convergence")

    monkeypatch.setattr(prompted, "minimize", unsuccessful)
    with pytest.raises(RuntimeError, match="stopped before convergence"):
        prompted.fit([(np.full(5, 0.2), 0)])


def assert_stationary(
    probabilities: measure.Floats,
    choices: np.ndarray[Any, np.dtype[np.int64]],
    result: prompted.Recalibration,
) -> None:
    floored = np.maximum(probabilities, person.FLOOR)
    logs = np.log(floored / floored.sum(axis=1, keepdims=True))
    residual = softmax(result.beta * logs + result.shifts, axis=1)
    residual[np.arange(len(choices)), choices] -= 1
    gradient = np.r_[
        np.mean(np.sum(residual * logs, axis=1)),
        (residual[:, :4] - residual[:, 4:]).mean(axis=0),
    ]
    gradient += (
        prompted.PENALTY * np.r_[2 * (result.beta - 1), 2 * (result.shifts[:4] - result.shifts[4])]
    )
    assert np.isfinite(result.numbers).all()
    assert np.max(np.abs(gradient[1:])) <= prompted.GRADIENT_LIMIT
    if result.beta == 0.0:
        assert gradient[0] >= -prompted.GRADIENT_LIMIT
    else:
        assert abs(gradient[0]) <= prompted.GRADIENT_LIMIT


def reproducibility_case(
    regime: str, seed: int
) -> tuple[measure.Floats, np.ndarray[Any, np.dtype[np.int64]], np.random.Generator]:
    rng = np.random.default_rng([seed, 20261120])
    n = (5, 20, 50, 512, 5000)[seed % 5]
    probabilities = rng.dirichlet(np.ones(5) * rng.uniform(0.3, 3), size=n)
    choices = rng.integers(0, 5, n)
    if regime == "well specified":
        true = softmax(
            rng.uniform(0.4, 2.5) * np.log(probabilities) + rng.normal(0, 0.5, 5), axis=1
        )
        choices = np.array([rng.choice(5, p=row) for row in true])
    elif regime == "exact zeros":
        probabilities[rng.random(probabilities.shape) < 0.2] = 0
        probabilities[probabilities.sum(axis=1) == 0] = 0.2
    elif regime == "one-hot rows":
        bands = rng.integers(0, 5, n) if seed % 2 else np.full(n, rng.integers(0, 5))
        probabilities = np.eye(5)[bands]
    elif regime == "identical non-uniform rows":
        probabilities = np.tile(rng.dirichlet(np.ones(5) * rng.uniform(0.2, 2)), (n, 1))
    elif regime == "unobserved bands":
        # Cycle through one, two, three and four bands never observed.
        choices = rng.choice(rng.choice(5, size=4 - seed % 4, replace=False), n)
    elif regime == "anti-informative":
        choices = np.argmin(probabilities, axis=1)
    elif regime == "extremely sharp":
        probabilities = softmax(rng.normal(0, 50, (n, 5)), axis=1)
    elif regime == "perfectly predictive":
        probabilities = softmax(rng.normal(0, 3, (n, 5)), axis=1)
        choices = np.argmax(probabilities, axis=1)
    else:
        raise AssertionError(regime)
    return probabilities, choices, rng


@pytest.mark.parametrize(
    "regime",
    [
        "well specified",
        "exact zeros",
        "one-hot rows",
        "identical non-uniform rows",
        "unobserved bands",
        "anti-informative",
        "extremely sharp",
        "perfectly predictive",
    ],
)
def test_fit_reproduces_across_regimes(
    regime: str, record_property: Callable[[str, object], None]
) -> None:
    worst = 0.0
    for seed in range(25):
        probabilities, choices, rng = reproducibility_case(regime, seed)
        pairs = list(zip(probabilities, choices.tolist(), strict=True))
        starts = (
            np.array(prompted.FIT_START),
            np.array(prompted.FIT_START) + 0.1,
            rng.normal(0, 5, 5),
        )
        results = [prompted.fit(pairs, start=start) for start in starts]
        for result in results:
            assert_stationary(probabilities, choices, result)
        difference = float(np.max(np.ptp([result.numbers for result in results], axis=0)))
        worst = max(worst, difference)
        assert difference < 1e-5, (regime, seed, difference)
    record_property("worst_coordinate_difference", worst)


def test_identical_rows_reproduction(record_property: Callable[[str, object], None]) -> None:
    rng = np.random.default_rng(20261120)
    n = int(rng.choice([5, 20, 50, 512]))
    single = rng.dirichlet(np.ones(5) * rng.uniform(0.2, 2.0))
    probabilities = np.tile(single, (n, 1))
    choices = rng.integers(0, 5, size=n)
    pairs = list(zip(probabilities, choices.tolist(), strict=True))
    first = prompted.fit(pairs)
    second = prompted.fit(pairs, start=np.array(prompted.FIT_START) + 0.1)
    difference = float(np.max(np.abs(first.numbers - second.numbers)))
    assert difference < 1e-5
    assert_stationary(probabilities, choices, first)
    assert_stationary(probabilities, choices, second)
    record_property("worst_coordinate_difference", difference)


def test_anti_informative_fit_is_exactly_at_bound() -> None:
    probabilities = np.full((500, 5), 0.24)
    choices = np.arange(500) % 5
    probabilities[np.arange(500), choices] = 0.04
    result = prompted.fit(list(zip(probabilities, choices.tolist(), strict=True)))
    assert result.beta == 0.0
    assert result.free_shifts == pytest.approx(np.zeros(4), abs=1e-10)
    assert_stationary(probabilities, choices, result)


def test_fit_refuses_uncertified_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompted, "POLISH_STEPS", 0)
    monkeypatch.setattr(prompted, "FIT_OPTIONS", {"gtol": 1.0, "maxiter": 1000})
    with pytest.raises(RuntimeError, match=r"remaining gradient .* exceeds 1e-10"):
        prompted.fit([(np.full(5, 0.2), 0)])


def test_solver_uses_method_and_options_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompted, "FIT_METHOD", "changed method")
    monkeypatch.setattr(prompted, "FIT_OPTIONS", {"gtol": 0.5, "maxiter": 3})
    called = False

    def inspect_settings(fun: Any, x0: measure.Floats, **kwargs: Any) -> SimpleNamespace:
        nonlocal called
        called = True
        assert kwargs["method"] == "changed method"
        assert kwargs["options"] == {"gtol": 0.5, "maxiter": 3}
        return SimpleNamespace(success=True, x=x0)

    monkeypatch.setattr(prompted, "minimize", inspect_settings)
    result = prompted.fit([(np.full(5, 0.2), band) for band in range(5)])
    assert result.numbers == pytest.approx(prompted.FIT_START, abs=1e-12)
    assert called


def test_polish_uses_gradient_limit_and_step_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    assert prompted.GRADIENT_LIMIT == 1e-10
    assert prompted.POLISH_STEPS == 20
    calls = 0

    def incomplete(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=True, x=np.array([1e-6]))

    def half_step(hessian: measure.Floats, gradient: measure.Floats) -> measure.Floats:
        nonlocal calls
        calls += 1
        return gradient / 2

    monkeypatch.setattr(prompted, "minimize", incomplete)
    monkeypatch.setattr(np.linalg, "solve", half_step)

    def objective(x: measure.Floats) -> tuple[float, measure.Floats]:
        return float(x @ x / 2), x

    def hessian(x: measure.Floats) -> measure.Floats:
        return np.eye(len(x))

    solve = prompted._solve  # pyright: ignore[reportPrivateUsage]
    result = solve(objective, hessian, [0.0])
    assert calls == 14
    assert np.max(np.abs(result)) <= 1e-10
    monkeypatch.setattr(prompted, "GRADIENT_LIMIT", 1e-5)
    assert solve(objective, hessian, [0.0])[0] == 1e-6
    assert calls == 14
    monkeypatch.setattr(prompted, "GRADIENT_LIMIT", 1e-10)
    monkeypatch.setattr(prompted, "POLISH_STEPS", 13)
    with pytest.raises(RuntimeError, match="remaining gradient"):
        solve(objective, hessian, [0.0])
    assert calls == 27


def test_beta_lower_bound_is_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    assert prompted.BETA_LOWER_BOUND == 0.0
    monkeypatch.setattr(prompted, "BETA_LOWER_BOUND", 0.5)
    with pytest.raises(ValueError, match="inverse temperature"):
        prompted.Recalibration(0.4)
    rng = np.random.default_rng(5)
    choices = rng.choice(5, size=80, p=[0.4, 0.1, 0.2, 0.1, 0.2])
    probabilities = rng.dirichlet(np.ones(5), size=80)
    probabilities[np.arange(80), choices] *= 0.05
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    pairs = list(zip(probabilities, choices.tolist(), strict=True))
    result = prompted.fit(pairs)
    assert result.beta == 0.5
    # The shifts are the best ones at that bound, not the ones for a bound of zero.
    logs = np.log(probabilities)

    def loss(free: measure.Floats) -> float:
        shifts: measure.Floats = np.r_[free, -free.sum()]
        logits = 0.5 * logs + shifts
        return float(
            np.mean(np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(80), choices])
            + prompted.PENALTY * (0.25 + float(np.sum(shifts**2)))
        )

    best = scipy.optimize.minimize(loss, np.zeros(4), method="BFGS", options={"gtol": 1e-9})
    assert np.max(np.abs(np.array(result.free_shifts) - best.x)) < 1e-5
    monkeypatch.setattr(prompted, "BETA_LOWER_BOUND", 0.0)
    assert np.max(np.abs(np.array(prompted.fit(pairs).free_shifts) - best.x)) > 1e-3
    # A beta that is positive but under the bound is moved to the bound as well.
    weak = rng.dirichlet(np.ones(5), size=2000)
    drawn = [int(rng.choice(5, p=row)) for row in softmax(0.2 * np.log(weak), axis=1)]
    weak_pairs = list(zip(weak, drawn, strict=True))
    assert 0.05 < prompted.fit(weak_pairs).beta < 0.4
    monkeypatch.setattr(prompted, "BETA_LOWER_BOUND", 0.5)
    assert prompted.fit(weak_pairs).beta == 0.5


@pytest.mark.parametrize("stage", [1, 2])
def test_solver_failure_in_either_stage_is_refused(
    stage: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    minimize: Any = prompted.minimize
    calls = 0

    def fail_stage(fun: Any, x0: measure.Floats, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == stage:
            return SimpleNamespace(success=False, message="requested stage failed")
        return minimize(fun, x0, **kwargs)

    monkeypatch.setattr(prompted, "minimize", fail_stage)
    probabilities = np.full((50, 5), 0.24)
    choices = np.arange(50) % 5
    probabilities[np.arange(50), choices] = 0.04
    with pytest.raises(RuntimeError, match="requested stage failed"):
        prompted.fit(list(zip(probabilities, choices.tolist(), strict=True)))
    assert calls == stage


def near_bound_pairs() -> tuple[list[tuple[measure.Floats, int]], prompted.Recalibration]:
    """Decisions whose fitted beta is positive and close to its bound, with their fit."""
    rng = np.random.default_rng(31)
    count = 120
    choices = rng.choice(5, size=count, p=[0.1, 0.3, 0.2, 0.1, 0.3])
    wrong = rng.dirichlet(np.ones(5), size=count)
    wrong[np.arange(count), choices] = 0.0
    wrong /= wrong.sum(axis=1, keepdims=True)
    right = wrong.copy()
    right[np.arange(count), choices] += 1.0
    right /= right.sum(axis=1, keepdims=True)

    def pairs(weight: float) -> list[tuple[measure.Floats, int]]:
        return list(zip((1 - weight) * wrong + weight * right, choices.tolist(), strict=True))

    low, high = 0.0, 1.0
    assert prompted.fit(pairs(low)).beta == 0.0 < prompted.fit(pairs(high)).beta
    for _ in range(60):
        middle = (low + high) / 2
        if 1e-5 < (beta := prompted.fit(pairs(middle)).beta) < 1e-3:
            return pairs(middle), prompted.fit(pairs(middle))
        low, high = (middle, high) if beta <= 1e-5 else (low, middle)
    raise AssertionError("no weight puts beta just inside its bound")


def outside(reference: prompted.Recalibration) -> Any:
    """A first solve that stops a rounding error outside the bound; later solves are real."""
    solve: Any = prompted._solve  # pyright: ignore[reportPrivateUsage]

    def stand_in(objective: Any, hessian: Any, start: measure.Floats) -> measure.Floats:
        if len(start) == 5:
            return np.r_[-1e-12, reference.free_shifts]
        return solve(objective, hessian, start)

    return stand_in


def test_an_optimum_just_inside_the_bound_is_found_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, reference = near_bound_pairs()
    monkeypatch.setattr(prompted, "_solve", outside(reference))
    polish: Any = prompted._polish  # pyright: ignore[reportPrivateUsage]
    bounds: list[float | None] = []

    def recorded(objective: Any, hessian: Any, numbers: Any, lower: float | None = None) -> Any:
        bounds.append(lower)
        return polish(objective, hessian, numbers, lower)

    monkeypatch.setattr(prompted, "_polish", recorded)
    result = prompted.fit(pairs)
    assert result.beta > 0
    assert np.max(np.abs(result.numbers - reference.numbers)) < 1e-8
    # The boundary solve polishes without a bound; the step back inside is held to it.
    assert bounds == [None, prompted.BETA_LOWER_BOUND]


def test_decisions_a_rounding_error_from_the_bound_are_fitted() -> None:
    """Found in review: the first solve stops just outside the bound and the fit raised."""
    rng = np.random.default_rng(2026)
    choices: Any = np.zeros(0, dtype=int)
    right: measure.Floats = np.zeros((0, 5))
    wrong: measure.Floats = np.zeros((0, 5))
    scale: measure.Floats = np.ones(5)
    # The draws before the last are spent only to reach the stream position of the finding.
    for _ in range(265):
        count = int(rng.choice([5, 10, 20, 50, 512]))
        choices = rng.integers(0, 5, size=count)
        right = rng.dirichlet(rng.uniform(0.02, 1.5, size=5), size=count)
        wrong = right.copy()
        right[np.arange(count), choices] += rng.uniform(0.3, 3.0)
        right /= right.sum(axis=1, keepdims=True)
        wrong[np.arange(count), choices] = 0.0
        wrong[wrong.sum(axis=1) == 0] = 0.2
        wrong /= wrong.sum(axis=1, keepdims=True)
    for _ in range(39):
        scale = np.exp(rng.normal(0, 1.5, size=5)) ** 1.053
    right, wrong = right * scale, wrong * scale
    right /= right.sum(axis=1, keepdims=True)
    wrong /= wrong.sum(axis=1, keepdims=True)
    weight = 0.09322158215367386
    pairs = list(zip((1 - weight) * wrong + weight * right, choices.tolist(), strict=True))
    result = prompted.fit(pairs)
    other = prompted.fit(pairs, start=np.array(prompted.FIT_START) + 0.1)
    assert 0 <= result.beta < 1e-9
    assert np.max(np.abs(result.numbers - other.numbers)) < 1e-8


def test_the_bound_is_kept_within_the_gradient_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, reference = near_bound_pairs()
    monkeypatch.setattr(prompted, "_solve", outside(reference))
    monkeypatch.setattr(prompted, "GRADIENT_LIMIT", 1.0)
    assert prompted.fit(pairs).beta == prompted.BETA_LOWER_BOUND


def test_polish_refuses_a_point_that_stays_outside_and_a_gradient_that_is_not_a_number() -> None:
    polish: Any = prompted._polish  # pyright: ignore[reportPrivateUsage]

    def bowl(numbers: measure.Floats) -> tuple[float, measure.Floats]:
        return float((numbers[0] + 1) ** 2), 2 * (numbers + 1)

    def curvature(numbers: measure.Floats) -> measure.Floats:
        return np.array([[2.0]])

    assert polish(bowl, curvature, np.array([3.0]))[0] == pytest.approx(-1.0)
    with pytest.raises(RuntimeError, match=r"outside its bound 0\.0"):
        polish(bowl, curvature, np.array([3.0]), lower=0.0)

    def shallow(numbers: measure.Floats) -> tuple[float, measure.Floats]:
        return float((numbers[0] - 1e-12) ** 2), 2 * (numbers - 1e-12)

    # Within the gradient limit already, but outside: the bound alone makes it step.
    assert polish(shallow, curvature, np.array([-1e-12]))[0] == -1e-12
    assert polish(shallow, curvature, np.array([-1e-12]), lower=0.0)[0] == pytest.approx(1e-12)

    def broken(numbers: measure.Floats) -> tuple[float, measure.Floats]:
        return 0.0, np.array([float("nan")])

    with pytest.raises(RuntimeError, match="remaining gradient nan"):
        polish(broken, curvature, np.array([3.0]))


def test_fit_starts_at_the_committed_start_and_floors_at_the_shared_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ast
    import inspect

    # An equal literal is the same object once compiled, so the source is what is checked.
    definition = ast.parse(inspect.getsource(prompted.fit)).body[0]
    assert isinstance(definition, ast.FunctionDef)
    assert [ast.unparse(default) for default in definition.args.kw_defaults if default] == [
        "FIT_START"
    ]
    raw = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
    before = prompted.Recalibration().apply(raw)
    monkeypatch.setattr(person, "FLOOR", 0.25)
    assert prompted.Recalibration().apply(raw) == pytest.approx([0.2] * 5)
    assert before[0] < 1e-8


def test_recalibration_refuses_a_scalar_and_a_wrong_number_of_shifts() -> None:
    with pytest.raises(ValueError, match="five finite non-negative"):
        prompted.Recalibration().apply(np.array(0.2))
    wrong: Any = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="four finite free shifts"):
        prompted.Recalibration(1.0, wrong)
    longer: Any = (0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="four finite free shifts"):
        prompted.Recalibration(1.0, longer)


def test_format_check_refuses_stub_and_live_together(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*arguments: object) -> clients.StubClient:
        raise AssertionError("no client may be built")

    monkeypatch.setattr(clients, "DecisionClient", refuse)
    monkeypatch.setattr(clients, "StubClient", refuse)
    with pytest.raises(SystemExit):
        prompted.main(["smoke", "--stub", "--live"])
