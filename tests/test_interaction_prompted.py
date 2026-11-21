"""Prompted requests on made-up groups and the stub client."""

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from tests.test_interaction_kernel import training
from zoon.interaction import kernel, measure, prompted
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
    assert prompted.Recalibration().apply(p) == pytest.approx(p, abs=1e-15)
    value = prompted.Recalibration(np.log(2), (0.2, -0.3, 0.4, -0.1))
    assert value.shifts.sum() == pytest.approx(0, abs=1e-15)
    assert value.temperature == pytest.approx(2)
    assert value.numbers.shape == (5,)
    assert value.apply(p) == pytest.approx(
        softmax(
            np.log(np.maximum(p, 1e-300)) / 2 + value.shifts,
            axis=1,
        ),
        abs=1e-15,
    )


def synthetic_pairs() -> tuple[list[tuple[measure.Floats, int]], prompted.Recalibration]:
    rng = np.random.default_rng(27)
    weights = np.array([1, 2, 3, 4, 5])
    shifts = np.log(weights) - np.log(weights).mean()
    known = prompted.Recalibration(
        np.log(0.5), (float(shifts[0]), float(shifts[1]), float(shifts[2]), float(shifts[3]))
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
    assert reference.numbers == pytest.approx(known.numbers, abs=2e-6)
    assert reference.shifts.sum() == pytest.approx(0, abs=1e-15)
    for other in (prompted.fit(pairs, prompted.START_NUDGE), prompted.fit(list(reversed(pairs)))):
        assert other.numbers.shape == (5,)
        assert (
            np.max(np.abs(other.numbers - reference.numbers)) < prompted.REPRODUCIBILITY_TOLERANCE
        )
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
        assert (
            prompts.decision_bytes(decision)
            == (prompted.DIRECTORY / "requests" / name).read_bytes()
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
    prompted.write_fixtures()
    for directory in ("instructions", "requests"):
        expected = {p.name: p.read_bytes() for p in (original / directory).iterdir()}
        written = {p.name: p.read_bytes() for p in (tmp_path / directory).iterdir()}
        assert written == expected
