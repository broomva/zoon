"""Simulation and recognition obey the registered step and transcript contracts."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.interaction_run_support import repository
from zoon.interaction import chat, run
from zoon.interaction.guard import digest
from zoon.pilot import clients


def test_steps_complete_in_order_and_refuse_repetition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions: list[chat.Session] = []
    original_init = chat.Session.__init__

    def initialize(
        self: chat.Session,
        transcript: chat.Transcript,
        client: chat.Client,
        cap: chat.Decimal | None = None,
    ) -> None:
        original_init(self, transcript, client, cap)
        sessions.append(self)

    monkeypatch.setattr(chat.Session, "__init__", initialize)
    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True, simulation_groups=2)
    for step in ("closed-loop", "recall"):
        with pytest.raises(clients.GuardRefusal, match="prerequisite"):
            runner.execute(step)
    calibration = runner.execute("calibrate")
    assert not sessions
    assert set(calibration["transcripts"]) == {"decision"}
    with pytest.raises(clients.GuardRefusal, match="prerequisite closed-loop"):
        runner.execute("recall")
    simulation = runner.execute("closed-loop")
    assert len(sessions) == 1
    assert simulation["status"] == "completed"
    payload = simulation["payload"]
    assert set(payload["simulations"]) == {"zoon", "alone", "prompted"}
    assert len(payload["digests"]) == 3 * 8 * 2
    assert len(payload["chat_logs"]) == 8 * 2
    for name, expected in payload["digests"].items():
        assert digest((runner.guard.storage / name).read_bytes()) == expected
    assert all(r["status"] == "not asked" for r in payload["recall"])
    assert len(payload["recall"]) == len(chat.GAMES)
    assert payload["chat"]["answered_decisions"] == 880
    assert payload["calls"] < 1024 + 880
    assert payload["calls"] > calibration["payload"]["calls"]
    recognition = runner.execute("recall")
    assert len(sessions) == 2
    assert recognition["status"] == "completed"
    assert recognition["payload"]["stopped_by"] is None
    transcript = chat.Transcript(
        runner.guard.transcript_directory,
        chat.RunIdentity(recognition["registered_commit"], "stub"),
    )
    assert recognition["payload"]["recall"] == [asdict(r) for r in chat.read_recall(transcript)]
    assert all(r["status"] == "judged" for r in recognition["payload"]["recall"])
    raw = transcript.read_bytes()
    assert recognition["payload"]["chat"]["transcript"]["lines"] == 889
    assert recognition["payload"]["chat"]["transcript"]["digest"] == digest(raw)
    for step in ("calibrate", "closed-loop", "recall"):
        with pytest.raises(clients.GuardRefusal, match="completed"):
            runner.execute(step)
    state = json.loads((runner.guard.directory / "closed-loop.json").read_bytes())
    assert state == simulation


def test_new_steps_have_no_data_sealing_or_source_access() -> None:
    import ast

    from zoon.interaction import simulation, steps

    for module in (simulation, steps):
        assert module.__file__ is not None
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr
            in (
                "read_held_out",
                "open_held_out",
                "load_held_out",
                "permit",
                "require_permit",
                "seal",
            )
            for node in ast.walk(tree)
        )
        forbidden = {
            "open_held_out",
            "read_held_out",
            "load_held_out",
            "require_permit",
            "HeldOutPermit",
            "RAW_DIR",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.alias):
                assert node.name.rsplit(".", 1)[-1] not in forbidden
            if isinstance(node, ast.Name):
                assert node.id not in forbidden
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden
        assert "HeldOutPermit" not in source and "groups/raw" not in source
        assert "/raw" not in source and "data.load" not in source


def test_channels_streams_committed_materials_and_saved_recalibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    import numpy as np

    from zoon.interaction import data, kernel, person, prompted, registered, score, simulation

    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True, simulation_groups=1)
    calibrated = runner.execute()
    numbers = calibrated["payload"]["recalibration"]
    recalibration = prompted.Recalibration(numbers[0], tuple(numbers[1:]))
    calls: list[tuple[str, bool, int]] = []
    original = score.simulate

    def simulate(
        agent: kernel.Agent,
        amounts: kernel.Amounts,
        *,
        open_channel: bool,
        groups: int = score.SIMULATED_GROUPS,
    ) -> score.Simulation:
        assert amounts == registered.read_amounts(
            root / "interaction/amounts.json", root / "interaction/data-manifest.json"
        )
        arm = (
            "prompted"
            if isinstance(agent, simulation.Prompted)
            else "alone"
            if isinstance(agent, kernel.Coded) and agent.law.candidate.twin
            else "zoon"
        )
        calls.append((arm, open_channel, groups))
        if isinstance(agent, kernel.Coded):
            law = registered.read_fitted(
                root / "interaction/fitted.json", root / "interaction/data-manifest.json"
            )[arm]
            np.testing.assert_array_equal(agent.law.numbers, law.numbers)
        else:
            assert isinstance(agent, simulation.Prompted)
            np.testing.assert_array_equal(agent.recalibration.numbers, recalibration.numbers)
        return original(agent, amounts, open_channel=open_channel, groups=groups)

    monkeypatch.setattr(score, "simulate", simulate)

    # Regenerating the guard's amount table is allowed; refitting either law is not.
    def no_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("the committed fits and saved recalibration must be reused")

    monkeypatch.setattr(registered, "fit", no_fit)
    monkeypatch.setattr(person, "fit", no_fit)
    monkeypatch.setattr(prompted, "fit", no_fit)
    original_build = kernel.Amounts.build
    original_verify = runner.guard.verify_materials
    regenerating = False

    def verify() -> list[data.Group]:
        nonlocal regenerating
        regenerating = True
        try:
            return original_verify()
        finally:
            regenerating = False

    def build(_cls: type[kernel.Amounts], groups: Sequence[data.Group]) -> kernel.Amounts:
        assert regenerating, "simulation must read the committed amount table"
        return original_build(groups)

    monkeypatch.setattr(runner.guard, "verify_materials", verify)
    monkeypatch.setattr(kernel.Amounts, "build", classmethod(build))
    state = runner.execute("closed-loop")
    assert calls == [("zoon", True, 1), ("alone", False, 1), ("prompted", True, 1)]
    transcript = runner.steps.decision
    assert transcript is not None
    for arm, names in state["payload"]["simulations"].items():
        for name in names:
            raw = (runner.guard.storage / name).read_bytes()
            header = json.loads(raw.splitlines()[0])
            cell = header["cell"]
            assert header["group"] == f"registered|{cell}|0"
            if arm == "chat":
                continue
            assert header["seed"] == 20261120
            assert header["open"] == (arm != "alone")
            replayed = kernel.replay(raw)
            assert replayed.bytes == raw
            if arm == "prompted":
                history: tuple[kernel.Observation, ...] = ()
                intents = [
                    json.loads(line)
                    for line in raw.splitlines()
                    if json.loads(line)["kind"] == "intent"
                ]
                for r, row in enumerate(replayed.rounds):
                    game = chat.GAMES[cell]
                    for seat in range(game.size):
                        request = prompted.request(kernel.View(game, seat, history))
                        from zoon.interaction import decisions

                        key = decisions.request_key((digest(clients.canonical(request)), request))
                        reply = transcript.transcript.entries[key].response
                        expected = recalibration.apply(
                            np.array([reply.probabilities[a] for a in prompted.ANSWERS])
                        )
                        np.testing.assert_allclose(
                            intents[r * game.size + seat]["returned"], expected, rtol=0, atol=1e-12
                        )
                    history += (kernel.Observation(tuple(enumerate(row)), game.payoffs(row)),)


def test_chat_receipt_identity_incomplete_line_and_checkpoint_are_guarded(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True, simulation_groups=1)
    runner.execute()
    runner.execute("closed-loop")
    path = runner.guard.transcript_directory / "chat.jsonl"
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    header = json.loads(lines[0])
    path.write_bytes(chat.kernel.encoded({**header, "mode": "live"}).encode() + b"".join(lines[1:]))
    with pytest.raises(clients.GuardRefusal, match="identity"):
        runner.execute("recall")
    path.write_bytes(original[:-1])
    with pytest.raises(clients.GuardRefusal, match="incomplete transcript line"):
        runner.execute("recall")
    assert path.read_bytes() == original[:-1]
    entry = json.loads(lines[1])
    entry["response"]["input_tokens"] += 1
    path.write_bytes(lines[0] + chat.kernel.encoded(entry).encode() + b"".join(lines[2:]))
    with pytest.raises(clients.GuardRefusal, match="checkpoint changed"):
        runner.execute("recall")


def test_registered_group_count_and_stub_only_test_sizes(tmp_path: Path) -> None:
    from zoon.interaction import score

    root = repository(tmp_path / "repo")
    assert run.Run(root, stub=True).steps.groups == score.SIMULATED_GROUPS == 100
    assert run.Run(root, stub=True, simulation_groups=2).steps.groups == 2
    with pytest.raises(ValueError, match="registered ensemble"):
        run.Run(root, simulation_groups=2)
    for count in (0, -1, 101, True):
        with pytest.raises(ValueError, match="group count"):
            run.Run(root, stub=True, simulation_groups=count)


@pytest.mark.parametrize("step", ["calibrate", "held-out", "report"])
def test_step_driver_refuses_other_commands_before_writing(tmp_path: Path, step: str) -> None:
    root = repository(tmp_path / "repo")
    runner = run.Run(root, stub=True)
    with pytest.raises(clients.GuardRefusal, match="unimplemented step"):
        runner.steps.execute(step)
    assert not runner.guard.directory.exists()


def test_log_publication_keeps_previous_bytes_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zoon.interaction import simulation

    path = tmp_path / "game.jsonl"
    path.write_bytes(b"previous complete log\n")

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("storage failed before publication")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="before publication"):
        simulation.write(path, b"new complete log\n")
    assert path.read_bytes() == b"previous complete log\n"
    assert path.with_suffix(".tmp").read_bytes() == b"new complete log\n"
