"""Operator authority through the real HTTP client, helper and step saves."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest

from tests.test_interaction_step_faults import Factory, state
from zoon.interaction import chat, guard, run, service_answers, simulation
from zoon.interaction.service_answers import Answer
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)


@contextmanager
def connected(
    runner: run.Run, stage: str = "generating request", answers: tuple[Answer, ...] = (Answer(503),)
) -> Generator[list[str]]:
    with service_answers.network(stage, answers) as requests:
        runner.steps.chat_client = service_answers.client()
        yield requests


def interrupted(runner: run.Run, step: str) -> None:
    with connected(runner), pytest.raises(chat.TransportInterruption):
        runner.execute(step)


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_first_flag_refuses_before_requests(runs: Factory, step: str) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "calibrated")
    path = runner.guard.directory / f"{step}.json"
    before = path.read_bytes() if path.exists() else None
    with connected(runner) as requests, pytest.raises(clients.GuardRefusal, match="only after"):
        runner.execute(step, end_chat="Unavailable")
    assert not requests
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize(
    "stage,answer",
    [
        (stage, answer)
        for stage in service_answers.STAGES
        for answer in (Answer(503), Answer(403), Answer(200, "secret in text"))
        if stage != "metadata check" or answer.kind != "secret in text"
    ],
)
def test_first_request_can_end_after_chat_interruption(
    runs: Factory,
    capsys: pytest.CaptureFixture[str],
    step: str,
    stage: str,
    answer: Answer,
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    with connected(runner, stage, (answer,)), pytest.raises(chat.CHAT_REQUEST_FAILURES):
        runner.execute(step)
    before = state(runner, step)["payload"]["chat"]
    capsys.readouterr()
    with connected(runner, stage, (answer,)) as requests:
        done = runner.execute(step, end_chat="Unavailable")
    event = done["payload"]["chat_endings"][-1]
    assert event == {
        "cause": "OPERATOR",
        "step": step,
        "reason": "Unavailable",
        "chat_request_failures": 2,
        "failure_class": (
            "configuration"
            if answer.status == 403
            else "secret refusal"
            if answer.kind == "secret in text"
            else "transport"
        ),
        "failure_stage": stage,
        "last_status": answer.status,
        "answered_decisions": before["answered_decisions"],
        **({"answered_recall": 0} if step == "recall" else {}),
    }
    assert done["status"] == "completed" and done["payload"]["chat"]["cause"] == "OPERATOR"
    assert done["payload"]["chat"]["status"] == "unusable"
    assert "outcome" not in done["payload"]
    assert len(requests) == (
        (4 if stage == "metadata check" else 5)
        if answer.status == 503
        else (1 if stage == "metadata check" else 2)
    )
    assert json.loads(capsys.readouterr().out.removeprefix("[chat ending] ")) == event
    if step == "closed-loop":
        assert "chat_logs" not in done["payload"]
    else:
        assert done["payload"]["stopped_by"] == "OPERATOR"
        assert all(row["status"] == "not asked" for row in done["payload"]["recall"])


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("flag", [False, True])
def test_success_and_replayed_replies_control_attempt_authority(
    runs: Factory, capsys: pytest.CaptureFixture[str], step: str, flag: bool
) -> None:
    runner = runs(prepared="interrupted-" + step)
    capsys.readouterr()
    with (
        connected(runner, answers=(Answer(200), Answer(503))) as requests,
        pytest.raises(chat.TransportInterruption),
    ):
        runner.execute(step, end_chat="Unavailable" if flag else None)
    assert len(requests) == 6
    assert capsys.readouterr().out == (
        "[chat ending] flag used up: a chat request succeeded in this attempt\n" if flag else ""
    )
    saved = state(runner, step)
    assert saved["payload"]["chat_request_failures"] == 2
    assert not saved["payload"].get("chat_endings")
    before = saved["payload"]["chat"]
    with connected(runner) as requests:
        done = runner.execute(step, end_chat="Unavailable")
    assert len(requests) == 5
    event = done["payload"]["chat_endings"][-1]
    assert event["chat_request_failures"] == 3
    assert event["answered_decisions"] == before["answered_decisions"]
    if step == "recall":
        assert event["answered_recall"] == 1
        assert [r["status"] for r in done["payload"]["recall"]] == ["judged"] + ["not asked"] * 7
    else:
        assert event["answered_decisions"] == 1


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("handler_fails", [False, True])
def test_http_ending_survives_only_successful_saves(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    handler_fails: bool,
) -> None:
    runner = runs(prepared="interrupted-" + step)
    original = runner.steps.save
    failed = False

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        nonlocal failed
        if runner.steps.payload.get("chat_endings") and (
            not failed or (status == "interrupted" and handler_fails)
        ):
            failed = True
            raise OSError("state unavailable")
        return original(status, reason)

    def local(**kwargs: Any) -> run.Run:
        return runner

    with monkeypatch.context() as patch, connected(runner):
        patch.setattr(runner.steps, "save", save)
        patch.setattr(run, "Run", local)
        assert run.main([step, "--stub", "--end-chat", "Unavailable"]) == 1
    output = capsys.readouterr()
    assert output.out == "" and output.err == "OSError: state unavailable\n"
    saved = state(runner, step)
    assert saved["status"] == ("running" if handler_fails else "interrupted")
    assert bool(saved["payload"].get("chat_endings")) == (not handler_fails)
    with connected(runner) as requests:
        if handler_fails:
            with pytest.raises(clients.GuardRefusal, match="only after"):
                runner.execute(step, end_chat="Unavailable")
            assert not requests
        else:
            done = runner.execute(step, end_chat="Unavailable")
            assert (
                not requests and done["payload"]["chat_endings"] == saved["payload"]["chat_endings"]
            )
            plain = runs("plain", prepared="interrupted-" + step)
            with connected(plain):
                expected = plain.execute(step, end_chat="Unavailable")
            assert done == expected
            capsys.readouterr()

    assert not capsys.readouterr().out
    if handler_fails:
        interrupted(runner, step)
        with connected(runner):
            done = runner.execute(step, end_chat="Unavailable")
        event = done["payload"]["chat_endings"][-1]
        assert event["chat_request_failures"] == 3
        assert json.loads(capsys.readouterr().out.removeprefix("[chat ending] ")) == event


def test_operator_endings_in_both_steps_preserve_the_first(
    runs: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = runs()
    interrupted(runner, "closed-loop")
    with connected(runner):
        closed = runner.execute("closed-loop", end_chat="Decisions unavailable")
    first = closed["payload"]["chat_endings"][0]
    with connected(runner) as requests, pytest.raises(clients.GuardRefusal, match="only after"):
        runner.execute("recall", end_chat="Recall unavailable")
    assert not requests
    with connected(runner, answers=(Answer(403),)), pytest.raises(chat.ConfigurationInterruption):
        runner.execute("recall")
    with connected(runner):
        recalled = runner.execute("recall", end_chat="Recall unavailable")
    assert recalled["payload"]["chat_endings"][0] == first
    assert [e["step"] for e in recalled["payload"]["chat_endings"]] == ["closed-loop", "recall"]
    assert state(runner) == closed
    events = recalled["payload"]["chat_endings"]
    assert [event["chat_request_failures"] for event in events] == [2, 2]
    printed = [json.loads(block) for block in capsys.readouterr().out.split("[chat ending] ")[1:]]
    assert printed == events


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("last", [503, None])
def test_http_last_attempt_status_replaces_prior_answers(
    runs: Factory, step: str, last: int | None
) -> None:
    runner = runs(prepared="interrupted-" + step)
    with connected(runner, answers=(Answer(200, "empty body"),) * 3 + (Answer(last),)):
        done = runner.execute(step, end_chat="Unavailable")
    assert done["payload"]["chat_endings"][-1]["last_status"] == last


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("status", ["running", "interrupted", "completed", "terminal"])
@pytest.mark.parametrize("cause", ["OPERATOR", "WITHDRAWN"])
def test_saved_ending_never_causes_another_service_request(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    status: str,
    cause: str,
) -> None:
    runner = runs(prepared="interrupted-" + step)
    original = runner.steps.save

    def stop(status: str = "running", reason: str = "") -> dict[str, Any]:
        if runner.steps.payload.get("chat_endings") and status in ("completed", "interrupted"):
            raise SystemExit("stopped")
        return original(status, reason)

    with (
        monkeypatch.context() as patch,
        connected(runner, answers=(Answer(404 if cause == "WITHDRAWN" else 503),)),
    ):
        patch.setattr(runner.steps, "save", stop)
        with pytest.raises(SystemExit):
            runner.execute(step, end_chat="Unavailable")
    saved = state(runner, step)
    saved["status"] = status
    if status == "terminal":
        saved["payload"]["outcome"] = {"H2": "no verdict", "H3": "no verdict", "reason": "CAP"}
        saved["reason"] = "decision model spending cap reached"
    saved["payload_digest"] = guard.digest(guard.encoded(saved["payload"]))
    saved.pop("digest")
    guard.atomic(
        runner.guard.directory / f"{step}.json",
        {**saved, "digest": guard.digest(guard.encoded(saved))},
    )
    with connected(runner) as requests:
        if status in ("completed", "terminal"):
            with pytest.raises(clients.GuardRefusal, match=r"completed|terminal"):
                runner.execute(step, end_chat="Unavailable")
        else:
            done = runner.execute(step, end_chat="Unavailable")
            assert done["payload"]["chat_endings"] == saved["payload"]["chat_endings"]
            assert not runner.steps.ended
            capsys.readouterr()

            def local(**kwargs: Any) -> run.Run:
                return runner

            monkeypatch.setattr(run, "Run", local)

            def retained(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return done

            monkeypatch.setattr(runner, "execute", retained)
            assert run.main([step, "--stub", "--end-chat", "Unavailable"]) == 0
            output = capsys.readouterr().out
            assert "ending already recorded: " in output and "flag had no effect" not in output
            assert guard.encoded(done["payload"]["chat_endings"][-1]).decode().strip() in output
        assert not requests


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("stage", service_answers.STAGES)
def test_redirect_target_failure_has_no_final_status(runs: Factory, step: str, stage: str) -> None:
    runner = runs(prepared="interrupted-" + step)
    with connected(runner, stage, (Answer(302, "redirect to unreachable"),)):
        done = runner.execute(step, end_chat="Unavailable")
    event = done["payload"]["chat_endings"][-1]
    assert event["failure_stage"] == stage and event["last_status"] is None


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("fault", ["first flag", "ending", "consumed"])
def test_http_command_prints_the_attempt_result(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    fault: str,
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    if fault != "first flag":
        interrupted(runner, step)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    answers = (
        (Answer(200), Answer(503))
        if fault == "consumed"
        else (Answer(403 if fault == "configuration" else 503),)
    )
    capsys.readouterr()
    with connected(runner, answers=answers) as requests:
        result = run.main([step, "--stub", "--end-chat", "Unavailable"])
    output = capsys.readouterr()
    assert result == int(fault != "ending")
    if fault == "first flag":
        assert not requests and not output.out
        assert "GuardRefusal: flag is accepted only after" in output.err
    elif fault == "ending":
        assert not output.err and "flag had no effect" not in output.out
        assert '"cause": "OPERATOR"' in output.out
        assert output.out.count("[chat ending] ") == 1
    elif fault == "consumed":
        assert (
            output.out == "[chat ending] flag used up: a chat request succeeded in this attempt\n"
        )
        assert output.err == "TransportInterruption: chat transport failure after bounded retries\n"
    else:
        assert not output.out and output.err == "ConfigurationInterruption: chat request refused\n"


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("reason", ["", " \t\n"])
def test_blank_cli_reason_precedes_runner_construction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], step: str, reason: str
) -> None:
    def unexpected(**kwargs: Any) -> run.Run:
        pytest.fail("blank reason constructed a runner")

    monkeypatch.setattr(run, "Run", unexpected)
    assert run.main([step, "--stub", "--end-chat", reason]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err == "GuardRefusal: ending chat requires a nonempty reason\n"


def test_flagged_recall_never_dispatches_the_decision_model(runs: Factory) -> None:
    class Decision(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            pytest.fail("recall dispatched a decision-model request")

    runner = runs(prepared="closed-loop")
    runner.steps.client = Decision("decision")
    interrupted(runner, "recall")
    with connected(runner):
        done = runner.execute("recall", end_chat="Unavailable")
    assert done["payload"]["chat_endings"][-1]["cause"] == "OPERATOR"


def test_decision_failure_with_flag_prints_no_chat_ending(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = runs()
    interrupted(runner, "closed-loop")

    def bands(*args: Any, **kwargs: Any) -> None:
        raise clients.StepFailure("decision transport unavailable")

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(simulation.Store, "bands", bands)
    monkeypatch.setattr(run, "Run", local)
    capsys.readouterr()
    with connected(runner) as requests:
        assert run.main(["closed-loop", "--stub", "--end-chat", "Unavailable"]) == 1
    output = capsys.readouterr()
    assert not requests and not output.out
    assert output.err == "StepFailure: decision transport unavailable\n"
    saved = state(runner)
    assert saved["reason"] == "StepFailure" and not saved["payload"].get("chat_endings")


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_missing_credential_failure_allows_ending(runs: Factory, step: str) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    with connected(runner) as requests:
        client = runner.steps.chat_client
        assert isinstance(client, chat.VertexClient)
        client.token = lambda: ""
        with pytest.raises(chat.ConfigurationInterruption):
            runner.execute(step)
        assert not requests
        done = runner.execute(step, end_chat="Unavailable")
        assert not requests
    event = done["payload"]["chat_endings"][-1]
    assert event["failure_class"] == "configuration" and event["chat_request_failures"] == 2


@pytest.mark.parametrize(
    "step,flag",
    [
        ("closed-loop", "end_chat"),
        ("recall", "end_chat"),
        ("closed-loop", "repair"),
        ("recall", "repair"),
        ("calibrate", "repair"),
    ],
)
def test_project_reason_refuses_before_work(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, flag: str
) -> None:
    runner = runs()
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture-project")

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("project reason reached work")

    monkeypatch.setattr(runner.guard, "_checkout", unexpected)
    with (
        connected(runner) as requests,
        pytest.raises(clients.GuardRefusal, match="connection metadata"),
    ):
        runner.execute(step, **{flag: "contains fixture-project"})
    assert not requests


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_bearer_text_in_reason_is_recorded(runs: Factory, step: str) -> None:
    runner = runs(prepared="interrupted-" + step)
    with connected(runner):
        done = runner.execute(step, end_chat="contains dummy-bearer")
    assert done["payload"]["chat_endings"][-1]["reason"] == "contains dummy-bearer"


def test_ending_counts_open_session_without_reopening(
    runs: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runs(prepared="closed-loop")
    interrupted(runner, "recall")
    original = runner.steps.operator_ending

    def ending(reason: str | None, error: Any) -> None:
        record = runner.steps.chat_record()

        def chat_record() -> dict[str, Any]:
            return record

        with monkeypatch.context() as patch:
            patch.setattr(runner.steps, "chat_record", chat_record)

            def unexpected(**kwargs: Any) -> chat.Transcript:
                pytest.fail("ending reopened the transcript")

            patch.setattr(runner.steps, "transcript", unexpected)

            def skip_save(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return {}

            patch.setattr(runner.steps, "save", skip_save)
            original(reason, error)

    monkeypatch.setattr(runner.steps, "operator_ending", ending)
    with (
        connected(runner, answers=(Answer(200), Answer(503))),
        pytest.raises(chat.TransportInterruption),
    ):
        runner.execute("recall", end_chat="Unavailable")
    with connected(runner):
        done = runner.execute("recall", end_chat="Unavailable")
    assert done["payload"]["chat_endings"][-1]["answered_recall"] == 1
