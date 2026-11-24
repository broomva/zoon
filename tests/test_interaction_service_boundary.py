"""Service status classes and operator authority at the client boundary."""

import json
from typing import Any

import pytest

from tests.test_interaction_step_faults import Factory, interrupt_chat, state
from zoon.interaction import chat, run
from zoon.interaction.steps import PreflightClient
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)

# Stage, answer, exception at the client, saved status/reason, flag accepted, ending.
STATUS_CASES = [
    (stage, answer, error, saved_status, reason, accepted, ends)
    for stage, rows in (
        (
            "metadata check",
            [
                (200, None, "completed", "", False, False),
                (
                    400,
                    chat.ConfigurationInterruption,
                    "interrupted",
                    "ConfigurationInterruption",
                    False,
                    False,
                ),
                (
                    401,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    403,
                    chat.ConfigurationInterruption,
                    "interrupted",
                    "ConfigurationInterruption",
                    False,
                    False,
                ),
                (404, None, "completed", "", False, False),
                (410, None, "completed", "", False, False),
                (
                    429,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    500,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    503,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    302,
                    chat.ConfigurationInterruption,
                    "interrupted",
                    "ConfigurationInterruption",
                    False,
                    False,
                ),
                (
                    None,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
            ],
        ),
        (
            "generating request",
            [
                (200, None, "completed", "", False, False),
                (
                    400,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    401,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    403,
                    chat.ConfigurationInterruption,
                    "interrupted",
                    "ConfigurationInterruption",
                    False,
                    False,
                ),
                (404, chat.Unusable, "completed", "", False, False),
                (410, chat.Unusable, "completed", "", False, False),
                (
                    429,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    500,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    503,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
                (
                    302,
                    chat.ConfigurationInterruption,
                    "interrupted",
                    "ConfigurationInterruption",
                    False,
                    False,
                ),
                (
                    None,
                    chat.TransportInterruption,
                    "interrupted",
                    "TransportInterruption",
                    True,
                    True,
                ),
            ],
        ),
    )
    for answer, error, saved_status, reason, accepted, ends in rows
]


def reply(text: str = "3", *, version: str = chat.MODEL + "-stub") -> dict[str, Any]:
    return {
        "modelVersion": version,
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1},
        "candidates": [{"content": {"parts": [{"text": text}]}}],
    }


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("stage,answer,error,saved_status,reason,accepted,ends", STATUS_CASES)
def test_service_status_boundary(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    stage: str,
    answer: int | None,
    error: type[Exception] | None,
    saved_status: str,
    reason: str,
    accepted: bool,
    ends: bool,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    pauses: list[float] = []
    tokens: list[str] = []
    answers: list[int | None] = []

    def token() -> str:
        tokens.append("dummy")
        return "dummy"

    def failing(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        answers.append(answer)
        if answer is None:
            raise ConnectionError("offline")
        return answer, reply()

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            failing if stage == "generating request" else ok,
            preflight_transport=failing if stage == "metadata check" else ok,
            token=token,
            pause=pauses.append,
        )

    direct = client()

    def contact() -> None:
        direct.preflight()
        direct.ask("hello")

    if error is None:
        contact()
    else:
        with pytest.raises(error) as caught:
            contact()
        if isinstance(caught.value, chat.Unusable):
            assert caught.value.cause == chat.Cause.WITHDRAWN
        if isinstance(caught.value, chat.TransportInterruption):
            assert (caught.value.stage, caught.value.last_status) == (stage, answer)
    retried = error in (chat.TransportInterruption, chat.Unusable)
    assert answers == [answer] * (4 if retried else 1)
    assert pauses == ([1, 2, 4] if retried else [])
    assert len(tokens) == (5 if answer == 401 else 1)

    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    runner.steps.chat_client = client()
    if saved_status == "interrupted":
        assert error is not None
        with pytest.raises(error):
            runner.execute(step)
    else:
        runner.execute(step)
    saved = state(runner, step)
    assert (saved["status"], saved["reason"]) == (saved_status, reason)
    assert saved["payload"]["chat"]["cause"] == ("WITHDRAWN" if error is chat.Unusable else None)
    capsys.readouterr()
    runner.steps.chat_client = client()
    if accepted:
        done = runner.execute(step, end_chat="Service unavailable")
        assert done["status"] == "completed"
        event = done["payload"]["chat_endings"][-1]
        assert ends and runner.steps.ended
        assert event["cause"] == "OPERATOR"
        assert (event["failure_stage"], event["last_status"]) == (stage, answer)
        assert event["chat_transport_failures"] == 2
        assert done["payload"]["chat"]["status"] == "unusable"
        assert event == json.loads(capsys.readouterr().out.split("[chat ending] ")[1])
    else:
        with pytest.raises(clients.GuardRefusal):
            runner.execute(step, end_chat="Service unavailable")
        assert not ends and not runner.steps.ended
        assert state(runner, step) == saved
        if error is chat.ConfigurationInterruption:
            with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="prerequisite"):
                runner.guard.before("held-out")
            assert "outcome" not in saved["payload"]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("kind", ["usable", "unusable", "blocked", "version"])
def test_returned_reply_uses_flag_up(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    kind: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    calls = 0

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls > 1:
            return 503, {}
        result = reply("invalid" if kind == "unusable" else "3")
        if kind == "blocked":
            result["promptFeedback"] = {"blockReason": "SAFETY"}
        if kind == "version":
            result["modelVersion"] = chat.MODEL + "-changed" if step == "recall" else "other"
        return 200, result

    def metadata(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, {}

    runner.steps.chat_client = chat.VertexClient(
        transport,
        preflight_transport=metadata,
        token=lambda: "dummy",
        pause=lambda _: None,
    )

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    capsys.readouterr()
    assert run.main([step, "--stub", "--end-chat", "Unavailable"]) == (
        0 if kind == "version" else 1
    )
    output = capsys.readouterr()
    saved = state(runner, step)
    assert not saved["payload"].get("chat_endings")
    if kind == "version":
        assert runner.steps.session is not None
        assert isinstance(runner.steps.session.client, PreflightClient)
        assert runner.steps.session.client.succeeded
        assert output.out.count("flag had no effect") == 1
        assert "flag used up" not in output.out
        assert saved["payload"]["chat"]["cause"] == "VERSION"
    else:
        assert output.out.count("flag used up: a chat request succeeded in this attempt") == 1
        assert "flag had no effect" not in output.out
        assert output.err.startswith("TransportInterruption:")
        assert (saved["status"], saved["reason"]) == ("interrupted", "TransportInterruption")
        assert saved["payload"]["chat_transport_failures"] == 2


@pytest.mark.parametrize("stage", ["metadata check", "generating request"])
@pytest.mark.parametrize("last", [503, None])
def test_operator_ending_uses_last_answer(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    last: int | None,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    runner = runs()
    interrupt_chat(runner, "closed-loop")
    answers = [None if last == 503 else 503] * 3 + [last]

    def failing(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        answer = answers.pop(0)
        if answer is None:
            raise ConnectionError("offline")
        return answer, {}

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    runner.steps.chat_client = chat.VertexClient(
        failing if stage == "generating request" else ok,
        preflight_transport=failing if stage == "metadata check" else ok,
        token=lambda: "dummy",
        pause=lambda _: None,
    )
    capsys.readouterr()
    done = runner.execute("closed-loop", end_chat="Unavailable")
    assert answers == []
    event = done["payload"]["chat_endings"][-1]
    assert (event["failure_stage"], event["last_status"]) == (stage, last)
    assert json.loads(capsys.readouterr().out.removeprefix("[chat ending] ")) == event


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("fault", ["append", "completion"])
def test_consumed_flag_is_announced_before_storage_failure(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    fault: str,
) -> None:
    from zoon.interaction.guard import StateReplacedError

    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            raise StateReplacedError("completion unavailable")
        return original(status, reason)

    def append(*args: Any, **kwargs: Any) -> None:
        raise OSError("append unavailable")

    if fault == "append":
        monkeypatch.setattr(chat.Transcript, "append", append)
    else:
        monkeypatch.setattr(runner.steps, "save", save)
    capsys.readouterr()
    with pytest.raises(OSError if fault == "append" else StateReplacedError):
        runner.execute(step, end_chat="Unavailable")
    assert (
        capsys.readouterr().out.count("flag used up: a chat request succeeded in this attempt") == 1
    )


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("stage", ["metadata check", "generating request"])
def test_unconsumed_flag_is_not_announced(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    stage: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)

    def refusal(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 403, {}

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    runner.steps.chat_client = chat.VertexClient(
        refusal if stage == "generating request" else ok,
        preflight_transport=refusal if stage == "metadata check" else ok,
        token=lambda: "dummy",
        pause=lambda _: None,
    )
    capsys.readouterr()
    with pytest.raises(chat.ConfigurationInterruption):
        runner.execute(step, end_chat="Unavailable")
    assert "flag used up" not in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["metadata check", "generating request"])
@pytest.mark.parametrize("fault", ["json", "connection after headers"])
def test_operator_ending_retains_http_status_when_body_fails(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    fault: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    runner = runs()
    interrupt_chat(runner, "closed-loop")

    class HTTPReply:
        status = 200

        def __enter__(self) -> "HTTPReply":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self) -> bytes:
            if fault == "connection after headers":
                raise ConnectionError("body unavailable")
            return b"invalid JSON"

    def urlopen(*args: Any, **kwargs: Any) -> HTTPReply:
        return HTTPReply()

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    monkeypatch.setattr(chat.urllib.request, "urlopen", urlopen)
    runner.steps.chat_client = chat.VertexClient(
        chat.post if stage == "generating request" else ok,
        preflight_transport=chat.get if stage == "metadata check" else ok,
        token=lambda: "dummy",
        pause=lambda _: None,
    )
    capsys.readouterr()
    done = runner.execute("closed-loop", end_chat="Unavailable")
    event = done["payload"]["chat_endings"][-1]
    assert (event["failure_stage"], event["last_status"]) == (stage, 200)
    assert json.loads(capsys.readouterr().out.removeprefix("[chat ending] ")) == event
