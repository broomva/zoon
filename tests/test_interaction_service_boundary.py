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

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            transport, preflight_transport=metadata, token=lambda: "dummy", pause=lambda _: None
        )

    direct = PreflightClient(client())
    response = direct.ask("hello")
    assert direct.succeeded
    assert response.text == ("" if kind == "blocked" else "invalid" if kind == "unusable" else "3")
    calls = 0
    runner.steps.chat_client = client()

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
@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("last", [503, None])
def test_operator_ending_uses_last_answer(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    step: str,
    last: int | None,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
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
    done = runner.execute(step, end_chat="Unavailable")
    assert answers == []
    event = done["payload"]["chat_endings"][-1]
    assert (event["failure_stage"], event["last_status"]) == (stage, last)
    assert json.loads(capsys.readouterr().out.removeprefix("[chat ending] ")) == event


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("fault", ["append", "completion"])
@pytest.mark.parametrize("flagged", [False, True])
def test_consumed_flag_is_announced_before_storage_failure(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    fault: str,
    flagged: bool,
) -> None:
    from zoon.interaction.guard import StateReplacedError

    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            transport, preflight_transport=transport, token=lambda: "dummy", pause=lambda _: None
        )

    direct = PreflightClient(client())
    assert direct.ask("hello").text == "3" and direct.succeeded
    runner.steps.chat_client = client()
    before = state(runner, step)["payload"]["chat"]
    original = runner.steps.save

    def save(status: str = "running", reason: str = "") -> dict[str, Any]:
        if status == "completed":
            raise StateReplacedError("completion unavailable")
        return original(status, reason)

    def append(*args: Any, **kwargs: Any) -> None:
        assert runner.steps.session is not None
        assert isinstance(runner.steps.session.client, PreflightClient)
        assert runner.steps.session.client.succeeded
        raise OSError("append unavailable")

    if fault == "append":
        monkeypatch.setattr(chat.Transcript, "append", append)
    else:
        monkeypatch.setattr(runner.steps, "save", save)
    capsys.readouterr()
    with pytest.raises(OSError if fault == "append" else StateReplacedError):
        runner.execute(step, end_chat="Unavailable" if flagged else None)
    assert capsys.readouterr().out.count(
        "flag used up: a chat request succeeded in this attempt"
    ) == int(flagged)
    if fault == "append":
        assert state(runner, step)["payload"]["chat"] == before


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


def status_rule(stage: str, status: int) -> str:
    """Classify the registration's stated sets, independently of client predicates."""
    if stage == "metadata check":
        if status in {200, 404, 410}:
            return "reply"
        if status in {401, 429} or status // 100 == 5:
            return "transport"
        return "configuration"
    if status == 200:
        return "reply"
    if status in {404, 410}:
        return "withdrawn"
    if status == 403 or status // 100 not in {4, 5}:
        return "configuration"
    return "transport"


@pytest.mark.parametrize("stage", ["metadata check", "generating request"])
@pytest.mark.parametrize("status", range(100, 600))
def test_every_service_status_matches_registration(
    monkeypatch: pytest.MonkeyPatch, stage: str, status: int
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[int] = []
    tokens: list[str] = []
    pauses: list[float] = []

    def token() -> str:
        tokens.append(f"dummy-{len(tokens)}")
        return tokens[-1]

    def answer(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(status)
        assert headers["Authorization"] == "Bearer " + tokens[-1]
        return status, reply()

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    client = chat.VertexClient(
        answer if stage == "generating request" else ok,
        preflight_transport=answer if stage == "metadata check" else ok,
        token=token,
        pause=pauses.append,
    )
    expected = status_rule(stage, status)
    errors = {
        "transport": chat.TransportInterruption,
        "configuration": chat.ConfigurationInterruption,
        "withdrawn": chat.Unusable,
    }
    if expected == "reply":
        client.preflight()
        assert client.ask("hello") == chat.Response("3", chat.MODEL + "-stub", 12, 1)
    else:
        with pytest.raises(errors[expected]) as caught:
            client.preflight()
            client.ask("hello")
        if isinstance(caught.value, chat.TransportInterruption):
            assert (caught.value.stage, caught.value.last_status) == (stage, status)
        if isinstance(caught.value, chat.Unusable):
            assert caught.value.cause == chat.Cause.WITHDRAWN
    retried = expected in {"transport", "withdrawn"}
    assert calls == [status] * (4 if retried else 1)
    assert pauses == ([1, 2, 4] if retried else [])
    assert len(tokens) == (5 if status == 401 else 1)


def invalid_answer(kind: str) -> dict[str, Any]:
    result = reply()
    if kind == "missing version":
        del result["modelVersion"]
    elif kind == "blank version":
        result["modelVersion"] = " \t\n"
    elif kind == "non-string version":
        result["modelVersion"] = 1
    elif kind == "missing tokens":
        result["usageMetadata"] = {}
    elif kind == "negative tokens":
        result["usageMetadata"]["promptTokenCount"] = -1
    elif kind == "boolean tokens":
        result["usageMetadata"]["candidatesTokenCount"] = True
    else:
        assert kind == "unprocessable body"
        result["candidates"] = [None]
    return result


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("recovers", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "missing version",
        "blank version",
        "non-string version",
        "missing tokens",
        "negative tokens",
        "boolean tokens",
        "unprocessable body",
    ],
)
def test_response_checks_control_flag_consumption(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    recovers: bool,
    kind: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[bytes] = []
    pauses: list[float] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(body)
        if recovers and len(calls) > 1:
            return (200, reply()) if len(calls) == 2 else (503, {})
        return 200, invalid_answer(kind)

    def metadata(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, {}

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            transport, preflight_transport=metadata, token=lambda: "dummy", pause=pauses.append
        )

    direct = PreflightClient(client())
    if recovers:
        assert direct.ask("hello").text == "3"
        assert direct.succeeded
        assert len(calls) == 2 and pauses == [1]
    else:
        with pytest.raises(chat.TransportInterruption) as caught:
            direct.ask("hello")
        assert (caught.value.stage, caught.value.last_status) == ("generating request", 200)
        assert not direct.succeeded
        assert len(calls) == 4 and pauses == [1, 2, 4]
    calls.clear()
    pauses.clear()
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    before = state(runner, step)["payload"]["chat"]["calls"]
    runner.steps.chat_client = client()
    capsys.readouterr()
    if recovers:
        with pytest.raises(chat.TransportInterruption):
            runner.execute(step, end_chat="Unavailable")
        saved = state(runner, step)
        assert (saved["status"], saved["reason"]) == ("interrupted", "TransportInterruption")
        assert not saved["payload"].get("chat_endings")
        assert saved["payload"]["chat"]["calls"] == before + 1
        assert len(calls) == 6 and pauses == [1, 1, 2, 4]
        assert capsys.readouterr().out.count("flag used up") == 1
    else:
        done = runner.execute(step, end_chat="Unavailable")
        event = done["payload"]["chat_endings"][-1]
        assert done["status"] == "completed" and runner.steps.ended
        assert (event["cause"], event["failure_stage"], event["last_status"]) == (
            "OPERATOR",
            "generating request",
            200,
        )
        assert done["payload"]["chat"]["calls"] == before
        assert len(calls) == 4 and pauses == [1, 2, 4]
        assert "flag used up" not in capsys.readouterr().out
        assert runner.steps.session is not None
        assert isinstance(runner.steps.session.client, PreflightClient)
        assert not runner.steps.session.client.succeeded
    assert calls[0] == calls[1]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("secret", ["fixture-project", "dummy-bearer"])
@pytest.mark.parametrize("shape", ["valid", "missing version", "invalid tokens"])
def test_secret_answer_interrupts_without_record_or_flag_consumption(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    secret: str,
    shape: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture-project")
    calls: list[bytes] = []
    pauses: list[float] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(body)
        result = reply(secret)
        if shape == "missing version":
            del result["modelVersion"]
        elif shape == "invalid tokens":
            result["usageMetadata"]["promptTokenCount"] = -1
        return 200, result

    def metadata(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, {}

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            transport,
            preflight_transport=metadata,
            token=lambda: "dummy-bearer",
            pause=pauses.append,
        )

    direct = PreflightClient(client())
    with pytest.raises(chat.GuardRefusal, match="no record was written"):
        direct.ask("hello")
    assert not direct.succeeded and len(calls) == 1 and not pauses
    calls.clear()
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    before = state(runner, step)["payload"]["chat"]
    runner.steps.chat_client = client()
    capsys.readouterr()
    with pytest.raises(chat.GuardRefusal):
        runner.execute(step, end_chat="Unavailable")
    saved = state(runner, step)
    assert (saved["status"], saved["reason"]) == ("interrupted", "GuardRefusal")
    assert saved["payload"]["chat"] == before
    assert not saved["payload"].get("chat_endings")
    assert "flag used up" not in capsys.readouterr().out
    assert len(calls) == 1
    with pytest.raises(clients.GuardRefusal, match="only after"):
        runner.execute(step, end_chat="Unavailable")
    assert len(calls) == 1 and state(runner, step) == saved
    runner.steps.chat_client = client()
    with pytest.raises(chat.GuardRefusal):
        runner.execute(step)
    assert len(calls) == 2 and calls[0] == calls[1] and not pauses
    assert state(runner, step)["payload"]["chat"] == before


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("stage", ["metadata check", "generating request"])
@pytest.mark.parametrize("fault", ["obtain", "refresh", "unreachable", "empty"])
def test_credentials_refuse_without_retry_or_operator_ending(
    runs: Factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
    stage: str,
    fault: str,
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[str] = []
    tokens: list[str] = []
    pauses: list[float] = []

    def token() -> str:
        tokens.append("fetch")
        if fault != "obtain" and len(tokens) == 1:
            return "dummy"
        if fault == "empty":
            return ""
        if fault == "unreachable":
            raise ConnectionError("token source unavailable")
        raise ValueError("credentials unavailable")

    def refused(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        calls.append(stage)
        return 401, {}

    def ok(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        return 200, reply()

    def client() -> chat.VertexClient:
        return chat.VertexClient(
            refused if stage == "generating request" else ok,
            preflight_transport=refused if stage == "metadata check" else ok,
            token=token,
            pause=pauses.append,
        )

    direct = PreflightClient(client())
    with pytest.raises(chat.ConfigurationInterruption, match="credentials unavailable"):
        direct.ask("hello")
    assert not direct.succeeded and not pauses
    assert len(tokens) == (1 if fault == "obtain" else 2)
    assert calls == ([] if fault == "obtain" else [stage])
    tokens.clear()
    calls.clear()
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    interrupt_chat(runner, step)
    before = state(runner, step)["payload"]["chat"]
    runner.steps.chat_client = client()
    capsys.readouterr()
    with pytest.raises(chat.ConfigurationInterruption):
        runner.execute(step, end_chat="Unavailable")
    saved = state(runner, step)
    assert (saved["status"], saved["reason"]) == ("interrupted", "ConfigurationInterruption")
    assert saved["payload"]["chat"] == before and "outcome" not in saved["payload"]
    assert not saved["payload"].get("chat_endings") and not pauses
    assert len(tokens) == (1 if fault == "obtain" else 2)
    assert calls == ([] if fault == "obtain" else [stage])
    assert "flag used up" not in capsys.readouterr().out
    with pytest.raises(clients.GuardRefusal, match="only after"):
        runner.execute(step, end_chat="Unavailable")
    assert len(tokens) == (1 if fault == "obtain" else 2)
    assert state(runner, step) == saved
    with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="prerequisite"):
        runner.guard.before("held-out")
