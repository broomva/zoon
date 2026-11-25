"""HTTP failures cross the real client boundary without recording a reply."""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import Any

import pytest

from tests.test_interaction_chat import payload
from tests.test_interaction_step_faults import Factory, state
from zoon.interaction import chat
from zoon.interaction.steps import PreflightClient
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)
Wire = Callable[[str, int], tuple[int, bytes]]


def answer(text: str = "3") -> bytes:
    return json.dumps(
        payload(
            modelVersion=chat.MODEL + "-stub",
            candidates=[{"content": {"parts": [{"text": text}]}}],
        )
    ).encode()


def connection(
    monkeypatch: pytest.MonkeyPatch, wire: Wire
) -> tuple[chat.VertexClient, list[str], list[float]]:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture-project")
    methods: list[str] = []
    pauses: list[float] = []

    class Reply:
        def __init__(self, status: int, body: bytes) -> None:
            self.status, self.body = status, body

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> "Reply":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def urlopen(request: urllib.request.Request, *, timeout: int) -> Reply:
        assert timeout == 30
        method = request.get_method()
        methods.append(method)
        status, body = wire(method, methods.count(method))
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "refused", Message(), None)
        return Reply(status, body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = chat.VertexClient(
        chat.post, token=lambda: "dummy-bearer", preflight_transport=chat.get, pause=pauses.append
    )
    return client, methods, pauses


@pytest.mark.parametrize("metadata", [404, 410])
@pytest.mark.parametrize("generation", ["reply", "withdrawn", "mixed"])
def test_http_missing_metadata_defers_to_generating_attempts(
    monkeypatch: pytest.MonkeyPatch, metadata: int, generation: str
) -> None:
    def wire(method: str, count: int) -> tuple[int, bytes]:
        if method == "GET":
            return metadata, b""
        if generation == "mixed" and count == 4:
            raise urllib.error.URLError("connection unavailable")
        return (200, answer()) if generation == "reply" else (404, b"")

    client, methods, pauses = connection(monkeypatch, wire)
    wrapped = PreflightClient(client)
    if generation == "reply":
        assert wrapped.ask("hello").text == "3"
        assert methods == ["GET", "POST"] and not pauses
    else:
        with pytest.raises(
            chat.Unusable if generation == "withdrawn" else chat.TransportInterruption
        ) as caught:
            wrapped.ask("hello")
        if generation == "withdrawn":
            assert isinstance(caught.value, chat.Unusable)
            assert caught.value.cause == chat.Cause.WITHDRAWN
        assert methods == ["GET"] + ["POST"] * 4
        assert pauses == [1, 2, 4]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize(
    "fault",
    ["transport", "service", "secret", "missing version", "blank version", "unprocessable body"],
)
def test_persistent_http_failure_interrupts_and_blocks_following_steps(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, fault: str
) -> None:
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    before = state(runner)["payload"]["chat"]
    recovering = False

    def wire(method: str, count: int) -> tuple[int, bytes]:
        if method == "GET":
            return 200, b"{}"
        if recovering:
            return 200, answer()
        if fault == "transport":
            raise urllib.error.URLError("connection unavailable")
        if fault == "service":
            return 403, b""
        if fault == "secret":
            return 200, answer("dummy-bearer")
        result = json.loads(answer())
        if fault == "missing version":
            del result["modelVersion"]
        elif fault == "blank version":
            result["modelVersion"] = " \t\n"
        else:
            result["candidates"] = [None]
        return 200, json.dumps(result).encode()

    client, methods, pauses = connection(monkeypatch, wire)
    runner.steps.chat_client = client
    error = (
        chat.ConfigurationInterruption
        if fault == "service"
        else chat.GuardRefusal
        if fault == "secret"
        else chat.TransportInterruption
    )
    attempts = 1 if fault in ("service", "secret") else 4
    for _ in range(2):
        with pytest.raises(error):
            runner.execute(step)
        saved = state(runner, step)
        assert (saved["status"], saved["reason"]) == ("interrupted", error.__name__)
        assert saved["payload"]["chat"] == before
        assert not saved["payload"].get("chat_endings")
        assert "outcome" not in saved["payload"]
        with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="prerequisite"):
            runner.guard.before("held-out")
    assert methods == (["GET"] + ["POST"] * attempts) * 2
    assert pauses == ([] if attempts == 1 else [1, 2, 4] * 2)
    recovering = True
    completed = runner.execute(step)
    assert completed["status"] == "completed"
    assert completed["payload"]["chat"]["cause"] is None
    assert completed["payload"]["chat"]["calls"] > before["calls"]
    if step == "recall":
        assert completed["payload"]["chat"]["answered_decisions"] == before["answered_decisions"]


@pytest.mark.parametrize("stage", ["GET", "POST"])
@pytest.mark.parametrize("status,body", [(200, b"not JSON"), (200, b"[]"), (204, b"")])
def test_http_unreadable_success_is_a_transport_fault(
    monkeypatch: pytest.MonkeyPatch, stage: str, status: int, body: bytes
) -> None:
    def wire(method: str, count: int) -> tuple[int, bytes]:
        return (status, body) if method == stage else (200, b"{}")

    client, methods, pauses = connection(monkeypatch, wire)
    with pytest.raises(chat.TransportInterruption):
        PreflightClient(client).ask("hello")
    assert methods == ([] if stage == "GET" else ["GET"]) + [stage] * 4
    assert pauses == [1, 2, 4]


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("stage", ["GET", "POST"])
@pytest.mark.parametrize("fault", ["obtain", "refresh", "unreachable", "empty"])
def test_credentials_refuse_at_once_without_recording(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, stage: str, fault: str
) -> None:
    client, methods, pauses = connection(
        monkeypatch, lambda method, count: (401, b"") if method == stage else (200, answer())
    )
    tokens: list[str] = []

    def token() -> str:
        tokens.append("fetch")
        if fault != "obtain" and len(tokens) == 1:
            return "dummy-bearer"
        if fault == "empty":
            return ""
        if fault == "unreachable":
            raise ConnectionError("token source unavailable")
        raise ValueError("credentials unavailable")

    client.token = token
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    runner.steps.chat_client = client
    before = state(runner)["payload"]["chat"]
    with pytest.raises(chat.ConfigurationInterruption, match="credentials unavailable"):
        runner.execute(step)
    saved = state(runner, step)
    assert (saved["status"], saved["reason"]) == ("interrupted", "ConfigurationInterruption")
    assert saved["payload"]["chat"] == before
    assert not saved["payload"].get("chat_endings") and not pauses
    assert len(tokens) == (1 if fault == "obtain" else 2)
    assert methods == ([] if fault == "obtain" else ["GET"] if stage == "GET" else ["GET", "POST"])


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
@pytest.mark.parametrize("secret", ["fixture-project", "dummy-bearer"])
@pytest.mark.parametrize(
    "shape", ["valid", "missing version", "invalid tokens", "unprocessable body"]
)
def test_secret_refusal_precedes_response_validation_and_can_repeat(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str, secret: str, shape: str
) -> None:
    def wire(method: str, count: int) -> tuple[int, bytes]:
        if method == "GET":
            return 200, b"{}"
        result = json.loads(answer(secret))
        if shape == "missing version":
            del result["modelVersion"]
        elif shape == "invalid tokens":
            result["usageMetadata"]["promptTokenCount"] = -1
        elif shape == "unprocessable body":
            result["candidates"][0]["content"]["parts"].append(None)
        return 200, json.dumps(result).encode()

    client, methods, pauses = connection(monkeypatch, wire)
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    runner.steps.chat_client = client
    before = state(runner)["payload"]["chat"]
    for _ in range(2):
        with pytest.raises(chat.GuardRefusal, match="no record was written"):
            runner.execute(step)
        saved = state(runner, step)
        assert (saved["status"], saved["reason"]) == ("interrupted", "GuardRefusal")
        assert saved["payload"]["chat"] == before
        assert not saved["payload"].get("chat_endings")
        with runner.guard.lock(), pytest.raises(clients.GuardRefusal, match="prerequisite"):
            runner.guard.before("held-out")
    assert methods == ["GET", "POST"] * 2 and not pauses


@pytest.mark.parametrize("step", ["closed-loop", "recall"])
def test_paid_reply_lost_before_append_is_requested_again(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    client, methods, _ = connection(monkeypatch, lambda method, count: (200, answer()))
    runner = runs(prepared="closed-loop" if step == "recall" else "bands")
    runner.steps.chat_client = client
    before = state(runner)["payload"]["chat"]
    path = runner.guard.transcript_directory / "chat.jsonl"
    raw = path.read_bytes() if path.exists() else b""

    def append(*args: Any, **kwargs: Any) -> None:
        raise OSError("append unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(chat.Transcript, "append", append)
        with pytest.raises(OSError, match="append unavailable"):
            runner.execute(step)
    saved = state(runner, step)
    assert saved["status"] == "interrupted"
    assert saved["payload"]["chat"] == before
    assert (path.read_bytes() if path.exists() else b"") == raw
    assert methods == ["GET", "POST"]
    completed = runner.execute(step)
    sent = methods.count("POST")
    recorded = completed["payload"]["chat"]["calls"] - before["calls"]
    assert completed["status"] == "completed"
    assert sent == recorded + 1
    assert recorded == (8 if step == "recall" else 440)
