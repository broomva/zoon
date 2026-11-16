"""Offline regression cases for request and terminal-state contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.test_clients import REQUEST, response, vertex_shape
from zoon.pilot import budget, clients, prices, prompts, run

SYSTEM = (
    "Reply with the answer only: a single digit, word or letter, with nothing before or after it."
)


def test_probability_system_is_rendered_hashed_budgeted_and_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    bodies: list[dict[str, object]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        bodies.append(clients.obj(json.loads(body)))
        return 200, vertex_shape()

    request = prompts.second_request(REQUEST)
    assert asdict(request) == {"system": SYSTEM, "text": prompts.second(REQUEST)}
    assert budget.input_tokens(request, "second") == (len(SYSTEM + request.text) + 2) // 3
    assert budget.input_tokens(REQUEST, "second") == budget.input_tokens(request, "second")
    assert clients.estimate("second", REQUEST) == prices.SECOND.dollars(
        budget.input_tokens(request, "second"), 1
    )
    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "fixture-token", transport)])
    session.ask_many("second", [REQUEST])
    row = json.loads(session.transcripts["second"].path.read_text())
    assert row["request"] == asdict(request)
    assert row["key"] == clients.key(request)
    assert clients.key(replace(request, system="changed")) != row["key"]
    assert bodies[0]["systemInstruction"] == {"parts": [{"text": SYSTEM}]}
    assert bodies[0]["contents"] == [{"role": "user", "parts": [{"text": request.text}]}]
    session.ask_many("second", ["recognition"], recognition=True)
    assert "systemInstruction" not in bodies[-1]


def test_smoke_probability_calls_carry_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    bodies: list[dict[str, object]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        bodies.append(clients.obj(json.loads(body)))
        return 200, vertex_shape()

    class Second(clients.VertexClient):
        def __init__(self) -> None:
            super().__init__(lambda: "fixture-token", transport)

        def preflight(self) -> None:
            pass

    monkeypatch.setattr(clients, "DecisionClient", lambda: clients.StubClient("decision"))
    monkeypatch.setattr(clients, "VertexClient", Second)
    run.smoke(False)
    assert len(bodies) == 3
    assert all(b["systemInstruction"] == {"parts": [{"text": SYSTEM}]} for b in bodies)


@pytest.mark.parametrize("content", [None, {}, {"parts": []}, {"parts": [{"text": None}]}])
@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS"])
def test_probability_reply_needs_no_text(content: object, finish: str) -> None:
    shape = vertex_shape()
    candidate = clients.obj(clients.items(shape["candidates"])[0])
    candidate["content"] = content
    candidate["finishReason"] = finish
    reply = clients.VertexClient().read(REQUEST, shape)
    assert reply.share == pytest.approx(0.8)
    assert reply.probabilities == pytest.approx({"Yes": 0.75, "No": 0.25})


class DecisionReply:
    def __init__(
        self, raw: dict[str | int, float], output: int | None = 0, model: str = ""
    ) -> None:
        self.raw, self.output, self.model = raw, output, model or prices.DECISION.model

    def Choice(self, **kwargs: object) -> object:
        return object()

    def TypeSafeClient(self, **kwargs: object) -> object:
        return self

    def system_one(self, *args: object) -> object:
        return SimpleNamespace(
            answers={"q": SimpleNamespace(probabilities=self.raw)},
            usage=SimpleNamespace(input_tokens=2, output_tokens=self.output),
            model=self.model,
        )


@pytest.mark.parametrize("raw", [{1: 0.2, "1": 0.4, "2": 0.6}, {"1": 1.0}])
def test_decision_label_fault_is_resumable(
    monkeypatch: pytest.MonkeyPatch, raw: dict[str | int, float]
) -> None:
    original = clients.retry

    def fast_retry(call: clients.Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=lambda delay: None)

    monkeypatch.setattr(clients, "retry", fast_retry)
    request = replace(REQUEST, answers=prompts.declared(("1", "2")))
    with pytest.raises(clients.StepFailure):
        clients.DecisionClient(cast(clients.DecisionSDK, DecisionReply(raw))).ask(request)


def test_null_output_usage_is_zero() -> None:
    reply = clients.DecisionClient(
        cast(clients.DecisionSDK, DecisionReply({"Yes": 0.5, "No": 0.5}, None))
    ).ask(REQUEST)
    assert reply.output_tokens == 0


def test_decision_secret_echo_is_never_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-key-marker")
    session = clients.Session(
        tmp_path,
        [
            clients.DecisionClient(
                cast(
                    clients.DecisionSDK,
                    DecisionReply({"Yes": 0.5, "No": 0.5}, model="fixture-key-marker"),
                )
            )
        ],
    )
    with pytest.raises(clients.StepFailure, match="connection metadata"):
        session.ask_many("decision", [REQUEST])
    assert not session.transcripts["decision"].entries
    assert not session.transcripts["decision"].path.exists()


@pytest.mark.parametrize("recognition", [False, True])
def test_cap_replay_checks_even_without_pending(tmp_path: Path, recognition: bool) -> None:
    session = clients.Session(tmp_path, [])
    request = "recognition" if recognition else REQUEST
    session.transcripts["decision"].append(request, replace(response(), input_tokens=4_000_000_000))
    with pytest.raises(clients.StopRun, match="cap reached"):
        clients.Session(tmp_path, []).ask_many("decision", [request], recognition=recognition)
