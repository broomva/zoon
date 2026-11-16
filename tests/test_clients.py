"""Recorded-shape responses only; no service is contacted."""

import json
import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from zoon.pilot import clients, prices, prompts

REQUEST = prompts.Decision("A state", "A question?", prompts.declared(("Yes", "No")))


def response(family: clients.Family = "decision") -> clients.Response:
    return clients.StubClient(family).ask(REQUEST)


def test_first_token_variants_missing_answers_and_rescaling() -> None:
    candidates = [
        {"token": "Yes", "logProbability": math.log(0.2)},
        {"token": " Yes", "logProbability": math.log(0.4)},
        {"token": "No", "logProbability": math.log(0.1)},
        {"token": "yes", "logProbability": math.log(0.05)},
        {"token": "  No", "logProbability": math.log(0.05)},
    ]
    p, share = clients.token_probabilities(REQUEST, candidates)
    assert share == pytest.approx(0.7)
    assert p == pytest.approx({"Yes": 6 / 7, "No": 1 / 7})
    p, share = clients.token_probabilities(REQUEST, candidates[:2])
    assert p == {"Yes": 1, "No": 0}
    assert share == pytest.approx(0.6)
    assert clients.token_probabilities(REQUEST, []) == ({"Yes": 0, "No": 0}, 0)
    context = replace(REQUEST, answers=prompts.declared(prompts.contexts()))
    p, share = clients.token_probabilities(context, [{"token": " B", "logProbability": -1}])
    assert p["Accident"] == 1 and sum(p.values()) == 1
    assert share == pytest.approx(math.exp(-1))


@pytest.mark.parametrize("share,usable", [(0.5, True), (0.4999, False), (0, False)])
def test_share_gate(tmp_path: Path, share: float, usable: bool) -> None:
    r = replace(response("second"), share=share)
    session = clients.Session(tmp_path, [clients.StubClient("second", {clients.key(REQUEST): r})])
    if usable:
        assert session.ask_many("second", [REQUEST]) == [r]
    else:
        with pytest.raises(clients.FamilyUnusable, match="one half"):
            session.ask_many("second", [REQUEST])
        assert session.ask_many("second", ["recognition"], recognition=True)[0].text == "unknown"


def test_replay_spending_digest_and_pool_order(tmp_path: Path) -> None:
    requests = [replace(REQUEST, state=str(i)) for i in range(33)]
    digests: list[str] = []
    for workers in (1, 2, 7):
        directory = tmp_path / str(workers)
        session = clients.Session(directory, [clients.StubClient("decision")], workers)
        actual = session.ask_many("decision", requests + requests[:1])
        assert actual == [clients.StubClient("decision").ask(r) for r in requests + requests[:1]]
        journal = session.transcripts["decision"]
        assert len(journal.entries) == len(requests)
        assert journal.spent == 33 * prices.DECISION.dollars(12, 1)
        digests.append(journal.digest())
        before = journal.path.read_bytes()
        restarted = clients.Session(directory, [])
        assert restarted.spent == session.spent
        assert restarted.ask_many("decision", list(reversed(requests))) == list(
            reversed(actual[:-1])
        )
        assert journal.path.read_bytes() == before
    assert len(set(digests)) == 1


def test_cap_prevents_sending_and_is_shared(tmp_path: Path) -> None:
    session = clients.Session(tmp_path, [clients.StubClient("decision")])
    session.ask_many("decision", [REQUEST])
    spent = session.spent
    next_request = replace(REQUEST, state="next")
    session.cap = spent + clients.estimate("decision", next_request) - Decimal("0.000000001")
    with pytest.raises(clients.StopRun, match="cap reached"):
        session.ask_many("decision", [next_request])
    assert len(session.transcripts["decision"].entries) == 1
    assert session.spent == spent
    # An answered request remains available even at the cap.
    assert session.ask_many("decision", [REQUEST]) == [response()]


@pytest.mark.parametrize("family", clients.budget.FAMILIES)
def test_version_changes_survive_restart(tmp_path: Path, family: clients.Family) -> None:
    r = replace(response(family), version="other-version")
    session = clients.Session(tmp_path, [clients.StubClient(family, {clients.key(REQUEST): r})])
    with pytest.raises(clients.StopRun, match="version changed"):
        session.ask_many(family, [REQUEST])
    with pytest.raises(clients.StopRun, match="version changed"):
        clients.Session(tmp_path, []).ask_many(family, [REQUEST])


def test_second_version_must_stay_fixed(tmp_path: Path) -> None:
    first = replace(response("second"), version=prices.SECOND.model + "-001")
    second = replace(first, version=prices.SECOND.model + "-002")
    next_request = replace(REQUEST, state="next")
    stub = clients.StubClient(
        "second", {clients.key(REQUEST): first, clients.key(next_request): second}
    )
    session = clients.Session(tmp_path, [stub])
    session.ask_many("second", [REQUEST])
    with pytest.raises(clients.FamilyUnusable):
        session.ask_many("second", [next_request])


def vertex_shape() -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": "Yes"}]},
                "logprobsResult": {
                    "topCandidates": [
                        {
                            "candidates": [
                                {"token": "Yes", "logProbability": math.log(0.6)},
                                {"token": "No", "logProbability": math.log(0.2)},
                            ]
                        }
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 31, "candidatesTokenCount": 1},
        "modelVersion": prices.SECOND.model,
    }


def test_vertex_shape_refresh_and_transcript_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, token = "marker-project-private", "marker-token-private"
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", project)
    calls: list[dict[str, object]] = []
    tokens: list[int] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        assert project in url and token in headers["Authorization"]
        calls.append(clients.obj(json.loads(body)))
        return (401, {}) if len(calls) == 1 else (200, vertex_shape())

    def provider() -> str:
        tokens.append(1)
        return token

    session = clients.Session(tmp_path, [clients.VertexClient(provider, transport)])
    r = session.ask_many("second", [REQUEST])[0]
    assert r.probabilities == pytest.approx({"Yes": 0.75, "No": 0.25})
    assert r.share == pytest.approx(0.8)
    assert (r.input_tokens, r.output_tokens, r.text) == (31, 1, "")
    assert len(tokens) == 2
    config = clients.obj(calls[-1]["generationConfig"])
    assert config == {
        "temperature": 0,
        "maxOutputTokens": 1,
        "thinkingConfig": {"thinkingBudget": 0},
        "responseLogprobs": True,
        "logprobs": 20,
    }
    transcript = session.transcripts["second"].path.read_text()
    assert project not in transcript and token not in transcript
    session.ask_many("second", ["recognise this"], recognition=True)
    assert calls[-1]["generationConfig"] == {
        "temperature": 0,
        "maxOutputTokens": 40,
        "thinkingConfig": {"thinkingBudget": 0},
    }


def test_transport_exception_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "private-marker")

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        raise ValueError("private-marker secret-token " + url)

    with pytest.raises(clients.StopRun) as error:
        clients.VertexClient(lambda: "secret-token", transport).ask(REQUEST)
    assert "private-marker" not in str(error.value)
    assert "secret-token" not in str(error.value)
    assert error.value.__suppress_context__


def test_decision_recorded_shape_integer_keys() -> None:
    scale = replace(REQUEST, answers=prompts.declared(tuple(map(str, range(1, 8)))))

    class SDK:
        def Choice(self, *, instructions: str, criteria: dict[str, str]) -> object:
            assert instructions == scale.question
            assert criteria == {a.answer: a.description for a in scale.answers}
            return object()

        def TypeSafeClient(self, *, model: str, timeout: float) -> object:
            assert model == prices.DECISION.model and timeout == 30
            return self

        def system_one(self, state: str, questions: dict[str, object]) -> object:
            assert state == scale.state and set(questions) == {"q"}
            return SimpleNamespace(
                answers={"q": SimpleNamespace(probabilities={i: 1 / 7 for i in range(1, 8)})},
                usage=SimpleNamespace(input_tokens=22),
                model=prices.DECISION.model,
            )

    r = clients.DecisionClient(cast(clients.DecisionSDK, SDK())).ask(scale)
    assert r.probabilities == dict.fromkeys(map(str, range(1, 8)), 1 / 7)
    assert (r.input_tokens, r.output_tokens, r.share, r.text) == (22, 0, None, "")


def test_bounded_retry_and_delays() -> None:
    calls: list[int] = []
    delays: list[float] = []

    def call() -> clients.Response:
        calls.append(1)
        if len(calls) < 3:
            raise clients.TransientError
        return response()

    assert clients.retry(call, pause=delays.append) == response()
    assert len(calls) == 3 and delays == [1, 2]
    calls.clear()
    with pytest.raises(clients.StopRun, match="bounded retries"):
        clients.retry(call, attempts=2, pause=delays.append)
    assert len(calls) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_is_rejected(bad: float) -> None:
    with pytest.raises(clients.StopRun, match="invalid"):
        clients.validate(REQUEST, replace(response(), probabilities={"Yes": bad, "No": 0}))
