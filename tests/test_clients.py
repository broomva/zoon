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
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many("decision", [next_request])
    assert len(session.transcripts["decision"].entries) == 1
    assert session.spent == spent
    # An answered request remains available even at the cap.
    assert session.ask_many("decision", [REQUEST]) == [response()]


@pytest.mark.parametrize("family", clients.budget.FAMILIES)
def test_version_changes_survive_restart(tmp_path: Path, family: clients.Family) -> None:
    r = replace(response(family), version="other-version")
    session = clients.Session(tmp_path, [clients.StubClient(family, {clients.key(REQUEST): r})])
    with pytest.raises(
        clients.TerminalStop if family == "decision" else clients.FamilyUnusable,
        match="version changed",
    ):
        session.ask_many(family, [REQUEST])
    with pytest.raises(
        clients.TerminalStop if family == "decision" else clients.FamilyUnusable,
        match="version changed",
    ):
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
                "finishReason": "STOP",
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

    with pytest.raises(clients.StepFailure) as error:
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
    assert r.probabilities == pytest.approx(dict.fromkeys(map(str, range(1, 8)), 1 / 7))
    assert (r.input_tokens, r.output_tokens, r.text) == (22, 0, "")
    assert r.share == pytest.approx(1)


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
    with pytest.raises(clients.StepFailure, match="bounded retries"):
        clients.retry(call, attempts=2, pause=delays.append)
    assert len(calls) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_is_rejected(bad: float) -> None:
    with pytest.raises(clients.StepFailure, match="invalid"):
        clients.validate(REQUEST, replace(response(), probabilities={"Yes": bad, "No": 0}))


def test_completed_order_and_cap_do_not_depend_on_worker_count(tmp_path: Path) -> None:
    import time

    class Delayed(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            assert isinstance(request, prompts.Decision)
            time.sleep((int(request.state) % 3) * 0.001)
            return replace(
                super().ask(request), input_tokens=clients.budget.input_tokens(request, "decision")
            )

    requests = [replace(REQUEST, state=str(i)) for i in range(20)]
    digests: list[str] = []
    for workers in (1, 5):
        session = clients.Session(
            tmp_path / str(workers),
            [Delayed("decision")],
            workers,
            cap=clients.estimate("decision", requests[0]) * 10,
        )
        with pytest.raises(clients.TerminalStop, match="cap reached"):
            session.ask_many("decision", requests)
        digests.append(session.transcripts["decision"].digest())
    assert len(set(digests)) == 1


@pytest.mark.parametrize(
    "status,exception",
    [
        (400, clients.StepFailure),
        (403, clients.StepFailure),
        (404, clients.FamilyUnusable),
        (410, clients.FamilyUnusable),
    ],
)
def test_permanent_http_errors_are_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch, status: int, exception: type[Exception]
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "project-marker")
    calls: list[int] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        calls.append(1)
        return status, {"error": "project-marker token-marker " + url}

    with pytest.raises(exception) as error:
        clients.VertexClient(lambda: "token-marker", transport).ask(REQUEST)
    assert len(calls) == (1 if status == 403 else 4)
    assert "marker" not in str(error.value)


def test_tokens_are_cached_until_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "marker-project")
    tokens: list[int] = []

    def provider() -> str:
        tokens.append(1)
        return "marker-token"

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        return 200, vertex_shape()

    client = clients.VertexClient(provider, transport)
    client.ask(REQUEST)
    client.ask(replace(REQUEST, state="other"))
    assert len(tokens) == 1


def test_corrupt_and_partial_transcripts_refuse(tmp_path: Path) -> None:
    path = tmp_path / "decision.jsonl"
    path.write_text('{"key":')
    with pytest.raises(clients.GuardRefusal, match="incomplete"):
        clients.Transcript(tmp_path, "decision")
    path.unlink()
    transcript = clients.Transcript(tmp_path, "decision")
    transcript.append(REQUEST, response())
    header, line = path.read_text().splitlines()
    row = json.loads(line)
    row["key"] = "wrong"
    path.write_text(header + "\n" + json.dumps(row) + "\n")
    with pytest.raises(clients.GuardRefusal, match="invalid"):
        clients.Transcript(tmp_path, "decision")


def test_private_metadata_in_reply_is_never_transcribed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "project-marker")

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        shape = vertex_shape()
        shape["candidates"] = [{"content": {"parts": [{"text": "token-marker project-marker"}]}}]
        return 200, shape

    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "token-marker", transport)])
    with pytest.raises(clients.StepFailure, match="connection metadata"):
        session.ask_many("second", ["recognition"], recognition=True)
    assert not session.transcripts["second"].path.exists()


@pytest.mark.parametrize("failure", [429, 500, 503, "timeout"])
def test_transient_http_and_timeout_backoff(
    monkeypatch: pytest.MonkeyPatch, failure: int | str
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "project-marker")
    calls: list[int] = []
    delays: list[float] = []
    original = clients.retry

    def fast_retry(call: clients.Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=delays.append)

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        calls.append(1)
        if failure == "timeout":
            raise TimeoutError
        return int(failure), {}

    monkeypatch.setattr(clients, "retry", fast_retry)
    with pytest.raises(clients.StepFailure, match="bounded retries"):
        clients.VertexClient(lambda: "token-marker", transport).ask(REQUEST)
    assert len(calls) == 4 and delays == [1, 2, 4]


def test_both_families_share_the_same_cap(tmp_path: Path) -> None:
    session = clients.Session(tmp_path, [clients.StubClient(f) for f in clients.budget.FAMILIES])
    session.ask_many("decision", [REQUEST])
    session.cap = session.spent + clients.estimate("second", REQUEST) - Decimal("0.000000001")
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many("second", [REQUEST])
    assert not session.transcripts["second"].entries


def test_cap_equality_allows_request(tmp_path: Path) -> None:
    session = clients.Session(
        tmp_path, [clients.StubClient("decision")], cap=clients.estimate("decision", REQUEST)
    )
    assert session.ask_many("decision", [REQUEST]) == [response()]


def test_numeric_first_tokens_use_the_same_rule() -> None:
    request = replace(REQUEST, answers=prompts.declared(tuple(map(str, range(1, 8)))))
    p, share = clients.token_probabilities(
        request,
        [
            {"token": "1", "logProbability": math.log(0.2)},
            {"token": " 7", "logProbability": math.log(0.6)},
        ],
    )
    assert share == pytest.approx(0.8)
    assert p["1"] == pytest.approx(0.25) and p["7"] == pytest.approx(0.75)
    assert sum(p[str(i)] for i in range(2, 7)) == 0


def test_reservations_release_after_actual_usage_is_known(tmp_path: Path) -> None:
    requests = [replace(REQUEST, state=str(i)) for i in range(20)]
    session = clients.Session(
        tmp_path, [clients.StubClient("decision")], cap=clients.estimate("decision", REQUEST) * 10
    )
    assert len(session.ask_many("decision", requests)) == 20
    assert session.spent < session.cap


def test_transcript_digest_is_the_file_sha256(tmp_path: Path) -> None:
    import hashlib

    transcript = clients.Transcript(tmp_path, "decision")
    assert transcript.digest() == hashlib.sha256(b"").hexdigest()
    transcript.append(REQUEST, response())
    assert transcript.digest() == hashlib.sha256(transcript.path.read_bytes()).hexdigest()


@pytest.mark.parametrize("total", [0.95, 0.99, 1.02, 1.05])
def test_rounded_decision_probabilities_keep_returned_values(tmp_path: Path, total: float) -> None:
    raw = {"Yes": total / 2, "No": total / 2}

    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            return SimpleNamespace(
                answers={"q": SimpleNamespace(probabilities=raw)},
                usage=SimpleNamespace(input_tokens=2),
                model=prices.DECISION.model,
            )

    session = clients.Session(tmp_path, [clients.DecisionClient(cast(clients.DecisionSDK, SDK()))])
    actual = session.ask_many("decision", [REQUEST])[0]
    assert actual.probabilities == {"Yes": 0.5, "No": 0.5}
    row = json.loads(session.transcripts["decision"].path.read_text().splitlines()[1])
    assert row["response"]["returned_probabilities"] == raw
    replay = clients.Transcript(tmp_path, "decision")
    assert replay.entries[clients.key(REQUEST)].response == actual


@pytest.mark.parametrize("bad", [0.0, 0.94, 1.06, float("nan"), float("inf"), "missing"])
def test_decision_reply_fault_retries_then_is_resumable(
    monkeypatch: pytest.MonkeyPatch, bad: float | str
) -> None:
    attempts: list[int] = []
    delays: list[float] = []
    original = clients.retry

    def fast_retry(call: clients.Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=delays.append)

    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            attempts.append(1)
            return SimpleNamespace(
                answers={}
                if bad == "missing"
                else {"q": SimpleNamespace(probabilities={"Yes": bad, "No": 0})},
                usage=SimpleNamespace(input_tokens=2),
                model=prices.DECISION.model,
            )

    monkeypatch.setattr(clients, "retry", fast_retry)
    with pytest.raises(clients.StepFailure):
        clients.DecisionClient(cast(clients.DecisionSDK, SDK())).ask(REQUEST)
    assert len(attempts) == 4 and delays == [1, 2, 4]


@pytest.mark.parametrize(
    "fault",
    ["candidates", "logprobs", "blocked", "finish", "missing finish", "probabilities"],
)
def test_unreadable_second_reply_records_zero_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    shape = vertex_shape()
    candidate = clients.obj(clients.items(shape["candidates"])[0])
    if fault == "candidates":
        shape["candidates"] = []
    elif fault == "content":
        candidate.pop("content")
    elif fault == "logprobs":
        candidate.pop("logprobsResult")
    elif fault == "blocked":
        shape["promptFeedback"] = {"blockReason": "SAFETY"}
    elif fault == "finish":
        candidate["finishReason"] = "SAFETY"
    elif fault == "missing finish":
        candidate.pop("finishReason")
    else:
        candidate["logprobsResult"] = {
            "topCandidates": [{"candidates": [{"token": "Yes", "logProbability": 1}]}]
        }

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        return 200, shape

    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "fixture-token", transport)])
    with pytest.raises(clients.FamilyUnusable):
        session.ask_many("second", [REQUEST])
    assert (
        session.transcripts["second"]
        .entries[clients.key(prompts.second_request(REQUEST))]
        .response.share
        == 0
    )
    with pytest.raises(clients.FamilyUnusable):
        clients.Session(tmp_path, []).ask_many("second", [REQUEST])


def test_second_request_disables_all_four_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    bodies: list[dict[str, object]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        bodies.append(clients.obj(json.loads(body)))
        return 200, vertex_shape()

    clients.VertexClient(lambda: "fixture-token", transport).ask(REQUEST)
    assert bodies[0]["safetySettings"] == [
        {"category": "HARM_CATEGORY_" + category, "threshold": "OFF"}
        for category in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
    ]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 429, 500, 503])
def test_second_service_faults_are_isolated_or_operator_faults(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    attempts: list[int] = []
    original = clients.retry

    def fast_retry(call: clients.Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=lambda delay: None)

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        attempts.append(1)
        return status, {}

    monkeypatch.setattr(clients, "retry", fast_retry)
    expected = (
        clients.StepFailure if status in (400, 401, 403, 429, 500, 503) else clients.FamilyUnusable
    )
    with pytest.raises(expected) as error:
        clients.VertexClient(lambda: "fixture-token", transport).ask(REQUEST)
    assert len(attempts) == (2 if status == 401 else 1 if status == 403 else 4)
    if status in (401, 403):
        assert str(error.value) == "ZOON_VERTEX_PROJECT"


@pytest.mark.parametrize("fault", ["filtered", "unknown version", "excess mass"])
def test_other_unreadable_second_replies_record_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    shape = vertex_shape()
    candidate = clients.obj(clients.items(shape["candidates"])[0])
    if fault in ("empty text", "null text"):
        candidate["content"] = {"parts": [{"text": None if fault == "null text" else " "}]}
    elif fault == "filtered":
        candidate["safetyRatings"] = [{"blocked": True}]
    elif fault == "unknown version":
        shape.pop("modelVersion")
    else:
        candidate["logprobsResult"] = {
            "topCandidates": [
                {
                    "candidates": [
                        {"token": "Yes", "logProbability": math.log(0.8)},
                        {"token": "other", "logProbability": math.log(0.8)},
                    ]
                }
            ]
        }

    def transport(*args: object) -> tuple[int, dict[str, object]]:
        return 200, shape

    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "fixture-token", transport)])
    with pytest.raises(clients.FamilyUnusable):
        session.ask_many("second", [REQUEST])
    assert (
        session.transcripts["second"]
        .entries[clients.key(prompts.second_request(REQUEST))]
        .response.share
        == 0
    )


def test_malformed_json_is_a_resumable_transport_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    class Reply(io.BytesIO):
        status = 200

    def opened(*args: object, **kwargs: object) -> Reply:
        return Reply(b"unreadable")

    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    monkeypatch.setattr(clients.urllib.request, "urlopen", opened)
    session = clients.Session(tmp_path, [clients.VertexClient(lambda: "fixture-token")])
    with pytest.raises(clients.StepFailure):
        session.ask_many("second", [REQUEST])
    assert not session.transcripts["second"].entries


def test_preflight_uses_non_generating_metadata_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    calls: list[tuple[str, dict[str, str], bytes]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
        calls.append((url, headers, body))
        return 200, {}

    clients.VertexClient(lambda: "fixture-token", preflight_transport=transport).preflight()
    url, headers, body = calls[0]
    assert url.endswith("/v1/publishers/google/models/" + prices.SECOND.model)
    assert headers["x-goog-user-project"] == "fixture"
    assert body == b""


def test_changed_decision_version_is_recorded_before_terminal_stop(tmp_path: Path) -> None:
    class SDK:
        def Choice(self, **kwargs: object) -> object:
            return object()

        def TypeSafeClient(self, **kwargs: object) -> object:
            return self

        def system_one(self, *args: object) -> object:
            return SimpleNamespace(
                answers={}, usage=SimpleNamespace(input_tokens=2), model="changed-version"
            )

    session = clients.Session(tmp_path, [clients.DecisionClient(cast(clients.DecisionSDK, SDK()))])
    with pytest.raises(clients.TerminalStop, match="version changed"):
        session.ask_many("decision", [REQUEST])
    recorded = session.transcripts["decision"].entries[clients.key(REQUEST)].response
    assert recorded.version == "changed-version"
    assert (recorded.input_tokens, recorded.output_tokens) == (2, 0)
    with pytest.raises(clients.TerminalStop, match="version changed"):
        clients.Session(tmp_path, []).ask_many("decision", [REQUEST])


def test_recorded_cost_is_checked_after_final_window_before_retryable_failure(
    tmp_path: Path,
) -> None:
    class Expensive(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            return replace(super().ask(request), input_tokens=100000, output_tokens=0)

    session = clients.Session(tmp_path, [Expensive("decision")], cap=Decimal("0.003"))
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many("decision", [REQUEST])
    assert session.spent == Decimal("0.0042")


def test_cap_precedes_terminal_failure_in_same_window(tmp_path: Path) -> None:
    requests = sorted([REQUEST, replace(REQUEST, state="another")], key=clients.key)

    class Expensive(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if request == requests[0]:
                raise clients.TerminalStop(clients.StopReason.WITHDRAWN)
            return replace(super().ask(request), input_tokens=4_000_000_000)

    session = clients.Session(tmp_path, [Expensive("decision")])
    with pytest.raises(clients.TerminalStop, match="cap reached"):
        session.ask_many("decision", requests)


def test_second_retry_exhaustion_is_resumable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOON_VERTEX_PROJECT", "fixture")
    original = clients.retry

    def fast_retry(call: clients.Callable[[], clients.Response]) -> clients.Response:
        return original(call, pause=lambda delay: None)

    monkeypatch.setattr(clients, "retry", fast_retry)

    def unavailable(*args: object) -> tuple[int, dict[str, object]]:
        return 503, {}

    with pytest.raises(clients.StepFailure, match="bounded retries"):
        clients.VertexClient(lambda: "fixture-token", unavailable).ask(REQUEST)
