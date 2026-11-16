"""Probability clients and resumable, budgeted transcripts."""

import hashlib
import importlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

from zoon.pilot import budget, prices, prompts

Request = prompts.Decision | str
Family = budget.Family
CAP = Decimal("150")
WINDOW = 16


class StopRun(ValueError):
    pass


class FamilyUnusable(StopRun):
    pass


class StepFailure(StopRun):
    pass


class TransientError(Exception):
    pass


@dataclass(frozen=True)
class Response:
    probabilities: dict[str, float]
    share: float | None
    text: str
    version: str
    input_tokens: int
    output_tokens: int


class Client(Protocol):
    family: Family

    def ask(self, request: Request) -> Response: ...


def canonical(request: Request) -> bytes:
    value = asdict(request) if isinstance(request, prompts.Decision) else request
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def key(request: Request) -> str:
    return hashlib.sha256(canonical(request)).hexdigest()


def cost(family: Family, response: Response) -> Decimal:
    price = prices.DECISION if family == "decision" else prices.SECOND
    return price.dollars(response.input_tokens, response.output_tokens)


def estimate(family: Family, request: Request) -> Decimal:
    price = prices.DECISION if family == "decision" else prices.SECOND
    output = 40 if isinstance(request, str) else 1
    return price.dollars(budget.input_tokens(wire(request, family), family), output)


def wire(request: Request, family: Family) -> Request:
    return (
        prompts.second(request)
        if family == "second" and isinstance(request, prompts.Decision)
        else request
    )


def validate(request: Request, response: Response) -> None:
    if not response.version or response.input_tokens < 0 or response.output_tokens < 0:
        raise StopRun("invalid reported version or token usage")
    if isinstance(request, str):
        if response.probabilities:
            raise StopRun("recognition returned unexpected probabilities")
        return
    labels = {a.answer for a in request.answers}
    values = response.probabilities
    if set(values) != labels or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()
    ):
        raise StopRun("invalid declared-answer probabilities")
    empty = response.share == 0 and sum(values.values()) == 0
    if not empty and not math.isclose(sum(values.values()), 1, abs_tol=1e-7):
        raise StopRun("declared-answer probabilities do not sum to one")
    if response.share is not None and (
        not math.isfinite(response.share) or not 0 <= response.share <= 1 + 1e-7
    ):
        raise StopRun("invalid declared-answer share")


def retry(
    call: Callable[[], Response], attempts: int = 4, pause: Callable[[float], None] = time.sleep
) -> Response:
    for attempt in range(attempts):
        try:
            return call()
        except (TransientError, TimeoutError):
            if attempt + 1 == attempts:
                raise StepFailure("transient request failure after bounded retries") from None
            pause(2.0**attempt)
    raise StopRun("request attempts must be positive")


class _Usage(Protocol):
    input_tokens: int


class _Answer(Protocol):
    probabilities: Mapping[str | int, float]


class _Result(Protocol):
    answers: Mapping[str, _Answer]
    usage: _Usage
    model: str


class _DecisionTransport(Protocol):
    def system_one(self, state: str, questions: dict[str, object]) -> _Result: ...


class DecisionSDK(Protocol):
    def Choice(self, *, instructions: str, criteria: dict[str, str]) -> object: ...
    def TypeSafeClient(self, *, model: str, timeout: float) -> _DecisionTransport: ...


class DecisionClient:
    family: Family = "decision"

    def __init__(self, sdk: DecisionSDK | None = None) -> None:
        self._sdk = sdk

    def ask(self, request: Request) -> Response:
        if not isinstance(request, prompts.Decision):
            raise StopRun("decision family cannot answer recognition")

        def call() -> Response:
            try:
                sdk = self._sdk or cast(DecisionSDK, importlib.import_module("typesafe_sdk"))
                client = sdk.TypeSafeClient(model=prices.DECISION.model, timeout=30.0)
                result = client.system_one(
                    request.state,
                    {
                        "q": sdk.Choice(
                            instructions=request.question,
                            criteria={a.answer: a.description for a in request.answers},
                        )
                    },
                )
                probabilities = {
                    str(label): float(p) for label, p in result.answers["q"].probabilities.items()
                }
                output = cast(int, getattr(result.usage, "output_tokens", 0))
                response = Response(
                    probabilities, None, "", result.model, result.usage.input_tokens, output
                )
                validate(request, response)
                return response
            except (StopRun, TransientError, TimeoutError):
                raise
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    raise TransientError from None
                if "timeout" in type(error).__name__.lower():
                    raise TimeoutError from None
                raise StepFailure(
                    "decision request failed; check service availability and live installation"
                ) from None

        return retry(call)


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StopRun("invalid response object")
    return cast(dict[str, object], value)


def items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise StopRun("invalid response list")
    return cast(list[object], value)


def number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise StopRun("invalid response number")
    return float(value)


def token_probabilities(
    request: prompts.Decision, candidates: Sequence[Mapping[str, object]]
) -> tuple[dict[str, float], float]:
    labels = [a.answer for a in request.answers]
    tokens = list("ABCDEFGHI") if tuple(labels) == prompts.contexts() else labels
    mass = dict.fromkeys(labels, 0.0)
    seen: set[str] = set()
    for candidate in candidates:
        token = str(candidate["token"])
        if token in seen:
            raise StopRun("duplicate token in top candidates")
        seen.add(token)
        logp = number(candidate["logProbability"])
        if not math.isfinite(logp) or logp > 0:
            raise StopRun("invalid token log probability")
        for label, expected in zip(labels, tokens, strict=True):
            if token in (expected, " " + expected):
                mass[label] += math.exp(logp)
    share = sum(mass.values())
    return {label: p / share if share else 0.0 for label, p in mass.items()}, share


HTTP = Callable[[str, dict[str, str], bytes], tuple[int, dict[str, object]]]


def post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
    try:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, obj(json.loads(reply.read()))
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (TimeoutError, urllib.error.URLError):
        raise TimeoutError from None
    except Exception:
        raise StepFailure("second-family transport failed") from None


def access_token() -> str:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.strip()
        if not token:
            raise ValueError
        return token
    except Exception:
        raise StepFailure("could not obtain an application-default access token") from None


class VertexClient:
    family: Family = "second"

    def __init__(self, token: Callable[[], str] = access_token, transport: HTTP = post) -> None:
        self._token = token
        self._transport = transport
        self._bearer: str | None = None
        self._token_lock = Lock()

    def bearer(self, rejected: str | None = None) -> str:
        with self._token_lock:
            if self._bearer is None or self._bearer == rejected:
                self._bearer = self._token()
            return self._bearer

    def ask(self, request: Request) -> Response:
        def call() -> Response:
            try:
                project = os.environ.get("ZOON_VERTEX_PROJECT", "")
                if not project:
                    raise StopRun("ZOON_VERTEX_PROJECT is required")
                url = (
                    "https://aiplatform.googleapis.com/v1/projects/"
                    + project
                    + "/locations/global/publishers/google/models/"
                    + prices.SECOND.model
                    + ":generateContent"
                )
                text = wire(request, self.family)
                config: dict[str, object] = {
                    "temperature": 0,
                    "maxOutputTokens": 40 if isinstance(request, str) else 1,
                    "thinkingConfig": {"thinkingBudget": 0},
                }
                if isinstance(request, prompts.Decision):
                    config.update(responseLogprobs=True, logprobs=20)
                body = json.dumps(
                    {
                        "contents": [{"role": "user", "parts": [{"text": text}]}],
                        "generationConfig": config,
                    }
                ).encode()
                status, data = 401, {}
                bearer = self.bearer()
                for refresh in range(2):
                    status, data = self._transport(
                        url,
                        {
                            "Authorization": "Bearer " + bearer,
                            "Content-Type": "application/json",
                        },
                        body,
                    )
                    if status != 401 or refresh:
                        break
                    bearer = self.bearer(rejected=bearer)
                if status == 429 or 500 <= status < 600:
                    raise TransientError
                if status in (404, 410):
                    raise FamilyUnusable("second family model is unavailable")
                if status != 200:
                    raise StepFailure(f"second-family request failed with HTTP {status}")
                candidate = obj(items(data["candidates"])[0])
                reply = str(obj(items(obj(candidate["content"])["parts"])[0])["text"])
                usage = obj(data["usageMetadata"])
                probabilities: dict[str, float] = {}
                share = None
                if isinstance(request, prompts.Decision):
                    top = obj(items(obj(candidate["logprobsResult"])["topCandidates"])[0])
                    probabilities, share = token_probabilities(
                        request, [obj(c) for c in items(top["candidates"])]
                    )
                response = Response(
                    probabilities,
                    share,
                    reply if isinstance(request, str) else "",
                    str(data["modelVersion"]),
                    int(number(usage["promptTokenCount"])),
                    int(number(usage["candidatesTokenCount"])),
                )
                if any(
                    secret and secret in json.dumps(asdict(response))
                    for secret in (project, bearer)
                ):
                    raise StopRun("connection metadata in response; no transcript was written")
                validate(request, response)
                return response
            except (StopRun, TransientError, TimeoutError):
                raise
            except Exception:
                raise StopRun("second-family response or transport invalid") from None

        return retry(call)


class StubClient:
    def __init__(self, family: Family, recorded: Mapping[str, Response] | None = None) -> None:
        self.family: Family = family
        self.recorded = recorded or {}

    def ask(self, request: Request) -> Response:
        if key(request) in self.recorded:
            return self.recorded[key(request)]
        version = prices.DECISION.model if self.family == "decision" else prices.SECOND.model
        if isinstance(request, str):
            return Response({}, None, "unknown", version, 12, 2)
        weights = [
            1 + int(hashlib.sha256((key(request) + a.answer).encode()).hexdigest()[:6], 16) % 100
            for a in request.answers
        ]
        probabilities = {
            a.answer: w / sum(weights) for a, w in zip(request.answers, weights, strict=True)
        }
        return Response(probabilities, 0.9 if self.family == "second" else None, "", version, 12, 1)


@dataclass(frozen=True)
class Entry:
    key: str
    request: object
    response: Response


class Transcript:
    def __init__(self, directory: Path, family: Family) -> None:
        self.family: Family = family
        self.path = directory / f"{family}.jsonl"
        self.entries: dict[str, Entry] = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise StopRun(
                    "incomplete transcript line; restore the last complete line before resuming"
                )
            for line in raw.splitlines():
                row = obj(json.loads(line))
                payload = obj(row["response"])
                response = Response(
                    cast(dict[str, float], payload["probabilities"]),
                    cast(float | None, payload["share"]),
                    str(payload["text"]),
                    str(payload["version"]),
                    int(number(payload["input_tokens"])),
                    int(number(payload["output_tokens"])),
                )
                digest = str(row["key"])
                encoded = json.dumps(
                    row["request"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
                if hashlib.sha256(encoded).hexdigest() != digest or digest in self.entries:
                    raise StopRun("invalid or duplicate transcript key")
                self.entries[digest] = Entry(digest, row["request"], response)
        self._spent = sum(
            (cost(self.family, e.response) for e in self.entries.values()), Decimal(0)
        )

    @property
    def spent(self) -> Decimal:
        return self._spent

    def append(self, request: Request, response: Response) -> None:
        digest = key(request)
        if digest in self.entries:
            return
        entry = Entry(digest, json.loads(canonical(request)), response)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(
                json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[digest] = entry
        self._spent += cost(self.family, response)

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes() if self.path.exists() else b"").hexdigest()


class Session:
    def __init__(
        self, directory: Path, clients: Sequence[Client], workers: int = 4, cap: Decimal = CAP
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.clients = {c.family: c for c in clients}
        self.transcripts: dict[Family, Transcript] = {
            f: Transcript(directory, f) for f in budget.FAMILIES
        }
        self.workers = workers
        self.cap = cap

    @property
    def spent(self) -> Decimal:
        return sum((t.spent for t in self.transcripts.values()), Decimal(0))

    def check_family(self, family: Family) -> None:
        entries = list(self.transcripts[family].entries.values())
        probabilities = [e for e in entries if not isinstance(e.request, str)]
        if family == "decision":
            if any(e.response.version != prices.DECISION.model for e in entries):
                raise StopRun("decision model version changed; no verdict")
        else:
            versions = {e.response.version for e in entries}
            if len(versions) > 1 or any(not v.startswith(prices.SECOND.model) for v in versions):
                raise FamilyUnusable("second family version changed")
            if any(e.response.share is None or e.response.share < 0.5 for e in probabilities):
                raise FamilyUnusable("second family declared-answer share below one half")

    def ask_many(
        self, family: Family, requests: Sequence[Request], *, recognition: bool = False
    ) -> list[Response]:
        transcript = self.transcripts[family]
        if not recognition:
            self.check_family(family)
        unique = {key(r): r for r in requests if key(r) not in transcript.entries}
        pending = [unique[k] for k in sorted(unique)]
        start = 0
        while start < len(pending):
            window: list[Request] = []
            reserved = Decimal(0)
            for request in pending[start : start + WINDOW]:
                addition = estimate(family, request)
                if self.spent + reserved + addition > self.cap:
                    break
                window.append(request)
                reserved += addition

            if not window:
                raise StopRun("spending cap reached; no verdict")

            # Admission and recording use a fixed window and canonical order, independent of
            # the number of workers or the order in which requests complete.
            def call(request: Request) -> Response | StopRun:
                try:
                    response = self.clients[family].ask(request)
                    validate(request, response)
                    return response
                except StopRun as error:
                    return error
                except Exception:
                    return StopRun("request failed")

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(call, window))
            failure: StopRun | None = None
            for request, result in zip(window, results, strict=True):
                if isinstance(result, StopRun):
                    failure = failure or result
                else:
                    transcript.append(request, result)
            if failure:
                raise failure
            if self.spent > self.cap:
                raise StopRun("spending cap reached; no verdict")
            if not recognition:
                self.check_family(family)
            start += len(window)
        if not recognition:
            self.check_family(family)
        return [transcript.entries[key(r)].response for r in requests]
