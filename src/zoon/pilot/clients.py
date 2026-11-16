"""Probability clients and resumable, budgeted transcripts."""

import hashlib
import http.client
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
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

from zoon.pilot import budget, prices, prompts

Request = prompts.Decision | str
Family = budget.Family
CAP = Decimal("150")
WINDOW = 16
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_" + category, "threshold": "OFF"}
    for category in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
]


class GuardRefusal(Exception):
    pass


class StepFailure(Exception):
    pass


class FamilyCause(Enum):
    SHARE = "second family declared-answer share below one half"
    VERSION = "second family version changed"
    WITHDRAWN = "second family withdrawn"
    CALIBRATION = "second family failed the stop rule"
    RECORDS = "second family complete-records failure"


class StopReason(Enum):
    WITHDRAWN = "decision model withdrawn; no verdict"
    VERSION = "decision model version changed; no verdict"
    CAP = "spending cap reached; no verdict"
    RECORDS = "complete-records failure; no verdict"
    CALIBRATION = "the judge could not be calibrated"


class FamilyUnusable(Exception):
    def __init__(self, cause: FamilyCause) -> None:
        if not isinstance(cast(object, cause), FamilyCause):
            raise TypeError("a registered family cause is required")
        self.cause = cause
        super().__init__(cause.value)


class TerminalStop(Exception):
    def __init__(
        self,
        reason: StopReason,
        *,
        response: "Response | None" = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(cast(object, reason), StopReason):
            raise TypeError("a registered stop reason is required")
        self.reason = reason
        self.response = response
        self.payload = payload or {}
        super().__init__(reason.value)


class TransientError(Exception):
    pass


class ModelMissing(TransientError):
    def __init__(self, family: Family) -> None:
        self.family = family
        super().__init__()


class UnreadableReply(Exception):
    """The second family's reply does not have the required shape or values."""


@dataclass(frozen=True)
class Response:
    probabilities: dict[str, float]
    share: float | None
    text: str
    version: str
    input_tokens: int
    output_tokens: int
    returned_probabilities: dict[str, float] | None = None


class Client(Protocol):
    family: Family

    def ask(self, request: Request) -> Response: ...


def canonical(request: Request | prompts.Second) -> bytes:
    value = asdict(request) if isinstance(request, (prompts.Decision, prompts.Second)) else request
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def key(request: Request | prompts.Second) -> str:
    return hashlib.sha256(canonical(request)).hexdigest()


def cost(family: Family, response: Response) -> Decimal:
    price = prices.DECISION if family == "decision" else prices.SECOND
    return price.dollars(response.input_tokens, response.output_tokens)


def estimate(family: Family, request: Request) -> Decimal:
    price = prices.DECISION if family == "decision" else prices.SECOND
    output = 40 if isinstance(request, str) else 1
    return price.dollars(budget.input_tokens(wire(request, family), family), output)


def wire(request: Request, family: Family) -> Request | prompts.Second:
    return (
        prompts.second_request(request)
        if family == "second" and isinstance(request, prompts.Decision)
        else request
    )


def validate(request: Request, response: Response) -> None:
    if (
        (not response.version and response.share != 0)
        or response.input_tokens < 0
        or response.output_tokens < 0
    ):
        raise StepFailure("invalid reported version or token usage")
    if isinstance(request, str):
        if response.probabilities:
            raise StepFailure("recognition returned unexpected probabilities")
        return
    labels = {a.answer for a in request.answers}
    values = response.probabilities
    if set(values) != labels or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()
    ):
        raise StepFailure("invalid declared-answer probabilities")
    empty = response.share == 0 and sum(values.values()) == 0
    if not empty and not math.isclose(sum(values.values()), 1, abs_tol=1e-7):
        raise StepFailure("declared-answer probabilities do not sum to one")
    if response.share is not None and (
        not math.isfinite(response.share)
        or not 0
        <= response.share
        <= (1.05 + 1e-12 if response.returned_probabilities else 1 + 1e-7)
    ):
        raise StepFailure("invalid declared-answer share")


def retry(
    call: Callable[[], Response], attempts: int = 4, pause: Callable[[float], None] = time.sleep
) -> Response:
    missing = 0
    for attempt in range(attempts):
        try:
            return call()
        except (
            TransientError,
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            http.client.IncompleteRead,
        ) as error:
            missing += isinstance(error, ModelMissing)
            if attempt + 1 == attempts:
                if isinstance(error, ModelMissing) and missing == attempts:
                    if error.family == "decision":
                        raise TerminalStop(StopReason.WITHDRAWN) from None
                    raise FamilyUnusable(FamilyCause.WITHDRAWN) from None
                raise StepFailure("transient request failure after bounded retries") from None
            pause(2.0**attempt)
    raise StepFailure("request attempts must be positive")


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
            raise StepFailure("decision family cannot answer recognition")

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
                reported = cast(object, getattr(result, "model", None))
                if not isinstance(reported, str) or not reported:
                    raise TransientError
                secret = os.environ.get("TYPESAFE_API_KEY", "")
                if secret and secret in repr((result.model, result.answers, result.usage)):
                    raise StepFailure("connection metadata in response; no transcript was written")
                if reported != prices.DECISION.model:
                    usage = cast(object, getattr(result, "usage", None))
                    incoming = cast(object, getattr(usage, "input_tokens", 0))
                    outgoing = cast(object, getattr(usage, "output_tokens", 0))
                    raise TerminalStop(
                        StopReason.VERSION,
                        response=Response(
                            {},
                            None,
                            "",
                            reported,
                            incoming if isinstance(incoming, int) and incoming >= 0 else 0,
                            outgoing if isinstance(outgoing, int) and outgoing >= 0 else 0,
                        ),
                    )
                raw = result.answers["q"].probabilities
                returned = {str(label): float(p) for label, p in raw.items()}
                total = sum(returned.values())
                if (
                    len(returned) != len(raw)
                    or set(returned) != {a.answer for a in request.answers}
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in returned.values())
                    or total <= 0
                    or abs(total - 1) > 0.05 + 1e-12
                ):
                    raise TransientError
                probabilities = {label: p / total for label, p in returned.items()}
                output = cast(int, getattr(result.usage, "output_tokens", 0) or 0)
                response = Response(
                    probabilities,
                    total,
                    "",
                    reported,
                    result.usage.input_tokens,
                    output,
                    returned,
                )
                try:
                    validate(request, response)
                except StepFailure:
                    raise TransientError from None
                return response
            except (StepFailure, TerminalStop, TransientError, TimeoutError):
                raise
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    raise TransientError from None
                if "timeout" in type(error).__name__.lower():
                    raise TimeoutError from None
            if status in (404, 410):
                raise ModelMissing("decision") from None
            if status in (401, 403):
                raise StepFailure("decision authentication failed") from None
            raise TransientError from None

        return retry(call)


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StepFailure("invalid response object")
    return cast(dict[str, object], value)


def items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise StepFailure("invalid response list")
    return cast(list[object], value)


def number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise StepFailure("invalid response number")
    return float(value)


def reply_obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UnreadableReply
    return cast(dict[str, object], value)


def reply_items(value: object, *, nonempty: bool = True) -> list[object]:
    if not isinstance(value, list) or (nonempty and not value):
        raise UnreadableReply
    return cast(list[object], value)


def reply_number(value: object) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise UnreadableReply
    return float(value)


def token_probabilities(
    request: prompts.Decision, candidates: Sequence[Mapping[str, object]]
) -> tuple[dict[str, float], float]:
    labels = [a.answer for a in request.answers]
    tokens = list("ABCDEFGHI") if tuple(labels) == prompts.contexts() else labels
    mass = dict.fromkeys(labels, 0.0)
    seen: set[str] = set()
    total = 0.0
    for candidate in candidates:
        token = candidate.get("token")
        if not isinstance(token, str):
            raise UnreadableReply("invalid candidate token")
        if token in seen:
            raise UnreadableReply("duplicate token in top candidates")
        seen.add(token)
        logp = reply_number(candidate.get("logProbability"))
        if not math.isfinite(logp) or logp > 0:
            raise UnreadableReply("invalid token log probability")
        total += math.exp(logp)
        if total > 1 + 1e-7:
            raise UnreadableReply("invalid top-candidate probability mass")
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
            try:
                return reply.status, obj(json.loads(reply.read()))
            except (ValueError, TypeError):
                raise TransientError from None
            except StepFailure:
                return reply.status, {}
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (TimeoutError, urllib.error.URLError):
        raise TimeoutError from None
    except (ConnectionError, http.client.IncompleteRead, OSError):
        raise TransientError from None


def get(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, object]]:
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, {}
    except urllib.error.HTTPError as error:
        return error.code, {}
    except (TimeoutError, urllib.error.URLError):
        raise TimeoutError from None


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
        raise StepFailure("ZOON_VERTEX_PROJECT") from None


class VertexClient:
    family: Family = "second"

    def __init__(
        self,
        token: Callable[[], str] = access_token,
        transport: HTTP = post,
        *,
        preflight_transport: HTTP = get,
    ) -> None:
        self._token = token
        self._transport = transport
        self._preflight_transport = preflight_transport
        self._bearer: str | None = None
        self._token_lock = Lock()

    def bearer(self, rejected: str | None = None) -> str:
        with self._token_lock:
            if self._bearer is None or self._bearer == rejected:
                try:
                    self._bearer = self._token()
                    if not self._bearer:
                        raise ValueError
                except Exception:
                    raise StepFailure("ZOON_VERTEX_PROJECT") from None
            return self._bearer

    def endpoint(self) -> str:
        project = os.environ.get("ZOON_VERTEX_PROJECT", "")
        if not project:
            raise StepFailure("ZOON_VERTEX_PROJECT")
        return (
            "https://aiplatform.googleapis.com/v1/projects/"
            + project
            + "/locations/global/publishers/google/models/"
            + prices.SECOND.model
        )

    def send(self, transport: HTTP, url: str, body: bytes) -> dict[str, object]:
        status, data = 401, {}
        bearer = self.bearer()
        for refresh in range(2):
            status, data = transport(
                url,
                {
                    "Authorization": "Bearer " + bearer,
                    "Content-Type": "application/json",
                    "x-goog-user-project": os.environ.get("ZOON_VERTEX_PROJECT", ""),
                },
                body,
            )
            if status != 401 or refresh:
                break
            bearer = self.bearer(rejected=bearer)
        if status in (401, 403):
            raise StepFailure("ZOON_VERTEX_PROJECT")
        if (
            status == 429
            or 500 <= status < 600
            or (400 <= status < 500 and status not in (404, 410))
        ):
            raise TransientError
        if status in (404, 410):
            raise ModelMissing("second")
        if status != 200:
            raise StepFailure("unexpected response status")
        return data

    def preflight(self) -> None:
        self.endpoint()
        try:
            retry(self._probe)
        except FamilyUnusable:
            pass
        except StepFailure as error:
            if str(error) == "ZOON_VERTEX_PROJECT":
                raise
        except Exception:
            pass

    def _probe(self) -> Response:
        metadata = (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/" + prices.SECOND.model
        )
        self.send(self._preflight_transport, metadata, b"")
        return Response({}, None, "", prices.SECOND.model, 0, 0)

    def ask(self, request: Request) -> Response:
        endpoint = self.endpoint()

        def call() -> Response:
            try:
                config: dict[str, object] = {
                    "temperature": 0,
                    "maxOutputTokens": 40 if isinstance(request, str) else 1,
                    "thinkingConfig": {"thinkingBudget": 0},
                }
                if isinstance(request, prompts.Decision):
                    config.update(responseLogprobs=True, logprobs=20)
                rendered = wire(request, self.family)
                payload: dict[str, object] = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": rendered.text
                                    if isinstance(rendered, prompts.Second)
                                    else rendered
                                }
                            ],
                        }
                    ],
                    "generationConfig": config,
                    "safetySettings": SAFETY_SETTINGS,
                }
                if isinstance(rendered, prompts.Second):
                    payload["systemInstruction"] = {"parts": [{"text": rendered.system}]}
                body = json.dumps(payload).encode()
                data = self.send(self._transport, endpoint + ":generateContent", body)
                return self.read(request, data)
            except (StepFailure, FamilyUnusable, TransientError, TimeoutError):
                raise
            except Exception:
                raise StepFailure("response processing failed") from None

        return retry(call)

    def read(self, request: Request, data: dict[str, object]) -> Response:
        project = os.environ.get("ZOON_VERTEX_PROJECT", "")
        if any(secret and secret in json.dumps(data) for secret in (project, self._bearer)):
            raise StepFailure("connection metadata in response; no transcript was written")
        version = data.get("modelVersion")
        version = version if isinstance(version, str) else ""
        input_tokens = output_tokens = 0
        try:
            usage = reply_obj(data.get("usageMetadata", {}))
            input_tokens = int(reply_number(usage.get("promptTokenCount", 0)))
            output_tokens = int(reply_number(usage.get("candidatesTokenCount", 0)))
            if "blockReason" in reply_obj(data.get("promptFeedback", {})):
                raise UnreadableReply
            candidate = reply_obj(reply_items(data.get("candidates"))[0])
            if candidate.get("finishReason") not in ("STOP", "MAX_TOKENS"):
                raise UnreadableReply
            for source in (data.get("promptFeedback", {}), candidate):
                if any(
                    reply_obj(rating).get("blocked")
                    for rating in reply_items(
                        reply_obj(source).get("safetyRatings", []), nonempty=False
                    )
                ):
                    raise UnreadableReply
            reply = ""
            if isinstance(request, str):
                reply = reply_obj(
                    reply_items(reply_obj(candidate.get("content")).get("parts"))[0]
                ).get("text")
                if not isinstance(reply, str) or not reply.strip():
                    raise UnreadableReply
            probabilities: dict[str, float] = {}
            share = None
            if isinstance(request, prompts.Decision):
                top = reply_obj(
                    reply_items(reply_obj(candidate.get("logprobsResult")).get("topCandidates"))[0]
                )
                candidates = reply_items(top.get("candidates"))
                if not candidates:
                    raise UnreadableReply
                probabilities, share = token_probabilities(
                    request, [reply_obj(c) for c in candidates]
                )
            response = Response(
                probabilities,
                share,
                reply if isinstance(request, str) else "",
                version,
                input_tokens,
                output_tokens,
            )
            if not version or input_tokens < 0 or output_tokens < 0:
                raise UnreadableReply
            return response
        except UnreadableReply:
            return Response(
                {a.answer: 0.0 for a in request.answers}
                if isinstance(request, prompts.Decision)
                else {},
                0.0,
                "",
                version,
                max(0, input_tokens),
                max(0, output_tokens),
            )


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


def entry_document(entry: Entry) -> dict[str, object]:
    document = asdict(entry)
    if entry.response.returned_probabilities is None:
        obj(document["response"]).pop("returned_probabilities")
    return document


class Transcript:
    def __init__(
        self, directory: Path, family: Family, identity: dict[str, str] | None = None
    ) -> None:
        self.identity = identity or {"registered_commit": "", "mode": "isolated"}
        self.family: Family = family
        self.path = directory / f"{family}.jsonl"
        self.entries: dict[str, Entry] = {}
        try:
            if self.path.exists():
                raw = self.path.read_bytes()
                if raw and not raw.endswith(b"\n"):
                    raise GuardRefusal(
                        "incomplete transcript line; restore the last complete line before resuming"
                    )
                lines = raw.splitlines()
                if not lines or json.loads(lines[0]) != self.identity:
                    raise GuardRefusal("transcript identity does not match this run")
                for line in lines[1:]:
                    row = obj(json.loads(line))
                    payload = obj(row["response"])
                    text, version = payload["text"], payload["version"]
                    if not isinstance(text, str) or not isinstance(version, str):
                        raise GuardRefusal("invalid transcript response text or version")
                    response = Response(
                        cast(dict[str, float], payload["probabilities"]),
                        cast(float | None, payload["share"]),
                        text,
                        version,
                        int(number(payload["input_tokens"])),
                        int(number(payload["output_tokens"])),
                        cast(dict[str, float] | None, payload.get("returned_probabilities")),
                    )
                    digest = str(row["key"])
                    encoded = json.dumps(
                        row["request"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    if hashlib.sha256(encoded).hexdigest() != digest or digest in self.entries:
                        raise GuardRefusal("invalid or duplicate transcript key")
                    self.entries[digest] = Entry(digest, row["request"], response)
        except GuardRefusal:
            raise
        except Exception as error:
            raise GuardRefusal("invalid transcript: " + type(error).__name__) from None

    @property
    def spent(self) -> Decimal:
        return sum((cost(self.family, e.response) for e in self.entries.values()), Decimal(0))

    def append(self, request: Request, response: Response) -> None:
        if not isinstance(cast(object, response.version), str) or not isinstance(
            cast(object, response.text), str
        ):
            raise StepFailure("invalid response text or version")
        rendered = wire(request, self.family)
        digest = key(rendered)
        if digest in self.entries:
            return
        entry = Entry(digest, json.loads(canonical(rendered)), response)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a") as handle:
            if new:
                handle.write(json.dumps(self.identity, sort_keys=True) + "\n")
            handle.write(
                json.dumps(
                    entry_document(entry), sort_keys=True, ensure_ascii=False, allow_nan=False
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[digest] = entry

    def digest(self) -> str:
        return hashlib.sha256(self.path.read_bytes() if self.path.exists() else b"").hexdigest()


class Session:
    def __init__(
        self,
        directory: Path,
        clients: Sequence[Client],
        workers: int = 4,
        cap: Decimal = CAP,
        *,
        identity: dict[str, str] | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.clients = {c.family: c for c in clients}
        self.transcripts: dict[Family, Transcript] = {
            f: Transcript(directory, f, identity) for f in budget.FAMILIES
        }
        self.workers = workers
        self.cap = cap

    @property
    def spent(self) -> Decimal:
        return sum((t.spent for t in self.transcripts.values()), Decimal(0))

    def check_cap(self) -> None:
        if self.spent > self.cap:
            raise TerminalStop(StopReason.CAP)

    def check_family(self, family: Family) -> None:
        entries = list(self.transcripts[family].entries.values())
        probabilities = [e for e in entries if not isinstance(e.request, str)]
        if family == "decision":
            if any(e.response.version != prices.DECISION.model for e in entries):
                raise TerminalStop(StopReason.VERSION)
        else:
            versions = {e.response.version for e in entries if e.response.version}
            if len(versions) > 1 or any(not v.startswith(prices.SECOND.model) for v in versions):
                raise FamilyUnusable(FamilyCause.VERSION)
            if any(e.response.share is None or e.response.share < 0.5 for e in probabilities):
                raise FamilyUnusable(FamilyCause.SHARE)

    def ask_many(
        self, family: Family, requests: Sequence[Request], *, recognition: bool = False
    ) -> list[Response]:
        self.check_cap()
        transcript = self.transcripts[family]

        def request_key(request: Request) -> str:
            return key(wire(request, family))

        if not recognition:
            self.check_family(family)
        unique = {request_key(r): r for r in requests if request_key(r) not in transcript.entries}
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
                raise TerminalStop(StopReason.CAP)

            # Admission and recording use a fixed window and canonical order, independent of
            # the number of workers or the order in which requests complete.
            def call(request: Request) -> Response | Exception:
                try:
                    response = self.clients[family].ask(request)
                    validate(request, response)
                    return response
                except Exception as error:
                    if isinstance(error, FamilyUnusable) and family == "decision":
                        return StepFailure("unexpected decision-family exclusion")
                    return error

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(call, window))
            failure: Exception | None = None
            for request, result in zip(window, results, strict=True):
                if isinstance(result, Exception):
                    if isinstance(result, TerminalStop) and result.response is not None:
                        transcript.append(request, result.response)

                    def priority(error: Exception) -> int:
                        return (
                            2
                            if isinstance(error, TerminalStop)
                            else 0
                            if isinstance(error, FamilyUnusable)
                            else 1
                        )

                    if failure is None or priority(result) > priority(failure):
                        failure = result
                else:
                    transcript.append(request, result)
            self.check_cap()
            if failure:
                raise failure
            if not recognition:
                self.check_family(family)
            start += len(window)
        if not recognition:
            self.check_family(family)
        return [transcript.entries[request_key(r)].response for r in requests]
