"""Decision transcripts with per-decision identity and serial budget admission.

The pilot supplies bounded retry, response validation, estimates and the
transcript reader. These records wrap a rendered request in its decision identity:
calibration calls each selected decision, including identical rendered requests.
Later callers can deduplicate by using the canonical request hash as identity.
"""

import importlib
import json
import math
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

from zoon.interaction.guard import digest
from zoon.pilot import clients, prices, prompts


class StorageFault(BaseException):
    """A transcript write failed; transport retry must never catch this fault."""


Request = tuple[str, prompts.Decision]


def document(request: Request) -> dict[str, object]:
    identifier, rendered = request
    return {"decision": identifier, "request": json.loads(clients.canonical(rendered))}


def request_key(request: Request) -> str:
    return digest(
        json.dumps(
            document(request), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    )


def append(transcript: clients.Transcript, request: Request, response: clients.Response) -> None:
    try:
        key = request_key(request)
        if key in transcript.entries:
            return
        entry = clients.Entry(key, document(request), response)
        transcript.path.parent.mkdir(parents=True, exist_ok=True)
        new = not transcript.path.exists() or transcript.path.stat().st_size == 0
        with transcript.path.open("a") as handle:
            if new:
                handle.write(json.dumps(transcript.identity, sort_keys=True) + "\n")
            handle.write(
                json.dumps(
                    clients.entry_document(entry),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        transcript.entries[key] = entry
    except OSError:
        raise StorageFault from None


class LiveClient:
    """The pilot client, with SDK observations for paid retries and admission.

    Live requests are serial so every SDK attempt can check the current ledger
    before transport. Credentials are left to the SDK.
    Injected SDKs exercise this path without a service or credential.
    """

    family: clients.Family = "decision"

    def __init__(self, sdk: Any = None) -> None:
        self.sdk = sdk

    def ask(self, request: clients.Request) -> clients.Response:
        return clients.DecisionClient(self.sdk).ask(request)

    def recorded(
        self,
        request: prompts.Decision,
        admit: Callable[[], None],
        received: Callable[[clients.Response, bool], None],
    ) -> clients.Response:
        sdk: Any = self.sdk or importlib.import_module("typesafe_sdk")

        def call() -> clients.Response:
            try:
                admit()
                transport = sdk.TypeSafeClient(model=prices.DECISION.model, timeout=30.0)
                result = transport.system_one(
                    request.state,
                    {
                        "q": sdk.Choice(
                            instructions=request.question,
                            criteria={a.answer: a.description for a in request.answers},
                        )
                    },
                )
                version = getattr(result, "model", None)
                usage = getattr(result, "usage", None)
                incoming: Any = getattr(usage, "input_tokens", None)
                outgoing: Any = getattr(usage, "output_tokens", 0)
                valid_incoming = type(incoming) is int and incoming >= 0
                valid_outgoing = type(outgoing) is int and outgoing >= 0
                secret = os.environ.get("TYPESAFE_API_KEY", "")
                if secret and secret in repr((version, getattr(result, "answers", None), usage)):
                    received(
                        clients.Response(
                            {},
                            None,
                            "",
                            version if isinstance(version, str) and secret not in version else "",
                            incoming if valid_incoming else 0,
                            outgoing if valid_outgoing else 0,
                        ),
                        False,
                    )
                    raise clients.StepFailure(
                        "connection metadata in response; reply content was not recorded"
                    )
                if isinstance(version, str) and version and version != prices.DECISION.model:
                    received(
                        clients.Response(
                            {},
                            None,
                            "",
                            version,
                            incoming if valid_incoming else 0,
                            outgoing if valid_outgoing else 0,
                        ),
                        True,
                    )
                    raise clients.TerminalStop(clients.StopReason.VERSION)
                if (
                    not version
                    or not isinstance(version, str)
                    or not valid_incoming
                    or not valid_outgoing
                ):
                    raise clients.TransientError
                returned: dict[str, float] = {}
                unique = False
                with suppress(AttributeError, KeyError, TypeError, ValueError):
                    raw = result.answers["q"].probabilities
                    returned = {str(k): float(v) for k, v in raw.items()}
                    unique = len(returned) == len(raw)
                if not set(returned) <= {a.answer for a in request.answers}:
                    returned = {}
                finite = all(math.isfinite(v) and 0 <= v <= 1 for v in returned.values())
                total = sum(returned.values())
                valid = (
                    unique
                    and finite
                    and set(returned) == {a.answer for a in request.answers}
                    and total > 0
                    and abs(total - 1) <= 0.05 + 1e-12
                )
                response = clients.Response(
                    {k: v / total for k, v in returned.items()} if valid else {},
                    total if finite else None,
                    "",
                    version,
                    incoming,
                    outgoing,
                    returned if finite else None,
                )
                received(response, valid)
                if not valid:
                    raise clients.TransientError
                return response
            except (
                clients.StepFailure,
                clients.TerminalStop,
                clients.TransientError,
                TimeoutError,
            ):
                raise
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    raise clients.TransientError from None
                if "timeout" in type(error).__name__.lower():
                    raise TimeoutError from None
                if status in (404, 410):
                    raise clients.ModelMissing("decision") from None
                if status in (401, 403):
                    raise clients.StepFailure("decision authentication failed") from None
                raise clients.TransientError from None

        return clients.retry(call)


class Session:
    """Dispatch serially so each recorded reply can stop all later requests."""

    def __init__(
        self,
        directory: Path,
        client: clients.Client,
        cap: Decimal,
        identity: dict[str, str],
    ) -> None:
        if client.family != "decision":
            raise ValueError("a decision client is required")
        self.transcript = clients.Transcript(directory, "decision", identity)
        self.client, self.cap = client, cap

    def check(self) -> None:
        # Replay in receipt order: the first terminal reply governs, including a
        # version reply whose cost would also cross the cap.
        spent = Decimal(0)
        for entry in self.transcript.entries.values():
            if entry.response.version != prices.DECISION.model:
                raise clients.TerminalStop(clients.StopReason.VERSION)
            spent += clients.cost("decision", entry.response)
            if spent > self.cap:
                raise clients.TerminalStop(clients.StopReason.CAP)

    def ask_many(self, requests: Sequence[Request]) -> list[clients.Response]:
        self.check()
        unique = {request_key(r): r for r in requests}
        pending = [unique[k] for k in sorted(unique) if k not in self.transcript.entries]
        for request in pending:

            def admit(current: Request = request) -> None:
                self.check()
                if self.transcript.spent + clients.estimate("decision", current[1]) > self.cap:
                    raise clients.TerminalStop(clients.StopReason.CAP)

            def received(
                response: clients.Response, accepted: bool, current: Request = request
            ) -> None:
                identifier, rendered = current
                attempt = 1 + sum(
                    str(clients.obj(e.request).get("decision", "")).startswith(
                        identifier + ":failed:"
                    )
                    for e in self.transcript.entries.values()
                )
                recorded = current if accepted else (identifier + f":failed:{attempt}", rendered)
                append(self.transcript, recorded, response)
                self.check()

            admit()
            try:
                response = (
                    self.client.recorded(request[1], admit, received)
                    if isinstance(self.client, LiveClient)
                    else self.client.ask(request[1])
                )
            except clients.TerminalStop as error:
                if error.response is not None:
                    append(self.transcript, request, error.response)
                raise
            if response.version != prices.DECISION.model:
                append(self.transcript, request, response)
                raise clients.TerminalStop(clients.StopReason.VERSION, response=response)
            clients.validate(request[1], response)
            append(self.transcript, request, response)
            self.check()
        return [self.transcript.entries[request_key(r)].response for r in requests]
