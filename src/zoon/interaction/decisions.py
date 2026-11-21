"""Decision transcripts with per-decision identity and deterministic budget windows.

The pilot supplies the transport, bounded retry, validation, estimates and
transcript reader. These records wrap a rendered request in its decision identity:
calibration calls each selected decision, including identical rendered requests.
Later callers can deduplicate by using the canonical request hash as identity.
"""

import importlib
import json
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Any

from zoon.interaction.guard import digest
from zoon.pilot import clients, prices, prompts

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
                clients.entry_document(entry), sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    transcript.entries[key] = entry


class LiveClient:
    """The pilot client, with SDK observations for paid retries and admission.

    Live requests are serial so every SDK attempt can check the current ledger
    before transport. The SDK and credentials are left to the pilot client.
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

        class Transport:
            def __init__(self, transport: Any) -> None:
                self.transport = transport

            def system_one(self, state: str, questions: dict[str, object]) -> Any:
                admit()
                result = self.transport.system_one(state, questions)
                version = getattr(result, "model", None)
                usage = getattr(result, "usage", None)
                incoming = getattr(usage, "input_tokens", None)
                outgoing = getattr(usage, "output_tokens", 0)
                if (
                    not isinstance(version, str)
                    or not version
                    or type(incoming) is not int
                    or incoming < 0
                    or type(outgoing) is not int
                    or outgoing < 0
                ):
                    raise clients.TransientError
                # Let the pilot check connection metadata before retaining any
                # unexpected version string. Its TerminalStop carries the paid reply.
                if version != prices.DECISION.model:
                    return result
                returned: dict[str, float] = {}
                with suppress(AttributeError, KeyError, TypeError, ValueError):
                    returned = {
                        str(k): float(v) for k, v in result.answers["q"].probabilities.items()
                    }
                if not set(returned) <= {a.answer for a in request.answers}:
                    returned = {}
                finite = all(math.isfinite(v) and 0 <= v <= 1 for v in returned.values())
                total = sum(returned.values())
                valid = (
                    finite
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
                received(response, valid or version != prices.DECISION.model)
                return result

        class SDK:
            def Choice(self, *, instructions: str, criteria: dict[str, str]) -> Any:
                return sdk.Choice(instructions=instructions, criteria=criteria)

            def TypeSafeClient(self, *, model: str, timeout: float) -> Transport:
                return Transport(sdk.TypeSafeClient(model=model, timeout=timeout))

        return clients.DecisionClient(SDK()).ask(request)


class Session:
    def __init__(
        self,
        directory: Path,
        client: clients.Client,
        workers: int,
        cap: Decimal,
        identity: dict[str, str],
    ) -> None:
        if workers < 1 or client.family != "decision":
            raise ValueError("a decision client and positive worker count are required")
        self.transcript = clients.Transcript(directory, "decision", identity)
        self.client, self.workers, self.cap = client, workers, cap

    def check(self) -> None:
        if self.transcript.spent > self.cap:
            raise clients.TerminalStop(clients.StopReason.CAP)
        if any(
            e.response.version != prices.DECISION.model for e in self.transcript.entries.values()
        ):
            raise clients.TerminalStop(clients.StopReason.VERSION)

    def ask_many(self, requests: Sequence[Request]) -> list[clients.Response]:
        self.check()
        unique = {request_key(r): r for r in requests}
        pending = [unique[k] for k in sorted(unique) if k not in self.transcript.entries]
        start = 0
        while start < len(pending):
            window: list[Request] = []
            reserved = Decimal(0)
            for request in pending[start : start + clients.WINDOW]:
                estimate = clients.estimate("decision", request[1])
                if self.transcript.spent + reserved + estimate > self.cap:
                    break
                window.append(request)
                reserved += estimate
            if not window:
                raise clients.TerminalStop(clients.StopReason.CAP)

            def call(request: Request) -> clients.Response | Exception:
                try:
                    if isinstance(self.client, LiveClient):

                        def admit() -> None:
                            self.check()
                            if (
                                self.transcript.spent + clients.estimate("decision", request[1])
                                > self.cap
                            ):
                                raise clients.TerminalStop(clients.StopReason.CAP)

                        def received(response: clients.Response, accepted: bool) -> None:
                            identifier, rendered = request
                            attempt = 1 + sum(
                                str(clients.obj(e.request).get("decision", "")).startswith(
                                    identifier + ":failed:"
                                )
                                for e in self.transcript.entries.values()
                            )
                            recorded = (
                                request
                                if accepted
                                else (identifier + f":failed:{attempt}", rendered)
                            )
                            append(self.transcript, recorded, response)

                        response = self.client.recorded(request[1], admit, received)
                    else:
                        response = self.client.ask(request[1])
                    if response.version != prices.DECISION.model:
                        raise clients.TerminalStop(clients.StopReason.VERSION, response=response)
                    clients.validate(request[1], response)
                    return response
                except Exception as error:
                    return error

            if isinstance(self.client, LiveClient):
                results = [call(r) for r in window]
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    results = list(pool.map(call, window))
            failure: Exception | None = None
            for request, result in zip(window, results, strict=True):
                if isinstance(result, Exception):
                    if isinstance(result, clients.TerminalStop) and result.response is not None:
                        append(self.transcript, request, result.response)
                    if failure is None or isinstance(result, clients.TerminalStop):
                        failure = result
                else:
                    append(self.transcript, request, result)
            self.check()
            if failure is not None:
                raise failure
            start += len(window)
        return [self.transcript.entries[request_key(r)].response for r in requests]
