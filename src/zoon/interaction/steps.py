"""Guarded simulation and recognition steps, with durable chat-only endings."""

from dataclasses import asdict
from decimal import Decimal
from typing import Any

from zoon.interaction import budget, chat, decisions, score, simulation
from zoon.interaction.guard import Guard, StateReplacedError, checkpoint, digest, encoded
from zoon.pilot import clients


class PreflightClient:
    def __init__(self, client: chat.Client) -> None:
        self.client = client
        self.ready = False
        self.succeeded = False

    def ask(self, text: str) -> chat.Response:
        if not self.ready:
            preflight = getattr(self.client, "preflight", None)
            if callable(preflight):
                preflight()
            self.ready = True
        response = self.client.ask(text)
        self.succeeded = True
        return response


class Steps:
    def __init__(
        self,
        guard: Guard,
        client: clients.Client,
        *,
        decision_cap: Decimal | None = None,
        chat_client: chat.Client | None = None,
        chat_cap: Decimal | None = None,
        groups: int = score.SIMULATED_GROUPS,
    ) -> None:
        self.guard, self.client = guard, client
        self.decision_cap = budget.DECISION_CAP if decision_cap is None else decision_cap
        self.chat_cap = budget.chat_cap() if chat_cap is None else chat_cap
        self.chat_client = chat_client
        if type(groups) is not int or not 1 <= groups <= score.SIMULATED_GROUPS:
            raise ValueError("simulation group count is outside the registered ensemble")
        if guard.mode == "live" and groups != score.SIMULATED_GROUPS:
            raise ValueError("live simulation requires the registered ensemble")
        self.groups = groups
        self.payload: dict[str, Any] = {}
        self.step = ""
        self.ended = False
        self.decision: decisions.Session | None = None
        self.session: chat.Session | None = None
        guard.register_transcripts("decision")
        guard.register_transcript("chat", self.chat_checkpoint)

    def transcript(self, *, read_only: bool = True) -> chat.Transcript:
        return chat.Transcript(
            self.guard.transcript_directory,
            chat.RunIdentity(
                self.guard.registered_commit, "stub" if self.guard.mode == "stub" else "live"
            ),
            read_only=read_only,
        )

    def chat_checkpoint(self, keys: list[str] | None) -> dict[str, Any]:
        transcript = self.transcript()
        entries = {digest(encoded(asdict(k))): asdict(e) for k, e in transcript.entries.items()}
        selected = sorted(entries) if keys is None else keys
        if selected != sorted(set(selected)) or any(k not in entries for k in selected):
            raise clients.GuardRefusal("chat checkpoint is missing or duplicates records")
        return {"keys": selected, "digest": digest(encoded([entries[k] for k in selected]))}

    def chat_record(self) -> dict[str, Any]:
        transcript = self.session.transcript if self.session else self.transcript()
        raw = transcript.read_bytes()
        endings = self.payload.get("chat_endings", [])
        cause = chat.status(transcript, self.chat_cap)
        reported = (endings[0]["cause"] if endings else None) or (cause.name if cause else None)
        if self.step == "recall":
            prior = self.guard.state("closed-loop")
            if prior:
                reported = prior["payload"]["chat"]["cause"] or reported
        return {
            "cause": reported,
            "status": "unusable" if reported else "usable",
            "endings": endings,
            "answered_decisions": sum(
                key.kind == "decision"
                and chat.points(entry.response.text, chat.GAMES[key.game].endowments[key.seat])
                is not None
                for key, entry in transcript.entries.items()
            ),
            "calls": len(transcript.entries),
            "input_tokens": sum(e.response.input_tokens for e in transcript.entries.values()),
            "output_tokens": sum(e.response.output_tokens for e in transcript.entries.values()),
            "thinking_tokens": sum(e.response.thinking_tokens for e in transcript.entries.values()),
            "spending": str(transcript.spent),
            "transcript": {
                "path": transcript.path.relative_to(self.guard.storage).as_posix(),
                "digest": digest(raw),
                "lines": raw.count(b"\n"),
            },
        }

    def save(self, status: str = "running", reason: str = "") -> dict[str, Any]:
        if self.decision:
            transcript = self.decision.transcript
            responses = [e.response for e in transcript.entries.values()]
            self.payload.update(
                calls=len(responses),
                input_tokens=sum(r.input_tokens for r in responses),
                output_tokens=sum(r.output_tokens for r in responses),
                spending=str(transcript.spent),
                versions=sorted({r.version for r in responses}),
                transcript=checkpoint(transcript),
            )
        self.payload["chat"] = self.chat_record()
        transcript = self.session.transcript if self.session else self.transcript()
        self.payload["recall"] = [asdict(reply) for reply in chat.read_recall(transcript)]
        return self.guard.save(self.step, self.payload, status=status, reason=reason)

    def open_chat(self) -> chat.Session:
        transcript = self.transcript(read_only=False)
        client = self.chat_client
        if client is None:
            client = (
                chat.StubClient() if self.guard.mode == "stub" else chat.VertexClient(chat.post)
            )
            if isinstance(client, chat.StubClient):
                # Restore the stub's stream from durable receipts in a new process.
                for key in transcript.entries:
                    if key.kind == "decision":
                        client.rng.randrange(chat.GAMES[key.game].endowments[key.seat] + 1)
        self.session = chat.Session(transcript, PreflightClient(client), self.chat_cap)
        return self.session

    def execute(
        self, step: str, repair: str | None = None, end_chat: str | None = None
    ) -> dict[str, Any]:
        if step not in ("closed-loop", "recall"):
            raise clients.GuardRefusal("unknown or unimplemented step")
        if end_chat is not None and not end_chat.strip():
            raise clients.GuardRefusal("ending chat requires a nonempty reason")
        self.step = step
        with self.guard.lock():
            self.guard.before(step, repair)
            old = self.guard.state(step)
            if end_chat is not None and not (
                old
                and (
                    (old["status"] == "interrupted" and old["reason"] == "TransportInterruption")
                    or any(e["step"] == step for e in old["payload"].get("chat_endings", []))
                )
            ):
                raise clients.GuardRefusal(
                    "flag is accepted only after an attempt of this step "
                    "that chat transport interrupted"
                )
            self.ended = False
            self.payload = old["payload"] if old else {}
            self.session = None
            self.decision = None
            try:
                self.decision = decisions.Session(
                    self.guard.transcript_directory,
                    self.client,
                    self.decision_cap,
                    self.guard.identity,
                )
                self.guard.transcripts["decision"] = self.decision.transcript
                session = self.open_chat()
                if step == "recall":
                    prior = self.guard.state("closed-loop")
                    if prior is None:
                        raise clients.GuardRefusal("completed simulation is required")
                    self.payload.setdefault(
                        "chat_endings", prior["payload"].get("chat_endings", []).copy()
                    )
                self.save()
                (self.closed_loop if step == "closed-loop" else self.recall)(session, end_chat)
                return self.save("completed")
            except StateReplacedError:
                raise
            except BaseException as error:
                self.session = None
                if isinstance(error, clients.TerminalStop) and error.reason in (
                    clients.StopReason.WITHDRAWN,
                    clients.StopReason.VERSION,
                    clients.StopReason.CAP,
                ):
                    self.payload["outcome"] = {
                        "H2": "no verdict",
                        "H3": "no verdict",
                        "reason": error.reason.name,
                    }
                    return self.save("terminal", error.reason.value)
                if isinstance(error, chat.TransportInterruption):
                    self.payload["chat_transport_failures"] = (
                        self.payload.get("chat_transport_failures", 0) + 1
                    )
                self.save("interrupted", type(error).__name__)
                if isinstance(error, decisions.StorageFault):
                    raise clients.StepFailure("could not persist decision transcript") from None
                raise

    def closed_loop(self, session: chat.Session, end_chat: str | None) -> None:
        if self.decision is None:
            raise clients.GuardRefusal("a guarded decision session is required")
        store = simulation.Store(self.guard, self.payload, self.save)
        store.verify()
        store.bands(self.decision, self.groups)
        try:
            if self.payload.get("chat_endings"):
                return
            session.check()
            store.chat(session, self.groups)
        except chat.TransportInterruption:
            self.operator_ending(end_chat)
        except chat.Unusable as error:
            if error.cause in (chat.Cause.WITHDRAWN, chat.Cause.REPLAY):
                self.add_ending(error.cause.name)
                self.save()

    def recall(self, session: chat.Session, end_chat: str | None) -> None:
        endings = self.payload["chat_endings"]
        for ending in endings:
            if ending["cause"] == "WITHDRAWN" or ending["step"] == "recall":
                self.payload["stopped_by"] = ending["cause"]
                return
        try:
            result = chat.recall(session)
        except chat.TransportInterruption:
            self.operator_ending(end_chat)
            self.payload["stopped_by"] = "OPERATOR"
            return
        if result.stopped_by == chat.Cause.WITHDRAWN:
            self.add_ending("WITHDRAWN")
            self.save()
        self.payload["stopped_by"] = result.stopped_by.name if result.stopped_by else None

    def add_ending(self, cause: str, **details: Any) -> None:
        self.payload.setdefault("chat_endings", []).append(
            {"cause": cause, "step": self.step, **details}
        )

    def operator_ending(self, reason: str | None) -> None:
        if reason is None or (
            self.session is not None
            and isinstance(self.session.client, PreflightClient)
            and self.session.client.succeeded
        ):
            raise
        self.ended = True
        self.payload["chat_transport_failures"] = self.payload.get("chat_transport_failures", 0) + 1
        details = {}
        if self.step == "recall":
            details["answered_recall"] = sum(
                key.kind == "recall" for key in self.transcript().entries
            )
        self.add_ending(
            "OPERATOR",
            chat_transport_failures=self.payload["chat_transport_failures"],
            **details,
            reason=reason,
            answered_decisions=self.chat_record()["answered_decisions"],
        )
        self.save()
        print(f"[chat ending] {encoded(self.payload['chat_endings'][-1]).decode()}")
