"""Guarded simulation and recognition steps, with durable chat-only endings."""

from dataclasses import asdict
from decimal import Decimal
from typing import Any

from zoon.interaction import budget, chat, decisions, score, simulation
from zoon.interaction.guard import Guard, checkpoint, digest, encoded
from zoon.pilot import clients


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
        ending = self.payload.get("chat_ending", {})
        cause = chat.status(transcript, self.chat_cap)
        return {
            "cause": ending.get("cause") or (cause.name if cause else None),
            "status": "unusable" if ending or cause else "usable",
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
        self.session = chat.Session(transcript, client, self.chat_cap)
        return self.session

    def preflight(self) -> None:
        if self.session is None:
            raise clients.GuardRefusal("a guarded chat session is required")
        preflight = getattr(self.session.client, "preflight", None)
        if callable(preflight):
            preflight()

    def execute(self, step: str, repair: str | None = None) -> dict[str, Any]:
        if step not in ("closed-loop", "recall"):
            raise clients.GuardRefusal("unknown or unimplemented step")
        self.step = step
        with self.guard.lock():
            self.guard.before(step, repair)
            old = self.guard.state(step)
            self.payload = old["payload"] if old else {}
            self.decision = decisions.Session(
                self.guard.transcript_directory, self.client, self.decision_cap, self.guard.identity
            )
            self.guard.transcripts["decision"] = self.decision.transcript
            session = self.open_chat()
            self.save()
            try:
                if step == "closed-loop":
                    self.closed_loop(session)
                else:
                    self.recall(session)
                self.payload.pop("last_failure", None)
                return self.save("completed")
            except BaseException as error:
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
                self.payload["last_failure"] = {
                    "kind": "chat_transport"
                    if isinstance(error, chat.TransportInterruption)
                    else "other",
                    "type": type(error).__name__,
                }
                self.save("interrupted", type(error).__name__)
                if isinstance(error, decisions.StorageFault):
                    raise clients.StepFailure("could not persist decision transcript") from None
                raise

    def closed_loop(self, session: chat.Session) -> None:
        if self.decision is None:
            raise clients.GuardRefusal("a guarded decision session is required")
        store = simulation.Store(self.guard, self.payload, self.save)
        store.verify()
        store.bands(self.decision, self.groups)
        try:
            ending = self.payload.get("chat_ending")
            if ending:
                return
            session.check()
            self.preflight()
            store.chat(session, self.groups)
        except chat.Unusable as error:
            if error.cause in (chat.Cause.WITHDRAWN, chat.Cause.REPLAY):
                self.payload["chat_ending"] = {"cause": error.cause.name, "step": self.step}
                self.save()
            # Partial chat games remain unscored. They are never promoted to digests.
            self.payload["chat_pending"] = {}

    def recall(self, session: chat.Session) -> None:
        prior = self.guard.state("closed-loop")
        if prior is None:
            raise clients.GuardRefusal("completed simulation is required")
        ending = self.payload.get("chat_ending") or prior["payload"].get("chat_ending")
        if ending:
            self.payload["chat_ending"] = ending
        if ending and (ending["cause"] == "WITHDRAWN" or ending["step"] == "recall"):
            self.payload["stopped_by"] = ending["cause"]
            return
        cause = chat.status(session.transcript, session.cap, recall=True)
        if cause:
            self.payload["stopped_by"] = cause.name
            return
        try:
            self.preflight()
            result = chat.recall(session)
            if result.stopped_by == chat.Cause.WITHDRAWN:
                self.payload["chat_ending"] = {"cause": "WITHDRAWN", "step": self.step}
            self.payload["stopped_by"] = result.stopped_by.name if result.stopped_by else None
        except chat.Unusable as error:
            # Withdrawal cannot be represented by a paid transcript entry.
            self.payload["chat_ending"] = {"cause": error.cause.name, "step": self.step}
            self.payload["stopped_by"] = error.cause.name
            self.save()

    def end_chat(self, reason: str, repair: str | None = None) -> dict[str, Any]:
        if not reason.strip():
            raise clients.GuardRefusal("ending chat requires a nonempty reason")
        with self.guard.lock():
            if (self.guard.directory / "held-out.json").exists():
                raise clients.GuardRefusal("chat cannot be ended once held-out has started")
            self.step = (
                "recall" if (self.guard.directory / "recall.json").exists() else "closed-loop"
            )
            self.guard.before(self.step, repair)
            old = self.guard.state(self.step)
            if not old or old["status"] != "interrupted":
                raise clients.GuardRefusal("chat can be ended only while its step is interrupted")
            self.payload = old["payload"]
            if self.payload.get("last_failure", {}).get("kind") != "chat_transport":
                raise clients.GuardRefusal("the last failure was not a chat transport failure")
            if self.payload.get("chat_ending"):
                raise clients.GuardRefusal("an ending of the chat arm is already recorded")
            self.payload["chat_ending"] = {
                "cause": "OPERATOR",
                "reason": reason,
                "step": self.step,
                "answered_decisions": self.chat_record()["answered_decisions"],
            }
            return self.save("interrupted", old["reason"])
