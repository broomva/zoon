"""Guarded interaction steps. Repeating an interrupted command retains its answers."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from zoon.interaction import budget, data, decisions, measure, prompted, score
from zoon.interaction.guard import Guard, StateReplacedError, checkpoint, digest, encoded
from zoon.pilot import clients


class Outcome(Enum):
    WITHDRAWN = "decision model withdrawn"
    VERSION = "decision model version changed"
    CAP = "decision model spending cap reached"
    FIT = "prompted.fit raised on the recorded recalibration answers"
    RECORDS = "held-out groups are unreadable or differ from the pinned split"


class Run:
    def __init__(
        self,
        root: Path = data.REPO,
        *,
        stub: bool = False,
        storage: Path | None = None,
        client: clients.Client | None = None,
        cap: Decimal | None = None,
    ) -> None:
        self.guard = Guard(root, stub=stub, storage=storage)
        self.guard.register_transcripts("decision")
        self.client = client or (clients.StubClient("decision") if stub else decisions.LiveClient())
        self.cap = cap
        self.session: decisions.Session | None = None
        self.payload: dict[str, Any] = {}

    def totals(self) -> dict[str, Any]:
        if self.session is None:
            return {}
        transcript = self.session.transcript
        responses = [e.response for e in transcript.entries.values()]
        return {
            "calls": len(responses),
            "input_tokens": sum(r.input_tokens for r in responses),
            "output_tokens": sum(r.output_tokens for r in responses),
            "spending": str(transcript.spent),
            "versions": sorted({r.version for r in responses}),
            "transcript": checkpoint(transcript),
        }

    def terminal(self, outcome: Outcome) -> dict[str, Any]:
        self.payload["outcome"] = {"H2": "no verdict", "H3": "no verdict", "reason": outcome.name}
        return self.guard.save(
            "calibrate", {**self.payload, **self.totals()}, status="terminal", reason=outcome.value
        )

    def execute(self, step: str = "calibrate", *, repair: str | None = None) -> dict[str, Any]:
        if step != "calibrate":
            raise clients.GuardRefusal("unknown or unimplemented step")
        with self.guard.lock():
            self.guard.before(step, repair)
            old = self.guard.state(step)
            self.payload = old["payload"] if old else {}
            groups = data.load(
                self.guard.root / "data" / "groups" / "training.json",
                manifest=self.guard.root / "interaction" / "data-manifest.json",
            )
            cap = budget.DECISION_CAP
            self.session = decisions.Session(
                self.guard.transcript_directory,
                self.client,
                cap if self.cap is None else self.cap,
                self.guard.identity,
            )
            self.guard.transcripts = {"decision": self.session.transcript}
            self.guard.save(step, {**self.payload, **self.totals()}, status="running")
            try:
                return self.calibrate(groups)
            except clients.TerminalStop as error:
                outcome = {
                    clients.StopReason.WITHDRAWN: Outcome.WITHDRAWN,
                    clients.StopReason.VERSION: Outcome.VERSION,
                    clients.StopReason.CAP: Outcome.CAP,
                }.get(error.reason)
                if outcome is None:
                    self.guard.save(
                        step,
                        {**self.payload, **self.totals()},
                        status="interrupted",
                        reason=type(error).__name__,
                    )
                    raise
                return self.terminal(outcome)
            except decisions.StorageFault:
                self.guard.save(
                    step,
                    {**self.payload, **self.totals()},
                    status="interrupted",
                    reason="StorageFault",
                )
                raise clients.StepFailure(
                    "could not persist decision transcript; restore writable storage and resume"
                ) from None
            except data.RecordFault:
                return self.terminal(Outcome.RECORDS)
            except StateReplacedError:
                raise
            except BaseException as error:
                self.guard.save(
                    step,
                    {**self.payload, **self.totals()},
                    status="interrupted",
                    reason=type(error).__name__,
                )
                raise

    def calibrate(self, groups: Sequence[data.Group]) -> dict[str, Any]:
        if self.session is None:
            raise clients.GuardRefusal("a guarded session is required")
        sample = prompted.subsample(groups)
        if len(sample) != prompted.SUBSAMPLE_COUNT:
            raise clients.StepFailure("the recalibration sample must contain exactly 512 decisions")
        requests = [
            (
                "original:" + d.digest,
                prompted.request(prompted.training_view(d.group, d.seat, d.round)),
            )
            for d in sample
        ]
        request_digest = digest(encoded([decisions.request_key(r) for r in requests]))
        if "recalibration" in self.payload and self.payload.get("fit_requests") != request_digest:
            raise clients.GuardRefusal("saved recalibration was fitted to different requests")
        original = self.session.ask_many(requests)
        pairs = [
            (
                np.array([r.probabilities[a] for a in prompted.ANSWERS]),
                measure.band(
                    d.group.rounds[d.round - 1][d.seat],
                    prompted.GAMES[d.group.cell].endowments[d.seat],
                ),
            )
            for d, r in zip(sample, original, strict=True)
        ]
        self.payload["answers"] = checkpoint(
            self.session.transcript, sorted({decisions.request_key(r) for r in requests})
        )
        if "recalibration" not in self.payload:
            try:
                recalibration = prompted.fit(pairs)
            except Exception:
                return self.terminal(Outcome.FIT)
            self.payload["recalibration"] = recalibration.numbers.tolist()
            self.payload["fit_requests"] = request_digest
            self.guard.save("calibrate", {**self.payload, **self.totals()}, status="running")
        values = self.payload["recalibration"]
        recalibration = prompted.Recalibration(values[0], tuple(values[1:]))
        disguised_requests = [
            (
                "disguise:" + d.digest,
                prompted.request(
                    prompted.training_view(d.group, d.seat, d.round), prompted.DISGUISE
                ),
            )
            for d in sample
        ]
        disguised = self.session.ask_many(disguised_requests)
        self.payload["disguise_answers"] = checkpoint(
            self.session.transcript,
            sorted({decisions.request_key(r) for r in disguised_requests}),
        )
        plain = recalibration.apply(np.array([p for p, _ in pairs]))
        changed = recalibration.apply(
            np.array([[r.probabilities[a] for a in prompted.ANSWERS] for r in disguised])
        )
        bands = np.array([b for _, b in pairs])
        log_change = score.log_scores(changed, bands) - score.log_scores(plain, bands)
        plain = np.maximum(plain, score.SCORING_FLOOR)
        plain /= plain.sum(axis=1, keepdims=True)
        changed = np.maximum(changed, score.SCORING_FLOOR)
        changed /= changed.sum(axis=1, keepdims=True)
        metrics = {
            "mean_log_probability_change": float(np.mean(log_change)),
            "mean_total_variation": float(np.mean(np.sum(0.5 * np.abs(changed - plain), axis=1))),
        }
        if not all(np.isfinite(v) for v in metrics.values()):
            raise clients.StepFailure("non-finite disguise summary")
        self.payload["disguise"] = metrics
        return self.guard.save("calibrate", {**self.payload, **self.totals()}, status="completed")


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise clients.GuardRefusal(message)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = Parser(description=__doc__)
    parser.add_argument("step", choices=("calibrate",))
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--repair", help="reason for the committed repair snapshot")
    try:
        args = parser.parse_args(arguments)
        storage = os.environ.get("ZOON_INTERACTION_DATA")
        state = Run(stub=args.stub, storage=Path(storage) if storage else None).execute(
            args.step, repair=args.repair
        )
        payload = state["payload"]
        print(f"[interaction {state['mode']}] {args.step} = {state['status']}")
        print(
            f"[decision] calls = {payload['calls']} | input tokens = {payload['input_tokens']} | "
            f"output tokens = {payload['output_tokens']} | USD = {payload['spending']}"
        )
        if state["status"] == "terminal":
            print(f"[outcome] H2 = no verdict | H3 = no verdict | reason = {state['reason']}")
        else:
            print(f"[recalibration] numbers = {payload['recalibration']}")
            print(f"[disguise] {json.dumps(payload['disguise'], sort_keys=True)}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        # No traceback, exception chain, frame, request or connection metadata.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
