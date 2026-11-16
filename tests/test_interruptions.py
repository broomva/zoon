"""Terminal failures take precedence, while interrupted steps can resume."""

from dataclasses import replace

import pytest

from tests.test_clients import response
from tests.test_run import runner as runner
from zoon.pilot import clients, prompts, run


@pytest.mark.parametrize("terminal_first", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_terminal_window_failure_wins_in_both_orders(
    runner: run.Run, terminal_first: bool, changed: bool
) -> None:
    requests = sorted(
        [prompts.evaluative(r) for r in runner.records if r["set"] == "outside"], key=clients.key
    )
    terminal_key = clients.key(requests[0 if terminal_first else 1])

    class Interrupted(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if clients.key(request) == terminal_key:
                if changed:
                    raise clients.ModelChanged(replace(response(), version="changed"))
                raise clients.StopRun("decision model withdrawn; no verdict")
            raise clients.StepFailure("try again")

    runner.session.clients["decision"] = Interrupted("decision")
    assert "stopped" in runner.execute("judge-check")[0]
    assert "decision model" in str(runner.state("judge-check")["reason"])


@pytest.mark.parametrize("fault", ["transport", "authentication", "unexpected"])
def test_second_request_fault_interrupts_and_reuses_recorded_answers(
    runner: run.Run, fault: str
) -> None:
    runner.execute("judge-check")
    original = runner.session.clients["second"]
    calls: list[str] = []

    class Interrupted(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            calls.append(clients.key(request))
            if len(calls) == 2:
                if fault == "unexpected":
                    raise RuntimeError("interrupted")
                raise clients.StepFailure(
                    "ZOON_VERTEX_PROJECT" if fault == "authentication" else "bounded retries"
                )
            return original.ask(request)

    runner.session.clients["second"] = Interrupted("second")
    assert "failed" in runner.execute("judge")[0]
    recorded = set(runner.session.transcripts["second"].entries)
    calls.clear()
    runner.session.clients["second"] = original
    assert "completed" in runner.execute("judge")[0]
    assert recorded <= runner.session.transcripts["second"].entries.keys()
