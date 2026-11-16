"""Recognition eligibility and replies remain visible beside numerical exclusions."""

import numpy as np
import pytest

from tests.test_run import finish
from tests.test_run import runner as runner
from zoon.pilot import clients, fetch, result, run


@pytest.mark.parametrize("empty", [False, True])
def test_recognition_report_keeps_only_replies_without_flags(runner: run.Run, empty: bool) -> None:
    states = finish(runner)
    recognised = clients.obj(clients.obj(states["judge"]["payload"])["recognition"])
    flagged, missing = fetch.DILEMMAS[:2]
    clients.obj(recognised[flagged])["flagged"] = True
    if empty:
        clients.obj(recognised[missing])["reply"] = " "
    else:
        del recognised[missing]
    lines = result.render(runner, states)
    assert f"[stub recognition] flagged dilemmas = {flagged}" in lines
    assert f"[stub recognition] dilemmas with no reply = {missing}" in lines
    assert f"[stub recognition] {missing} reply and flag = no reply" in lines
    held = clients.obj(states["held-out"]["payload"])
    logs = clients.obj(held["logs"])
    first, second = "decision/zoon-evaluative", "laws-only"
    expected = run.score.compare(
        run.floats(logs[first]),
        run.floats(logs[second]),
        [
            (int(clients.number(clients.items(p)[0])), str(clients.items(p)[1]))
            for p in clients.items(held["identities"])
        ],
        np.array(held["dilemma"]),
        keep=list(range(2, 13)),
    )
    assert (
        f"[stub recognition filtered] {first} minus {second} = {result.display(expected)}" in lines
    )
    # No replies leaves no eligible target even though every stored flag is false.
    for value in recognised.values():
        clients.obj(value)["reply"] = ""
        clients.obj(value)["flagged"] = False
    lines = result.render(runner, states)
    assert all(
        "not estimable" in line for line in lines if line.startswith("[stub recognition filtered]")
    )


def test_unusable_second_family_still_prints_recognition_flags(runner: run.Run) -> None:
    states = finish(runner)
    judged = clients.obj(states["judge"]["payload"])
    judged["second"] = {"usable": False, "reason": "unusable"}
    recognised = clients.obj(judged["recognition"])
    for value in recognised.values():
        clients.obj(value).update(reply="Unknown, possibly the 1977 kidnapping", flagged=True)
    lines = result.render(runner, states)
    assert sum("1977 kidnapping" in line and '"flagged": true' in line for line in lines) == 13
    assert not any("second" in line and "mean +" in line for line in lines)


def test_partial_recognition_preserves_numerical_results(runner: run.Run) -> None:
    runner.execute("judge-check")
    original = runner.session.clients["second"]
    requests = [run.prompts.recognition(r) for r in runner.records if r["set"] == "task"]
    missing_key = sorted(map(clients.key, requests))[1]

    class Withdrawn(clients.StubClient):
        def ask(self, request: clients.Request) -> clients.Response:
            if isinstance(request, str) and clients.key(request) == missing_key:
                raise clients.FamilyUnusable(clients.FamilyCause.WITHDRAWN)
            return original.ask(request)

    runner.session.clients["second"] = Withdrawn("second")
    assert "completed" in runner.execute("judge")[0]
    payload = clients.obj(runner.state("judge")["payload"])
    assert len(clients.obj(payload["recognition"])) == 12
    assert clients.obj(payload["second"])["usable"]
    assert len(clients.items(clients.obj(payload["second"])["evaluative"])) == 13
    assert len(clients.items(clients.obj(payload["second"])["strict"])) == 13
    runner.save("calibrate", {}, status="stopped", reason="fixture stop")
    lines = runner.execute("report")
    assert sum('"reply":' in line and '"flagged":' in line for line in lines) == 12
    assert sum("reply and flag = no reply" in line for line in lines) == 1
