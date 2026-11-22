"""Commands select live by default and exercise guarded stub calibration."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.interaction_run_support import commit, repository
from tests.test_interaction_decision_faults import SDK, reply
from zoon.interaction import budget, decisions, run
from zoon.pilot import clients


def test_run_default_builds_live_client(tmp_path: Path) -> None:
    runner = run.Run(repository(tmp_path / "repo"))
    assert type(runner.client) is decisions.LiveClient
    assert runner.guard.mode == "live"


def test_cli_default_runs_live_and_passes_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = repository(tmp_path / "repo")
    (root / "src/code.py").write_text("value = 2\n")
    commit(root)
    sdk = SDK(lambda n: reply(version="another-version"))
    original = run.Run

    class LocalRun(original):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(root, **kwargs)
            assert type(self.client) is decisions.LiveClient
            assert self.guard.mode == "live"
            self.client = decisions.LiveClient(sdk)

    monkeypatch.delenv("ZOON_INTERACTION_DATA", raising=False)
    monkeypatch.setattr(run, "Run", LocalRun)
    assert run.main(["calibrate", "--repair", "Resolve execution fault"]) == 0
    output = capsys.readouterr()
    assert "[interaction live] calibrate = terminal" in output.out
    assert (
        "[outcome] H2 = no verdict | H3 = no verdict | reason = decision model version changed"
        in output.out
    )
    assert output.err == "" and sdk.calls == 1
    binding = json.loads((root / "data/interaction/run/live/binding.json").read_bytes())
    assert binding["repairs"][0]["reason"] == "Resolve execution fault"


@pytest.mark.parametrize("terminal", [False, True])
def test_stub_command_uses_storage_and_prints_completed_or_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminal: bool,
) -> None:
    root = repository(tmp_path / "repo")
    storage = tmp_path / "rehearsal"
    monkeypatch.setenv("ZOON_INTERACTION_DATA", str(storage))
    original = run.Run

    class LocalRun(original):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(root, **kwargs)

    monkeypatch.setattr(run, "Run", LocalRun)

    def no_bound(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("commands must use the registered cap constant")

    monkeypatch.setattr(budget, "estimate", no_bound)
    if terminal:
        monkeypatch.setattr(budget, "DECISION_CAP", 0)
    assert run.main(["calibrate", "--stub"]) == 0
    output = capsys.readouterr()
    status = "terminal" if terminal else "completed"
    assert output.err == ""
    assert f"[interaction stub] calibrate = {status}" in output.out
    assert "[decision] calls = " in output.out
    if terminal:
        assert (
            "[outcome] H2 = no verdict | H3 = no verdict | reason = "
            "decision model spending cap reached" in output.out
        )
    else:
        assert "[recalibration] numbers = " in output.out and "[disguise] " in output.out
    assert json.loads((storage / "run/stub/calibrate.json").read_bytes())["status"] == status
    assert not (root / "data/interaction").exists()


@pytest.mark.parametrize("fault", ["interrupt", "unmapped"])
def test_step_interruptions_are_saved_before_propagating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root = repository(tmp_path / "repo")

    def failed(*args: Any) -> Any:
        if fault == "interrupt":
            raise KeyboardInterrupt
        raise clients.TerminalStop(clients.StopReason.CALIBRATION)

    monkeypatch.setattr(run.Run, "calibrate", failed)
    runner = run.Run(root, stub=True)
    with pytest.raises(KeyboardInterrupt if fault == "interrupt" else clients.TerminalStop):
        runner.execute()
    state = json.loads((runner.guard.directory / "calibrate.json").read_bytes())
    assert state["status"] == "interrupted"
    assert "outcome" not in state["payload"]
