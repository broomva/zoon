"""Every live step compares disk bytes with the same registered commit."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.tracked import copy_tracked
from zoon.pilot import clients, fetch, run


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(root: Path) -> str:
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Change",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> run.Run:
    copy_tracked(tmp_path)
    git(tmp_path, "init", "-q")
    commit(tmp_path)
    git(tmp_path, "tag", "registered")
    runner = run.Run(
        tmp_path, supplied_clients=[clients.StubClient(f) for f in ("decision", "second")]
    )
    monkeypatch.setattr(runner, "verify_materials", lambda: None)
    return runner


def restart(live: run.Run, monkeypatch: pytest.MonkeyPatch) -> run.Run:
    fresh = run.Run(live.root, supplied_clients=[])
    monkeypatch.setattr(fresh, "verify_materials", lambda: None)
    return fresh


def change(live: run.Run, name: str, *, hidden: bool = False) -> None:
    path = live.root / name
    if hidden:
        git(live.root, "update-index", "--skip-worktree", name)
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n# changed\n")
    if not hidden:
        commit(live.root)


@pytest.mark.parametrize("step", run.STEPS)
@pytest.mark.parametrize(
    "fault,pattern",
    [
        ("no-tag", "tag"),
        ("wrong-tag", "tag"),
        ("unrelated", "ancestor"),
        ("registered", "registered files"),
        ("hidden-registered", "clean work tree"),
        ("table", "registered files"),
        ("code", "repair"),
        ("init", "repair"),
        ("added", "repair"),
        ("deleted", "repair"),
        ("hidden-code", "repair|clean work tree"),
        ("dirty", "clean work tree"),
    ],
)
def test_every_live_step_refuses_before_requests_or_loader(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, step: str, fault: str, pattern: str
) -> None:
    tagged = git(live.root, "rev-parse", "HEAD")
    if fault == "no-tag":
        git(live.root, "tag", "-d", "registered")
    elif fault == "wrong-tag":
        change(live, "src/zoon/pilot/score.py")
        git(live.root, "tag", "-f", "registered")
    elif fault == "unrelated":
        git(live.root, "checkout", "--orphan", "other")
        (live.root / "other.txt").write_text("unrelated history")
        commit(live.root)
    elif fault == "dirty":
        (live.root / "src/zoon/pilot/score.py").write_text("changed")
    else:
        names = {
            "registered": "prereg/h1.md",
            "hidden-registered": "pilot/task-texts.json",
            "table": "src/zoon/pilot/korner_deutsch_2022.csv",
            "code": "src/zoon/pilot/score.py",
            "init": "src/zoon/__init__.py",
            "added": "src/zoon/added.py",
            "deleted": "src/zoon/pilot/score.py",
            "hidden-code": "src/zoon/pilot/score.py",
        }
        if fault == "deleted":
            (live.root / names[fault]).unlink()
            commit(live.root)
        else:
            change(live, names[fault], hidden=fault.startswith("hidden"))
    reached: list[str] = []

    def loader(directory: Path) -> None:
        reached.append("loader")

    def request(*args: object, **kwargs: object) -> None:
        reached.append("request")

    monkeypatch.setattr(fetch, "load_registered_held_out", loader)
    monkeypatch.setattr(live, "request", request)
    # A mismatching explicit commit matters only to held-out; other steps need a missing tag.
    if fault == "wrong-tag" and step != "held-out":
        git(live.root, "tag", "-d", "registered")
    with pytest.raises(clients.GuardRefusal, match=pattern):
        live.execute(step, tagged, repair="Restore operation" if "registered" in fault else None)
    assert not reached
    assert not any(t.entries for t in live.session.transcripts.values())


def reach(live: run.Run, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reached: list[str] = []

    def stopped() -> run.State:
        reached.append("work")
        return live.save("judge-check", {}, status="stopped", reason="fixture stop")

    monkeypatch.setattr(live, "_judge_check", stopped)
    return reached


def test_equal_tag_and_matching_repair_snapshot_go_on(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached = reach(live, monkeypatch)
    live.execute("judge-check")
    assert reached == ["work"]
    change(live, "src/zoon/pilot/score.py")
    live.guard("report", None, "Restore scoring")
    assert live.pending_binding is not None
    live.binding(live.pending_binding)
    live.pending_binding = None
    live.execute("judge-check", repair="Restore scoring")
    live.execute("judge-check")
    restarted = run.Run(live.root, supplied_clients=[])
    monkeypatch.setattr(restarted, "verify_materials", lambda: None)
    assert "stopped" in restarted.execute("judge-check")[0]
    repair = clients.obj(clients.items(live.binding()["repairs"])[0])
    assert repair["files"] == ["src/zoon/pilot/score.py"]
    assert repair["code"] == {
        "src/zoon/pilot/score.py": run.digest((live.root / "src/zoon/pilot/score.py").read_bytes())
    }


def test_two_commits_require_the_full_difference_and_report_it(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    reach(live, monkeypatch)
    first = "src/zoon/pilot/score.py"
    second = "src/zoon/pilot/clients.py"
    change(live, first)
    live = restart(live, monkeypatch)
    reach(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match=r"score\.py"):
        live.execute("judge-check")
    change(live, second)
    live = restart(live, monkeypatch)
    reach(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match=r"score\.py"):
        live.execute("judge-check")
    live.execute("judge-check", repair="Restore scoring and requests")
    repair = clients.obj(clients.items(live.binding()["repairs"])[0])
    assert repair["files"] == sorted([first, second])
    assert repair["commit"] == git(live.root, "rev-parse", "HEAD")
    assert set(clients.obj(repair["code"])) == {first, second}
    (live.root / "note.txt").write_text("Reporting note")
    commit(live.root)
    lines = live.execute("report", repair="Restore reporting")
    assert any("difference from registered" in line and "note.txt" in line for line in lines)
    assert any(
        "Repairs" in line
        and "difference from registered" in line
        and first in line
        and second in line
        for line in lines
    )
    # The report computes its list again instead of trusting stored repair file names.
    binding = live.binding()
    clients.obj(clients.items(binding["repairs"])[0])["files"] = []
    (live.directory / "binding.json").write_bytes(run.encoded(binding))
    assert any(
        "difference from registered" in line and first in line and second in line
        for line in run_result(live)
    )


def run_result(live: run.Run) -> list[str]:
    from zoon.pilot.result import render

    return render(live, {"judge-check": live.state("judge-check")})


@pytest.mark.parametrize("changed", ["src/zoon/pilot/clients.py", "src/zoon/pilot/score.py"])
def test_previous_repair_does_not_cover_later_code(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    reach(live, monkeypatch)
    change(live, "src/zoon/pilot/score.py")
    live.execute("judge-check", repair="Restore scoring")
    change(live, changed)
    live = restart(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match="repair"):
        live.execute("report")


def test_report_without_repairs_says_none(live: run.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    reach(live, monkeypatch)
    live.execute("judge-check")
    assert "[Repairs] history = none" in live.execute("report")
    saved = json.loads((live.directory / "judge-check.json").read_text())
    assert saved["registered_commit"] == live.registered_commit
    assert saved["code_snapshot"] == {}


@pytest.mark.parametrize("step", ["judge", "calibrate", "held-out"])
@pytest.mark.parametrize("entry", ["guard", "command"])
def test_one_guard_checks_prerequisites_before_materials(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, step: str, entry: str
) -> None:
    earlier = run.STEPS[: run.STEPS.index(step)]
    for previous in earlier:
        live.save(previous, {}, status="stopped" if previous == earlier[-1] else "completed")

    def forbidden() -> None:
        raise AssertionError("materials reached before the prerequisite gate")

    monkeypatch.setattr(live, "verify_materials", forbidden)
    registered = git(live.root, "rev-parse", "HEAD")
    with pytest.raises(clients.GuardRefusal, match="did not pass"):
        if entry == "guard":
            live.guard(step, registered, None)
        else:
            live.execute(step, registered)


@pytest.mark.parametrize("name", ["pyproject.toml", "uv.lock"])
def test_deleted_code_inputs_can_be_declared_as_repairs(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    (live.root / name).unlink()
    commit(live.root)
    live = restart(live, monkeypatch)
    reach(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match="repair"):
        live.execute("judge-check")
    live.execute("judge-check", repair="Restore the execution environment")
    repair = clients.obj(clients.items(live.binding()["repairs"])[0])
    assert repair["code"] == {name: "deleted"}


def test_malformed_registered_material_is_refused_by_the_guard_before_parsing(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    next((live.root / "pilot" / "rewrites").glob("*.json")).write_text("{")
    commit(live.root)
    live = restart(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match="registered files"):
        live.execute("judge-check", repair="Restore operation")


def test_new_registration_has_new_state_and_transcript_identity(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = live.directory
    live.save("judge-check", {})
    live.session.transcripts["decision"].append(
        "old recognition", clients.Response({}, None, "unknown", "old", 1, 1)
    )
    (live.root / "docs" / "new.md").write_text("A new registration.\n")
    commit(live.root)
    git(live.root, "tag", "-f", "registered")
    fresh = restart(live, monkeypatch)
    assert fresh.directory != old
    assert fresh.directory.name == "live-" + git(live.root, "rev-parse", "HEAD")
    assert not fresh.state("judge-check", required=False)
    assert not any(t.entries for t in fresh.session.transcripts.values())
    calls: list[int] = []

    def complete() -> run.State:
        calls.append(1)
        return fresh.save("judge-check", {})

    monkeypatch.setattr(fresh, "_judge_check", complete)
    assert "completed" in fresh.execute("judge-check")[0]
    assert calls == [1]
    assert old.exists()


@pytest.mark.parametrize("field", ["registered_commit", "code_snapshot"])
def test_state_refuses_unbound_identity(live: run.Run, field: str) -> None:
    live.save("judge-check", {})
    path = live.directory / "judge-check.json"
    document = json.loads(path.read_text())
    document[field] = "different" if field == "registered_commit" else {"example.py": "different"}
    path.write_bytes(run.encoded(document))
    with pytest.raises(clients.GuardRefusal, match=r"registered commit|snapshot"):
        live.state("judge-check")


def test_return_from_abandoned_repair_requires_its_own_declaration(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    tagged = git(live.root, "rev-parse", "HEAD")
    change(live, "src/zoon/pilot/clients.py")

    def complete() -> run.State:
        return live.save("judge-check", {})

    monkeypatch.setattr(live, "_judge_check", complete)
    live.execute("judge-check", repair="Restore request handling")
    saved = live.state("judge-check")["code_snapshot"]
    assert saved
    git(live.root, "checkout", "-q", tagged)
    fresh = restart(live, monkeypatch)
    with pytest.raises(clients.GuardRefusal, match="repair"):
        fresh.state("judge-check")
    with pytest.raises(clients.GuardRefusal, match="repair"):
        fresh.execute("judge")
    fresh.guard("judge", None, "Return to the registered implementation")
    assert fresh.pending_binding is not None
    fresh.binding(fresh.pending_binding)
    fresh.guard("judge", None, None)
    assert clients.obj(clients.items(fresh.binding()["repairs"])[-1])["code"] == {}


@pytest.mark.parametrize("name", ["src/zoon/extra.py", "untracked.txt"])
@pytest.mark.parametrize("repair", [None, "Restore operation"])
def test_untracked_files_anywhere_refuse_without_writing(
    live: run.Run, name: str, repair: str | None
) -> None:
    (live.root / name).write_text("unexpected\n")
    with pytest.raises(clients.GuardRefusal, match="clean work tree"):
        live.execute("judge-check", repair=repair)
    assert not live.directory.exists()


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
@pytest.mark.parametrize("repair", [None, "Restore request handling"])
def test_hidden_disk_edits_cannot_be_declared_as_repairs(
    live: run.Run, flag: str, repair: str | None
) -> None:
    name = "src/zoon/pilot/clients.py"
    git(live.root, "update-index", flag, name)
    (live.root / name).write_text("changed\n")
    assert git(live.root, "status", "--porcelain", "--untracked-files=all") == ""
    with pytest.raises(clients.GuardRefusal, match="matching HEAD"):
        live.execute("judge-check", repair=repair)
    assert not live.directory.exists()


def test_root_level_tracked_code_requires_repair_and_is_reported(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    (live.root / "settings.toml").write_text("enabled = true\n")
    commit(live.root)
    with pytest.raises(clients.GuardRefusal, match="repair"):
        live.execute("judge-check")

    def stop() -> run.State:
        raise clients.TerminalStop(clients.StopReason.CALIBRATION)

    monkeypatch.setattr(live, "_judge_check", stop)
    live.execute("judge-check", repair="Restore configuration")
    repair = clients.obj(clients.items(live.binding()["repairs"])[0])
    assert repair["files"] == ["settings.toml"]
    assert set(clients.obj(repair["code"])) == {"settings.toml"}
    assert any(
        "difference from registered" in line and "settings.toml" in line
        for line in live.execute("report")
    )


@pytest.mark.parametrize("folder", ["raw", "outside"])
def test_missing_dataset_refuses_before_state_or_requests_then_resumes(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, folder: str
) -> None:
    from tests.synthetic import write_outside, write_raw
    from zoon.pilot import outside

    raw, external = live.root / "data" / "raw", live.root / "data" / "outside"
    write_raw(raw, 3)
    write_outside(external)
    (live.root / "data-manifest.json").unlink()
    (live.root / "data-manifest-outside.json").unlink()
    fetch.fetch(raw, live.root / "data-manifest.json")
    outside.fetch(external, live.root / "data-manifest-outside.json")
    commit(live.root)
    git(live.root, "tag", "-f", "registered")
    fresh = run.Run(live.root, supplied_clients=[])
    original = run.Run.verify_materials
    reached: list[str] = []

    def training() -> tuple[list[run.cohorts.Participant], run.cohorts.Cohorts]:
        reached.append("training")
        people = run.stub.people(48, 41)
        return people, run.cohorts.generate(people, calibration_size=8)

    monkeypatch.setattr(fresh, "training", training)
    # Regenerate only synthetic fixtures; no held-out loader is reachable.
    people, document = training()
    (live.root / "prereg" / "cohorts.json").write_bytes(run.cohorts.serialise(document))
    records = run.rewrites.load(live.root / "pilot" / "rewrites")
    target = live.root / "prereg" / "requests"
    for path in target.iterdir():
        if path.name != "README.md":
            path.unlink()
    for name, value in run.prompts.fixtures(
        people, document, run.materials.load(), records
    ).items():
        (target / name).write_bytes(value)
    commit(live.root)
    git(live.root, "tag", "-f", "registered")
    fresh = run.Run(live.root, supplied_clients=[])
    monkeypatch.setattr(fresh, "training", training)

    def accept(*args: object) -> None:
        pass

    monkeypatch.setattr(outside, "check_against_paper", accept)
    monkeypatch.setattr(fresh, "verify_materials", lambda: original(fresh))
    path = next((live.root / "data" / folder).iterdir())
    content = path.read_bytes()
    path.unlink()
    reached.clear()
    with pytest.raises(clients.GuardRefusal, match="manifest"):
        fresh.execute("judge-check")
    assert not fresh.directory.exists()
    assert not reached
    path.write_bytes(content)

    def complete() -> run.State:
        reached.append("body")
        return fresh.save("judge-check", {})

    monkeypatch.setattr(fresh, "_judge_check", complete)
    assert "completed" in fresh.execute("judge-check")[0]
    assert reached == ["training", "body"]
    assert not any(t.entries for t in fresh.session.transcripts.values())


def test_refused_guard_cannot_leave_a_pending_repair_for_a_later_command(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    tagged = git(live.root, "rev-parse", "HEAD")
    change(live, "src/zoon/pilot/clients.py")

    def refuse() -> None:
        raise clients.GuardRefusal("materials unavailable")

    monkeypatch.setattr(live, "verify_materials", refuse)
    with pytest.raises(clients.GuardRefusal, match="materials"):
        live.execute("judge-check", repair="Restore requests")
    assert live.pending_binding is None
    assert not live.directory.exists()
    git(live.root, "checkout", "-q", tagged)
    monkeypatch.setattr(live, "verify_materials", lambda: None)

    def complete() -> run.State:
        return live.save("judge-check", {})

    monkeypatch.setattr(live, "_judge_check", complete)
    assert "completed" in live.execute("judge-check")[0]
    assert live.binding()["repairs"] == []
    assert live.state("judge-check")["code_snapshot"] == {}
