"""Every live step compares disk bytes with the same registered commit."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

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
    for directory in ("src", "prereg", "pilot"):
        shutil.copytree(fetch.REPO / directory, tmp_path / directory)
    for name in (
        "pyproject.toml",
        "uv.lock",
        *[p.name for p in fetch.REPO.glob("data-manifest*.json")],
    ):
        shutil.copyfile(fetch.REPO / name, tmp_path / name)
    (tmp_path / ".gitignore").write_text("data/\n__pycache__/\n")
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
        ("hidden-registered", "registered files"),
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
    live = restart(live, monkeypatch)
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
    with pytest.raises(clients.StopRun, match=pattern):
        live.execute(step, tagged, repair="Restore operation" if "registered" in fault else None)
    assert not reached
    assert not any(t.entries for t in live.session.transcripts.values())


def reach(live: run.Run, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reached: list[str] = []

    def stopped() -> run.State:
        reached.append("work")
        return live.save("judge-check", {}, status="stopped", reason="fixture stop")

    monkeypatch.setattr(live, "judge_check", stopped)
    return reached


def test_equal_tag_and_matching_repair_snapshot_go_on(
    live: run.Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached = reach(live, monkeypatch)
    live.execute("judge-check")
    assert reached == ["work"]
    change(live, "src/zoon/pilot/score.py")
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
    with pytest.raises(clients.StopRun, match=r"score\.py"):
        live.execute("judge-check")
    change(live, second)
    live = restart(live, monkeypatch)
    reach(live, monkeypatch)
    with pytest.raises(clients.StopRun, match=r"score\.py"):
        live.execute("judge-check")
    live.execute("judge-check", repair="Restore scoring and requests")
    repair = clients.obj(clients.items(live.binding()["repairs"])[0])
    assert repair["files"] == sorted([first, second])
    assert repair["commit"] == git(live.root, "rev-parse", "HEAD")
    assert set(clients.obj(repair["code"])) == {first, second}
    (live.root / "note.txt").write_text("Reporting note")
    commit(live.root)
    lines = live.execute("report")
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
        for line in live.execute("report")
    )


@pytest.mark.parametrize("changed", ["src/zoon/pilot/clients.py", "src/zoon/pilot/score.py"])
def test_previous_repair_does_not_cover_later_code(
    live: run.Run, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    reach(live, monkeypatch)
    change(live, "src/zoon/pilot/score.py")
    live.execute("judge-check", repair="Restore scoring")
    change(live, changed)
    live = restart(live, monkeypatch)
    with pytest.raises(clients.StopRun, match="repair"):
        live.execute("judge-check")


def test_report_without_repairs_says_none(live: run.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    reach(live, monkeypatch)
    live.execute("judge-check")
    assert "[Repairs] history = none" in live.execute("report")
    saved = json.loads((live.directory / "judge-check.json").read_text())
    assert saved["registered_files"] and saved["code_files"]
