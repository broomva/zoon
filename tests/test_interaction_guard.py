"""Every guard rule is exercised in an isolated git repository."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.interaction_run_support import commit, git, repository
from zoon.interaction import data, guard, registered
from zoon.interaction.guard import Guard, atomic, digest, encoded
from zoon.pilot import clients


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return repository(tmp_path / "repo")


def passed(root: Path, *, stub: bool = False, repair: str | None = None) -> Guard:
    gate = Guard(root, stub=stub)
    with gate.lock():
        gate.before("calibrate", repair)
    return gate


@pytest.mark.parametrize("stub", [False, True])
@pytest.mark.parametrize("damage", ["untracked", "tracked", "assume", "deleted"])
def test_clean_checkout_and_disk_bytes(root: Path, stub: bool, damage: str) -> None:
    if damage == "untracked":
        (root / "extra").write_text("untracked")
    elif damage == "deleted":
        (root / "code.py").rename(root / "ignored.py")
    else:
        if damage == "assume":
            git(root, "update-index", "--assume-unchanged", "code.py")
        (root / "code.py").write_text("value = 2\n")
    with pytest.raises(clients.GuardRefusal, match="clean work tree"):
        passed(root, stub=stub)


def test_ignored_files_are_allowed_and_only_stub_waives_the_tag(tmp_path: Path) -> None:
    root = repository(tmp_path / "repo", tag=False)
    with pytest.raises(clients.GuardRefusal):
        passed(root)
    assert passed(root, stub=True).mode == "stub"
    assert not (root / "data" / "run").exists()
    assert not (root / "data" / "transcripts").exists()


def test_tag_must_be_an_ancestor(root: Path) -> None:
    git(root, "checkout", "--orphan", "other")
    commit(root)
    with pytest.raises(clients.GuardRefusal):
        passed(root)


@pytest.mark.parametrize("name", ["registration.md", "fitted.json", "amounts.json"])
def test_registered_bytes_cannot_be_repaired(root: Path, name: str) -> None:
    path = root / "interaction" / name
    path.write_bytes(path.read_bytes() + b"\n")
    commit(root)
    with pytest.raises(clients.GuardRefusal, match="registered files"):
        passed(root, repair="Resolve execution fault")


def test_code_snapshots_require_new_ordered_committed_declarations(root: Path) -> None:
    original = git(root, "rev-parse", "HEAD")
    passed(root)
    for number in (2, 3, 1):
        (root / "code.py").write_text(f"value = {number}\n")
        head = commit(root)
        with pytest.raises(clients.GuardRefusal, match="repair reason"):
            passed(root)
        with pytest.raises(clients.GuardRefusal, match="repair reason"):
            passed(root, repair=" ")
        gate = passed(root, repair="Resolve execution fault")
        record = gate.repairs[-1]
        assert record["commit"] == head
        assert record["registered_commit"] == original
        assert record["mode"] == "live"
        assert record["reason"] == "Resolve execution fault"
        expected = (
            {} if number == 1 else {"code.py": "100644:" + digest((root / "code.py").read_bytes())}
        )
        assert record["code"] == expected and record["files"] == ["code.py"]
        assert passed(root).repairs == gate.repairs
    assert [r["number"] for r in gate.repairs] == [1, 2, 3]
    assert gate.repairs[0]["previous"] is None
    assert all(
        b["previous"] == a["digest"] for a, b in zip(gate.repairs, gate.repairs[1:], strict=False)
    )


def test_docs_and_tests_are_recorded_differences_but_not_code(root: Path) -> None:
    for name in ("docs", "tests"):
        (root / name).mkdir()
        (root / name / "record.txt").write_text("record")
    commit(root)
    gate = passed(root)
    assert gate.code == {}
    assert set(gate.differences) == {"docs/record.txt", "tests/record.txt"}


def test_repair_history_cannot_be_reordered_or_rebound(root: Path) -> None:
    (root / "code.py").write_text("value = 2\n")
    commit(root)
    gate = passed(root, repair="Resolve execution fault")
    path = gate.directory / "binding.json"
    document = json.loads(path.read_bytes())
    document["repairs"][0]["code"] = {}
    atomic(path, document)
    with pytest.raises(clients.GuardRefusal, match="ordered"):
        passed(root)


@pytest.mark.parametrize(
    "kind", ["source", "training", "instructions", "requests", "amounts", "chat"]
)
def test_guard_verifies_hashes_and_regenerates_materials(root: Path, kind: str) -> None:
    # Material changes are committed and tagged here to distinguish regeneration
    # checks from the independent byte comparison against an earlier tag.
    directory = root / "interaction"
    if kind == "source":
        next((root / "data" / "groups" / "raw").iterdir()).write_bytes(b"changed")
    elif kind == "training":
        training = root / "data" / "groups" / "training.json"
        document = json.loads(training.read_bytes())
        document["dataset"] = "changed source label"
        training.write_text(json.dumps(document))
    else:
        path = {
            "instructions": directory / "instructions" / "2-AI.txt",
            "requests": directory / "requests" / "2-AI-type-0-round-1.json",
            "chat": directory / "chat-requests" / "2-AI-type-0-round-1.txt",
            "amounts": directory / "amounts.json",
        }[kind]
        if kind == "amounts":
            document = json.loads(path.read_bytes())
            document.pop("sha256")
            document["rows"][0][2][0] += 1
            path.write_bytes(registered.sealed(document))
        else:
            path.write_bytes(path.read_bytes() + b" ")
        commit(root)
        git(root, "tag", "-f", guard.TAG)
    with pytest.raises(clients.GuardRefusal):
        passed(root)


def test_guard_never_refits_the_primary_arms(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("primary fits must not be regenerated")

    monkeypatch.setattr(registered, "fit", refuse)
    from zoon.interaction.run import Run

    Run(root, stub=True).execute()


def test_step_state_identity_outputs_and_exact_prior_states(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        first = gate.save("calibrate", {"value": 1}, status="completed")
        assert first["previous"] == {}
        gate.before("closed-loop")
        log = gate.storage / "simulation.jsonl"
        log.write_bytes(b"synthetic simulation\n")
        second = gate.save(
            "closed-loop", {"digests": {log.name: digest(log.read_bytes())}}, status="completed"
        )
        assert second["previous"] == {
            "calibrate": digest((gate.directory / "calibrate.json").read_bytes())
        }
        gate.before("recall")
        gate.save("recall", {}, status="completed")
        gate.before("held-out")
        assert isinstance(gate.permit(), guard.HeldOutPermit)
    assert first["registered_commit"] == git(root, "rev-parse", guard.TAG)
    assert first["commit"] == git(root, "rev-parse", "HEAD")
    assert first["mode"] == "live" and first["step"] == "calibrate"
    assert first["payload_digest"] == digest(encoded(first["payload"]))
    path = gate.directory / "calibrate.json"
    first["payload"]["value"] = 2
    atomic(path, first)
    with pytest.raises(clients.GuardRefusal, match="digest"):
        passed(root)


@pytest.mark.parametrize(
    "field,value", [("registered_commit", "b" * 40), ("mode", "stub"), ("step", "recall")]
)
def test_foreign_state_refused_even_with_a_valid_digest(root: Path, field: str, value: str) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        state = gate.save("calibrate", {}, status="interrupted")
    state[field] = value
    state.pop("digest")
    atomic(gate.directory / "calibrate.json", {**state, "digest": digest(encoded(state))})
    with pytest.raises(clients.GuardRefusal, match="another"):
        passed(root)


def test_later_steps_require_completion_and_report_accepts_terminal_absence(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        with pytest.raises(clients.GuardRefusal, match="prerequisite"):
            gate.before("closed-loop")
        gate.before("calibrate")
        gate.save(
            "calibrate",
            {"outcome": {"H2": "no verdict", "H3": "no verdict", "reason": "FIT"}},
            status="terminal",
            reason="fit failed",
        )
        with pytest.raises(clients.GuardRefusal, match="terminal"):
            gate.before("closed-loop")
        gate.before("report")
        report = gate.save("report", {"result": "no verdict"}, status="completed")
        assert report["previous"]["closed-loop"] is None
        gate.before("report")
        assert gate.save("report", {"result": "no verdict"}, status="completed") == report
        with pytest.raises(clients.GuardRefusal, match="reproduce"):
            gate.save("report", {"result": "changed"}, status="completed")
        # An earlier absence is part of the report binding too.
        atomic(gate.directory / "closed-loop.json", {})
        with pytest.raises(clients.GuardRefusal, match="earlier-step"):
            gate.state("report")


def test_second_process_cannot_hold_lock_and_release_follows_failure(root: Path) -> None:
    gate = Guard(root)
    source = (
        "from pathlib import Path\nfrom zoon.interaction.guard import Guard\n"
        "from zoon.pilot.clients import GuardRefusal\nimport sys\n"
        "try:\n with Guard(Path(sys.argv[1])).lock(): pass\n"
        "except GuardRefusal:\n sys.exit(7)\n"
    )
    with gate.lock():
        result = subprocess.run(
            [sys.executable, "-c", source, str(root)],
            env={**os.environ, "PYTHONPATH": str(data.REPO / "src")},
            capture_output=True,
        )
        assert result.returncode == 7
    passed(root)


def test_state_replace_is_atomic_on_write_failure(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        gate.save("calibrate", {"old": True}, status="interrupted")
        path = gate.directory / "calibrate.json"
        before = path.read_bytes()
        replace = Path.replace

        def fail(self: Path, target: Path) -> Path:
            if target == path:
                raise OSError("interrupted replace")
            return replace(self, target)

        monkeypatch.setattr(Path, "replace", fail)
        with pytest.raises(OSError):
            gate.save("calibrate", {"old": False}, status="interrupted")
        assert path.read_bytes() == before


def test_permit_is_live_locked_and_step_scoped(root: Path) -> None:
    with pytest.raises(clients.GuardRefusal):
        data.open_held_out()
    for stub in (False, True):
        gate = Guard(root, stub=stub)
        with gate.lock():
            gate.before("calibrate")
            with pytest.raises(clients.GuardRefusal):
                gate.permit()
        with pytest.raises(clients.GuardRefusal):
            gate.permit()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("payload_digest", "a" * 64, "output digest"),
        ("code_snapshot", {"changed.py": "a" * 64}, "undeclared"),
        ("repair", "a" * 64, "undeclared"),
        ("previous", {"calibrate": None}, "earlier-step"),
        ("status", "unknown", "status"),
        ("schema", 2, "another"),
    ],
)
def test_state_field_checks_are_independent_of_envelope_checksum(
    root: Path, field: str, value: Any, match: str
) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        state = gate.save("calibrate", {"value": 1}, status="interrupted")
    state[field] = value
    state.pop("digest")
    atomic(gate.directory / "calibrate.json", {**state, "digest": digest(encoded(state))})
    with pytest.raises(clients.GuardRefusal, match=match):
        passed(root)


def test_tag_cannot_move_after_a_binding_exists(root: Path) -> None:
    gate = passed(root)
    (root / "docs").mkdir()
    (root / "docs" / "record.txt").write_text("record")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    with pytest.raises(clients.GuardRefusal, match="registered commit"):
        passed(root)
    assert gate.registered_commit != git(root, "rev-parse", guard.TAG)


def test_interrupted_prerequisite_is_not_completion(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        gate.save("calibrate", {}, status="interrupted")
        with pytest.raises(clients.GuardRefusal, match="prerequisite"):
            gate.before("closed-loop")
        with pytest.raises(clients.GuardRefusal, match="report requires"):
            gate.before("report")


def test_pilot_state_is_never_reused_or_changed(root: Path) -> None:
    for name in ("run", "transcripts"):
        directory = root / "data" / name
        directory.mkdir()
        (directory / "live.json").write_bytes(b"invalid pilot state; do not parse")
    gate = passed(root)
    assert gate.directory == root / "data" / "interaction" / "run" / "live"
    assert gate.transcript_directory == root / "data" / "interaction" / "transcripts" / "live"
    assert not (gate.directory / "calibrate.json").exists()
    for name in ("run", "transcripts"):
        assert (
            root / "data" / name / "live.json"
        ).read_bytes() == b"invalid pilot state; do not parse"
        with pytest.raises(clients.GuardRefusal, match="separate"):
            Guard(root, stub=True, storage=root / "data" / name)


@pytest.mark.parametrize(
    "field,value",
    [
        ("digest", "a" * 64),
        ("previous", "a" * 64),
        ("number", 2),
        ("registered_commit", "a" * 40),
        ("mode", "stub"),
        ("reason", " "),
    ],
)
def test_each_repair_record_is_identity_and_order_checked(
    root: Path, field: str, value: Any
) -> None:
    (root / "code.py").write_text("value = 2\n")
    commit(root)
    gate = passed(root, repair="Resolve execution fault")
    path = gate.directory / "binding.json"
    binding = json.loads(path.read_bytes())
    record = binding["repairs"][0]
    record[field] = value
    if field != "digest":
        content = {k: v for k, v in record.items() if k != "digest"}
        record["digest"] = digest(encoded(content))
    atomic(path, binding)
    with pytest.raises(clients.GuardRefusal, match="ordered"):
        passed(root)


def test_repair_snapshot_is_compared_with_its_commit(root: Path) -> None:
    (root / "code.py").write_text("value = 2\n")
    commit(root)
    gate = passed(root, repair="Resolve execution fault")
    path = gate.directory / "binding.json"
    binding = json.loads(path.read_bytes())
    record = binding["repairs"][0]
    record["code"] = {}
    record["files"] = []
    content = {k: v for k, v in record.items() if k != "digest"}
    record["digest"] = digest(encoded(content))
    atomic(path, binding)
    with pytest.raises(clients.GuardRefusal, match="snapshot"):
        passed(root)


def test_step_envelope_checksum_is_checked(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        state = gate.save("calibrate", {}, status="interrupted")
    state["digest"] = "a" * 64
    atomic(gate.directory / "calibrate.json", state)
    with pytest.raises(clients.GuardRefusal, match="state digest"):
        passed(root)


@pytest.mark.parametrize("change", ["registered", "code"])
def test_stub_waives_only_tag_absence(root: Path, change: str) -> None:
    path = root / "interaction" / "registration.md" if change == "registered" else root / "code.py"
    path.write_bytes(path.read_bytes() + b"\n")
    commit(root)
    with pytest.raises(
        clients.GuardRefusal,
        match="registered files" if change == "registered" else "repair reason",
    ):
        passed(root, stub=True)


def test_report_repetition_keeps_its_original_commit_and_output(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        gate.save(
            "calibrate",
            {"outcome": {"H2": "no verdict", "H3": "no verdict", "reason": "FIT"}},
            status="terminal",
            reason="fit failed",
        )
        gate.before("report")
        report = gate.save("report", {"result": "no verdict"}, status="completed")
    (root / "docs").mkdir()
    (root / "docs" / "result.md").write_text("no verdict")
    commit(root)
    with gate.lock():
        gate.before("report")
        assert gate.save("report", {"result": "no verdict"}, status="completed") == report


def test_live_state_cannot_be_redirected(root: Path, tmp_path: Path) -> None:
    with pytest.raises(clients.GuardRefusal, match="live state"):
        Guard(root, storage=tmp_path / "another")


def test_terminal_state_requires_a_reason_and_both_outcomes(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        with pytest.raises(clients.GuardRefusal, match="both no-verdict"):
            gate.save("calibrate", {}, status="terminal")
        with pytest.raises(clients.GuardRefusal, match="both no-verdict"):
            gate.save(
                "calibrate",
                {"outcome": {"H2": "no verdict", "H3": "no verdict", "reason": "FIT"}},
                status="terminal",
            )
        assert gate.state("calibrate") is None


def test_cited_inputs_are_read_from_the_tag_after_docs_change(root: Path) -> None:
    (root / "docs").mkdir()
    path = root / "docs" / "input.txt"
    path.write_bytes(b"registered input\n\n")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    path.write_bytes(b"later text\n")
    commit(root)
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        assert gate.input_bytes("docs/input.txt") == b"registered input\n\n"
        with pytest.raises(clients.GuardRefusal):
            gate.input_bytes("missing")


def test_tracked_mode_change_requires_a_new_repair_snapshot(root: Path) -> None:
    path = root / "code.py"
    path.write_text("value = 2\n")
    commit(root)
    passed(root, repair="Resolve execution fault")
    path.chmod(0o755)
    commit(root)
    with pytest.raises(clients.GuardRefusal, match="repair reason"):
        passed(root)
    gate = passed(root, repair="Resolve execution fault")
    assert gate.repairs[-1]["code"]["code.py"] == "100755:" + digest(path.read_bytes())
