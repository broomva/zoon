"""Guard refusals distinguish checkout, material and persistence faults."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.interaction_run_support import commit, git, repository
from zoon.interaction import data, guard, registered
from zoon.interaction.guard import Guard, atomic, checkpoint, digest, encoded
from zoon.pilot import clients


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return repository(tmp_path / "repo")


def before(root: Path, **kwargs: Any) -> Guard:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate", **kwargs)
    return gate


@pytest.mark.parametrize("keys", [["b", "a"], ["a", "a"]])
def test_checkpoint_refuses_noncanonical_keys(tmp_path: Path, keys: list[str]) -> None:
    ledger = clients.Transcript(tmp_path, "decision")
    for key in ("a", "b"):
        ledger.entries[key] = clients.Entry(key, {}, clients.Response({}, None, "", "v", 0, 0))
    with pytest.raises(clients.GuardRefusal, match="duplicates"):
        checkpoint(ledger, keys)


def test_terminal_requires_each_hypothesis(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        with pytest.raises(clients.GuardRefusal, match="both no-verdict"):
            gate.save(
                "calibrate",
                {"outcome": {"H2": "no verdict", "reason": "FIT"}},
                status="terminal",
                reason="fit failed",
            )


def test_before_requires_lock_before_inspecting_checkout(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = Guard(root)

    def unexpected(*args: Any) -> None:
        pytest.fail("checkout inspected without the lock")

    monkeypatch.setattr(gate, "_checkout", unexpected)
    with pytest.raises(clients.GuardRefusal, match="lock"):
        gate.before("calibrate")


def test_save_requires_the_same_step_to_pass_before(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        with pytest.raises(clients.GuardRefusal, match="passed the guard"):
            gate.save("calibrate", {}, status="running")
        gate.before("calibrate")
        with pytest.raises(clients.GuardRefusal, match="passed the guard"):
            gate.save("recall", {}, status="running")


@pytest.mark.parametrize("name", ["src/code.py", "pyproject.toml", "uv.lock"])
def test_implementation_paths_require_declared_repairs(root: Path, name: str) -> None:
    (root / name).write_text("changed implementation\n")
    commit(root)
    with pytest.raises(clients.GuardRefusal, match="repair reason"):
        before(root)
    assert name in before(root, repair="Resolve execution fault").code


def test_paths_tracked_only_at_tag_are_compared_with_disk(root: Path) -> None:
    path = root / "data/tagged.py"
    path.write_text("value = 1\n")
    git(root, "add", "-f", "data/tagged.py")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    git(root, "rm", "data/tagged.py")
    commit(root)
    path.write_text("value = 1\n")
    assert git(root, "status", "--porcelain", "--untracked-files=all") == ""
    with pytest.raises(clients.GuardRefusal, match="disk bytes"):
        before(root, repair="Resolve execution fault")


def test_declared_repair_must_remain_an_ancestor(root: Path) -> None:
    (root / "src/code.py").write_text("value = 2\n")
    commit(root)
    before(root, repair="Resolve execution fault")
    # Same tree, different history: the declared commit has been replaced.
    replacement = git(
        root, "commit-tree", "HEAD^{tree}", "-p", guard.TAG, "-m", "Replace implementation"
    )
    git(root, "update-ref", "HEAD", replacement)
    with pytest.raises(clients.GuardRefusal, match="repair commit must remain an ancestor"):
        before(root)


@pytest.mark.parametrize("directory", ["instructions", "requests", "chat-requests"])
def test_registered_directories_refuse_added_files(root: Path, directory: str) -> None:
    (root / "interaction" / directory / "extra.txt").write_text("extra")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    with pytest.raises(clients.GuardRefusal, match="file set"):
        before(root)


def test_amounts_are_regenerated_even_when_the_reader_accepts_them(root: Path) -> None:
    path = root / "interaction/amounts.json"
    document = json.loads(path.read_bytes())
    document.pop("sha256")
    document["rows"][0][2].append(document["rows"][0][2][-1])
    path.write_bytes(registered.sealed(document))
    registered.read_amounts(path, root / "interaction/data-manifest.json")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    with pytest.raises(clients.GuardRefusal, match="do not regenerate"):
        before(root)


def test_cited_input_needs_a_guarded_step(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.registered_commit = git(root, "rev-parse", guard.TAG)
        with pytest.raises(clients.GuardRefusal, match="guarded registration"):
            gate.input_bytes("src/code.py")


def test_terminal_outcomes_are_validated_when_read(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        gate.before("calibrate")
        state = gate.save("calibrate", {}, status="interrupted")
        state.update(status="terminal", reason="fit failed")
        state.pop("digest")
        atomic(gate.directory / "calibrate.json", {**state, "digest": digest(encoded(state))})
        with pytest.raises(clients.GuardRefusal, match="both no-verdict"):
            gate.state("calibrate")


def test_stub_never_reads_raw_sources_and_live_requires_them(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.rmtree(root / "data/groups/raw")
    real_verify = guard.verify_data

    def unexpected(*args: Any) -> None:
        pytest.fail("stub read raw source hashes")

    monkeypatch.setattr(guard, "verify_data", unexpected)
    gate = Guard(root, stub=True)
    with gate.lock():
        gate.before("calibrate")
    monkeypatch.setattr(guard, "verify_data", real_verify)
    with pytest.raises(clients.GuardRefusal, match="dataset unavailable"):
        before(root)


@pytest.mark.parametrize(
    "name,allowed",
    [
        ("src/pkg/__pycache__/module.cpython-312.pyc", True),
        ("src/pkg/__pycache__/café.cpython-312.pyc", True),
        ("src/pkg/__pycache__/café.txt", False),
        ("interaction/__pycache__/module.pyc", True),
        ("src/pkg/module.pyc", False),
        ("src/pkg/__pycache__/module.py", False),
        ("interaction/__pycache__/other.txt", False),
        ("src/zoon/interaction/data/__init__.py", False),
        ("interaction/hidden.py", False),
    ],
)
def test_ignored_paths_allow_only_interpreter_bytecode_cache(
    root: Path, name: str, allowed: bool
) -> None:
    (root / ".git/info/exclude").write_text(name + "\n")
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ignored content")
    if allowed:
        before(root)
    else:
        with pytest.raises(clients.GuardRefusal, match="ignored files"):
            before(root)


def test_steps_register_and_reload_their_transcript_families(root: Path) -> None:
    gate = Guard(root)
    gate.register_transcripts("second")
    with gate.lock():
        gate.before("calibrate")
        gate.transcripts["second"].append(
            "synthetic recall", clients.StubClient("second").ask("synthetic recall")
        )
        saved = gate.save("calibrate", {}, status="completed")
        assert set(saved["transcripts"]) == {"second"}
    resumed = Guard(root)
    with resumed.lock():
        resumed.before("closed-loop")
        assert resumed.state("calibrate") == saved
        assert "second" in resumed.transcripts


@pytest.mark.parametrize("family", ["decision", "second"])
@pytest.mark.parametrize("field", [None, "mode", "registered_commit"])
def test_registered_family_identity_is_checked_before_any_step_record(
    root: Path, family: clients.Family, field: str | None
) -> None:
    initial = before(root)
    identity = dict(initial.identity)
    if field is not None:
        identity[field] = "stub" if field == "mode" else "b" * 40
    path = initial.transcript_directory / f"{family}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity) + "\n")
    gate = Guard(root)
    gate.register_transcripts(family)
    assert not (gate.directory / "calibrate.json").exists()
    with gate.lock():
        if field is None:
            gate.before("calibrate")
            assert family in gate.transcripts
            assert gate.transcripts[family].identity == identity
        else:
            with pytest.raises(clients.GuardRefusal, match="identity"):
                gate.before("calibrate")
            assert gate.checked_step is None
    assert not (gate.directory / "calibrate.json").exists()


def test_removing_held_lock_does_not_admit_another_command(root: Path) -> None:
    gate = Guard(root)
    with gate.lock():
        (gate.storage / "run/lock").unlink()
        with pytest.raises(clients.GuardRefusal, match="another command"), Guard(root).lock():
            pytest.fail("second command entered")


def test_lock_handle_must_still_name_the_lock_file(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = guard.fcntl.flock
    path = root / "data/interaction/run/lock"
    count = 0

    def replaced(descriptor: Any, operation: int) -> None:
        nonlocal count
        original(descriptor, operation)
        count += 1
        if count == 2:
            path.rename(path.with_name("previous-lock"))
            path.touch()

    monkeypatch.setattr(guard.fcntl, "flock", replaced)
    with pytest.raises(clients.GuardRefusal, match="lock path changed"), Guard(root).lock():
        pytest.fail("replaced lock entered")


def test_steps_register_their_own_transcript_format(root: Path) -> None:
    # A different record format can supply the same checkpoint shape.
    records = {"turn-1": {"text": "synthetic chat reply", "tokens": 3}}

    def read(keys: list[str] | None) -> dict[str, Any]:
        selected = sorted(records) if keys is None else keys
        return {"keys": selected, "digest": digest(encoded([records[k] for k in selected]))}

    gate = Guard(root)
    gate.register_transcript("chat", read)
    with gate.lock():
        gate.before("calibrate")
        gate.save("calibrate", {}, status="completed")
        gate.before("closed-loop")
        saved = gate.save("closed-loop", {}, status="completed")
        assert saved["transcripts"] == {"chat": read(None)}
    resumed = Guard(root)
    resumed.register_transcript("chat", read)
    with resumed.lock():
        resumed.before("recall")
        assert resumed.state("closed-loop") == saved
        records["turn-1"]["text"] = "changed"
        with pytest.raises(clients.GuardRefusal, match="checkpoint"):
            resumed.state("closed-loop")


def test_root_data_ignore_does_not_hide_source_packages(root: Path) -> None:
    (root / ".gitignore").write_bytes((data.REPO / ".gitignore").read_bytes())
    assert git(root, "check-ignore", "data/groups/training.json") == "data/groups/training.json"
    path = root / "src/zoon/interaction/data/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n")
    assert (
        "src/zoon/interaction/data/__init__.py"
        in git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    )


def test_clean_checkout_accepts_literal_tracked_paths(root: Path) -> None:
    for name in ("café.md", " spaced\tname\n.md "):
        path = root / "docs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("document\n")
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    assert before(root).differences == {}


def test_head_cannot_move_during_checkout_verification(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = git(root, "rev-parse", "HEAD")
    original = guard.git
    calls: list[tuple[str, ...]] = []
    snapshots: list[str] = []
    original_snapshot = guard.snapshot_tree

    def snapshot(directory: Path, commit: str) -> dict[str, str]:
        snapshots.append(commit)
        return original_snapshot(directory, commit)

    def moving(directory: Path, *arguments: str) -> str:
        calls.append(arguments)
        result = original(directory, *arguments)
        if arguments == ("rev-parse", "HEAD") and result == captured:
            # Move history while leaving the disk and index unchanged.
            replacement = git(
                root, "commit-tree", "HEAD^{tree}", "-p", captured, "-m", "Add history"
            )
            git(root, "update-ref", "HEAD", replacement)
        return result

    monkeypatch.setattr(guard, "git", moving)
    monkeypatch.setattr(guard, "snapshot_tree", snapshot)
    with pytest.raises(clients.GuardRefusal, match="HEAD moved"):
        before(root)
    assert ("merge-base", "--is-ancestor", captured, captured) in calls
    assert snapshots and set(snapshots) == {captured}
    assert not (root / "data/interaction/run/live/binding.json").exists()
