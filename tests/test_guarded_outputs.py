"""Both guarded-output consumers use the same named refusal and storage boundary."""

import errno
import os
import socket
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.test_interaction_step_faults import Factory, state
from zoon.interaction import chat, run, simulation
from zoon.pilot import clients

pytest_plugins = ("tests.test_interaction_step_faults",)


@pytest.mark.parametrize("consumer", ["store", "permit"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "directory",
        "parent file",
        "permission",
        "loop",
        "storage",
        "named pipe",
        "socket",
        "device",
        "digest",
    ],
)
def test_guarded_output_faults_share_the_read_boundary(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, consumer: str, damage: str
) -> None:
    runner = runs()
    saved = state(runner)
    name = next(iter(saved["payload"]["digests"]))
    path = runner.guard.storage / name
    raw = path.read_bytes()
    if damage == "digest":
        path.write_bytes(b"changed")
    elif damage == "device":
        original_stat = Path.stat

        def device(current: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            result = original_stat(current, follow_symlinks=follow_symlinks)
            if current == path:
                values = list(result)
                values[0] = stat.S_IFCHR | 0o600
                return os.stat_result(values)
            return result

        monkeypatch.setattr(Path, "stat", device)
    elif damage in ("missing", "directory", "loop", "named pipe", "socket"):
        path.unlink()
        if damage == "directory":
            path.mkdir()
        elif damage == "named pipe":
            os.mkfifo(path)
        elif damage == "socket":
            with monkeypatch.context() as patch, socket.socket(socket.AF_UNIX) as server:
                patch.chdir(path.parent)
                server.bind(path.name)
        elif damage == "loop":
            path.symlink_to(path.name)
    elif damage == "parent file":
        # The test needs only this output; preserve its neighbors for fixture cleanup.
        moved = path.parent.with_name(path.parent.name + "-saved")
        path.parent.rename(moved)
        path.parent.write_bytes(b"not a directory")
    else:
        original = Path.read_bytes

        def read(current: Path) -> bytes:
            if current == path:
                error = PermissionError if damage == "permission" else OSError
                raise error(
                    errno.EACCES if damage == "permission" else errno.EIO,
                    "storage unavailable",
                    str(current),
                )
            return original(current)

        monkeypatch.setattr(Path, "read_bytes", read)
    if damage in ("named pipe", "socket", "device"):

        def unexpected(current: Path) -> bytes:
            pytest.fail("nonregular output was opened")

        monkeypatch.setattr(Path, "read_bytes", unexpected)
    storage_fault = damage in ("loop", "storage")
    message = (
        f"guarded output {name} storage failed"
        if storage_fault
        else f"missing guarded output {name}"
        if damage == "missing"
        else "recorded simulation digest differs from its output"
        if damage == "digest"
        else f"guarded output {name} cannot be read"
    )
    with runner.guard.lock():
        if consumer == "permit":
            runner.guard.mode, runner.guard.checked_step = "live", "held-out"

            def completed(step: str) -> dict[str, Any]:
                return {**saved, "status": "completed"}

            monkeypatch.setattr(runner.guard, "state", completed)
        with pytest.raises(
            chat.StorageInterruption if storage_fault else clients.GuardRefusal
        ) as caught:
            if consumer == "store":
                simulation.Store(runner.guard, saved["payload"], lambda: None).verify()
            else:
                runner.guard.permit()
        assert str(caught.value) == message
        assert str(runner.guard.storage) not in str(caught.value)
    assert raw


@pytest.mark.parametrize("damage", ["loop", "storage", "parent file"])
def test_guarded_output_command_hides_storage_paths(
    runs: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    runner = runs()
    name = next(iter(state(runner)["payload"]["digests"]))
    path = runner.guard.storage / name
    if damage == "loop":
        path.unlink()
        path.symlink_to(path.name)
    elif damage == "parent file":
        path.parent.rename(path.parent.with_name(path.parent.name + "-saved"))
        path.parent.write_bytes(b"not a directory")
    else:
        original = Path.read_bytes

        def read(current: Path) -> bytes:
            if current == path:
                raise OSError(errno.EIO, "storage unavailable", str(current))
            return original(current)

        monkeypatch.setattr(Path, "read_bytes", read)

    def local(**kwargs: Any) -> run.Run:
        return runner

    monkeypatch.setattr(run, "Run", local)
    assert run.main(["closed-loop", "--stub"]) == 1
    output = capsys.readouterr()
    error = "GuardRefusal" if damage == "parent file" else "StorageInterruption"
    suffix = "cannot be read" if damage == "parent file" else "storage failed"
    assert output.out == ""
    assert output.err == f"{error}: guarded output {name} {suffix}\n"
    assert str(runner.guard.storage) not in output.err
    assert state(runner)["status"] == "interrupted" and state(runner)["reason"] == error


def test_validate_accepts_only_the_response() -> None:
    response = chat.Response("3", chat.MODEL, 12, 1)
    chat.validate(response)
    with pytest.raises(TypeError, match="bearer"):
        chat.validate(response, bearer="dummy")  # type: ignore[call-arg]


def test_later_save_cannot_add_a_transcript_family(runs: Factory) -> None:
    runner = runs(prepared="calibrated")
    with runner.guard.lock():
        runner.guard.before("closed-loop")
        runner.guard.save("closed-loop", {}, families=("decision",), status="running")
        with pytest.raises(clients.GuardRefusal, match="families changed"):
            runner.guard.save("closed-loop", {}, families=("decision", "chat"), status="running")
