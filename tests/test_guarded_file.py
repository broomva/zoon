"""Special files are refused before any output bytes are read."""

import os
from pathlib import Path

import pytest

from zoon.interaction import guard
from zoon.pilot import clients


def test_named_pipe_is_refused_without_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.mkfifo(tmp_path / "output")

    def unexpected(self: Path) -> bytes:
        pytest.fail("nonregular output was opened")

    monkeypatch.setattr(Path, "read_bytes", unexpected)
    with pytest.raises(clients.GuardRefusal, match="cannot be read"):
        guard.guarded_output(tmp_path, "output", "a" * 64)
