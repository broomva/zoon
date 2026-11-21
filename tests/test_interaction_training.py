"""Training-dependent checks must run when required by the environment."""

import os
from pathlib import Path

import pytest

from zoon.interaction import data


def require_training() -> None:
    if not data.TRAINING.is_file():
        message = "training file is absent"
        if os.environ.get("ZOON_REQUIRE_TRAINING") == "1":
            pytest.fail(message)
        pytest.skip(message)


def test_training_requirement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    training = tmp_path / "training.json"
    monkeypatch.setattr(data, "TRAINING", training)
    monkeypatch.delenv("ZOON_REQUIRE_TRAINING", raising=False)
    with pytest.raises(pytest.skip.Exception, match="training file is absent"):
        require_training()
    monkeypatch.setenv("ZOON_REQUIRE_TRAINING", "1")
    with pytest.raises(pytest.fail.Exception, match="training file is absent"):
        require_training()
    training.write_bytes(b"present")
    require_training()
