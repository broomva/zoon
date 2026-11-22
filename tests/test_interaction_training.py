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


def outcome() -> str:
    """What the helper does, caught here so that a wrong outcome cannot skip this test."""
    try:
        require_training()
    except pytest.skip.Exception:
        return "skip"
    except pytest.fail.Exception:
        return "fail"
    return "run"


def test_training_requirement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    training = tmp_path / "training.json"
    monkeypatch.setattr(data, "TRAINING", training)
    monkeypatch.delenv("ZOON_REQUIRE_TRAINING", raising=False)
    assert outcome() == "skip"
    monkeypatch.setenv("ZOON_REQUIRE_TRAINING", "0")
    assert outcome() == "skip"
    monkeypatch.setenv("ZOON_REQUIRE_TRAINING", "1")
    assert outcome() == "fail"
    training.write_bytes(b"present")
    assert outcome() == "run"
    monkeypatch.delenv("ZOON_REQUIRE_TRAINING")
    assert outcome() == "run"


@pytest.mark.parametrize("required", [False, True])
def test_repository_checks_the_training_requirement_at_its_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    from tests.interaction_run_support import repository

    monkeypatch.setattr(data, "TRAINING", tmp_path / "absent.json")
    monkeypatch.setenv("ZOON_REQUIRE_TRAINING", "1" if required else "0")
    with pytest.raises(
        pytest.fail.Exception if required else pytest.skip.Exception,
        match="training file is absent",
    ):
        repository(tmp_path / "repo")
    assert not (tmp_path / "repo").exists()
