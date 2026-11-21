"""Isolated checkouts with fake source bytes and verified training-only materials."""

import json
import shutil
import subprocess
from pathlib import Path

from zoon.interaction import data
from zoon.interaction.guard import TAG, digest


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def commit(root: Path, message: str = "Change implementation") -> str:
    git(root, "add", ".")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


def repository(root: Path, *, tag: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "Example")
    git(root, "config", "user.email", "example@example.org")
    (root / ".gitignore").write_text("data/\n")
    (root / "code.py").write_text("value = 1\n")
    shutil.copytree(data.REPO / "interaction", root / "interaction")
    directory = root / "data" / "groups"
    (directory / "raw").mkdir(parents=True)
    # Training may be used before registration; actual source bytes are never copied.
    shutil.copyfile(data.TRAINING, directory / "training.json")
    manifest_path = root / "interaction" / "data-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for name in manifest["files"]:
        raw = b"synthetic source; not a participant record"
        (directory / "raw" / name).write_bytes(raw)
        manifest["files"][name].update(size=len(raw), sha256=digest(raw))
    manifest_path.write_text(json.dumps(manifest))
    commit(root, "Register materials")
    if tag:
        git(root, "tag", TAG)
    return root
