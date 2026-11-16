"""Copy repository inputs without generated files or local datasets."""

import shutil
import subprocess
from pathlib import Path

from zoon.pilot import fetch


def copy_tracked(target: Path) -> None:
    names = (
        subprocess.run(["git", "ls-files", "-z"], cwd=fetch.REPO, check=True, capture_output=True)
        .stdout.decode()
        .split("\0")
    )
    for name in filter(None, names):
        assert name.split("/")[0] not in {"data", ".venv"}
        source = fetch.REPO / name
        if source.is_file():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
