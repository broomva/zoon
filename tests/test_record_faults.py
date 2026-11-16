"""Only malformed held-out records end the run under the complete-records rule."""

import csv
from pathlib import Path

import pytest

from tests.synthetic import write_raw
from tests.test_run import COMMIT
from tests.test_run import runner as runner
from zoon.pilot import clients, cohorts, fetch, run, split


@pytest.mark.parametrize("fault", ["short", "nonfinite", "unreadable"])
def test_held_out_record_fault_is_terminal(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    raw = runner.root / "data/raw"
    intended = write_raw(raw, 40)
    experiment, label = next(k for k in intended if split.assign(*k) == "held_out")
    source = fetch.SOURCES[experiment]
    path = raw / source.filename
    with path.open(newline="", encoding="latin-1") as handle:
        reader = csv.DictReader(handle, delimiter=source.delimiter)
        fields = reader.fieldnames
        assert fields is not None
        rows = list(reader)
    row = next(r for r in rows if r[""] == label)
    if fault == "short":
        column = next(c for c in source.order_columns if row[c])
        blocks = row[column].split("|")
        blocks.remove(next(b for b in blocks if b != "AttnCheck"))
        row[column] = "|".join(blocks)
    else:
        column = next(c for c in fields if "_V_" in c and row[c])
        row[column] = "nan" if fault == "nonfinite" else "unreadable"
    with path.open("w", newline="", encoding="latin-1") as handle:
        writer = csv.DictWriter(handle, fields, delimiter=source.delimiter)
        writer.writeheader()
        writer.writerows(rows)
    # Use only these synthetic files, with the normal execution boundary.
    monkeypatch.undo()
    loader = fetch.load_registered_held_out
    monkeypatch.setattr(run, "git", lambda *args: COMMIT)
    monkeypatch.setattr(run, "changed_files", lambda *args: [])
    for step in run.STEPS[:3]:
        runner.save(step, {})
    monkeypatch.setattr(runner, "guard", lambda *args: None)
    monkeypatch.setattr(fetch, "load_registered_held_out", loader)
    runner.synthetic = False
    assert "stopped" in runner.execute("held-out", COMMIT)[0]
    assert runner.state("held-out")["reason"] == clients.StopReason.RECORDS.value
    runner.synthetic = True
    assert "no verdict: complete-records failure" in runner.execute("report")[0]
    assert (runner.directory / "result.md").exists()


def test_training_record_fault_is_not_a_terminal_stop(tmp_path: Path) -> None:
    intended = write_raw(tmp_path, 40)
    identity = next(k for k in intended if split.assign(*k) == "train")
    write_raw(tmp_path, 40, unreadable=frozenset({identity}))
    with pytest.raises(ValueError) as error:
        cohorts.participants(fetch.load(tmp_path))
    assert not isinstance(error.value, clients.TerminalStop)
