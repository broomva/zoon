"""Only malformed held-out records end the run under the complete-records rule."""

import csv
from pathlib import Path

import pytest

from tests.synthetic import write_raw
from tests.test_run import COMMIT
from tests.test_run import runner as runner
from zoon.pilot import clients, cohorts, fetch, run, split
from zoon.pilot.fetch import load_registered_held_out


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

    def git(*args: object) -> str:
        return COMMIT

    def changed(*args: object) -> list[str]:
        return []

    def guarded(*args: object) -> None:
        pass

    monkeypatch.setattr(run, "git", git)
    monkeypatch.setattr(run, "changed_files", changed)
    for step in run.STEPS[:3]:
        runner.save(step, {})
    monkeypatch.setattr(runner, "guard", guarded)
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


@pytest.mark.parametrize("kind", ["held-out", "excluded", "training"])
def test_short_raw_row_only_stops_for_analysed_participants(
    runner: run.Run, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    for step in run.STEPS[:3]:
        assert "completed" in runner.execute(step)[0]
    training = runner.training()
    raw = runner.root / "data/raw"
    intended = write_raw(raw, 40)
    source = fetch.SOURCES[1]
    path = raw / source.filename
    with path.open(newline="", encoding="latin-1") as handle:
        rows = list(csv.reader(handle, delimiter=source.delimiter))
    header = rows[0]
    for index, row in enumerate(rows[1:], 1):
        identity = (1, row[0])
        matches = (
            identity not in intended
            if kind == "excluded"
            else identity in intended
            and split.assign(*identity) == ("held_out" if kind == "held-out" else "train")
        )
        if matches:
            gate_end = max(header.index(name) for name in ("", *source.exclusion_columns)) + 1
            rows[index] = row[:gate_end]
            break
    else:
        raise AssertionError("synthetic row not found")
    with path.open("w", newline="", encoding="latin-1") as handle:
        csv.writer(handle, delimiter=source.delimiter).writerows(rows)
    monkeypatch.setattr(fetch, "load_registered_held_out", load_registered_held_out)
    monkeypatch.setattr(runner, "guard", lambda *args: None)
    monkeypatch.setattr(runner, "training", lambda: training)
    monkeypatch.setattr(
        run.budget,
        "HELD_OUT_PARTICIPANTS",
        sum(split.assign(*identity) == "held_out" for identity in intended),
    )
    runner.synthetic = False
    lines = runner.execute("held-out", COMMIT)
    state = runner.state("held-out")
    if kind == "held-out":
        assert "stopped" in lines[0]
        assert state["reason"] == clients.StopReason.RECORDS.value
    else:
        assert "completed" in lines[0]
        assert state["reason"] == ""
