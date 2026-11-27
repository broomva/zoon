"""The second seal is exercised on synthetic sources only, under live temporary tags."""

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.interaction_run_support import commit, git, repository
from tests.synthetic_groups import FakeMatio, write_raw
from zoon.interaction import data, guard, registered, split
from zoon.interaction.games import GAMES
from zoon.interaction.guard import Guard, digest
from zoon.pilot import clients


def sealed_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, damage: str = "", small: bool = False
) -> tuple[Path, dict[tuple[str, str], Any]]:
    root = repository(tmp_path / "repo")
    original = json.loads(data.TRAINING.read_bytes())
    counts = {
        c: sum(g["cell"] == c for g in original["groups"]) + original["held_out"][c] for c in GAMES
    }
    if small:
        counts = dict.fromkeys(GAMES, 11)
    directory = root / "data" / "groups"
    raw = directory / "raw"
    held = {(c, g) for c, n in counts.items() for g in split.held_out(c, map(str, range(1, n + 1)))}
    damaged = next(k for k in sorted(held) if k[0].startswith("4-" if damage == "shape" else "2-"))
    intended = write_raw(raw, counts, frozenset({damaged}) if damage == "text" else frozenset())
    if damage == "shape":
        path = raw / data.FOURS_FILE
        frame = pd.read_pickle(path)
        keep = ~(
            (frame.Treatment == damaged[0][2:])
            & (frame.GroupID == int(damaged[1]))
            & (frame.PlayerID == 4)
            & (frame.Session == 1)
        )
        frame[keep].to_pickle(path)
    manifest = root / "interaction" / "data-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    n: {"size": (raw / n).stat().st_size, "sha256": digest((raw / n).read_bytes())}
                    for n in data.FILES
                }
            }
        )
    )
    monkeypatch.setattr(data, "MANIFEST", manifest)
    monkeypatch.setitem(sys.modules, "matio", FakeMatio)
    training = directory / "training.json"
    data.seal(raw, training)
    sha = registered.training_digest(manifest)
    fitted_path = root / "interaction" / "fitted.json"
    fitted = json.loads(fitted_path.read_bytes())
    fitted.pop("sha256")
    fitted["training_sha256"] = sha
    fitted_path.write_bytes(registered.sealed(fitted))
    (root / "interaction" / "amounts.json").write_bytes(
        registered.amounts_bytes(data.load(training), sha)
    )
    commit(root)
    git(root, "tag", "-f", guard.TAG)
    return root, intended


def ready(gate: Guard, *, digests: bool = True) -> None:
    log = gate.storage / "simulation.jsonl"
    log.write_bytes(b"synthetic simulation\n")
    for step in guard.ORDER[:3]:
        gate.before(step)
        gate.save(
            step,
            {"digests": {log.name: digest(log.read_bytes())}}
            if step == "closed-loop" and digests
            else {},
            families=tuple(gate.transcripts.keys() | gate._readers.keys()),
            status="completed",
        )
    gate.before("held-out")


def test_second_seal_returns_exact_pinned_groups_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, intended = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        found = data.open_held_out(permit)
        expected = {
            (cell, split.group_index(cell, raw)): rounds
            for (cell, raw), rounds in intended.items()
            if raw in split.held_out(cell, [g for c, g in intended if c == cell])
        }
        assert len(found) == 143
        assert [(g.cell, g.index) for g in found] == sorted((g.cell, g.index) for g in found)
        assert {(g.cell, g.index): g.rounds for g in found} == expected
        assert {c: sum(g.cell == c for g in found) for c in GAMES} == data.held_out_counts(
            root / "data" / "groups" / "training.json"
        )
    with pytest.raises(clients.GuardRefusal):
        data.open_held_out(permit)


@pytest.mark.parametrize("damage", ["text", "shape"])
def test_unreadable_held_out_group_refuses_every_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch, damage=damage)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        with pytest.raises(data.RecordFault):
            data.open_held_out(gate.permit())


def test_seal_refuses_wrong_counts_and_changed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        training = root / "data" / "groups" / "training.json"
        original_training = training.read_bytes()
        document = json.loads(original_training)
        document["held_out"]["2-FE"] -= 1
        training.write_text(json.dumps(document))
        manifest = root / "interaction" / "data-manifest.json"
        pinned = json.loads(manifest.read_bytes())
        pinned["training"]["sha256"] = digest(training.read_bytes())
        manifest.write_text(json.dumps(pinned))
        with pytest.raises(ValueError, match="does not match"):
            data.open_held_out(permit)
        training.write_bytes(original_training)
        source = root / "data" / "groups" / "raw" / data.PAIRS_FILE
        source.write_bytes(b"changed")
        with pytest.raises(ValueError, match="does not match"):
            data.open_held_out(permit)


def test_simulation_digests_and_live_mode_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    for stub, digests in ((False, False), (True, True)):
        gate = Guard(root, stub=stub)
        with gate.lock():
            ready(gate, digests=digests)
            with pytest.raises(clients.GuardRefusal):
                gate.permit()
    with pytest.raises(clients.GuardRefusal):
        guard.HeldOutPermit(object(), Guard(root))


def test_no_source_module_can_call_the_second_seal() -> None:
    """The later held-out driver will be the first allowed caller outside data."""
    restricted = {
        "open_held_out",
        "_pairs",
        "_fours",
        "HeldOutPermit",
        "_AUTHORITY",
        "_SECOND_SEAL",
    }
    for path in (data.REPO / "src").rglob("*.py"):
        names = restricted.copy()
        if path == Path(data.__file__):
            names = {"_AUTHORITY"}
        elif path == Path(guard.__file__):
            names = {"open_held_out", "_pairs", "_fours", "_SECOND_SEAL"}
        tree = ast.parse(path.read_text())
        references = [
            n
            for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id in names)
            or (
                isinstance(n, ast.Attribute)
                and (n.attr in names or (n.attr == "permit" and path != Path(guard.__file__)))
            )
            or (
                isinstance(n, ast.ImportFrom)
                and any(a.name in names or a.name == "*" for a in n.names)
            )
        ]
        assert not references, path.name


def test_permit_checks_the_recorded_simulation_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        (gate.storage / "simulation.jsonl").write_bytes(b"changed")
        with pytest.raises(clients.GuardRefusal, match="simulation digest"):
            gate.permit()


def test_second_seal_requires_the_registered_total_even_with_consistent_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch, small=True)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        with pytest.raises(data.RecordFault, match="pinned split"):
            data.open_held_out(gate.permit())


@pytest.mark.parametrize("damage", ["duplicate", "training"])
def test_seal_refuses_duplicate_indices_and_training_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from dataclasses import replace

    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        module: Any = data
        pairs = module._pairs

        def changed(*args: Any, **kwargs: Any) -> Any:
            groups, counts = pairs(*args, **kwargs)
            if damage == "duplicate":
                groups[0] = groups[1]
            else:
                training = data.load(root / "data" / "groups" / "training.json")
                index = next(g.index for g in training if g.cell == groups[0].cell)
                groups[0] = replace(groups[0], index=index)
            return groups, counts

        monkeypatch.setattr(data, "_pairs", changed)
        with pytest.raises(data.RecordFault, match="pinned split"):
            data.open_held_out(gate.permit())


@pytest.mark.parametrize("reader", ["_pairs", "_fours"])
def test_private_readers_refuse_foreign_and_direct_held_out_permits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    from types import SimpleNamespace

    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    module: Any = data
    read = getattr(module, reader)
    # A foreign permit must fail before even looking for the source.
    with pytest.raises(clients.GuardRefusal, match="live held-out"):
        read(tmp_path / "absent", object())
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        assert not hasattr(permit, "authority")
        with pytest.raises(clients.GuardRefusal, match="internal to the second seal"):
            read(tmp_path / "absent", permit)
        permit.guard = SimpleNamespace(active=True, mode="live", checked_step="held-out")
        with pytest.raises(clients.GuardRefusal, match="live held-out"):
            guard.require_permit(permit)


def test_only_held_out_step_can_issue_or_use_a_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        gate.checked_step = "recall"
        with pytest.raises(clients.GuardRefusal, match="only the live held-out"):
            gate.permit()
        with pytest.raises(clients.GuardRefusal, match="live held-out"):
            guard.require_permit(permit)


def test_simulation_digest_paths_cannot_leave_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"simulation")
    with gate.lock():
        for step in guard.ORDER[:3]:
            gate.before(step)
            gate.save(
                step,
                {"digests": {str(outside): digest(outside.read_bytes())}}
                if step == "closed-loop"
                else {},
                families=tuple(gate.transcripts.keys() | gate._readers.keys()),
                status="completed",
            )
        gate.before("held-out")
        with pytest.raises(clients.GuardRefusal, match="simulation digest"):
            gate.permit()


@pytest.mark.parametrize("filename", data.FILES)
def test_second_seal_uses_the_registered_manifest_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        source = root / "data/groups/raw" / filename
        if filename == data.PAIRS_FILE:
            import openpyxl

            book = openpyxl.load_workbook(source)
            sheet = book[data.PAIR_SHEETS["FE"]]
            sheet.cell(row=2, column=3, value=0)
            book.save(source)
            book.close()
        else:
            frame = pd.read_pickle(source)
            frame.loc[0, data.FOUR_VALUE] = 0
            frame.to_pickle(source)
        manifest = root / "interaction/data-manifest.json"
        document = json.loads(manifest.read_bytes())
        document["files"][filename].update(
            size=source.stat().st_size, sha256=digest(source.read_bytes())
        )
        manifest.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="does not match"):
            data.open_held_out(permit)


def test_held_out_permit_expires_on_the_next_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = sealed_repository(tmp_path, monkeypatch)
    gate = Guard(root)
    with gate.lock():
        ready(gate)
        permit = gate.permit()
        assert guard.require_permit(permit) is gate
    with gate.lock():
        gate.before("held-out")
        with pytest.raises(clients.GuardRefusal, match="live held-out"):
            guard.require_permit(permit)
        assert guard.require_permit(gate.permit()) is gate
