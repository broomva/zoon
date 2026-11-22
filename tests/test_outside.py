"""The outside data: the committed tables, the loader that keeps personal data out, and the
manifest."""

import dataclasses
import inspect
from pathlib import Path

import pytest

from tests.synthetic import OUTSIDE_PERSONAL_COLUMNS, SECRET, write_outside, write_raw
from zoon.pilot import fetch, outside, split


def _with_published(
    dilemmas: dict[str, outside.Dilemma], means: dict[int, dict[str, tuple[int, float]]]
) -> dict[str, outside.Dilemma]:
    """The committed table with its printed means replaced by given ones, to two decimals."""
    return {
        name: dataclasses.replace(
            dilemma,
            published={study: round(means[study][name][1], 2) for study in dilemma.columns},
        )
        for name, dilemma in dilemmas.items()
    }


def test_committed_tables_hold_what_the_analysis_assumes() -> None:
    """The original paper has 38 factual dilemmas with codes, and trolley-type dilemmas
    without. Each of the task's 13 maps to a different factual original, the task table
    agrees with the labels the analysis uses, and exactly one adaptation reversed the
    suggested action. Every published answer has a number.

    Protects `outside.published`, `originals`, `action_reversed` and the GRADES table: a
    missing row, a renamed original or an answer without a grade fails here."""
    dilemmas = outside.published()
    factual = {name for name, d in dilemmas.items() if d.codes is not None}
    assert len(factual) == 38
    assert {d.kind for name, d in dilemmas.items() if name not in factual} == {"trolley"}
    assert all(set(d.published) == set(d.columns) for d in dilemmas.values())
    assert all(set(dilemmas[name].columns) == {1, 2} for name in factual)

    originals = outside.originals()
    assert tuple(originals) == fetch.DILEMMAS
    assert len(set(originals.values())) == 13 and set(originals.values()) <= factual

    assert [name for name, turned in outside.action_reversed().items() if turned] == ["Endow"]
    for codes in (outside.original_codes(), outside.task_codes()):
        assert tuple(codes) == fetch.DILEMMAS
        for name, one in codes.items():
            assert (one.action_is_utilitarian == "Yes") == fetch.CBR_IS_ACTION[name]
    for name in factual:
        codes = dilemmas[name].codes
        assert codes is not None
        encoded = outside.encode(codes)
        assert encoded.shape == (len(outside.FEATURES),)
        assert ((encoded >= 0) & (encoded <= 1)).all()


def test_task_table_must_agree_with_the_labels_in_use(tmp_path: Path) -> None:
    """Protects the check in `action_reversed`: a task table that calls a rule option the
    utilitarian one is refused, not silently used to reflect a rating."""
    wrong = tmp_path / "task.csv"
    text = outside.TASK_TABLE.read_text(encoding="utf-8")
    assert "Animal rights,Animals,No,No,Yes" in text
    wrong.write_text(
        text.replace("Animal rights,Animals,No,No,Yes", "Animal rights,Animals,No,Yes,Yes"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CBR_IS_ACTION disagree"):
        outside.action_reversed(wrong)


def test_grades() -> None:
    """A hedge sits between yes and no, a qualification after the answer is dropped, and an
    answer nobody has graded is an error. Protects `outside.grade`."""
    assert outside.grade("Yes") == 1.0 and outside.grade("No") == 0.0
    assert outside.grade("No, self in victim-group") == 0.0
    assert outside.grade("Yes but will also benefit") == 1.0
    assert outside.grade("Yes, plus self potentially in victim group") == 1.0
    assert 0.0 < outside.grade("Probably no") < outside.grade("Maybe") < outside.grade("Yes")
    assert outside.grade("somewhat") == outside.grade("?") == 0.5
    with pytest.raises(ValueError, match="no grade"):
        outside.grade("Perhaps")


def test_ratings_are_the_counts_and_means_of_the_judgments(tmp_path: Path) -> None:
    """Protects the two lines in `outside.ratings` that add a judgment to a count and a sum,
    and the one that skips a dilemma a participant was not shown. A dilemma whose judgment
    is split over several columns is one dilemma."""
    intended = write_outside(tmp_path)
    rated = outside.ratings(tmp_path)
    assert set(rated) == set(outside.STUDIES)
    for study, by_name in rated.items():
        assert set(by_name) == set(intended[study])
        for name, rating in by_name.items():
            count, mean = intended[study][name]
            assert rating.count == count
            assert rating.mean == pytest.approx(mean)
    assert len(outside.published()["Fumes"].columns[2]) == 3


def test_personal_data_cannot_reach_the_ratings(tmp_path: Path) -> None:
    """The files carry personal columns and free-text comments filled with a marker. What
    comes back is a count and a mean per dilemma and nothing else."""
    write_outside(tmp_path)
    for study in outside.STUDIES.values():
        text = (tmp_path / study.filename).read_text(encoding="latin-1")
        assert all(column in text.splitlines()[0] for column in OUTSIDE_PERSONAL_COLUMNS)
        assert SECRET in text
    rated = outside.ratings(tmp_path)
    assert SECRET not in repr(rated)
    fields = {field.name for field in dataclasses.fields(outside.Rating)}
    assert fields == {"count", "mean"}
    assert list(inspect.signature(outside.ratings).parameters) == ["raw_dir", "dilemmas"]


@pytest.mark.parametrize(
    "column",
    [*OUTSIDE_PERSONAL_COLUMNS, "New_911.open", "New_911.absurd", "Filler_Heinz.filler"],
)
def test_only_judgment_columns_can_be_read(tmp_path: Path, column: str) -> None:
    """Protects the refusal in `outside.allowed_columns`: naming a personal column, a comment
    or any other rating as a dilemma's judgment is an error, not a leak."""
    write_outside(tmp_path)
    dilemmas = outside.published()
    dilemmas["911"] = dataclasses.replace(dilemmas["911"], columns={1: (column,), 2: (column,)})
    with pytest.raises(ValueError, match="only judgment columns may be read"):
        outside.ratings(tmp_path, dilemmas)
    for study in outside.STUDIES:
        allowed = outside.allowed_columns(study, outside.published())
        assert all(
            outside.JUDGMENT_COLUMN.fullmatch(name) for names in allowed.values() for name in names
        )


def test_a_cell_off_the_scale_is_an_error(tmp_path: Path) -> None:
    """Protects the scale check in `outside.ratings`: a judgment column holding anything but
    1 to 7 raises. Here the column named is real but belongs to another rating."""
    write_outside(tmp_path)
    study = outside.STUDIES[1]
    path = tmp_path / study.filename
    lines = path.read_text(encoding="latin-1").splitlines()
    header = lines[0].split(study.delimiter)
    judgment = header.index("New_911.dilemmaNew")
    filled = next(i for i, line in enumerate(lines[1:], 1) if line.split(";")[judgment])
    cells = lines[filled].split(study.delimiter)
    cells[judgment] = "9"
    lines[filled] = study.delimiter.join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    with pytest.raises(ValueError, match="holds '9'"):
        outside.ratings(tmp_path)


def test_means_are_checked_against_the_paper(tmp_path: Path) -> None:
    """Protects `outside.discrepancy` and the threshold in `check_against_paper`. Synthetic
    judgments do not reproduce the paper's means, so the check raises. With printed means
    equal to the synthetic ones to two decimals it passes, and the largest difference is
    within rounding. Moving one printed mean by 0.02 is caught and named."""
    intended = write_outside(tmp_path)
    rated = outside.ratings(tmp_path)
    with pytest.raises(ValueError, match="more than rounding allows"):
        outside.check_against_paper(rated, outside.published())

    matching = _with_published(outside.published(), intended)
    outside.check_against_paper(rated, matching)
    worst = outside.discrepancy(rated, matching)
    assert all(0 <= gap <= outside.ROUNDING + 1e-9 for gap, _ in worst.values())

    matching["Trolley"] = dataclasses.replace(
        matching["Trolley"],
        published={1: matching["Trolley"].published[1] + 0.02, 2: matching["Trolley"].published[2]},
    )
    gap, name = outside.discrepancy(rated, matching)[1]
    assert name == "Trolley" and gap == pytest.approx(0.02, abs=outside.ROUNDING + 1e-9)
    with pytest.raises(ValueError, match="study 1, Trolley"):
        outside.check_against_paper(rated, matching)


def test_fetch_records_then_verifies_hashes(tmp_path: Path) -> None:
    """THE MANIFEST CHECK for the outside files. With the files present nothing is
    downloaded. The first run records each file's URL, size and hash; a later run refuses a
    file that has changed. Protects the comparison with the manifest in `outside.fetch`."""
    raw_dir, manifest = tmp_path / "outside", tmp_path / "data-manifest-outside.json"
    write_outside(raw_dir, people=5)
    outside.fetch(raw_dir, manifest)
    recorded = manifest.read_text()
    assert all(study.filename in recorded for study in outside.STUDIES.values())
    assert all(study.url in recorded for study in outside.STUDIES.values())
    outside.fetch(raw_dir, manifest)
    assert manifest.read_text() == recorded

    changed = raw_dir / outside.STUDIES[2].filename
    changed.write_text(changed.read_text(encoding="latin-1") + "extra\n", encoding="latin-1")
    with pytest.raises(ValueError, match="does not match"):
        outside.fetch(raw_dir, manifest)


def test_committed_manifest_names_the_outside_files() -> None:
    """The committed manifest pins both outside files, and they are kept apart from the
    task's own raw files and under the ignored data directory."""
    recorded = outside.MANIFEST.read_text()
    assert all(study.filename in recorded for study in outside.STUDIES.values())
    assert outside.RAW_DIR != fetch.RAW_DIR
    assert outside.RAW_DIR.parent == fetch.RAW_DIR.parent == fetch.REPO / "data"
    assert "/data/" in (fetch.REPO / ".gitignore").read_text().split()


def test_outside_loader_stays_out_of_the_task_data(tmp_path: Path) -> None:
    """THE SEAL. The outside loader sits beside the task's raw files, in which every
    held-out record is unreadable and every personal cell is marked. It returns the same
    ratings as without them, and exports nothing that names the held-out set."""
    intended = write_outside(tmp_path)
    everyone = [(e, str(n)) for e in fetch.SOURCES for n in range(1, 41)]
    held_out = frozenset(key for key in everyone if split.assign(*key) == "held_out")
    write_raw(tmp_path, 40, unreadable=held_out)
    rated = outside.ratings(tmp_path)
    for study, by_name in rated.items():
        assert {name: rating.count for name, rating in by_name.items()} == {
            name: count for name, (count, _) in intended[study].items()
        }
    assert not {name for name in vars(outside) if "held" in name.lower()}
    source = Path(inspect.getfile(outside)).read_text()
    assert "SOURCES" not in source and "fetch.load" not in source and "_scan" not in source
