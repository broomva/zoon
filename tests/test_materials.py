"""Source fidelity and the answer/condition-to-outcome mapping."""

import json
from pathlib import Path

import pytest

from zoon.pilot import fetch, materials


def test_committed_digest_and_inventory() -> None:
    assert materials.digest(materials.TEXTS.read_bytes()) == materials.DIGEST.read_text().strip()
    tasks = materials.load()
    assert tuple(tasks) == fetch.DILEMMAS
    document = json.loads(materials.TEXTS.read_text())
    assert document["source"] == materials.SOURCE
    for task in tasks.values():
        assert task["text"].endswith("?")
        assert set(task["outcomes"]) == {"Yes", "No"}
        assert all(
            set(outcomes) == {"success", "failure"} for outcomes in task["outcomes"].values()
        )
        assert all(text for outcomes in task["outcomes"].values() for text in outcomes.values())


def test_every_committed_text_occurs_in_pdf() -> None:
    if not materials.PDF.exists():
        pytest.skip("published materials PDF absent; run python -m zoon.pilot.materials fetch")
    columns = materials.extracted_columns()
    for name, task in materials.load().items():
        body, yes, no = columns[name]
        assert materials.normalise(task["text"]) in materials.normalise(body)
        for answer, column in (("Yes", yes), ("No", no)):
            for text in task["outcomes"][answer].values():
                assert materials.normalise(text) in materials.normalise(column), (name, answer)
    assert materials.serialise(materials.extract()) == materials.TEXTS.read_bytes()
    manifest = json.loads(materials.MANIFEST.read_text())
    entry = manifest["files"][materials.PDF.name]
    assert entry == {
        "url": materials.URL,
        "size": materials.PDF.stat().st_size,
        "sha256": materials.digest(materials.PDF.read_bytes()),
    }


@pytest.mark.parametrize("name", fetch.DILEMMAS)
@pytest.mark.parametrize("condition", fetch.CONDITIONS)
@pytest.mark.parametrize("answer", ["Yes", "No"])
def test_outcome_mapping(name: str, condition: str, answer: materials.Answer) -> None:
    cbr = (answer == "Yes") == fetch.CBR_IS_ACTION[name]
    succeeds = cbr == (condition == "cbr_success")
    expected = materials.load()[name]["outcomes"][answer]["success" if succeeds else "failure"]
    assert materials.outcome_text(name, answer, condition=condition) == expected
    assert materials.outcome_text(name, answer, success=succeeds) == expected
    with pytest.raises(ValueError, match="disagree"):
        materials.outcome_text(name, answer, condition=condition, success=not succeeds)


def test_invalid_outcome_inputs() -> None:
    with pytest.raises(ValueError, match="required"):
        materials.outcome_text("Ladder", "Yes")
    with pytest.raises(ValueError, match="condition"):
        materials.outcome_text("Ladder", "Yes", condition="other")
    with pytest.raises(ValueError, match="answer"):
        materials.outcome_text("Ladder", "Maybe", success=True)  # type: ignore[arg-type]


def test_normalisation() -> None:
    assert materials.normalise("some\nwhite-space and hyphen-\nation\u00ad") == materials.normalise(
        "some white space and hyphenation"
    )


def test_manifest_pins_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, manifest = tmp_path / materials.PDF.name, tmp_path / "manifest.json"
    source.write_bytes(b"published PDF")
    materials.fetch(tmp_path, manifest)
    before = manifest.read_bytes()
    materials.fetch(tmp_path, manifest)
    assert manifest.read_bytes() == before
    source.write_bytes(b"changed PDF")
    with pytest.raises(ValueError, match="manifest"):
        materials.fetch(tmp_path, manifest)


def test_missing_pdf_is_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    def download(url: str) -> io.BytesIO:
        assert url == materials.URL
        return io.BytesIO(b"PDF")

    monkeypatch.setattr(materials.urllib.request, "urlopen", download)
    materials.fetch(tmp_path / "materials", tmp_path / "manifest.json")
    assert (tmp_path / "materials" / materials.PDF.name).read_bytes() == b"PDF"
