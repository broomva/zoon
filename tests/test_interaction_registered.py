"""Committed materials are bound to the training file and reproduce its fits."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.test_interaction_training import require_training
from zoon.interaction import data, kernel, person, registered, score


def test_amounts_regenerate_byte_for_byte() -> None:
    require_training()
    expected = registered.amounts_bytes(data.load(), registered.training_digest())
    assert registered.AMOUNTS.read_bytes() == expected
    assert registered.read_amounts() == kernel.Amounts.build(data.load())
    assert expected.endswith(b"\n")


def test_committed_fits_reproduce_all_training_probabilities_and_log_likelihoods() -> None:
    require_training()
    groups = data.load()
    committed, refitted = registered.read_fitted(), registered.fit(groups)
    assert {name: len(law.numbers) for name, law in committed.items()} == {"zoon": 20, "alone": 18}
    for arm, law in committed.items():
        fresh = refitted[arm]
        predictions = score.coded(law, groups)
        np.testing.assert_allclose(
            predictions.probabilities,
            score.coded(fresh, groups).probabilities,
            atol=registered.PROBABILITY_TOLERANCE,
            rtol=0,
        )
        np.testing.assert_allclose(
            predictions.logs(person.FLOOR), person.log_likelihoods(law, groups), atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("name", ["fitted", "amounts"])
def test_one_changed_digit_is_rejected(name: str, tmp_path: Path) -> None:
    source = registered.FITTED if name == "fitted" else registered.AMOUNTS
    content = source.read_bytes()
    # A single final fractional digit is far below the fit tolerance, but is
    # still a corrupted registered material. Change one byte, not a parsed value.
    field = b'"numbers": [' if name == "fitted" else b'"rows": ['
    start = content.index(field) + len(field)
    match = re.search(rb"\d+\.\d+" if name == "fitted" else rb"\d+", content[start:])
    assert match
    position = start + match.end() - 1
    replacement = b"0" if content[position : position + 1] != b"0" else b"1"
    changed = content[:position] + replacement + content[position + 1 :]
    assert sum(a != b for a, b in zip(content, changed, strict=True)) == 1
    altered = tmp_path / source.name
    altered.write_bytes(changed)
    reader = registered.read_fitted if name == "fitted" else registered.read_amounts
    with pytest.raises(ValueError, match="checksum"):
        reader(altered)


@pytest.mark.parametrize("name", ["fitted", "amounts"])
def test_wrong_training_digest_is_rejected_even_with_valid_checksum(
    name: str, tmp_path: Path
) -> None:
    source = registered.FITTED if name == "fitted" else registered.AMOUNTS
    document = json.loads(source.read_bytes())
    document.pop("sha256")
    document["training_sha256"] = "0" * 64
    altered = tmp_path / source.name
    altered.write_bytes(registered.sealed(document))
    reader = registered.read_fitted if name == "fitted" else registered.read_amounts
    with pytest.raises(ValueError, match="training digest"):
        reader(altered)


def test_material_digest_is_the_real_training_digest() -> None:
    require_training()
    assert hashlib.sha256(data.TRAINING.read_bytes()).hexdigest() == registered.training_digest()


def test_fitted_reader_retains_exact_committed_numbers() -> None:
    assert (
        registered.fitted_bytes(registered.read_fitted(), registered.training_digest())
        == registered.FITTED.read_bytes()
    )


@pytest.mark.parametrize("name", ["fitted", "amounts"])
@pytest.mark.parametrize("change", ["newline", "space", "compact", "duplicate"])
def test_material_bytes_must_be_canonical(name: str, change: str, tmp_path: Path) -> None:
    source = registered.FITTED if name == "fitted" else registered.AMOUNTS
    raw = source.read_bytes()
    if change == "newline":
        raw = raw[:-1]
    elif change == "space":
        raw += b" "
    elif change == "compact":
        raw = json.dumps(json.loads(raw), separators=(",", ":")).encode()
    else:
        key = "arms" if name == "fitted" else "rows"
        raw = raw.replace(
            b"{",
            ("{\n" + json.dumps(key) + ": " + json.dumps(json.loads(raw)[key]) + ",").encode(),
            1,
        )
    path = tmp_path / source.name
    path.write_bytes(raw)
    reader = registered.read_fitted if name == "fitted" else registered.read_amounts
    with pytest.raises(ValueError, match="checksum"):
        reader(path)


@pytest.mark.parametrize(
    "change",
    [
        "arms",
        "form",
        "count",
        "missing",
        "string",
        "boolean",
        "fraction",
        "extra",
        "row_extra",
        "twin_type",
        "count_type",
        "first_fraction",
        "negative_fraction",
    ],
)
def test_fitted_refuses_invalid_schema(change: str, tmp_path: Path) -> None:
    document = json.loads(registered.FITTED.read_bytes())
    document.pop("sha256")
    row = document["arms"]["zoon"]
    if change == "arms":
        document["arms"]["other"] = document["arms"].pop("alone")
    elif change == "form":
        row["letter"] = "C"
    elif change == "count":
        row["count"] -= 1
    elif change == "twin_type":
        row["twin"] = int(row["twin"])
    elif change == "count_type":
        row["count"] = float(row["count"])
    elif change == "first_fraction":
        row["numbers"][-len(registered.FORMS["zoon"].traces)] = 1.01
    elif change == "negative_fraction":
        row["numbers"][-1] = -0.01
    elif change == "missing":
        del row["numbers"]
    elif change == "string":
        row["numbers"][0] = str(row["numbers"][0])
    elif change == "boolean":
        row["numbers"][0] = True
    elif change == "fraction":
        row["numbers"][-1] = 7.5
    elif change == "extra":
        document["extra"] = 0
    else:
        row["extra"] = 0
    path = tmp_path / "fitted.json"
    path.write_bytes(registered.sealed(document))
    with pytest.raises(ValueError, match=r"material|primary|fraction"):
        registered.read_fitted(path)


@pytest.mark.parametrize(
    "change",
    [
        "unsorted",
        "missing",
        "duplicate",
        "string",
        "boolean",
        "values",
        "extra",
        "key",
        "band_boolean",
        "empty",
    ],
)
def test_amounts_refuse_invalid_schema(change: str, tmp_path: Path) -> None:
    document = json.loads(registered.AMOUNTS.read_bytes())
    document.pop("sha256")
    rows = document["rows"]
    if change == "unsorted":
        rows.reverse()
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "string":
        rows[0][2][0] = str(rows[0][2][0])
    elif change == "boolean":
        rows[0][0] = True
    elif change == "band_boolean":
        rows[0][1] = True
    elif change == "empty":
        rows[0][2] = []
    elif change == "values":
        rows[0][2].reverse()
    elif change == "extra":
        document["extra"] = 0
    else:
        del document["rows"]
    path = tmp_path / "amounts.json"
    path.write_bytes(registered.sealed(document))
    with pytest.raises(ValueError, match=r"material|amount"):
        registered.read_amounts(path)


def test_write_materials_writes_each_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_interaction_score import synthetic

    groups = synthetic()
    laws = registered.read_fitted()

    def fit(_: object) -> dict[str, person.Person]:
        return laws

    monkeypatch.setattr(registered, "fit", fit)
    monkeypatch.setattr(registered, "FITTED", tmp_path / "fitted.json")
    monkeypatch.setattr(registered, "AMOUNTS", tmp_path / "amounts.json")
    digest = registered.training_digest()
    registered.write_materials(groups, digest)
    assert registered.FITTED.read_bytes() == registered.fitted_bytes(laws, digest)
    assert registered.AMOUNTS.read_bytes() == registered.amounts_bytes(groups, digest)


def test_writer_refuses_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        registered.encoded({"number": float("nan")})


@pytest.mark.parametrize("name", ["fitted", "amounts"])
def test_material_reader_refuses_missing_checksum(name: str, tmp_path: Path) -> None:
    source = registered.FITTED if name == "fitted" else registered.AMOUNTS
    document = json.loads(source.read_bytes())
    document.pop("sha256")
    path = tmp_path / source.name
    path.write_bytes(registered.encoded(document))
    reader = registered.read_fitted if name == "fitted" else registered.read_amounts
    with pytest.raises(ValueError, match="invalid material document"):
        reader(path)
