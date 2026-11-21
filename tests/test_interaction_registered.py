"""Committed materials are bound to the training file and reproduce its fits."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.test_interaction_training import require_training
from zoon.interaction import data, person, registered, score


def test_amounts_regenerate_byte_for_byte() -> None:
    require_training()
    expected = registered.amounts_bytes(data.load(), registered.training_digest())
    assert registered.AMOUNTS.read_bytes() == expected
    assert registered.read_amounts().rows
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
            person.log_likelihoods(law, groups),
            person.log_likelihoods(fresh, groups),
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
