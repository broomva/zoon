"""Fit boundaries and feedback histories of the registered arms."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit

from tests.synthetic import simulate
from zoon.pilot import arms, fetch, lean, models, power


def test_all_arms_fit_training_targets_and_known_share_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = simulate(8, 0.2, 1, 21)
    constant = models.fit_constant
    learning = models.fit_learning
    lean_fit = models.fit_lean
    calls: list[tuple[str, tuple[int, ...]]] = []

    def record(targets: models.Bools, kind: str) -> None:
        omitted = tuple(d for d in range(13) if not targets[training.dilemma == d].any())
        assert omitted == () or len(omitted) == 2
        calls.append((kind, omitted))

    def fit_constant(
        data: models.Sequences, targets: models.Bools, dilemmas: int, dilemma_intercepts: bool
    ) -> models.Parameters:
        assert data is training and dilemma_intercepts
        record(targets, "constant")
        return constant(data, targets, dilemmas, dilemma_intercepts)

    def fit_learning(
        data: models.Sequences, targets: models.Bools, dilemmas: int, dilemma_intercepts: bool
    ) -> models.Parameters:
        assert data is training and dilemma_intercepts
        record(targets, "learning")
        return learning(data, targets, dilemmas, dilemma_intercepts)

    def fit_lean(
        data: models.Sequences,
        targets: models.Bools,
        sign: models.Floats,
        measure: models.Floats,
        moving: bool,
    ) -> models.LeanFit:
        assert data is training
        record(targets, "lean")
        return lean_fit(data, targets, sign, measure, moving)

    monkeypatch.setattr(models, "fit_constant", fit_constant)
    monkeypatch.setattr(models, "fit_learning", fit_learning)
    monkeypatch.setattr(models, "fit_lean", fit_lean)
    rotations = arms.fit_rotations(
        training,
        {"sign": np.zeros(13), "decision/zoon-evaluative": np.arange(13, dtype=np.float64)},
    )
    assert [list(r.held_out) for r in rotations] == power.held_out_sets(True, 13)
    assert set(rotations[0].arms) == {
        "pooled",
        "laws-only",
        "known share",
        "known share + learning",
        "sign",
        "sign + learning",
        "decision/zoon-evaluative",
        "decision/no-state-evaluative",
    }
    assert sum(not omitted for _, omitted in calls) == 26
    assert rotations[0].arms["laws-only"].count == 13
    assert rotations[0].arms["known share + learning"].count == 15
    assert rotations[0].arms["sign"].count == 1
    assert rotations[0].arms["decision/zoon-evaluative"].count == 4
    assert rotations[0].arms["decision/no-state-evaluative"].count == 2
    scored = simulate(4, 0.2, 1, 22)
    predicted = arms.predict(rotations, scored)
    assert all(len(r) == 13 for r in predicted.values())
    assert all(p.shape == (4, 13) for values in predicted.values() for p in values)


def test_feedback_on_held_dilemma_changes_later_predictions() -> None:
    training = simulate(1, 0, 0, 1)
    data = replace(
        training,
        choice=np.ones((1, 13), dtype=np.int64),
        dilemma=np.arange(13, dtype=np.int64)[None, :],
        feedback=np.ones((1, 13)),
    )
    fitted = arms.Fit(models.Parameters(np.zeros(13), 0.5, 2), 2)
    result = arms.predict([arms.Rotation((0, 1), {"learning": fitted})], data)["learning"][0]
    np.testing.assert_allclose(result[0, :3], [0.5, expit(1), expit(1.5)])
    changed = replace(data, feedback=np.zeros_like(data.feedback))
    np.testing.assert_allclose(
        arms.predict([arms.Rotation((0, 1), {"learning": fitted})], changed)["learning"][0], 0.5
    )


def test_reference_ratings_and_reflected_norm() -> None:
    ratings = arms.reference_ratings()
    assert set(ratings) == {"sign", "coded", "norm"}
    np.testing.assert_array_equal(ratings["sign"], np.zeros(13))
    assert np.ptp(ratings["coded"]) == pytest.approx(0)
    endow = fetch.DILEMMAS.index("Endow")
    from zoon.pilot import outside

    assert (
        ratings["norm"][endow] == 8 - outside.published()[outside.originals()["Endow"]].published[2]
    )
    assert lean.SIGN[endow] == 1
