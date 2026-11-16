"""Small arithmetic examples for every stage-two numerical transform."""

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.synthetic import simulate
from zoon.pilot import clients, judge, lean, outside, power, prompted, prompts, score
from zoon.pilot.models import Floats


def test_evaluative_and_strict_features() -> None:
    r = clients.Response({str(i): 1 / 7 for i in range(1, 8)}, None, "", "v", 0, 0)
    assert judge.evaluative(r) == pytest.approx(4)
    assert (
        judge.evaluative(replace(r, probabilities={str(i): float(i == 6) for i in range(1, 8)}))
        == 6
    )
    yes = [replace(r, probabilities={"Yes": p, "No": 1 - p}) for p in (0.1, 0.2, 0.3, 0.4)]
    context = replace(r, probabilities=dict.fromkeys(outside.CONTEXTS, 1 / 9))
    actual = judge.features([*yes, context], False)
    np.testing.assert_allclose(actual, [0.1, 0, 0.2, 0.3, 0.4, *([1 / 9] * 9)])
    assert judge.features([*yes, context], True)[1] == 1
    with pytest.raises(ValueError):
        judge.features(yes, True)


def test_rotation_standardising_population_spread_and_sign_afterwards() -> None:
    raw = np.array([1.0, 3.0, 7.0])
    # Fit on 1 and 3: mean 2, population spread 1; the unseen 7 is five units above.
    np.testing.assert_array_equal(judge.standardise(raw, [2]), [-1, 1, 5])
    np.testing.assert_array_equal(lean.SIGN[:3] * judge.standardise(raw, [2]), [-1, 1, 5])
    np.testing.assert_array_equal(judge.standardise(np.array([2.0, 2.0, 7.0]), [2]), [0, 0, 0])
    np.testing.assert_array_equal(
        judge.standardise(np.array([2.0, 2.0 + 1e-10, 7.0]), [2]), [0, 0, 0]
    )
    with pytest.raises(clients.StepFailure):
        judge.standardise(np.array([math.nan, 2.0, 3.0]), [2])


def test_pearson_stop_rule_threshold_and_nonfinite() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.0, -1.0, 1.0])
    assert judge.calibration(x, y).correlation == pytest.approx(0.5)
    assert judge.calibration(x, y).passed
    assert not judge.calibration(x, -x).passed
    assert not judge.calibration(np.ones(3), y).passed
    assert not judge.calibration(np.array([1.0, math.nan, 3.0]), y).passed
    assert judge.correlation(x, y[:2]) is None


def test_strict_map_reuses_penalty_selection_and_infinite_map() -> None:
    features = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    target = np.array([1.0, 3.0, 5.0, 7.0])
    result = judge.fit_map(("a", "b", "c", "d"), features, target)
    assert result.fitted.penalty == math.inf
    assert result.fitted.offset == 4
    np.testing.assert_array_equal(result.fitted.weights, [0, 0])
    np.testing.assert_array_equal(result.fitted.predict(features), [4, 4, 4, 4])
    assert result.finite.penalty == lean.FINITE_PENALTIES[0]
    assert result.finite_error == pytest.approx(math.sqrt(80 / 9))
    assert result.finite_correlation == pytest.approx(-1)
    predicted = np.zeros((2, len(outside.FEATURES)))
    published = np.ones_like(predicted)
    assert set(judge.agreement(predicted, published).values()) == {1}
    assert "action is the utilitarian option" not in judge.agreement(predicted, published)


def test_prompted_three_steps() -> None:
    yes = np.array([0.0, 1.0, 0.2, 0.2])
    sign = np.array([True, False, True, False])
    identity = prompted.Fit(1, 0, True)
    np.testing.assert_allclose(
        prompted.probabilities(yes, sign, identity), [0.001, 0.001, 0.2, 0.8]
    )
    np.testing.assert_allclose(
        prompted.probabilities(yes, sign, prompted.Fit(-1, 0, True)), [0.999, 0.999, 0.8, 0.2]
    )
    shifted = prompted.probabilities(
        np.array([0.25]), np.array([True]), prompted.Fit(1, math.log(3), True)
    )
    np.testing.assert_allclose(shifted, [0.5])
    np.testing.assert_allclose(
        prompted.probabilities(yes, sign, prompted.Fit(10, 10, True))[:2], [0.001, 0.001]
    )


def test_prompted_fit_start_bounds_clamped_likelihood_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def minimize(fun: Any, x0: Floats, **kwargs: Any) -> Any:
        np.testing.assert_array_equal(x0, [1, 0])
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["bounds"] == [(-10, 10), (-10, 10)]
        value, gradient = fun(np.array([1.0, 0.0]))
        assert value == pytest.approx(-math.log(0.25) - math.log(0.75))
        np.testing.assert_allclose(gradient, [0.5 * math.log(3), -0.5])
        saturated, derivative = fun(np.array([10.0, 10.0]))
        assert math.isfinite(saturated)
        assert derivative.shape == (2,)
        return SimpleNamespace(success=False, x=np.array([9.0, 9.0]))

    monkeypatch.setattr(prompted, "minimize", minimize)
    assert prompted.fit(
        np.array([0.25, 0.25]), np.ones(2, dtype=np.bool_), np.array([1, 0])
    ) == prompted.Fit(1, 0, False)


def test_prompted_fit_can_choose_negative_slope() -> None:
    yes = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    fitted = prompted.fit(yes, np.ones(6, dtype=np.bool_), np.array([1, 1, 0, 0, 0, 1]))
    assert fitted.success and -10 <= fitted.a < 0 and -10 <= fitted.c <= 10
    np.testing.assert_allclose(
        prompted.probabilities(yes, np.ones(6, dtype=np.bool_), fitted),
        [2 / 3] * 3 + [1 / 3] * 3,
        atol=1e-5,
    )


def test_rotations_average_logs_not_probabilities_and_clamp() -> None:
    data = simulate(2, 0, 0, 1)
    p = [np.full(data.choice.shape, 0.5) for _ in range(13)]
    for i, pair in enumerate(power.held_out_sets(True, 13)):
        for d in pair:
            # Rotation d and its predecessor give different probabilities for d.
            p[i][data.dilemma == d] = 0.25 if i == d else 0.75
    actual = score.rotation_logs(p, data)
    np.testing.assert_allclose(actual, (math.log(0.25) + math.log(0.75)) / 2)
    certain = [np.ones(data.choice.shape) for _ in range(13)]
    np.testing.assert_allclose(
        score.rotation_logs(certain, data), np.log(np.where(data.choice, 0.999, 0.001))
    )
    p[0][0, np.flatnonzero(data.dilemma[0] == 0)[0]] = math.nan
    with pytest.raises(clients.TerminalStop, match="complete-records"):
        score.rotation_logs(p, data)
    with pytest.raises(clients.TerminalStop, match="complete-records"):
        score.rotation_logs(certain[:-1], data)
    with pytest.raises(clients.TerminalStop, match="complete-records"):
        score.complete(replace(data, choice=data.choice[:, :-1]))


def test_participant_ordering_averages_breakdowns_and_dilemma_interval() -> None:
    dilemma = np.tile(np.arange(13), (3, 1))
    first = np.array([[float(i)] * 13 for i in [1, 3, 2]])
    first[:, 0] += 13
    second = np.zeros_like(first)
    ids = [(2, "a"), (1, "b"), (1, "a")]
    result = score.compare(first, second, ids, dilemma)
    assert result is not None and result.mean == 3
    assert result.interval == power.bootstrap_interval(np.array([3.0, 4.0, 2.0]))
    later = score.compare(first, second, ids, dilemma, later=True)
    assert later is not None and later.mean == 2
    experiment = score.compare(first, second, ids, dilemma, experiment=1)
    assert experiment is not None and experiment.mean == 3.5
    assert score.compare(first, second, ids, dilemma, keep=[]) is None
    assert score.dilemma_interval(first, second, dilemma, []) is None
    assert score.dilemma_interval(first, second, dilemma) == power.bootstrap_interval(
        np.array([15.0] + [2.0] * 12)
    )
    filtered = score.compare(first, second, ids, dilemma, keep=list(range(1, 13)))
    assert filtered is not None and filtered.mean == 2


@pytest.mark.parametrize(
    "mean,low,expected",
    [
        (0.002, 0.0001, "pass"),
        (0.001999, 0.0001, "fail"),
        (0.1, 0.0, "fail"),
        (0.1, -0.1, "fail"),
        (math.nan, 0.1, "fail"),
    ],
)
def test_exact_pass_rule(mean: float, low: float, expected: str) -> None:
    assert score.verdict([score.rule(mean, (low, 0.2))] * 3) == expected


def test_bootstrap_registered_resamples_and_percentiles() -> None:
    assert power.RESAMPLES == 10000 and power.SEED == 20261115
    values = np.array([0.0, 2.0])
    assert power.bootstrap_interval(values) == (0.0, 2.0)
    with pytest.raises(ValueError):
        score.verdict([])


def test_evaluative_declared_labels() -> None:
    request = prompts.Decision("state", "question", prompts.declared(tuple(map(str, range(1, 8)))))
    assert math.isfinite(judge.evaluative(clients.StubClient("decision").ask(request)))
