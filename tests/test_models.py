"""The models: the learning rule, the fit, and the per-dilemma intercepts."""

import numpy as np
import pytest

from tests.synthetic import DILEMMAS, simulate
from zoon.pilot.models import (
    Parameters,
    Sequences,
    fit_constant,
    fit_learning,
    log_likelihood,
    probabilities,
)


def _all(data: Sequences) -> np.ndarray:
    return np.ones(data.choice.shape, dtype=np.bool_)


def test_learning_update_matches_a_hand_computation() -> None:
    """Protects the update line in `models._run`:

        value[who, chosen] += learning_rate * error

    One participant, learning rate 0.5, temperature 2, by hand:

    trial  values (rule, cbr)  P(cbr)           choice  feedback  update
    1      0, 0                sigmoid(0)       cbr     +1.0      cbr -> 0.5
    2      0, 0.5              sigmoid(1)       cbr     -1.0      cbr -> -0.25
    3      0, -0.25            sigmoid(-0.5)    rule    +0.5      rule -> 0.25
    4      0.25, -0.25         sigmoid(-1)      rule

    Flipping the sign of the update, skipping it, or updating the unchosen option changes
    the probabilities on trials 2 to 4.
    """
    data = Sequences(
        choice=np.array([[1, 1, 0, 0]]),
        feedback=np.array([[1.0, -1.0, 0.5, 0.0]]),
        dilemma=np.array([[0, 1, 2, 3]]),
    )
    parameters = Parameters(np.zeros(DILEMMAS), learning_rate=0.5, temperature=2.0)

    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    expected = [0.5, sigmoid(1.0), 1 - sigmoid(-0.5), 1 - sigmoid(-1.0)]
    assert probabilities(parameters, data)[0] == pytest.approx(expected, abs=1e-12)


def test_fit_recovers_learning_rate_and_temperature() -> None:
    """Stated tolerance: with 2,000 simulated people, the learning rate is recovered within
    0.05 of 0.3 and the temperature within 0.3 of 3.0, with and without dilemma intercepts.
    Fitted intercepts are within 0.25 of the simulated ones."""
    plain = simulate(2000, learning_rate=0.3, temperature=3.0, seed=1)
    fit = fit_learning(plain, _all(plain), DILEMMAS, dilemma_intercepts=False)
    assert fit.learning_rate == pytest.approx(0.3, abs=0.05)
    assert fit.temperature == pytest.approx(3.0, abs=0.3)
    assert not fit.intercept.any()

    lean = np.linspace(-1.0, 1.0, DILEMMAS)
    leaning = simulate(2000, learning_rate=0.3, temperature=3.0, seed=2, intercept=lean)
    fit = fit_learning(leaning, _all(leaning), DILEMMAS, dilemma_intercepts=True)
    assert fit.learning_rate == pytest.approx(0.3, abs=0.05)
    assert fit.temperature == pytest.approx(3.0, abs=0.3)
    assert fit.intercept == pytest.approx(lean, abs=0.25)


def test_fit_is_at_least_as_good_as_a_grid_search() -> None:
    """The optimiser and its hand-written gradient reach the best point of a brute-force
    search over learning rate and temperature."""
    data = simulate(500, learning_rate=0.2, temperature=1.5, seed=3)
    fit = fit_learning(data, _all(data), DILEMMAS, dilemma_intercepts=False)

    def total(learning_rate: float, temperature: float) -> float:
        candidate = Parameters(np.zeros(DILEMMAS), learning_rate, temperature)
        return float(np.log(probabilities(candidate, data)).sum())

    grid = max(total(a, t) for a in np.linspace(0.01, 1, 34) for t in np.linspace(0.1, 6, 60))
    assert total(fit.learning_rate, fit.temperature) >= grid - 1e-6


def test_unseen_dilemma_takes_the_pooled_share() -> None:
    """A dilemma with no fitting target gets the share of choices over all fitting targets,
    as a logit, and not the mean of the fitted logits, which overshoots when leans are
    lopsided. The learning model adds the average shift of its fitted intercepts, and still
    learns from that dilemma's feedback."""
    lean = np.linspace(-0.5, 2.5, DILEMMAS)
    data = simulate(800, learning_rate=0.3, temperature=2.0, seed=4, intercept=lean)
    unseen = [2, 9]
    targets = ~np.isin(data.dilemma, unseen)
    seen = [d for d in range(DILEMMAS) if d not in unseen]
    share = float(data.choice[targets].mean())
    pooled = np.log(share / (1 - share))

    constant = fit_constant(data, targets, DILEMMAS, dilemma_intercepts=True)
    assert constant.intercept[unseen] == pytest.approx(pooled)
    assert np.ptp(constant.intercept[seen]) > 1.0
    assert abs(constant.intercept[seen].mean() - pooled) > 0.05

    learning = fit_learning(data, targets, DILEMMAS, dilemma_intercepts=True)
    shift = float((learning.intercept[seen] - constant.intercept[seen]).mean())
    assert learning.intercept[unseen] == pytest.approx(pooled + shift)

    fit = fit_learning(data, targets, DILEMMAS, dilemma_intercepts=True)
    forgotten = Sequences(data.choice, np.where(targets, data.feedback, 0.0), data.dilemma)
    assert not np.allclose(probabilities(fit, data), probabilities(fit, forgotten))


def test_constant_model_is_the_share_of_choices() -> None:
    data = simulate(300, learning_rate=0.0, temperature=0.0, seed=5, intercept=np.full(13, 0.8))
    fit = fit_constant(data, _all(data), DILEMMAS, dilemma_intercepts=False)
    share = data.choice.mean()
    expected = np.where(data.choice == 1, share, 1 - share)
    assert probabilities(fit, data) == pytest.approx(expected)
    assert fit.learning_rate == 0.0 and fit.temperature == 0.0


def test_log_likelihood_is_clamped() -> None:
    """A model that is certain and wrong scores log(0.001), not minus infinity, and a model
    that is certain and right scores log(0.999), not zero."""
    data = Sequences(np.array([[0, 1]]), np.zeros((1, 2)), np.array([[0, 1]]))
    certain = Parameters(np.full(DILEMMAS, 50.0), learning_rate=0.0, temperature=0.0)
    assert log_likelihood(certain, data)[0] == pytest.approx([np.log(1e-3), np.log(1 - 1e-3)])


@pytest.mark.parametrize("dilemma_intercepts", [False, True])
def test_every_choice_gets_a_probability(dilemma_intercepts: bool) -> None:
    data = simulate(200, learning_rate=0.3, temperature=3.0, seed=6)
    for fit in (
        fit_constant(data, _all(data), DILEMMAS, dilemma_intercepts),
        fit_learning(data, _all(data), DILEMMAS, dilemma_intercepts),
    ):
        chance = probabilities(fit, data)
        assert chance.shape == data.choice.shape
        assert ((chance > 0) & (chance < 1)).all()
        assert (log_likelihood(fit, data) >= np.log(1e-3)).all()
        assert (log_likelihood(fit, data) <= np.log(1 - 1e-3)).all()
