"""The power calculation: negative controls, the splits, and the power arithmetic."""

import numpy as np
import pytest

from tests.synthetic import DILEMMAS, simulate
from zoon.pilot import power
from zoon.pilot.models import Sequences

# "Within noise of zero" for the negative controls, in nats per choice. The positive control
# below must exceed forty times this.
NOISE = 0.001


@pytest.mark.parametrize("by_dilemma", [False, True])
@pytest.mark.parametrize("dilemma_intercepts", [False, True])
def test_no_learning_in_the_data_gives_no_advantage(
    by_dilemma: bool, dilemma_intercepts: bool
) -> None:
    """NEGATIVE CONTROL. Data simulated with learning rate 0: the learning model's
    cross-validated advantage over the constant model is within noise of zero."""
    lean = np.linspace(-1.0, 1.0, DILEMMAS) if dilemma_intercepts else None
    data = simulate(600, learning_rate=0.0, temperature=3.0, seed=11, intercept=lean)
    effect = power.summarise(power.cross_validate(data, by_dilemma, dilemma_intercepts))
    assert abs(effect.mean) < NOISE


def test_shuffling_feedback_removes_most_of_the_advantage() -> None:
    """NEGATIVE CONTROL. Data simulated with learning: the advantage is clearly positive,
    and permuting each participant's feedback across their trials removes most of it.

    It does not remove all of it, and should not be expected to. Shuffling within a person
    keeps how often they chose each option and how good their feedback was overall, and those
    two together still say a little about which way the person drifted. The test therefore
    asserts that less than a quarter of the advantage survives, not that none does.
    """
    data = simulate(1500, learning_rate=0.3, temperature=1.0, seed=12)
    real = power.summarise(power.cross_validate(data, False, False))
    control = power.summarise(power.cross_validate(power.shuffle_feedback(data), False, False))
    assert real.interval[0] > 40 * NOISE
    assert control.mean < 0.25 * real.mean

    shuffled = power.shuffle_feedback(data)
    assert np.array_equal(shuffled.choice, data.choice)
    assert np.allclose(np.sort(shuffled.feedback, axis=1), np.sort(data.feedback, axis=1))
    assert not np.array_equal(shuffled.feedback, data.feedback)


def test_learning_shows_only_after_feedback() -> None:
    """On trial 1 nobody has seen an outcome, so the nested models predict the same thing.
    The advantage appears later."""
    lean = np.linspace(-1.0, 1.0, DILEMMAS)
    data = simulate(1500, learning_rate=0.3, temperature=1.0, seed=13, intercept=lean)
    by_trial = power.cross_validate(data, False, True).mean(axis=0)
    assert abs(by_trial[0]) < NOISE
    assert by_trial[7:].mean() > 20 * NOISE


@pytest.mark.parametrize("by_dilemma", [False, True])
def test_every_trial_is_scored_out_of_sample(by_dilemma: bool) -> None:
    """Each participant-trial gets a finite score under both splits, and the result is the
    same when the run is repeated."""
    data = simulate(150, learning_rate=0.3, temperature=1.0, seed=14)
    gain = power.cross_validate(data, by_dilemma, True)
    assert gain.shape == data.choice.shape
    assert np.isfinite(gain).all()
    assert np.array_equal(gain, power.cross_validate(data, by_dilemma, True))


def test_dilemma_split_matches_an_independent_computation() -> None:
    """Checks what holding out a dilemma means, on trial 1, where the answer can be written
    down. The learning model without intercepts predicts 0.5 on trial 1 whatever its
    parameters. The constant model predicts the share of cost-benefit choices among the
    fitting participants on the dilemmas that are not held out. Each dilemma is held out in
    two sets, with the neighbour on either side, and the two scores are averaged.

    This fails if held-out dilemmas enter the fit, if dilemmas that are not held out are
    scored, or if the scoring participant's fold is used for fitting."""
    lean = np.where(np.arange(DILEMMAS) < 4, 3.0, -1.0)
    data = simulate(100, learning_rate=0.3, temperature=1.0, seed=17, intercept=lean)
    gain = power.cross_validate(data, by_dilemma=True, dilemma_intercepts=False)
    fold = power.folds(len(data.choice))
    for person in range(len(data.choice)):
        first, chose_cbr = data.dilemma[person, 0], data.choice[person, 0] == 1
        fitting = fold != fold[person]
        scores: list[float] = []
        for held_out in ({first, (first + 1) % DILEMMAS}, {first, (first - 1) % DILEMMAS}):
            kept = ~np.isin(data.dilemma[fitting], list(held_out))
            share = data.choice[fitting][kept].mean()
            scores.append(np.log(0.5) - np.log(share if chose_cbr else 1 - share))
        assert gain[person, 0] == pytest.approx(np.mean(scores), abs=1e-9)


def test_a_participant_is_never_scored_by_a_model_fitted_on_their_fold() -> None:
    """Changing one participant's choices cannot change the score of anyone else in the same
    fold, because their fold is not in the fit. It does change the scores of other folds."""
    data = simulate(60, learning_rate=0.3, temperature=1.0, seed=16)
    fold = power.folds(len(data.choice))
    changed = Sequences(data.choice.copy(), data.feedback, data.dilemma)
    changed.choice[0] = 1 - changed.choice[0]
    before = power.cross_validate(data, False, True)
    after = power.cross_validate(changed, False, True)
    same_fold = fold == fold[0]
    same_fold[0] = False
    assert np.array_equal(before[same_fold], after[same_fold])
    assert not np.allclose(before[fold != fold[0]], after[fold != fold[0]])


def test_power_arithmetic() -> None:
    """No effect is detected at the rate alpha; the detectable effect has the target power
    and matches the normal approximation (1.960 + 0.842) * sd / sqrt(n) for large n; a huge
    effect has power one, not a numerical failure."""
    assert power.power_to_detect(0.0, 1.0, 500) == pytest.approx(power.ALPHA, abs=1e-9)
    detectable = power.detectable_effect(0.05, 680)
    assert power.power_to_detect(detectable, 0.05, 680) == pytest.approx(0.80, abs=1e-6)
    assert detectable == pytest.approx(2.8016 * 0.05 / np.sqrt(680), rel=0.01)
    assert power.power_to_detect(1.0, 0.05, 680) == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    values = np.random.default_rng(15).normal(0.01, 0.05, 400)
    low, high = power.bootstrap_interval(values)
    assert (low, high) == power.bootstrap_interval(values)
    assert low < values.mean() < high
    assert high - low == pytest.approx(2 * 1.96 * 0.05 / np.sqrt(400), rel=0.15)
