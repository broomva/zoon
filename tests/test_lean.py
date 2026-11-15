"""The lean: the sign rule, the fit, the map from codes, and the nested comparisons."""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from tests.synthetic import DILEMMAS, simulate, write_raw
from zoon.pilot import fetch, lean, models, outside, power, report, split
from zoon.pilot.models import Floats, Parameters, Sequences, fit_lean, probabilities

# "Within noise of zero" for the negative controls, in nats per choice, as in test_power.
NOISE = 0.001
ALPHA, BETA = 0.4, 0.9
# A measure that varies over the dilemmas and is not the sign in disguise.
MEASURE: Floats = lean.standardise(np.random.default_rng(21).normal(size=DILEMMAS))
NOTHING: Floats = np.zeros(DILEMMAS)


def _generated(alpha: float, beta: float) -> Floats:
    """The lean the simulator is given, written out here and not taken from the model."""
    return lean.SIGN * alpha + lean.SIGN * beta * MEASURE


def _all(data: Sequences) -> np.ndarray:
    return np.ones(data.choice.shape, dtype=np.bool_)


def _effect(scores: lean.Scores, first: str, second: str) -> power.Effect:
    return power.summarise(scores.log_likelihood[first] - scores.log_likelihood[second])


def test_measure_is_not_the_sign() -> None:
    assert abs(np.corrcoef(MEASURE, lean.SIGN)[0, 1]) < 0.5
    assert lean.SIGN.tolist().count(1.0) == 8 and lean.SIGN.tolist().count(-1.0) == 5


def test_flipping_the_sign_flips_the_probability() -> None:
    """THE SIGN RULE. Protects the line in `models.lean`:

        return sign * (alpha + beta * measure)

    For the same alpha, beta and m, the probability of the cost-benefit option where the
    suggested action is the cost-benefit option, and where it is the rule option, sum to one.
    The form `alpha + beta * sign * measure`, which signs only the measure, fails this: the
    two would be sigmoid(1.03) and sigmoid(-0.23).
    """
    data = Sequences(np.array([[1]]), np.array([[0.0]]), np.array([[0]]))

    def chance(sign: float) -> float:
        intercept = models.lean(np.array([sign]), np.array([0.7]), alpha=0.4, beta=0.9)
        return float(probabilities(Parameters(intercept, 0.0, 0.0), data)[0, 0])

    assert chance(+1.0) == pytest.approx(expit(0.4 + 0.9 * 0.7))
    assert chance(-1.0) == pytest.approx(expit(-(0.4 + 0.9 * 0.7)))
    assert chance(+1.0) + chance(-1.0) == pytest.approx(1.0)


def test_fit_recovers_alpha_and_beta() -> None:
    """Stated tolerance: with 3,000 simulated people, alpha and beta are recovered within
    0.06, with and without learning in the data; the learning rate within 0.05 and the
    temperature within 0.3. Protects the two gradient lines for alpha and beta in
    `models.fit_lean`: with either sign wrong the fit does not move toward the truth."""
    still = simulate(3000, 0.0, 0.0, seed=31, intercept=_generated(ALPHA, BETA))
    fit = fit_lean(still, _all(still), lean.SIGN, MEASURE, learning=False)
    assert fit.alpha == pytest.approx(ALPHA, abs=0.06)
    assert fit.beta == pytest.approx(BETA, abs=0.06)
    assert fit.parameters.learning_rate == 0.0 and fit.parameters.temperature == 0.0
    assert fit.parameters.intercept == pytest.approx(lean.SIGN * (fit.alpha + fit.beta * MEASURE))

    moving = simulate(3000, 0.3, 3.0, seed=32, intercept=_generated(ALPHA, BETA))
    fit = fit_lean(moving, _all(moving), lean.SIGN, MEASURE, learning=True)
    assert fit.alpha == pytest.approx(ALPHA, abs=0.06)
    assert fit.beta == pytest.approx(BETA, abs=0.06)
    assert fit.parameters.learning_rate == pytest.approx(0.3, abs=0.05)
    assert fit.parameters.temperature == pytest.approx(3.0, abs=0.3)


def test_fit_is_at_least_as_good_as_a_grid_search() -> None:
    """The optimiser and its hand-written gradient reach the best point of a brute-force
    search over alpha and beta."""
    data = simulate(400, 0.0, 0.0, seed=33, intercept=_generated(ALPHA, BETA))
    fit = fit_lean(data, _all(data), lean.SIGN, MEASURE, learning=False)

    def total(alpha: float, beta: float) -> float:
        candidate = Parameters(lean.SIGN * (alpha + beta * MEASURE), 0.0, 0.0)
        return float(np.log(probabilities(candidate, data)).sum())

    grid = max(total(a, b) for a in np.linspace(0, 0.8, 41) for b in np.linspace(0.5, 1.3, 41))
    assert total(fit.alpha, fit.beta) >= grid - 1e-6


def test_fit_uses_the_targets_only() -> None:
    """Protects `residual = np.where(targets, ...)` in `models.fit_lean`. Choices on dilemmas
    outside the targets do not move alpha or beta; choices on a dilemma inside them do."""
    data = simulate(300, 0.0, 0.0, seed=34, intercept=_generated(ALPHA, BETA))
    targets = ~np.isin(data.dilemma, [2, 3])
    fit = fit_lean(data, targets, lean.SIGN, MEASURE, learning=False)

    def flipped(dilemma: int) -> Sequences:
        choice = np.where(data.dilemma == dilemma, 1 - data.choice, data.choice)
        return Sequences(choice, data.feedback, data.dilemma)

    outside_targets = fit_lean(flipped(2), targets, lean.SIGN, MEASURE, learning=False)
    assert (outside_targets.alpha, outside_targets.beta) == (fit.alpha, fit.beta)
    inside_targets = fit_lean(flipped(5), targets, lean.SIGN, MEASURE, learning=False)
    assert abs(inside_targets.alpha - fit.alpha) + abs(inside_targets.beta - fit.beta) > 0.02


def test_a_lean_with_signal_beats_the_pooled_share_on_held_out_dilemmas() -> None:
    """POSITIVE CONTROL. Data simulated with a known beta: on dilemmas and participants the
    models were not fitted to, the lean built from the measure beats the pooled share and
    beats the sign alone, and learning adds to it. On trial 1 learning adds nothing."""
    data = simulate(600, 0.3, 1.0, seed=35, intercept=_generated(ALPHA, BETA))
    scores = lean.cross_validate(data, lean.SIGN, {"sign": NOTHING, "measure": MEASURE})
    assert _effect(scores, "measure", lean.POOLED).interval[0] > 40 * NOISE
    assert _effect(scores, "measure", "sign").interval[0] > 40 * NOISE
    on_top = scores.log_likelihood["measure + learning"] - scores.log_likelihood["measure"]
    assert abs(on_top[:, 0].mean()) < NOISE
    assert on_top[:, 7:].mean() > 5 * NOISE

    fitted = scores.fitted["measure"]
    assert fitted.shape == (power.FOLDS * DILEMMAS, 4)
    assert fitted[:, 1].mean() == pytest.approx(BETA, abs=0.15)
    assert not fitted[:, 2:].any() and scores.fitted["measure + learning"][:, 3].all()


def test_a_measure_without_signal_gives_no_advantage() -> None:
    """NEGATIVE CONTROL. Data simulated with beta = 0.

    With alpha = 0 as well, the lean gives no advantage over the pooled share. With alpha
    not 0, people favour the suggested action whatever the dilemma: the sign alone then
    beats the pooled share, and the measure adds nothing to the sign. That second case is
    why `sign` is reported: an advantage over `pooled` is not by itself evidence that the
    measure carries anything.
    """
    measures = {"sign": NOTHING, "measure": MEASURE}
    flat = simulate(600, 0.0, 0.0, seed=36, intercept=_generated(0.0, 0.0))
    scores = lean.cross_validate(flat, lean.SIGN, measures)
    assert abs(_effect(scores, "measure", lean.POOLED).mean) < NOISE
    assert abs(_effect(scores, "measure", "sign").mean) < NOISE

    biased = simulate(600, 0.0, 0.0, seed=37, intercept=_generated(0.8, 0.0))
    scores = lean.cross_validate(biased, lean.SIGN, measures)
    assert abs(_effect(scores, "measure", "sign").mean) < NOISE
    assert _effect(scores, "sign", lean.POOLED).interval[0] > 40 * NOISE


def test_held_out_dilemmas_and_scoring_participants_stay_out_of_the_fit() -> None:
    """Checks the split against a computation written out here, for the two models that
    have a closed form.

    `sign` without learning has one number: the share of fitting choices that took the
    suggested action, over the dilemmas that are not held out. Each dilemma is held out in
    two sets, with the neighbour on either side, and the two scores are averaged. `known
    share` is the share on that same dilemma among the fitting participants.

    Protects `targets` and `scored` in `lean.cross_validate`. This fails if a held-out
    dilemma enters the fit, if the scoring participant's fold does, or if a dilemma that is
    not held out is scored.
    """
    strong = lean.SIGN * np.where(np.arange(DILEMMAS) < 4, 3.0, -1.0)
    data = simulate(100, 0.0, 0.0, seed=38, intercept=strong)
    scores = lean.cross_validate(data, lean.SIGN, {"sign": NOTHING})
    fold = power.folds(len(data.choice))
    took_action = (data.choice == 1) == (lean.SIGN[data.dilemma] == 1)
    for person in range(len(data.choice)):
        fitting = fold != fold[person]
        for trial in range(DILEMMAS):
            d = int(data.dilemma[person, trial])
            chose_action = bool(took_action[person, trial])
            parts: list[float] = []
            for held_out in ({d, (d + 1) % DILEMMAS}, {d, (d - 1) % DILEMMAS}):
                kept = ~np.isin(data.dilemma[fitting], list(held_out))
                share = took_action[fitting][kept].mean()
                parts.append(float(np.log(share if chose_action else 1 - share)))
            assert scores.log_likelihood["sign"][person, trial] == pytest.approx(
                np.mean(parts), abs=1e-6
            )
            own = data.choice[fitting][data.dilemma[fitting] == d].mean()
            chose_cbr = data.choice[person, trial] == 1
            assert scores.log_likelihood[lean.KNOWN][person, trial] == pytest.approx(
                np.log(own if chose_cbr else 1 - own), abs=1e-9
            )


def test_pooled_pair_is_the_existing_power_calculation() -> None:
    """The pooled models here are the nested pair of `power.cross_validate`, on the same
    folds and held-out sets, so their difference is the existing effect."""
    data = simulate(150, 0.3, 1.0, seed=39, intercept=_generated(ALPHA, BETA))
    scores = lean.cross_validate(data, lean.SIGN, {})
    gain = scores.log_likelihood[lean.POOLED + lean.LEARNING] - scores.log_likelihood[lean.POOLED]
    assert gain == pytest.approx(power.cross_validate(data, True, True), abs=1e-12)
    assert set(scores.log_likelihood) == {lean.KNOWN, lean.POOLED, lean.POOLED + lean.LEARNING}


def _printed(study: int) -> dict[str, outside.Rating]:
    """Ratings made from the means printed in the paper, so that no raw file is needed."""
    return {
        name: outside.Rating(80, dilemma.published[study])
        for name, dilemma in outside.published().items()
        if study in dilemma.published
    }


def test_the_map_is_never_fitted_on_the_tasks_dilemmas() -> None:
    """Protects `and name not in task` in `lean.outside_names`, and its use in `fit_map`.

    The map is fitted on the 25 factual dilemmas that are not originals of the task's 13,
    and on no trolley-type dilemma. Whatever the judgments of the 13 are, the map is the
    same; a judgment of a dilemma it is fitted on does change it."""
    dilemmas, originals = outside.published(), outside.originals()
    rated = _printed(2)
    fitted = lean.fit_map(dilemmas, originals, rated, lean.FINITE_PENALTIES)
    assert len(fitted.fitted_on) == 25
    assert set(fitted.fitted_on).isdisjoint(originals.values())
    assert all(dilemmas[name].kind == "factual" for name in fitted.fitted_on)
    assert set(fitted.fitted_on) | set(originals.values()) == {
        name for name, dilemma in dilemmas.items() if dilemma.kind == "factual"
    }
    assert fitted.weights.any()

    moved = rated | {name: outside.Rating(80, 1.0) for name in originals.values()}
    same = lean.fit_map(dilemmas, originals, moved, lean.FINITE_PENALTIES)
    assert same.penalty == fitted.penalty and same.offset == fitted.offset
    assert np.array_equal(same.weights, fitted.weights)

    moved = rated | {fitted.fitted_on[0]: outside.Rating(80, 1.0)}
    other = lean.fit_map(dilemmas, originals, moved, lean.FINITE_PENALTIES)
    assert not np.array_equal(other.weights, fitted.weights)


def test_ridge() -> None:
    """A weak penalty recovers a linear relation, a stronger one shrinks the weights, an
    infinite one predicts the mean, and a code that does not vary gets no weight. Protects
    `lean.ridge`."""
    rng = np.random.default_rng(41)
    features = np.column_stack([rng.normal(size=(40, 3)), np.ones(40)])
    truth = np.array([0.5, -1.0, 2.0, 0.0])
    target = 4.0 + features @ truth
    offset, weights = lean.ridge(features, target, 1e-9)
    assert weights == pytest.approx(truth, abs=1e-6)
    assert offset + features[0] @ weights == pytest.approx(target[0], abs=1e-6)
    assert weights[3] == 0.0

    _, weak = lean.ridge(features, target, 1.0)
    _, strong = lean.ridge(features, target, 100.0)
    assert np.linalg.norm(strong) < np.linalg.norm(weak) < np.linalg.norm(truth)

    offset, weights = lean.ridge(features, target, np.inf)
    assert offset == pytest.approx(target.mean()) and not weights.any()


def test_penalty_is_chosen_without_the_dilemma_being_predicted() -> None:
    """Protects `rest` in `lean.left_out_predictions` and the choice in `choose_penalty`.

    A linear relation gets the weakest penalty. A target that says nothing gets the
    strongest, which is the map that ignores the codes. Changing one dilemma's judgment
    does not change the prediction made for it, and does change the others."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(25, 3))
    linear = 4.0 + features @ np.array([0.5, -1.0, 2.0])
    assert lean.choose_penalty(features, linear) == min(lean.PENALTIES)
    assert lean.choose_penalty(features, np.full(25, 4.0)) == np.inf
    assert lean.PENALTIES[0] == np.inf and np.inf not in lean.FINITE_PENALTIES

    noisy = linear + rng.normal(0, 0.5, 25)
    before = lean.left_out_predictions(features, noisy)
    moved = noisy.copy()
    moved[0] += 3.0
    after = lean.left_out_predictions(features, moved)
    assert after[0] == before[0]
    assert not np.allclose(after[1:], before[1:])
    assert np.corrcoef(before, noisy)[0, 1] > 0.9


def test_rating_of_a_reversed_action_is_reflected() -> None:
    """Protects `np.where(reversed_, low + high - rating, rating)` in
    `lean.suggested_action_rating` and its use in `norm_lean`. One of the task's dilemmas
    asks about the opposite of the original's suggested action; its rating, and no other,
    is reflected about the middle of the scale."""
    assert lean.suggested_action_rating(
        np.array([2.5, 4.0, 5.5]), np.array([True, True, False])
    ) == pytest.approx([5.5, 4.0, 5.5])

    originals, rated = outside.originals(), _printed(2)
    made = lean.norm_lean(originals, rated, outside.action_reversed())
    for d, name in enumerate(fetch.DILEMMAS):
        judged = rated[originals[name]].mean
        assert made.rating[d] == pytest.approx(8 - judged if name == "Endow" else judged)
    assert made.measure.mean() == pytest.approx(0.0, abs=1e-12)
    assert made.measure.std() == pytest.approx(1.0)
    assert np.argsort(made.measure).tolist() == np.argsort(made.rating).tolist()


def test_a_map_that_ignores_the_codes_gives_no_lean() -> None:
    """A rating with no spread has m(d) = 0 for every dilemma, which is the `sign` lean.
    Protects the early return in `lean.standardise`. The codes handed to the map describe
    the action the task asks about, so nothing is reflected and nothing varies."""
    assert not lean.standardise(np.full(DILEMMAS, 4.57)).any()
    dilemmas, originals = outside.published(), outside.originals()
    constant = lean.fit_map(dilemmas, originals, _printed(2), (np.inf,))
    made = lean.coded_lean(constant, outside.original_codes())
    assert np.ptp(made.rating) == 0.0 and not made.measure.any()

    codes = outside.original_codes()
    action = outside.FEATURES.index("action is the utilitarian option")
    assert [outside.encode(codes[name])[action] for name in fetch.DILEMMAS] == [
        float(fetch.CBR_IS_ACTION[name]) for name in fetch.DILEMMAS
    ]
    finite = lean.fit_map(dilemmas, originals, _printed(2), lean.FINITE_PENALTIES)
    assert lean.coded_lean(finite, codes).measure.std() == pytest.approx(1.0)


def test_the_seal_holds_through_the_lean_analysis(tmp_path: Path) -> None:
    """THE SEAL. In the raw files every held-out record is unreadable. The lean analysis
    runs, scores exactly the training participants, and its per-dilemma shares are those of
    the training participants alone. Nothing in the new modules names the held-out set, and
    the report takes its participants from `fetch.load` and their number from
    `fetch.held_out_counts`."""
    everyone = [(e, str(n)) for e in fetch.SOURCES for n in range(1, 61)]
    held_out = frozenset(key for key in everyone if split.assign(*key) == "held_out")
    intended = write_raw(tmp_path, 60, unreadable=held_out)
    training = {key: chosen for key, chosen in intended.items() if key not in held_out}
    assert 0 < len(training) < len(intended)

    data = power.sequences(fetch.load(tmp_path))
    assert len(data.choice) == len(training)
    scores = lean.cross_validate(data, lean.SIGN, {"measure": MEASURE})
    for values in scores.log_likelihood.values():
        assert values.shape == (len(training), DILEMMAS) and np.isfinite(values).all()
    expected = [np.mean([chosen[name] for chosen in training.values()]) for name in fetch.DILEMMAS]
    assert lean.shares(data) == pytest.approx(expected)

    assert {name for name in vars(lean) if "held" in name.lower()} == {"held_out_sets"}
    assert not {name for name in vars(models) if "held" in name.lower()}
    called = set(re.findall(r"\bfetch\.(\w+)\(", Path(inspect.getfile(report)).read_text()))
    assert called == {"fetch", "load", "held_out_counts"}
