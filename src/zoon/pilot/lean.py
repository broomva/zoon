"""If a dilemma's lean came from published human norms or codes, how much would it help?

No language model is involved. A dilemma's lean is its logit of choosing the cost-benefit
option before any feedback. The existing baseline gives a dilemma it was not fitted to the
pooled share. Here the lean of such a dilemma is set from information published outside the
task:

    logit P(cost-benefit option) = s(d) * (alpha + beta * m(d)) + tau * (v[1] - v[0])

`s(d)` is +1 where the suggested action is the cost-benefit option and -1 where it is the rule
option. `m(d)` says how appropriate the suggested action is, in one of three ways:

    sign    m(d) = 0: nothing but which option the question asks about
    norm    the mean appropriateness judgment of the original dilemma in a published sample
    coded   the judgment predicted from the original paper's feature codes by a ridge map
            fitted on the original dilemmas that are not among the task's 13

`alpha` and `beta`, and the learning rate and temperature where the model learns, are fitted
on training-fold participants and on dilemmas that are not held out, in the nested scheme of
`zoon.pilot.power`.
"""

import math
from dataclasses import dataclass

import numpy as np

from zoon.pilot import outside
from zoon.pilot.fetch import CBR_IS_ACTION, DILEMMAS
from zoon.pilot.models import (
    Bools,
    Floats,
    Sequences,
    fit_constant,
    fit_lean,
    fit_learning,
    log_likelihood,
)
from zoon.pilot.power import FOLDS, SEED, bootstrap_interval, folds, held_out_sets

SIGN: Floats = np.where([CBR_IS_ACTION[name] for name in DILEMMAS], 1.0, -1.0)
POOLED = "pooled"
KNOWN = "known share"
LEARNING = " + learning"
# Tried from the strongest penalty down, so that a tie goes to the stronger one. An infinite
# penalty is the map that ignores the codes and predicts the mean judgment for every dilemma.
FINITE_PENALTIES: tuple[float, ...] = (
    1000.0,
    300.0,
    100.0,
    30.0,
    10.0,
    3.0,
    1.0,
    0.3,
    0.1,
    0.03,
    0.01,
)
PENALTIES: tuple[float, ...] = (math.inf, *FINITE_PENALTIES)


@dataclass(frozen=True)
class Map:
    """A ridge map from a dilemma's published codes to its mean appropriateness judgment."""

    fitted_on: tuple[str, ...]  # names of the dilemmas it was fitted on
    penalty: float
    offset: float
    weights: Floats  # rating points per unit of each code, in the order of outside.FEATURES

    def predict(self, features: Floats) -> Floats:
        return self.offset + features @ self.weights


def ridge(features: Floats, target: Floats, penalty: float) -> tuple[float, Floats]:
    """Fit a ridge regression and return its offset and weights in the units of the codes.

    Each code is centred and scaled to unit spread within the dilemmas given, so that one
    penalty means the same for every code. The offset is not penalised. A code that does not
    vary among the dilemmas given gets weight zero: nothing was learned about it. With an
    infinite penalty every weight is zero and the offset is the mean judgment.
    """
    if math.isinf(penalty):
        return float(target.mean()), np.zeros(features.shape[1])
    centre = features.mean(axis=0)
    spread = features.std(axis=0)
    varies = spread > 0
    scaled = (features[:, varies] - centre[varies]) / spread[varies]
    gram = scaled.T @ scaled + penalty * np.eye(int(varies.sum()))
    solved = np.linalg.solve(gram, scaled.T @ (target - target.mean()))
    weights = np.zeros(features.shape[1])
    weights[varies] = solved / spread[varies]
    return float(target.mean() - centre @ weights), weights


def _left_out_errors(features: Floats, target: Floats, penalty: float) -> Floats:
    """Squared error on each dilemma of a map fitted on all the others."""
    errors = np.empty(len(target))
    for i in range(len(target)):
        rest = np.arange(len(target)) != i
        offset, weights = ridge(features[rest], target[rest], penalty)
        errors[i] = (offset + features[i] @ weights - target[i]) ** 2
    return errors


def choose_penalty(
    features: Floats, target: Floats, penalties: tuple[float, ...] = PENALTIES
) -> float:
    """Return the penalty with the smallest leave-one-out error among the dilemmas given."""

    def error(penalty: float) -> float:
        return float(_left_out_errors(features, target, penalty).sum())

    return min(penalties, key=error)


def left_out_predictions(
    features: Floats, target: Floats, penalties: tuple[float, ...] = PENALTIES
) -> Floats:
    """Predict each dilemma from a map that never saw it.

    The penalty is chosen again inside each fit, without the dilemma being predicted, so the
    choice of penalty cannot flatter the result.
    """
    predictions = np.empty(len(target))
    for i in range(len(target)):
        rest = np.arange(len(target)) != i
        penalty = choose_penalty(features[rest], target[rest], penalties)
        offset, weights = ridge(features[rest], target[rest], penalty)
        predictions[i] = offset + features[i] @ weights
    return predictions


def outside_names(
    dilemmas: dict[str, outside.Dilemma], originals: dict[str, str]
) -> tuple[str, ...]:
    """Return the dilemmas the map may be fitted on: the factual ones, which have published
    codes, minus the originals of the task's own dilemmas."""
    task = set(originals.values())
    return tuple(
        name for name, dilemma in dilemmas.items() if dilemma.codes is not None and name not in task
    )


def design(
    dilemmas: dict[str, outside.Dilemma], names: tuple[str, ...], rated: dict[str, outside.Rating]
) -> tuple[Floats, Floats]:
    """Return the codes and the mean judgments of the named dilemmas."""
    rows: list[Floats] = []
    for name in names:
        codes = dilemmas[name].codes
        assert codes is not None
        rows.append(outside.encode(codes))
    return np.array(rows), np.array([rated[name].mean for name in names])


def fit_map(
    dilemmas: dict[str, outside.Dilemma],
    originals: dict[str, str],
    rated: dict[str, outside.Rating],
    penalties: tuple[float, ...] = PENALTIES,
) -> Map:
    """Fit the map on the outside dilemmas only, with the penalty chosen among them."""
    names = outside_names(dilemmas, originals)
    features, target = design(dilemmas, names, rated)
    penalty = choose_penalty(features, target, penalties)
    offset, weights = ridge(features, target, penalty)
    return Map(names, penalty, offset, weights)


def suggested_action_rating(rating: Floats, reversed_: Bools) -> Floats:
    """Turn a rating of the original's suggested action into one of the task's.

    Where the task asks about the opposite action, the rating is reflected about the middle
    of the scale, as the original paper does when it recodes judgments as utilitarian.
    """
    low, high = outside.SCALE
    return np.where(reversed_, low + high - rating, rating)


def standardise(rating: Floats) -> Floats:
    """Return m(d): the rating minus its mean over the dilemmas, in units of its spread over
    them. A rating with no spread says nothing, and m(d) is zero throughout.

    Any other centre would give the same predictions, since `alpha` is fitted. Centring at
    the mean keeps `alpha` and `beta` from trading off in the fit.
    """
    spread = float(rating.std())
    if spread < 1e-9:
        return np.zeros_like(rating)
    return (rating - rating.mean()) / spread


@dataclass(frozen=True)
class Lean:
    """One way of saying, for each of the task's dilemmas, how appropriate its suggested
    action is."""

    rating: Floats  # on the 1 to 7 scale of the outside samples, judged or predicted
    measure: Floats  # m(d), the rating standardised


def norm_lean(
    originals: dict[str, str], rated: dict[str, outside.Rating], reversed_: dict[str, bool]
) -> Lean:
    """The lean from the mean judgment of each dilemma's original in an outside sample."""
    judged = np.array([rated[originals[name]].mean for name in DILEMMAS])
    turned = np.array([reversed_[name] for name in DILEMMAS])
    rating = suggested_action_rating(judged, turned)
    return Lean(rating, standardise(rating))


def coded_lean(fitted: Map, codes: dict[str, outside.Codes]) -> Lean:
    """The lean from the map applied to each dilemma's published codes.

    The codes must describe the action the task asks about, so the prediction is already a
    rating of that action and nothing is reflected.
    """
    rating = fitted.predict(np.array([outside.encode(codes[name]) for name in DILEMMAS]))
    return Lean(rating, standardise(rating))


@dataclass(frozen=True)
class Scores:
    """What cross-validation returns for a set of models."""

    # model -> log-likelihood of each participant's choice on each trial, from fits that saw
    # neither the participant nor the trial's dilemma
    log_likelihood: dict[str, Floats]
    # model -> one row per fit: alpha, beta, learning rate, temperature
    fitted: dict[str, Floats]


def cross_validate(
    data: Sequences,
    sign: Floats,
    measures: dict[str, Floats],
    seed: int = SEED,
    dilemmas: int = len(DILEMMAS),
) -> Scores:
    """Score every model on every trial under the split by participant and by dilemma.

    The models are the pooled baseline with and without learning, as in
    `power.cross_validate`, and for each entry of `measures` a lean model with and without
    learning. Folds and held-out dilemma sets are those of `power.cross_validate`. A trial
    is scored once for each set its dilemma is held out in, and the scores are averaged.

    One more model is scored as a ceiling and is not a candidate: `KNOWN` gives each dilemma
    its own share among the fitting participants, with no dilemma held out. No lean without
    learning can do better than that, and nothing that has not seen choices on the dilemma
    can have it.
    """
    names = [POOLED, *measures]
    names += [name + LEARNING for name in names]
    fold = folds(len(data.choice), seed)
    total = {name: np.zeros(data.choice.shape) for name in names}
    count = np.zeros(data.choice.shape)
    known = np.zeros(data.choice.shape)
    fitted: dict[str, list[tuple[float, float, float, float]]] = {name: [] for name in names}
    for k in range(FOLDS):
        fitting, scoring = data.take(fold != k), data.take(fold == k)
        everything = np.ones(fitting.choice.shape, dtype=np.bool_)
        own_share = fit_constant(fitting, everything, dilemmas, dilemma_intercepts=True)
        known[fold == k] = log_likelihood(own_share, scoring)
        for held_out in held_out_sets(True, dilemmas):
            targets: Bools = ~np.isin(fitting.dilemma, held_out)
            scored: Bools = np.isin(scoring.dilemma, held_out)
            count[fold == k] += scored
            parameters = {
                POOLED: fit_constant(fitting, targets, dilemmas, dilemma_intercepts=True),
                POOLED + LEARNING: fit_learning(fitting, targets, dilemmas, True),
            }
            for name, measure in measures.items():
                for suffix, learning in (("", False), (LEARNING, True)):
                    fit = fit_lean(fitting, targets, sign, measure, learning)
                    parameters[name + suffix] = fit.parameters
                    fitted[name + suffix].append(
                        (
                            fit.alpha,
                            fit.beta,
                            fit.parameters.learning_rate,
                            fit.parameters.temperature,
                        )
                    )
            for name, model in parameters.items():
                total[name][fold == k] += np.where(scored, log_likelihood(model, scoring), 0.0)
    return Scores(
        {KNOWN: known} | {name: total[name] / count for name in names},
        {name: np.array(rows) for name, rows in fitted.items() if rows},
    )


def by_dilemma(gain: Floats, data: Sequences, dilemmas: int = len(DILEMMAS)) -> Floats:
    """Mean of a participants x trials array on each dilemma. Every participant met every
    dilemma once, so the mean of these is the overall mean."""
    return np.array([float(gain[data.dilemma == d].mean()) for d in range(dilemmas)])


def dilemma_interval(gain: Floats, data: Sequences) -> tuple[float, float]:
    """95% percentile interval for the mean, resampling dilemmas instead of participants."""
    return bootstrap_interval(by_dilemma(gain, data))


def shares(data: Sequences, dilemmas: int = len(DILEMMAS)) -> Floats:
    """Share of choices of the cost-benefit option on each dilemma."""
    return np.array([float(data.choice[data.dilemma == d].mean()) for d in range(dilemmas)])
