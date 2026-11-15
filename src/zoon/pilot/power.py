"""How large is the effect of learning from feedback, and could held-out data detect it?

Everything here runs on training participants. The effect is, for each participant, the mean
log-likelihood per choice under the learning model minus the same under the no-learning
model, with both fitted on other participants.

Two ways of splitting, both nested inside the training participants:

- by participant: five folds of participants; fit on four, score the fifth;
- by participant and dilemma: as above, and in addition two dilemmas at a time are held out
  in rotation. The models are fitted on the other eleven dilemmas of the fitting
  participants and scored on the two held-out dilemmas of the scoring participants.

What holding out a dilemma means for a model that learns across trials. A held-out dilemma is
withheld as a thing to predict and as a source of fitted parameters: its choices add nothing
to the fitted likelihood, and it gets no intercept of its own. It is not withheld as
experience. The participant met it at some point in their own random order, chose, and saw
the outcome, and their later choices depend on that. So the learning model updates its values
on every trial in the order the participant saw them, held-out dilemmas included, both when
fitting and when scoring. When the model predicts a held-out dilemma on trial t it has used
the participant's choices and feedback on trials 1 to t-1 and nothing from trial t.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct
from scipy.stats import t as student_t

from zoon.pilot.fetch import DILEMMAS
from zoon.pilot.models import (
    Bools,
    Floats,
    Ints,
    Sequences,
    fit_constant,
    fit_learning,
    log_likelihood,
)

FOLDS = 5
HELD_OUT_DILEMMAS = 2
SEED = 20261115
RESAMPLES = 10_000
ALPHA = 0.05
TARGET_POWER = 0.80


def sequences(table: pd.DataFrame) -> Sequences:
    """Arrange a trial-level table as participants x trials arrays, in trial order."""
    ordered = table.sort_values(["experiment", "participant", "trial"])
    trials = int(ordered["trial"].max())
    codes = ordered["dilemma"].map({name: i for i, name in enumerate(DILEMMAS)})

    def grid(values: pd.Series, kind: type) -> np.ndarray:
        return values.to_numpy(dtype=kind).reshape(-1, trials)

    return Sequences(
        choice=grid(ordered["cbr_choice"], np.int64),
        feedback=grid(ordered["valence"], np.float64),
        dilemma=grid(codes, np.int64),
    )


def participants(table: pd.DataFrame) -> pd.DataFrame:
    """One row per participant, in the row order of `sequences`."""
    ordered = table.sort_values(["experiment", "participant", "trial"])
    first = ordered["trial"] == 1
    return ordered.loc[first, ["experiment", "participant", "condition"]].reset_index(drop=True)


def shuffle_feedback(data: Sequences, seed: int = SEED) -> Sequences:
    """Permute each participant's feedback across their own trials. A negative control: it
    keeps every choice and every feedback value and breaks which feedback followed which
    choice."""
    rng = np.random.default_rng(seed)
    return Sequences(data.choice, rng.permuted(data.feedback, axis=1), data.dilemma)


def _held_out_sets(by_dilemma: bool, dilemmas: int) -> list[list[int]]:
    """Dilemma sets to hold out in turn. Neighbours in a fixed circular order, so that every
    dilemma is held out the same number of times."""
    if not by_dilemma:
        return [[]]
    return [[(d + k) % dilemmas for k in range(HELD_OUT_DILEMMAS)] for d in range(dilemmas)]


def folds(people: int, seed: int = SEED) -> Ints:
    """Assign each participant to one of FOLDS groups of near-equal size."""
    return np.random.default_rng(seed).permutation(people) % FOLDS


def cross_validate(
    data: Sequences,
    by_dilemma: bool,
    dilemma_intercepts: bool,
    seed: int = SEED,
    dilemmas: int = len(DILEMMAS),
) -> Floats:
    """Return learning minus no-learning log-likelihood for each participant and trial.

    Each entry comes only from models fitted without that participant and, when `by_dilemma`
    is set, without any choice on that trial's dilemma. Where a trial is scored more than
    once (each dilemma is held out in more than one set), the entry is the mean.
    """
    fold = folds(len(data.choice), seed)
    total = np.zeros(data.choice.shape)
    count = np.zeros(data.choice.shape)
    for k in range(FOLDS):
        fitting, scoring = data.take(fold != k), data.take(fold == k)
        for held_out in _held_out_sets(by_dilemma, dilemmas):
            targets: Bools = ~np.isin(fitting.dilemma, held_out)
            scored: Bools = np.isin(scoring.dilemma, held_out) | (not by_dilemma)
            learning = fit_learning(fitting, targets, dilemmas, dilemma_intercepts)
            constant = fit_constant(fitting, targets, dilemmas, dilemma_intercepts)
            gain = log_likelihood(learning, scoring) - log_likelihood(constant, scoring)
            total[fold == k] += np.where(scored, gain, 0.0)
            count[fold == k] += scored
    return total / count


def bootstrap_interval(values: Floats, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval for the mean, resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    means = np.empty(RESAMPLES)
    for start in range(0, RESAMPLES, 500):
        draws = rng.integers(0, len(values), size=(500, len(values)))
        means[start : start + 500] = values[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def power_to_detect(effect: float, sd: float, participants: int) -> float:
    """Power of a two-sided paired t-test across participants at level ALPHA."""
    df = participants - 1
    critical = float(student_t.ppf(1 - ALPHA / 2, df))
    shift = effect / sd * np.sqrt(participants)
    # Both tails. P(T < -c | shift) is written as P(T > c | -shift), which stays finite.
    return float(nct.sf(critical, df, shift) + nct.sf(critical, df, -shift))


def detectable_effect(sd: float, participants: int) -> float:
    """Smallest effect the same test detects with power TARGET_POWER."""
    upper = 6 * sd / np.sqrt(participants)  # six standard errors: power is close to one
    return float(brentq(lambda e: power_to_detect(e, sd, participants) - TARGET_POWER, 0.0, upper))


@dataclass(frozen=True)
class Effect:
    """The effect of learning in one group of participants under one split."""

    mean: float  # nats per choice
    interval: tuple[float, float]
    sd: float  # across participants

    def power(self, held_out: int, fraction: float = 1.0) -> float:
        """Power to detect this effect, or a fraction of it, with `held_out` participants."""
        return power_to_detect(abs(self.mean) * fraction, self.sd, held_out)

    def detectable(self, held_out: int) -> float:
        return detectable_effect(self.sd, held_out)


def summarise(gain: Floats) -> Effect:
    """Summarise a participants x trials array of gains: each participant counts once."""
    per_person = gain.mean(axis=1)
    return Effect(
        mean=float(per_person.mean()),
        interval=bootstrap_interval(per_person),
        sd=float(per_person.std(ddof=1)),
    )
