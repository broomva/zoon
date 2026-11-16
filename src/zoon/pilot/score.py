"""Paired log scores, registered intervals and the deciding pass rule."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from zoon.pilot import clients, fetch, power
from zoon.pilot.models import PROBABILITY_FLOOR, Floats, Ints, Sequences

Identity = tuple[int, str]


@dataclass(frozen=True)
class Comparison:
    mean: float
    interval: tuple[float, float]
    floor: bool
    positive: bool

    @property
    def passed(self) -> bool:
        return self.floor and self.positive


def rule(mean: float, interval: tuple[float, float]) -> Comparison:
    finite = bool(np.isfinite([mean, *interval]).all())
    return Comparison(mean, interval, finite and mean >= 0.002, finite and interval[0] > 0)


def verdict(comparisons: Sequence[Comparison]) -> str:
    if len(comparisons) != 3:
        raise ValueError("the verdict requires three comparisons")
    return "pass" if all(c.passed for c in comparisons) else "fail"


def complete(data: Sequences) -> None:
    shape = data.choice.shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] != len(fetch.DILEMMAS):
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    if data.feedback.shape != shape or data.dilemma.shape != shape:
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    expected = np.arange(len(fetch.DILEMMAS))
    if (
        not np.isin(data.choice, [0, 1]).all()
        or not np.isfinite(data.feedback).all()
        or not (np.sort(data.dilemma, axis=1) == expected).all()
        or (np.abs(data.feedback) > 1).any()
    ):
        raise clients.TerminalStop(clients.StopReason.RECORDS)


def rotation_logs(probabilities: Sequence[Floats], data: Sequences) -> Floats:
    complete(data)
    rotations = power.held_out_sets(True, len(fetch.DILEMMAS))
    if len(probabilities) != len(rotations):
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    total = np.zeros(data.choice.shape)
    count = np.zeros(data.choice.shape, dtype=np.int64)
    for pair, p in zip(rotations, probabilities, strict=True):
        target = np.isin(data.dilemma, pair)
        if (
            p.shape != data.choice.shape
            or not np.isfinite(p[target]).all()
            or ((p[target] < 0) | (p[target] > 1)).any()
        ):
            raise clients.TerminalStop(clients.StopReason.RECORDS)
        chosen = np.where(data.choice == 1, p, 1 - p)
        total[target] += np.log(np.clip(chosen[target], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))
        count += target
    if not (count == 2).all():
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    return total / count


def compare(
    first: Floats,
    second: Floats,
    identities: Sequence[Identity],
    dilemma: Ints,
    *,
    keep: Sequence[int] | None = None,
    later: bool = False,
    experiment: int | None = None,
) -> Comparison | None:
    if (
        first.shape != second.shape
        or first.shape != dilemma.shape
        or first.shape[0] != len(identities)
    ):
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    if len(set(identities)) != len(identities):
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    ordered = sorted(range(len(identities)), key=lambda i: identities[i])
    means: list[float] = []
    for i in ordered:
        if experiment is not None and identities[i][0] != experiment:
            continue
        targets = np.ones(first.shape[1], dtype=np.bool_)
        if later:
            targets[0] = False
        if keep is not None:
            targets &= np.isin(dilemma[i], keep)
        if not targets.any():
            continue
        values = first[i, targets] - second[i, targets]
        if not np.isfinite(values).all():
            raise clients.TerminalStop(clients.StopReason.RECORDS)
        means.append(float(values.mean()))
    if not means:
        return None
    values = np.array(means)
    return rule(float(values.mean()), power.bootstrap_interval(values))


def dilemma_interval(
    first: Floats, second: Floats, dilemma: Ints, keep: Sequence[int] | None = None
) -> tuple[float, float] | None:
    retained = list(range(len(fetch.DILEMMAS))) if keep is None else list(keep)
    if not retained:
        return None
    gain = first - second
    means = np.array([float(gain[dilemma == d].mean()) for d in retained])
    if not np.isfinite(means).all():
        raise clients.TerminalStop(clients.StopReason.RECORDS)
    return power.bootstrap_interval(means)
