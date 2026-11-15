"""Print the pilot's numbers. Run as `uv run python -m zoon.pilot.report power`.

Every number quoted in `docs/pilot/power.md` is printed here, one per labelled line.
"""

import sys

import numpy as np

from zoon.pilot import fetch, power
from zoon.pilot.models import (
    LEARNING_RATE_BOUNDS,
    PROBABILITY_FLOOR,
    TEMPERATURE_BOUNDS,
    Bools,
    Floats,
    Sequences,
    fit_constant,
    fit_learning,
    log_likelihood,
)
from zoon.pilot.split import HELD_OUT_FRACTION

VARIANTS = (("plain", False), ("dilemma intercepts", True))
SPLITS = (("participant", False), ("participant and dilemma", True))
EARLY, LATE = slice(1, 7), slice(7, 13)  # trials 2-7 and 8-13, zero-based


def _interval(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:+.5f}, {interval[1]:+.5f}]"


def _fitted(scope: str, data: Sequences) -> None:
    """Print the parameters fitted to every training participant in the group."""
    everything = np.ones(data.choice.shape, dtype=np.bool_)
    dilemmas = len(fetch.DILEMMAS)
    print(f"[{scope}] constant | fitted P(cost-benefit option) = {data.choice.mean():.4f}")
    for variant, intercepts in VARIANTS:
        fit = fit_learning(data, everything, dilemmas, intercepts)
        base = fit_constant(data, everything, dilemmas, intercepts)
        gain = log_likelihood(fit, data).mean() - log_likelihood(base, data).mean()
        label = f"[{scope}] learning, {variant}"
        print(f"{label} | fitted learning rate = {fit.learning_rate:.4f}")
        print(f"{label} | fitted temperature = {fit.temperature:.4f}")
        print(f"{label} | in-sample gain over constant = {gain:+.5f} nats/choice")


def _effect(label: str, gain: Floats, held_out: int) -> None:
    """Print the effect, its uncertainty, and what the held-out participants could detect."""
    effect = power.summarise(gain)
    print(f"{label} | effect = {effect.mean:+.5f} nats/choice")
    print(f"{label} | effect 95% interval = {_interval(effect.interval)}")
    print(f"{label} | sd across participants = {effect.sd:.5f}")
    print(f"{label} | detectable effect at 80% power = {effect.detectable(held_out):.5f}")
    print(f"{label} | power for the observed effect = {effect.power(held_out):.3f}")
    print(f"{label} | power for half the observed effect = {effect.power(held_out, 0.5):.3f}")
    print(f"{label} | effect on trial 1 = {gain[:, 0].mean():+.5f}")
    for name, trials in (("trials 2-7", EARLY), ("trials 8-13", LATE)):
        part = power.summarise(gain[:, trials])
        print(f"{label} | effect on {name} = {part.mean:+.5f}")
        print(f"{label} | effect on {name} 95% interval = {_interval(part.interval)}")


def _breakdown(label: str, gain: Floats, data: Sequences, conditions: dict[str, Bools]) -> None:
    """Print the effect by trial number, by condition, and by dilemma within condition."""
    for trial, value in enumerate(gain.mean(axis=0)[1:], start=2):
        print(f"{label} | effect on trial {trial} = {value:+.5f}")
    for name, members in conditions.items():
        part = power.summarise(gain[members])
        print(f"{label} | effect, {name} = {part.mean:+.5f}")
        print(f"{label} | effect, {name} 95% interval = {_interval(part.interval)}")
        shown = data.dilemma[members]
        each = [float(gain[members][shown == d].mean()) for d in range(len(fetch.DILEMMAS))]
        low, high = int(np.argmin(each)), int(np.argmax(each))
        print(f"{label} | effect, {name}, lowest dilemma = {each[low]:+.5f}")
        print(f"{label} | effect, {name}, highest dilemma = {each[high]:+.5f}")


def run_power() -> None:
    fetch.fetch()
    table = fetch.load()
    held_out = fetch.held_out_counts()
    people = power.participants(table)
    everyone = power.sequences(table)

    print("zoon pilot: power calculation, training participants only")
    print("effect = mean log-likelihood per choice, learning model minus constant model,")
    print("         both fitted on other participants; positive favours learning")
    print("note: the plain models are not nested (the learning model has no intercept), so")
    print("      their difference on trial 1, before any feedback, is not an effect of learning")
    print(f"seed = {power.SEED}")
    print(f"held-out fraction aimed for = {HELD_OUT_FRACTION}")
    print(f"participant folds = {power.FOLDS}")
    print(f"dilemmas = {len(fetch.DILEMMAS)}")
    print(f"dilemmas held out at a time = {power.HELD_OUT_DILEMMAS}")
    print(f"times each dilemma is held out = {power.HELD_OUT_DILEMMAS}")
    print(f"learning rate bounds = {list(LEARNING_RATE_BOUNDS)}")
    print(f"temperature bounds = {list(TEMPERATURE_BOUNDS)}")
    print(f"bootstrap resamples = {power.RESAMPLES}")
    print(f"probability clamp = [{PROBABILITY_FLOOR}, {1 - PROBABILITY_FLOOR}]")
    print(f"test = paired t across participants, two-sided alpha = {power.ALPHA}")
    print(f"target power = {power.TARGET_POWER}")
    print()

    scopes: dict[str, tuple[Bools, int]] = {
        f"experiment {e}": ((people["experiment"] == e).to_numpy(), count)
        for e, count in sorted(held_out.items())
    }
    scopes["pooled"] = (np.ones(len(people), dtype=np.bool_), sum(held_out.values()))
    conditions: dict[str, Bools] = {
        name: (people["condition"] == name).to_numpy() for name in fetch.CONDITIONS
    }

    for scope, (members, count) in scopes.items():
        print(f"[{scope}] participants after exclusions = {members.sum() + count}")
        print(f"[{scope}] training participants = {members.sum()}")
        print(f"[{scope}] held-out participants = {count}")
    for name, members in conditions.items():
        print(f"[pooled] training participants, {name} = {members.sum()}")
    print(f"[pooled] training choices = {len(table)}")
    low, high = everyone.feedback.min(), everyone.feedback.max()
    print(f"[pooled] feedback range in training data = [{low:+.2f}, {high:+.2f}]")
    print()

    for scope, (members, count) in scopes.items():
        data = everyone.take(members)
        _fitted(scope, data)
        for split, by_dilemma in SPLITS:
            for variant, intercepts in VARIANTS:
                label = f"[{scope}] split = {split} | models = {variant}"
                gain = power.cross_validate(data, by_dilemma, intercepts)
                _effect(label, gain, count)
                if scope == "pooled":
                    _breakdown(label, gain, data, conditions)
                if scope == "pooled" and not by_dilemma:
                    shuffled = power.shuffle_feedback(data)
                    control = power.summarise(power.cross_validate(shuffled, False, intercepts))
                    name = "effect with feedback shuffled within participant"
                    print(f"{label} | {name} = {control.mean:+.5f}")
                    print(f"{label} | {name} 95% interval = {_interval(control.interval)}")
        print()


if __name__ == "__main__":
    if sys.argv[1:] != ["power"]:
        sys.exit("usage: python -m zoon.pilot.report power")
    run_power()
