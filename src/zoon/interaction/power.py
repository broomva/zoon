"""Training-only power and coverage under the registered comparison rules."""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from types import ModuleType

import numpy as np

from zoon.interaction import compare, data, games, kernel, measure, person, registered, score
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES
from zoon.interaction.measure import Floats

H2_TRIALS = 100_000
H3_TRIALS = 5_000
TRIAL_RESAMPLES = 1_000
H2_BATCH = 1_000
POOL_GROUPS = 2000
TRIAL_SEED = 20261121
HALF_EFFECT = 0.5
NULL_EFFECT = 0.0
TRAINING_EFFECT = 1.0
POOL_PREFIX = "power-pool"
POOL_REPLACEMENT = False
EXPECTED_H2 = ("0.0209", "0.0133", "0.0281")
EXPECTED_RESCORE = "0.0253"
TITLE = "zoon interaction: registered-rule power on training groups"
TRAINING_H2 = "[training, H2, out of fold] zoon minus alone"
TRAINING_RESCORE = "[training, H2, out of fold, person floor] zoon minus alone"
TRAINING_H3 = "[training, H3, in-sample laws] alone minus zoon"
OUT_OF_FOLD_H3 = "[training, H3, out of fold] alone minus zoon"
POOL_H3 = "[H3] finite-pool trial expectation"
EXPECTED_H3 = "0.0124"
RATE_LABELS = (
    "coverage of the trial expectation",
    "lower end above truth",
    "upper end below truth",
    "pass at zero",
    "pass at half the out-of-fold effect",
    "pass at the out-of-fold effect",
)

# Resolve attributes at rendering time, so no pasted setting can silently become stale.
SETTINGS: tuple[tuple[str, ModuleType, str], ...] = (
    ("registered interval seed", score, "SEED"),
    ("power trial seed", sys.modules[__name__], "TRIAL_SEED"),
    ("registered interval replicates", score, "RESAMPLES"),
    ("H2 trial interval replicates", score, "RESAMPLES"),
    ("H2 batch size", sys.modules[__name__], "H2_BATCH"),
    ("fold and out-of-fold simulation seed", measure, "SEED"),
    ("H2 power trials", sys.modules[__name__], "H2_TRIALS"),
    ("H3 power trials", sys.modules[__name__], "H3_TRIALS"),
    ("H3 trial interval replicates", sys.modules[__name__], "TRIAL_RESAMPLES"),
    ("pool groups per game and arm", sys.modules[__name__], "POOL_GROUPS"),
    ("pool prefix", sys.modules[__name__], "POOL_PREFIX"),
    ("pool draws with replacement", sys.modules[__name__], "POOL_REPLACEMENT"),
    ("simulated groups per game", score, "SIMULATED_GROUPS"),
    ("half ensemble", score, "HALF_ENSEMBLE"),
    ("human groups omitted per game", score, "HUMAN_DEFICIT"),
    ("interval percentiles", score, "PERCENTILES"),
    ("H2 threshold", score, "H2_THRESHOLD"),
    ("H3 threshold", score, "H3_THRESHOLD"),
    ("scoring floor", score, "SCORING_FLOOR"),
    ("rescoring floor", score, "RESCORING_FLOOR"),
    ("folds", measure, "FOLDS"),
    ("rounds", games, "ROUNDS"),
    ("null effect multiplier", sys.modules[__name__], "NULL_EFFECT"),
    ("half effect multiplier", sys.modules[__name__], "HALF_EFFECT"),
    ("out-of-fold effect multiplier", sys.modules[__name__], "TRAINING_EFFECT"),
    ("registered simulation prefix", score, "SIMULATION_PREFIX"),
)


def settings() -> list[str]:
    return [f"[settings] {label} = {getattr(source, name)}" for label, source, name in SETTINGS] + [
        entry.render() for entry in compare.settings() if entry.label.startswith("[fit]")
    ]


def out_of_fold(groups: Sequence[Group], candidate: person.Candidate) -> score.Decisions:
    """The same folds and fit order as person.out_of_sample, retaining all bands."""
    folds = np.array([measure.fold_of(g) for g in groups])
    decisions = measure.decisions(groups)
    probabilities = np.empty((len(decisions.choice), measure.BANDS))
    for fold in sorted(set(folds.tolist())):
        train = [g for i, g in enumerate(groups) if folds[i] != fold]
        test = [g for i, g in enumerate(groups) if folds[i] == fold]
        probabilities[folds[decisions.group] == fold] = score.coded(
            person.fit(train, candidate), test
        ).probabilities
    return score.Decisions(decisions.group, decisions.choice, probabilities)


def h2_training(groups: Sequence[Group]) -> tuple[Floats, score.Interval, score.Interval]:
    zoon = out_of_fold(groups, registered.FORMS["zoon"])
    alone = out_of_fold(groups, registered.FORMS["alone"])
    common = score.group_means((zoon.logs() - alone.logs())[:, None], zoon.group, groups)
    original = score.group_means(
        (zoon.logs(person.FLOOR) - alone.logs(person.FLOOR))[:, None], zoon.group, groups
    )
    return common, score.h2_interval(common, groups), score.h2_interval(original, groups)


def checked_training_h2(result: score.Interval, rescore: score.Interval) -> None:
    quoted = tuple(f"{v:.4f}" for v in result.table()[0])
    if quoted != EXPECTED_H2 or f"{rescore.estimate[0]:.4f}" != EXPECTED_RESCORE:
        raise RuntimeError(f"registered training H2 check failed: {quoted}")


def sample_groups(
    groups: Sequence[Group], counts: Mapping[str, int], rng: np.random.Generator
) -> tuple[list[Group], measure.Ints]:
    cells = score.by_game(groups)
    if set(counts) != set(cells) or any(n <= score.HUMAN_DEFICIT for n in counts.values()):
        raise ValueError("held-out counts must cover the scored games with at least two each")
    indices = np.concatenate(
        [rng.choice(cells[cell], counts[cell], replace=True) for cell in cells]
    ).astype(np.int64)
    # Each resampled occurrence is a distinct group with a stable within-game index.
    trial = [
        Group(groups[int(i)].cell, str(n).zfill(len(str(len(indices)))), groups[int(i)].rounds)
        for n, i in enumerate(indices)
    ]
    return trial, indices


@dataclass(frozen=True)
class Rates:
    coverage: float
    lower_above_truth: float
    upper_below_truth: float
    null: float
    half: float
    full: float

    def values(self) -> tuple[float, ...]:
        return (
            self.coverage,
            self.lower_above_truth,
            self.upper_below_truth,
            self.null,
            self.half,
            self.full,
        )


def trial_rates(
    estimates: Floats,
    lows: Floats,
    highs: Floats,
    truth: float,
    threshold: float,
    training_effect: float | None = None,
) -> Rates:
    """Location shifts preserve all game's residuals, interval widths and pairing."""
    if not len(estimates) or estimates.shape != lows.shape or lows.shape != highs.shape:
        raise ValueError("nonempty matched trial estimates and bounds are required")
    score.finite(estimates)
    score.finite(lows)
    score.finite(highs)
    score.finite(np.array([truth, threshold]))
    target = truth if training_effect is None else training_effect
    score.finite(np.array([target]))
    coverage = float(np.mean((lows <= truth) & (truth <= highs)))
    passed: list[float] = []
    for multiplier in (NULL_EFFECT, HALF_EFFECT, TRAINING_EFFECT):
        shift = multiplier * target - truth
        passed.append(
            float(np.mean(score.comparison_passes(estimates + shift, lows + shift, threshold)))
        )
    return Rates(coverage, float(np.mean(lows > truth)), float(np.mean(highs < truth)), *passed)


def h2_trial_intervals(
    groups: Sequence[Group],
    differences: Floats,
    counts: Mapping[str, int],
    *,
    trials: int = H2_TRIALS,
    resamples: int = score.RESAMPLES,
) -> tuple[Floats, Floats, Floats]:
    """Apply the registered draws to batches by their exact multiplicity matrices."""
    if trials <= 0 or differences.shape != (len(groups), 1):
        raise ValueError("positive trials and one difference per group are required")
    score.finite(differences)
    draws = score.h2_indices(counts, resamples)
    total = sum(n * GAMES[c].size for c, n in counts.items())
    weights = {c: n * GAMES[c].size / total for c, n in counts.items()}
    matrices = {
        c: np.column_stack([(idx == j).sum(axis=1) for j in range(counts[c])])
        * (weights[c] / idx.shape[1])
        for c, idx in draws.items()
    }
    rng = np.random.default_rng(TRIAL_SEED)
    estimates, lows, highs = (np.empty(trials) for _ in range(3))
    for start in range(0, trials, H2_BATCH):
        stop = min(start + H2_BATCH, trials)
        indices = np.array([sample_groups(groups, counts, rng)[1] for _ in range(stop - start)])
        sampled = differences[indices, 0]
        values = np.zeros((stop - start, resamples))
        point = np.zeros(stop - start)
        offset = 0
        for cell in sorted(draws):
            n = counts[cell]
            cell_values = sampled[:, offset : offset + n]
            values += cell_values @ matrices[cell].T
            point += weights[cell] * cell_values.mean(axis=1)
            offset += n
        estimates[start:stop] = point
        lows[start:stop], highs[start:stop] = np.percentile(values, score.PERCENTILES, axis=1)
    return estimates, lows, highs


def h2_rates(
    groups: Sequence[Group],
    differences: Floats,
    counts: Mapping[str, int],
    *,
    trials: int = H2_TRIALS,
    resamples: int = score.RESAMPLES,
) -> Rates:
    cells = score.interval_groups(groups)
    means = {c: float(differences[mine, 0].mean()) for c, mine in cells.items()}
    total = sum(n * GAMES[c].size for c, n in counts.items())
    truth = sum(counts[c] * GAMES[c].size / total * means[c] for c in cells)
    training_weights = score.h2_weights(groups)
    training_effect = sum(training_weights[c] * means[c] for c in cells)
    estimates, lows, highs = h2_trial_intervals(
        groups, differences, counts, trials=trials, resamples=resamples
    )
    return trial_rates(
        estimates, lows, highs, truth, score.H2_THRESHOLD, training_effect=training_effect
    )


def pool(
    laws: Mapping[str, person.Person], amounts: kernel.Amounts, count: int = POOL_GROUPS
) -> dict[str, dict[str, Floats]]:
    """Simulate once; distinct pool strings, shared group numbers across arms."""
    result: dict[str, dict[str, Floats]] = {}
    for arm, law in laws.items():
        result[arm] = {}
        for cell in sorted(GAMES):
            game = GAMES[cell]
            agents = [kernel.Coded(law) for _ in range(game.size)]
            values = np.array(
                [
                    compare.summaries(
                        kernel.play(
                            game,
                            agents,
                            amounts,
                            score.SEED,
                            f"{POOL_PREFIX}|{cell}|{n}",
                            open_channel=not law.candidate.twin,
                        ).rounds,
                        game,
                    )
                    for n in range(count)
                ]
            )
            score.validate_summaries(values, cell)
            result[arm][cell] = values
    return result


def pool_truth(
    groups: Sequence[Group],
    simulations: Mapping[str, Mapping[str, Floats]],
    counts: Mapping[str, int],
) -> float:
    # Fair CRPS on the whole finite pool is the expectation of fair CRPS for a
    # subset drawn without replacement. Human expectations use each game's training
    # distribution; weights use the registered held-out counts.
    trial, _ = sample_groups(groups, counts, np.random.default_rng(TRIAL_SEED))
    weights = score.h3_weights(trial)
    humans = score.observed(groups)
    scores: dict[str, Floats] = {}
    for arm, simulated in simulations.items():
        total = np.zeros(len(compare.SUMMARIES))
        for cell, human in humans.items():
            width = len(compare.SUMMARIES) - int(cell in score.UNDEFINED_GAMES)
            for s in range(width):
                mean = np.mean([compare.crps(simulated[cell][:, s], float(y)) for y in human[:, s]])
                total[s] += weights[cell][s] * mean
        scores[arm] = total
    return float((scores["alone"] - scores["zoon"]).mean())


def h3_rates(
    groups: Sequence[Group],
    simulations: Mapping[str, Mapping[str, Floats]],
    counts: Mapping[str, int],
    *,
    training_effect: float,
    trials: int = H3_TRIALS,
    resamples: int = TRIAL_RESAMPLES,
) -> Rates:
    rng = np.random.default_rng(TRIAL_SEED)
    truth = pool_truth(groups, simulations, counts)
    estimates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(trials):
        trial, _ = sample_groups(groups, counts, rng)
        draws: dict[str, dict[str, Floats]] = {arm: {} for arm in simulations}
        for cell in score.by_game(groups):
            sizes = {len(simulated[cell]) for simulated in simulations.values()}
            if len(sizes) != 1:
                raise ValueError("paired arms require equal pool sizes")
            indices = rng.choice(sizes.pop(), score.SIMULATED_GROUPS, replace=POOL_REPLACEMENT)
            for arm in simulations:
                draws[arm][cell] = simulations[arm][cell][indices]
        intervals = score.h3_interval(trial, draws, resamples)
        result = score.difference(intervals["alone"], intervals["zoon"])
        low, high = result.bounds()[:, 0]
        estimates.append(result.estimate[0])
        lows.append(low)
        highs.append(high)
    return trial_rates(
        np.array(estimates),
        np.array(lows),
        np.array(highs),
        truth,
        score.H3_THRESHOLD,
        training_effect=training_effect,
    )


def formatted(label: str, interval: score.Interval) -> str:
    point, low, high = interval.table()[0]
    return f"{label} = {point:.4f} [{low:.4f}, {high:.4f}]"


def formatted_rate(value: float, trials: int) -> str:
    se = sqrt(value * (1 - value) / trials)
    return f"{value:.3f} (se {se:.3f})"


def labels() -> list[str]:
    return [
        TITLE,
        *[line.split(" = ")[0] for line in settings()],
        "[data] training groups",
        "[data] held-out groups",
        TRAINING_H2,
        TRAINING_RESCORE,
        TRAINING_H3,
        OUT_OF_FOLD_H3,
        POOL_H3,
        *[f"[{hypothesis}] {label}" for hypothesis in ("H2", "H3") for label in RATE_LABELS],
    ]


def h3_out_of_fold(groups: Sequence[Group]) -> float:
    """The candidate note's folds, simulations and composite, without a second rule."""
    values = {arm: compare.evaluate(groups, form).closed for arm, form in registered.FORMS.items()}
    effect = float(compare.combine(values["alone"] - values["zoon"])[0])
    if f"{effect:.4f}" != EXPECTED_H3:
        raise RuntimeError(f"out-of-fold H3 check failed: {effect:.4f}")
    return effect


def main() -> None:
    groups, counts = data.load(), data.held_out_counts()
    print(TITLE, flush=True)
    for line in settings():
        print(line, flush=True)
    print(f"[data] training groups = {len(groups)}", flush=True)
    print(f"[data] held-out groups = {sum(counts.values())}", flush=True)
    differences, h2, original = h2_training(groups)
    checked_training_h2(h2, original)
    print(formatted(TRAINING_H2, h2), flush=True)
    print(formatted(TRAINING_RESCORE, original), flush=True)
    laws, amounts = registered.read_fitted(), registered.read_amounts()
    simulations = {
        arm: score.simulate(
            kernel.Coded(law), amounts, open_channel=not law.candidate.twin
        ).summaries
        for arm, law in laws.items()
    }
    intervals = score.h3_interval(groups, simulations)
    h3 = score.difference(intervals["alone"], intervals["zoon"])
    # The printed row is composite only; each summary remains available to callers.
    print(formatted(TRAINING_H3, score.Interval(h3.estimate[:1], h3.values[:, :1])), flush=True)
    effect = h3_out_of_fold(groups)
    print(f"{OUT_OF_FOLD_H3} = {effect:.4f}", flush=True)
    h2_power = h2_rates(groups, differences, counts)
    simulations_pool = pool(laws, amounts)
    print(f"{POOL_H3} = {pool_truth(groups, simulations_pool, counts):.4f}", flush=True)
    h3_power = h3_rates(groups, simulations_pool, counts, training_effect=effect)
    for hypothesis, rates, trials in (("H2", h2_power, H2_TRIALS), ("H3", h3_power, H3_TRIALS)):
        for label, value in zip(RATE_LABELS, rates.values(), strict=True):
            print(f"[{hypothesis}] {label} = {formatted_rate(value, trials)}", flush=True)


if __name__ == "__main__":
    main()
