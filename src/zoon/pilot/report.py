"""Print the pilot's numbers. Run as `uv run python -m zoon.pilot.report power` or
`uv run python -m zoon.pilot.report lean`.

Every number quoted in `docs/pilot/power.md` and `docs/pilot/lean.md` is printed here, on a
labelled line.
"""

import math
import sys

import numpy as np

from zoon.pilot import fetch, lean, outside, power
from zoon.pilot.models import (
    LEARNING_RATE_BOUNDS,
    PROBABILITY_FLOOR,
    TEMPERATURE_BOUNDS,
    TEMPERATURE_PRIOR_SCALE,
    Bools,
    Floats,
    Sequences,
    fit_constant,
    fit_lean,
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
        product = fit.learning_rate * fit.temperature
        print(f"{label} | fitted learning rate x temperature = {product:.4f}")
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
    print(f"temperature prior = lognormal(0, {TEMPERATURE_PRIOR_SCALE}), as in the paper's model")
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


SIGN, NORM, CODED = "sign", "norm", "coded"
GERMAN, FORCED, TASK_CODES = "German sample", "penalty forced finite", "task authors' codes"
PRIMARY, CHECK = 2, 1  # the US sample is primary, the German sample is the check
LEARNING = lean.LEARNING
Pair = tuple[str, str]  # (first model, second model): the effect is the first minus the second

READING: tuple[Pair, ...] = (
    (NORM, lean.POOLED),
    (CODED, lean.POOLED),
    (NORM, SIGN),
    (CODED, SIGN),
    (lean.KNOWN, lean.POOLED),
)
ON_TOP: tuple[Pair, ...] = (
    (NORM + LEARNING, NORM),
    (CODED + LEARNING, CODED),
    (lean.POOLED + LEARNING, lean.POOLED),
)
WHOLE: tuple[Pair, ...] = (
    (NORM + LEARNING, lean.POOLED + LEARNING),
    (CODED + LEARNING, lean.POOLED + LEARNING),
)
# Comparisons with the `sign` lean, which separate what m(d) adds from what s(d) adds.
# Printed in short form, and for the pooled group only.
WITH_SIGN: dict[str, tuple[Pair, ...]] = {
    "reading, no learning": ((SIGN, lean.POOLED),),
    "learning on top of a lean": ((SIGN + LEARNING, SIGN),),
    "the whole thing": ((NORM + LEARNING, SIGN + LEARNING),),
}
GROUPS: tuple[tuple[str, tuple[Pair, ...]], ...] = (
    ("reading, no learning", READING),
    ("learning on top of a lean", ON_TOP),
    ("the whole thing", WHOLE),
)


def _number(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def _correlation(a: Floats, b: Floats) -> str:
    if a.std() == 0 or b.std() == 0:
        return "undefined, m(d) is zero for every dilemma"
    return f"{float(np.corrcoef(a, b)[0, 1]):+.3f}"


def _compare(
    label: str, scores: lean.Scores, pair: Pair, data: Sequences, held_out: int, full: bool
) -> None:
    """Print one comparison: the effect, its uncertainty over participants, and what the
    held-out participants could detect. In full, also its spread, the smallest detectable
    effect, trial 1, and its uncertainty over dilemmas."""
    first, second = pair
    label = f"{label} | {first} minus {second}"
    gain = scores.log_likelihood[first] - scores.log_likelihood[second]
    effect = power.summarise(gain)
    print(f"{label} | effect = {effect.mean:+.5f} nats/choice")
    if not gain.any():
        print(f"{label} | the two are the same model, so there is nothing to detect")
        return
    print(f"{label} | effect 95% interval = {_interval(effect.interval)}")
    if full:
        print(f"{label} | sd across participants = {effect.sd:.5f}")
        print(f"{label} | detectable effect at 80% power = {effect.detectable(held_out):.5f}")
    print(f"{label} | power for the observed effect = {effect.power(held_out):.3f}")
    print(f"{label} | power for half the observed effect = {effect.power(held_out, 0.5):.3f}")
    later = power.summarise(gain[:, 1:])
    if full:
        print(f"{label} | effect on trial 1 = {gain[:, 0].mean():+.5f}")
    print(f"{label} | effect on trials 2-13 = {later.mean:+.5f}")
    if not full:
        return
    print(f"{label} | effect on trials 2-13 95% interval = {_interval(later.interval)}")
    each = lean.by_dilemma(gain, data)
    print(f"{label} | dilemmas with a positive effect = {int((each > 0).sum())} of {len(each)}")
    across = lean.dilemma_interval(gain, data)
    print(f"{label} | effect 95% interval, resampling dilemmas = {_interval(across)}")


def _fitted_lean(scope: str, name: str, data: Sequences, measure: Floats, rows: Floats) -> None:
    """Print alpha and beta fitted to everyone in the group, and how they vary over the
    cross-validation fits, whose rows hold alpha and beta in the first two columns."""
    everything = np.ones(data.choice.shape, dtype=np.bool_)
    learning = name.endswith(LEARNING)
    fit = fit_lean(data, everything, lean.SIGN, measure, learning)
    label = f"[{scope}] {name}"
    for column, (symbol, value) in enumerate((("alpha", fit.alpha), ("beta", fit.beta))):
        if symbol == "beta" and not measure.any():
            continue  # with m(d) = 0 there is no beta to fit
        spread = rows[:, column]
        print(f"{label} | {symbol}, fitted to all of the group = {value:+.4f}")
        print(
            f"{label} | {symbol} over the {len(rows)} cross-validation fits: "
            f"mean = {spread.mean():+.4f}, sd = {spread.std(ddof=1):.4f}, "
            f"min = {spread.min():+.4f}, max = {spread.max():+.4f}"
        )
    if learning:
        rate, temperature = fit.parameters.learning_rate, fit.parameters.temperature
        print(f"{label} | fitted learning rate = {rate:.4f}")
        print(f"{label} | fitted temperature = {temperature:.4f}")
        print(f"{label} | fitted learning rate x temperature = {rate * temperature:.4f}")


def _map(study: int, dilemmas: dict[str, outside.Dilemma], rated: outside.Rated) -> None:
    """Print how well the published codes predict the published judgments among the
    dilemmas that are not the task's."""
    label = f"[coded map, {outside.STUDIES[study].label} sample]"
    originals = outside.originals()
    names = lean.outside_names(dilemmas, originals)
    features, target = lean.design(dilemmas, names, rated[study])
    chosen = lean.fit_map(dilemmas, originals, rated[study])
    forced = lean.fit_map(dilemmas, originals, rated[study], lean.FINITE_PENALTIES)
    others = np.array([np.delete(target, i).mean() for i in range(len(target))])

    def error(predictions: Floats) -> float:
        return float(np.sqrt(((predictions - target) ** 2).mean()))

    nested = lean.left_out_predictions(features, target)
    finite = lean.left_out_predictions(features, target, (forced.penalty,))
    print(f"{label} sd of the mean judgments of the dilemmas fitted on = {target.std():.3f}")
    print(f"{label} penalty chosen by leave-one-out = {_number(chosen.penalty)}")
    print(f"{label} left-out error, predicting the mean of the others = {error(others):.3f}")
    print(f"{label} left-out error, penalty chosen inside each fit = {error(nested):.3f}")
    print(
        f"{label} left-out correlation, penalty chosen inside each fit = "
        f"{_correlation(nested, target)}"
    )
    print(f"{label} best finite penalty = {_number(forced.penalty)}")
    print(f"{label} left-out error at the best finite penalty = {error(finite):.3f}")
    print(
        f"{label} left-out correlation at the best finite penalty = {_correlation(finite, target)}"
    )
    task = np.array([outside.encode(codes) for codes in outside.original_codes().values()])
    for name, fitted in (("penalty chosen", chosen), ("best finite penalty", forced)):
        predicted = fitted.predict(task)
        print(
            f"{label} predicted rating of the task's 13, {name}: "
            f"sd = {predicted.std():.3f}, min = {predicted.min():.3f}, "
            f"max = {predicted.max():.3f}"
        )
    if study != PRIMARY:
        return
    for column, feature in enumerate(outside.FEATURES):
        print(
            f"{label} code: {feature} | above zero in {int((features[:, column] > 0).sum())} "
            f"of the {len(names)} fitted on and {int((task[:, column] > 0).sum())} of the "
            f"task's {len(task)} | weight at the best finite penalty = "
            f"{forced.weights[column]:+.4f}"
        )


def run_lean() -> None:
    fetch.fetch()
    outside.fetch()
    table = fetch.load()
    held_out = fetch.held_out_counts()
    people = power.participants(table)
    everyone = power.sequences(table)

    dilemmas = outside.published()
    originals = outside.originals()
    reversed_ = outside.action_reversed()
    rated = outside.ratings()
    outside.check_against_paper(rated, dilemmas)
    names = lean.outside_names(dilemmas, originals)
    factual = [d for d in dilemmas.values() if d.codes is not None]
    sets = len(power.held_out_sets(True, len(fetch.DILEMMAS)))

    print("zoon pilot: a lean from published human norms and codes, training participants only")
    print("no language model is involved")
    print(
        "model: logit P(cost-benefit option) = s(d) * (alpha + beta * m(d)) + tau * (v[1] - v[0])"
    )
    print("  s(d)         +1 where the suggested action is the cost-benefit option, -1 where it")
    print("               is the rule option; the sign multiplies the whole bracket")
    print("  pooled       no lean: a dilemma that was not fitted takes the share over the fitted")
    print("               dilemmas (the existing baseline)")
    print("  sign         m(d) = 0: only which option the question asks about")
    print("  norm         m(d) = mean appropriateness of the original dilemma in an outside")
    print("               sample, standardised over the 13; human judgments of nearly the same")
    print("               text, which a reader of the text does not have")
    print("  coded        m(d) = that judgment predicted from the original paper's codes by a")
    print("               ridge map fitted on other dilemmas, standardised over the 13")
    print("  known share  each dilemma's own share among the fitting participants; a ceiling,")
    print("               not available for a dilemma nobody has answered")
    print("  + learning   tau and the learning rate are fitted together with alpha and beta")
    print("effect = mean log-likelihood per choice, first model minus second, each choice scored")
    print("         by fits that saw neither the participant nor the dilemma (known share: not")
    print("         the participant)")
    print(f"seed = {power.SEED}")
    print(f"held-out fraction aimed for = {HELD_OUT_FRACTION}")
    print(f"participant folds = {power.FOLDS}")
    print(f"dilemmas = {len(fetch.DILEMMAS)}")
    print(f"dilemmas held out at a time = {power.HELD_OUT_DILEMMAS}")
    print(f"times each dilemma is held out = {power.HELD_OUT_DILEMMAS}")
    print(f"cross-validation fits per model = {power.FOLDS * sets}")
    print(f"learning rate bounds = {list(LEARNING_RATE_BOUNDS)}")
    print(f"temperature bounds = {list(TEMPERATURE_BOUNDS)}")
    print(f"temperature prior = lognormal(0, {TEMPERATURE_PRIOR_SCALE}), as in the paper's model")
    print(f"bootstrap resamples = {power.RESAMPLES}")
    print(f"probability clamp = [{PROBABILITY_FLOOR}, {1 - PROBABILITY_FLOOR}]")
    print(f"test = paired t across participants, two-sided alpha = {power.ALPHA}")
    print(f"target power = {power.TARGET_POWER}")
    print(f"ridge penalties tried = {', '.join(_number(p) for p in lean.PENALTIES)}")
    print()

    print("[outside] source = Körner and Deutsch 2022, https://osf.io/cg5tq/")
    tables = {1: "Tables 3 and 4", 2: "Tables 7 and 8"}
    for study, (gap, where) in outside.discrepancy(rated, dilemmas).items():
        label = f"[outside] study {study}, {outside.STUDIES[study].label} sample"
        counts = [rating.count for rating in rated[study].values()]
        own = [rated[study][d.name].count for d in factual]
        print(f"{label} | file = {outside.STUDIES[study].url}")
        print(f"{label} | dilemmas with a judgment = {len(counts)}")
        print(f"{label} | judgments per dilemma = {min(counts)} to {max(counts)}")
        print(f"{label} | judgments per factual dilemma = {min(own)} to {max(own)}")
        print(
            f"{label} | largest difference between a mean computed from the file and the "
            f"paper's {tables[study]} = {gap:.4f} ({where})"
        )
    print(f"[outside] difference that rounding to two decimals allows = {outside.ROUNDING}")
    print()

    for name, sign in zip(fetch.DILEMMAS, lean.SIGN, strict=True):
        turned = "yes" if reversed_[name] else "no"
        print(
            f"[mapping] {name} | original = {originals[name]} | s = {sign:+.0f} "
            f"| suggested action reversed in adaptation = {turned}"
        )
    print()

    print(f"[coded map] factual dilemmas, with published codes = {len(factual)}")
    print(f"[coded map] of those, originals of the task's dilemmas, left out = {len(originals)}")
    print(
        f"[coded map] trolley-type dilemmas, without published codes, left out = "
        f"{len(dilemmas) - len(factual)}"
    )
    print(f"[coded map] dilemmas fitted on = {len(names)}")
    for start in range(0, len(names), 5):
        print(f"[coded map] fitted on: {'; '.join(names[start : start + 5])}")
    print(f"[coded map] codes = {len(outside.FEATURES)}")
    _map(PRIMARY, dilemmas, rated)
    _map(CHECK, dilemmas, rated)
    print()

    codes = outside.original_codes()
    blank = np.zeros(len(fetch.DILEMMAS))
    leans: dict[str, lean.Lean] = {
        SIGN: lean.Lean(blank, blank),
        NORM: lean.norm_lean(originals, rated[PRIMARY], reversed_),
        CODED: lean.coded_lean(lean.fit_map(dilemmas, originals, rated[PRIMARY]), codes),
    }
    forced = lean.fit_map(dilemmas, originals, rated[PRIMARY], lean.FINITE_PENALTIES)
    checks: dict[str, lean.Lean] = {
        f"{NORM}, {GERMAN}": lean.norm_lean(originals, rated[CHECK], reversed_),
        f"{CODED}, {GERMAN}": lean.coded_lean(
            lean.fit_map(dilemmas, originals, rated[CHECK]), codes
        ),
        f"{CODED}, {FORCED}": lean.coded_lean(forced, codes),
        f"{CODED}, {FORCED}, {TASK_CODES}": lean.coded_lean(forced, outside.task_codes()),
    }

    share = lean.shares(everyone)
    logit = np.log(share / (1 - share))
    german = checks[f"{NORM}, {GERMAN}"]
    finite = checks[f"{CODED}, {FORCED}"]
    print("[dilemmas] rating = mean appropriateness of the original's suggested action, 1 to 7,")
    print("[dilemmas]          computed from the outside files")
    print("[dilemmas] m      = the rating of the task's suggested action, standardised over the")
    print("[dilemmas]          13; where the action was reversed the rating is reflected first")
    print("[dilemmas] share  = share of training participants choosing the cost-benefit option")
    print(
        f"[dilemmas] {'dilemma':<11}{'s':>3}{'rating US':>11}{'rating German':>15}"
        f"{'m norm':>9}{'m norm German':>15}{'m coded':>9}{'m coded forced':>16}{'share':>8}"
    )
    for d, name in enumerate(fetch.DILEMMAS):
        original = originals[name]
        print(
            f"[dilemmas] {name:<11}{lean.SIGN[d]:>+3.0f}{rated[PRIMARY][original].mean:>11.2f}"
            f"{rated[CHECK][original].mean:>15.2f}{leans[NORM].measure[d]:>+9.3f}"
            f"{german.measure[d]:>+15.3f}{leans[CODED].measure[d]:>+9.3f}"
            f"{finite.measure[d]:>+16.3f}{share[d]:>8.3f}"
        )
    kept = np.array([not reversed_[name] for name in fetch.DILEMMAS])
    for name, one in (leans | checks).items():
        if name == SIGN:
            continue
        signed = lean.SIGN * one.measure
        label = f"[dilemmas] correlation over the 13 of s(d) * m(d) with logit share | {name}"
        print(f"{label} = {_correlation(signed, logit)}")
        if name.startswith(NORM):
            without = _correlation(signed[kept], logit[kept])
            print(f"{label}, without the reversed dilemma = {without}")
    print(
        f"[dilemmas] correlation over the 13 of s(d) with logit share = "
        f"{_correlation(lean.SIGN, logit)}"
    )
    print()

    scopes: dict[str, tuple[Bools, int]] = {
        "pooled": (np.ones(len(people), dtype=np.bool_), sum(held_out.values()))
    }
    for e, count in sorted(held_out.items()):
        scopes[f"experiment {e}"] = ((people["experiment"] == e).to_numpy(), count)
    for scope, (members, count) in scopes.items():
        print(f"[{scope}] training participants = {members.sum()}")
        print(f"[{scope}] held-out participants = {count}")
    print()

    measures = {name: one.measure for name, one in leans.items()}
    for scope, (members, count) in scopes.items():
        data = everyone.take(members)
        scores = lean.cross_validate(data, lean.SIGN, measures)
        whole = scope == "pooled"
        if whole:
            for name in (NORM, NORM + LEARNING, SIGN, SIGN + LEARNING):
                measure = measures[name.removesuffix(LEARNING)]
                _fitted_lean(scope, name, data, measure, scores.fitted[name])
            if not measures[CODED].any():
                print(f"[{scope}] {CODED} | m(d) is zero for every dilemma: this is `{SIGN}`")
        for group, pairs in GROUPS:
            for pair in pairs:
                _compare(f"[{scope}] {group}", scores, pair, data, count, full=whole)
            if whole:
                for pair in WITH_SIGN[group]:
                    _compare(f"[{scope}] {group}", scores, pair, data, count, full=False)
        print()
        if not whole:
            continue
        likelihood = scores.log_likelihood
        reached = (likelihood[NORM] - likelihood[lean.POOLED]).mean()
        ceiling = (likelihood[lean.KNOWN] - likelihood[lean.POOLED]).mean()
        print(
            f"[pooled] ({NORM} minus {lean.POOLED}) as a share of "
            f"({lean.KNOWN} minus {lean.POOLED}) = {reached / ceiling:.3f}"
        )
        print()
        shown = (READING[0], READING[2], READING[4], ON_TOP[0])
        columns = [lean.by_dilemma(likelihood[a] - likelihood[b], data) for a, b in shown]
        print(
            f"[pooled, by dilemma] {'dilemma':<11}{'norm - pooled':>15}{'norm - sign':>13}"
            f"{'known share - pooled':>22}{'norm + learning - norm':>24}"
        )
        for d, name in enumerate(fetch.DILEMMAS):
            print(
                f"[pooled, by dilemma] {name:<11}{columns[0][d]:>+15.5f}{columns[1][d]:>+13.5f}"
                f"{columns[2][d]:>+22.5f}{columns[3][d]:>+24.5f}"
            )
        print()
        extra = {name: one.measure for name, one in checks.items() if one.measure.any()}
        for name, one in checks.items():
            if not one.measure.any():
                print(f"[pooled] check | {name} | m(d) is zero for every dilemma: this is `sign`")
        again = lean.cross_validate(data, lean.SIGN, {SIGN: measures[SIGN]} | extra)
        for name in extra:
            for pair in (
                (name, lean.POOLED),
                (name, SIGN),
                (name + LEARNING, name),
                (name + LEARNING, lean.POOLED + LEARNING),
            ):
                _compare("[pooled] check", again, pair, data, count, full=False)
        print()


if __name__ == "__main__":
    if sys.argv[1:] == ["power"]:
        run_power()
    elif sys.argv[1:] == ["lean"]:
        run_lean()
    else:
        sys.exit("usage: python -m zoon.pilot.report power|lean")
