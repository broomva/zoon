"""Selection, fair scores and complete runs on groups made through the kernel."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from tests.test_interaction_kernel import training
from tests.test_interaction_person import law
from zoon.interaction import compare, kernel, measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats


def simulation_stand_ins(
    monkeypatch: pytest.MonkeyPatch,
    rows: Callable[[Game, str, kernel.Amounts, bool, person.Candidate], Sequence[Sequence[int]]]
    | None = None,
) -> None:
    def fitted(groups: Sequence[Group], c: person.Candidate) -> person.Person:
        return law(c.letter, c.twin)

    monkeypatch.setattr(person, "fit", fitted)
    if rows is None:
        return
    simulate = rows

    def played(
        game: Game,
        agents: Sequence[kernel.Agent],
        amounts: kernel.Amounts,
        seed: int,
        group: str,
        *,
        open_channel: bool = True,
    ) -> kernel.Run:
        assert seed == measure.SEED
        assert isinstance(agents[0], kernel.Coded)
        result = simulate(game, group, amounts, open_channel, agents[0].law.candidate)
        return kernel.Run(tuple(tuple(row) for row in result), ())

    monkeypatch.setattr(kernel, "play", played)


def generated(others: float, count: int = 8, seed: int = 31) -> list[Group]:
    """An independent last-round law. No fitting or person-state helper generates play."""
    amounts = kernel.Amounts.build(training())

    def independent(view: kernel.View) -> Floats:
        score = 0.0
        if view.history:
            previous = dict(view.history[-1].contributions)
            own = previous[view.seat] / view.game.endowments[view.seat]
            other = np.mean(
                [previous[s] / e for s, e in enumerate(view.game.endowments) if s != view.seat]
            )
            score = 2 * (own - 0.5) + others * (other - 0.5)
        cuts = np.array([-1.5, -0.5, 0.5, 1.5])
        return np.diff(np.r_[0.0, expit(cuts - score), 1.0])

    return [
        Group(
            cell,
            f"{cell}-{g}",
            kernel.play(game, [independent] * game.size, amounts, seed, f"{cell}-{g}").rounds,
        )
        for cell, game in GAMES.items()
        for g in range(count)
    ]


def test_selection_uses_earliest_reference_then_count_then_score_then_letter() -> None:
    group = np.arange(4, dtype=np.int64)
    # Exact top scores: input insertion order must not select B as the reference.
    scores = {
        "B": np.array([1.0, 0.0, 1.0, 0.0]),
        "A": np.array([0.0, 1.0, 0.0, 1.0]),
        "C": np.array([-0.1, 0.9, -0.1, 0.9]),
        "D": np.full(4, -3.0),
    }
    selected = compare.select(scores, {"A": 5, "B": 5, "C": 4, "D": 1}, group, 4)
    assert selected.reference == "A"
    # C is always worse than A: its interval cannot include zero.
    assert selected.tied == ("A", "B") and selected.winner == "A"
    scores["C"] = np.array([-0.3, 1.1, -0.3, 1.1])
    selected = compare.select(scores, {"A": 5, "B": 5, "C": 4, "D": 1}, group, 4)
    assert selected.tied == ("A", "B", "C") and selected.winner == "C"
    # With equal counts, the higher score wins; with equal scores the earlier letter wins.
    scores["B"] += 0.02
    selected = compare.select(scores, {name: 5 for name in scores}, group, 4)
    assert selected.winner == "B"
    scores["B"] -= 0.02
    assert compare.select(scores, {name: 5 for name in scores}, group, 4).winner == "A"


def test_fair_crps_against_distinct_pairs_and_count_independence() -> None:
    for values in ((0.1, 0.8), (0.2, 0.2, 0.9), (-1.1, 0.4, 0.5, 0.8, 1.2)):
        sample = np.array(values)
        y = 0.3
        brute = sum(abs(x - y) for x in values) / len(values) - sum(
            abs(a - b) for i, a in enumerate(values) for j, b in enumerate(values) if i != j
        ) / (2 * len(values) * (len(values) - 1))
        assert compare.crps(sample, y) == pytest.approx(brute)
    # Enumerate every iid binary sample. Expected score must not change with sample count.
    from itertools import product

    expected: list[float] = []
    for n in (2, 3, 5):
        expected.append(
            float(
                np.mean(
                    [compare.crps(np.array(v, dtype=float), 0.3) for v in product((0, 1), repeat=n)]
                )
            )
        )
    assert expected == pytest.approx([0.25] * 3)
    with pytest.raises(ValueError, match="two finite"):
        compare.crps(np.array([0.2]), 0.3)


def test_combine_summaries_before_averaging_with_undefined_groups() -> None:
    differences = np.array([[0.1] * 4 + [0.2], [0.3] * 4 + [0.6], [0.5] * 4 + [np.nan]])
    assert compare.combine(differences) == pytest.approx([0.32, 0.3, 0.3, 0.3, 0.3, 0.4])


def test_all_summaries_match_hand_worked_exact_shares() -> None:
    # A productivity-only difference defines the type gap even with equal endowments.
    for cell in ("2-PI", "2-FE"):
        game = GAMES[cell]
        rounds = [(0, 24)] * 10 + [(12, 6)] * 10
        result = compare.summaries(rounds, game)
        assert result[:4] == pytest.approx([0.4375, 0.625, -0.125, 1.25 / 38])
        if cell == "2-PI":
            assert result[4] == pytest.approx(-0.375)
        else:
            assert np.isnan(result[4])
    game = GAMES["4-AI"]
    result = compare.summaries([(36, 18, 0, 12)] * ROUNDS, game)
    assert result == pytest.approx([0.625, 7 / 12, 0, 0, 0.25])


def test_intervals_resample_within_game_and_preserve_summary_missingness() -> None:
    groups = [Group("2-PI", "a", ()), Group("2-PI", "b", ()), Group("2-FE", "c", ())]
    values = np.array([[0.1] * 4 + [0.2], [0.3] * 4 + [0.6], [0.5] * 4 + [np.nan]])
    estimate = compare.interval(values, groups, resamples=500)
    assert estimate[:, 0] == pytest.approx(compare.combine(values))
    assert estimate[1, 1:] == pytest.approx([7 / 30, 11 / 30])
    assert estimate[5, 1:] == pytest.approx([0.2, 0.6])
    assert np.array_equal(estimate, compare.interval(values, groups, resamples=500))


def test_folds_keep_both_fits_and_amount_tables_away_from_the_group() -> None:
    groups = generated(others=2, count=2)
    target = 0
    changed = list(groups)
    changed[target] = replace(
        groups[target], rounds=tuple(tuple(0 for _ in row) for row in groups[target].rounds)
    )
    c = person.Candidate("B")
    before, after = (
        compare.evaluate(groups, c, simulations=2),
        compare.evaluate(changed, c, simulations=2),
    )
    rows = measure.decisions(groups).group
    fold = np.array([measure.fold_of(g) for g in groups])
    same = (fold[rows] == fold[target]) & (rows != target)
    assert same.any()
    assert np.array_equal(before.open[same], after.open[same])
    # Equal NaNs are structural: FE has no type gap.
    other = (fold == fold[target]) & (np.arange(len(groups)) != target)
    assert np.array_equal(before.closed[other], after.closed[other], equal_nan=True)
    assert np.array_equal(before.covered[other], after.covered[other], equal_nan=True)
    assert not np.allclose(
        before.open[fold[rows] != fold[target]], after.open[fold[rows] != fold[target]]
    )
    assert not np.allclose(
        before.closed[fold != fold[target]], after.closed[fold != fold[target]], equal_nan=True
    )


def score_candidates(groups: Sequence[Group]) -> tuple[compare.Selection, dict[str, Floats]]:
    scores: dict[str, Floats] = {}
    for letter in person.LETTERS:
        scores[letter] = person.out_of_sample(groups, person.Candidate(letter))
    d = measure.decisions(groups)
    selection = compare.select(
        scores, {c.letter: c.count for c in person.CANDIDATES if not c.twin}, d.group, len(groups)
    )
    return selection, scores


@pytest.mark.parametrize("others,seed", ((0.0, 21), (4.0, 22)))
def test_independent_and_following_groups_are_controls_in_both_loops(
    others: float, seed: int
) -> None:
    groups = generated(others=others, count=4, seed=seed)
    selection, scores = score_candidates(groups)
    c = person.Candidate(selection.winner)
    simulations = 12 if others == 0 else 4
    a, b = (
        compare.evaluate(groups, c, simulations=simulations),
        compare.evaluate(groups, replace(c, twin=True), simulations=simulations),
    )
    assert a.open == pytest.approx(scores[c.letter])
    d = measure.decisions(groups)
    h2 = measure.paired(a.open - b.open, d.group, len(groups))
    h3 = compare.interval(b.closed - a.closed, groups, resamples=200)[0]
    if others == 0:
        assert h2[0] < 0.01 and h2[1] <= 0
        assert h3[0] < 0.01 and h3[1] <= 0
    else:
        assert h2[1] > 0.01
        assert h3[1] > 0


def split_note() -> tuple[str, str]:
    text = (Path(__file__).parent.parent / "docs/interaction/candidates.md").read_text()
    prefix, paste = text.rsplit("\n```\n", 2)[:2]
    return prefix, paste


def labels(text: str) -> list[str]:
    return [line.split(" = ")[0] for line in text.splitlines() if line]


def test_live_report_labels_and_power_use_defined_training_and_sealed_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    groups = generated(others=1.0, count=2, seed=51)
    monkeypatch.setattr(compare.data, "load", lambda: groups)
    monkeypatch.setattr(compare.data, "held_out_counts", lambda: {cell: 3 for cell in GAMES})
    counts: list[tuple[int, int]] = []
    real_power = measure.power

    def checked_power(
        found: tuple[float, float, float], training: int, sealed: int
    ) -> tuple[float, float]:
        counts.append((training, sealed))
        return real_power(found, training, sealed)

    monkeypatch.setattr(measure, "power", checked_power)
    compare.main(simulations=2, resamples=20)
    assert counts == ([(16, 24)] * 6 + [(12, 18)]) * 2
    printed = capsys.readouterr().out
    _, paste = split_note()
    assert labels(printed) == labels(paste)


@pytest.mark.parametrize(
    "sample,covered",
    (
        ((0.0, 0.25, 0.5, 1.0), (0.0, 1.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
    ),
)
def test_coverage_uses_the_inclusive_central_ninety_percent(
    monkeypatch: pytest.MonkeyPatch,
    sample: tuple[float, ...],
    covered: tuple[float, ...],
) -> None:
    groups = [
        Group(
            cell,
            f"{cell}-{n}",
            (tuple(e * n // 2 if n < 3 else 1 for e in game.endowments),) * ROUNDS,
        )
        for cell, game in GAMES.items()
        for n in range(4)
    ]

    def rows(
        game: Game, group: str, amounts: kernel.Amounts, opened: bool, c: person.Candidate
    ) -> Sequence[Sequence[int]]:
        fraction = sample[int(group.rsplit("|", 1)[1])]
        return (tuple(int(e * fraction) for e in game.endowments),) * ROUNDS

    simulation_stand_ins(monkeypatch, rows)
    result = compare.evaluate(groups, person.Candidate("B"), simulations=4)
    assert result.covered[:, 0] == pytest.approx(list(covered) * len(GAMES))
    assert result.covered[:, 2:4] == pytest.approx(np.ones((len(groups), 2)))
    # The productivity-only pair still has equal shares when both give one point.
    assert np.nanmean(result.covered[:, 4]) == pytest.approx(19 / 24)
    for n, group in enumerate(groups):
        if group.cell.endswith("FE"):
            assert np.isnan(result.covered[n, 4])
        observed = (
            (0.0, 0.5, 1.0)[n % 4]
            if n % 4 < 3
            else sum(1 / e for e in GAMES[group.cell].endowments) / GAMES[group.cell].size
        )
        brute = (
            sum(abs(x - observed) for x in sample) / 4
            - sum(abs(x - y) for i, x in enumerate(sample) for j, y in enumerate(sample) if i != j)
            / 24
        )
        assert result.closed[n, 0] == pytest.approx(brute)


def test_trend_uses_exactly_first_and_last_five_rounds() -> None:
    rounds = [(0, 0)] * 4 + [(24, 24)] + [(12, 12)] * 10 + [(0, 0)] + [(24, 24)] * 4
    assert compare.summaries(rounds, GAMES["2-FE"])[2] == pytest.approx(3 / 5)


def test_composite_interval_matches_enumerated_stratified_ninety_five_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from itertools import product

    groups = [Group("2-FE", str(i), ()) for i in range(2)] + [
        Group("2-PI", str(i), ()) for i in range(4)
    ]
    values = np.array(
        [
            [0, 1, 2, 3, np.nan],
            [8, 4, 0, 1, np.nan],
            [0, 1, 0, 2, 0],
            [1, 2, 1, 0, 2],
            [3, 5, 0, 1, 8],
            [20, 0, 9, 3, 32],
        ],
        dtype=float,
    )
    samples = np.array(
        [a + b for a in product(range(2), repeat=2) for b in product(range(2, 6), repeat=4)]
    )
    # Enumerate independently, giving each defined summary equal weight in each draw.
    brute = np.array(
        [
            [np.mean([row[j] for row in values[index] if np.isfinite(row[j])]) for j in range(5)]
            for index in samples
        ]
    )
    composite = brute.mean(axis=1)
    expected = np.percentile(composite, [2.5, 97.5])
    assert not np.allclose(expected, np.percentile(composite, [5, 95]))
    assert not np.allclose(expected, np.percentile(composite, [20, 80]))

    class Enumerated:
        def choice(self, mine: np.ndarray, size: tuple[int, int]) -> np.ndarray:
            assert size == (len(samples), len(mine))
            return samples[:, :2] if len(mine) == 2 else samples[:, 2:]

    def enumerated(seed: int) -> Enumerated:
        return Enumerated()

    monkeypatch.setattr(compare.np.random, "default_rng", enumerated)
    result = compare.interval(values, groups, resamples=len(samples))
    assert result[0, 1:] == pytest.approx(expected)
    assert result[1:, 1:] == pytest.approx(np.percentile(brute, [2.5, 97.5], axis=0).T)


def test_evaluate_excludes_scored_amounts_and_uses_fold_labels_and_correct_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = training()
    groups[0] = replace(groups[0], rounds=((8, 8),) * ROUNDS)
    all_rows = kernel.Amounts.build(groups).rows
    calls: list[str] = []

    def rows(
        game: Game, group: str, amounts: kernel.Amounts, opened: bool, c: person.Candidate
    ) -> Sequence[Sequence[int]]:
        fold_label, cell, index = group.split("|")
        assert fold_label in {f"fold-{measure.fold_of(g)}" for g in groups}
        fold = int(fold_label.removeprefix("fold-"))
        assert cell == game.cell
        assert index in ("0", "1")
        train = [g for g in groups if measure.fold_of(g) != fold]
        assert amounts.rows == kernel.Amounts.build(train).rows
        assert amounts.rows != all_rows
        assert opened == (not c.twin)
        calls.append(group)
        return ((0,) * game.size,) * ROUNDS

    simulation_stand_ins(monkeypatch, rows)
    for twin in (False, True):
        compare.evaluate(groups, person.Candidate("B", twin), simulations=2)
    assert calls
    assert len(calls) == 4 * len(GAMES)
    assert len(set(calls)) == 2 * len(GAMES)


def test_coverage_excludes_values_between_ninety_and_ninety_five_percent_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [Group("2-FE", str(i), ((1, 1),) * ROUNDS) for i in range(4)]

    def rows(
        game: Game, group: str, amounts: kernel.Amounts, opened: bool, c: person.Candidate
    ) -> Sequence[Sequence[int]]:
        n = int(group.rsplit("|", 1)[1])
        return ((0 if n == 0 else 12 if n == 1 else 24,) * 2,) * ROUNDS

    simulation_stand_ins(monkeypatch, rows)
    result = compare.evaluate(groups, person.Candidate("B"), simulations=4)
    assert np.percentile([0, 0.5, 1, 1], 2.5) < 1 / 24 < np.percentile([0, 0.5, 1, 1], 5)
    assert np.array_equal(result.covered[:, 0], np.zeros(4))


def test_game_left_out_fits_exclude_every_group_of_the_scored_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_interaction_person import law, tiny

    groups = tiny()
    calls: list[tuple[str, ...]] = []

    def fitted(train: Sequence[Group], c: person.Candidate) -> person.Person:
        calls.append(tuple(sorted({g.cell for g in train})))
        p = law(c.letter, c.twin)
        return replace(p, numbers=p.numbers + len(train) / 100)

    monkeypatch.setattr(person, "fit", fitted)
    c = person.Candidate("B")
    found = person.out_of_sample(groups, c, leave_out_game=True)
    observed_calls = list(calls)
    expected = np.empty_like(found)
    decisions = measure.decisions(groups)
    for cell in sorted(GAMES):
        train, test = [g for g in groups if g.cell != cell], [g for g in groups if g.cell == cell]
        mask = np.array([groups[int(i)].cell == cell for i in decisions.group])
        expected[mask] = person.log_likelihoods(fitted(train, c), test)
    assert found == pytest.approx(expected)
    assert observed_calls == [tuple(sorted(set(GAMES) - {cell})) for cell in sorted(GAMES)]


def test_report_recomputes_every_entry_and_main_only_prints_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_interaction_person import law, tiny

    groups = tiny()
    decisions = measure.decisions(groups)
    monkeypatch.setattr(compare.data, "load", lambda: groups)
    monkeypatch.setattr(compare.data, "held_out_counts", lambda: {cell: 3 for cell in GAMES})
    monkeypatch.setattr(person, "FUNCTION_TOLERANCE", 2e-10)
    calls: list[str] = []
    results: dict[str, compare.Evaluation] = {}

    def fitted(train: Sequence[Group], c: person.Candidate) -> person.Person:
        p = law(c.letter, c.twin)
        # Train composition changes both rules; a game-left-out fit differs from a fold fit.
        shift = sum((list(GAMES).index(g.cell) + 1) ** 2 for g in train) / 1000
        index = 0 if c.twin and c.letter in "CD" else person.LETTERS.index(c.letter)
        return replace(p, numbers=p.numbers + shift + index / 100 + c.twin / 1000)

    def evaluated(
        groups: Sequence[Group], c: person.Candidate, simulations: int
    ) -> compare.Evaluation:
        calls.append(c.name)
        n = person.LETTERS.index(c.letter)
        base = -0.1 if c.name == "E" else -0.4 if c.name == "B twin" else -2.0 - n / 10
        opened = base + (n + 1 + c.twin) * decisions.group / 1000
        closed = np.full((len(groups), 5), 0.1 + n / 10 + c.twin / 20)
        closed += np.arange(len(groups))[:, None] * (n + 1 + c.twin) / 1000
        covered = np.broadcast_to(
            np.arange(1, 6) / 7 + n / 100 + c.twin / 1000, closed.shape
        ).copy()
        for i, g in enumerate(groups):
            if g.cell.endswith("FE"):
                closed[i, 4] = covered[i, 4] = np.nan
        result = compare.Evaluation(opened, closed, covered, fitted(groups, c))
        results[c.name] = result
        return result

    monkeypatch.setattr(person, "fit", fitted)
    monkeypatch.setattr(compare, "evaluate", evaluated)
    power_calls: list[tuple[tuple[float, float, float], int, int]] = []
    real_power = measure.power

    def checked_power(
        found: tuple[float, float, float], training: int, sealed: int
    ) -> tuple[float, float]:
        power_calls.append((found, training, sealed))
        return real_power(found, training, sealed)

    monkeypatch.setattr(measure, "power", checked_power)

    report = compare.report(groups, {cell: 3 for cell in GAMES}, simulations=2, resamples=100)
    assert "C twin" not in calls and "D twin" not in calls
    expected: list[tuple[str, int | float | str | tuple[float, ...], str]] = []

    def add(label: str, value: int | float | str | tuple[float, ...], fmt: str = "") -> None:
        expected.append((label, value, fmt))

    add("[data] training groups", len(groups))
    add("[data] decisions", len(decisions.choice))
    add("[data] held-out groups", 3 * len(GAMES))
    add("[simulation] groups per game and fold", 2)
    for label, value, fmt in (
        ("ridge", person.RIDGE, ""),
        ("probability floor", person.FLOOR, ".10f"),
        ("log gap limit", person.LOG_GAP_LIMIT, ""),
        ("iteration limit", person.MAX_ITERATIONS, ""),
        ("gradient tolerance", person.TOLERANCE, ".8f"),
        ("function tolerance", 2e-10, ".10f"),
    ):
        add(f"[fit] {label}", value, fmt)
    add("[fit check] absolute tolerance", 0.02, ".2f")
    add("[fit check] start nudge", 1e-6, ".6f")
    from collections import Counter

    for (e, band), values in kernel.Amounts.build(groups).rows.items():
        add(
            f"[amounts, endowment {e}, band {band}] points:count",
            ", ".join(f"{x}:{n}" for x, n in sorted(Counter(values).items())),
        )
    for c in person.CANDIDATES:
        result = results["A twin" if c.twin and c.letter in "CD" else c.name]
        add(f"[count] {c.name}", c.count)
        p = fitted(groups, c)
        base = (
            ("constant", "endowment", "return", "product", "four seats")
            if c.letter == "B"
            else tuple(f"constant {cell} type {kind}" for cell, kind in measure.STRATA)
        )
        names = [
            f"{phase} {name}"
            for phase in ("first", "later")
            for name in (*base, "log gap 1", "log gap 2", "log gap 3")
        ]
        names += [f"weight {x}" for x in c.features] + [f"trace fraction {x}" for x in c.traces]
        for name, value in zip(names, p.numbers, strict=True):
            add(f"[numbers, {c.name}] {name}", float(value), "+.6f")
        add(f"[open loop] {c.name}", measure.paired(result.open, decisions.group, len(groups)))
        for metric, value in zip(
            compare.METRICS, compare.interval(result.closed, groups, 100), strict=True
        ):
            add(f"[closed loop, {metric}] {c.name}", tuple(value))
        for j, summary in enumerate(compare.SUMMARIES):
            add(f"[coverage, {summary}] {c.name}", float(np.nanmean(result.covered[:, j])), ".4f")
    selections: list[compare.Selection] = []
    for twin in (False, True):
        candidates = [c for c in person.CANDIDATES if c.twin == twin]
        selected = compare.select(
            {
                c.name: results["A twin" if c.twin and c.letter in "CD" else c.name].open
                for c in candidates
            },
            {c.name: c.count for c in candidates},
            decisions.group,
            len(groups),
        )
        selections.append(selected)
        label = "twins" if twin else "candidates"
        add(f"[selection, {label}] highest", selected.reference)
        add(f"[selection, {label}] tied at step 2", ", ".join(selected.tied))
        add(f"[selection, {label}] winner", selected.winner)
    assert [selection.winner for selection in selections] == ["E", "B twin"]
    expected_power_calls: list[tuple[tuple[float, float, float], int, int]] = []
    for comparison, opponent in (("selected twin", "B twin"), ("own twin", "E twin")):
        a, b = results["E"], results[opponent]
        h2 = measure.paired(a.open - b.open, decisions.group, len(groups))
        add(f"[gain, {comparison}, open loop] candidate minus twin", h2)
        h3 = compare.interval(b.closed - a.closed, groups, 100)
        for metric, value in zip(compare.METRICS, h3, strict=True):
            add(f"[gain, {comparison}, closed loop, {metric}] twin minus candidate", tuple(value))
        for metric, value in (("H2", h2), *zip(compare.METRICS, h3, strict=True)):
            # Two equal-seat games lack a type gap; all others define every summary.
            n = (
                sum(g.cell not in ("2-FE", "4-FE") for g in groups)
                if metric == "type gap"
                else len(groups)
            )
            sealed = 18 if metric == "type gap" else 24
            effect_interval = (float(value[0]), float(value[1]), float(value[2]))
            expected_power_calls.append((effect_interval, n, sealed))
            full, half = real_power(effect_interval, n, sealed)
            for effect, power in (("training effect", full), ("half effect", half)):
                add(f"[power, {comparison}, {metric}] {effect}", power, ".3f")
    for arm, name in (("candidate", "E"), ("twin", "B twin")):
        c = next(c for c in person.CANDIDATES if c.name == name)
        left = person.out_of_sample(groups, c, leave_out_game=True)
        add(f"[game left out, {arm}] score", measure.paired(left, decisions.group, len(groups)))
        add(
            f"[game left out, {arm}] loss",
            measure.paired(results[name].open - left, decisions.group, len(groups)),
        )
        for cell in sorted(GAMES):
            mask = np.array([groups[int(i)].cell == cell for i in decisions.group])
            add(f"[game left out, {arm}, {cell}] score", float(left[mask].mean()), "+.4f")
    assert power_calls == expected_power_calls
    # Exact ordered equality also rejects extra, missing, duplicate or reordered entries.
    assert [(entry.label, entry.value, entry.format) for entry in report] == expected
    rendered: list[str] = []
    for label, value, fmt in expected:
        if isinstance(value, tuple):
            mean, low, high = value
            rendered.append(f"{label} = {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
        else:
            rendered.append(f"{label} = {value:{fmt}}")

    def built(*args: object, **kwargs: object) -> list[compare.Entry]:
        return report

    monkeypatch.setattr(compare, "report", built)
    compare.main(simulations=2, resamples=100)
    assert (
        capsys.readouterr().out
        == "zoon interaction: candidates on training groups\n" + "\n".join(rendered) + "\n"
    )


def test_main_explains_undefined_power_for_zero_width_intervals(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_interaction_person import law, tiny

    groups = tiny()
    decisions = measure.decisions(groups)
    monkeypatch.setattr(compare.data, "load", lambda: groups)
    monkeypatch.setattr(compare.data, "held_out_counts", lambda: {cell: 3 for cell in GAMES})

    simulation_stand_ins(monkeypatch)

    def scored(
        groups: Sequence[Group], c: person.Candidate, leave_out_game: bool = False
    ) -> Floats:
        return np.zeros(len(decisions.choice))

    def evaluated(
        groups: Sequence[Group], c: person.Candidate, simulations: int
    ) -> compare.Evaluation:
        return compare.Evaluation(
            np.zeros(len(decisions.choice)),
            np.zeros((len(groups), 5)),
            np.ones((len(groups), 5)),
            law(c.letter, c.twin),
        )

    monkeypatch.setattr(person, "out_of_sample", scored)
    monkeypatch.setattr(compare, "evaluate", evaluated)
    compare.main(simulations=2, resamples=10)
    printed = capsys.readouterr().out
    assert "nan" not in printed.lower()
    power = [line for line in printed.splitlines() if line.startswith("[power,")]
    assert len(power) == 28
    assert all(line.endswith("undefined (zero-width interval)") for line in power)
