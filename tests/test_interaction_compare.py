"""Selection, fair scores and complete runs on groups made through the kernel."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from tests.test_interaction_kernel import training
from zoon.interaction import compare, kernel, measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS
from zoon.interaction.measure import Floats


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


def test_live_report_has_the_pasted_labels(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    groups = generated(others=1.0, count=1, seed=51)
    monkeypatch.setattr(compare.data, "load", lambda: groups)
    monkeypatch.setattr(compare.data, "held_out_counts", lambda: {cell: 3 for cell in GAMES})
    compare.main(simulations=2, resamples=20)
    printed = capsys.readouterr().out
    _, paste = split_note()
    assert labels(printed) == labels(paste)
