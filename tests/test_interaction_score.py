"""Literal resampling oracles, independent of the subset-sum implementation."""

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from zoon.interaction import compare, kernel, person, score
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats, Ints

SMALL_R = 7


def synthetic() -> list[Group]:
    rng = np.random.default_rng(81)
    return [
        Group(
            cell,
            str(n),
            tuple(tuple(int(rng.integers(e + 1)) for e in game.endowments) for _ in range(ROUNDS)),
        )
        for cell, game in reversed(sorted(GAMES.items()))
        for n in reversed(range(2 + game.size))
    ]


def ensembles() -> dict[str, dict[str, Floats]]:
    rng = np.random.default_rng(97)
    result: dict[str, dict[str, Floats]] = {}
    for arm in ("zoon", "alone", "prompted", "chat"):
        result[arm] = {}
        for cell in sorted(GAMES):
            values = rng.uniform(-0.2, 0.8, (100, 5))
            if cell in ("2-FE", "4-FE"):
                values[:, -1] = np.nan
            result[arm][cell] = values
    return result


def literal_h2(differences: Floats, groups: Sequence[Group], r_count: int) -> Floats:
    # Literal registration: sorted games, sorted group indices, then n_g - 1.
    rng = np.random.default_rng(20261120)
    idx: dict[str, Ints] = {}
    mine: dict[str, list[int]] = {}
    for g in sorted(GAMES):
        mine[g] = sorted(
            [i for i, group in enumerate(groups) if group.cell == g], key=lambda i: groups[i].index
        )
        n_g = len(mine[g])
        idx[g] = rng.integers(0, n_g, size=(r_count, n_g - 1))
    result = np.zeros((r_count, differences.shape[1]))
    total = sum(len(mine[g]) * GAMES[g].size * ROUNDS for g in sorted(GAMES))
    for r in range(r_count):
        for c in range(differences.shape[1]):
            for g in sorted(GAMES):
                n_g = len(mine[g])
                w_g = n_g * GAMES[g].size * ROUNDS / total
                m_g = sum(differences[mine[g][int(i)], c] for i in idx[g][r]) / (n_g - 1)
                result[r, c] += w_g * m_g
    return result


def literal_h3(
    groups: Sequence[Group],
    arms: Mapping[str, Mapping[str, Floats]],
    r_count: int,
) -> dict[str, Floats]:
    rng = np.random.default_rng(20261120)
    human: dict[str, Ints] = {}
    half: dict[str, list[Ints]] = {}
    observed: dict[str, list[Floats]] = {}
    for g in sorted(GAMES):
        mine = sorted([group for group in groups if group.cell == g], key=lambda group: group.index)
        n_g = len(mine)
        human[g] = rng.integers(0, n_g, size=(r_count, n_g - 1))
        half[g] = []
        for _r in range(r_count):
            half[g].append(rng.permutation(100)[:50])
        observed[g] = [compare.summaries(group.rounds, GAMES[g]) for group in mine]
    result = {arm: np.zeros((r_count, 6)) for arm in arms}
    for r in range(r_count):
        for arm in arms:
            for s in range(5):
                count = sum(
                    len(observed[g]) for g in sorted(GAMES) if s != 4 or g not in ("2-FE", "4-FE")
                )
                for g in sorted(GAMES):
                    if s == 4 and g in ("2-FE", "4-FE"):
                        continue
                    x = arms[arm][g][half[g][r], s]
                    # Plain pair loops, independent of compare.crps and distance matrices.
                    pair = sum(abs(x[i] - x[j]) for i in range(50) for j in range(i + 1, 50))
                    scores: list[float] = []
                    for i in human[g][r]:
                        y = observed[g][int(i)][s]
                        scores.append(sum(abs(value - y) for value in x) / 50 - pair / (50 * 49))
                    m = sum(scores) / len(scores)
                    result[arm][r, s + 1] += len(observed[g]) / count * m
            result[arm][r, 0] = sum(result[arm][r, 1:]) / 5
    return result


def test_h2_every_replicate_matches_literal_registration() -> None:
    groups = synthetic()
    differences = np.random.default_rng(53).normal(size=(len(groups), 2))
    result = score.h2_interval(differences, groups, SMALL_R)
    np.testing.assert_allclose(
        result.values, literal_h2(differences, groups, SMALL_R), atol=1e-12, rtol=0
    )
    counts = np.array([GAMES[g.cell].size * ROUNDS for g in groups])
    np.testing.assert_allclose(result.estimate, np.average(differences, axis=0, weights=counts))
    np.testing.assert_array_equal(
        result.bounds(), np.percentile(result.values, [2.5, 97.5], axis=0)
    )
    assert sum(score.h2_weights(groups).values()) == pytest.approx(1)


def test_h3_every_replicate_and_summary_matches_literal_registration() -> None:
    groups, arms = synthetic(), ensembles()
    actual = score.h3_interval(groups, arms, SMALL_R)
    expected = literal_h3(groups, arms, SMALL_R)
    for arm in arms:
        np.testing.assert_allclose(actual[arm].values, expected[arm], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            actual[arm].estimate,
            compare.combine(score.fair_scores(groups, arms[arm])),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual[arm].bounds(),
            np.percentile(expected[arm], [2.5, 97.5], axis=0),
            atol=1e-12,
            rtol=0,
        )
    np.testing.assert_allclose(sum(score.h3_weights(groups).values()), np.ones(5), atol=1e-15)
    delta = score.difference(actual["alone"], actual["zoon"])
    np.testing.assert_allclose(
        delta.values, expected["alone"] - expected["zoon"], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("threshold", [score.H2_THRESHOLD, score.H3_THRESHOLD])
def test_verdict_edges_and_missing_comparisons(threshold: float) -> None:
    assert score.comparison_verdict(threshold, 1e-15, threshold) == "passed"
    assert score.comparison_verdict(threshold, 0.0, threshold) == "failed"
    assert score.comparison_verdict(np.nextafter(threshold, -np.inf), 1.0, threshold) == "failed"
    assert score.hypothesis_verdict("passed", "passed") == "passed"
    assert score.hypothesis_verdict("passed", "failed") == "failed"
    assert score.hypothesis_verdict(None, "failed") == "no verdict"
    assert score.hypothesis_verdict("failed", None) == "no verdict"
    assert score.hypothesis_verdict("no verdict", "passed") == "no verdict"


def test_floor_is_applied_to_every_band_before_normalising_and_counted_before_floor() -> None:
    probabilities = np.array([[0.0, 0.001, 0.004, 0.005, 0.99]] * 5)
    chosen = np.arange(5)
    floored = np.maximum(probabilities, 0.005)
    np.testing.assert_array_equal(
        score.log_scores(probabilities, chosen),
        np.log(floored[np.arange(5), chosen] / floored.sum(axis=1)),
    )
    assert score.below_floor(probabilities, chosen) == 3
    assert np.exp(score.log_scores(probabilities, chosen)[0]) < 0.005
    original = np.maximum(probabilities, person.FLOOR)
    np.testing.assert_array_equal(
        score.log_scores(probabilities, chosen, person.FLOOR),
        np.log(original[np.arange(5), chosen] / original.sum(axis=1)),
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_probabilities_scores_and_defined_summaries_raise(value: float) -> None:
    probabilities = np.ones((1, 5)) / 5
    probabilities[0, 1] = value
    with pytest.raises(ValueError, match="non-finite"):
        score.log_scores(probabilities, np.array([0]))
    groups = synthetic()
    differences = np.ones((len(groups), 1))
    differences[0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        score.h2_interval(differences, groups, SMALL_R)
    arms = ensembles()
    arms["zoon"]["2-AI"][0, 4] = value
    with pytest.raises(ValueError, match="non-finite"):
        score.h3_interval(groups, arms, SMALL_R)
    arms = ensembles()
    arms["zoon"]["2-FE"][0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        score.h3_interval(groups, arms, SMALL_R)


def test_undefined_type_gap_is_only_in_the_two_equal_games() -> None:
    for cell in GAMES:
        values = np.ones((3, 5))
        if cell in ("2-FE", "4-FE"):
            with pytest.raises(ValueError, match="undefined"):
                score.validate_summaries(values, cell)
            values[:, -1] = np.nan
            score.validate_summaries(values, cell)
        else:
            values[:, -1] = np.nan
            with pytest.raises(ValueError, match="non-finite"):
                score.validate_summaries(values, cell)


def test_coded_predictions_observe_only_real_earlier_rounds() -> None:
    from tests.test_interaction_person import law

    groups = synthetic()
    fitted = law("B")
    scores = score.coded(fitted, groups)
    np.testing.assert_allclose(
        scores.logs(person.FLOOR), person.log_likelihoods(fitted, groups), atol=1e-12, rtol=0
    )
    values = np.column_stack((scores.logs(), scores.logs() * 2))
    means = score.group_means(values, scores.group, groups)
    assert means.shape == (len(groups), 2)
    with pytest.raises(ValueError, match="every decision"):
        score.group_means(values[:-1], scores.group[:-1], groups)


@pytest.mark.parametrize("opened", [False, True])
def test_registered_simulation_streams_logs_summaries_and_sizes(
    monkeypatch: pytest.MonkeyPatch,
    opened: bool,
) -> None:
    seen: list[tuple[str, str, int]] = []
    amounts = kernel.Amounts({})

    def agent(view: kernel.View) -> Floats:
        return np.array([1.0, 0, 0, 0, 0])

    def play(
        game: Game,
        agents: Sequence[kernel.Agent],
        table: kernel.Amounts,
        seed: int,
        group: str,
        *,
        open_channel: bool = True,
    ) -> kernel.Run:
        assert open_channel == opened
        assert all(a is agent for a in agents)
        assert table is amounts
        seen.append((game.cell, group, seed))
        return kernel.Run(((0,) * game.size,) * ROUNDS, (group,))

    monkeypatch.setattr(kernel, "play", play)
    result = score.simulate(agent, amounts, open_channel=opened)
    assert seen == [
        (cell, f"registered|{cell}|{n}", 20261120) for cell in sorted(GAMES) for n in range(100)
    ]
    for cell in GAMES:
        assert len(result.logs[cell]) == 100
        assert result.summaries[cell].shape == (100, 5)
        np.testing.assert_allclose(
            result.summaries[cell][0],
            compare.summaries(result.logs[cell][0].rounds, GAMES[cell]),
            equal_nan=True,
        )


def test_interval_constants_are_bound_to_the_registration() -> None:
    import re

    from zoon.interaction import data, prompted, registered

    text = (data.REPO / "interaction" / "registration.md").read_text()
    quoted = re.search(r"`R = ([\d_]+)`", text)
    assert quoted and int(quoted[1].replace("_", "")) == score.RESAMPLES
    calls = re.findall(r"rng\.permutation\((\d+)\)\[:(\d+)\]", text)
    assert calls == [(str(score.SIMULATED_GROUPS), str(score.HALF_ENSEMBLE))]
    assert "size=(R, n_g - 1)" in text and score.HUMAN_DEFICIT == 1
    assert f"at least {score.H2_THRESHOLD}" in text
    assert f"at least {score.H3_THRESHOLD}" in text
    assert score.PERCENTILES == (2.5, 97.5)
    assert "[2.5, 97.5]" in text
    assert score.SCORING_FLOOR == prompted.PROMPTED_FLOOR
    assert score.RESCORING_FLOOR == person.FLOOR
    assert f"`{registered.PROBABILITY_TOLERANCE:.0e}".replace("e-03", "e-3") in text


def test_fair_point_scores_allow_one_human_but_intervals_need_two() -> None:
    groups = [Group("2-AI", "one", ((18, 6),) * ROUNDS)]
    simulations = ensembles()["zoon"]
    actual = score.fair_scores(groups, simulations)
    human = compare.summaries(groups[0].rounds, GAMES["2-AI"])
    expected = [compare.crps(simulations["2-AI"][:, s], float(y)) for s, y in enumerate(human)]
    np.testing.assert_allclose(actual[0], expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="at least two groups"):
        score.h3_interval(groups, {"zoon": simulations}, SMALL_R)
    with pytest.raises(ValueError, match="at least two groups"):
        score.h2_interval(np.ones((1, 1)), groups, SMALL_R)
