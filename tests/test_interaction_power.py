"""Power mechanics on synthetic groups; never rerun the long pasted calculation."""

import ast
import inspect
import re
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from tests.test_interaction_score import ensembles, synthetic
from zoon.interaction import compare, data, kernel, measure, person, power, registered, score
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats

NOTE = data.REPO / "docs" / "interaction" / "power.md"


def note() -> tuple[str, str]:
    text = NOTE.read_text()
    prose, paste, _ = text.rsplit("\n```\n", 2)
    return prose, paste


def test_power_table_is_the_paste_and_no_prose_digits() -> None:
    prose, paste = note()
    table = [line for line in prose.splitlines() if line.startswith("|")]
    expected = ["| Quantity | Result |", "|---|---|"]
    printed = dict(
        line.split(" = ", 1)
        for line in paste.splitlines()[1:]
        if not line.startswith(("[settings]", "[fit]"))
    )
    expected += [f"| {label} | {value} |" for label, value in printed.items()]
    assert table == expected
    prose_lines = [line for line in prose.splitlines() if line and not line.startswith("|")]
    assert "\n".join(prose_lines) == (
        "# Power under the registered rules\n"
        "The program reads training groups, held-out counts, and committed fits and amounts.\n"
        "Training groups are resampled with replacement within each game at the held-out counts\n"
        "and treated as independent. Open-loop scores use out-of-fold fits.\n"
        "Closed-loop trial scores use laws and amounts fitted on those same training groups;\n"
        "paired ensembles are drawn without replacement from a finite pool.\n"
        "Coverage is of the trial expectation. Effects are shifts of location that keep\n"
        "the full-effect variation; a real smaller effect would not keep that variation.\n"
        "The effect rates are a guide, not the power of the registered test.\n"
        "Each figure concerns one comparison with `alone`; nothing concerns the prompted arm.\n"
        "The paste records one run. A test recomputes both open-loop training interval lines\n"
        "and the in-sample closed-loop training interval line, and renders the title, settings\n"
        "and fit lines from the code; it does not recompute the other lines."
    )
    assert not re.search(r"\d", "\n".join(prose_lines))
    assert paste.splitlines()[0] == power.TITLE
    assert [line.split(" = ", 1)[0] for line in paste.splitlines()] == power.labels()
    assert [
        line for line in paste.splitlines() if line.startswith(("[settings]", "[fit]"))
    ] == power.settings()


def test_each_settings_line_resolves_its_current_named_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len({label for label, _, _ in power.SETTINGS}) == len(power.SETTINGS)
    for label, source, attribute in power.SETTINGS:
        assert isinstance(source, ModuleType)
        assert attribute.isidentifier()
        assert hasattr(source, attribute)
        with monkeypatch.context() as patch:
            patch.setattr(source, attribute, "changed source")
            assert f"[settings] {label} = changed source" in power.settings()
    # All settings really name assignments; the table cannot hold its own literal values.
    for _, source, attribute in power.SETTINGS:
        tree = ast.parse(inspect.getsource(source))
        assert any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == attribute
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            for node in tree.body
        )


def test_out_of_fold_retains_probabilities_and_uses_person_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_interaction_person import law

    groups = synthetic()
    fitted = law("B")
    calls: list[list[str]] = []

    def fit(train: list[data.Group], candidate: person.Candidate) -> person.Person:
        assert candidate == fitted.candidate
        calls.append([g.index + g.cell for g in train])
        return fitted

    monkeypatch.setattr(person, "fit", fit)
    found = power.out_of_fold(groups, fitted.candidate)
    folds = sorted({measure.fold_of(g) for g in groups})
    assert calls == [
        [g.index + g.cell for g in groups if measure.fold_of(g) != fold] for fold in folds
    ]
    np.testing.assert_allclose(found.probabilities, score.coded(fitted, groups).probabilities)
    np.testing.assert_allclose(
        found.logs(person.FLOOR), person.out_of_sample(groups, fitted.candidate), atol=1e-12, rtol=0
    )


def test_rates_use_coverage_and_both_pass_conditions() -> None:
    result = power.trial_rates(
        np.array([0.01, 0.03]), np.array([0.0, 0.01]), np.array([0.02, 0.04]), 0.02, 0.0052
    )
    assert result.coverage == 1
    assert result.null == 0
    assert result.half == 0
    assert result.full == 0.5


def test_h2_power_resamples_whole_groups_at_given_counts() -> None:
    groups = synthetic()
    counts = dict.fromkeys(GAMES, 3)
    differences = np.arange(len(groups), dtype=float)[:, None] / len(groups)
    first = power.h2_rates(groups, differences, counts, trials=3, resamples=9)
    assert first == power.h2_rates(groups, differences, counts, trials=3, resamples=9)
    assert all(0 <= v <= 1 for v in first.values())
    trial, indices = power.sample_groups(groups, counts, np.random.default_rng(21))
    for cell in GAMES:
        assert sum(g.cell == cell for g in trial) == 3
    for group, i in zip(trial, indices, strict=True):
        assert group.rounds == groups[int(i)].rounds
    assert len({g.index for g in trial}) == len(trial)


def test_pool_is_kernel_simulated_with_distinct_paired_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []
    amounts = kernel.Amounts({})
    laws = {arm: person.Person(form, form.start()) for arm, form in registered.FORMS.items()}

    def play(
        game: Game,
        agents: list[kernel.Coded],
        table: kernel.Amounts,
        seed: int,
        group: str,
        *,
        open_channel: bool = True,
    ) -> kernel.Run:
        assert open_channel == (not agents[0].law.candidate.twin)
        assert table is amounts and seed == score.SEED
        assert len(agents) == game.size
        arm = "alone" if agents[0].law.candidate.twin else "zoon"
        calls.append((arm, group, len(agents)))
        return kernel.Run(((0,) * game.size,) * ROUNDS, ())

    monkeypatch.setattr(kernel, "play", play)
    result = power.pool(laws, amounts, count=3)
    assert len(calls) == 2 * len(GAMES) * 3
    assert {group for arm, group, _ in calls if arm == "zoon"} == {
        group for arm, group, _ in calls if arm == "alone"
    }
    assert all(group.startswith("power-pool|") for _, group, _ in calls)
    for arm in result:
        for cell in GAMES:
            assert result[arm][cell].shape == (3, 5)


def test_h3_trials_draw_the_same_pool_indices_for_both_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups, source = synthetic(), ensembles()
    simulations = {arm: source[arm] for arm in ("zoon", "alone")}
    # Unique level markers reveal the selected pool indices without peeking at RNG state.
    for cell in GAMES:
        for arm, shift in (("zoon", 0), ("alone", 1)):
            simulations[arm][cell][:, 0] = np.arange(100) / 100 + shift
    original = score.h3_interval
    calls = 0

    def interval(
        groups: list[data.Group], arms: dict[str, dict[str, Floats]], resamples: int
    ) -> dict[str, score.Interval]:
        nonlocal calls
        calls += 1
        for cell in GAMES:
            np.testing.assert_allclose(
                arms["alone"][cell][:, 0] - 1, arms["zoon"][cell][:, 0], atol=1e-14
            )
        return original(groups, arms, resamples)

    monkeypatch.setattr(score, "h3_interval", interval)
    result = power.h3_rates(
        groups, simulations, dict.fromkeys(GAMES, 3), training_effect=0.01, trials=3, resamples=7
    )
    assert calls == 3
    assert all(0 <= value <= 1 for value in result.values())


def test_main_prints_current_labels_without_long_calculation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    groups = synthetic()
    interval = score.Interval(np.array([0.02]), np.full((3, 1), 0.02))
    monkeypatch.setattr(data, "load", lambda: groups)
    monkeypatch.setattr(data, "held_out_counts", lambda: dict.fromkeys(GAMES, 3))

    def training(_groups: object) -> tuple[Floats, score.Interval, score.Interval]:
        return np.ones((len(groups), 1)), interval, interval

    checked_calls: list[tuple[object, ...]] = []

    def checked(*args: object) -> None:
        checked_calls.append(args)

    def simulate(
        agent: kernel.Coded, table: kernel.Amounts, *, open_channel: bool
    ) -> score.Simulation:
        assert open_channel == (not agent.law.candidate.twin)
        return score.Simulation({}, source["alone" if agent.law.candidate.twin else "zoon"])

    def pool(*_args: object) -> dict[str, dict[str, Floats]]:
        return source

    def rates(*args: object, **kwargs: object) -> power.Rates:
        assert args[2] == dict.fromkeys(GAMES, 3)
        if kwargs:
            assert kwargs == {"training_effect": 0.0124}
        return power.Rates(1, 0, 0, 0, 0.5, 1)

    monkeypatch.setattr(power, "h2_training", training)
    monkeypatch.setattr(power, "checked_training_h2", checked)
    laws = {arm: person.Person(form, form.start()) for arm, form in registered.FORMS.items()}
    monkeypatch.setattr(registered, "read_fitted", lambda: laws)
    monkeypatch.setattr(registered, "read_amounts", lambda: kernel.Amounts({}))
    source = ensembles()
    monkeypatch.setattr(score, "simulate", simulate)
    monkeypatch.setattr(power, "pool", pool)

    def effect(_: object) -> float:
        return 0.0124

    monkeypatch.setattr(power, "h3_out_of_fold", effect)

    def truth(*args: object) -> float:
        return 0.0131

    monkeypatch.setattr(power, "pool_truth", truth)
    expected_h3 = score.Interval(np.array([0.014, 0.03]), np.array([[0.009, 0.02], [0.019, 0.04]]))

    def h3_interval(
        groups: object, simulations: object, resamples: int = score.RESAMPLES
    ) -> dict[str, score.Interval]:
        assert resamples == score.RESAMPLES
        return {"alone": expected_h3, "zoon": score.Interval(np.zeros(2), np.zeros((2, 2)))}

    monkeypatch.setattr(score, "h3_interval", h3_interval)

    def forbidden(*args: object) -> None:
        pytest.fail("committed materials must be read")

    monkeypatch.setattr(registered, "fit", forbidden)
    monkeypatch.setattr(kernel.Amounts, "build", forbidden)
    monkeypatch.setattr(power, "h2_rates", rates)
    monkeypatch.setattr(power, "h3_rates", rates)
    power.main()
    output = capsys.readouterr().out.splitlines()
    assert checked_calls == [(interval, interval)]
    assert power.formatted(power.TRAINING_H3, expected_h3) in output
    assert f"{power.OUT_OF_FOLD_H3} = 0.0124" in output
    assert f"{power.POOL_H3} = 0.0131" in output
    assert [line.split(" = ", 1)[0] for line in output] == power.labels()
    assert "[H2] pass at half the out-of-fold effect = 0.500 (se 0.002)" in output
    assert "[H3] pass at half the out-of-fold effect = 0.500 (se 0.007)" in output


def test_changed_required_training_h2_stops() -> None:
    wrong = score.Interval(np.array([0.1]), np.ones((3, 1)))
    with pytest.raises(RuntimeError, match="training H2 check failed"):
        power.checked_training_h2(wrong, wrong)


def test_committed_null_channel_controls() -> None:
    laws, amounts = registered.read_fitted(), registered.read_amounts()
    law = laws["zoon"]
    numbers = law.numbers.copy()
    numbers[2 * law.candidate.phase_size + law.candidate.features.index("others")] = 0
    null = replace(law, numbers=numbers)
    for cell in sorted(GAMES):
        game = GAMES[cell]
        for n in range(score.SIMULATED_GROUPS):
            runs = [
                kernel.play(
                    game,
                    [kernel.Coded(null) for _ in range(game.size)],
                    amounts,
                    score.SEED,
                    f"registered|{cell}|{n}",
                    open_channel=opened,
                )
                for opened in (True, False)
            ]
            assert runs[0].rounds == runs[1].rounds
            import json

            intents = [
                [
                    (row["band"], row["points"])
                    for line in run.log
                    if (row := json.loads(line))["kind"] == "intent"
                ]
                for run in runs
            ]
            assert intents[0] == intents[1]


def test_power_targets_the_printed_training_effect_rather_than_pool_truth() -> None:
    result = power.trial_rates(
        np.array([0.02]),
        np.array([0.01]),
        np.array([0.03]),
        0.02,
        0.0052,
        training_effect=0.04,
    )
    assert result.values() == (1, 0, 0, 0, 1, 1)


def test_coverage_and_one_sided_errors_partition_trials_with_inclusive_bounds() -> None:
    result = power.trial_rates(
        np.full(5, 0.02),
        np.array([0.0, 0.02, 0.021, 0.0, 0.03]),
        np.array([0.02, 0.04, 0.04, 0.019, 0.04]),
        0.02,
        0.0052,
    )
    assert result.coverage == 0.4
    assert result.lower_above_truth == 0.4
    assert result.upper_below_truth == 0.2
    assert sum(result.values()[:3]) == pytest.approx(1)


def test_trial_generator_is_independent_of_registered_interval_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert power.TRIAL_SEED != score.SEED
    groups = synthetic()
    differences = np.arange(len(groups), dtype=float)[:, None]
    counts = dict.fromkeys(GAMES, 3)
    sampled: list[list[int]] = []
    original = power.sample_groups

    def sample(
        groups: list[data.Group], counts: dict[str, int], rng: np.random.Generator
    ) -> tuple[list[data.Group], measure.Ints]:
        trial, indices = original(groups, counts, rng)
        sampled.append(indices.tolist())
        return trial, indices

    monkeypatch.setattr(power, "sample_groups", sample)
    power.h2_rates(groups, differences, counts, trials=2, resamples=7)
    first = sampled.copy()
    sampled.clear()
    monkeypatch.setattr(score, "SEED", score.SEED + 1)
    power.h2_rates(groups, differences, counts, trials=2, resamples=7)
    assert sampled == first


@pytest.mark.parametrize(
    ("value", "trials", "expected"),
    [
        (0.032, power.H2_TRIALS, "0.032 (se 0.001)"),
        (0.032, power.H3_TRIALS, "0.032 (se 0.002)"),
        (0.0, power.H2_TRIALS, "0.000 (se 0.000)"),
        (1.0, power.H3_TRIALS, "1.000 (se 0.000)"),
    ],
)
def test_formatted_rate_uses_its_trial_count(value: float, trials: int, expected: str) -> None:
    assert power.formatted_rate(value, trials) == expected


@pytest.mark.parametrize("opened", [False, True])
def test_replacing_other_seats_through_play_with_committed_law(opened: bool) -> None:
    laws, amounts = registered.read_fitted(), registered.read_amounts()
    for cell, game in GAMES.items():
        for seat in range(game.size):
            traces: list[list[Floats]] = []
            for band in (0, 4):
                focal = kernel.Coded(laws["zoon"])
                probabilities: list[Floats] = []

                def agent(
                    view: kernel.View,
                    focal: kernel.Coded = focal,
                    probabilities: list[Floats] = probabilities,
                ) -> Floats:
                    p = focal(view)
                    probabilities.append(p.copy())
                    return p

                def fixed(view: kernel.View, band: int = band) -> Floats:
                    return np.eye(5)[band]

                agents = [agent if s == seat else fixed for s in range(game.size)]
                kernel.play(
                    game, agents, amounts, score.SEED, f"control|{cell}|{seat}", open_channel=opened
                )
                traces.append(probabilities)
            if opened:
                assert not np.array_equal(traces[0], traces[1]), (cell, seat)
            else:
                np.testing.assert_array_equal(traces[0], traces[1])
            assert len(traces[0]) == ROUNDS


def test_trial_rates_threshold_decides_with_positive_lower_end() -> None:
    threshold = score.H2_THRESHOLD
    estimates = np.array([np.nextafter(threshold, -np.inf), threshold, threshold * 2])
    result = power.trial_rates(estimates, np.full(3, 0.001), np.full(3, 0.1), 0.0, threshold)
    assert result.full == pytest.approx(2 / 3)
    for estimate in estimates:
        assert bool(score.comparison_passes(np.array(estimate), np.array(0.001), threshold)) == (
            score.comparison_verdict(float(estimate), 0.001, threshold) == "passed"
        )


def test_fast_h2_matches_fifty_registered_intervals() -> None:
    groups = synthetic()
    differences = np.random.default_rng(65).normal(size=(len(groups), 1))
    counts = {cell: 2 + n for n, cell in enumerate(sorted(GAMES))}
    estimates, lows, highs = power.h2_trial_intervals(groups, differences, counts, trials=50)
    rng = np.random.default_rng(power.TRIAL_SEED)
    for n in range(50):
        trial, indices = power.sample_groups(groups, counts, rng)
        interval = score.h2_interval(differences[indices], trial)
        np.testing.assert_allclose(
            [estimates[n], lows[n], highs[n]], interval.table()[0], atol=1e-12, rtol=0
        )


def test_h2_rates_use_held_out_truth_training_effect_and_h2_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = synthetic()
    differences = np.arange(len(groups), dtype=float)[:, None] ** 2
    counts = {cell: 2 + n for n, cell in enumerate(sorted(GAMES))}
    rng = np.random.default_rng(power.TRIAL_SEED)
    expected: list[Floats] = []
    for _ in range(4):
        trial, indices = power.sample_groups(groups, counts, rng)
        expected.append(score.h2_interval(differences[indices], trial, 13).table()[0])
    weights = np.array([counts[c] * GAMES[c].size for c in sorted(GAMES)])
    means = np.array([differences[m].mean() for m in score.by_game(groups).values()])
    truth = float(np.average(means, weights=weights))
    training = float(
        sum(
            score.h2_weights(groups)[c] * differences[m, 0].mean()
            for c, m in score.by_game(groups).items()
        )
    )
    table = np.array(expected)
    captured: list[bool] = []

    def rates(
        estimates: Floats,
        lows: Floats,
        highs: Floats,
        actual_truth: float,
        threshold: float,
        training_effect: float,
    ) -> power.Rates:
        captured.append(True)
        np.testing.assert_allclose(
            np.column_stack((estimates, lows, highs)), table, atol=1e-12, rtol=0
        )
        assert actual_truth == pytest.approx(truth, abs=1e-14)
        assert training_effect == pytest.approx(training, abs=1e-14)
        assert threshold == score.H2_THRESHOLD
        return power.Rates(1, 0, 0, 0, 0, 1)

    monkeypatch.setattr(power, "trial_rates", rates)
    assert power.h2_rates(groups, differences, counts, trials=4, resamples=13).coverage == 1
    assert captured == [True]


def test_sample_groups_draws_with_replacement() -> None:
    groups = synthetic()
    counts = dict.fromkeys(GAMES, 20)
    trial, indices = power.sample_groups(groups, counts, np.random.default_rng(9))
    assert len(set(indices.tolist())) < len(indices) == len(trial)


def test_expected_training_values_are_parsed_from_registration() -> None:
    text = (data.REPO / "interaction" / "registration.md").read_text()
    match = re.search(
        r"under this scoring floor, ([\d.]+) with interval ([\d.]+) to ([\d.]+)", text
    )
    assert match and match.groups() == power.EXPECTED_H2
    match = re.search(r"with `person.FLOOR` and gives ([\d.]+)\.", text)
    assert match and match[1] == power.EXPECTED_RESCORE
    match = re.search(r"a quarter of the training estimate ([\d.]+) in\ncandidates.md", text)
    assert match and match[1] == power.EXPECTED_H3
    assert f"{power.OUT_OF_FOLD_H3} = {power.EXPECTED_H3}" in note()[1].splitlines()


def test_regenerate_training_output() -> None:
    from tests.test_interaction_training import require_training

    require_training()
    groups = data.load()
    _, h2, rescore = power.h2_training(groups)
    power.checked_training_h2(h2, rescore)
    laws, amounts = registered.read_fitted(), registered.read_amounts()
    simulations = {
        arm: score.simulate(
            kernel.Coded(law), amounts, open_channel=not law.candidate.twin
        ).summaries
        for arm, law in laws.items()
    }
    intervals = score.h3_interval(groups, simulations)
    h3 = score.difference(intervals["alone"], intervals["zoon"])
    paste = note()[1].splitlines()
    for label, interval in (
        (power.TRAINING_H2, h2),
        (power.TRAINING_RESCORE, rescore),
        (power.TRAINING_H3, h3),
    ):
        assert power.formatted(label, interval) in paste


@pytest.mark.parametrize("which", ["common", "rescore"])
def test_checked_training_h2_refuses_either_mismatch(which: str) -> None:
    common = score.Interval(np.array([0.0209]), np.array([[0.0133], [0.0281]] * 100))
    rescore = score.Interval(np.array([0.0253]), np.array([[0.0178], [0.0326]]))
    wrong = score.Interval(np.array([0.1]), np.ones((3, 1)))
    with pytest.raises(RuntimeError, match="training H2"):
        power.checked_training_h2(
            wrong if which == "common" else common, wrong if which == "rescore" else rescore
        )


def test_pool_truth_all_values_sign_and_held_out_summary_weights() -> None:
    groups = synthetic()
    arms = ensembles()
    simulations = {
        arm: {c: np.tile(v, (2, 1)) for c, v in arms[arm].items()} for arm in ("zoon", "alone")
    }
    for arm in simulations:
        for cell in GAMES:
            simulations[arm][cell][100:, :4] += 0.7 if arm == "alone" else 0.1
    counts = {c: 2 + n for n, c in enumerate(sorted(GAMES))}
    expected: list[float] = []
    for s in range(5):
        total, weight = 0.0, 0
        for cell, mine in score.by_game(groups).items():
            if s == 4 and cell in score.UNDEFINED_GAMES:
                continue
            human = [compare.summaries(groups[int(i)].rounds, GAMES[cell])[s] for i in mine]
            delta = np.mean(
                [
                    compare.crps(simulations["alone"][cell][:, s], float(y))
                    - compare.crps(simulations["zoon"][cell][:, s], float(y))
                    for y in human
                ]
            )
            total += counts[cell] * delta
            weight += counts[cell]
        expected.append(total / weight)
    assert power.pool_truth(groups, simulations, counts) == pytest.approx(
        np.mean(expected), abs=1e-14
    )


def test_h3_trials_inputs_composite_sign_threshold_and_independent_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups, source = synthetic(), ensembles()
    simulations = {arm: source[arm] for arm in ("zoon", "alone")}
    counts = dict.fromkeys(GAMES, 3)
    sampled: list[list[tuple[str, str, tuple[tuple[int, ...], ...]]]] = []

    def interval(
        trial: list[data.Group], arms: dict[str, dict[str, Floats]], resamples: int
    ) -> dict[str, score.Interval]:
        assert resamples == 11
        assert {c: len(m) for c, m in score.by_game(trial).items()} == counts
        sampled.append([(g.cell, g.index, g.rounds) for g in trial])
        for cell in GAMES:
            assert len(np.unique(arms["zoon"][cell][:, 0])) == 100
        return {
            "alone": score.Interval(np.array([0.004, 0.9]), np.tile([0.002, 0.8], (11, 1))),
            "zoon": score.Interval(np.zeros(2), np.zeros((11, 2))),
        }

    monkeypatch.setattr(score, "h3_interval", interval)

    def truth(*args: object) -> float:
        return 0.01

    monkeypatch.setattr(power, "pool_truth", truth)
    monkeypatch.setattr(score, "H3_THRESHOLD", 0.0031)
    result = power.h3_rates(
        groups, simulations, counts, training_effect=0.01, trials=3, resamples=11
    )
    assert result == power.trial_rates(
        np.full(3, 0.004), np.full(3, 0.002), np.full(3, 0.002), 0.01, 0.0031, training_effect=0.01
    )
    first = sampled.copy()
    sampled.clear()
    monkeypatch.setattr(score, "SEED", score.SEED + 1)
    power.h3_rates(groups, simulations, counts, training_effect=0.01, trials=3, resamples=11)
    assert first == sampled


def test_formatted_interval_has_point_lower_upper() -> None:
    assert (
        power.formatted("example", score.Interval(np.array([3.0]), np.array([[1.0], [5.0]])))
        == "example = 3.0000 [1.1000, 4.9000]"
    )


def test_h3_out_of_fold_reuses_comparison_and_checks_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = synthetic()
    seen: list[person.Candidate] = []

    def evaluate(g: object, form: person.Candidate) -> compare.Evaluation:
        assert g is groups
        seen.append(form)
        # The composite differs from the level alone.
        closed = (
            np.tile([0.002, 0.02, 0.01, 0.02, 0.01], (len(groups), 1))
            if form.twin
            else np.zeros((len(groups), 5))
        )
        closed[[g.cell in score.UNDEFINED_GAMES for g in groups], -1] = np.nan
        covered = np.full_like(closed, 0.9 if form.twin else 0.1)
        covered[np.isnan(closed)] = np.nan
        return compare.Evaluation(np.zeros(1), closed, covered, person.Person(form, form.start()))

    monkeypatch.setattr(compare, "evaluate", evaluate)
    assert power.h3_out_of_fold(groups) == pytest.approx(0.0124)
    assert seen == [registered.FORMS["zoon"], registered.FORMS["alone"]]
    monkeypatch.setattr(power, "EXPECTED_H3", ".9999")
    with pytest.raises(RuntimeError, match="out-of-fold H3"):
        power.h3_out_of_fold(groups)


def test_training_h2_forms_floor_sign_and_registered_replicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = synthetic()
    identities = np.repeat(np.arange(len(groups)), [ROUNDS * GAMES[g.cell].size for g in groups])
    chosen = np.zeros(len(identities), dtype=np.int64)
    probabilities = [
        np.tile(row, (len(identities), 1))
        for row in ([0.001, 0.199, 0.2, 0.2, 0.4], [0.002, 0.198, 0.2, 0.2, 0.4])
    ]
    decisions = [score.Decisions(identities, chosen, p) for p in probabilities]

    def out_of_fold(g: object, form: person.Candidate) -> score.Decisions:
        assert g is groups
        return decisions[1 if form.twin else 0]

    monkeypatch.setattr(power, "out_of_fold", out_of_fold)
    common, h2, rescore = power.h2_training(groups)
    expected = score.group_means(
        (decisions[0].logs() - decisions[1].logs())[:, None], identities, groups
    )
    original = score.group_means(
        (decisions[0].logs(person.FLOOR) - decisions[1].logs(person.FLOOR))[:, None],
        identities,
        groups,
    )
    np.testing.assert_array_equal(common, expected)
    np.testing.assert_array_equal(h2.table(), score.h2_interval(expected, groups).table())
    np.testing.assert_array_equal(rescore.table(), score.h2_interval(original, groups).table())
    assert len(h2.values) == score.RESAMPLES == len(rescore.values)


def test_h2_batches_match_single_batch_and_literal_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = synthetic()
    differences = np.random.default_rng(65).normal(size=(len(groups), 1))
    counts = {cell: 2 + n for n, cell in enumerate(sorted(GAMES))}
    trials, resamples = 7, 13
    monkeypatch.setattr(power, "H2_BATCH", trials)
    single = power.h2_trial_intervals(
        groups, differences, counts, trials=trials, resamples=resamples
    )
    monkeypatch.setattr(power, "H2_BATCH", 2)
    batched = power.h2_trial_intervals(
        groups, differences, counts, trials=trials, resamples=resamples
    )
    rng = np.random.default_rng(power.TRIAL_SEED)
    expected: list[Floats] = []
    for _ in range(trials):
        trial, indices = power.sample_groups(groups, counts, rng)
        expected.append(score.h2_interval(differences[indices], trial, resamples).table()[0])
    for actual in (single, batched):
        np.testing.assert_allclose(np.column_stack(actual), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("estimates", "lower", "truth"),
    [([0.006, 0.009], 0.0055, 0.004), ([0.002, 0.004], 0.001, 0.0)],
)
def test_trial_rates_shifts_estimates_across_threshold(
    estimates: list[float], lower: float, truth: float
) -> None:
    result = power.trial_rates(
        np.array(estimates),
        np.full(2, lower),
        np.full(2, 0.02),
        truth=truth,
        threshold=0.0052,
        training_effect=0.008,
    )
    assert (result.null, result.half, result.full) == (0.0, 1.0, 1.0)


@pytest.mark.parametrize("hypothesis", ["H2", "H3"])
def test_rate_defaults_match_printed_replicate_settings(hypothesis: str) -> None:
    rates = power.h2_rates if hypothesis == "H2" else power.h3_rates
    defaults = inspect.signature(rates).parameters
    printed = dict(line.split(" = ", 1) for line in power.settings())
    assert defaults["trials"].default == int(printed[f"[settings] {hypothesis} power trials"])
    assert defaults["resamples"].default == int(
        printed[f"[settings] {hypothesis} trial interval replicates"]
    )
    if hypothesis == "H2":
        interval_defaults = inspect.signature(power.h2_trial_intervals).parameters
        for name in ("trials", "resamples"):
            assert interval_defaults[name].default == defaults[name].default
