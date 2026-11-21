"""Power mechanics on synthetic groups; never rerun the long pasted calculation."""

import ast
import inspect
import re
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from tests.test_interaction_score import ensembles, synthetic
from zoon.interaction import data, kernel, measure, person, power, registered, score
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
        line.split(" = ", 1) for line in paste.splitlines()[1:] if not line.startswith("[settings]")
    )
    expected += [f"| {label} | {value} |" for label, value in printed.items()]
    assert table == expected
    prose_lines = [line for line in prose.splitlines() if line and not line.startswith("|")]
    assert len(prose_lines) <= 12
    assert not re.search(r"\d", "\n".join(prose_lines))
    assert paste.splitlines()[0] == power.TITLE
    assert [line.split(" = ", 1)[0] for line in paste.splitlines()] == power.labels()
    assert [
        line for line in paste.splitlines() if line.startswith("[settings]")
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

    def checked(*_args: object) -> None:
        pass

    def simulate(*_args: object, **_kwargs: object) -> score.Simulation:
        return score.Simulation({}, source["zoon"])

    def pool(*_args: object) -> dict[str, dict[str, Floats]]:
        return source

    def rates(*_args: object, **_kwargs: object) -> power.Rates:
        return power.Rates(1, 0, 0, 0, 0.5, 1)

    monkeypatch.setattr(power, "h2_training", training)
    monkeypatch.setattr(power, "checked_training_h2", checked)
    laws = {arm: person.Person(form, form.start()) for arm, form in registered.FORMS.items()}
    monkeypatch.setattr(registered, "read_fitted", lambda: laws)
    monkeypatch.setattr(registered, "read_amounts", lambda: kernel.Amounts({}))
    source = ensembles()
    monkeypatch.setattr(score, "simulate", simulate)
    monkeypatch.setattr(power, "pool", pool)
    monkeypatch.setattr(power, "h2_rates", rates)
    monkeypatch.setattr(power, "h3_rates", rates)
    power.main()
    output = capsys.readouterr().out.splitlines()
    assert [line.split(" = ", 1)[0] for line in output] == power.labels()
    assert "[H2] pass at half effect = 0.500 (se 0.007)" in output
    assert "[H3] pass at half effect = 0.500 (se 0.011)" in output


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
        for seat in range(game.size):
            before = np.empty(measure.BANDS)
            for other in (0, 1):
                row = tuple(
                    e * other if s != seat else e // 2 for s, e in enumerate(game.endowments)
                )
                view = kernel.View(
                    game,
                    seat,
                    (kernel.Observation(((seat, row[seat]),), None),) * 3,
                    open_channel=False,
                )
                p = kernel.Coded(law)(view)
                if other == 0:
                    before = p
                else:
                    np.testing.assert_array_equal(before, p)


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
        (0.032, power.H2_TRIALS, "0.032 (se 0.002)"),
        (0.032, power.H3_TRIALS, "0.032 (se 0.004)"),
        (0.0, power.H2_TRIALS, "0.000 (se 0.000)"),
        (1.0, power.H3_TRIALS, "1.000 (se 0.000)"),
    ],
)
def test_formatted_rate_uses_its_trial_count(value: float, trials: int, expected: str) -> None:
    assert power.formatted_rate(value, trials) == expected
