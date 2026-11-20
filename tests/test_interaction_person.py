"""Behavioural controls for each person, using exact made-up contributions."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative  # type: ignore[reportMissingTypeStubs]

from zoon.interaction import measure, person
from zoon.interaction.data import Group
from zoon.interaction.games import GAMES, ROUNDS


def law(letter: str, twin: bool = False) -> person.Person:
    c = person.Candidate(letter, twin)
    numbers = c.start()
    numbers[2 * c.phase_size : 2 * c.phase_size + len(c.features)] = np.arange(
        1, len(c.features) + 1
    )
    return person.Person(c, numbers)


@pytest.mark.parametrize("letter", person.LETTERS)
def test_exact_shares_and_first_round_have_no_traces(letter: str) -> None:
    p, game = law(letter), GAMES["2-FE"]
    low = p.update(person.State(), game, 0, (2, 2))
    high = p.update(person.State(), game, 0, (7, 7))
    assert measure.band(2, 24) == measure.band(7, 24)
    assert not np.array_equal(p.probabilities(game, 0, low), p.probabilities(game, 0, high))
    assert p.probabilities(game, 0, person.State((99.0,) * len(p.fractions))) == pytest.approx(
        p.probabilities(game, 0)
    )
    assert p.probabilities(game, 0).sum() == pytest.approx(1.0)


def test_c_two_traces_and_missing_kind() -> None:
    p = law("C")
    for cell, row in (("2-AI", (18, 3)), ("2-FE", (12, 6)), ("4-FE", (0, 6, 12, 24))):
        game = GAMES[cell]
        state = p.update(person.State(), game, 0, row)
        expected = np.mean([row[s] / game.endowments[s] for s in range(1, game.size)])
        assert state.traces[1:] == pytest.approx((expected, expected))
    game = GAMES["4-AI"]
    state = p.update(person.State(), game, 0, (0, 36, 0, 0))
    assert state.traces == (0, 1, 0)
    switched = p.update(person.State(), game, 0, (0, 0, 12, 12))
    assert switched.traces == (0, 0, 1)
    assert not np.allclose(p.probabilities(game, 0, state), p.probabilities(game, 0, switched))


def test_d_cap_and_reference_endowment_for_both_seats() -> None:
    p, game = law("D"), GAMES["2-AI"]
    for seat, row, expected in ((0, (0, 12), 1 / 3), (1, (24, 0), 1), (1, (6, 0), 0.5)):
        state = p.update(person.State(), game, seat, row)
        assert state.traces[1] == pytest.approx(expected)
    a = p.update(person.State(), game, 1, (24, 0))
    b = p.update(person.State(), game, 1, (36, 0))
    assert p.probabilities(game, 1, a) == pytest.approx(p.probabilities(game, 1, b))


def test_e_keeps_the_exact_first_share() -> None:
    p, game = law("E"), GAMES["2-FE"]
    numbers = p.numbers.copy()
    numbers[-len(p.fractions) :] = 1
    p = replace(p, numbers=numbers)
    low, high = person.State(), person.State()
    for r in range(ROUNDS - 1):
        low = p.update(low, game, 0, (2, 12) if r == 0 else (12, 12))
        high = p.update(high, game, 0, (7, 12) if r == 0 else (12, 12))
    assert low.traces == high.traces
    assert (low.anchor, high.anchor) == (2 / 24, 7 / 24)
    assert not np.allclose(p.probabilities(game, 0, low), p.probabilities(game, 0, high))


def test_f_twin_has_no_state_after_first_round() -> None:
    p, game = law("F", True), GAMES["2-AI"]
    low = p.update(person.State(), game, 0, (0, 0))
    high = p.update(person.State(), game, 0, (36, 12))
    assert p.candidate.features == () and p.fractions.size == 0
    assert p.probabilities(game, 0, low) == pytest.approx(p.probabilities(game, 0, high))
    f = law("F")
    a, b = f.update(person.State(), game, 0, (0, 0)), f.update(person.State(), game, 0, (36, 0))
    assert f.probabilities(game, 0, a) == pytest.approx(f.probabilities(game, 0, b))


@pytest.mark.parametrize("c", person.CANDIDATES, ids=lambda c: c.name)
def test_disguise_scales_every_point_without_changing_predictions(c: person.Candidate) -> None:
    p = law(c.letter, c.twin)
    for game in GAMES.values():
        # B uses the rules in their declared units, so canonicalise their common point unit.
        scaled = replace(game, endowments=tuple(10 * e for e in game.endowments))
        a, b = person.State(), person.State()
        for r in range(ROUNDS):
            row = tuple((r * 7 + s * 3) % (e + 1) for s, e in enumerate(game.endowments))
            assert p.probabilities(game, 0, a) == pytest.approx(p.probabilities(scaled, 0, b))
            a = p.update(a, game, 0, row)
            b = p.update(b, scaled, 0, tuple(10 * v for v in row))


def tiny() -> list[Group]:
    return [
        Group(
            cell,
            f"{cell}-{g}",
            tuple(
                tuple((r + g + s) % (e + 1) for s, e in enumerate(game.endowments))
                for r in range(ROUNDS)
            ),
        )
        for cell, game in GAMES.items()
        for g in range(2)
    ]


@pytest.mark.parametrize("c", person.CANDIDATES, ids=lambda c: c.name)
def test_likelihood_gradient_and_live_state_agree(c: person.Candidate) -> None:
    groups = tiny()
    record = person.records(groups, c)
    numbers = c.start()
    numbers[2 * c.phase_size : 2 * c.phase_size + len(c.features)] = 0.7
    loss, gradient, _ = person.likelihood(numbers, record, c)
    numeric = np.asarray(approx_derivative(lambda x: person.likelihood(x, record, c)[0], numbers))
    assert np.isfinite(loss)
    assert gradient == pytest.approx(numeric.ravel(), abs=2e-5)
    fitted = person.Person(c, numbers)
    live: list[float] = []
    for group in groups:
        game = GAMES[group.cell]
        states = [person.State() for _ in range(game.size)]
        for row in group.rounds:
            for seat, state in enumerate(states):
                live.append(
                    float(
                        np.log(
                            fitted.probabilities(game, seat, state)[
                                measure.band(row[seat], game.endowments[seat])
                            ]
                        )
                    )
                )
            states = [fitted.update(state, game, seat, row) for seat, state in enumerate(states)]
    assert person.log_likelihoods(fitted, groups) == pytest.approx(live)
    assert len(numbers) == c.count


def test_b_predicts_an_absent_game_and_a_missing_constant_stays_zero() -> None:
    groups = [g for g in tiny() if g.cell != "2-MI"]
    for letter in ("A", "B"):
        p = person.fit(groups, person.Candidate(letter))
        assert np.isfinite(p.probabilities(GAMES["2-MI"], 0)).all()
        if letter == "A":
            idx = measure.STRATA["2-MI", 0]
            assert p.numbers[idx] == 0 and p.numbers[p.candidate.phase_size + idx] == 0


def test_nonconvergence_includes_the_optimiser_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def failed(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(success=False, message="iteration limit", x=np.zeros(2))

    monkeypatch.setattr(person, "minimize", failed)
    with pytest.raises(RuntimeError, match="did not converge: iteration limit"):
        person.fit(tiny(), person.Candidate("A"))
