"""Spending counts on small made-up groups."""

from collections.abc import Iterator
from decimal import Decimal
from fractions import Fraction
from itertools import product

import pytest

from zoon.interaction import budget, kernel, prompted
from zoon.interaction.data import Group
from zoon.interaction.games import Game
from zoon.pilot import budget as pilot_budget
from zoon.pilot import prices


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Group]]:
    game = Game("2-FE", "FE", (2, 2), (Fraction(1), Fraction(1)))
    monkeypatch.setitem(prompted.GAMES, game.cell, game)
    monkeypatch.setattr(budget, "GAMES", {game.cell: game})
    monkeypatch.setattr(budget, "ROUNDS", 3)
    monkeypatch.setattr(prompted, "ROUNDS", 3)
    monkeypatch.setattr(budget, "SIMULATED_GROUPS", 3)
    monkeypatch.setattr(prompted, "SUBSAMPLE_COUNT", 4)
    yield [Group(game.cell, "made-up", ((0, 0), (1, 1), (2, 2)))]
    budget.longest_row.cache_clear()


def test_every_count_and_cost_against_small_case(small: list[Group]) -> None:
    rows = {r.step: r for r in budget.estimate(small, {"2-FE": 2})}
    assert {step: row.calls for step, row in rows.items()} == {
        "open loop": 12,
        "recalibration": 4,
        "disguise": 4,
        "closed loop": 18,
        "recall": 1,
    }
    assert {step: row.output_tokens for step, row in rows.items()} == {
        "open loop": 12,
        "recalibration": 4,
        "disguise": 4,
        "closed loop": 18,
        "recall": 40,
    }
    assert {step: row.input_tokens for step, row in rows.items()} == {
        "open loop": 14764,
        "recalibration": 4855,
        "disguise": 4862,
        "closed loop": 22146,
        "recall": 980,
    }
    assert sum(row.calls for row in rows.values()) == 39
    game = budget.GAMES["2-FE"]
    per_group = sum(
        pilot_budget.input_tokens(budget.bound_request(game, s, r), "decision")
        for r in (1, 2, 3)
        for s in (0, 1)
    )
    assert rows["open loop"].input_tokens == 2 * per_group
    assert rows["closed loop"].input_tokens == 3 * per_group
    sample = prompted.subsample(small)
    for step, multiplier in (("recalibration", 1), ("disguise", prompted.DISGUISE)):
        tokens = sum(
            pilot_budget.input_tokens(
                prompted.request(prompted.training_view(d.group, d.seat, d.round), multiplier),
                "decision",
            )
            for d in sample
        )
        assert rows[step].input_tokens == tokens
    assert rows["recall"].input_tokens == pilot_budget.input_tokens(
        prompted.recall(game), "decision"
    )
    for row in rows.values():
        assert row.dollars == prices.DECISION.dollars(row.input_tokens, row.output_tokens)
    total = sum((r.dollars for r in rows.values()), Decimal())
    assert total == Decimal("0.042") * sum(r.input_tokens for r in rows.values()) / 1_000_000


def test_longest_rendering_and_growing_history(small: list[Group]) -> None:
    game = budget.GAMES["2-FE"]
    longest = budget.longest_row(game)
    assert len(prompted.round_text(game, longest)) == max(
        len(prompted.round_text(game, row)) for row in product(range(3), repeat=2)
    )
    previous = 0
    for r in (1, 2, 3):
        bound = budget.bound_request(game, 1, r)
        tokens = pilot_budget.input_tokens(bound, "decision")
        assert tokens > previous
        previous = tokens
        for row in product(range(3), repeat=2):
            view = kernel.View(
                game, 1, (kernel.Observation(tuple(enumerate(row)), game.payoffs(row)),) * (r - 1)
            )
            assert pilot_budget.input_tokens(prompted.request(view), "decision") <= tokens


def test_ceiling_comparison_and_main_reads_only_training_and_counts(
    small: list[Group],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    accesses: list[str] = []

    def load() -> list[Group]:
        accesses.append("training")
        return small

    def held() -> dict[str, int]:
        accesses.append("counts")
        return {"2-FE": 2}

    monkeypatch.setattr(budget.data, "load", load)
    monkeypatch.setattr(budget.data, "held_out_counts", held)
    for ceiling, within in ((Decimal("120"), True), (Decimal("0.00000001"), False)):
        monkeypatch.setattr(budget, "CEILING", ceiling)
        budget.main()
        output = capsys.readouterr().out
        total = sum((r.dollars for r in budget.estimate(small, {"2-FE": 2})), Decimal())
        assert f"[budget] share of proposed ceiling = {total / ceiling:.8f}" in output
        assert f"[budget] total at most proposed ceiling = {within}" in output
        assert (
            f"[total] calls = 39 | input tokens <= 47607 | output tokens <= 78 | "
            f"USD <= {budget.dollars(total)}"
        ) in output
        assert "[budget] largest component = closed loop | USD <= 0.00093014" in output
        assert f"[settings] proposed ceiling USD = {ceiling}" in output
    assert accesses == ["training", "counts"] * 2
    assert budget.SIMULATED_GROUPS == 3
    settings = "\n".join(budget.settings())
    assert "[settings] rounds = 3" in settings
    assert "[settings] simulated groups per game = 3" in settings
    assert "[settings] recalibration count limit = 4" in settings


def test_dollars_bound_rounds_up() -> None:
    assert budget.dollars(Decimal("0.040320924")) == "0.04032093"
    assert Decimal(budget.dollars(Decimal("0.000000001"))) >= Decimal("0.000000001")
