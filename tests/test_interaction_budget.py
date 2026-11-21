"""Spending counts on small made-up groups."""

import ast
import builtins
import inspect
import io
import os
from collections.abc import Iterator
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zoon.interaction import budget, compare, games, kernel, prompted
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
    }
    assert {step: row.output_tokens for step, row in rows.items()} == {
        "open loop": 12,
        "recalibration": 4,
        "disguise": 4,
        "closed loop": 18,
    }
    assert {step: row.input_tokens for step, row in rows.items()} == {
        "open loop": 14764,
        "recalibration": 4855,
        "disguise": 4862,
        "closed loop": 22146,
    }
    assert sum(row.calls for row in rows.values()) == 38
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

    def refuse_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected file read")

    monkeypatch.setattr(builtins, "open", refuse_read)
    monkeypatch.setattr(io, "open", refuse_read)
    monkeypatch.setattr(os, "open", refuse_read)
    monkeypatch.setattr(Path, "open", refuse_read)
    total = sum((r.dollars for r in budget.estimate(small, {"2-FE": 2})), Decimal())
    for ceiling, within in ((Decimal("120"), True), (Decimal("0.00000001"), False), (total, True)):
        monkeypatch.setattr(budget, "CEILING", ceiling)
        budget.main()
        output = capsys.readouterr().out
        total = sum((r.dollars for r in budget.estimate(small, {"2-FE": 2})), Decimal())
        assert f"[budget] share of proposed ceiling = {total / ceiling:.8f}" in output
        assert f"[budget] total at most proposed ceiling = {within}" in output
        assert (
            f"[total] calls = 38 | input tokens <= 46627 | output tokens <= 38 | "
            f"USD <= {budget.dollars(total)}"
        ) in output
        assert "[budget] largest component = closed loop | USD <= 0.00093014" in output
        assert f"[settings] proposed ceiling USD = {ceiling}" in output
    assert accesses == ["training", "counts"] * 3
    settings = "\n".join(budget.settings())
    assert "[settings] rounds = 3" in settings
    assert "[settings] simulated groups per game = 3" in settings
    assert "[settings] recalibration count limit = 4" in settings


def test_dollars_bound_rounds_up() -> None:
    assert budget.dollars(Decimal("0.040320924")) == "0.04032093"
    assert Decimal(budget.dollars(Decimal("0.000000001"))) >= Decimal("0.000000001")


def test_longest_row_includes_full_endowment() -> None:
    game = Game("example", "example", (10, 1), (Fraction(1), Fraction(1)))
    rows = list(product(range(11), range(2)))
    longest = budget.longest_row(game)
    assert longest[0] == 10
    assert len(prompted.round_text(game, longest)) == max(
        len(prompted.round_text(game, row)) for row in rows
    )
    assert len(prompted.round_text(game, longest)) > max(
        len(prompted.round_text(game, row)) for row in rows if row[0] < 10
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_held_counts_must_have_exactly_the_games(unknown: bool) -> None:
    held = dict.fromkeys(budget.GAMES, 1)
    if unknown:
        held["unknown"] = 1
    else:
        held.pop(next(iter(held)))
    with pytest.raises(ValueError, match="exactly"):
        budget.estimate([], held)


def test_each_settings_line_reads_its_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = {(label, path) for label, source, path in budget.SETTINGS if source is prompted}
    assert {
        ("recalibration start coordinates", "FIT_START"),
        ("recalibration optimiser", "FIT_METHOD"),
        ("recalibration optimiser options", "FIT_OPTIONS"),
        ("recalibration gradient limit", "GRADIENT_LIMIT"),
        ("recalibration polish steps", "POLISH_STEPS"),
        ("recalibration beta lower bound", "BETA_LOWER_BOUND"),
        ("recalibration penalty", "PENALTY"),
    } <= sources
    for label, source, path in budget.SETTINGS:
        with monkeypatch.context() as patch:
            sentinel = "changed source"
            parts = path.split(".")
            if len(parts) == 1:
                patch.setattr(source, parts[0], sentinel)
            else:
                parent: Any = getattr(source, parts[0])
                patch.setattr(source, parts[0], replace(parent, **{parts[1]: sentinel}))
            assert f"[settings] {label} = {sentinel}" in budget.settings()


def test_settings_table_resolves_named_attributes_outside_the_table() -> None:
    assert len({label for label, _, _ in budget.SETTINGS}) == len(budget.SETTINGS)
    for _, source, path in budget.SETTINGS:
        assert isinstance(source, ModuleType)
        value: object = source
        for attribute in path.split("."):
            assert attribute.isidentifier()
            value = getattr(value, attribute)
        assert value is not source
    # A value held by this module is an alias of another module's constant, never typed here.
    own = {path: getattr(budget, path) for _, source, path in budget.SETTINGS if source is budget}
    assert own == {
        "ROUNDS": games.ROUNDS,
        "SIMULATED_GROUPS": compare.SIMULATED_GROUPS,
        "CEILING": pilot_budget.CEILING,
    }
    tree = ast.parse(inspect.getsource(budget))
    ceiling = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CEILING" for target in node.targets)
    )
    assert ast.unparse(ceiling.value) == "pilot_budget.CEILING"
