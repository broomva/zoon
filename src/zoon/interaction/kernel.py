"""Simultaneous rounds, exact transfers, restricted observations and replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from numbers import Integral
from typing import Any, Protocol

import numpy as np

from zoon.interaction import measure, person
from zoon.interaction.data import Group, Rounds
from zoon.interaction.games import GAMES, ROUNDS, Game
from zoon.interaction.measure import Floats


@dataclass(frozen=True)
class Amounts:
    """Empirical amounts with repetitions retained, by endowment and middle band."""

    rows: Mapping[tuple[int, int], tuple[int, ...]]

    def __post_init__(self) -> None:
        for (endowment, band), values in self.rows.items():
            if (
                not isinstance(endowment, Integral)
                or endowment <= 0
                or not isinstance(band, Integral)
                or band not in (1, 2, 3)
            ):
                raise ValueError("amount table needs a positive endowment and a middle band")
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or not 0 <= value <= endowment
                    or measure.band(value, endowment) != band
                ):
                    raise ValueError("amount is illegal or outside its band")
        object.__setattr__(
            self,
            "rows",
            {
                (int(e), int(b)): tuple(int(v) for v in values)
                for (e, b), values in self.rows.items()
            },
        )

    @classmethod
    def build(cls, groups: Sequence[Group]) -> Amounts:
        rows: dict[tuple[int, int], list[int]] = {}
        for group in groups:
            game = GAMES[group.cell]
            for row in group.rounds:
                for points, endowment in zip(row, game.endowments, strict=True):
                    b = measure.band(points, endowment)
                    if b in (1, 2, 3):
                        rows.setdefault((endowment, b), []).append(points)
        return cls({key: tuple(sorted(values)) for key, values in sorted(rows.items())})

    def draw(self, endowment: int, band: int, uniform: float) -> int:
        if band == 0:
            return 0
        if band == 4:
            return endowment
        values = self.rows.get((endowment, band))
        if not values:
            raise ValueError(f"no training amounts for endowment {endowment}, band {band}")
        return values[min(int(uniform * len(values)), len(values) - 1)]


@dataclass(frozen=True)
class Observation:
    contributions: tuple[tuple[int, int], ...]  # seat and points, only visible seats
    payoffs: tuple[Fraction, ...] | None


@dataclass(frozen=True)
class View:
    game: Game
    seat: int
    history: tuple[Observation, ...]
    open_channel: bool = True


class Agent(Protocol):
    def __call__(self, view: View) -> Floats: ...


@dataclass
class Coded:
    """A fitted person called with a view."""

    law: person.Person
    _history: tuple[Observation, ...] = field(default=(), init=False, repr=False)
    _state: person.State = field(default_factory=person.State, init=False, repr=False)
    _key: tuple[Game, int] | None = field(default=None, init=False, repr=False)

    def __call__(self, view: View) -> Floats:
        # Reuse a matching prefix.
        key = (view.game, view.seat)
        if key == self._key and view.history and view.history[:-1] == self._history:
            state, observations = self._state, view.history[-1:]
        else:
            state, observations = person.State(), view.history
        for observation in observations:
            visible = dict(observation.contributions)
            row = tuple(visible.get(s, 0) for s in range(view.game.size))
            state = self.law.update(state, view.game, view.seat, row)
            if set(visible) == {view.seat}:
                state = replace(
                    state,
                    traces=tuple(
                        value if name == "own" else 0.5
                        for name, value in zip(self.law.candidate.traces, state.traces, strict=True)
                    ),
                )
        self._history, self._state, self._key = view.history, state, key
        return self.law.probabilities(view.game, view.seat, state)


def uniform(seed: int, cell: str, group: str, round_: int, seat: int, purpose: str) -> float:
    key = json.dumps([seed, cell, group, round_, seat, purpose], separators=(",", ":"))
    # A 53-bit binary fraction is exactly representable and strictly below one.
    integer = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 11
    return integer / 2**53


def distribution(values: Floats) -> Floats:
    if (
        values.shape != (5,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or values.sum() <= 0
        or not np.isfinite(values.sum())
    ):
        raise ValueError("an agent must return five finite non-negative probabilities")
    p = np.maximum(values, person.FLOOR)
    return p / p.sum()


def balance(game: Game, points: tuple[int, ...]) -> tuple[Fraction, ...]:
    payoffs = game.payoffs(points)
    pool = sum((p * c for p, c in zip(game.productivities, points, strict=True)), Fraction())
    expected = tuple(
        Fraction(e - c) + pool / game.size for e, c in zip(game.endowments, points, strict=True)
    )
    if payoffs != expected:
        raise ArithmeticError("round ledger does not balance")
    return payoffs


def encoded(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


@dataclass(frozen=True)
class Run:
    rounds: Rounds
    log: tuple[str, ...]

    @property
    def bytes(self) -> bytes:
        return "".join(self.log).encode()


def play(
    game: Game,
    agents: Sequence[Agent],
    amounts: Amounts,
    seed: int,
    group: str,
    *,
    open_channel: bool = True,
) -> Run:
    if len(agents) != game.size:
        raise ValueError("one agent is needed per seat")
    # The log is only extended. Each round uses the same frozen prefix of observations.
    log = [
        encoded(
            {
                "kind": "start",
                "cell": game.cell,
                "treatment": game.treatment,
                "endowments": game.endowments,
                "productivities": [str(p) for p in game.productivities],
                "seed": seed,
                "group": group,
                "open": open_channel,
                "amounts": [[e, b, values] for (e, b), values in sorted(amounts.rows.items())],
            }
        )
    ]
    histories: list[tuple[Observation, ...]] = [() for _ in agents]
    rounds: list[tuple[int, ...]] = []
    for r in range(ROUNDS):
        returned = [
            np.array(agent(View(game, s, histories[s], open_channel)), dtype=np.float64, copy=True)
            for s, agent in enumerate(agents)
        ]
        laws = [distribution(p) for p in returned]
        chosen: list[int] = []
        for s, law in enumerate(laws):
            u = uniform(seed, game.cell, group, r, s, "band")
            b = min(int(np.searchsorted(np.cumsum(law), u, side="right")), 4)
            points = amounts.draw(
                game.endowments[s], b, uniform(seed, game.cell, group, r, s, "amount")
            )
            chosen.append(points)
            log.append(
                encoded(
                    {
                        "kind": "intent",
                        "round": r,
                        "seat": s,
                        "returned": returned[s].tolist(),
                        "probabilities": law.tolist(),
                        "band": b,
                        "points": points,
                    }
                )
            )
        row = tuple(chosen)
        payoffs = balance(game, row)
        pool = sum((p * c for p, c in zip(game.productivities, row, strict=True)), Fraction())
        rounds.append(row)
        for s in range(game.size):
            log.append(
                encoded(
                    {
                        "kind": "transfer",
                        "round": r,
                        "seat": s,
                        "kept": str(Fraction(game.endowments[s] - row[s])),
                        "pool": str(pool),
                        "share": str(pool / game.size),
                        "payoff": str(payoffs[s]),
                    }
                )
            )
        for s in range(game.size):
            visible = tuple(enumerate(row)) if open_channel else ((s, row[s]),)
            observation = Observation(visible, payoffs if open_channel else None)
            histories[s] += (observation,)
            log.append(
                encoded(
                    {
                        "kind": "observation",
                        "round": r,
                        "seat": s,
                        "contributions": visible,
                        "payoffs": [str(p) for p in payoffs] if open_channel else None,
                    }
                )
            )
    return Run(tuple(rounds), tuple(log))


def replay(log: bytes) -> Run:
    """Replay the logged game and compare its bytes."""
    records: list[dict[str, Any]] = [json.loads(line) for line in log.decode().splitlines()]
    header = records[0]
    game = Game(
        header["cell"],
        header["treatment"],
        tuple(header["endowments"]),
        tuple(Fraction(p) for p in header["productivities"]),
    )
    if game != GAMES[game.cell]:
        raise ValueError("game rules differ")
    amounts = Amounts({(e, b): tuple(values) for e, b, values in header["amounts"]})
    laws = {
        (record["round"], record["seat"]): np.array(record["returned"])
        for record in records
        if record["kind"] == "intent"
    }

    def recorded(view: View) -> Floats:
        return laws[len(view.history), view.seat]

    result = play(
        game,
        [recorded] * game.size,
        amounts,
        header["seed"],
        header["group"],
        open_channel=header["open"],
    )
    if result.bytes != log:
        raise ValueError("log does not replay")
    return result
