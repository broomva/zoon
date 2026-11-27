"""Closed-loop logs from the committed arms and transcript-backed model answers."""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition
from typing import Any

import numpy as np

from zoon.interaction import chat, compare, decisions, kernel, prompted, registered, score
from zoon.interaction.games import GAMES, Game
from zoon.interaction.guard import Guard, digest, guarded_output
from zoon.interaction.measure import Floats
from zoon.pilot import clients

# Groups enter the shared session in order, including with more workers. This
# preserves transcript line references and the stateful stub stream on resume.
CHAT_THREADS = 1


class Prompted:
    def __init__(self, session: decisions.Session, recalibration: prompted.Recalibration) -> None:
        self.session, self.recalibration = session, recalibration

    def __call__(self, view: kernel.View) -> Floats:
        request = prompted.request(view)
        identifier = digest(clients.canonical(request))
        response = self.session.ask_many([(identifier, request)])[0]
        used = np.array([response.probabilities[a] for a in prompted.ANSWERS])
        score.finite(used)
        result = self.recalibration.apply(used)
        score.finite(result)
        return result


def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Store:
    def __init__(self, guard: Guard, payload: dict[str, Any], save: Callable[[], Any]) -> None:
        self.guard, self.payload, self.save = guard, payload, save
        payload.setdefault("digests", {})
        payload.setdefault("simulations", {})

    def read(self, name: str, expected: str) -> bytes:
        return guarded_output(self.guard.storage, name, expected)

    def verify(self) -> None:
        for name, expected in self.payload["digests"].items():
            self.read(name, expected)

    def record(self, arm: str, cell: str, runs: tuple[kernel.Run, ...]) -> dict[str, str]:
        values = np.array([compare.summaries(run.rounds, GAMES[cell]) for run in runs])
        score.validate_summaries(values, cell)
        outputs: dict[str, str] = {}
        for n, run in enumerate(runs):
            path = self.guard.directory / "closed-loop" / arm / cell / f"{n}.jsonl"
            write(path, run.bytes)
            outputs[path.relative_to(self.guard.storage).as_posix()] = digest(run.bytes)
        return outputs

    def bands(self, session: decisions.Session, groups: int) -> None:
        directory = self.guard.root / "interaction"
        manifest = directory / "data-manifest.json"
        laws = registered.read_fitted(directory / "fitted.json", manifest)
        amounts = registered.read_amounts(directory / "amounts.json", manifest)
        calibration = self.guard.state("calibrate")
        if calibration is None:
            raise clients.GuardRefusal("completed calibration is required")
        numbers = calibration["payload"]["recalibration"]
        recalibration = prompted.Recalibration(numbers[0], tuple(numbers[1:]))
        for arm in ("zoon", "alone", "prompted"):
            if arm in self.payload["simulations"]:
                continue
            agent = (
                Prompted(session, recalibration) if arm == "prompted" else kernel.Coded(laws[arm])
            )
            result = score.simulate(agent, amounts, open_channel=arm != "alone", groups=groups)
            outputs: dict[str, str] = {}
            for cell, runs in result.logs.items():
                score.validate_summaries(result.summaries[cell], cell)
                outputs.update(self.record(arm, cell, runs))
            self.payload["digests"].update(outputs)
            self.payload["simulations"][arm] = sorted(outputs)
            self.save()

    def chat(self, session: chat.Session, groups: int) -> None:
        games = {
            cell: play_chat_game(game, session, groups) for cell, game in sorted(GAMES.items())
        }
        verified = chat.replay_transcript(session.transcript, session.transcript.digest())
        for runs in games.values():
            for run in runs:
                chat.replay(run.bytes, verified, session.cap)
        outputs: dict[str, str] = {}
        for cell, runs in games.items():
            outputs.update(self.record("chat", cell, runs))
        self.payload["chat_logs"] = outputs


def play_chat_game(game: Game, session: chat.Session, groups: int) -> tuple[kernel.Run, ...]:
    turn = 0
    failed = False
    condition = Condition()

    def play(n: int) -> kernel.Run:
        nonlocal turn, failed
        with condition:
            condition.wait_for(lambda: n == turn or failed)
            if failed:
                raise chat.ConsistencyInterruption("earlier chat group interrupted")
            try:
                result = chat.play(game, session, f"{score.SIMULATION_PREFIX}|{game.cell}|{n}")
            except BaseException:
                failed = True
                raise
            finally:
                turn += 1
                condition.notify_all()
            return result

    with ThreadPoolExecutor(max_workers=CHAT_THREADS) as pool:
        return tuple(pool.map(play, range(groups)))
