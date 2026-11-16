"""The registered command sequence, with digest-bound state and a sealed fifth step."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import cast

import numpy as np

from zoon.pilot import (
    arms,
    budget,
    clients,
    cohorts,
    fetch,
    judge,
    lean,
    materials,
    models,
    outside,
    prompted,
    prompts,
    recognition,
    rewrites,
    score,
    stub,
)

STEPS = ("judge-check", "judge", "calibrate", "held-out", "report")
State = dict[str, object]


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_data(directory: Path, manifest: Path) -> None:
    try:
        files = clients.obj(clients.obj(json.loads(manifest.read_text()))["files"])
        for name, raw in files.items():
            expected = clients.obj(raw)
            path = directory / name
            with path.open("rb") as handle:
                found = hashlib.file_digest(handle, "sha256").hexdigest()
            if path.stat().st_size != expected["size"] or found != expected["sha256"]:
                raise clients.GuardRefusal("dataset does not match its registered manifest")
    except (OSError, ValueError, KeyError):
        raise clients.GuardRefusal(
            "dataset unavailable or inconsistent with its registered manifest"
        ) from None


def git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        raise clients.GuardRefusal("could not verify the registered tag or checkout") from None


def split_dependencies(deps: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    registered = {
        name: value
        for name, value in deps.items()
        if name.startswith(("prereg/", "pilot/", "data-manifest"))
        or (name.startswith("src/") and name.endswith(".csv"))
    }
    return registered, {name: value for name, value in deps.items() if name not in registered}


def tree(root: Path, commit: str) -> dict[str, str]:
    import io

    entries = [line.split("\t", 1) for line in git(root, "ls-tree", "-r", commit).splitlines()]
    objects = [metadata.split()[2] for metadata, _ in entries]
    stream = io.BytesIO(
        subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=("\n".join(objects) + "\n").encode(),
            capture_output=True,
            check=True,
        ).stdout
    )
    result: dict[str, str] = {}
    for _, name in entries:
        size = int(stream.readline().split()[2])
        result[name] = digest(stream.read(size))
        stream.read(1)
    return result


def changed_files(root: Path, registered: str, current: str) -> list[str]:
    return git(root, "diff", "--name-only", registered, current).splitlines()


def floats(value: object) -> models.Floats:
    return np.array(cast(list[object], value), dtype=np.float64)


def fit_document(rotations: Sequence[arms.Rotation]) -> list[State]:
    result: list[State] = []
    for rotation in rotations:
        rows: State = {}
        for name, fit in rotation.arms.items():
            rows[name] = {
                "intercept": fit.parameters.intercept.tolist(),
                "learning_rate": fit.parameters.learning_rate,
                "temperature": fit.parameters.temperature,
                "count": fit.count,
                "alpha": fit.alpha,
                "beta": fit.beta,
            }
        result.append({"held_out": list(rotation.held_out), "arms": rows})
    return result


def read_fits(value: object) -> list[arms.Rotation]:
    result: list[arms.Rotation] = []
    for raw in clients.items(value):
        row = clients.obj(raw)
        fitted: dict[str, arms.Fit] = {}
        for name, raw_fit in clients.obj(row["arms"]).items():
            fit = clients.obj(raw_fit)
            fitted[name] = arms.Fit(
                models.Parameters(
                    floats(fit["intercept"]),
                    clients.number(fit["learning_rate"]),
                    clients.number(fit["temperature"]),
                ),
                int(clients.number(fit["count"])),
                cast(float | None, fit["alpha"]),
                cast(float | None, fit["beta"]),
            )
        held_out = tuple(int(clients.number(i)) for i in clients.items(row["held_out"]))
        result.append(arms.Rotation(held_out, fitted))
    return result


def checkpoint(transcript: clients.Transcript, count: int | None = None) -> dict[str, object]:
    entries = list(transcript.entries.values())
    if count is not None:
        if len(entries) < count:
            raise clients.GuardRefusal("transcript checkpoint is missing requests")
        entries = entries[:count]
    return {
        "calls": len(entries),
        "digest": digest(encoded([clients.entry_document(e) for e in entries])),
    }


class Run:
    def __init__(
        self,
        root: Path = fetch.REPO,
        *,
        synthetic: bool = False,
        workers: int = 4,
        supplied_clients: Sequence[clients.Client] | None = None,
        cap: Decimal = clients.CAP,
    ) -> None:
        self.root = root
        self.synthetic = synthetic
        self.registered_commit = git(
            root, "rev-parse", "HEAD" if synthetic else "refs/tags/registered^{commit}"
        )
        self.mode = "stub" if synthetic else "live-" + self.registered_commit
        self.code: dict[str, str] = {}
        self.pending_binding: State | None = None
        self.directory = root / "data" / "run" / self.mode
        self.records = rewrites.load(root / "pilot" / "rewrites") if synthetic else []
        available = (
            supplied_clients
            if supplied_clients is not None
            else [stub.client(f, self.records) for f in budget.FAMILIES]
            if synthetic
            else [clients.DecisionClient(), clients.VertexClient()]
        )
        self.session = clients.Session(
            root / "data" / "transcripts" / self.mode, available, workers, cap
        )

    def line(self, tag: str, key: str, value: object) -> str:
        return f"[{'stub ' if self.synthetic else ''}{tag}] {key} = {value}"

    def binding(self, value: State | None = None) -> State:
        path = self.directory / "binding.json"
        if value is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded(value))
            temporary.replace(path)
        return clients.obj(json.loads(path.read_text())) if path.exists() else {"repairs": []}

    def guard(self, step: str, registered: str | None, repair: str | None) -> None:
        self.pending_binding = None
        try:
            self._guard(step, registered, repair)
        except BaseException as error:
            self.pending_binding = None
            if isinstance(error, clients.GuardRefusal):
                raise
            raise clients.GuardRefusal(
                "guard could not verify inputs: " + type(error).__name__
            ) from None

    def _guard(self, step: str, registered: str | None, repair: str | None) -> None:
        if step not in STEPS:
            raise clients.GuardRefusal("unknown step")
        if not self.synthetic:
            tagged = git(self.root, "rev-parse", "refs/tags/registered^{commit}")
            if tagged != self.registered_commit:
                raise clients.GuardRefusal("registered tag changed; start with its run identity")
            if step == "held-out" and registered != tagged:
                raise clients.GuardRefusal("--registered-commit must match the registered tag")
            try:
                git(self.root, "merge-base", "--is-ancestor", tagged, "HEAD")
            except clients.GuardRefusal:
                raise clients.GuardRefusal(
                    "registered commit must be an ancestor of HEAD"
                ) from None
            original, head = tree(self.root, tagged), tree(self.root, "HEAD")
            names = original.keys() | head.keys()
            if git(self.root, "status", "--porcelain", "--untracked-files=all") or any(
                (digest(path.read_bytes()) if (path := self.root / name).is_file() else "deleted")
                != head.get(name, "deleted")
                for name in names
            ):
                raise clients.GuardRefusal("requires a clean work tree matching HEAD")
            difference = {
                name: head.get(name, "deleted")
                for name in names
                if original.get(name) != head.get(name)
            }
            fixed, code = split_dependencies(difference)
            if fixed:
                raise clients.GuardRefusal(
                    "registered files differ from the registered commit: "
                    + ", ".join(sorted(fixed))
                )
            self.code = {
                name: value
                for name, value in code.items()
                if not name.startswith(("docs/", "tests/"))
            }
            binding = self.binding()
            repairs = clients.items(binding["repairs"])
            if self.code != (clients.obj(repairs[-1])["code"] if repairs else {}):
                if not repair or not repair.strip():
                    raise clients.GuardRefusal(
                        "code requires --repair with a reason: " + ", ".join(sorted(self.code))
                    )
                repairs.append(
                    {
                        "reason": repair,
                        "commit": git(self.root, "rev-parse", "HEAD"),
                        "files": sorted(self.code),
                        "code": self.code,
                    }
                )
                self.pending_binding = binding
        elif step == "held-out" and registered != self.registered_commit:
            raise clients.GuardRefusal(
                "--registered-commit must match the current full commit for a stub run"
            )
        if step != "report":
            for earlier in STEPS[: STEPS.index(step)]:
                if self.state(earlier)["status"] != "completed":
                    raise clients.GuardRefusal(f"step {earlier} did not pass its gate")
        if not self.synthetic:
            self.records = rewrites.load(self.root / "pilot" / "rewrites")
            self.verify_materials()
        if step == "report":
            interrupted = False
            for earlier in STEPS[:-1]:
                state = self.state(
                    earlier, required=not interrupted and self.session.spent <= self.session.cap
                )
                if state.get("status") == "failed":
                    raise clients.GuardRefusal(f"repeat step {earlier} before report")
                interrupted |= state.get("status") == "stopped"

    def state(self, step: str, *, required: bool = True) -> State:
        try:
            path = self.directory / f"{step}.json"
            if not path.exists():
                if not required:
                    return {}
                raise clients.GuardRefusal(f"step {step} has not completed")
            raw = path.read_bytes()
            state = clients.obj(json.loads(raw))
            if state.get("registered_commit") != self.registered_commit:
                raise clients.GuardRefusal("state belongs to another registered commit")
            if state.get("step") != step or state.get("mode") != self.mode:
                raise clients.GuardRefusal("state belongs to another step or mode")
            if state.get("status") not in ("completed", "failed", "stopped") or (
                step == "report" and state["status"] == "stopped"
            ):
                raise clients.GuardRefusal("invalid step status")
            snapshot = clients.obj(state["code_snapshot"])
            repairs = clients.items((self.pending_binding or self.binding())["repairs"])
            if not self.synthetic:
                changed = changed_files(self.root, self.registered_commit, "HEAD")
                _, code = split_dependencies(
                    {
                        name: digest(path.read_bytes())
                        if (path := self.root / name).is_file()
                        else "deleted"
                        for name in changed
                        if not name.startswith(("docs/", "tests/"))
                    }
                )
                if code != (clients.obj(repairs[-1])["code"] if repairs else {}):
                    raise clients.GuardRefusal("current code requires a declared repair")
            if snapshot != {} and not any(clients.obj(r).get("code") == snapshot for r in repairs):
                raise clients.GuardRefusal("state has an unbound code snapshot")
            previous_steps = clients.obj(state["previous"])
            if set(previous_steps) != set(STEPS[: STEPS.index(step)]):
                raise clients.GuardRefusal("previous step set changed")
            for earlier, expected in previous_steps.items():
                previous = self.directory / f"{earlier}.json"
                if expected is None and not previous.exists():
                    continue
                if not previous.exists() or digest(previous.read_bytes()) != expected:
                    raise clients.GuardRefusal("previous step changed")
            if digest(encoded(state["payload"])) != state["payload_digest"]:
                raise clients.GuardRefusal("step output digest changed")
            for family in budget.FAMILIES:
                expected = clients.obj(clients.obj(state["transcripts"])[family])
                found = checkpoint(
                    self.session.transcripts[family], int(clients.number(expected["calls"]))
                )
                if found != expected:
                    raise clients.GuardRefusal("transcript checkpoint changed")
            state["_digest"] = digest(raw)
            return state
        except clients.GuardRefusal:
            raise
        except Exception as error:
            raise clients.GuardRefusal("invalid state: " + type(error).__name__) from None

    def save(
        self, step: str, payload: State, *, status: str = "completed", reason: str = ""
    ) -> State:
        previous = {
            name: prior.get("_digest")
            for name in STEPS[: STEPS.index(step)]
            for prior in [self.state(name, required=False)]
        }
        state: State = {
            "step": step,
            "mode": self.mode,
            "registered_commit": self.registered_commit,
            "code_snapshot": self.code,
            "previous": previous,
            "status": status,
            "reason": reason,
            "transcripts": {f: checkpoint(t) for f, t in self.session.transcripts.items()},
            "payload": payload,
            "commit": git(self.root, "rev-parse", "HEAD"),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{step}.json"
        temporary = path.with_suffix(".tmp")
        state["payload_digest"] = digest(encoded(payload))
        temporary.write_bytes(encoded(state))
        temporary.replace(path)
        return state

    def training(self) -> tuple[list[cohorts.Participant], cohorts.Cohorts]:
        if self.synthetic:
            people = stub.people(48, 41)
            return people, cohorts.generate(people, calibration_size=8)
        people = cohorts.participants(fetch.load(self.root / "data" / "raw"))
        return people, cohorts.generate(people)

    def request(
        self,
        family: clients.Family,
        requests: Sequence[clients.Request],
        *,
        recognition_only: bool = False,
    ) -> list[clients.Response]:
        return self.session.ask_many(family, requests, recognition=recognition_only)

    def _judge_check(self) -> State:
        records = [r for r in self.records if r["set"] == "outside"]
        means = (
            judge.human_ratings()
            if self.synthetic
            else {
                name: rating.mean
                for name, rating in outside.ratings(self.root / "data" / "outside")[2].items()
            }
        )
        human = np.array([means[str(r["id"])] for r in records])
        result: State = {"human": means}
        for family in budget.FAMILIES:
            try:
                responses = self.request(family, [prompts.evaluative(r) for r in records])
                check = judge.calibration(np.array([judge.evaluative(r) for r in responses]), human)
                reason = (
                    clients.StopReason if family == "decision" else clients.FamilyCause
                ).CALIBRATION.value
                result[family] = {
                    **asdict(check),
                    "reason": "" if check.passed else reason,
                }
            except clients.FamilyUnusable as error:
                result[family] = {"correlation": None, "passed": False, "reason": str(error)}
        if not clients.obj(result["decision"])["passed"]:
            raise clients.TerminalStop(clients.StopReason.CALIBRATION, payload=result)
        return self.save("judge-check", result)

    def _judge_all(self) -> State:
        checked = clients.obj(self.state("judge-check")["payload"])
        outside_records = [r for r in self.records if r["set"] == "outside"]
        tasks = {str(r["id"]): r for r in self.records if r["set"] == "task"}
        outside_names = tuple(str(r["id"]) for r in outside_records)
        ordered = [*outside_records, *(tasks[n] for n in fetch.DILEMMAS)]
        means = cast(dict[str, float], checked["human"])
        human = np.array([means[n] for n in outside_names])
        result: State = {}
        for family in budget.FAMILIES:
            if not clients.obj(checked[family])["passed"]:
                result[family] = {"usable": False, "reason": clients.obj(checked[family])["reason"]}
                continue
            try:
                ratings = np.array(
                    [
                        judge.evaluative(r)
                        for r in self.request(
                            family, [prompts.evaluative(tasks[n]) for n in fetch.DILEMMAS]
                        )
                    ]
                )
                strict = self.request(
                    family, [prompts.strict(record, i) for record in ordered for i in range(5)]
                )
                published = outside.published()
                original_codes = outside.original_codes()
                codes = [
                    cast(outside.Codes, published[str(r["id"])].codes) for r in outside_records
                ] + [original_codes[n] for n in fetch.DILEMMAS]
                action = [outside.grade(c.action_is_utilitarian) == 1 for c in codes]
                descriptive = np.array(
                    [judge.features(strict[i * 5 : i * 5 + 5], a) for i, a in enumerate(action)]
                )
                mapping = judge.fit_map(outside_names, descriptive[:25], human)
                raw = mapping.fitted.predict(descriptive[25:])
                reversible = [r for r in ordered if "reversed" in r]
                reversed_ratings = self.request(
                    family, [prompts.evaluative(r, reverse=True) for r in reversible]
                )
                originals = self.request(family, [prompts.evaluative(r) for r in reversible])
                differences = {
                    str(record["id"]): judge.evaluative(reversed_) - judge.evaluative(original)
                    for record, reversed_, original in zip(
                        reversible, reversed_ratings, originals, strict=True
                    )
                }
                result[family] = {
                    "usable": True,
                    "evaluative": ratings.tolist(),
                    "strict": raw.tolist(),
                    "penalty": "inf"
                    if math.isinf(mapping.fitted.penalty)
                    else mapping.fitted.penalty,
                    "map_offset": mapping.fitted.offset,
                    "map_weights": mapping.fitted.weights.tolist(),
                    "finite_penalty": mapping.finite.penalty,
                    "finite_error": mapping.finite_error,
                    "finite_correlation": mapping.finite_correlation,
                    "finite_task_ratings": mapping.finite.predict(descriptive[25:]).tolist(),
                    "agreement_outside": judge.agreement(
                        descriptive[:25], np.array([outside.encode(c) for c in codes[:25]])
                    ),
                    "agreement_task": judge.agreement(
                        descriptive[25:], np.array([outside.encode(c) for c in codes[25:]])
                    ),
                    "reversal": differences,
                    "norm_correlation_evaluative": judge.correlation(
                        ratings, judge.norm_ratings(means)
                    ),
                    "norm_correlation_strict": judge.correlation(raw, judge.norm_ratings(means)),
                }
            except clients.FamilyUnusable as error:
                result[family] = {"usable": False, "reason": str(error)}
        try:
            self.request(
                "second",
                [prompts.recognition(tasks[n]) for n in fetch.DILEMMAS],
                recognition_only=True,
            )
        except clients.FamilyUnusable as error:
            result["recognition_error"] = str(error)
            result["second"] = {"usable": False, "reason": str(error)}
        result["recognition"] = {
            name: {
                "reply": entry.response.text,
                "flagged": recognition.flagged(
                    entry.response.text, recognition.keys(name, self.records)
                ),
            }
            for name in fetch.DILEMMAS
            if (
                entry := self.session.transcripts["second"].entries.get(
                    clients.key(prompts.recognition(tasks[name]))
                )
            )
            and entry.response.text.strip()
        }
        # Recognition also reports a version. A changed version removes the second family's
        # numerical results, while recognition replies remain reportable.
        try:
            self.session.check_family("second")
        except clients.FamilyUnusable as error:
            result["second"] = {"usable": False, "reason": str(error)}
        return self.save("judge", result)

    def collect_prompted(
        self,
        family: clients.Family,
        people: Sequence[cohorts.Participant],
        rotation: cohorts.Rotation,
        training: Sequence[cohorts.Participant],
        tasks: dict[str, materials.Task],
        *,
        held: bool,
    ) -> models.Floats:
        examples = prompts.examples(rotation, training, tasks)
        requests: list[clients.Request] = []
        positions: list[tuple[int, int]] = []
        for i, person in enumerate(people):
            for t, trial in enumerate(person.trials):
                if (trial.dilemma in rotation["held_out"]) == held:
                    positions.append((i, t))
                    requests.append(prompts.prompted(person, t + 1, examples, tasks))
        responses = self.request(family, requests)
        result = np.full((len(people), len(fetch.DILEMMAS)), 0.5)
        for (i, t), response in zip(positions, responses, strict=True):
            result[i, t] = response.probabilities["Yes"]
        return result

    def _calibrate(self) -> State:
        people, document = self.training()
        tasks = materials.load(self.root / "pilot" / "task-texts.json")
        judged = clients.obj(self.state("judge")["payload"])
        calibration_ids = {(p["experiment"], p["participant"]) for p in document["calibration"]}
        sample = [p for p in people if (p.experiment, p.participant) in calibration_ids]
        data = arms.sequences(sample)
        result: State = {}
        means = cast(dict[str, float], clients.obj(self.state("judge-check")["payload"])["human"])
        ratings = arms.reference_ratings(means)
        for family in budget.FAMILIES:
            judged_family = clients.obj(judged[family])
            if not judged_family["usable"]:
                result[family] = {"usable": False, "reason": judged_family["reason"]}
                continue
            try:
                fits: list[State] = []
                for rotation in document["rotations"]:
                    yes = self.collect_prompted(family, sample, rotation, people, tasks, held=False)
                    targets = ~np.isin(
                        data.dilemma, [fetch.DILEMMAS.index(d) for d in rotation["held_out"]]
                    )
                    action = lean.SIGN[data.dilemma] == 1
                    fit = prompted.fit(yes[targets], action[targets], data.choice[targets])
                    fits.append(asdict(fit))
                result[family] = {"usable": True, "fits": fits}
                for kind in ("evaluative", "strict"):
                    ratings[f"{family}/zoon-{kind}"] = floats(judged_family[kind])
            except clients.FamilyUnusable as error:
                result[family] = {"usable": False, "reason": str(error)}
        result["rotations"] = fit_document(arms.fit_rotations(arms.sequences(people), ratings))
        return self.save("calibrate", result)

    def _held_out(self, registered: str | None) -> State:
        self.session.check_family("decision")
        calibrated = clients.obj(self.state("calibrate")["payload"])
        # The synthetic branch has no route through the real held-out loader.
        people = (
            stub.people(12, 83)
            if self.synthetic
            else cohorts.participants(fetch.load_registered_held_out(self.root / "data" / "raw"))
        )
        if not self.synthetic and len(people) != budget.HELD_OUT_PARTICIPANTS:
            raise clients.TerminalStop(clients.StopReason.RECORDS)
        data = arms.sequences(people)
        training, document = self.training()
        tasks = materials.load(self.root / "pilot" / "task-texts.json")
        predicted = arms.predict(read_fits(calibrated["rotations"]), data)
        unavailable: State = {}
        for family in budget.FAMILIES:
            cal = clients.obj(calibrated[family])
            if not cal["usable"]:
                unavailable[family] = cal["reason"]
                continue
            try:
                probabilities: list[models.Floats] = []
                for rotation, raw_fit in zip(
                    document["rotations"], clients.items(cal["fits"]), strict=True
                ):
                    fit = clients.obj(raw_fit)
                    yes = self.collect_prompted(
                        family, people, rotation, training, tasks, held=True
                    )
                    probabilities.append(
                        prompted.probabilities(
                            yes,
                            lean.SIGN[data.dilemma] == 1,
                            prompted.Fit(
                                clients.number(fit["a"]),
                                clients.number(fit["c"]),
                                bool(fit["success"]),
                            ),
                        )
                    )
                predicted[f"{family}/prompted"] = probabilities
                self.session.check_family(family)
            except clients.FamilyUnusable as error:
                unavailable[family] = str(error)
        logs: State = {}
        for name, probabilities in predicted.items():
            family = name.split("/")[0]
            if family in unavailable:
                continue
            try:
                logs[name] = score.rotation_logs(probabilities, data).tolist()
            except clients.TerminalStop as error:
                if family != "second" or error.reason is not clients.StopReason.RECORDS:
                    raise
                unavailable["second"] = clients.StopReason.RECORDS.value
        if "second" in unavailable:
            logs = {name: values for name, values in logs.items() if not name.startswith("second/")}
        return self.save(
            "held-out",
            {
                "logs": logs,
                "identities": [[p.experiment, p.participant] for p in people],
                "dilemma": data.dilemma.tolist(),
                "unavailable": unavailable,
            },
        )

    def verify_materials(self) -> None:
        for folder, manifest in (
            ("raw", "data-manifest.json"),
            ("outside", "data-manifest-outside.json"),
        ):
            verify_data(self.root / "data" / folder, self.root / manifest)
        outside.check_against_paper(
            outside.ratings(self.root / "data" / "outside"), outside.published()
        )
        if (
            rewrites.digest(self.records)
            != (self.root / "pilot" / "rewrites.sha256").read_text().strip()
        ):
            raise clients.GuardRefusal("rewrite digest differs from computed materials")
        texts = self.root / "pilot" / "task-texts.json"
        if (
            digest(texts.read_bytes())
            != (self.root / "pilot" / "task-texts.sha256").read_text().strip()
        ):
            raise clients.GuardRefusal("task-text digest differs from computed materials")
        people, document = self.training()
        if cohorts.serialise(document) != (self.root / "prereg" / "cohorts.json").read_bytes():
            raise clients.GuardRefusal("training data do not regenerate registered cohorts")
        expected = prompts.fixtures(people, document, materials.load(texts), self.records)
        found = {
            path.name: path.read_bytes()
            for path in (self.root / "prereg" / "requests").iterdir()
            if path.name != "README.md"
        }
        if expected != found:
            raise clients.GuardRefusal("request fixtures differ from regenerated materials")

    def preflight(self) -> None:
        second = self.session.clients.get("second")
        if isinstance(second, clients.VertexClient):
            second.preflight()

    def execute(
        self,
        step: str,
        registered: str | None = None,
        *,
        repair: str | None = None,
    ) -> list[str]:
        lock = os.open(self.root, os.O_RDONLY)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise clients.GuardRefusal("another command is using this run") from None
            self.session.transcripts = {
                f: clients.Transcript(t.path.parent, f) for f, t in self.session.transcripts.items()
            }
            self.guard(step, registered, repair)
            try:
                state = self.state(step, required=False)
            except BaseException:
                self.pending_binding = None
                raise
            if self.pending_binding is not None:
                self.binding(self.pending_binding)
            self.pending_binding = None
            if state.get("status") in ("completed", "stopped") and step != "report":
                return self.status_lines(step, state)
            over_cap = step != "report" and self.session.spent > self.session.cap
            try:
                if over_cap:
                    raise clients.TerminalStop(clients.StopReason.CAP)
                if step == "report":
                    return self._report()
                self.preflight()
                if step == "judge-check":
                    state = self._judge_check()
                elif step == "judge":
                    state = self._judge_all()
                elif step == "calibrate":
                    state = self._calibrate()
                else:
                    state = self._held_out(registered)
            except clients.TerminalStop as error:
                if step == "report":
                    state = self.save(step, {}, status="failed", reason=type(error).__name__)
                else:
                    state = self.save(
                        step, error.payload, status="stopped", reason=error.reason.value
                    )
            except BaseException as error:
                state = self.save(step, {}, status="failed", reason=type(error).__name__)
            return self.status_lines(step, state)
        finally:
            os.close(lock)

    def status_lines(self, step: str, state: State) -> list[str]:
        return [self.line(step, key, state[key] or "none") for key in ("status", "reason")]

    def _report(self) -> list[str]:
        from zoon.pilot.result import render, write_report

        states: dict[str, State] = {}
        for step in STEPS[:-1]:
            state = self.state(step, required=self.session.spent <= self.session.cap)
            if not state:
                break
            states[step] = state
            if state["status"] != "completed":
                break
        lines = render(self, states)
        write_report(self, lines)
        self.save("report", {"lines": lines})
        return lines


def smoke(synthetic: bool) -> list[str]:
    request = prompts.Decision(
        "The number under discussion is seven.",
        "Is the number prime?",
        prompts.declared(("Yes", "No")),
    )
    available: list[clients.Client] = (
        [clients.StubClient(f) for f in budget.FAMILIES]
        if synthetic
        else [clients.DecisionClient(), clients.VertexClient()]
    )
    result: list[str] = []
    for client in available:
        if isinstance(client, clients.VertexClient):
            client.preflight()
    for client in available:
        seven = prompts.Decision(
            "A bag contains seven numbered balls, one of each number from one to seven.",
            "Which number will be drawn uniformly at random?",
            prompts.declared(tuple(map(str, range(1, 8)))),
        )
        nine = prompts.Decision(
            "Nine equally likely labelled boxes are available; their labels have no meaning.",
            "Which label will be drawn uniformly at random?",
            prompts.declared(prompts.contexts()),
        )
        for question in (request, seven, nine):
            response = client.ask(question)
            total = (
                sum(response.returned_probabilities.values())
                if response.returned_probabilities is not None
                else response.share
                if response.share is not None
                else sum(response.probabilities.values())
            )
            result.append(
                f"[{'stub ' if synthetic else ''}smoke {client.family}] "
                f"answers = {len(question.answers)} | version = {response.version} | "
                f"probabilities = {response.probabilities} | sum before division = {total} | "
                f"share = {response.share} | input tokens = {response.input_tokens} | "
                f"output tokens = {response.output_tokens} | "
                f"USD = {clients.cost(client.family, response):.8f}"
            )
    return result


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the registered pilot one step at a time.")
    parser.add_argument("step", choices=(*STEPS, "smoke"))
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--registered-commit")
    parser.add_argument("--repair")
    options = parser.parse_args(arguments)
    try:
        lines = (
            smoke(bool(options.stub))
            if options.step == "smoke"
            else Run(synthetic=bool(options.stub), workers=int(options.workers)).execute(
                str(options.step),
                cast(str | None, options.registered_commit),
                repair=cast(str | None, options.repair),
            )
        )
        for line in lines:
            print(line)
    except clients.GuardRefusal as error:
        print(f"[{'stub ' if options.stub else ''}run] refused = {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
