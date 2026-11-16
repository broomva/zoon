"""The registered command sequence, with digest-bound state and a sealed fifth step."""

import argparse
import fcntl
import hashlib
import json
import math
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


def dependencies(root: Path) -> dict[str, str]:
    paths = sorted(
        p
        for folder in ("src", "prereg", "pilot")
        for p in (root / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    paths += [root / "pyproject.toml", root / "uv.lock", *sorted(root.glob("data-manifest*.json"))]
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in paths}


def verify_data(directory: Path, manifest: Path) -> None:
    try:
        files = clients.obj(clients.obj(json.loads(manifest.read_text()))["files"])
        for name, raw in files.items():
            expected = clients.obj(raw)
            path = directory / name
            with path.open("rb") as handle:
                found = hashlib.file_digest(handle, "sha256").hexdigest()
            if path.stat().st_size != expected["size"] or found != expected["sha256"]:
                raise clients.StopRun("dataset does not match its registered manifest")
    except (OSError, ValueError, KeyError):
        raise clients.StopRun(
            "dataset unavailable or inconsistent with its registered manifest"
        ) from None


def git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        raise clients.StopRun("could not verify the registered checkout") from None


def split_dependencies(deps: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    registered = {
        name: value
        for name, value in deps.items()
        if name.startswith(("prereg/", "pilot/", "data-manifest"))
        or (name.startswith("src/") and name.endswith(".csv"))
    }
    return registered, {name: value for name, value in deps.items() if name not in registered}


def tree(root: Path, commit: str) -> dict[str, str]:
    entries = git(root, "ls-tree", "-r", commit).splitlines()
    return {
        name: metadata.split()[2]
        for line in entries
        for metadata, name in [line.split("\t", 1)]
        if name.startswith(("src/", "prereg/", "pilot/"))
        or name in ("pyproject.toml", "uv.lock")
        or (name.startswith("data-manifest") and name.endswith(".json") and "/" not in name)
    }


def blob(root: Path, object_id: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", object_id], cwd=root, capture_output=True, check=True
    ).stdout


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
        result.append(
            arms.Rotation(
                tuple(int(clients.number(i)) for i in clients.items(row["held_out"])), fitted
            )
        )
    return result


def checkpoint(transcript: clients.Transcript, count: int | None = None) -> dict[str, object]:
    entries = list(transcript.entries.values())
    if count is not None:
        if len(entries) < count:
            raise clients.StopRun("transcript checkpoint is missing requests")
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
        self.mode = "stub" if synthetic else "live"
        self.directory = root / "data" / "run" / self.mode
        self.records = rewrites.load(root / "pilot" / "rewrites")
        self.deps = dependencies(root)
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

    def binding(self) -> State:
        path = self.directory / "binding.json"
        if path.exists():
            return clients.obj(json.loads(path.read_text()))
        return {"repairs": []}

    def guard(self, step: str, registered: str | None, repair: str | None) -> None:
        self.session.transcripts = {
            family: clients.Transcript(transcript.path.parent, family)
            for family, transcript in self.session.transcripts.items()
        }
        if not self.synthetic:
            try:
                tagged = git(self.root, "rev-parse", "refs/tags/registered^{commit}")
            except clients.StopRun:
                raise clients.StopRun("registered tag is required") from None
            if step == "held-out" and registered != tagged:
                raise clients.StopRun("--registered-commit must match the registered tag")
            try:
                git(self.root, "merge-base", "--is-ancestor", tagged, "HEAD")
            except clients.StopRun:
                raise clients.StopRun("registered commit must be an ancestor of HEAD") from None
            original = tree(self.root, tagged)
            head = tree(self.root, "HEAD")
            names = original.keys() | head.keys()
            current = {
                name: digest((self.root / name).read_bytes())
                if (self.root / name).is_file()
                else "deleted"
                for name in names
            }
            difference = {
                name: value
                for name, value in current.items()
                if name not in original
                or name not in head
                or digest(blob(self.root, original[name])) != value
            }
            fixed, code = split_dependencies(difference)
            if fixed:
                raise clients.StopRun(
                    "registered files differ from the registered commit: "
                    + ", ".join(sorted(fixed))
                )
            if git(self.root, "status", "--porcelain", "--untracked-files=no"):
                raise clients.StopRun("requires a clean work tree")
            binding = self.binding()
            repairs = clients.items(binding["repairs"])
            if code and not any(clients.obj(r).get("code") == code for r in repairs):
                if not repair or not repair.strip():
                    raise clients.StopRun(
                        "code requires --repair with a reason: " + ", ".join(sorted(code))
                    )
                repairs.append(
                    {
                        "reason": repair,
                        "commit": git(self.root, "rev-parse", "HEAD"),
                        "files": sorted(code),
                        "code": code,
                    }
                )
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.directory / "binding.tmp"
                temporary.write_bytes(encoded(binding))
                temporary.replace(self.directory / "binding.json")
            self.deps = current
        elif step == "held-out" and registered != git(self.root, "rev-parse", "HEAD"):
            raise clients.StopRun(
                "--registered-commit must match the current full commit for a stub run"
            )
        if self.session.spent > self.session.cap and step != "report":
            raise clients.StopRun("spending cap reached; no verdict")

    def state(self, step: str) -> State:
        path = self.directory / f"{step}.json"
        if not path.exists():
            raise clients.StopRun(f"step {step} has not completed")
        state = clients.obj(json.loads(path.read_text()))
        if (
            split_dependencies(cast(dict[str, str], state["dependencies"]))[0]
            != split_dependencies(self.deps)[0]
            or state["mode"] != self.mode
        ):
            raise clients.StopRun("registration digests changed; previous state cannot be used")
        for earlier, expected in clients.obj(state["previous"]).items():
            previous = self.directory / f"{earlier}.json"
            if not previous.exists() or digest(previous.read_bytes()) != expected:
                raise clients.StopRun("previous step changed")
        payload_path = self.directory / f"{step}-values.json"
        if (
            not payload_path.exists()
            or digest(payload_path.read_bytes()) != state["payload_digest"]
        ):
            raise clients.StopRun("step output digest changed")
        state["payload"] = clients.obj(json.loads(payload_path.read_text()))
        for family in budget.FAMILIES:
            expected = clients.obj(clients.obj(state["transcripts"])[family])
            found = checkpoint(
                self.session.transcripts[family], int(clients.number(expected["calls"]))
            )
            if found != expected:
                raise clients.StopRun("transcript checkpoint changed")
        return state

    def save(
        self, step: str, payload: State, *, status: str = "completed", reason: str = ""
    ) -> State:
        previous = {
            name: digest((self.directory / f"{name}.json").read_bytes())
            for name in STEPS[: STEPS.index(step)]
            if (self.directory / f"{name}.json").exists()
        }
        state: State = {
            "step": step,
            "mode": self.mode,
            "dependencies": self.deps,
            "registered_files": split_dependencies(self.deps)[0],
            "code_files": split_dependencies(self.deps)[1],
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
        payload_path = self.directory / f"{step}-values.json"
        payload_bytes = encoded(payload)
        payload_temp = payload_path.with_suffix(".tmp")
        payload_temp.write_bytes(payload_bytes)
        payload_temp.replace(payload_path)
        stored = {k: v for k, v in state.items() if k != "payload"}
        stored["payload_digest"] = digest(payload_bytes)
        temporary.write_bytes(encoded(stored))
        temporary.replace(path)
        return state

    def training(self) -> tuple[list[cohorts.Participant], cohorts.Cohorts]:
        if self.synthetic:
            people = stub.people(48, 41)
            return people, cohorts.generate(people, calibration_size=8)
        verify_data(self.root / "data" / "raw", self.root / "data-manifest.json")
        people = cohorts.participants(fetch.load(self.root / "data" / "raw"))
        document = cohorts.load(self.root / "prereg" / "cohorts.json")
        if cohorts.serialise(cohorts.generate(people)) != cohorts.serialise(document):
            raise clients.StopRun("training data do not regenerate registered cohorts")
        return people, document

    def request(
        self,
        family: clients.Family,
        requests: Sequence[clients.Request],
        *,
        recognition_only: bool = False,
    ) -> list[clients.Response]:
        return self.session.ask_many(family, requests, recognition=recognition_only)

    def judge_check(self) -> State:
        records = [r for r in self.records if r["set"] == "outside"]
        if self.synthetic:
            means = judge.human_ratings()
        else:
            try:
                verify_data(
                    self.root / "data" / "outside", self.root / "data-manifest-outside.json"
                )
                rated = outside.ratings(self.root / "data" / "outside")
                outside.check_against_paper(rated, outside.published())
                means = {name: rating.mean for name, rating in rated[2].items()}
            except (ValueError, OSError):
                raise clients.StopRun(
                    "outside US ratings are unavailable or disagree with the paper"
                ) from None
        human = np.array([means[str(r["id"])] for r in records])
        result: State = {"human": means}
        for family in budget.FAMILIES:
            try:
                responses = self.request(family, [prompts.evaluative(r) for r in records])
                check = judge.calibration(np.array([judge.evaluative(r) for r in responses]), human)
                result[family] = {
                    **asdict(check),
                    "reason": "" if check.passed else "evaluative calibration failed",
                }
            except clients.FamilyUnusable as error:
                result[family] = {"correlation": None, "passed": False, "reason": str(error)}
        decision = clients.obj(result["decision"])
        passed = bool(decision["passed"])
        return self.save(
            "judge-check",
            result,
            status="completed" if passed else "stopped",
            reason="" if passed else "the judge could not be calibrated",
        )

    def judge_all(self) -> State:
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
            replies = self.request(
                "second",
                [prompts.recognition(tasks[n]) for n in fetch.DILEMMAS],
                recognition_only=True,
            )
            result["recognition"] = {
                name: {
                    "reply": reply.text,
                    "flagged": recognition.flagged(
                        reply.text, recognition.keys(name, self.records)
                    ),
                }
                for name, reply in zip(fetch.DILEMMAS, replies, strict=True)
                if reply.text.strip()
            }
        except clients.FamilyUnusable as error:
            result["recognition"] = {}
            result["recognition_error"] = str(error)
            result["second"] = {"usable": False, "reason": str(error)}
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

    def calibrate(self) -> State:
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

    def held_out(self, registered: str | None) -> State:
        for step in STEPS[:3]:
            if self.state(step)["status"] != "completed":
                raise clients.StopRun(f"step {step} did not pass its gate")
        checked = clients.obj(self.state("judge-check")["payload"])
        if "decision" not in checked or not clients.obj(checked["decision"])["passed"]:
            raise clients.StopRun("the judge could not be calibrated")
        self.session.check_family("decision")
        commit = registered
        self.preflight()
        calibrated = clients.obj(self.state("calibrate")["payload"])
        if not self.synthetic:
            verify_data(self.root / "data" / "raw", self.root / "data-manifest.json")
        # The synthetic branch has no route through the real held-out loader.
        people = (
            stub.people(12, 83)
            if self.synthetic
            else cohorts.participants(fetch.load_registered_held_out(self.root / "data" / "raw"))
        )
        if not self.synthetic and len(people) != budget.HELD_OUT_PARTICIPANTS:
            raise clients.StopRun(
                "complete-records failure: registered participant total differs; no verdict"
            )
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
            except clients.StopRun:
                if family != "second":
                    raise
                unavailable["second"] = "complete-records failure"
        if "second" in unavailable:
            logs = {name: values for name, values in logs.items() if not name.startswith("second/")}
        return self.save(
            "held-out",
            {
                "logs": logs,
                "identities": [[p.experiment, p.participant] for p in people],
                "dilemma": data.dilemma.tolist(),
                "unavailable": unavailable,
                "registered_commit": commit,
            },
        )

    def verify_materials(self) -> None:
        if (
            rewrites.digest(self.records)
            != (self.root / "pilot" / "rewrites.sha256").read_text().strip()
        ):
            raise clients.StopRun("rewrite digest differs from computed materials")
        texts = self.root / "pilot" / "task-texts.json"
        if (
            digest(texts.read_bytes())
            != (self.root / "pilot" / "task-texts.sha256").read_text().strip()
        ):
            raise clients.StopRun("task-text digest differs from computed materials")
        people, document = self.training()
        if cohorts.serialise(document) != (self.root / "prereg" / "cohorts.json").read_bytes():
            raise clients.StopRun("training data do not regenerate registered cohorts")
        expected = prompts.fixtures(people, document, materials.load(texts), self.records)
        found = {
            path.name: path.read_bytes()
            for path in (self.root / "prereg" / "requests").iterdir()
            if path.name != "README.md"
        }
        if expected != found:
            raise clients.StopRun("request fixtures differ from regenerated materials")

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
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "execution.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise clients.StopRun("another command is using this run") from None
            try:
                self.guard(step, registered, repair)
            except clients.StopRun as error:
                if str(error) != "spending cap reached; no verdict":
                    raise
                state = self.save(step, {}, status="stopped", reason=str(error))
                return [
                    self.line(step, "status", state["status"]),
                    self.line(step, "reason", state["reason"]),
                ]
            if not self.synthetic and step != "report":
                self.verify_materials()
            if step == "report":
                return self.report()
            for earlier in STEPS[: STEPS.index(step)]:
                state = self.state(earlier)
                if state["status"] != "completed":
                    raise clients.StopRun(f"step {earlier} did not pass its gate")
            if step == "held-out":
                self.session.check_family("decision")
            path = self.directory / f"{step}.json"
            if path.exists():
                state = self.state(step)
                if state["status"] != "failed":
                    return [
                        self.line(step, "status", state["status"]),
                        self.line(step, "reason", state["reason"] or "none"),
                    ]
            try:
                if step != "held-out":
                    self.preflight()
                if step == "judge-check":
                    state = self.judge_check()
                elif step == "judge":
                    state = self.judge_all()
                elif step == "calibrate":
                    state = self.calibrate()
                elif step == "held-out":
                    state = self.held_out(registered)
                else:
                    raise clients.StopRun("unknown step")
            except clients.StepFailure as error:
                state = self.save(step, {}, status="failed", reason=str(error))
            except clients.StopRun as error:
                state = self.save(step, {}, status="stopped", reason=str(error))
            except (ValueError, KeyError, IndexError):
                state = self.save(
                    step,
                    {},
                    status="stopped",
                    reason="complete-records or response-data failure; no verdict",
                )
            return [
                self.line(step, "status", state["status"]),
                self.line(step, "reason", state["reason"] or "none"),
            ]

    def report(self) -> list[str]:
        from zoon.pilot.result import render

        if self.session.spent <= self.session.cap:
            for step in STEPS[:-1]:
                if (self.directory / f"{step}.json").exists() and self.state(step)[
                    "status"
                ] == "failed":
                    raise clients.StopRun(f"repeat step {step} before report")
        states: dict[str, State] = {}
        stopped = False
        for step in STEPS[:-1]:
            if stopped:
                break
            if (
                not (self.directory / f"{step}.json").exists()
                and self.session.spent > self.session.cap
            ):
                break
            states[step] = self.state(step)
            stopped = states[step]["status"] != "completed"
        lines = render(self, states)
        target = (
            self.directory / "result.md"
            if self.synthetic
            else self.root / "docs" / "pilot" / "result.md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        preamble = (
            "This is a synthetic stub run. It is not the pilot result."
            if self.synthetic
            else "This is the registered pilot result. The command output is reproduced below."
        )
        repairs = [line for line in lines if line.startswith(("[Repairs]", "[stub Repairs]"))]
        target.write_text(
            preamble
            + "\n\n```text\n"
            + "\n".join(lines)
            + "\n```\n"
            + "\n## Repairs\n\n```text\n"
            + "\n".join(repairs)
            + "\n```\n"
        )
        self.save("report", {"result_digest": digest(target.read_bytes())})
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
    except clients.StopRun as error:
        print(f"[{'stub ' if options.stub else ''}run] refused = {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
