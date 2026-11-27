"""Checkout gates, repair history, atomic step records and a process-scoped seal permit.

The order is a state contract, not a list of implemented commands. A future step
uses the same before/save protocol; only the command driver selects implementations.
Live mode verifies raw source hashes before every step. Stub mode never needs or
reads the raw sources; both modes verify training and regenerate registered materials.
"""

from __future__ import annotations

import fcntl
import io
import json
import os
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from zoon.interaction import chat, data, prompted, registered
from zoon.interaction.games import GAMES
from zoon.pilot import clients, prompts
from zoon.pilot.run import digest, encoded, git, verify_data

ORDER = ("calibrate", "closed-loop", "recall", "held-out", "report")
TAG = "registered-interaction"
SCHEMA = 1
_AUTHORITY = object()


class StateReplacedError(OSError):
    """State was replaced, but syncing its directory failed."""


def atomic(path: Path, document: dict[str, Any]) -> None:
    """One replace commits the output and its checksum together; never truncate state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded(document))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise StateReplacedError("state replaced but directory sync failed") from error


def git_paths(root: Path, *arguments: str) -> list[bytes]:
    """Read NUL-delimited Git output without quoting or trimming file names."""
    try:
        raw = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        raise clients.GuardRefusal("could not read checkout paths") from None
    return raw.split(b"\0")[:-1]


def snapshot_tree(root: Path, commit: str) -> dict[str, str]:
    """Keep literal tracked paths, file modes and content digests."""
    entries = [line.split(b"\t", 1) for line in git_paths(root, "ls-tree", "-rz", commit)]
    objects = [metadata.split()[2] for metadata, _ in entries]
    stream = io.BytesIO(
        subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=b"\n".join(objects) + b"\n",
            capture_output=True,
            check=True,
        ).stdout
    )
    result: dict[str, str] = {}
    for metadata, name in entries:
        size = int(stream.readline().split()[2])
        result[os.fsdecode(name)] = metadata.split()[0].decode() + ":" + digest(stream.read(size))
        stream.read(1)
    return result


def checkpoint(transcript: clients.Transcript, keys: list[str] | None = None) -> dict[str, Any]:
    """Hash canonical records by request key, independent of append/completion order."""
    selected = sorted(transcript.entries) if keys is None else keys
    if selected != sorted(set(selected)) or any(k not in transcript.entries for k in selected):
        raise clients.GuardRefusal("transcript checkpoint is missing or duplicates records")
    return {
        "keys": selected,
        "digest": digest(
            encoded([clients.entry_document(transcript.entries[k]) for k in selected])
        ),
    }


def require_terminal(payload: dict[str, Any], reason: str) -> None:
    outcome = payload.get("outcome", {})
    if (
        not reason.strip()
        or outcome.get("H2") != "no verdict"
        or outcome.get("H3") != "no verdict"
        or not isinstance(outcome.get("reason"), str)
        or not outcome["reason"].strip()
    ):
        raise clients.GuardRefusal("terminal state requires both no-verdict outcomes and a reason")


class HeldOutPermit:
    """Opaque capability, valid only during the issuing guard's locked held-out step."""

    def __init__(self, authority: object, guard: Guard) -> None:
        if authority is not _AUTHORITY:
            raise clients.GuardRefusal("only the guard can issue a held-out permit")
        self.activation = guard.activation
        self.guard: object = guard
        self.__authority = authority

    def authorized(self) -> bool:
        return self.__authority is _AUTHORITY


def require_permit(permit: object) -> Guard:
    if (
        not isinstance(permit, HeldOutPermit)
        or not permit.authorized()
        or not isinstance(permit.guard, Guard)
        or permit.activation != permit.guard.activation
        or not permit.guard.active
        or permit.guard.mode != "live"
        or permit.guard.checked_step != "held-out"
    ):
        raise clients.GuardRefusal("a live held-out guard permit is required")
    return permit.guard


def guarded_output(storage: Path, name: str, expected: object) -> bytes:
    """Read one recorded output; messages expose only its recorded name."""
    try:
        path = (storage / name).resolve()
        if not path.is_relative_to(storage):
            raise clients.GuardRefusal("simulation path is outside run storage")
        raw = path.read_bytes()
    except FileNotFoundError:
        raise clients.GuardRefusal(f"missing guarded output {name}") from None
    except (IsADirectoryError, NotADirectoryError, PermissionError):
        raise clients.GuardRefusal(f"guarded output {name} cannot be read") from None
    except (OSError, RuntimeError):
        raise chat.StorageInterruption(f"guarded output {name} storage failed") from None
    if not isinstance(expected, str) or len(expected) != 64 or digest(raw) != expected:
        raise clients.GuardRefusal("recorded simulation digest differs from its output")
    return raw


class Guard:
    def __init__(self, root: Path, *, stub: bool = False, storage: Path | None = None) -> None:
        self.root = root.resolve()
        self.mode = "stub" if stub else "live"
        # An explicit alternate data directory is useful for an isolated rehearsal.
        self.storage = (storage or self.root / "data" / "interaction").resolve()
        if not stub and self.storage != self.root / "data" / "interaction":
            raise clients.GuardRefusal("live state must use the interaction run directory")
        if self.storage == self.root / "data" or self.storage.is_relative_to(
            self.root / "data" / "run"
        ):
            raise clients.GuardRefusal("interaction storage must be separate from pilot state")
        if self.storage.is_relative_to(self.root / "data" / "transcripts"):
            raise clients.GuardRefusal(
                "interaction storage must be separate from pilot transcripts"
            )
        self.directory = self.storage / "run" / self.mode
        self.transcript_directory = self.storage / "transcripts" / self.mode
        self.active = False
        self.activation = 0
        self.checked_step: str | None = None
        self.registered_commit = ""
        self.commit = ""
        self.code: dict[str, str] = {}
        self.differences: dict[str, str] = {}
        self.repairs: list[dict[str, Any]] = []
        self.transcripts: dict[str, clients.Transcript] = {}
        self._families: set[clients.Family] = set()
        self._readers: dict[str, Callable[[list[str] | None], dict[str, Any]]] = {}

    @property
    def identity(self) -> dict[str, str]:
        return {"registered_commit": self.registered_commit, "mode": self.mode}

    @contextmanager
    def lock(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Modes in the same storage share a lock, but never records or answers.
        path = self.storage / "run" / "lock"
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise clients.GuardRefusal("another command holds the run lock") from None
            with path.open("a+b") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise clients.GuardRefusal("another command holds the run lock") from None
                try:
                    opened, current = os.fstat(handle.fileno()), os.stat(path)
                except OSError:
                    raise clients.GuardRefusal("run lock path changed while acquiring it") from None
                if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                    raise clients.GuardRefusal("run lock path changed while acquiring it")
                self.active = True
                try:
                    yield
                finally:
                    self.active = False
                    self.checked_step = None
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _require_lock(self) -> None:
        if not self.active:
            raise clients.GuardRefusal("the run lock is required")

    def binding(self) -> dict[str, Any] | None:
        path = self.directory / "binding.json"
        if not path.exists():
            return None
        document = json.loads(path.read_bytes())
        if document.get("schema") != SCHEMA or document.get("mode") != self.mode:
            raise clients.GuardRefusal("binding belongs to another schema or mode")
        return document

    def register_transcripts(self, *families: clients.Family) -> None:
        """Steps declare their transcript families before entering the guard."""
        self._families.update(families)

    def register_transcript(
        self, family: str, reader: Callable[[list[str] | None], dict[str, Any]]
    ) -> None:
        """Register a checkpoint reader for a step's own transcript format.

        The reader verifies identity on every opening and returns the existing
        keys/digest checkpoint shape. It receives None to select all records.
        """
        self._readers[family] = reader

    def transcript_checkpoint(self, family: str, keys: list[str] | None = None) -> dict[str, Any]:
        if family in self._readers:
            return self._readers[family](keys)
        if family not in self.transcripts:
            raise clients.GuardRefusal("step must register its transcript reader")
        return checkpoint(self.transcripts[family], keys)

    def before(self, step: str, repair: str | None = None) -> None:
        self._require_lock()
        self.activation += 1
        self.checked_step = None
        if step not in ORDER:
            raise clients.GuardRefusal("unknown step")
        try:
            self._checkout(repair)
            # Identity is checked for families a step registered, even before its first record.
            self.register_transcripts(*cast(tuple[clients.Family, ...], tuple(self.transcripts)))
            # Earlier steps may have registered additional families.
            for name in ORDER:
                path = self.directory / f"{name}.json"
                if path.exists():
                    for family in json.loads(path.read_bytes())["transcripts"]:
                        if family in ("decision", "second") and family not in self._readers:
                            self.register_transcripts(family)
            self.transcripts = {
                family: clients.Transcript(self.transcript_directory, family, self.identity)
                for family in self._families
            }
            for reader in self._readers.values():
                reader(None)
            states = {name: self.state(name) for name in ORDER}
            terminal = any(s and s["status"] == "terminal" for s in states.values())
            if step != "report":
                if terminal:
                    raise clients.GuardRefusal("run has a terminal no-verdict outcome")
                current = states[step]
                if current and current["status"] == "completed":
                    raise clients.GuardRefusal("a completed calling step cannot be repeated")
                for earlier in ORDER[: ORDER.index(step)]:
                    prior = states[earlier]
                    if not prior or prior["status"] != "completed":
                        raise clients.GuardRefusal(f"prerequisite {earlier} is not completed")
            elif not terminal and any(
                not prior or prior["status"] != "completed"
                for n in ORDER[:-1]
                for prior in [states[n]]
            ):
                raise clients.GuardRefusal("report requires completed steps or a terminal outcome")
            if git(self.root, "rev-parse", "HEAD") != self.commit:
                raise clients.GuardRefusal("HEAD moved during checkout verification")
            atomic(
                self.directory / "binding.json",
                {**self.identity, "schema": SCHEMA, "repairs": self.repairs},
            )
            self.checked_step = step
        except chat.GuardRefusal as error:
            raise clients.GuardRefusal(str(error)) from None
        except clients.GuardRefusal:
            raise
        except Exception as error:
            raise clients.GuardRefusal(
                "guard could not verify inputs: " + type(error).__name__
            ) from None

    def _checkout(self, repair: str | None) -> None:
        self.commit = git(self.root, "rev-parse", "HEAD")
        binding = self.binding()
        try:
            tagged = git(self.root, "rev-parse", f"refs/tags/{TAG}^{{commit}}")
        except clients.GuardRefusal:
            if self.mode != "stub":
                raise
            tagged = None
        self.registered_commit = (
            str(binding["registered_commit"]) if binding else tagged or self.commit
        )
        if tagged is not None and tagged != self.registered_commit:
            raise clients.GuardRefusal("binding belongs to another registered commit")
        git(self.root, "merge-base", "--is-ancestor", self.registered_commit, self.commit)
        original, head = (
            snapshot_tree(self.root, self.registered_commit),
            snapshot_tree(self.root, self.commit),
        )
        names = original.keys() | head.keys()
        ignored = git_paths(
            self.root,
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "src/",
            "interaction/",
        )
        if any(
            not (
                "__pycache__" in Path(name).parts[:-1]
                and name.endswith(".pyc")
                and (self.root / name).is_file()
            )
            for raw in ignored
            for name in [os.fsdecode(raw)]
        ):
            raise clients.GuardRefusal("ignored files under src/ or interaction/ are refused")
        if git(self.root, "status", "--porcelain", "--untracked-files=all") or any(
            (digest(path.read_bytes()) if (path := self.root / name).is_file() else "deleted")
            != head.get(name, "deleted").split(":", 1)[-1]
            for name in names
        ):
            raise clients.GuardRefusal("requires a clean work tree with disk bytes matching HEAD")
        self.differences = {
            n: head.get(n, "deleted") for n in sorted(names) if original.get(n) != head.get(n)
        }
        if any(n.startswith("interaction/") for n in self.differences):
            raise clients.GuardRefusal("registered files differ from the tag")
        self.code = {
            n: v for n, v in self.differences.items() if not n.startswith(("docs/", "tests/"))
        }
        self.repairs = binding["repairs"] if binding else []
        previous: str | None = None
        last_code: dict[str, str] = {}
        for index, record in enumerate(self.repairs):
            content = {k: v for k, v in record.items() if k != "digest"}
            if (
                record["digest"] != digest(encoded(content))
                or record["previous"] != previous
                or record["number"] != index + 1
                or record["registered_commit"] != self.registered_commit
                or record["mode"] != self.mode
                or not record["reason"].strip()
            ):
                raise clients.GuardRefusal("repair history is not an ordered registered binding")
            try:
                git(self.root, "merge-base", "--is-ancestor", record["commit"], self.commit)
            except clients.GuardRefusal:
                raise clients.GuardRefusal(
                    "declared repair commit must remain an ancestor of HEAD"
                ) from None
            commit_tree = snapshot_tree(self.root, record["commit"])
            expected = {
                n: commit_tree.get(n, "deleted")
                for n in sorted(original.keys() | commit_tree.keys())
                if original.get(n) != commit_tree.get(n) and not n.startswith(("docs/", "tests/"))
            }
            changed = sorted(
                n for n in last_code.keys() | expected.keys() if last_code.get(n) != expected.get(n)
            )
            if record["code"] != expected or record["files"] != changed:
                raise clients.GuardRefusal("repair snapshot does not match its committed bytes")
            previous = record["digest"]
            last_code = expected
        if self.code != (self.repairs[-1]["code"] if self.repairs else {}):
            if not repair or not repair.strip():
                raise clients.GuardRefusal("changed code requires a declared repair reason")
            record = {
                **self.identity,
                "number": len(self.repairs) + 1,
                "previous": previous,
                "reason": repair,
                "commit": self.commit,
                "code": self.code,
                "files": sorted(
                    n
                    for n in last_code.keys() | self.code.keys()
                    if last_code.get(n) != self.code.get(n)
                ),
            }
            self.repairs.append({**record, "digest": digest(encoded(record))})
        # Verification failure must not publish a declaration. before() calls this
        # verifier before creating the binding or any step record.
        self.verify_materials()

    def verify_materials(self) -> list[data.Group]:
        manifest = self.root / "interaction" / "data-manifest.json"
        if self.mode == "live":
            verify_data(self.root / "data" / "groups" / "raw", manifest)
        groups = data.load(self.root / "data" / "groups" / "training.json", manifest=manifest)
        directory = self.root / "interaction"
        training_digest = registered.training_digest(manifest)
        registered.read_fitted(directory / "fitted.json", manifest)
        registered.read_amounts(directory / "amounts.json", manifest)
        expected: dict[str, bytes] = {
            f"instructions/{cell}.txt": prompted.instructions(game).encode()
            for cell, game in sorted(GAMES.items())
        }
        expected["instructions/sha256.json"] = (
            json.dumps(
                {
                    f"{c}.txt": digest(prompted.instructions(g).encode())
                    for c, g in sorted(GAMES.items())
                },
                indent=2,
            )
            + "\n"
        ).encode()
        expected.update(
            {f"requests/{n}": prompts.decision_bytes(r) for n, r in prompted.fixtures().items()}
        )
        expected.update(
            {f"chat-requests/{n}": text.encode() for n, text in chat.fixtures().items()}
        )
        expected["chat-requests/sha256.json"] = (
            json.dumps({n: digest(t.encode()) for n, t in chat.fixtures().items()}, indent=2) + "\n"
        ).encode()
        expected["amounts.json"] = registered.amounts_bytes(groups, training_digest)
        if any((directory / n).read_bytes() != raw for n, raw in expected.items()):
            raise clients.GuardRefusal("registered materials do not regenerate")
        for name in ("instructions", "requests", "chat-requests"):
            if {p.name for p in (directory / name).iterdir()} != {
                Path(n).name for n in expected if n.startswith(name + "/")
            }:
                raise clients.GuardRefusal("registered material file set changed")
        return groups

    def input_bytes(self, name: str) -> bytes:
        """Read a cited tracked input from the registered tree, even after docs change."""
        self._require_lock()
        if self.checked_step is None or name not in snapshot_tree(
            self.root, self.registered_commit
        ):
            raise clients.GuardRefusal("input must be tracked at the guarded registration")
        try:
            return subprocess.run(
                ["git", "cat-file", "blob", f"{self.registered_commit}:{name}"],
                cwd=self.root,
                capture_output=True,
                check=True,
            ).stdout
        except subprocess.CalledProcessError:
            raise clients.GuardRefusal("could not read the registered input bytes") from None

    def state(self, step: str) -> dict[str, Any] | None:
        self._require_lock()
        path = self.directory / f"{step}.json"
        if not path.exists():
            return None
        state = json.loads(path.read_bytes())
        content = {k: v for k, v in state.items() if k != "digest"}
        if state["digest"] != digest(encoded(content)):
            raise clients.GuardRefusal("step state digest changed")
        if any(
            state.get(k) != v for k, v in {**self.identity, "schema": SCHEMA, "step": step}.items()
        ):
            raise clients.GuardRefusal("state belongs to another step, commit or mode")
        if state["status"] not in ("running", "interrupted", "completed", "terminal"):
            raise clients.GuardRefusal("invalid step status")
        if state["status"] == "terminal":
            require_terminal(state["payload"], state["reason"])
        if state["payload_digest"] != digest(encoded(state["payload"])):
            raise clients.GuardRefusal("step output digest changed")
        allowed: dict[str | None, dict[str, str]] = {None: {}}
        allowed.update({r["digest"]: r["code"] for r in self.repairs})
        if state["repair"] not in allowed or state["code_snapshot"] != allowed[state["repair"]]:
            raise clients.GuardRefusal("state has an undeclared code snapshot")
        if state["previous"] != self.previous(step):
            raise clients.GuardRefusal("exact earlier-step states changed, including absence")
        for family, expected in state["transcripts"].items():
            if self.transcript_checkpoint(family, expected["keys"]) != expected:
                raise clients.GuardRefusal("recorded transcript checkpoint changed")
        return state

    def previous(self, step: str) -> dict[str, str | None]:
        return {
            earlier: digest(path.read_bytes()) if path.exists() else None
            for earlier in ORDER[: ORDER.index(step)]
            for path in [self.directory / f"{earlier}.json"]
        }

    def save(
        self,
        step: str,
        payload: dict[str, Any],
        *,
        families: tuple[str, ...],
        status: str,
        reason: str = "",
    ) -> dict[str, Any]:
        self._require_lock()
        if self.checked_step != step:
            raise clients.GuardRefusal("step has not passed the guard")
        if status == "terminal":
            require_terminal(payload, reason)
        old = self.state(step)
        if old and old["status"] in ("completed", "terminal") and step != "report":
            raise clients.GuardRefusal("a completed or terminal step cannot be rewritten")
        if old and set(old["transcripts"]) != set(families):
            raise clients.GuardRefusal("step transcript families changed")
        state = {
            **self.identity,
            "schema": SCHEMA,
            "step": step,
            "commit": self.commit,
            "code_snapshot": self.code,
            "repair": self.repairs[-1]["digest"] if self.repairs else None,
            "differences": self.differences,
            "previous": self.previous(step),
            "status": status,
            "reason": reason,
            "transcripts": {family: self.transcript_checkpoint(family) for family in families},
            "payload": payload,
            "payload_digest": digest(encoded(payload)),
        }
        result = {**state, "digest": digest(encoded(state))}
        if old and old["status"] == "completed":
            if result["payload_digest"] != old["payload_digest"]:
                raise clients.GuardRefusal("report must reproduce the same output")
            return old
        atomic(self.directory / f"{step}.json", result)
        return result

    def recall_chat_record(self) -> dict[str, Any]:
        """Return the completed recall record; later rebuild failures live in later states."""
        state = self.state("recall")
        if state is None or state["status"] != "completed":
            raise clients.GuardRefusal("completed recall is required for the recall chat record")
        return state["payload"]["chat"]

    def permit(self) -> HeldOutPermit:
        self._require_lock()
        if self.mode != "live" or self.checked_step != "held-out":
            raise clients.GuardRefusal("only the live held-out step receives a permit")
        simulation = self.state("closed-loop")
        if (
            not simulation
            or simulation["status"] != "completed"
            or not simulation["payload"].get("digests")
        ):
            raise clients.GuardRefusal("completed simulation with recorded digests is required")
        outputs = simulation["payload"]["digests"]
        if not isinstance(outputs, dict) or not outputs:
            raise clients.GuardRefusal("simulation digests must name recorded files")
        for name, expected in cast(dict[str, object], outputs).items():
            guarded_output(self.storage, name, expected)
        return HeldOutPermit(_AUTHORITY, self)
