"""Closed outcomes and exclusive readers are properties of the whole pilot."""

import ast
from pathlib import Path

from zoon.pilot import clients, fetch

PILOT = fetch.REPO / "src" / "zoon" / "pilot"


def calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, (ast.Name, ast.Attribute))
        and (child.func.id if isinstance(child.func, ast.Name) else child.func.attr) == name
    ]


def functions() -> list[tuple[Path, str, str, ast.AST]]:
    result: list[tuple[Path, str, str, ast.AST]] = []
    for path in PILOT.rglob("*.py"):
        module = ast.parse(path.read_text())
        for parent in [module, *[n for n in module.body if isinstance(n, ast.ClassDef)]]:
            owner = parent.name if isinstance(parent, ast.ClassDef) else ""
            outer = ast.Module(
                body=[
                    n
                    for n in parent.body
                    if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ],
                type_ignores=[],
            )
            result.append((path, owner, "<body>", outer))
            for node in parent.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.append(
                        (
                            path,
                            parent.name if isinstance(parent, ast.ClassDef) else "",
                            node.name,
                            node,
                        )
                    )
    return result


def test_only_execute_saves_stopped() -> None:
    sites: list[tuple[str, str, str]] = []
    matched = 0
    for path, owner, name, node in functions():
        for call in calls(node, "save"):
            matched += 1
            assert all(k.arg is not None for k in call.keywords), (path.name, call.lineno)
            status = next((k.value for k in call.keywords if k.arg == "status"), None)
            if status is not None and not (
                isinstance(status, ast.Constant) and status.value in ("failed", "completed")
            ):
                assert isinstance(status, ast.Constant) and status.value == "stopped"
                sites.append((path.name, owner, name))
    assert sites == [("run.py", "Run", "execute")]
    assert matched


def test_save_is_never_aliased_or_obtained_dynamically() -> None:
    matched = 0
    for path in PILOT.rglob("*.py"):
        module = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(module):
            if isinstance(node, ast.Attribute) and node.attr == "save":
                matched += 1
                parent = parents[node]
                assert isinstance(parent, ast.Call) and parent.func is node, path.name
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
            ):
                assert not any(
                    isinstance(a, ast.Constant) and a.value == "save" for a in node.args
                ), path.name
    assert matched


def test_outcome_types_have_no_subclasses() -> None:
    closed = {"GuardRefusal", "StepFailure", "FamilyUnusable", "TerminalStop"}
    matched: set[str] = set()
    for path in PILOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                matched.add(node.name)
                for base in node.bases:
                    assert not any(
                        (isinstance(child, ast.Name) and child.id in closed)
                        or (isinstance(child, ast.Attribute) and child.attr in closed)
                        for child in ast.walk(base)
                    ), (path.name, node.name)
    assert closed <= matched


def test_outcomes_use_enum_members() -> None:
    matched = {"TerminalStop": 0, "FamilyUnusable": 0}
    for path in PILOT.rglob("*.py"):
        node = ast.parse(path.read_text())
        for outcome, enum in (("TerminalStop", "StopReason"), ("FamilyUnusable", "FamilyCause")):
            for call in calls(node, outcome):
                matched[outcome] += 1
                assert call.args, (path.name, call.lineno)
                reason = call.args[0]
                assert isinstance(reason, ast.Attribute), (path.name, call.lineno)
                base = reason.value
                assert (isinstance(base, ast.Name) and base.id == enum) or (
                    isinstance(base, ast.Attribute) and base.attr == enum
                ), (path.name, call.lineno)
    assert all(matched.values())


def test_arbitrary_exceptions_never_become_closed_outcomes() -> None:
    forbidden = {
        "ValueError",
        "KeyError",
        "IndexError",
        "Exception",
        "BaseException",
        "OSError",
        "TypeError",
        "OverflowError",
        "RuntimeError",
    }
    matched = 0
    for path in PILOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = (
                {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
                if node.type
                else forbidden
            )
            if caught & forbidden:
                matched += 1
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise) and child.exc is not None:
                        assert not calls(child.exc, "TerminalStop")
                        assert not calls(child.exc, "FamilyUnusable")
    assert matched


def test_state_binding_and_transcript_readers_are_exclusive() -> None:
    matched = {"state": 0, "transcript": 0}
    for path, owner, name, function in functions():
        aliases: dict[str, set[str]] = {}

        def kinds(
            node: ast.AST, owner: str = owner, aliases: dict[str, set[str]] = aliases
        ) -> set[str]:
            source = ast.unparse(node)
            found: set[str] = set()
            if (
                ".directory" in source
                or "data/run/" in source
                or '"binding.json"' in source
                or "'binding.json'" in source
                or any(
                    f"'{step}.json'" in source
                    for step in ("judge-check", "judge", "calibrate", "held-out", "report")
                )
                or ("'run'" in source and "'data'" in source)
                or (
                    ".json" in source
                    and any(
                        isinstance(n, ast.Name) and n.id in {"step", "earlier"}
                        for n in ast.walk(node)
                    )
                )
            ):
                found.add("state")
            if "transcript" in source.lower() or (owner == "Transcript" and "self.path" in source):
                found.add("transcript")
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    found |= aliases.get(child.id, set())
            return found

        # A fixed point also follows aliases assigned inside branches and comprehensions.
        for _ in range(3):
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            aliases.setdefault(target.id, set()).update(kinds(node.value))
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, (ast.Name, ast.Attribute)
            ):
                continue
            reader = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            if reader not in {"load", "loads", "open", "read_text", "read_bytes"}:
                continue
            sources = kinds(node)
            if "state" in sources:
                matched["state"] += 1
                assert (owner, name) in {("Run", "state"), ("Run", "binding")}, (
                    path.name,
                    name,
                    node.lineno,
                )
            if "transcript" in sources:
                matched["transcript"] += 1
                assert owner == "Transcript", (path.name, name, node.lineno)
    assert all(matched.values())


def test_structure_scan_covers_every_module() -> None:
    found = {path for path, _, _, _ in functions()}
    assert found == set(PILOT.rglob("*.py"))
    assert found


def test_closed_reason_values_have_registered_sentences() -> None:
    registration = (fetch.REPO / "prereg/h1.md").read_text().splitlines()
    stops = [
        (
            "decision model withdrawn; no verdict",
            "decision model is withdrawn or reports another version; the "
            "spending cap is reached; the",
        ),
        (
            "decision model version changed; no verdict",
            "decision model is withdrawn or reports another version; the "
            "spending cap is reached; the",
        ),
        (
            "spending cap reached; no verdict",
            "decision model is withdrawn or reports another version; the "
            "spending cap is reached; the",
        ),
        (
            "complete-records failure; no verdict",
            "complete-records rule fails, which includes a held-out record "
            "that cannot be read as 13 valid",
        ),
        (
            "the judge could not be calibrated",
            'outcome is reported as "the judge could not be calibrated". It '
            "is not a result about either",
        ),
    ]
    families = [
        (
            "second family declared-answer share below one half",
            "half for any call to that family, as a judge or in the "
            "`prompted` arm, the family is reported",
        ),
        (
            "second family version changed",
            "before the run ends, the run stops and gives no verdict. If the "
            "second family does, at any",
        ),
        (
            "second family withdrawn",
            "request including the recognition request, its numerical results are not given. "
            "A model counts as withdrawn when the service answers that it does not",
        ),
        (
            "second family failed the stop rule",
            "hypothesis. If the second family fails the same test, its arms "
            "are not run and that is",
        ),
        (
            "second family complete-records failure",
            "stops and gives no verdict. A failure in an arm that uses the "
            "second family removes that",
        ),
    ]
    assert [r.value for r in clients.StopReason] == [v for v, _ in stops]
    assert [r.value for r in clients.FamilyCause] == [v for v, _ in families]
    for _, sentence in stops + families:
        assert sentence in registration
