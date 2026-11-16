"""Closed outcomes and exclusive readers are properties of the whole pilot."""

import ast
from pathlib import Path

from zoon.pilot import fetch

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
    for path, owner, name, node in functions():
        for call in calls(node, "save"):
            status = next((k.value for k in call.keywords if k.arg == "status"), None)
            if status is not None and not (
                isinstance(status, ast.Constant) and status.value in ("failed", "completed")
            ):
                assert isinstance(status, ast.Constant) and status.value == "stopped"
                sites.append((path.name, owner, name))
    assert sites == [("run.py", "Run", "execute")]


def test_outcomes_use_enum_members() -> None:
    for path in PILOT.rglob("*.py"):
        node = ast.parse(path.read_text())
        for outcome, enum in (("TerminalStop", "StopReason"), ("FamilyUnusable", "FamilyCause")):
            for call in calls(node, outcome):
                assert call.args, (path.name, call.lineno)
                reason = call.args[0]
                assert isinstance(reason, ast.Attribute), (path.name, call.lineno)
                base = reason.value
                assert (isinstance(base, ast.Name) and base.id == enum) or (
                    isinstance(base, ast.Attribute) and base.attr == enum
                ), (path.name, call.lineno)


def test_arbitrary_exceptions_never_become_closed_outcomes() -> None:
    forbidden = {"ValueError", "KeyError", "IndexError", "Exception"}
    for name in ("run.py", "clients.py"):
        for node in ast.walk(ast.parse((PILOT / name).read_text())):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = (
                {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
                if node.type
                else forbidden
            )
            if caught & forbidden:
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise) and child.exc is not None:
                        assert not calls(child.exc, "TerminalStop")
                        assert not calls(child.exc, "FamilyUnusable")


def test_state_binding_and_transcript_readers_are_exclusive() -> None:
    for path, owner, name, function in functions():
        aliases: dict[str, set[str]] = {}

        def kinds(
            node: ast.AST, owner: str = owner, aliases: dict[str, set[str]] = aliases
        ) -> set[str]:
            source = ast.unparse(node)
            found: set[str] = set()
            if (
                "self.directory" in source
                or "data/run/" in source
                or '"binding.json"' in source
                or "'binding.json'" in source
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
            if reader not in {"loads", "open", "read_text", "read_bytes"}:
                continue
            sources = kinds(node)
            if "state" in sources:
                assert (owner, name) in {("Run", "state"), ("Run", "binding")}, (
                    path.name,
                    name,
                    node.lineno,
                )
            if "transcript" in sources:
                assert owner == "Transcript", (path.name, name, node.lineno)
