"""The pilot notes quote only numbers that their pasted report output contains."""

import re
from pathlib import Path

import pytest

NOTES = Path(__file__).parent.parent / "docs" / "pilot"
NUMBER = re.compile(r"[+-]?\d+\.\d{3,}")
# Each note that pastes report output, with the first line of that output and a line that
# only a complete run prints.
PASTED = {
    "budget.md": ("zoon pilot: spending bound", "[budget] total at most 120 USD"),
    "power.md": ("zoon pilot: power calculation", "[pooled] held-out participants"),
    "lean.md": (
        "zoon pilot: a lean from published human norms and codes",
        "[experiment 4] the whole thing | coded + learning minus pooled + learning",
    ),
}


def split_note(name: str) -> tuple[str, str]:
    """Return the prose and the pasted report output, which is the last code block."""
    text = (NOTES / name).read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    paste = text[start : text.rindex("\n```")]
    return text[:start], paste


@pytest.mark.parametrize("name", PASTED)
def test_report_output_is_pasted(name: str) -> None:
    _, paste = split_note(name)
    first, later = PASTED[name]
    assert paste.startswith(first)
    assert later in paste


@pytest.mark.parametrize("name", PASTED)
def test_every_quoted_number_is_in_the_report_output(name: str) -> None:
    """Figures with three or more decimals, and every budget figure, must appear in the paste.
    Protects the entry for each note in PASTED: without it, a figure in that note that the
    report never printed would pass."""
    prose, paste = split_note(name)
    pattern = re.compile(r"[+-]?\d+(?:\.\d+)?") if name == "budget.md" else NUMBER
    printed = {number.lstrip("+") for number in pattern.findall(paste)}
    prose = re.sub(r"https?://\S+", "", prose)  # a DOI is not a figure
    quoted = {number.lstrip("+") for number in pattern.findall(prose)}
    assert quoted, "the note quotes no figures"
    assert sorted(quoted - printed) == []


# A note that pastes no output of its own may quote figures only from these notes' output.
CITING = {"arms.md": ("power.md", "lean.md")}


@pytest.mark.parametrize("name", CITING)
def test_a_note_that_cites_quotes_only_printed_figures(name: str) -> None:
    """Protects the entry for each note in CITING: a figure in that note that neither cited
    report printed would otherwise pass."""
    printed: set[str] = set()
    for cited in CITING[name]:
        _, paste = split_note(cited)
        printed |= {number.lstrip("+") for number in NUMBER.findall(paste)}
    prose = re.sub(r"https?://\S+", "", (NOTES / name).read_text())
    quoted = {number.lstrip("+") for number in NUMBER.findall(prose)}
    assert quoted, "the note quotes no figures"
    assert sorted(quoted - printed) == []


def test_interaction_registration_links_and_existing_paths() -> None:
    """Only the explicitly listed analysis additions may name paths not present today."""
    root = NOTES.parent.parent
    document = root / "interaction" / "registration.md"
    check_interaction_registration(root, document)
    text = document.read_text()
    for path in (
        "src/zoon/interaction/chat.py",
        "interaction/chat-requests/",
        "src/zoon/interaction/run.py",
        "src/zoon/interaction/guard.py",
        "src/zoon/interaction/decisions.py",
    ):
        assert f"| `{path}` | Existing today |" in text
        assert f"- `{path}`" not in text


def check_interaction_registration(root: Path, document: Path) -> None:
    text = document.read_text()
    heading = "### Paths to be added by the analysis code\n"
    assert text.count(heading) == 1, "analysis additions heading must occur once"
    additions = text.partition(heading)[2].split("\n## ", 1)[0]
    exempt = set(re.findall(r"^- `([^`]+)`$", additions, flags=re.MULTILINE))
    assert exempt, "analysis additions must be an explicit path list"
    assert "do not exist today" in additions
    assert "exempt from the existing-path check" in additions
    for path in exempt:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        candidate = root / path
        assert candidate.resolve().is_relative_to(root.resolve()), path
        assert not candidate.exists(), f"exempt path exists: {path}"

    added = set(
        re.findall(r"^\| `([^`]+)` \| Added by the analysis code \|", text, flags=re.MULTILINE)
    )
    assert exempt == added, "every exempt path must have an added-file row, and conversely"

    existing = re.findall(r"^\| `([^`]+)` \| Existing today \|", text, flags=re.MULTILINE)
    assert existing, "the file table must distinguish existing paths"
    if document.name == "registration.md" and root == NOTES.parent.parent:
        assert {"interaction/fitted.json", "interaction/amounts.json"} <= set(existing)
    for path in existing:
        assert path not in exempt, f"existing path is exempt: {path}"
        candidate = root / path
        assert candidate.resolve().is_relative_to(root.resolve()), path
        assert candidate.exists(), f"existing path is missing: {path}"

    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    assert links, "the registration must link its sources"
    for link in links:
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
            continue
        target, _, fragment = link.partition("#")
        linked = (document.parent / target).resolve()
        assert linked.is_relative_to(root.resolve()), f"link leaves repository: {link}"
        assert linked.exists(), link
        if fragment:
            headings = re.findall(r"^#+ (.+)$", linked.read_text(), flags=re.MULTILINE)
            anchors = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-") for h in headings}
            assert fragment in anchors, link

    # Links resolve from interaction/; all other named paths resolve from the root.
    prose = re.sub(r"!?\[[^\]]*\]\([^)]+\)", "", text)
    paths = set(
        re.findall(
            r"(?<![\w./-])(?:[\w.-]+/)+[\w.-]*|"
            r"(?<![\w./-])[\w.-]+\.(?:md|py|json|txt)(?![\w.-])",
            prose,
        )
    )
    assert paths and exempt <= paths
    for path in paths - exempt:
        # A short filename in prose names a note already linked in the document.
        candidates = [root / path]
        if "/" not in path:
            candidates.extend(
                document.parent / link.split("#", 1)[0]
                for link in links
                if Path(link.split("#", 1)[0]).name == path
            )
        assert any(candidate.exists() for candidate in candidates), path


@pytest.mark.parametrize(
    "mutation",
    [
        "existing exemption",
        "exemption on disk",
        "missing existing",
        "escape",
        "heading",
        "exemption without row",
    ],
)
def test_registration_path_check_rejects_mutations(tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "repository"
    folder = root / "interaction"
    folder.mkdir(parents=True)
    document = folder / "registration.md"
    (root / "README.md").write_text("# Source\n")
    text = """# Registration

[Source](../README.md)

| File | State | Holds |
|---|---|---|
| `interaction/registration.md` | Existing today | This file |
| `interaction/fitted.json` | Added by the analysis code | Fitted numbers |

### Paths to be added by the analysis code

These paths do not exist today and are exempt from the existing-path check:

- `interaction/fitted.json`
"""
    document.write_text(text)
    check_interaction_registration(root, document)
    if mutation == "existing exemption":
        text += "- `README.md`\n"
    elif mutation == "exemption without row":
        text += "- `interaction/unlisted.json`\n"
    elif mutation == "exemption on disk":
        (folder / "fitted.json").write_text("{}\n")
    elif mutation == "missing existing":
        text = text.replace(
            "| `interaction/registration.md` | Existing today | This file |",
            "| `interaction/fitted.json` | Existing today | Missing |",
        )
    elif mutation == "escape":
        (tmp_path / "outside.md").write_text("# Outside\n")
        text = text.replace("../README.md", "../../outside.md")
    else:
        text = text.replace("### Paths to be added by the analysis code", "### Future paths")
    document.write_text(text)
    with pytest.raises(AssertionError):
        check_interaction_registration(root, document)


def test_interaction_run_status_matches_available_steps() -> None:
    root = NOTES.parent.parent
    text = " ".join((root / "interaction" / "registration.md").read_text().split())
    assert (
        "The second sealing step, scoring and intervals, run guards, step state, calibration, "
        "closed-loop, recall and the operator chat-ending command are implemented; "
        "held-out and report remain open work."
    ) in text
