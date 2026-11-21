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
    document = root / "prereg" / "interaction.md"
    text = document.read_text()
    additions = text.split("### Paths to be added by the analysis code\n", 1)[1].split("\n## ", 1)[
        0
    ]
    exempt = set(re.findall(r"^- `([^`]+)`$", additions, flags=re.MULTILINE))
    assert exempt, "analysis additions must be an explicit path list"
    assert "do not exist today" in additions
    assert "exempt from the existing-path check" in additions
    for path in exempt:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts

    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    assert links, "the registration must link its sources"
    for link in links:
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
            continue
        target, _, fragment = link.partition("#")
        linked = document.parent / target
        assert linked.exists(), link
        if fragment:
            headings = re.findall(r"^#+ (.+)$", linked.read_text(), flags=re.MULTILINE)
            anchors = {re.sub(r"[^\w -]", "", h.lower()).replace(" ", "-") for h in headings}
            assert fragment in anchors, link

    # Links resolve from prereg/; all other named paths resolve from the root.
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
