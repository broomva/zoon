"""Published task texts and the outcomes selected by the recorded answer and condition.

The PDF labels its columns Yes and No and its rows Success and Failure. The raw
question/valence column suffixes are YS (Yes succeeds) and NS (No succeeds), as
`fetch._response_columns` documents. That function chooses YS exactly when
CBR_IS_ACTION[dilemma] == (condition == "cbr_success"). Thus the selected PDF cell
is the answer's column and Success iff that answer matches the suffix. This is
also the `success` field emitted by `fetch._trials`; ratings do not select cells.

Extraction uses the PDF's text-column origins to keep table cells together across
page breaks. It joins extracted words with spaces, removes headings and cell labels,
and otherwise preserves the source wording. No participant records are read here.
"""

import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from zoon.pilot.fetch import CBR_IS_ACTION, CONDITIONS, DILEMMAS, REPO

DIRECTORY = REPO / "data" / "materials"
PDF = DIRECTORY / "dilemmas-and-outcomes.pdf"
MANIFEST = REPO / "data-manifest-materials.json"
TEXTS = REPO / "pilot" / "task-texts.json"
DIGEST = REPO / "pilot" / "task-texts.sha256"
URL = "https://osf.io/download/sgf5y/"
Answer = Literal["Yes", "No"]

# Inclusive, one-based PDF pages, including continuations of the outcome tables.
PAGES = {
    "Ladder": (4, 4),
    "Tyran": (5, 5),
    "Lifeboat": (6, 7),
    "Suicide": (8, 9),
    "Medicine": (10, 10),
    "RAF": (11, 11),
    "Rwanda": (12, 12),
    "Ransom": (13, 13),
    "Tax": (18, 18),
    "Operations": (16, 17),
    "Vet": (19, 20),
    "Endow": (21, 22),
    "RobinHood": (14, 15),
}
SOURCE = {
    "authors": "Maier, Cheung and Lieder",
    "title": "dilemmas and outcomes.pdf",
    "url": URL,
    "materials_url": "https://osf.io/4up5z/",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "changes": "Mechanically extracted; headings and table labels omitted; whitespace joined.",
}


class Task(TypedDict):
    text: str
    outcomes: dict[str, dict[str, str]]  # answer -> success/failure -> text


class Document(TypedDict):
    source: dict[str, str]
    dilemmas: dict[str, Task]


class Manifest(TypedDict):
    dataset: str
    licence: str
    files: dict[str, dict[str, str | int]]


def fetch(directory: Path = DIRECTORY, manifest: Path = MANIFEST) -> None:
    """Download the PDF if absent; record its URL, size and hash, rejecting changes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PDF.name
    if not path.exists():
        with urllib.request.urlopen(URL) as response:
            path.write_bytes(response.read())
    found: dict[str, str | int] = {
        "url": URL,
        "size": path.stat().st_size,
        "sha256": digest(path.read_bytes()),
    }
    recorded: Manifest = (
        cast(Manifest, json.loads(manifest.read_text()))
        if manifest.exists()
        else {
            "dataset": SOURCE["materials_url"],
            "licence": SOURCE["license"],
            "files": {PDF.name: found},
        }
    )
    if recorded["files"].get(PDF.name) != found:
        raise ValueError("materials PDF does not match the manifest")
    manifest.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalise(text: str) -> str:
    """Ignore whitespace and hyphenation, including soft and line-break hyphens."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)[\-\u00ad\u2010]\s*\n\s*(?=\w)", "", text)
    return " ".join(re.sub(r"[\-\u00ad\u2010-\u2015]", " ", text).split())


def extracted_columns(path: Path = PDF) -> dict[str, tuple[str, str, str]]:
    """Extract the vignette and each answer column, including continuation pages."""
    # Only extraction needs this development dependency; rendering needs the JSON alone.
    from pypdf import PdfReader

    reader = PdfReader(path)
    result: dict[str, tuple[str, str, str]] = {}
    for name, (first, last) in PAGES.items():
        columns: list[list[str]] = [[], [], []]

        def visit(
            text: str,
            cm: list[float],
            tm: list[float],
            font: object,
            size: float,
            target: list[list[str]] = columns,
        ) -> None:
            if not text.strip() or cm[5] > 790:
                return  # page number
            origin = cm[4]
            column = 1 if 140 < origin < 150 else 2 if 320 < origin < 330 else 0
            if column or 71 < origin < 73:
                target[column].append(text.strip())

        for page in reader.pages[first - 1 : last]:
            page.extract_text(visitor_text=visit)
        result[name] = cast(tuple[str, str, str], tuple(" ".join(c) for c in columns))
    return result


def extract(path: Path = PDF) -> Document:
    """Build the committed document mechanically from the published PDF."""
    tasks: dict[str, Task] = {}
    for name, (body, yes, no) in extracted_columns(path).items():
        # Titles precede the first vignette sentence, which always starts It, You or After.
        start = re.search(r"\b(?:It|You|After)\b", body)
        if start is None:
            raise ValueError(f"{name}: no vignette start")
        text = body[start.start() :]
        if not text.endswith("?"):
            raise ValueError(f"{name}: no final question")
        outcomes: dict[str, dict[str, str]] = {}
        for answer, column in (("Yes", yes), ("No", no)):
            match = re.fullmatch(
                rf"{answer} {answer} success\s*: (.+) {answer} failure\s*: (.+)", column
            )
            if match is None:
                raise ValueError(f"{name}: cannot separate {answer} outcomes")
            outcomes[answer] = {"success": match[1], "failure": match[2]}
        tasks[name] = {"text": text, "outcomes": outcomes}
    return {"source": SOURCE.copy(), "dilemmas": tasks}


def load(path: Path = TEXTS) -> dict[str, Task]:
    document = cast(Document, json.loads(path.read_text()))
    if tuple(document["dilemmas"]) != DILEMMAS:
        raise ValueError("task IDs must match DILEMMAS in order")
    return document["dilemmas"]


def serialise(document: Document) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode()


def outcome_text(
    dilemma: str,
    answer: Answer,
    *,
    success: bool | None = None,
    condition: str | None = None,
    tasks: dict[str, Task] | None = None,
) -> str:
    """Select by recorded success or derive it from the feedback condition; reject conflicts."""
    if answer not in ("Yes", "No"):
        raise ValueError("answer must be Yes or No")
    if condition is not None:
        if condition not in CONDITIONS:
            raise ValueError("unknown feedback condition")
        cbr = (answer == "Yes") == CBR_IS_ACTION[dilemma]
        derived = cbr == (condition == "cbr_success")
        if success is not None and success != derived:
            raise ValueError("success and condition disagree")
        success = derived
    if success is None:
        raise ValueError("success or condition is required")
    texts = load() if tasks is None else tasks
    return texts[dilemma]["outcomes"][answer]["success" if success else "failure"]


def main(arguments: Sequence[str]) -> None:
    if list(arguments) == ["fetch"]:
        fetch()
        print("[materials] PDF manifest verified = True")
    elif list(arguments) == ["build"]:
        content = serialise(extract())
        TEXTS.write_bytes(content)
        DIGEST.write_text(digest(content) + "\n")
        print(f"[materials] dilemmas = {len(load())}")
    else:
        sys.exit("usage: python -m zoon.pilot.materials fetch|build")


if __name__ == "__main__":
    main(sys.argv[1:])
