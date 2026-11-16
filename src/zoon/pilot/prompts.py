"""Stable, pure request renderers; requests are data and never sent by this module.

Decision requests contain state, one question and declared answers with descriptions.
Second-family requests are a single text. Recognition has only the second-family form.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from zoon.pilot import cohorts, fetch, materials, outside, rewrites

DIRECTORY = fetch.REPO / "prereg" / "requests"
STRICT = (
    "Does the suggested action involve killing a person?",
    "Are the people who would be harmed by the suggested action among the people who would "
    "benefit from it?",
    "Would the person deciding benefit personally from the suggested action?",
    "Would it be reasonable to describe this dilemma the other way round, asking about the "
    "opposite action?",
    "Which of these contexts does the dilemma belong to?",
)
RECOGNITION = (
    "This describes a real case with the details removed. Name the case, its place and its "
    "year, or answer unknown."
)
PREDICT = "Which answer does this person give to the current dilemma?"
YES_NO = "Answer with one word: Yes or No."
SCALE = "Answer with one digit from 1 (completely inappropriate) to 7 (completely appropriate)."
INTRO = (
    "A person is taking part in a study. They read moral dilemmas one at a time. For each one\n"
    "they answer Yes or No to the question at the end. They are then told what happened as a\n"
    "result, and they rate how good or bad the consequences were, from -100 (very bad) to 100\n"
    "(very good)."
)


@dataclass(frozen=True)
class Answer:
    answer: str
    description: str


@dataclass(frozen=True)
class Decision:
    state: str
    question: str
    answers: tuple[Answer, ...]


def contexts() -> tuple[str, ...]:
    """Contexts in their first-appearance order in the committed source table."""
    return outside.CONTEXTS


def declared(labels: Sequence[str]) -> tuple[Answer, ...]:
    return tuple(Answer(label, label) for label in labels)


def evaluative(record: Mapping[str, object], *, reverse: bool = False) -> Decision:
    text = str(record["reversed" if reverse else "rewrite"])
    action = str(record["reversed_suggested_action" if reverse else "suggested_action"])
    return Decision(
        rewrites.question(text)[0].rstrip(),
        f"How appropriate is it to {action}?",
        tuple(
            Answer(
                str(i),
                f"{i} (completely inappropriate)"
                if i == 1
                else f"{i} (completely appropriate)"
                if i == 7
                else str(i),
            )
            for i in range(1, 8)
        ),
    )


def strict(record: Mapping[str, object], index: int, *, reverse: bool = False) -> Decision:
    if not 0 <= index < len(STRICT):
        raise ValueError("strict question index out of range")
    text = str(record["reversed" if reverse else "rewrite"])
    action = str(record["reversed_suggested_action" if reverse else "suggested_action"])
    question = STRICT[index]
    if index < 4:
        question += f" The suggested action is: {action}."
    return Decision(
        rewrites.question(text)[0].rstrip(),
        question,
        declared(contexts() if index == 4 else ("Yes", "No")),
    )


def second(request: Decision) -> str:
    labels = tuple(answer.answer for answer in request.answers)
    if labels == tuple(str(i) for i in range(1, 8)):
        closing = SCALE
    elif labels == ("Yes", "No"):
        closing = YES_NO
    elif labels == contexts():
        closing = (
            "\n".join(
                f"{letter}. {context}" for letter, context in zip("ABCDEFGHI", labels, strict=True)
            )
            + "\nAnswer with one letter."
        )
    else:
        raise ValueError("unsupported declared answers")
    return f"{request.state}\n\n{request.question}\n{closing}"


def recognition(record: Mapping[str, object]) -> str:
    return f"{record['rewrite']}\n\n{RECOGNITION}"


def trial(trial: cohorts.Trial, tasks: dict[str, materials.Task], *, complete: bool = True) -> str:
    text = f"Dilemma:\n{tasks[trial.dilemma]['text']}\nAnswer: {trial.answer}"
    if complete:
        outcome = materials.outcome_text(
            trial.dilemma, trial.answer, success=trial.success, tasks=tasks
        )
        text += f"\nWhat happened: {outcome}\nRating: {trial.rating}"
    return text


def example(
    person: cohorts.Participant, last: int, label: str, tasks: dict[str, materials.Task]
) -> str:
    if not 2 <= last <= len(person.trials):
        raise ValueError("example must end after its first trial")
    return f"Person {label}:\n" + "\n\n".join(
        trial(t, tasks, complete=i < last) for i, t in enumerate(person.trials[:last], 1)
    )


def prompted_state(examples: str, record: str, current: str) -> str:
    return "\n\n".join(
        (
            INTRO,
            "Here are records from other people in the same study.",
            examples,
            "Here is the record of the person you are asked about.",
            record,
            f"Current dilemma:\n{current}",
        )
    )


def examples(
    rotation: cohorts.Rotation,
    people: Sequence[cohorts.Participant],
    tasks: dict[str, materials.Task],
) -> str:
    if len(rotation["examples"]) != 2:
        raise ValueError("each rotation must have two examples")
    lookup = {(p.experiment, p.participant): p for p in people}
    result: list[str] = []
    for label, spec in zip(("A", "B"), rotation["examples"], strict=True):
        person = lookup[spec["experiment"], spec["participant"]]
        if any(t.dilemma in rotation["held_out"] for t in person.trials[: spec["last"]]):
            raise ValueError("example contains a held-out dilemma")
        result.append(example(person, spec["last"], label, tasks))
    return "\n\n".join(result)


def prompted(
    person: cohorts.Participant, current: int, example_text: str, tasks: dict[str, materials.Task]
) -> Decision:
    """Render a one-based current trial and all its earlier trials, without the target answer."""
    if not 1 <= current <= len(person.trials):
        raise ValueError("current trial out of range")
    history = "\n\n".join(trial(t, tasks) for t in person.trials[: current - 1])
    if not history:
        history = "This person has not answered any dilemma yet."
    state = prompted_state(example_text, history, tasks[person.trials[current - 1].dilemma]["text"])
    return Decision(state, PREDICT, declared(("Yes", "No")))


def decision_bytes(request: Decision) -> bytes:
    return (json.dumps(asdict(request), indent=2, ensure_ascii=False) + "\n").encode()


def fixtures(
    people: Sequence[cohorts.Participant],
    document: cohorts.Cohorts,
    tasks: dict[str, materials.Task],
    records: Sequence[rewrites.Record],
) -> dict[str, bytes]:
    """One request of each kind, using the first calibration participant and first rotation."""
    record = next(r for r in records if r["id"] == "Ladder")
    spec = document["calibration"][0]
    person = next(
        p
        for p in people
        if (p.experiment, p.participant) == (spec["experiment"], spec["participant"])
    )
    shown = examples(document["rotations"][0], people, tasks)
    requests = {"evaluative": evaluative(record)}
    requests.update({f"strict-{i + 1}": strict(record, i) for i in range(len(STRICT))})
    requests["prompted-empty"] = prompted(person, 1, shown, tasks)
    requests["prompted-history"] = prompted(person, 5, shown, tasks)
    result: dict[str, bytes] = {}
    for name, request in requests.items():
        result[f"decision-{name}.json"] = decision_bytes(request)
        result[f"second-{name}.txt"] = second(request).encode()
    result["second-recognition.txt"] = recognition(record).encode()
    return result


def write_fixtures(directory: Path = DIRECTORY) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in fixtures(
        cohorts.load_training(), cohorts.load(), materials.load(), rewrites.load()
    ).items():
        (directory / name).write_bytes(content)


if __name__ == "__main__":
    write_fixtures()
    print("[requests] fixtures regenerated = True")
