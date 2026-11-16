"""A deterministic spending bound built from training records and published materials only.

Decision characters are the complete canonical JSON fixture, including answer descriptions.
The held-out bound uses the longest trial renderings over all dilemma/answer/outcome cells,
with the longest allowed rating text. Repeated dilemmas in that artificial history are
permitted to make an upper bound; no real person's record is used for it.
Reversal calls are evaluative only, for every committed reversible task or outside rewrite.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from zoon.pilot import cohorts, fetch, materials, prices, prompts, rewrites

Family = Literal["decision", "second"]
FAMILIES: tuple[Family, ...] = ("decision", "second")
STEPS = (
    "outside judge",
    "task and strict judges",
    "reversal",
    "recognition",
    "calibration",
    "held-out bound",
)
HELD_OUT_PARTICIPANTS = 680  # fixed by the registration; never counted from raw data here
EXAMPLES = 2
CEILING = Decimal("120")


@dataclass(frozen=True)
class Cost:
    family: Family
    step: str
    calls: int
    input_tokens: int
    output_tokens: int

    @property
    def dollars(self) -> Decimal:
        price = prices.DECISION if self.family == "decision" else prices.SECOND
        return price.dollars(self.input_tokens, self.output_tokens)


def input_tokens(request: prompts.Decision | prompts.Second | str, family: Family) -> int:
    if isinstance(request, prompts.Decision) and family == "second":
        request = prompts.second_request(request)
    text = (
        request.system + request.text
        if isinstance(request, prompts.Second)
        else prompts.decision_bytes(request).decode()
        if isinstance(request, prompts.Decision)
        else request
    )
    return (len(text) + 2) // 3 + (400 if family == "decision" else 0)


def longest_trials(tasks: dict[str, materials.Task]) -> list[str]:
    renderings = [
        prompts.trial(cohorts.Trial(name, answer, success, -100), tasks)
        for name in fetch.DILEMMAS
        for answer in ("Yes", "No")
        for success in (False, True)
    ]
    return sorted(renderings, key=len, reverse=True)[: len(fetch.DILEMMAS) - 1]


def held_out_request(example_text: str, tasks: dict[str, materials.Task]) -> prompts.Decision:
    state = prompts.prompted_state(
        example_text,
        "\n\n".join(longest_trials(tasks)),
        max((t["text"] for t in tasks.values()), key=len),
    )
    return prompts.Decision(state, prompts.PREDICT, prompts.declared(("Yes", "No")))


def estimate(
    people: Sequence[cohorts.Participant],
    document: cohorts.Cohorts,
    tasks: dict[str, materials.Task],
    records: Sequence[rewrites.Record],
) -> list[Cost]:
    """Count every planned call without making a call or accessing held-out records."""
    tallies = {(family, step): [0, 0, 0] for family in FAMILIES for step in STEPS}

    def add(
        family: Family,
        step: str,
        request: prompts.Decision | prompts.Second | str,
        *,
        calls: int = 1,
        output: int = 1,
    ) -> None:
        tally = tallies[family, step]
        tally[0] += calls
        tally[1] += calls * input_tokens(request, family)
        tally[2] += calls * output if family == "second" else 0

    def both(step: str, request: prompts.Decision, *, calls: int = 1) -> None:
        add("decision", step, request, calls=calls)
        add("second", step, prompts.second_request(request), calls=calls)

    for record in records:
        if record["set"] == "outside":
            both("outside judge", prompts.evaluative(record))
        else:
            both("task and strict judges", prompts.evaluative(record))
            add("second", "recognition", prompts.recognition(record), output=40)
        for i in range(len(prompts.STRICT)):
            both("task and strict judges", prompts.strict(record, i))
        if "reversed" in record:
            both("reversal", prompts.evaluative(record, reverse=True))

    lookup = {(p.experiment, p.participant): p for p in people}
    for rotation in document["rotations"]:
        example_text = prompts.examples(rotation, people, tasks)
        for identity in document["calibration"]:
            person = lookup[identity["experiment"], identity["participant"]]
            for i, trial in enumerate(person.trials, 1):
                if trial.dilemma not in rotation["held_out"]:
                    both("calibration", prompts.prompted(person, i, example_text, tasks))
        # Each dilemma appears in two rotations. Each rotation has two held-out trials
        # per participant; the histories are bounded by materials, not by reading records.
        both(
            "held-out bound",
            held_out_request(example_text, tasks),
            calls=HELD_OUT_PARTICIPANTS * len(rotation["held_out"]),
        )
    return [Cost(family, step, *tallies[family, step]) for family in FAMILIES for step in STEPS]


def total(rows: Sequence[Cost], family: Family | None = None) -> Decimal:
    return sum((r.dollars for r in rows if family is None or r.family == family), Decimal(0))


def run() -> None:
    document = cohorts.load()
    people = cohorts.load_training()
    if cohorts.serialise(cohorts.generate(people)) != cohorts.FILE.read_bytes():
        raise ValueError("committed cohorts differ from the registered draw")
    rows = estimate(people, document, materials.load(), rewrites.load())
    print("zoon pilot: spending bound")
    print(f"[budget] examples per rotation = {EXAMPLES}")
    print(f"[budget] calibration participants = {len(document['calibration'])}")
    print(f"[budget] held-out participants, fixed by registration = {HELD_OUT_PARTICIPANTS}")
    print("[budget] input estimate = ceil(characters / 3); decision adds 400 per call")
    print("[budget] decision characters = canonical request JSON, including descriptions")
    print(
        "[budget] second characters = system instruction plus user text; recognition is user text"
    )
    print("[budget] reversal questions = evaluative only, all committed reversed rewrites")
    for family, price in (("decision", prices.DECISION), ("second", prices.SECOND)):
        print(f"[{family} price] model = {price.model} | read = {price.read} | page = {price.page}")
        print(
            f"[{family} price] USD per million input = {price.input_per_million} | "
            f"USD per million output = {price.output_per_million}"
        )
        own = [r for r in rows if r.family == family]
        for row in own:
            print(
                f"[{family} {row.step}] calls = {row.calls} | input tokens = {row.input_tokens} | "
                f"output tokens = {row.output_tokens} | USD = {row.dollars:.6f}"
            )
        print(
            f"[{family} total] calls = {sum(r.calls for r in own)} | "
            f"input tokens = {sum(r.input_tokens for r in own)} | "
            f"output tokens = {sum(r.output_tokens for r in own)} | USD = {total(own):.6f}"
        )
    print(f"[budget] total USD = {total(rows):.6f}")
    print(f"[budget] registration ceiling USD = {CEILING}")
    print(f"[budget] total at most {CEILING} USD = {total(rows) <= CEILING}")
