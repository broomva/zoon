"""The registration states the implemented step and persistence contracts."""

from pathlib import Path

from tests.test_interaction_docs import kernel_view_for_docs

NOTE = Path(__file__).parent.parent / "docs" / "interaction" / "measure.md"


def test_registration_chat_contract_has_only_the_registered_rules() -> None:
    from zoon.interaction import chat

    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    section = text.split("### The chat arm\n\n", 1)[1].split("\n## Spending", 1)[0]
    assert section == (
        "It plays one seat at a time. Its request is the participant instructions, the\n"
        'line "You are Player n.", the history as rendered for `prompted`, and then:\n'
        "\n"
        "> How many of your E points do you contribute to the common pool in round r? "
        "Answer with one whole number of points and nothing else.\n"
        "\n"
        "Substitute that seat's endowment for E, seat for n and round for r.\n"
        "The instructions already end in `\\n`: append\n"
        "one `\\n`, then `You are Player n.`, then `\\n\\n`, then the history, then `\\n\\n`,\n"
        "then the question, with no final newline and no system instruction. The history\n"
        "joins rounds with `\\n\\n`; each round starts with `Round r\\n`.\n"
        "Use the same empty-history line as `prompted` when there is no earlier round.\n"
        "Safety settings are the pilot's second-family client's: `HARM_CATEGORY_HARASSMENT`,\n"
        "`HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT` and\n"
        "`HARM_CATEGORY_DANGEROUS_CONTENT` each have threshold `OFF`. The reply is the\n"
        "concatenated text parts of the first candidate, in their returned order.\n"
        "A candidate with text and no finish reason is read as its text.\n"
        "\n"
        "After trimming white space (space, tab, line feed and carriage return only), "
        "the reply must be a whole number from zero to E.\n"
        "Fixed here: accept only a non-empty run of ASCII digits, with no sign, decimal\n"
        "point or other text; leading zeroes are allowed. Retry unusable replies under\n"
        "the three-reply rule in Data; transport failures and arm-only outcomes follow\n"
        "that section.\n"
        "\n"
        "Every seat of every round of every simulated group is its own call. Identical\n"
        "requests are NOT reused across decisions. The transcript identifies the game,\n"
        "group, round, seat and attempt, and a resumed decision uses its recorded replies.\n"
        "The incomplete-line rule under Guard, repairs and persistence applies.\n"
        "Replay requires the recorded whole-transcript SHA-256 and exact line count, and\n"
        "uses one verified copy re-read from disk without fresh calls. Rendered "
        "decision and recall\n"
        "requests must match; attempts for each decision must be consecutive from one,\n"
        "with no attempt after an accepted reply and no more than three attempts.\n"
        "A later round of a group requires accepted replies for every seat of its earlier rounds.\n"
        "Changing text or token counts on any line fails replay. A game cannot be scored\n"
        "when the arm is unusable; its whole transcript can still pass replay.\n"
        "The arm's status is derived from the transcript and the cap, in file order: the\n"
        "first invalid or changed version, third unusable reply for one decision, or\n"
        "the chat cap reached under Spending. At the same entry, that is the order of precedence.\n"
        "The transcript cannot show a withdrawal or an operator ending; the run\n"
        "step records them, and that record governs later steps.\n"
        'Each recall question is labelled "judged", "not judged: version", or "not asked".\n'
        "The accepted points bypass recalibration and the band-to-points step.\n"
        "Run closed loop only, with the H3 simulation count. Score against `zoon` by the\n"
        "same H3 composite, difference and interval. It changes no verdict.\n"
    )
    normal = " ".join(text.split())
    assert (
        "The decision-model cap is also reached when recorded spending is above it; "
        "equality is allowed."
    ) in normal
    assert (
        "answered questions are kept and judged and the questions not yet "
        'answered are recorded "not asked".'
    ) in normal
    assert {c.name for c in chat.Cause} == {
        "REPLIES",
        "WITHDRAWN",
        "VERSION",
        "CAP",
        "REPLAY",
    }
    assert chat.points(" \t\n\r00036\r", 36) == 36
    assert all(chat.points(t, 36) is None for t in ("+1", "1.", "37", "\x0c1", "1\xa0"))
    assert chat.request(kernel_view_for_docs()).count("How many of your") == 1


def test_registration_recall_contract() -> None:
    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    passage = text.split("Recall questions with no recorded answer", 1)[1].split(
        "\n\nA non-finite probability", 1
    )[0]
    passage = "Recall questions with no recorded answer" + passage
    assert passage == (
        "Recall questions with no recorded answer are sent unless the model is withdrawn,\n"
        "its first reported version is invalid, its version has changed,\n"
        "or the chat cap is reached. An arm made unusable by the three-reply rule, or ended\n"
        "by the operator during `closed-loop`, is still asked. "
        "If the operator ends the arm\n"
        "during `recall`, answered questions are kept and judged and the questions not yet\n"
        'answered are recorded "not asked". `closed-loop` and `recall` complete with the\n'
        'chat status recorded and "not asked" for each recall question not sent, and later\n'
        "steps proceed. This changes no hypothesis verdict. The chat reply rule is under\n"
        'The chat arm. A paid reply with an invalid version is retained as "not judged: version",\n'
        "without judging or flagging it. Recall reports separately what stopped it asking."
    )


def test_registration_incomplete_line_contract() -> None:
    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    passage = text.split("Transcripts are append-only", 1)[1].split(
        " State is separate for this registration", 1
    )[0]
    assert "Transcripts are append-only" + passage == (
        "Transcripts are append-only except for the incomplete-line cut below, and bound\n"
        "to the full registered commit and mode; check that identity before replay.\n"
        "An incomplete final line, left by a write the system cut short, is not a record;\n"
        "every opening refuses it and changes no bytes. The operator cuts the file back\n"
        "to the byte after its last line feed, or to empty if it has none, changes nothing\n"
        "else, and repeats the command, and the decision is asked again. If the cut bytes\n"
        "held a reply, it was paid for and is not counted. The operator notes each cut and\n"
        "its size in bytes in the message of the commit that adds the result. This is the\n"
        "one change to a transcript the run allows; other damage is refused."
    )
    normal = " ".join(text.split())
    assert (
        "Apart from a declared repair, the incomplete-line cut and restoring a guarded output "
        "under Guard, repairs and persistence, this is the only operator decision the run allows, "
        "and it cannot be taken once `held-out` has started."
    ) in normal
    assert (
        "Do not delete or reset state or transcripts during the run, except for that cut; "
        "restoring a guarded output does not change state or transcripts."
    ) in normal


def test_available_step_modules_and_remaining_work_are_explicit() -> None:
    root = NOTE.parents[2]
    registration = (root / "interaction/registration.md").read_text()
    normal = " ".join(registration.split())
    assert (
        "Calibration, closed-loop and recall "
        "are implemented today; held-out and report remain requirements."
    ) in normal
    assert (
        "calibration, closed-loop and recall are implemented; held-out and report remain open work."
    ) in normal
    assert (
        "- The held-out step and report, including the path marked as added above," in registration
    )
    for name in ("steps.py", "simulation.py"):
        assert f"| `src/zoon/interaction/{name}` | Existing today |" in registration
        assert (root / "src/zoon/interaction" / name).is_file()


def test_registration_resume_order_retries_and_files() -> None:
    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    for passage in (
        (
            "For both models, a request has at most 4 attempts, with pauses of 1, 2 and 4\n"
            "seconds between failed attempts, as in `src/zoon/pilot/clients.py`, "
            "function `retry`.\n"
            "The analysis code must use the same counts and pauses."
        ),
        (
            "Record the version each response reports. The decision model must report exactly\n"
            "its name above. For chat, fix the version to the first response's "
            "reported version,\n"
            "which must begin with its model name, as in the pilot. Withdrawal means the\n"
            "service says the model does not exist on every bounded attempt at a request."
        ),
        (
            "Everything else interrupts a step and is resumed by repeating its command.\n"
            "Recorded answers are kept. Only missing answers are requested. A change that is\n"
            "not a permitted repair ends confirmatory use; what follows is exploratory and\n"
            "decides neither hypothesis. This is an operator rule; the guard cannot "
            "judge intent."
        ),
        (
            "2. `python -m zoon.interaction.run closed-loop`: simulate all usable arms. Write\n"
            "   each coded and prompted arm's logs and guarded digests when that arm "
            "finishes.\n"
            "   Write chat logs and their separate digests only on completion, after "
            "replay against\n"
            "   the verified transcript. Finish all simulation before any held-out "
            "group is read.\n"
            "3. `python -m zoon.interaction.run recall`: ask the chat model the "
            "recall questions.\n"
            "4. `python -m zoon.interaction.run held-out`: the first step to read "
            "held-out groups.\n"
            "   Only this step may call the second sealing step that produces them. Make the\n"
            "   open-loop calls and score H2 and H3. An unusable chat arm does not block it."
        ),
        (
            "A completed calling step is not repeated; `report` may be run again and must\n"
            "reproduce the same file. An interrupted step resumes with recorded\n"
            "answers and requests only those missing. A later calling step requires completed\n"
            "prerequisites; report records terminal outcomes without requiring later calling\n"
            "steps. The held-out prerequisite is completed simulation with recorded\n"
            "digests."
        ),
        (
            "Each model has its own cap, so exhausting the chat cap cannot end the run;\n"
            "a configuration refusal can still prevent completion as described above."
        ),
        ("| `interaction/registration.md` | Existing today | This registration |"),
        ("| `interaction/` | Existing today | All interaction materials |"),
        ("| `interaction/data-manifest.json` | Existing today | Source and training hashes |"),
        ("| `interaction/instructions/` | Existing today | Eight rendered instructions |"),
        ("| `interaction/instructions/sha256.json` | Existing today | Instruction digests |"),
        ("| `interaction/requests/` | Existing today | Rendered request fixtures |"),
        (
            "| `interaction/fitted.json` | Existing today | B and twin numbers, from "
            "the committed fitting procedure |"
        ),
        ("| `interaction/amounts.json` | Existing today | The training band-to-points table |"),
        (
            "| `src/zoon/interaction/score.py` | Existing today | Registered scores, "
            "intervals and verdicts |"
        ),
        (
            "| `src/zoon/interaction/registered.py` | Existing today | Committed "
            "material readers and regeneration |"
        ),
        (
            "| `src/zoon/interaction/power.py` | Existing today | Training power and "
            "coverage calculation |"
        ),
        (
            "| `docs/interaction/power.md` | Existing today | Power assumptions and "
            "one recorded run |"
        ),
        (
            "| `src/zoon/interaction/run.py` | Existing today | Calibration and step "
            "command dispatch |"
        ),
        (
            "| `src/zoon/interaction/steps.py` | Existing today | "
            "Closed-loop, recall and operator chat endings |"
        ),
        (
            "| `src/zoon/interaction/simulation.py` | Existing today | Committed-arm "
            "simulation and game log persistence |"
        ),
        (
            "| `src/zoon/interaction/guard.py` | Existing today | Run guards, "
            "repairs, state and lock |"
        ),
        (
            "| `src/zoon/interaction/decisions.py` | Existing today | Budgeted "
            "decision calls and append-only records |"
        ),
        (
            "| `src/zoon/interaction/chat.py` | Existing today | Chat requests, "
            "client, transcript, replay and recall |"
        ),
        (
            "| `docs/interaction/result.md` | Added by the analysis code | The "
            "report, outside the registered directory |"
        ),
        ("| `interaction/chat-requests/` | Existing today | Committed chat request fixtures |"),
        (
            "| `src/zoon/interaction/service_answers.py` | Existing today | "
            "Service-answer table regeneration |"
        ),
        (
            "| `interaction/service-answers.md` | Existing today | "
            "Generated service-answer outcomes |"
        ),
    ):
        assert passage in text


def test_registration_step_state_and_http_contracts() -> None:
    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    for passage in (
        (
            "For a resumable step, the flag is accepted only after a saved chat transport\n"
            "interruption of that step, or when that step already has a saved ending.\n"
            "Otherwise repeat the step without the flag; a later chat transport interruption\n"
            "makes the flag acceptable on the next attempt."
        ),
        (
            "A transport failure does not count toward the three attempts. An unusable\n"
            "reply is a recorded response whose text fails the reply rule, or is empty or\n"
            "blocked. Three unusable replies for one decision, the model's withdrawal, a\n"
            "version change, reaching the chat cap, a failed replay of a chat game, or a first\n"
            "reported version that does not begin with the model name, or an operator ending,\n"
            "make `chat` unusable with no scores.\n"
            "Chat game logs are written from the chat transcript when `closed-loop` completes,\n"
            "if the arm is still usable then, and are not read back by any run step. A later\n"
            "step that scores them first rebuilds them from the transcript, checked against\n"
            "`recall.json` `payload.chat.transcript.digest`, and requires the rebuilt bytes to\n"
            "match `closed-loop.json` `payload.chat_logs`. A failed rebuild makes `chat`\n"
            "unusable; that step records REPLAY in its own state if recall records no earlier\n"
            "cause, and the report prints both the recall record and the later step's record."
        ),
        (
            "Chat game logs are written from the chat transcript when `closed-loop` completes,\n"
            "if the arm is still usable then, and are not read back by any run step. A later\n"
            "step that scores them first rebuilds them from the transcript, checked against\n"
            "`recall.json` `payload.chat.transcript.digest`, and requires the rebuilt bytes to\n"
            "match `closed-loop.json` `payload.chat_logs`. A failed rebuild makes `chat`\n"
            "unusable; that step records REPLAY in its own state if recall records no earlier\n"
            "cause, and the report prints both the recall record and the later step's record."
        ),
        (
            "Each step keeps at most one ending event, written once. Recall carries earlier\n"
            "ending events forward unchanged and may add its own.\n"
            "The earliest cause, however detected, governs the arm's reported cause; once a\n"
            "completed step records a cause, later steps retain it. A "
            "withdrawal or an ending event in `recall` stops further\n"
            "recall questions. A withdrawal, an ending already in the current step, the cap,\n"
            "a version change or an invalid first version prevents further recall questions.\n"
            "Completed state files remain unchanged; the recall state exposes all ending events.\n"
            "An arm usable at the end of `closed-loop` can become unusable during `recall`.\n"
            "`Guard.recall_chat_record()` reads the status at the end of "
            "recall from `recall.json`\n"
            "fields `payload.chat.status` and `payload.chat.cause` and refuses unless recall is\n"
            "completed; an arm unusable there has no scores. A later failed rebuild is recorded\n"
            "in that later step's state as described above. The arm is unusable if completed\n"
            "recall says so or any later step records a failed rebuild; the recall record alone\n"
            "is not final once a later step has recorded one. If the run ends with no verdict\n"
            'before recall completes, the report prints the chat arm as "not run" and labels\n'
            "any earlier chat record with its step; it does not call that record final."
        ),
        (
            "If the run cannot record either model's withdrawal, a later attempt records a\n"
            "withdrawal only if that model is still withdrawn. A failed ending save is carried\n"
            "by the handler's one normal save; if that also fails, the last successful state\n"
            "stands. REPLAY is derived again on retry. A decision version "
            "change or cap is detected again\n"
            "from the decision transcript. Unreadable or incomplete chat transcripts prevent\n"
            "the handler's save; the next attempt refuses with the cut-back instruction for an\n"
            "incomplete line."
        ),
        (
            "A reply whose append fails before any byte is written is not recorded, is not\n"
            "counted toward the cap, and is requested again; actual spending can exceed\n"
            "recorded spending by such replies. No bound is claimed on\n"
            "paid but unrecorded spending across repeated storage failures."
        ),
        (
            "The client has five outcome classes: an accepted generating reply is returned\n"
            "for recording and uses up the flag; withdrawal counted on every bounded generating\n"
            "attempt makes chat unusable; a transport interruption leaves the request unrecorded\n"
            "and interrupts the step unless the eligible flag ends the arm; configuration and\n"
            "guard refusals leave the request unrecorded and interrupt the step "
            "without ending the arm."
        ),
        (
            "The generated [service-answer table](service-answers.md) records which answers fall\n"
            "in each class, their request counts, and whether a persistent answer leaves the flag\n"
            "acceptable on the next attempt."
        ),
        (
            "For a JSON-object generating answer entering response processing, the secret check\n"
            "runs first on all readable text parts joined together: a secret refuses the reply\n"
            "even if processing would otherwise fail, with no record or use of the flag."
        ),
    ):
        assert passage in text
