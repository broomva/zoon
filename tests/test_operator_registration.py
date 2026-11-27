"""Exact registered wording for implemented operator and answer-class behavior."""

from pathlib import Path

import pytest

PINNED = (
    ("The `chat` arm decides nothing."),
    (
        "Three unusable replies for one decision, the model's withdrawal, a\nver"
        "sion change, reaching the chat cap, a failed replay of a chat game, or"
        " a first\nreported version that does not begin with the model name, or "
        "an operator ending,\nmake `chat` unusable with no scores."
    ),
    (
        "A blank reason is refused before any work.\n"
        "A reason for `--end-chat` or `--repair` that contains the project identifier is\n"
        "refused before any work; the run does not otherwise inspect the operator's text.\n"
        "A chat request failure is a transport interrupti"
        "on, a configuration refusal\n(including an unavailable credential)"
        ", or a secret refusal of a reply from the\nmetadata check or a gen"
        "erating request, leaving no recorded reply and no counted withdra"
        "wal;\nthe flag applies to all three classes.\nFor an unfinished clo"
        "sed-loop or recall step, the flag is accepted only after\nthat ste"
        "p saved an interruption by a chat request failure, or an ending o"
        "f that\nstep is already saved; otherwise repeat the step without t"
        "he flag.\nIt ends the arm only if a chat request fails in this att"
        "empt before any generating\nrequest returns a reply; a returned re"
        "ply uses up the flag before transcript append,\nwhile replies reus"
        "ed from the transcript do not use it up.\nDecision-model and stora"
        "ge failures, three unusable replies, the cap, a version\nchange an"
        "d withdrawal never create an operator ending.\n\nThe run cannot che"
        "ck the operator's reason or distinguish a service failure from\nan"
        " operator-side fault such as a broken network or a rejected or mi"
        "ssing credential.\nThe ending records OPERATOR, the class, stage, "
        "last HTTP status, reason, step,\nthis step's count of recorded cha"
        "t request failures including this one, and answered\nchat decision"
        "s (and answered recall questions in recall); it removes chat scor"
        "es\nand changes no verdict.\nThe last status is the final HTTP stat"
        "us of the last attempt after redirects,\nor null if that attempt r"
        "eceived no final HTTP answer.\nAn operator ending completes this s"
        'tep; recall retains and judges its answers\nand marks the rest "no'
        't asked".\nOn completion the command prints the chat status and en'
        'ding records, and prints\n"flag had no effect" once if this flagge'
        "d attempt has no ending.\nIf a flagged attempt raises after a gene"
        'rating request returned in that attempt,\nit prints "flag used up:'
        ' a chat request succeeded in this attempt" before propagating the'
        " failure.\nAn ending already saved in this unfinished step is reta"
        'ined without new chat requests\nand printed as "ending already rec'
        'orded"; a completed or terminal step is still refused.\n'
    ),
    (
        "A reply whose text holds a secret is never recorded; a secret is the project\n"
        "identifier or a bearer this process has sent.\nThe generated [se"
        "rvice-answer table](service-answers.md) records the outcome class,\nreq"
        "uest count and eligibility of the flag on the next attempt for each an"
        "swer.\n"
    ),
    (
        "| `src/zoon/interaction/steps.py` | Existing today | Closed-loop, reca"
        "ll and operator chat endings |"
    ),
    (
        "| `interaction/chat-requests/` | Existing today | Committed chat reque"
        "st fixtures |\n| `src/zoon/interaction/service_answers.py` | Existing t"
        "oday | Service-answer table regeneration |\n| `interaction/service-answ"
        "ers.md` | Existing today | Generated service-answer outcomes |"
    ),
    (
        "The run module dispatches calibration, closed-loop and recall today. T"
        "he existing spending script\n`src/zoon/interaction/budget.py` produces "
        "both bounds. The second sealing step,\nscoring and intervals, run guard"
        "s, step state, calibration, closed-loop and recall are implemented; he"
        "ld-out and report remain\nopen work. A material-regeneration test must "
        "compare registered instructions,\nfixtures and the amount table with th"
        "eir committed bytes; the fit test uses the\nprobability tolerance above"
        "."
    ),
    (
        "Merge the registration, analysis code and registered materials, then c"
        "reate the\ntag as specified above. Calibration, closed-loop and recall "
        "are implemented today; held-out and report remain requirements.\nRun th"
        "em in this order:"
    ),
    (
        "The transcript cannot show a withdrawal or an operator ending; the run"
        "\nstep records them, and that record governs later steps."
    ),
)


@pytest.mark.parametrize("passage", PINNED)
def test_operator_registration_contract(passage: str) -> None:
    text = (Path(__file__).parents[1] / "interaction/registration.md").read_text()
    assert passage in text
