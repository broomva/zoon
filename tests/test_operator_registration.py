"""Exact registered wording for implemented operator and answer-class behavior."""

from pathlib import Path

import pytest

PINNED = (
    ("The `chat` arm decides nothing."),
    (
        "Three unusable replies for one decision, the model's withdrawal, a\nversion "
        "change, reaching the chat cap, a failed replay of a chat game, or a first\nr"
        "eported version that does not begin with the model name, or an operator end"
        "ing,\nmake `chat` unusable with no scores."
    ),
    (
        "A blank reason is refused before any work.\nA reason for `--end-chat` or `--"
        "repair` that contains the project identifier is\nrefused before any work; th"
        "e run does not otherwise inspect the operator's text.\nThe metadata check is"
        " the model metadata HTTP request made before generation.\nA chat request fai"
        "lure is a transport interruption, a configuration refusal\n(including an una"
        "vailable credential), or a secret refusal of a reply from the\nmetadata chec"
        "k or a generating request, leaving no recorded reply and no counted withdra"
        "wal;\nthe flag applies to all three classes.\nFor an unfinished closed-loop o"
        "r recall step, the flag is accepted only after\nthat step saved an interrupt"
        "ion by a chat request failure, or an ending of that\nstep is already saved; "
        "otherwise repeat the step without the flag.\nIt ends the arm only if a chat "
        "request fails in this attempt before any generating\nrequest returns a reply"
        "; a returned reply uses up the flag before transcript append,\nwhile replies"
        " reused from the transcript do not use it up.\nDecision-model and storage fa"
        "ilures, three unusable replies, the cap, a version\nchange and withdrawal ne"
        "ver create an operator ending.\n\nThe run cannot distinguish a service failur"
        "e from an operator-side fault such as\na broken network or a rejected or mis"
        "sing credential.\nThe ending records OPERATOR, the class, stage, last HTTP s"
        "tatus, reason, step,\nthis step's count of recorded chat request failures in"
        "cluding this one, and answered\nchat decisions (and answered recall question"
        "s in recall); it removes chat scores\nand changes no verdict.\nThe stage is n"
        "ull if failure precedes a request, and the last status is the final\nHTTP st"
        "atus of the last attempt after redirects, or null if that attempt received\n"
        "no final HTTP answer.\nAn operator ending completes this step; recall retain"
        's and judges its answers\nand marks the rest "not asked".\nOn completion the '
        'command prints the chat status and ending records, and prints\n"flag had no '
        'effect" once if this flagged attempt neither uses the flag nor retains\nan e'
        "nding saved before the attempt.\nIf a flagged attempt raises after a generat"
        'ing request returned in that attempt,\nit prints "flag used up: a chat reque'
        'st succeeded in this attempt" before propagating the failure.\nAn ending alr'
        "eady saved in this unfinished step is retained without new chat requests\nan"
        'd printed as "ending already recorded"; a completed or terminal step is sti'
        "ll refused.\n"
    ),
    (
        "A reply whose text holds a secret is never recorded; a secret is the projec"
        "t\nidentifier or a bearer this process has sent.\nThe generated [service-answ"
        "er table](service-answers.md) records the outcome class,\nrequest count and "
        "eligibility of the flag on the next attempt for each answer.\n"
    ),
    (
        "| `src/zoon/interaction/steps.py` | Existing today | Closed-loop, recall an"
        "d operator chat endings |"
    ),
    (
        "| `interaction/chat-requests/` | Existing today | Committed chat request fi"
        "xtures |\n| `src/zoon/interaction/service_answers.py` | Existing today | Ser"
        "vice-answer table regeneration |\n| `interaction/service-answers.md` | Exist"
        "ing today | Generated service-answer outcomes |"
    ),
    (
        "The run module dispatches calibration, closed-loop and recall today. The ex"
        "isting spending script\n`src/zoon/interaction/budget.py` produces both bound"
        "s. The second sealing step,\nscoring and intervals, run guards, step state, "
        "calibration, closed-loop and recall are implemented; held-out and report re"
        "main\nopen work. A material-regeneration test must compare registered instru"
        "ctions,\nfixtures and the amount table with their committed bytes; the fit t"
        "est uses the\nprobability tolerance above."
    ),
    (
        "Merge the registration, analysis code and registered materials, then create"
        " the\ntag as specified above. Calibration, closed-loop and recall are implem"
        "ented today; held-out and report remain requirements.\nRun them in this orde"
        "r:"
    ),
    (
        "The transcript cannot show a withdrawal or an operator ending; the run\nstep"
        " records them, and that record governs later steps."
    ),
    ("completed step records a cause, later steps retain it."),
    (
        "Recall questions with no recorded answer are sent unless the model is withd"
        "rawn,\nan ending is already saved in recall, its first reported version is i"
        "nvalid,\nits version has changed, or the chat cap is reached."
    ),
    ("A completed step that sends model requests is not repeated;"),
    (
        "A guarded output that is missing, is not a regular file (including a direct"
        "ory,\nnamed pipe, socket or device), has a non-directory parent, cannot be r"
        "ead for lack\nof permission, or differs from its recorded digest refuses"
    ),
    ("The metadata check is the model metadata HTTP request made before generation."),
)


@pytest.mark.parametrize("passage", PINNED)
def test_operator_registration_contract(passage: str) -> None:
    text = (Path(__file__).parents[1] / "interaction/registration.md").read_text()
    assert passage in text
