"""The interaction notes quote only numbers that their pasted output contains."""

import ast
import inspect
import re
from decimal import Decimal
from pathlib import Path

import pytest

from tests.test_interaction_measure import played_groups
from tests.test_interaction_training import require_training
from zoon.interaction import kernel, measure

NOTE = Path(__file__).parent.parent / "docs" / "interaction" / "measure.md"
# Lines the note's argument rests on. A run must print each one.
REPORTED = (
    "[open loop] own history adds, entered first",
    "[open loop] own history adds, entered second",
    "[open loop] others' history adds, entered first",
    "[open loop] others' history adds, entered second",
    "[open loop, round 1 only] per game, no history",
    "[open loop, round 1 only] per game, own and others",
    "[members apart, round 1] closer than chance by",
    "[members apart, rounds 16 to 20] closer than chance by",
    "[power] closed loop, all three",
    "[power] closed loop, members apart",
)
NUMBER = re.compile(r"[+-]?\d+\.\d{3,}")


def _split() -> tuple[str, str]:
    """Return the prose and the pasted output, which is the last code block."""
    text = NOTE.read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    return text[:start], text[start : text.rindex("\n```")]


def test_measure_paste_has_heading_and_power_label() -> None:
    _, paste = _split()
    assert paste.startswith("zoon interaction: what the others' history adds")
    assert "[power] closed loop" in paste


def test_every_quoted_number_is_in_the_pasted_output() -> None:
    "A figure with three or more decimals in the prose must have been printed by the run."
    prose, paste = _split()
    printed = {number.lstrip("+-") for number in NUMBER.findall(paste)}
    quoted = {number.lstrip("+-") for number in NUMBER.findall(prose)}
    assert len(quoted) > 10, "the note quotes its figures"
    assert sorted(quoted - printed) == []


def test_the_pasted_output_reports_what_the_note_relies_on() -> None:
    """Check the required labels in the paste."""
    _, paste = _split()
    for label in REPORTED:
        assert label in paste, label


def test_a_live_run_prints_what_the_note_relies_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same lines are printed by the program itself, run here on synthetic groups, and
    the pasted output has a line for every label a run prints."""
    groups = played_groups(others=1.0, count=6, seed=51)
    monkeypatch.setattr(measure.data, "load", lambda: groups)
    monkeypatch.setattr(measure.data, "held_out_counts", lambda: {"2-FE": 4, "4-FE": 3})
    measure.main()
    printed = capsys.readouterr().out
    for label in REPORTED:
        assert label in printed, label
    _, paste = _split()

    def labels(text: str) -> list[str]:
        return [line.split(" = ")[0] for line in text.splitlines()]

    assert labels(printed) == labels(paste)


def test_notes_exclude_phrases_asserting_influence() -> None:
    """Reject listed phrases in the three notes."""
    notes = NOTE.parent
    text = " ".join(
        (notes / name).read_text() for name in ("measure.md", "task-selection.md", "candidates.md")
    )
    text = " ".join(text.split()).lower()
    for phrase in (
        "influenced one another",
        "it rules out",
        "is ruled out",
        "separates influence",
        "this is influence",
        "as strangers",
        "moved toward the people",
    ):
        assert phrase not in text, phrase


def candidate_note() -> tuple[str, str]:
    text = (NOTE.parent / "candidates.md").read_text()
    start = text.rindex("\n```\n", 0, text.rindex("\n```")) + len("\n```\n")
    return text[:start], text[start : text.rindex("\n```")]


def test_candidate_figures_are_printed_and_every_form_is_reported() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    decimals = NUMBER
    printed = {n.lstrip("+-") for n in decimals.findall(paste)}
    assert {n.lstrip("+-") for n in decimals.findall(prose)} <= printed
    assert paste.startswith("zoon interaction: candidates on training groups\n")
    lines = dict(line.split(" = ", 1) for line in paste.splitlines()[1:])
    for c in person.CANDIDATES:
        assert int(lines[f"[count] {c.name}"]) == c.count
        numbers = {
            label: value
            for label, value in lines.items()
            if label.startswith(f"[numbers, {c.name}]")
        }
        assert len(numbers) == c.count
        for trace in c.traces:
            assert 0 <= float(lines[f"[numbers, {c.name}] trace fraction {trace}"]) <= 1
        assert f"[open loop] {c.name}" in lines
        for summary in compare.METRICS:
            assert f"[closed loop, {summary}] {c.name}" in lines
        for summary in compare.SUMMARIES:
            assert 0 <= float(lines[f"[coverage, {summary}] {c.name}"]) <= 1
    for label in ("candidates", "twins"):
        tied = lines[f"[selection, {label}] tied at step 2"].split(", ")
        winner = lines[f"[selection, {label}] winner"]
        reference = lines[f"[selection, {label}] highest"]
        assert reference in tied and winner in tied
        assert int(lines[f"[count] {winner}"]) == min(
            int(lines[f"[count] {name}"]) for name in tied
        )
    for comparison in ("selected twin", "own twin"):
        assert f"[gain, {comparison}, open loop] candidate minus twin" in lines
        for summary in compare.METRICS:
            assert f"[gain, {comparison}, closed loop, {summary}] twin minus candidate" in lines
        for summary in ("H2", *compare.METRICS):
            assert f"[power, {comparison}, {summary}] training effect" in lines
            assert f"[power, {comparison}, {summary}] half effect" in lines
    for arm in ("candidate", "twin"):
        assert f"[game left out, {arm}] loss" in lines


def test_candidate_outcomes_and_tables_match_the_paste() -> None:
    from zoon.interaction import compare, person

    prose, paste = candidate_note()
    lines = dict(line.split(" = ", 1) for line in paste.splitlines()[1:])
    winner = lines["[selection, candidates] winner"]
    twin = lines["[selection, twins] winner"]
    normal = " ".join(prose.split())
    training = re.search(r"on (\d+) training groups", normal)
    held = re.search(r"The (\d+) held-out groups", normal)
    first_cut = re.search(r"The first cut is now fixed at (\d+)\.", normal)
    assert training and held and first_cut
    assert training[1] == lines["[data] training groups"]
    assert held[1] == lines["[data] held-out groups"]
    assert first_cut[1] == lines["[fit] first cut"]
    selection = re.search(
        r"Among candidates, (.*?) tie at step 2; (.*?) has the highest score; "
        r"the rule chooses (.*?) because it has the fewest fitted numbers among those tied\.",
        normal,
    )
    assert selection
    assert selection.groups() == (
        lines["[selection, candidates] tied at step 2"],
        lines["[selection, candidates] highest"],
        winner,
    )
    tied = lines["[selection, candidates] tied at step 2"].split(", ")
    assert int(lines[f"[count] {winner}"]) < min(
        int(lines[f"[count] {name}"]) for name in tied if name != winner
    )
    twins = re.search(
        r"Among twins, (.*?) has the highest score, is alone at step 2, "
        r"and is chosen by the rule\.",
        normal,
    )
    assert twins
    assert (
        twins[1]
        == lines["[selection, twins] highest"]
        == lines["[selection, twins] tied at step 2"]
        == twin
    )

    tables = [
        [[value.strip() for value in row.strip("|").split("|")] for row in block.splitlines()]
        for block in re.findall(r"(?:^\|[^\n]*\n)+", prose, re.M)
    ]
    expected: list[list[list[str]]] = []
    expected.append(
        [
            [
                c.name,
                lines[f"[count] {c.name}"],
                lines[f"[open loop] {c.name}"],
                lines[f"[closed loop, composite] {c.name}"],
            ]
            for c in person.CANDIDATES
        ]
    )
    gains: list[list[str]] = []
    powers: list[list[str]] = []
    for metric in ("H2", *compare.METRICS):
        values: list[str] = []
        for opponent in ("selected twin", "own twin"):
            label = (
                f"[gain, {opponent}, open loop] candidate minus twin"
                if metric == "H2"
                else f"[gain, {opponent}, closed loop, {metric}] twin minus candidate"
            )
            values.append(lines[label])
        gains.append([metric, *values])
        powers.append(
            [
                metric,
                *[
                    lines[f"[power, {opponent}, {metric}] {effect}"]
                    for opponent in ("selected twin", "own twin")
                    for effect in ("training effect", "half effect")
                ],
            ]
        )
    expected.append(gains)
    expected.append(
        [
            [
                summary,
                *[
                    lines[f"[coverage, {summary}] {name}"]
                    for name in (winner, twin, winner + " twin")
                ],
            ]
            for summary in compare.SUMMARIES
        ]
    )
    expected.append(powers)
    expected.append(
        [
            [arm, lines[f"[game left out, {arm}] score"], lines[f"[game left out, {arm}] loss"]]
            for arm in ("candidate", "twin")
        ]
    )
    expected.append(
        [
            [
                cell,
                lines[f"[game left out, candidate, {cell}] score"],
                lines[f"[game left out, twin, {cell}] score"],
            ]
            for cell in sorted(compare.GAMES)
        ]
    )
    assert [table[2:] for table in tables] == expected
    assert [table[0] for table in tables] == [
        ["Form", "Fitted numbers", "Open-loop nats per decision", "Closed-loop composite"],
        ["Measure", "Against selected twin", "Against own twin"],
        ["Summary", "Selected candidate", "Selected twin", "Own twin"],
        [
            "Measure",
            "Selected twin, training effect",
            "Selected twin, half effect",
            "Own twin, training effect",
            "Own twin, half effect",
        ],
        ["Arm", "Nats per decision", "Loss from leaving the game out"],
        ["Game", "Candidate", "Twin"],
    ]
    reproduced = re.search(
        r"The largest absolute fitted-number changes over reversed groups and a nudged start "
        r"are ([\d.]+) for (.*?) and ([\d.]+) for (.*?)\.",
        normal,
    )
    assert reproduced
    assert reproduced.groups() == (
        lines[f"[reproducibility, {winner}] largest absolute change"],
        winner,
        lines[f"[reproducibility, {twin}] largest absolute change"],
        twin,
    )


def test_candidate_pasted_settings_match_the_current_code() -> None:
    from zoon.interaction import compare

    _, paste = candidate_note()
    settings = [
        line
        for line in paste.splitlines()
        if line.startswith(("[fit]", "[simulation]", "[interval]"))
    ]
    assert settings == [entry.render() for entry in compare.settings()]


def test_candidate_pasted_amounts_match_the_training_file() -> None:
    from zoon.interaction import compare, data

    require_training()
    _, paste = candidate_note()
    amounts = [line for line in paste.splitlines() if line.startswith("[amounts,")]
    assert amounts == [entry.render() for entry in compare.amount_entries(data.load())]


def prompted_note() -> tuple[str, str]:
    text = (NOTE.parent / "prompted.md").read_text()
    prefix, paste = text.rsplit("\n```\n", 2)[:2]
    return prefix, paste.rstrip("`\n")


def test_prompted_table_and_quoted_figures_match_the_paste() -> None:
    prose, paste = prompted_note()
    rows: list[str] = []
    for line in paste.splitlines():
        match = re.fullmatch(
            r"\[(open loop|recalibration|disguise|closed loop|total|"
            r"chat closed loop|chat recall|chat total)\] calls = (\d+) "
            r"\| input tokens <= (\d+) \| output tokens <= (\d+) \| USD <= (\S+)",
            line,
        )
        if match:
            rows.append("| " + " | ".join(match.groups()) + " |")
    quoted = [line for line in prose.splitlines() if re.match(r"\| (?!Call|---)", line)]
    assert len(rows) == 8
    assert quoted == rows
    # Attribution identifies the source and its licence; these are not measured figures.
    # It is exempt from the digit rule, so it is pinned whole.
    citation = (
        "The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),\n"
        "e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.107"
        "3/pnas.2525760123),\n"
        "supplementary appendix, sections 5B and 5D. The other games use substituted "
        "parameters.\n\n"
    )
    assert prose.count(citation) == 1
    prose = prose.replace(citation, "")
    table = [
        "| Call | Calls | Input tokens at most | Output tokens at most | USD at most |",
        "|---|---:|---:|---:|---:|",
        *quoted,
    ]
    lines = prose.splitlines()
    first = lines.index(table[0])
    # The header says what the figures are, so it is checked with them, in place.
    assert lines[first : first + len(table)] == table
    prose = "\n".join(lines[:first] + lines[first + len(table) :])
    assert not re.search(r"\d", prose)
    assert paste.startswith("zoon interaction: prompted spending bound\n")
    assert "[budget] total at most proposed ceiling = " in paste


def test_prompted_pasted_settings_match_current_objects() -> None:
    from zoon.interaction import budget

    _, paste = prompted_note()
    assert [
        line for line in paste.splitlines() if line.startswith("[settings]")
    ] == budget.settings()


def test_prompted_full_paste_matches_training_only_bound() -> None:
    from zoon.interaction import budget, data

    require_training()
    _, paste = prompted_note()
    assert paste == "\n".join(budget.report(data.load(), data.held_out_counts()))


def test_registration_constants_match_current_code() -> None:
    from zoon.interaction import person, prompted, split
    from zoon.pilot import clients

    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    constants = {
        "prompted.PROMPTED_FLOOR": prompted.PROMPTED_FLOOR,
        "SUBSAMPLE_COUNT": prompted.SUBSAMPLE_COUNT,
        "SUBSAMPLE_SALT": prompted.SUBSAMPLE_SALT,
        "PENALTY": prompted.PENALTY,
        "GRADIENT_LIMIT": prompted.GRADIENT_LIMIT,
        "POLISH_STEPS": prompted.POLISH_STEPS,
        "person.FLOOR": person.FLOOR,
        "person.RIDGE": person.RIDGE,
        "split.SALT": split.SALT,
        "LOG_GAP_LIMIT": person.LOG_GAP_LIMIT,
        "MAX_ITERATIONS": person.MAX_ITERATIONS,
        "FIT_METHOD": prompted.FIT_METHOD,
        "BETA_LOWER_BOUND": prompted.BETA_LOWER_BOUND,
        "prompted.DISGUISE": prompted.DISGUISE,
    }
    for name, expected in constants.items():
        quoted = re.findall(rf"`{re.escape(name)} = ([^`]+)`", text)
        assert quoted, name
        assert all(ast.literal_eval(value) == expected for value in quoted), name

    seeds = re.findall(r"default_rng\((\d+)\)|Fixed here: seed (\d+);", text)
    assert len(seeds) == 3
    assert all(int(first or second) == measure.SEED for first, second in seeds)

    normal = " ".join(text.split())
    retry = re.search(
        r"For both models, a request has at most (\d+) attempts, "
        r"with pauses of ([\d, ]+) and (\d+) seconds between failed attempts",
        normal,
    )
    assert retry
    attempts = int(retry[1])
    pauses = [int(value.strip()) for value in retry[2].split(",")] + [int(retry[3])]
    assert attempts == inspect.signature(clients.retry).parameters["attempts"].default
    observed: list[float] = []
    calls = 0

    def fail() -> clients.Response:
        nonlocal calls
        calls += 1
        raise clients.TransientError

    with pytest.raises(clients.StepFailure):
        clients.retry(fail, pause=observed.append)
    assert calls == attempts
    assert observed == pauses


def test_registration_chat_settings_rules_and_figures_match_code() -> None:
    from zoon.interaction import budget, chat, chat_prices, data

    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    normal = " ".join(text.split())
    for name, value in {
        "chat.MAX_OUTPUT_TOKENS": chat.MAX_OUTPUT_TOKENS,
        "budget.DECISION_CAP_MULTIPLE": budget.DECISION_CAP_MULTIPLE,
        "chat_prices.LOW_THROUGH": chat_prices.LOW_THROUGH,
    }.items():
        quoted = re.findall(rf"`{re.escape(name)} = ([^`]+)`", text)
        assert quoted and all(ast.literal_eval(q) == value for q in quoted)
    assert f"Thinking level {chat.THINKING_LEVEL}; temperature {chat.TEMPERATURE};" in normal
    assert f"at most {chat.MAX_OUTPUT_TOKENS} output tokens" in normal
    assert f"with at most {chat.MAX_OUTPUT_TOKENS} output tokens per call" in normal
    words = {3: "Three", 4: "Four"}
    assert f"{words[chat.REPLY_ATTEMPTS]} unusable replies for one decision" in normal
    assert "leading zeroes are allowed" in normal
    assert "space, tab, line feed and carriage return only" in normal
    retry = re.search(
        r"For both models, a request has at most (\d+) attempts, "
        r"with pauses of (\d+), (\d+) and (\d+) seconds",
        normal,
    )
    assert retry and int(retry[1]) == chat.TRANSPORT_ATTEMPTS
    assert tuple(map(float, retry.groups()[1:])) == chat.TRANSPORT_PAUSES
    assert f"each have threshold `{chat.SAFETY_SETTINGS[0]['threshold']}`" in normal
    keys = re.search(
        r'Flag a reply containing the run "([^"]+)", "([^"]+)" or "([^"]+)", '
        r'or the adjacent runs "([^"]+)"',
        normal,
    )
    assert keys and keys.groups() == chat.RECALL_KEYS
    question = re.search(r"^> How many of your .*$", text, re.M)
    assert question
    rendered = question[0][2:].replace(" E ", " 36 ").replace("round r?", "round 1?")
    assert chat.request(kernel_view_for_docs()).endswith(rendered)
    separators = re.search(
        r"one `(.*?)`, then `You are Player n\.`, then `(.*?)`, "
        r"then the history, then `(.*?)`, then the question",
        normal,
    )
    assert separators
    assert tuple(ast.literal_eval('"' + part + '"') for part in separators.groups()) == (
        chat.INSTRUCTION_SEPARATOR,
        chat.SECTION_SEPARATOR,
        chat.SECTION_SEPARATOR,
    )
    rounds = re.search(r"history joins rounds with `(.*?)`", normal)
    assert rounds and ast.literal_eval('"' + rounds[1] + '"') == chat.ROUND_SEPARATOR
    assert f"on {chat_prices.READ} from the provider's" in normal
    assert f"{chat_prices.HIGH_FROM}; the bound uses the higher price." in normal
    multiple = {4: "four", 3: "three"}
    assert (
        f"The decision-model cap is {multiple[budget.DECISION_CAP_MULTIPLE]} times its bound"
        in normal
    )
    require_training()
    rows = budget.estimate(data.load(), data.held_out_counts())
    assert f"**Decision-model cap in US dollars: {budget.decision_cap(rows)}.**" in normal
    assert f"recall included: {budget.chat_cap()}.**" in normal
    bound = sum((r.dollars for r in budget.chat_estimate()), start=Decimal())
    assert f"**Chat-model bound in US dollars: {budget.dollars(bound)}.**" in normal


def kernel_view_for_docs() -> "kernel.View":
    from zoon.interaction import kernel
    from zoon.interaction.games import GAMES

    return kernel.View(GAMES["2-AI"], 0, ())


def test_registered_format_check_counts_equal_committed_record() -> None:
    import hashlib

    from zoon.interaction import chat

    raw = (NOTE.parent / "format-check-chat.txt").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "daf600cdcd5486ff7cc0fe381838bd074f9be1f7151c4833755dc50b00660d1d"
    )
    lines = raw.decode().splitlines()
    normal = " ".join((NOTE.parents[2] / "interaction" / "registration.md").read_text().split())
    paragraph = re.search(
        r"was run once with at most 256 output tokens, the limit first chosen: (\d+) calls, .*?"
        r"All (\d+) replies were usable whole numbers\. (\w+) reported thinking tokens, "
        r"from (\d+) to (\d+)",
        normal,
    )
    assert paragraph
    assert int(paragraph[1]) == len(lines) == 12
    assert int(paragraph[2]) == sum("usable = True" in line for line in lines) == 12
    thoughts = [int(re.search(r"thinking tokens = (\d+)", line)[1]) for line in lines]  # type: ignore[index]
    nonzero = [value for value in thoughts if value]
    assert paragraph[3] == {3: "Three"}[len(nonzero)]
    assert (int(paragraph[4]), int(paragraph[5])) == (min(nonzero), max(nonzero))
    assert len(lines) == len(chat.smoke(chat.StubClient()))
    shape_raw = (NOTE.parent / "reply-shape.txt").read_bytes()
    assert (
        hashlib.sha256(shape_raw).hexdigest()
        == "3655492d7bfeba544e188e1ff34bc1522510f5c294af128ff5c0d6bf74889fd5"
    )
    shape_count = len(re.findall(r"^reply (?:with|without) thinking:$", shape_raw.decode(), re.M))
    sentence = re.search(
        "On (\\d{4}-\\d{2}-\\d{2}) (\\w+) further calls at that limit were made to record "
        r"which usage fields a reply carries; the record is "
        r"\[`docs/interaction/reply-shape.txt`\]\(../docs/interaction/reply-shape.txt\)\.",
        normal,
    )
    assert sentence and sentence[1] == "2026-11-21"
    assert sentence[2] == {2: "two"}[shape_count]
    assert "others were printed as zero" in normal

    assert (
        "[`docs/interaction/format-check-chat.txt`](../docs/interaction/format-check-chat.txt)"
        in normal
    )


def test_output_limit_paragraph_counts_equal_committed_record() -> None:
    import hashlib

    from zoon.interaction import chat

    raw = (NOTE.parent / "output-limit.txt").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "448de8ed4571fe337008d4c267c225be77c8ff7b1f1706aee80389a34dac9eed"
    )
    sections: dict[int, list[str]] = {}
    for block in raw.decode().split("\n\n")[1:]:
        head, *lines = block.splitlines()
        limit = re.fullmatch(r"(?:second run|probe|check) with at most (\d+) output tokens:", head)
        assert limit and all(line.startswith("[smoke ") for line in lines)
        sections[int(limit[1])] = lines
    assert list(sections) == [256, 4096, chat.MAX_OUTPUT_TOKENS]

    def thinking(lines: list[str]) -> list[int]:
        return [int(re.search(r"thinking tokens = (\d+)", line)[1]) for line in lines]  # type: ignore[index]

    def usable(lines: list[str]) -> int:
        return sum("usable = True" in line for line in lines)

    second, probe, check = sections.values()
    normal = " ".join((NOTE.parents[2] / "interaction" / "registration.md").read_text().split())
    paragraph = re.search(
        r"A second run of the same (\d+) calls with at most (\d+) output tokens returned (\d+) "
        r"empty replies, each with (\d+) thinking tokens\. The same four requests were then sent "
        r"(\d+) times each with at most (\d+) output tokens: all (\d+) replies were usable, and "
        r"the most thinking tokens in one reply was (\d+)\. The limit was set to (\d+) and the "
        r"(\d+) calls were run again at it: all (\d+) replies were usable, and the most "
        r"thinking tokens in one reply was (\d+)\. These (\d+) calls are recorded in "
        r"\[`docs/interaction/output-limit.txt`\]\(../docs/interaction/output-limit.txt\)\.",
        normal,
    )
    assert paragraph
    stated = [int(value) for value in paragraph.groups()]
    empty = [line for line in second if 'text = ""' in line]
    assert len(empty) == len(second) - usable(second) and len(set(thinking(empty))) == 1
    names = {line.split()[1] for line in probe}
    assert len(names) == 4 and len(probe) == 4 * stated[4]
    assert stated == [
        len(second),
        256,
        len(empty),
        thinking(empty)[0],
        len(probe) // 4,
        4096,
        usable(probe),
        max(thinking(probe)),
        chat.MAX_OUTPUT_TOKENS,
        len(check),
        usable(check),
        max(thinking(check)),
        len(second) + len(probe) + len(check),
    ]
    assert len(probe) == usable(probe) and len(check) == usable(check)
    assert len(second) == len(check) == len(chat.smoke(chat.StubClient()))
    assert max(thinking(probe)) > 256 and max(thinking(check)) > 256
    assert 4 * max(thinking(probe) + thinking(check)) < chat.MAX_OUTPUT_TOKENS


def test_prompted_prose_is_scope_and_record_only() -> None:
    text = (NOTE.parent / "prompted.md").read_text()
    assert len(re.findall(r"\n```\n.*?\n```", text, re.S)) == 1
    prose = re.sub(r"\n```\n.*?\n```", "", text, flags=re.S)
    prose = "\n".join(line for line in prose.splitlines() if not line.startswith("|"))
    assert prose == (
        "# The prompted arm: requests and spending bound\n"
        "\n"
        "This is preparation for the prompted prediction arm of the draft interaction design.\n"
        "The model's instructions are in [the instruction "
        "files](../../interaction/instructions/).\n"
        "Its question and declared answers, with an empty history and a made-up later history,\n"
        "are in [the request files](../../interaction/requests/). These are the materials given\n"
        "and the prediction asked for.\n"
        "\n"
        "The chat arm's rules are in [the registration](../../interaction/registration.md),\n"
        "and its format check is in [the record](format-check-chat.txt).\n"
        "\n"
        "The instructions are used under CC BY 4.0: Wang, Hilbe and Zhang, *PNAS* 123(5),\n"
        "e2525760123, [doi:10.1073/pnas.2525760123](https://doi.org/10.1073/pnas.2525760123),\n"
        "supplementary appendix, sections 5B and 5D. The other games use substituted parameters.\n"
        "\n"
        "The spending program reads training groups and held-out counts only. Its settings,\n"
        "assumptions and output are recorded below. Vertex price confirmation remains due\n"
        "on registration day. The format checks are outside the bound.\n"
        "\n"
        "\n"
        "Output of `python -m zoon.interaction.budget`:\n"
    )


def test_draft_status_does_not_reopen_completed_chat_work() -> None:
    root = NOTE.parents[2]
    readme = (root / "README.md").read_text()
    assert (
        "chat requests, transcripts and replay, scoring and intervals, the fitted files, "
        "and power and coverage under the registered rules are implemented and tested; "
        "the second seal, run guards and step state, and the report remain open" in readme
    )
    design = " ".join((NOTE.parent / "design.md").read_text().split())
    assert (
        "the chat bound are filled from the spending script, and both format checks are DONE;"
        in design
    )
    assert "The chat request check with the committed renderer is DONE;" in design
    assert (
        "The registration fixes separate decision and chat caps from that script's output."
        in design
    )


def test_registration_chat_contract_has_only_the_registered_rules() -> None:
    from zoon.interaction import chat

    text = (NOTE.parents[2] / "interaction" / "registration.md").read_text()
    section = text.split("### The chat arm\n\n", 1)[1].split("\n## Spending", 1)[0]
    assert section == (
        "It plays one seat at a time. Its request is the participant instructions, the\n"
        'line "You are Player n.", the history as rendered for `prompted`, and then:\n'
        "\n"
        "> How many of your E points do you contribute to the common pool in round "
        "r? Answer with one whole number of points and nothing else.\n"
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
        "After trimming white space (space, tab, line feed and carriage return "
        "only), the reply must be a whole number from zero to E.\n"
        "Fixed here: accept only a non-empty run of ASCII digits, with no sign, decimal\n"
        "point or other text; leading zeroes are allowed. Retry unusable replies under\n"
        "the three-reply rule in Data; transport failures and arm-only outcomes follow\n"
        "that section.\n"
        "\n"
        "Every seat of every round of every simulated group is its own call. Identical\n"
        "requests are NOT reused across decisions. The transcript identifies the game,\n"
        "group, round, seat and attempt, and a resumed decision uses its recorded replies.\n"
        "An incomplete final line, left by a write the system cut short, is not a record;\n"
        "it is removed on the next writable opening and the decision is asked again.\n"
        "Each removal is recorded with its byte offset, length and SHA-256 digest before\n"
        "the line is removed, and the report lists them.\n"
        "Read-only opening, including replay and digest computation, refuses an incomplete\n"
        "final line and changes no bytes. Other damage is refused without repair.\n"
        "Replay requires the recorded whole-transcript SHA-256, the exact line count and\n"
        "on-disk entries, rendered decision and recall requests, rounds in order and no\n"
        "attempt after an accepted reply; it uses every retained attempt without fresh calls.\n"
        "Changing text or token counts on any line fails replay.\n"
        "The first cause that makes the arm unusable stands and is never overwritten.\n"
        "The run machinery stores the cause member's name.\n"
        "Recall is still asked after the three-reply rule, failed replay or an operator stop\n"
        "during closed-loop, but not after withdrawal, a version change, the cap or an\n"
        "operator stop during recall; answered recall questions are kept and judged.\n"
        'Recall reports separately what stopped recall, labels unsent questions "not asked",\n'
        'and retains a version-invalid paid reply as "not judged: version", without judging\n'
        "or flagging it.\n"
        "The accepted points bypass recalibration and the band-to-points step.\n"
        "Run closed loop only, with the H3 simulation count. Score against `zoon` by the\n"
        "same H3 composite, difference and interval. It changes no verdict.\n"
    )
    reply_rules = (
        "Before generating," + text.split("Before generating,", 1)[1].split("\nThe outcomes", 1)[0]
    )
    assert reply_rules == (
        "Before generating, check model metadata without generating text; missing metadata\n"
        "does not establish withdrawal.\n"
        "A success without a non-empty string modelVersion or integer promptTokenCount is\n"
        "a transient transport fault and is retried within the bounded attempts, with no\n"
        "transcript record. Missing candidatesTokenCount or thoughtsTokenCount means zero,\n"
        "with or without candidate text; a present non-integer count, including a boolean,\n"
        "is the same transient fault."
    )
    normal = " ".join(text.split())
    assert "Spending is above the cap when it is reached; equality is allowed." in normal
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
        "OPERATOR_CLOSED_LOOP",
        "OPERATOR_RECALL",
    }
    assert "run machinery stores the member's name" in (chat.Session.__doc__ or "")
    assert chat.points(" \t\n\r00036\r", 36) == 36
    assert all(chat.points(t, 36) is None for t in ("+1", "1.", "37", "\x0c1", "1\xa0"))
    assert chat.request(kernel_view_for_docs()).count("How many of your") == 1


def test_chat_probe_history_is_not_a_claim_about_registered_thinking() -> None:
    text = " ".join((NOTE.parents[2] / "interaction" / "registration.md").read_text().split())
    assert "that reported no thinking tokens in those probes." in text
    assert "no thinking tokens at any setting" not in text
