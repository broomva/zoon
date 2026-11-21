# Chat review R1 implementation record

Base: `ee176fa2d80b6d9fd3fe3ab41199ffae8306c2f5`, branch `chat-arm`.
This record covers the fixes requested after the independent code and design
reviews of that commit. All work and mutations ran in the authorized clone.
No model or network calls were made, no held-out records were read, and no
subagents were started. The protected-path diff against the base is zero bytes.

## Result and interfaces

All requested decisions are implemented; none were deferred or substituted.
`Session(..., terminal=cause)` restores a persisted terminal cause. Every session
stop latches its cause. `recall` returns `RecallResult(replies, terminal)` and
labels each question as judged, not judged because of version, or not asked.
Paid version-invalid replies are retained without a recognition flag.

The future run machinery must call `VertexClient.preflight()` before its first
generating request, persist the terminal cause and tail-removal count in step
state, and record `Transcript.digest()` for `replay_transcript`. No run machinery
was added in this change. The digest is SHA-256 of the whole transcript's exact
bytes, including its canonical header and final newlines. Game intent records
also bind every attempted reply through `Choice.attempts`, including unusable
replies before an accepted answer. Recall receipts include their entry digests.

Append writes complete lines without buffering and fsyncs them. On a partial or
short write, interruption or fsync failure, it reopens the file, truncates to the
prior length and fsyncs before raising `StorageInterruption`. A failed rollback
raises `ConsistencyInterruption`. Only an incomplete final line is removed at
load; complete damage is refused without repairing it. `Session.tail_removals`
exposes the number removed at that opening for the report.

Transport, configuration, consistency and storage faults have distinct types.
A malformed success is retried without recording a reply. Missing output or
thinking counts default to zero only when there is no candidate text.
Exception sanitization is tested by walking both context and cause chains.

## Validation

The bounded suite command was:

```sh
TMPDIR=/private/tmp/zoon-w16b/tmp ZOON_REQUIRE_TRAINING=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/zoon-w16b/pytest tests/test_interaction*.py tests/test_docs.py tests/test_structure.py
```

Result: **552 passed** in 64.89 seconds. The focused review baseline passed
242 tests before the final three additional write-failure cases were added;
the bounded suite above includes those cases.

- `.venv/bin/ruff check`: all checks passed.
- `.venv/bin/ruff format --check`: 102 files already formatted.
- `PATH=/opt/homebrew/opt/node@22/bin:$PATH .venv/bin/pyright --pythonpath .venv/bin/python`:
  zero errors and zero warnings. The default Node could not start because of a
  missing Homebrew library; the installed Node 22 was used without changing the environment.
- `TMPDIR=/private/tmp/zoon-w16b/tmp .venv/bin/python -m zoon.interaction.budget`:
  exact match to the updated paste in `prompted.md`.
- `TMPDIR=/private/tmp/zoon-w16b/tmp .venv/bin/python -m zoon.interaction.chat smoke --stub`:
  12 calls, 12 usable replies, no service access.
- Byte comparison of the supplied format-check record and
  `docs/interaction/format-check-chat.txt`: identical. SHA-256:
  `daf600cdcd5486ff7cc0fe381838bd074f9be1f7151c4833755dc50b00660d1d`.

The decision-model figures and all chat call, token and cost figures are
unchanged. The decision cap remains 17 USD, the chat cap 966 USD, and the chat
bound 965.85657600 USD. The program/paste diff adds only the decision cap multiple,
the stub-check default, and two labelled assumptions (byte-fallback tokenisation
and the service honouring the combined candidate-plus-thinking output limit).

## Survivor rerun

Every one of the review's 56 non-equivalent survivors was revisited sequentially.
Each row used the same bounded pytest invocation and the six files below, with
`-x -k <selector>` added. A killed result requires an assertion failure, not a
collection error or timeout. Every mutation's original bytes were restored in a
`finally` block before the next case.

```text
tests/test_interaction_chat.py
tests/test_interaction_chat_repairs.py
tests/test_interaction_docs.py
tests/test_interaction_budget.py
tests/test_docs.py
tests/test_structure.py
```

**55 killed; 1 left equivalent after the implementation changed.** Mutant 104
removed `flush()` before `fsync()` from buffered append. Append now uses
unbuffered file objects, so there is no append buffer or flush to remove. Its
no-op rerun passed; it is left for that reason, not claimed as killed. Complete
writes before fsync and rollback after sync failure have separate passing tests.

Changed code was re-anchored to preserve the mutation's meaning. The five removed
prose claims were tested by reinserting the mutated claim before the scope
paragraph; the scope-only test rejects each. The README status mutation was
similarly reapplied to its corrected sentence. Header mutations that now have a
second occurrence in preflight targeted the generating request.

The review's explicitly equivalent or immaterial cases were excluded as requested:
54, 57, 97, 111, 112, 113, 141, 142, 144, 149, 153, 161, 179, 185, 239, 240, 254.

| Mutant | Result | Pytest selector | Adaptation |
|---:|---|---|---|
| 17 | killed | `recall_adjacent_runs` | original mutation |
| 31 | killed | `precise_trimming` | original mutation |
| 38 | killed | `precise_trimming` | explicit trim anchor |
| 43 | killed | `admission_estimate_independent` | original mutation |
| 44 | killed | `admission_estimate_independent` | original mutation |
| 53 | killed | `transcript_leak_guard` | original mutation |
| 59 | killed | `outgoing_request_guard` | outgoing bearer guard anchor |
| 60 | killed | `live_wire` | original mutation |
| 65 | killed | `client_response_guard` | original mutation |
| 70 | killed | `missing_finish_reason` | original mutation |
| 82 | killed | `live_wire` | original mutation |
| 84 | killed | `malformed_success_retried` | strict version parser anchor |
| 91 | killed | `short_registered_commit` | original mutation |
| 104 | left: equivalent | `fsync_follows` | equivalent after unbuffered append rewrite: no buffer or flush remains |
| 129 | killed | `withdrawal_is_latched` | original mutation |
| 138 | killed | `ledger_each` | original mutation |
| 139 | killed | `ledger_each` | original mutation |
| 145 | killed | `replay_checks_request` | original mutation |
| 163 | killed | `fixtures_command_regenerates` | original mutation |
| 164 | killed | `direct_json_and_http_errors` | original mutation |
| 165 | killed | `live_wire` | original mutation |
| 166 | killed | `direct_json_and_http_errors` | JSON wrapper anchor |
| 167 | killed | `live_wire` | original mutation |
| 168 | killed | `sanitized_faults` | typed credential wrapper anchor |
| 203 | killed | `exact_request_and_same` | original mutation |
| 207 | killed | `exact_request_and_same` | original mutation |
| 208 | killed | `exact_request_and_same` | original mutation |
| 210 | killed | `registration_chat_settings` | original mutation |
| 211 | killed | `registration_chat_settings` | original mutation |
| 212 | killed | `registration_chat_settings` | original mutation |
| 213 | killed | `registration_chat_settings` | original mutation |
| 216 | killed | `registration_chat_settings` | original mutation |
| 217 | killed | `registration_chat_settings` | original mutation |
| 218 | killed | `registration_chat_settings` | original mutation |
| 219 | killed | `registration_chat_settings` | original mutation |
| 221 | killed | `registration_chat_settings` | original mutation |
| 222 | killed | `registration_chat_settings` | original mutation |
| 223 | killed | `registration_chat_settings` | original mutation |
| 224 | killed | `registration_chat_settings` | original mutation |
| 225 | killed | `registered_format_check_counts` | original mutation |
| 226 | killed | `registration_chat_settings` | original mutation |
| 230 | killed | `prompted_prose_is_scope` | original prose removed; reinsert mutated claim |
| 232 | killed | `prompted_prose_is_scope` | original prose removed; reinsert mutated claim |
| 233 | killed | `prompted_prose_is_scope` | original prose removed; reinsert mutated claim |
| 234 | killed | `prompted_prose_is_scope` | original prose removed; reinsert mutated claim |
| 235 | killed | `prompted_prose_is_scope` | original prose removed; reinsert mutated claim |
| 236 | killed | `draft_status` | original mutation |
| 237 | killed | `draft_status` | status corrected; reintroduce stale claim |
| 238 | killed | `transcript_leak_guard` | updated all guard anchors |
| 241 | killed | `transcript_leak_guard` | updated all guard anchors |
| 244 | killed | `short_registered_commit` | original mutation |
| 245 | killed | `usable_third_reply` | original mutation |
| 246 | killed | `ledger_each` | original mutation |
| 251 | killed | `live_wire` | generating-path occurrence |
| 252 | killed | `live_wire` | generating-path occurrence |
| 253 | killed | `live_wire` | original mutation |

## Additional findings

The implementation check exposed a further write-recovery concern: buffered
output could complicate rollback after an interrupted flush. The final path
uses unbuffered complete-line writes and a separate rollback handle. Tests also
cover a short return from write, KeyboardInterrupt, and a first fsync failure,
with byte-identical rollback and a successful next append.

No outstanding fault was found by the checks above. Live service behaviour was
not exercised under the task's offline boundary. Future step-state integration
remains outside this task, as required by the prohibition on adding run machinery.
