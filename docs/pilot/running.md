# Running the registered pilot

Use Python 3.12 and uv. Keep the registered checkout clean.
The published raw task files and outside-study files must already be downloaded.
They must match their manifests.
Live calibration and references use US means computed from the outside-study files.
These commands do not download data.

Install the optional live dependency before the run:

```sh
uv sync --extra live
```

Set TYPESAFE_API_KEY and ZOON_VERTEX_PROJECT.
The second family uses application-default authentication through gcloud.
No project name or token belongs in a command, report or transcript.
Each calling step about to send requests checks ZOON_VERTEX_PROJECT and authentication using a
non-generating model-metadata request, before sending any question.
A 401 receives one refresh; a repeated 401 or a 403 fails the step resumably.
A request fault fails the step resumably; its saved reason is the exception type only.

An optional smoke command sends three unrelated requests to each family:
a binary question, a seven-answer question and a nine-answer question.
The second family uses digits for the seven options and letters for the nine options.
It writes no transcript and uses neither judge materials nor participant records.
It prints the reported versions, probabilities, sum before division, declared-answer
share, tokens and cost. The four content-filter settings are kept together in
SAFETY_SETTINGS; a live smoke test must confirm that the service accepts OFF.

```sh
uv run --extra live python -m zoon.pilot.run smoke
```

The owner creates the git tag registered on the merged registration commit, before judge-check.
Run these commands in order. Replace REGISTERED_COMMIT with the full commit that
the registered tag points to. That commit must be HEAD or an ancestor of HEAD;
every registered file must remain byte-identical. Any code difference requires a recorded
repair covering its full current snapshot.

```sh
uv run --extra live python -m zoon.pilot.run judge-check
uv run --extra live python -m zoon.pilot.run judge
uv run --extra live python -m zoon.pilot.run calibrate
uv run --extra live python -m zoon.pilot.run held-out --registered-commit REGISTERED_COMMIT
uv run python -m zoon.pilot.run report
```

The first four commands need TYPESAFE_API_KEY and ZOON_VERTEX_PROJECT.
The report needs no credentials and makes no model calls.
Use --workers to set the thread pool size. The default is four.
Run only one command at a time.

judge-check checks the outside evaluative correlation.
The decision family must have a finite correlation of at least 0.50.
A failure ends the run as “the judge could not be calibrated”.
A second-family correlation failure removes that family's numerical results.
judge obtains the task ratings, strict features, reversals and recognition replies.
Recognition runs even if the second family's numerical results are unusable.
calibrate fits prompted recalibration and all coded arms on training participants.
Every live step requires a clean work tree, including all untracked files
that git does not ignore. Before writing state or sending a request, its guard verifies
both datasets against their manifests, checks the outside ratings against the paper,
and regenerates the materials digests, request fixtures and cohorts. Report uses the same
guard. Stub commands accept any work tree.
Each of the five live commands checks the registered tag, ancestry, registered files and repair
snapshot before doing its work, even above the cap. held-out also checks
--registered-commit against the tag and the earlier gates before loading records.
It is the only step that loads held-out participants.
report writes docs/pilot/result.md and prints the same labelled lines.
After a stopped step, run report to record the outcome.
Report refuses any failed step, including above the cap; repeat that step before reporting.
After its guard passes, report always writes the result file and its state.
The report can only be completed or failed; an exception in its body interrupts it resumably.
Commit the report after inspecting it; the report command does not commit it.

Every step checks registration identity, code snapshots and previous state before returning
a recorded outcome or considering the cap. A later calling step requires every earlier step
to be completed, including above the cap.
A decision-model version change stops the run with no verdict.
A second-family version change or declared-answer share below one half removes its results.
An unreadable or blocked second-family reply is recorded as a share of zero.
A request failure after bounded retries in either family interrupts the step resumably.
Decision probabilities are divided by their sum when it is within 0.05 of one.
Both the returned and used probabilities are retained in the transcript.
Invalid decision replies are retried; exhaustion fails the step resumably.
Recognition replies are retained regardless of those numerical gates.
If withdrawal or a version change is detected during recognition,
the second family's numerical results are withheld; replies already recorded remain in the report.
A reply cut off at the token limit is used as it stands. A blocked or empty reply
is listed as having no reply and does not remove numerical results. A recognition reply
is flagged whenever it contains a registered key, even if it also says unknown.
Every comparison is repeated on the dilemmas that have a reply and no flag.
Flagged dilemmas and dilemmas with no reply are listed separately. If none remain,
the repeated comparisons are not estimable. Nothing is refitted or requested again.
An unusable second family still has its recognition replies and flags reported.
The spending cap is shared across both families and is recomputed from transcripts.
After the guard passes, the execution boundary checks spending before a pending calling step
starts. It saves that step as stopped if recorded spending is above the cap. Request handling
checks spending before replay or admission and after each recorded window, before raising
any request failure.
Every comparison uses the session's cap, which defaults to 150 dollars.
Report gives no verdict above that cap regardless of the saved step statuses.
Admission reserves the registered estimate for outstanding requests.
A request that would exceed the session's cap is not sent.
A cap stop gives no verdict.

Transcripts are append-only JSON lines in data/transcripts/live-REGISTERED_COMMIT/.
State files are in data/run/live-REGISTERED_COMMIT/.
REGISTERED_COMMIT is the full commit at the registered tag. Another registration has
separate directories and cannot reuse these files. Each transcript file's first line records the registered commit and mode;
it is checked before any replay.
A state file must name that commit, its step and its mode; its code snapshot must be the
original empty snapshot or match a repair snapshot recorded in the binding.
Each step stores its output and output digest in one atomic state file.
It binds the exact set of earlier steps, including an explicit absence for any unrun step.
Both are ignored by git. Do not delete either during a run.
Each transcript entry is keyed by the canonical request's SHA-256.
If interrupted, repeat the same command. Completed requests are read from the transcript.
Completed and stopped calling steps return their recorded status after the guard passes,
without overwriting state. Report is regenerated after its guard passes and still gives
no verdict above the session's cap. A live report writes docs/pilot/result.md and makes
the work tree dirty. Commit or remove that file before running report again.
Transient request failures receive four attempts with delays of one, two and four seconds.
An exhausted transport fault in either family records a failed step.
Local transport errors, timeouts, 5xx, 429, 404 and 410 responses are retried.
Other 4xx responses and bodies that are not JSON are also retried, except for the
authentication failures described above; exhaustion interrupts the step.
A model is withdrawn only if 404 or 410 is returned on every attempt at one request.
A mix of missing-model and other retryable faults interrupts the step after exhaustion.
An exhausted decision reply fault also records a failed step.
An authentication preflight fault also records a failed step.
Repeat that command after service or authentication recovers.
Completed requests are reused. Later steps refuse a failed prerequisite.
A refusal happens before the step body and writes nothing: neither state nor a repair
binding nor a transcript. Restore the inputs and repeat the same command.
A failed step is interrupted and resumable. Inside a calling step, every exception other
than a registered terminal stop saves failed, naming only the exception type, never its message.
Report saves failed for every exception, including a terminal-stop exception.
A stopped step is terminal. Run report for that outcome.
Every live step uses one guard measured from the commit tagged registered,
including report. Its bound set is every tracked file at the tag or at HEAD.
The whole work tree must have an empty git status, including untracked files that are
not ignored. Disk bytes must also equal the HEAD blobs for every bound path; this catches
skip-worktree and assume-unchanged edits even when git status hides them. A hidden edit
cannot be declared as a repair: it must first be committed. Added and deleted files count.
Registered files are prereg/, pilot/, the manifests and CSV tables under src/.
A difference in any registered file always refuses continuation, even with --repair.
Code is every other tracked difference outside docs/ and tests/, including root-level files.
Do not reset state to continue a confirmatory run.
If only code differs, commit the repair and keep the work tree clean.
Resume the same command with --repair "Reason for the repair".
The command records the reason, current commit and full difference from the registered commit
for code, including current digests. It reuses recorded answers. The difference must
exactly match the last recorded repair snapshot; any other difference needs another repair.
The binding is an ordered chain; returning to an earlier repair requires another declaration.
Returning to the registered code after a repair also requires a declared empty snapshot,
even after a previous declared return. An empty difference defines a return, including
a revert commit whose identity differs from the registered commit.
There is no baseline taken from the first step.
The report's Repairs heading lists every repair, or none, and computes afresh the full
list of files differing between the registered commit and the report's commit. It also
reports the registered commit and the commit each step ran at. Readers judge whether
each repair preserved all registered definitions; the guard cannot certify that.

The closed run-ending reasons are decision-model withdrawal, decision-model version
change, the spending cap, complete-records failure, and failed judge calibration.
The first four give no verdict. Calibration has its own outcome:
“the judge could not be calibrated”. Only the execution boundary saves stopped.
Only a declared repair can continue a changed implementation. The registration's rule
that any other change ends the confirmatory run requires the operator's judgment;
the guard refuses an undeclared difference and cannot certify repair intent.
The second family's closed exclusion causes are declared-answer share below one half
(including an unreadable or blocked reply counted as zero), version change, withdrawal,
failed judge calibration, and complete-records failure in a second-family arm.
A held-out record that cannot be read as 13 valid trials stops with the complete-records
reason. Training record faults interrupt or refuse; they do not end the run.
Connection metadata in a response interrupts the step before that reply is recorded.
Anything else only interrupts a step: a request that fails after the retries, an unreadable
decision reply after the retries, or a fault of authentication or configuration in either family.
An unreadable second-family probability reply counts as a share of zero, including when
it has no reported version. A program fault while reading a reply interrupts the step.
The step is then run again. It keeps every answer already recorded and sends only the requests that have none,
so running a step again cannot change an answer. A terminal stop wins over a retryable
failure in the same request window; among terminal outcomes the cap comes first.
Spending counts the tokens in recorded responses; discarded unreadable replies and the
six unrelated connection-test requests are not counted.
An empty transcript file needs no recovery; repeat the command.
A partial final transcript line requires recovery of the last complete durable record.
For a truncated state file, restore its last complete durable copy before repeating the command.
A service reply lost before it was durably recorded cannot be recovered by this program.

For a synthetic dry run, add --stub to each of the five commands.
Use the current full commit for --registered-commit.
No environment variables or downloaded data are needed.
The dry run uses 48 synthetic training participants, eight for calibration and 12 for scoring.
It uses all 13 rotations and the committed materials.
Every printed line starts with a stub label.
It writes data/transcripts/stub/, data/run/stub/ and data/run/stub/result.md.
These files cannot replace the live result: state and transcripts carry the stub mode,
and every rendered line identifies the synthetic run. A live run refuses copied stub files.
