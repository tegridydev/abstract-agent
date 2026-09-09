# Model output and recovery

A response can arrive successfully and still be unusable. The compiler keeps that separate
from a connection failure so recovery can address the actual problem.

## Reading a response

The compiler accepts exact JSON, fenced JSON, balanced JSON and bounded tagged text.
Partial JSON is considered only when transport metadata confirms truncation. A short prose
or Markdown prefix can precede an unfinished object. Duplicate keys and schema echoes are
rejected on every route.

Each stage can normalise content, project it into its contract or use a conservative fallback.
The run records which route was used and whether the result needs human review. Gap mapping
can keep complete validated items from a truncated response, discard the unfinished tail
and record `validated_partial` with a warning.

## Keeping model requests small

Candidate generation separates the causal idea from the experimental protocol. Critique
and novelty checks work on one candidate at a time. The scientific record names the system,
intervention, comparator, outcomes, metrics, decision rules, mechanism, evidence, confounders,
risks and experiment.

Python renders the canonical hypothesis and composes the proposal abstract from the validated
candidate. The writing model supplies rationale and additional limitations. It cannot rename
the candidate, invent paper identifiers or replace the scientific claim in the final record.

Generation across domains requires independently attributable evidence from both domains.
When that evidence is missing, `cross-domain-application` becomes
`boundary-condition-or-failure-mode`. The configured axis, effective axis and reason are
recorded. If no valid scope survives, the question stays broad. Missing gaps stay unknown.

## Compatibility profiles

Version 3 profiles track schema, JSON and tagged text routes by model digest, prompt version
and contract. Two structural failures in distinct runs can demote a route. A confirmed
failure to initialise the grammar can demote it immediately. Content quality failures are
recorded separately, so a weak answer does not teach the transport layer the wrong lesson.

Fallback is a decision for an individual call, not a learned model capability. Profiles
store hashes, counters and route metadata, without prompts or responses. Older profiles
are ignored. You can reset learned routes from the models screen.

## Repair and stopping

A failed unit can be omitted or marked uncertain while successful units remain available.
Completed mapped units are checkpointed for resume. A wider transport interruption saves
a diagnostic; invalid output and connection failures retain separate categories.

Scientific checks look for placeholder methods, duplicated relations, undefined outcomes,
inconsistent metrics or controls, missing decision rules and weak evidence alignment.
Revision must address the recorded blocking findings. A finding is resolved only when the
scientific content it concerns has changed.

The final gate allows one focused repair with the candidate, unresolved findings and at
most six evidence cards. Repair cannot change identity or the innovation axis. If the same
checks still fail, the run keeps the evidence and reports `insufficient_valid_candidates`.
It does not turn the failed candidate into a finished proposal.

Custom pipelines can use `draft_conditioned` generation. Projection and fallback remain
visible in the provenance. See [architecture and output](architecture.md) for saved records
and [usage](usage.md) for resume commands.
