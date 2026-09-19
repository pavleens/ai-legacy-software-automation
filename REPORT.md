# Design Write-Up

## 1. Architecture

The system separates expensive, adaptive discovery from cheap, deterministic replay.
During discovery, a model receives accessibility-derived observations and chooses from a
closed action set. A compiler converts the successful run into a typed
`CapabilityArtifact`. Discovery always produces a draft. A human reviews and approves the
artifact before unattended replay.

Replay imports no model client and accepts no provider configuration. It resolves recorded
semantic targets, runs actions through policy, verifies state predicates, and returns one of
four typed results: success, known business outcome, escalation, or hard failure. The
boundary between these phases is the product: a versioned, readable artifact that can be
reviewed, diffed, tested, and reused.

The main abstraction is `Surface`. It exposes observations as role, name, value, state,
container, row, and column information, with opaque live handles. Browser selectors remain
inside the Playwright implementation. The rest of the system only knows portable
accessibility concepts and six actions: navigate, click, type, select, read, and wait.

The implementation is intentionally synchronous and single-process. The assignment's hard
problems are compilation, reliable replay, safety, and handoff; a queue or service layer
would add deployment machinery without improving those properties.

## 2. Artifact Schema

The artifact is a function contract rather than a click transcript. It declares typed
inputs and outputs, ordered steps, a success checkpoint, known business outcomes, bounded
recovery rules, allowed origins, tenant overrides, version, approval state, and discovery
provenance.

Each target is an ordered ladder of strategies. Semantic role/name and visible-label
strategies are preferred, structural row/column addressing handles data grids, and ordinal
position is a last resort. Every rung is resolved against the discovery surface before it
is admitted to the artifact. A strategy that is already ambiguous is discarded.

Compiling live references is essential. Playwright's ARIA references are exact within one
snapshot but ephemeral and positional. The same reference identified a textbox on the
search screen and a table cell after navigation. Persisting it could act on the wrong
element without raising an error.

Data cells require stricter handling. A cell's accessible name is often the customer value
it contains. The first prototype therefore encoded one observed balance in its locator and
worked for only one member. The compiler now emits only value-free `ROW_COLUMN` strategies
for content-named elements. It describes the target as, for example, the Balance column of
the Savings row within Accounts On File. If no structural strategy resolves, compilation
fails instead of storing the value as a fallback.

Inputs are bound when the action payload equals a supplied discovery input. Replay validates
required, unknown, and malformed inputs before touching the surface. Required outputs are
also validated after the checkpoint; running all clicks is not success if the declared
balance is absent or malformed.

## 3. Determinism and Error Handling

Replay is deterministic because the model is absent, waits are state predicates rather than
fixed sleeps, target strategies are tried in a recorded order, and all branching is encoded
in the artifact and executor. The trace records every attempted strategy and the rung that
resolved, making degraded targeting observable.

Known outcomes are checked before recovery and before failure. "Member not found" and
"access denied" are valid answers to a valid request, not broken automation. They return a
business outcome and exit successfully so callers do not create false operational alarms.

Recovery rules have a closed remedy set: dismiss a declared control, reload, or wait and
retry. Attempts are bounded per rule and per step, with an absolute step re-entry ceiling
to stop interacting rules from cycling. Recovery actions pass through the same policy
engine as ordinary steps.

A final checkpoint proves the intended state was reached. On a frameset, the top-level URL
does not change as child content changes, so the sample capability verifies the structural
result target rather than trusting navigation. Hard failures identify the failure kind,
step, expected state, observed state, and complete strategy history.

The curated evidence covers successful replay with two different member IDs, known
outcomes, bounded recovery, invalid input, unresolved targets, and human handoff. One failed
discovery is retained because discovery is stochastic; replay is the repeatable path.

## 4. Heterogeneity and Multi-Tenant Support

The mock application exercises legacy conditions rather than a modern test-friendly DOM. It
uses a frameset, table layout, duplicate text, non-semantic classes, no test IDs, and
unassociated form labels. Perception scans every frame and augments the accessibility tree
with a conservative proximity-label pass for controls whose accessible name is empty.

The artifact vocabulary is portable. Role/name maps to UI Automation Name and ControlType
or AT-SPI role/name. `ROW_COLUMN` maps to grid or table patterns. Opaque handles isolate
Playwright ARIA references today and would isolate UIA runtime IDs in a desktop adapter.
Desktop support is not implemented, but the artifact, executor, policy, outcomes, evidence,
and handoff layers do not depend on the DOM.

Artifacts identify a vendor product, while origins are policy inputs. A tenant override can
replace the origin or selected step targets without mutating the base artifact. Consistent
fallback to lower-ranked strategies is recorded as drift evidence and indicates that a
tenant's version has diverged.

## 5. Escalation and Handoff

Escalation occurs when policy requires human confirmation, no target resolves, or bounded
recovery is exhausted. The automation publishes an `InterventionRequest` containing the
goal, step intent, reason, observed context, suggested action, and evidence location.

Control is represented as a session lease. Ceding transfers ownership from automation to
human while preserving the same browser context. Automated actions are forbidden while the
human holds the lease. Reclaim always creates a fresh observation because the operator may
have changed fields or navigated.

The operator resolves an intervention as `PERFORMED`, `UNBLOCKED`, or `ABORT`.
`PERFORMED` skips the step, preventing an irreversible action from running twice.
`UNBLOCKED` retries from fresh state. `ABORT` terminates the flow. If the performed step
declares an output, the operator must return that named output through the intervention;
otherwise final output validation rejects the run.

The file broker is deliberately mocked behind an interface that a queue-backed operator
console could implement. The lease, wait, dispositions, output transfer, fresh observation,
timeout behavior, and handoff audit record are implemented. A timeout returns `ESCALATED`
and never silently resumes.

## 6. Safety

Policy is data and denies by default. Every action checks the allowed action type, exact
origin, path segment boundary, and every actionable frame URL. This prevents an allowed top
page from concealing an action inside a cross-origin child frame. Irreversible steps require
human confirmation unless an operator explicitly enables unattended risky actions.
Discovery uses the same restriction and cannot silently execute a model-classified risky
step.

Model privacy is explicit. Local Ollama on a loopback address with a non-cloud model is the
default. Hosted Ollama tags, non-loopback hosts, OpenAI, and Anthropic are classified as
remote and require `--allow-remote-model`. The evidence records the provider used.

Evidence is append-only and redacted at write time. In the default real-data mode it omits
screenshots, page source, form values, text digests, and content-derived labels. Full
evidence requires `--synthetic-evidence`, which is documented for the mock target only.
Declared sensitive inputs and common credential/account patterns are still scrubbed from
all structured and text output. Binary replacement is not claimed to redact pixels; pixels
are disabled instead.

No credentials ship in the repository. Environment files are ignored, sample values are
placeholders, absolute workstation paths have been removed, and the public repository
includes a security policy, pinned dependency snapshot, CI, and a clean secret scan.

## 7. Cuts and Next Steps

The largest deliberate cuts are a real operator console, a desktop `Surface`
implementation, worker/queue infrastructure, and automatic discovery of known outcomes.
Known outcomes require institutional knowledge that one successful run cannot observe, so
they are added through a human annotation command that increments the artifact version and
resets approval.

The next engineering priorities are step-scoped recovery rules, relevance-ranked
observation truncation, batched extraction of control values, signed artifact approvals,
and an agent-facing capability catalogue generated from input/output schemas. Production
deployment would additionally require authentication, encrypted evidence storage with
retention controls, tenant isolation, rate limits, dependency scanning, and an external
security review.

Running the system exposed the most important design lesson. An early compiler resolved a
live reference after performing the action, so the reference was interpreted against the
next screen and a Search step recorded a target named Joined. Nothing raised an exception.
That failure is why stable targets are compiled from the pre-action observation, verified
before storage, and prohibited from carrying observed record values.
