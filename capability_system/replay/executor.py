"""Deterministic replay of a capability artifact.

=======================================================================================
THERE IS NO LLM IN THIS MODULE. NOT IMPORTED, NOT CALLED, NOT OPTIONAL, NOT BEHIND A
FLAG. This is the production execution path.
=======================================================================================

That constraint is the whole design, not a detail of it. Discovery is allowed to be
expensive, slow and probabilistic because it happens once, under observation, and its
output goes through human review. Replay happens thousands of times a day, unattended,
against a bank's systems of record. Three things follow, and they are the reason this file
looks the way it does:

  * **Reproducibility.** The same artifact against the same surface state must do the same
    thing every time. A model in the loop makes that untrue by construction, and an
    automation whose behaviour cannot be predicted cannot be approved for unattended use.
  * **Auditability.** Every decision here reduces to a declared predicate in the artifact
    that a human read and signed off. "Why did it click that?" has a diffable answer.
  * **Cost and latency.** The per-run cost of a capability is a page load, not a token
    bill, which is what makes running one ten thousand times a day reasonable.

The executor's only job is to run declared steps, evaluate declared predicates, and sort
what it sees into the four statuses in `outcomes.py`. Anything it cannot sort goes to a
human via the escalation path rather than being guessed at.

THREE ORDERING DECISIONS WORTH DEFENDING OUT LOUD
-------------------------------------------------

1.  **Known outcomes are checked BEFORE recovery rules, and before anything is treated as
    an error.** "No such member" is a correct answer to a correct question. If recovery ran
    first, a screen showing a legitimate business result could match a generic "something
    looks wrong, reload" rule, and the capability would burn its retry budget on a page
    that was already telling it the answer -- then report a hard failure instead of the
    result the caller asked for. Outcome detection is therefore the first thing that
    happens after every step, and a terminal outcome returns immediately.

2.  **Recovery attempts are tracked per rule, per step.** Not globally, and not per rule.
    A global counter means a flow that legitimately dismisses one interstitial on each of
    six screens exhausts its budget halfway through, for no reason other than that it is
    long. A per-rule counter that spans steps has the same defect. Scoping the counter to
    (step, rule) gives each rule a fresh, bounded budget at each step, which is the unit an
    operator reasons about when they write `max_attempts: 2` -- "twice, here", not "twice,
    ever". The ceiling is still absolute: it can never become an unbounded loop against a
    core banking system.

3.  **After reclaiming control from a human the executor re-observes, and uses only the
    fresh observation.** A human who resolved a stuck state has by definition changed the
    screen, and may have navigated somewhere else entirely, submitted something, or logged
    back in. Resuming against the observation captured before the handoff is how an
    automation clicks "Confirm" on a page it has never actually seen. `reclaim()` returns a
    fresh `Observation` precisely so this file can be written to require it, and the cached
    one is discarded rather than merged.
"""

from __future__ import annotations

import time
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from capability_system.artifact.schema import (
    ApprovalState,
    CapabilityArtifact,
    KnownOutcome,
    ParamType,
    RecoveryRule,
    RiskClass,
    Step,
    TargetSpec,
    TargetStrategy,
)
from capability_system.escalation.control import (
    InterventionResolution,
    InterventionRequest,
    SessionController,
    StuckReason,
)
from capability_system.perception.base import Action, ActionKind, Observation
from capability_system.replay.conditions import describe as describe_condition
from capability_system.replay.conditions import evaluate
from capability_system.replay.outcomes import FailureKind, ReplayResult, StepTrace
from capability_system.safety.policy import Decision, PolicyEngine, redact

# Defaults chosen to be overridable per instance rather than per call: a test wants a
# 50ms wait budget, a slow legacy screen wants seconds, and neither should require a
# different code path.
DEFAULT_WAIT_TIMEOUT_S = 10.0
DEFAULT_POLL_INTERVAL_S = 0.1
DEFAULT_ESCALATION_TIMEOUT_S = 300.0

# Hard ceiling on how many times a single step may be re-entered, on top of whatever the
# recovery rules themselves allow. This exists so that a pathological artifact (two rules
# whose remedies re-trigger each other) still terminates.
STEP_REENTRY_CEILING = 12


def strategy_label(strategy: TargetStrategy) -> str:
    """Stable, human-readable name for one ranked strategy.

    This string ends up in `StepTrace.strategy_used`, which is the evidence that the
    ranked-hypothesis design actually pays for itself: over many runs it shows which tier
    of strategy is carrying the capability, and a capability that has quietly fallen
    through to ORDINAL everywhere is one re-skin away from breaking and should be
    re-recorded before it does.
    """
    label = f"{strategy.kind.value}:{strategy.value}"
    if strategy.container:
        label += f"@{strategy.container}"
    if strategy.ordinal is not None:
        label += f"#{strategy.ordinal}"
    return label


class _StepAborted(Exception):
    """Internal control-flow signal carrying a finished `ReplayResult`.

    Used rather than threading an optional result through every helper's return type. It
    never escapes `run()`: the top level catches it and returns the carried result, so the
    public API still returns a `ReplayResult` on every path.
    """

    def __init__(self, result: ReplayResult) -> None:
        super().__init__(result.status.value)
        self.result = result


class ReplayExecutor:
    """Runs an approved `CapabilityArtifact` against a `Surface`. No model involved.

    The constructor takes collaborators rather than constructing them, because every one of
    them is the seam this system is arguing for: the surface makes it portable to desktop,
    the policy engine is the choke point that cannot be routed around, the recorder makes
    the run auditable, and the controller is what turns "stuck" into "a human has it"
    instead of "a failed job".
    """

    def __init__(
        self,
        surface: Any,
        policy: PolicyEngine,
        recorder: Any = None,
        controller: SessionController | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.surface = surface
        self.policy = policy
        self.recorder = recorder
        self.controller = controller
        self.tenant_id = tenant_id

        self.wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S
        self.poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
        self.escalation_timeout_s: float = DEFAULT_ESCALATION_TIMEOUT_S
        self._last_disposition: InterventionResolution | None = None
        self._last_operator_outputs: dict[str, Any] = {}

        # Per-run mutable state, reset at the top of `run`.
        self._artifact: CapabilityArtifact | None = None
        self._sensitive: set[str] = set()
        self._trace: list[StepTrace] = []
        self._outputs: dict[str, Any] = {}
        self._recovery_attempts: dict[tuple[str, str], int] = {}

    # -- recorder plumbing (duck-typed, never allowed to break a run) ------------------

    def _event(self, kind: str, **fields: Any) -> None:
        """Best-effort structured event. A recorder that can break the run it documents
        would trade reliability for observability, which is the wrong direction."""
        if self.recorder is None:
            return
        try:
            self.recorder.event(kind, **fields)
        except Exception:  # pragma: no cover - defensive
            pass

    def _record_step(self, index: int, step_id: str, **fields: Any) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.step(index, step_id, **fields)
        except Exception:  # pragma: no cover - defensive
            pass

    @property
    def _evidence_dir(self) -> str | None:
        reference = getattr(self.recorder, "reference", None)
        if reference is not None:
            return str(reference)
        run_dir = getattr(self.recorder, "run_dir", None)
        return str(run_dir) if run_dir is not None else None

    def _scrub(self, text: str) -> str:
        """Redact before anything reaches a trace, an event or an intervention request.

        Traces are read by operators and shipped to evidence storage, so they are an
        egress point for regulated data exactly like a log line is.
        """
        return redact(text, self._sensitive)

    # -- result helpers ---------------------------------------------------------------

    def _result_kwargs(self, started: float) -> dict[str, Any]:
        return {
            "trace": list(self._trace),
            "evidence_dir": self._evidence_dir,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    def _fail(
        self,
        kind: FailureKind,
        step_id: str | None,
        expected: str,
        observed: str,
        started: float,
    ) -> ReplayResult:
        artifact = self._artifact
        assert artifact is not None
        result = ReplayResult.failure(
            artifact.id,
            artifact.version,
            kind,
            step_id,
            self._scrub(expected),
            self._scrub(observed),
            **self._result_kwargs(started),
        )
        self._event(
            "replay_failed",
            failure_kind=kind.value,
            step_id=step_id,
            expected=result.expected,
            observed=result.observed,
        )
        return result

    # -- surface plumbing -------------------------------------------------------------

    def _observe(self) -> Observation:
        """Every observation goes through here so the lease is honoured on every read.

        Asking the controller first, rather than trusting call sites, is what makes the
        lease a gate instead of a convention -- the same argument the policy engine makes
        about actions.
        """
        if self.controller is not None:
            self.controller.assert_automation_may_act()
        return self.surface.observe()

    def _resolve_target(self, target: TargetSpec, trace: StepTrace) -> str | None:
        """Walk the ranked strategies in order, recording every one tried.

        Strategies are offered to the surface ONE AT A TIME rather than as a list, even
        though `Surface.resolve` accepts a sequence. Handing over the whole list would let
        the surface resolve the target without telling us which hypothesis won, and that
        attribution is the entire empirical case for ranked targeting. The cost is one
        extra call per failed tier, which is nothing next to a page load.
        """
        trace.strategies_tried = []
        for strategy in target.strategies:
            label = strategy_label(strategy)
            trace.strategies_tried.append(label)
            handle = self.surface.resolve([strategy])
            if handle:
                trace.strategy_used = label
                if label != trace.strategies_tried[0]:
                    # Worth an explicit event: a capability that routinely degrades past
                    # its first strategy is drifting away from what was recorded.
                    self._event(
                        "strategy_degraded",
                        step_id=trace.step_id,
                        used=label,
                        tried=list(trace.strategies_tried),
                    )
                return handle
        trace.strategy_used = None
        return None

    # -- tenant specialisation --------------------------------------------------------

    def _apply_tenant(self, artifact: CapabilityArtifact) -> CapabilityArtifact:
        """Return a tenant-specialised deep copy. The caller's artifact is never mutated.

        Mutating would be a live bug in the reuse story this schema exists for: one
        artifact object is shared across hundreds of institutions running the same vendor
        product, so writing tenant A's origin into it would leak into tenant B's next run.
        """
        if not self.tenant_id:
            return artifact
        override = next(
            (o for o in artifact.tenant_overrides if o.tenant_id == self.tenant_id), None
        )
        if override is None:
            return artifact

        specialised = artifact.model_copy(deep=True)
        recorded = (artifact.surface.recorded_origin or "").rstrip("/")
        new_origin = (override.origin or "").rstrip("/")

        if new_origin:
            specialised.surface.recorded_origin = new_origin
            for step in specialised.steps:
                # Only the origin prefix is rewritten. The path is part of the recorded
                # flow and identical across tenants running the same product version.
                if step.literal and recorded and step.literal.startswith(recorded):
                    step.literal = new_origin + step.literal[len(recorded) :]

        for step in specialised.steps:
            if step.id in override.step_target_overrides:
                step.target = override.step_target_overrides[step.id].model_copy(deep=True)

        self._event(
            "tenant_override_applied",
            tenant_id=self.tenant_id,
            origin=new_origin or None,
            overridden_steps=sorted(override.step_target_overrides),
        )
        return specialised

    # -- input contract ---------------------------------------------------------------

    def _validate_inputs(
        self, artifact: CapabilityArtifact, inputs: dict[str, Any], started: float
    ) -> ReplayResult | None:
        """Check the call against the declared contract BEFORE touching the surface.

        Deliberately strict in both directions. A missing required argument is obvious; an
        *unknown* argument is rejected too, because silently ignoring it means a caller who
        typoed `member_id_` as an argument gets a confident lookup of the wrong thing. For
        a capability that moves money, a typo must be a rejected call, not a successful one
        against a default.
        """
        declared = artifact.input_map()
        missing = [p.name for p in artifact.inputs if p.required and p.name not in inputs]
        unknown = [name for name in inputs if name not in declared]
        invalid_types = [
            p.name
            for p in artifact.inputs
            if p.name in inputs and not self._matches_type(inputs[p.name], p.type)
        ]
        if not missing and not unknown and not invalid_types:
            return None

        parts: list[str] = []
        if missing:
            parts.append(f"missing required: {sorted(missing)}")
        if unknown:
            parts.append(f"not in contract: {sorted(unknown)}")
        if invalid_types:
            expected_types = {
                p.name: p.type.value for p in artifact.inputs if p.name in invalid_types
            }
            parts.append(f"wrong type or format: {expected_types}")
        return self._fail(
            FailureKind.INVALID_INPUT,
            None,
            f"inputs matching contract {sorted(declared)}",
            "; ".join(parts),
            started,
        )

    @staticmethod
    def _matches_type(value: Any, expected: ParamType) -> bool:
        if expected is ParamType.STRING:
            return isinstance(value, str)
        if expected is ParamType.BOOLEAN:
            return isinstance(value, bool) or (
                isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}
            )
        if expected in {ParamType.NUMBER, ParamType.MONEY}:
            if isinstance(value, bool):
                return False
            try:
                cleaned = str(value).strip().replace(",", "")
                if expected is ParamType.MONEY:
                    cleaned = cleaned.replace("$", "").replace("CAD", "").strip()
                Decimal(cleaned)
                return True
            except (InvalidOperation, ValueError):
                return False
        if expected is ParamType.DATE:
            try:
                date.fromisoformat(str(value))
                return True
            except ValueError:
                return False
        return False

    # -- payload and action -----------------------------------------------------------

    def _payload_for(self, step: Step, inputs: dict[str, Any]) -> str | None:
        """`literal` wins over `binds_input` only because a step never legitimately has both.

        Ordering them this way keeps a navigate step (fixed URL) and a type step (bound
        argument) on one code path.
        """
        if step.literal is not None:
            return step.literal
        if step.binds_input is not None:
            value = inputs.get(step.binds_input)
            return None if value is None else str(value)
        return None

    def _build_action(self, step: Step, handle: str | None, payload: str | None) -> Action:
        kind = ActionKind(step.action)
        if kind is ActionKind.SELECT:
            return Action(kind=kind, handle=handle, option=payload)
        return Action(kind=kind, handle=handle, text=payload)

    # -- predicates -------------------------------------------------------------------

    def _match_outcome(self, artifact: CapabilityArtifact, obs: Observation) -> KnownOutcome | None:
        for outcome in artifact.known_outcomes:
            if evaluate(outcome.detect, obs, self.surface):
                return outcome
        return None

    def _match_recovery(self, artifact: CapabilityArtifact, obs: Observation) -> RecoveryRule | None:
        for rule in artifact.recovery_rules:
            if evaluate(rule.detect, obs, self.surface):
                return rule
        return None

    def _wait_for(self, step: Step, trace: StepTrace) -> tuple[bool, Observation]:
        """Poll a declared predicate to a bounded deadline. Never a bare sleep.

        A fixed sleep is simultaneously too slow on the common path and too short on the
        slow one, and it makes the run's duration a function of the artifact's guesswork
        rather than of the application. Polling a predicate is what lets the same artifact
        run against a fast test bank and a loaded production one unchanged.
        """
        assert step.wait_for is not None
        deadline = time.monotonic() + self.wait_timeout_s
        obs = self._observe()
        while True:
            if evaluate(step.wait_for, obs, self.surface):
                return True, obs
            if time.monotonic() >= deadline:
                return False, obs
            time.sleep(self.poll_interval_s)
            obs = self._observe()

    # -- escalation -------------------------------------------------------------------

    def _escalate(
        self,
        artifact: CapabilityArtifact,
        step: Step | None,
        reason: StuckReason,
        detail: str,
        suggested_action: str,
        started: float,
    ) -> Observation:
        """Hand the live session to a human, wait, and return a FRESH observation.

        Raises `_StepAborted` carrying an ESCALATED result if nobody resolves in time.
        Timing out must not silently resume: an unattended automation picking a session
        back up after a human abandoned it mid-edit is the single most dangerous thing
        this system could do, so the run ends and the caller is told the work is in flight.
        """
        if self.controller is None:
            raise AssertionError("_escalate called without a controller")  # pragma: no cover

        request = InterventionRequest(
            session_id="",  # cede() stamps the real session id it owns.
            capability_id=artifact.id,
            capability_version=artifact.version,
            goal=artifact.description or artifact.name,
            step_id=step.id if step else None,
            step_intent=step.intent if step else None,
            reason=reason,
            detail=self._scrub(detail),
            suggested_action=suggested_action,
            evidence_dir=self._evidence_dir,
        )
        published = self.controller.cede(request)
        self._event(
            "escalated",
            intervention_id=published.id,
            reason=reason.value,
            step_id=step.id if step else None,
            detail=request.detail,
        )

        if not self.controller.await_resume(published.id, timeout_s=self.escalation_timeout_s):
            artifact_ = artifact
            raise _StepAborted(
                ReplayResult.escalated(
                    artifact_.id,
                    artifact_.version,
                    published.id,
                    step.id if step else None,
                    self._scrub(detail),
                    expected=suggested_action,
                    **self._result_kwargs(started),
                )
            )

        # THE RULE THAT PREVENTS THE WORST BUG: use what the surface looks like NOW, not
        # what it looked like before a human touched it. See the module docstring.
        resolved = self.controller.broker.poll(published.id)
        disposition = getattr(resolved, "resolution", None)
        fresh, handoff = self.controller.reclaim()
        # The operator's disposition is an INSTRUCTION, not a status. "I performed the
        # step" and "I cleared what blocked you" require different next moves, and only
        # the human knows which happened. Stored here for the call site to branch on.
        self._last_disposition = disposition
        self._last_operator_outputs = dict(getattr(resolved, "operator_outputs", {}) or {})
        self._event(
            "control_reclaimed",
            intervention_id=published.id,
            navigated=handoff.navigated,
            duration_ms=handoff.duration_ms,
            resolution=getattr(disposition, "value", None),
        )
        return fresh

    # -- the run ----------------------------------------------------------------------

    def run(
        self,
        artifact: CapabilityArtifact,
        inputs: dict[str, Any],
        allow_draft: bool = False,
    ) -> ReplayResult:
        """Execute the capability and return exactly one `ReplayResult`.

        `allow_draft` is a parameter rather than a policy field on purpose: approving a
        capability is a governance decision recorded in the artifact, while running a draft
        is an operational decision taken by whoever is watching this particular run. Making
        the caller type it at the call site means a draft can never reach a scheduled,
        unattended job by inheriting someone's config.
        """
        started = time.monotonic()
        self._trace = []
        self._outputs = {}
        self._last_disposition = None
        self._last_operator_outputs = {}
        self._recovery_attempts = {}
        self._sensitive = {
            str(inputs[name]) for name in artifact.sensitive_names() if name in inputs
        }
        self._artifact = artifact

        # 1. Contract check first: an invalid call must not open a bank screen at all.
        invalid = self._validate_inputs(artifact, inputs, started)
        if invalid is not None:
            return invalid

        # 2. Approval gate. A DRAFT artifact is one a human has not signed off; letting one
        #    run unattended against a bank is the precise failure mode this gate exists for.
        if artifact.approval is not ApprovalState.APPROVED and not allow_draft:
            return self._fail(
                FailureKind.POLICY_BLOCKED,
                None,
                f"approval={ApprovalState.APPROVED.value}",
                f"approval={artifact.approval.value}; unattended replay refused "
                "(pass allow_draft=True for a supervised run)",
                started,
            )

        # 3. Tenant specialisation, on a copy.
        artifact = self._apply_tenant(artifact)
        self._artifact = artifact

        if len(artifact.steps) > self.policy.policy.max_steps:
            return self._fail(
                FailureKind.POLICY_BLOCKED,
                None,
                f"at most {self.policy.policy.max_steps} steps per run",
                f"artifact declares {len(artifact.steps)} steps",
                started,
            )

        self._event(
            "replay_started",
            capability_id=artifact.id,
            capability_version=artifact.version,
            tenant_id=self.tenant_id,
            approval=artifact.approval.value,
            steps=len(artifact.steps),
        )

        try:
            for index, step in enumerate(artifact.steps):
                terminal = self._run_step(artifact, index, step, inputs, started)
                if terminal is not None:
                    return terminal
            return self._verify_checkpoint(artifact, started)
        except _StepAborted as aborted:
            return aborted.result

    # -- one step ---------------------------------------------------------------------

    def _run_step(
        self,
        artifact: CapabilityArtifact,
        index: int,
        step: Step,
        inputs: dict[str, Any],
        started: float,
    ) -> ReplayResult | None:
        """Run a single step to completion. Returns a terminal result, or None to continue.

        The step is a loop rather than a straight line because recovery is defined as
        "remedy the condition, then do the step again". Re-running after a remedy is the
        conservative reading and the right one: if an interstitial or a stale-session
        banner was present, the step's effect is in doubt even when the click reported
        success, and these back-office steps (type a field, press Search) are re-runnable.
        """
        trace = StepTrace(step_id=step.id, intent=step.intent, action=step.action)
        self._trace.append(trace)
        step_started = time.monotonic()
        escalation_retry_used = False
        reentries = 0

        def finish(ok: bool, detail: str) -> None:
            trace.ok = ok
            trace.detail = self._scrub(detail)
            trace.elapsed_ms = int((time.monotonic() - step_started) * 1000)

        def fail(kind: FailureKind, expected: str, observed: str) -> ReplayResult:
            finish(False, f"{kind.value}: {observed}")
            return self._fail(kind, step.id, expected, observed, started)

        def note_recovery(rule: RecoveryRule) -> bool:
            """Charge one attempt to (this step, this rule). False when the ceiling is hit."""
            key = (step.id, rule.code)
            used = self._recovery_attempts.get(key, 0)
            if used >= rule.max_attempts:
                return False
            self._recovery_attempts[key] = used + 1
            return True

        while True:
            reentries += 1
            if reentries > STEP_REENTRY_CEILING:
                return fail(
                    FailureKind.RECOVERY_EXHAUSTED,
                    f"step {step.id} to complete within {STEP_REENTRY_CEILING} attempts",
                    "step re-entered past the absolute ceiling; recovery rules are cycling",
                )

            obs = self._observe()

            # (a) payload, (b) action + policy BEFORE anything touches the surface.
            payload = self._payload_for(step, inputs)
            try:
                action = self._build_action(step, None, payload)
            except ValueError:
                return fail(
                    FailureKind.INVALID_INPUT,
                    "a known ActionKind",
                    f"artifact declares action {step.action!r}",
                )

            verdict = self.policy.check_action(
                action,
                risk=step.risk,
                current_url=obs.url,
                surface_urls=obs.frame_urls,
            )
            if verdict.decision is Decision.BLOCK:
                return fail(
                    FailureKind.POLICY_BLOCKED,
                    "an action permitted by policy",
                    verdict.reason,
                )
            if verdict.decision is Decision.REQUIRE_CONFIRMATION:
                if self.controller is None:
                    # No human is reachable, so the only safe answer is to refuse. An
                    # irreversible step is never completed by default.
                    return fail(
                        FailureKind.POLICY_BLOCKED,
                        "human confirmation for an irreversible step",
                        f"{verdict.reason}; no controller is attached to obtain it",
                    )
                if escalation_retry_used:
                    return fail(
                        FailureKind.POLICY_BLOCKED,
                        "human confirmation for an irreversible step",
                        f"{verdict.reason}; already escalated once for this step",
                    )
                escalation_retry_used = True
                self._escalate(
                    artifact,
                    step,
                    StuckReason.POLICY_REQUIRES_CONFIRMATION,
                    f"Step {step.id} ({step.intent}) is classified "
                    f"{RiskClass.RISKY_IRREVERSIBLE.value}. {verdict.reason}",
                    "Review the pending action and either complete it yourself or "
                    "resolve the request to authorise the automation to proceed.",
                    started,
                )
                if self._last_disposition is InterventionResolution.ABORT:
                    finish(False, "operator declined to authorise the step")
                    return ReplayResult.escalated(
                        artifact.id, artifact.version,
                        escalation_id=self.controller.lease.intervention_id or "",
                        step_id=step.id,
                        reason="operator declined to authorise the irreversible step",
                        **self._result_kwargs(started),
                    )
                if self._last_disposition is InterventionResolution.PERFORMED:
                    # The human completed it in the live session. Retrying would attempt an
                    # irreversible action a SECOND time, which is the single thing this
                    # whole path exists to prevent. Record it and move on to the next step.
                    trace.recoveries.append("escalation:performed_by_operator")
                    if step.reads_into and step.reads_into in self._last_operator_outputs:
                        self._outputs[step.reads_into] = self._last_operator_outputs[
                            step.reads_into
                        ]
                    finish(True, "completed by the operator during handoff")
                    return None
                continue  # UNBLOCKED: re-observe and re-check; retry the step as recorded.

            # (c) target resolution through the ranked strategies.
            handle: str | None = None
            if step.target is not None:
                handle = self._resolve_target(step.target, trace)
                if handle is None:
                    rule = self._match_recovery(artifact, obs)
                    if rule is not None:
                        if not note_recovery(rule):
                            exhausted = self._exhausted(
                                artifact, step, rule, trace, fail, started
                            )
                            if exhausted is not None:
                                return exhausted
                            continue
                        self._apply_remedy(artifact, rule, trace, step)
                        continue
                    if self.controller is not None and not escalation_retry_used:
                        escalation_retry_used = True
                        self._escalate(
                            artifact,
                            step,
                            StuckReason.TARGET_NOT_FOUND,
                            f"Expected control {step.target.description!r}; none of "
                            f"{trace.strategies_tried} resolved on {obs.url}.",
                            "Bring the session to the screen containing this control, "
                            "then resolve the request.",
                            started,
                        )
                        continue
                    return fail(
                        FailureKind.TARGET_NOT_FOUND,
                        f"control {step.target.description!r} via {trace.strategies_tried}",
                        f"no strategy resolved on {obs.url!r} "
                        f"({len(obs.elements)} elements observed)",
                    )
                action = self._build_action(step, handle, payload)

            self._record_step(index, step.id, observation=obs, action=action)

            # (d) act.
            result = self.surface.act(action)
            if not result.ok:
                # A surface error is checked against the recovery rules first: "the app
                # blinked" is exactly what a `wait_retry` rule is for.
                after = self._observe()
                rule = self._match_recovery(artifact, after)
                if rule is not None:
                    if not note_recovery(rule):
                        exhausted = self._exhausted(artifact, step, rule, trace, fail, started)
                        if exhausted is not None:
                            return exhausted
                        continue
                    self._apply_remedy(artifact, rule, trace, step)
                    continue
                return fail(
                    FailureKind.SURFACE_ERROR,
                    f"{step.action} on {step.target.description if step.target else payload!r}",
                    result.error or "surface reported the action failed",
                )

            if step.reads_into:
                self._outputs[step.reads_into] = result.read_value

            # (e) wait on the declared predicate, if any.
            wait_failed_detail: str | None = None
            if step.wait_for is not None:
                met, obs_after = self._wait_for(step, trace)
                if not met:
                    # Do NOT fail yet. A wait that never comes true is very often the
                    # *shape* of a business outcome -- the results table never appears
                    # because there is no such member -- so the outcome check below gets
                    # first refusal on this state.
                    wait_failed_detail = (
                        f"condition never held within {self.wait_timeout_s:.1f}s: "
                        f"{describe_condition(step.wait_for)}"
                    )
            else:
                obs_after = self._observe()

            # (f) KNOWN OUTCOMES BEFORE ERRORS AND BEFORE RECOVERY. See module docstring.
            outcome = self._match_outcome(artifact, obs_after)
            if outcome is not None and outcome.terminal:
                finish(True, f"known outcome {outcome.code}")
                self._event("business_outcome", code=outcome.code, step_id=step.id)
                return ReplayResult.business(
                    artifact.id,
                    artifact.version,
                    outcome.code,
                    outcome.description,
                    **self._result_kwargs(started),
                )
            if outcome is not None:
                trace.detail = self._scrub(f"non-terminal outcome {outcome.code}")
                self._event("known_outcome_observed", code=outcome.code, step_id=step.id)

            # (g) then, and only then, recovery rules.
            rule = self._match_recovery(artifact, obs_after)
            if rule is not None:
                if not note_recovery(rule):
                    exhausted = self._exhausted(artifact, step, rule, trace, fail, started)
                    if exhausted is not None:
                        return exhausted
                    continue
                self._apply_remedy(artifact, rule, trace, step)
                continue

            if wait_failed_detail is not None:
                return fail(
                    FailureKind.PRECONDITION_FAILED,
                    describe_condition(step.wait_for) if step.wait_for else "",
                    f"{wait_failed_detail}; url={obs_after.url!r}",
                )

            finish(True, "ok")
            return None

    def _apply_remedy(
        self,
        artifact: CapabilityArtifact,
        rule: RecoveryRule,
        trace: StepTrace,
        step: Step,
    ) -> None:
        """Perform one pre-authorised remedy. Recorded in the trace either way.

        Remedies are deliberately the only actions in the system that were not recorded as
        steps, which is why the set is closed at three and each one is declared in the
        artifact with its own ceiling. An open-ended "do something sensible" here would be
        an unreviewed action against a bank.
        """
        trace.recoveries.append(rule.code)
        self._event(
            "recovery_fired",
            code=rule.code,
            remedy=rule.remedy,
            step_id=step.id,
            attempt=self._recovery_attempts.get((step.id, rule.code), 0),
            max_attempts=rule.max_attempts,
        )

        if rule.remedy == "dismiss" and rule.dismiss_target is not None:
            probe = StepTrace(step_id=f"{step.id}:recovery:{rule.code}", intent=rule.description,
                              action=ActionKind.CLICK.value)
            handle = self._resolve_target(rule.dismiss_target, probe)
            if handle:
                obs = self._observe()
                action = Action(kind=ActionKind.CLICK, handle=handle)
                verdict = self.policy.check_action(
                    action, current_url=obs.url, surface_urls=obs.frame_urls
                )
                if verdict.permitted:
                    self.surface.act(action)
        elif rule.remedy == "reload":
            obs = self._observe()
            if obs.url:
                # Reload is re-navigation, so it goes through the same policy choke point
                # as any other navigation rather than around it.
                action = Action(kind=ActionKind.NAVIGATE, text=obs.url)
                if self.policy.check_action(
                    action, current_url=obs.url, surface_urls=obs.frame_urls
                ).permitted:
                    self.surface.act(action)

        if rule.backoff_ms:
            time.sleep(rule.backoff_ms / 1000.0)

    def _exhausted(
        self,
        artifact: CapabilityArtifact,
        step: Step,
        rule: RecoveryRule,
        trace: StepTrace,
        fail: Any,
        started: float,
    ) -> ReplayResult | None:
        """A bounded remedy that outlived its ceiling. Escalate if a human is reachable.

        Escalating rather than failing is the right default here specifically because the
        condition is *known*: the artifact predicted this state and declared a remedy, so
        the fact that the remedy stopped working is information an operator can act on
        immediately, and the session is still live and positioned where they need it.
        """
        detail = (
            f"Recovery {rule.code} still matches after {rule.max_attempts} "
            f"attempt(s) at step {step.id}."
        )
        if self.controller is not None:
            self._escalate(
                artifact,
                step,
                StuckReason.RECOVERY_EXHAUSTED,
                detail,
                f"Clear the condition described as {rule.description or rule.code!r}, "
                "then resolve the request so replay can retry this step.",
                started,
            )
            # Resumed: give the step exactly one more pass with the human's fresh state.
            self._recovery_attempts.pop((step.id, rule.code), None)
            trace.recoveries.append(f"{rule.code}:escalated")
            return None  # None means "retry this step with the human's fresh state".
        return fail(FailureKind.RECOVERY_EXHAUSTED, f"{rule.code} cleared within ceiling", detail)

    # -- checkpoint -------------------------------------------------------------------

    def _verify_checkpoint(self, artifact: CapabilityArtifact, started: float) -> ReplayResult:
        """The success condition. Running every step is not the same as having succeeded.

        Without this, a capability "succeeds" whenever its clicks land, which is how a
        silent app change turns into a stream of confidently wrong results. The checkpoint
        is the artifact's own declaration of what done looks like.
        """
        obs = self._observe()

        # Outcomes get one last look before anything is called a failure: a screen that
        # says "no such member" is a correct answer, not a broken automation, even when it
        # surfaces only at the end of the flow.
        outcome = self._match_outcome(artifact, obs)
        if outcome is not None and outcome.terminal:
            self._event("business_outcome", code=outcome.code, step_id=None)
            return ReplayResult.business(
                artifact.id,
                artifact.version,
                outcome.code,
                outcome.description,
                **self._result_kwargs(started),
            )

        if not evaluate(artifact.checkpoint, obs, self.surface):
            observed = (
                f"url={obs.url!r} title={obs.title!r} "
                f"elements={len(obs.elements)} "
                f"text_digest={obs.text_digest[:200]!r}"
            )
            return self._fail(
                FailureKind.CHECKPOINT_NOT_MET,
                artifact.steps[-1].id if artifact.steps else None,
                describe_condition(artifact.checkpoint),
                observed,
                started,
            )

        declared = {field.name for field in artifact.outputs}
        outputs = {k: v for k, v in self._outputs.items() if k in declared}
        missing = [
            field.name
            for field in artifact.outputs
            if field.required and (field.name not in outputs or outputs[field.name] is None)
        ]
        malformed = [
            field.name
            for field in artifact.outputs
            if field.name in outputs
            and outputs[field.name] is not None
            and not self._matches_type(outputs[field.name], field.type)
        ]
        if missing or malformed:
            return self._fail(
                FailureKind.INVALID_OUTPUT,
                artifact.steps[-1].id if artifact.steps else None,
                f"declared outputs {sorted(declared)} with their specified types",
                f"missing={sorted(missing)} malformed={sorted(malformed)}",
                started,
            )
        self._event("replay_succeeded", outputs=sorted(outputs))
        return ReplayResult.success(
            artifact.id, artifact.version, outputs, **self._result_kwargs(started)
        )


__all__ = ["ReplayExecutor", "strategy_label"]
