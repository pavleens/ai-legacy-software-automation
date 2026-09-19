"""Tests for the deterministic replay executor.

WHY A STUB SURFACE RATHER THAN THE MOCK BANK. The behaviour under test is the executor's
taxonomy -- what counts as a business outcome, when recovery is allowed to fire, when a
human is called -- and none of that is a property of a browser. Driving these cases
through Playwright would make them slow, flaky, and would require building an HTML fixture
for every error state rather than simply declaring the state. The stub implements the same
`Surface` protocol the real one does, which is the point of having the protocol.
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import pytest

from capability_system.artifact.schema import (
    ApprovalState,
    CapabilityArtifact,
    InputParam,
    KnownOutcome,
    OutputField,
    ParamType,
    RecoveryRule,
    RiskClass,
    StateCondition,
    Step,
    StrategyKind,
    SurfaceDescriptor,
    TargetSpec,
    TargetStrategy,
)
from capability_system.perception.base import (
    Action,
    ActionKind,
    ActionResult,
    Observation,
    UIElement,
)
from capability_system.replay import executor as executor_module
from capability_system.replay.conditions import evaluate
from capability_system.replay.executor import ReplayExecutor, strategy_label
from capability_system.replay.outcomes import FailureKind, ResultStatus
from capability_system.safety.policy import Policy, PolicyEngine

ORIGIN = "https://bank.example.com"
SEARCH_URL = f"{ORIGIN}/members/search"


# ---------------------------------------------------------------------------------------
# Stub surface
# ---------------------------------------------------------------------------------------


class StubSurface:
    """Scripted `Surface`. Observations come from a list; targets from a dict.

    `resolvable` maps a strategy label (the same string the executor puts in the trace) to
    a handle. Anything not in the dict does not resolve, which is how a "first strategy
    misses, second hits" case is expressed without inventing markup.
    """

    def __init__(
        self,
        observations: Sequence[Observation],
        resolvable: dict[str, str] | None = None,
        act_ok: bool = True,
    ) -> None:
        self._observations = list(observations)
        self.resolvable = dict(resolvable or {})
        self.act_ok = act_ok
        self.actions: list[Action] = []
        self.observe_calls = 0
        self.resolve_calls: list[str] = []
        self.read_value = "1,234.56"
        self.closed = False

    # The last scripted observation repeats, so a test only scripts the states it cares
    # about instead of counting how many times the executor happens to look.
    def observe(self) -> Observation:
        self.observe_calls += 1
        index = min(self.observe_calls - 1, len(self._observations) - 1)
        return self._observations[index]

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if not self.act_ok:
            return ActionResult(ok=False, error="stub surface refused the action")
        if action.kind is ActionKind.READ:
            return ActionResult(ok=True, read_value=self.read_value)
        return ActionResult(ok=True)

    def resolve(self, strategies: Sequence[Any]) -> str | None:
        for strategy in strategies:
            label = strategy_label(strategy)
            self.resolve_calls.append(label)
            if label in self.resolvable:
                return self.resolvable[label]
        return None

    def close(self) -> None:
        self.closed = True


class ExplodingSurface:
    """Any contact with this surface is a test failure.

    Used to prove the input-contract check happens before the first observation, which is
    a claim about ordering that a mock-call assertion states more weakly.
    """

    def observe(self) -> Observation:  # pragma: no cover - must never run
        raise AssertionError("surface was touched")

    def act(self, action: Action) -> ActionResult:  # pragma: no cover
        raise AssertionError("surface was touched")

    def resolve(self, strategies: Sequence[Any]) -> str | None:  # pragma: no cover
        raise AssertionError("surface was touched")

    def close(self) -> None:  # pragma: no cover
        raise AssertionError("surface was touched")


# ---------------------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------------------


def strategy(kind: StrategyKind, value: str) -> TargetStrategy:
    return TargetStrategy(kind=kind, value=value, confidence=0.9, rationale="test")


SEARCH_BOX = TargetSpec(
    description="Member ID field",
    strategies=[strategy(StrategyKind.LABEL, "Member ID")],
)
SEARCH_BUTTON = TargetSpec(
    description="Search button",
    strategies=[strategy(StrategyKind.ROLE_NAME, "button|Search")],
)
BALANCE_FIELD = TargetSpec(
    description="Balance cell",
    strategies=[strategy(StrategyKind.TEXT, "Available balance")],
)

RESOLVABLE = {
    "label:Member ID": "h-input",
    "role_name:button|Search": "h-search",
    "text:Available balance": "h-balance",
}


def observation(
    url: str = SEARCH_URL,
    text: str = "Member 88 Available balance 1,234.56",
    title: str = "Member search",
    elements: Sequence[UIElement] | None = None,
    frame_urls: Sequence[str] | None = None,
) -> Observation:
    return Observation(
        url=url,
        title=title,
        text_digest=text,
        elements=list(elements or [UIElement(handle="h-balance", role="cell", name="Available balance", value="1,234.56")]),
        frame_urls=list(frame_urls or []),
    )


def policy() -> PolicyEngine:
    return PolicyEngine(Policy(allowed_origins=[ORIGIN]))


def artifact(**overrides: Any) -> CapabilityArtifact:
    """A three-step lookup capability: type an id, search, read the balance."""
    base: dict[str, Any] = dict(
        id="lookup_member_balance",
        version=3,
        name="Look up member balance",
        description="Find a member by id and return their available balance.",
        surface=SurfaceDescriptor(kind="web", product="acme-core", recorded_origin=ORIGIN),
        approval=ApprovalState.APPROVED,
        inputs=[
            InputParam(name="member_id", required=True, description="Member number"),
            InputParam(name="note", required=False, description="Optional note"),
        ],
        outputs=[OutputField(name="balance", from_step="read_balance")],
        steps=[
            Step(id="type_id", intent="Enter the member id", action="type",
                 target=SEARCH_BOX, binds_input="member_id"),
            Step(id="submit", intent="Run the search", action="click", target=SEARCH_BUTTON),
            Step(id="read_balance", intent="Read the available balance", action="read",
                 target=BALANCE_FIELD, reads_into="balance"),
        ],
        checkpoint=StateCondition(kind="text_present", value="Available balance"),
        known_outcomes=[],
        recovery_rules=[],
        allowed_origins=[ORIGIN],
    )
    base.update(overrides)
    return CapabilityArtifact(**base)


def make_executor(surface: Any, **kwargs: Any) -> ReplayExecutor:
    ex = ReplayExecutor(surface, policy(), **kwargs)
    # Tests must not pay for real waiting; the budget is an instance knob for this reason.
    ex.wait_timeout_s = 0.05
    ex.poll_interval_s = 0.0
    return ex


# ---------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------


def test_conditions_cover_every_kind_and_negate() -> None:
    surface = StubSurface([observation()], RESOLVABLE)
    obs = observation()

    assert evaluate(StateCondition(kind="text_present", value="Available balance"), obs, surface)
    assert not evaluate(StateCondition(kind="text_present", value="No such member"), obs, surface)
    assert evaluate(
        StateCondition(kind="text_present", value="No such member", negate=True), obs, surface
    )
    assert evaluate(StateCondition(kind="url_matches", value=r"/members/\w+"), obs, surface)
    assert not evaluate(StateCondition(kind="url_matches", value=r"/transfers"), obs, surface)
    assert evaluate(
        StateCondition(kind="element_present", value="", target=BALANCE_FIELD), obs, surface
    )
    missing = TargetSpec(description="Ghost", strategies=[strategy(StrategyKind.TEXT, "nope")])
    assert evaluate(StateCondition(kind="element_absent", value="", target=missing), obs, surface)
    assert evaluate(
        StateCondition(kind="value_equals", value="1,234.56", target=BALANCE_FIELD), obs, surface
    )
    assert not evaluate(
        StateCondition(kind="value_equals", value="0.00", target=BALANCE_FIELD), obs, surface
    )


# ---------------------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------------------


def test_happy_path_returns_success_with_declared_outputs() -> None:
    surface = StubSurface([observation()], RESOLVABLE)
    result = make_executor(surface).run(artifact(), {"member_id": "88"})

    assert result.status is ResultStatus.SUCCESS
    assert result.outputs == {"balance": "1,234.56"}
    assert [t.step_id for t in result.trace] == ["type_id", "submit", "read_balance"]
    assert all(t.ok for t in result.trace)
    # The bound input actually reached the surface, rather than the literal being reused.
    typed = [a for a in surface.actions if a.kind is ActionKind.TYPE]
    assert typed and typed[0].text == "88"


def test_undeclared_read_output_is_not_returned() -> None:
    cap = artifact(outputs=[])
    surface = StubSurface([observation()], RESOLVABLE)
    result = make_executor(surface).run(cap, {"member_id": "88"})
    assert result.status is ResultStatus.SUCCESS
    assert result.outputs == {}


# ---------------------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------------------


def test_missing_required_input_is_invalid_and_never_touches_the_surface() -> None:
    result = make_executor(ExplodingSurface()).run(artifact(), {"note": "hi"})

    assert result.status is ResultStatus.HARD_FAILURE
    assert result.failure_kind is FailureKind.INVALID_INPUT
    assert "member_id" in (result.observed or "")
    assert result.trace == []


def test_undeclared_input_is_rejected() -> None:
    result = make_executor(ExplodingSurface()).run(
        artifact(), {"member_id": "88", "sort_code": "00-00-00"}
    )
    assert result.failure_kind is FailureKind.INVALID_INPUT
    assert "sort_code" in (result.observed or "")


def test_wrong_input_type_is_rejected_before_touching_surface() -> None:
    cap = artifact(inputs=[InputParam(name="member_id", type=ParamType.NUMBER)])
    result = make_executor(ExplodingSurface()).run(cap, {"member_id": "not-a-number"})
    assert result.failure_kind is FailureKind.INVALID_INPUT
    assert "wrong type" in (result.observed or "")


# ---------------------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------------------


def test_draft_artifact_is_refused_unattended_but_runs_with_allow_draft() -> None:
    draft = artifact(approval=ApprovalState.DRAFT)

    refused = make_executor(ExplodingSurface()).run(draft, {"member_id": "88"})
    assert refused.status is ResultStatus.HARD_FAILURE
    assert refused.failure_kind is FailureKind.POLICY_BLOCKED
    assert "draft" in (refused.observed or "").lower()

    surface = StubSurface([observation()], RESOLVABLE)
    supervised = make_executor(surface).run(draft, {"member_id": "88"}, allow_draft=True)
    assert supervised.status is ResultStatus.SUCCESS


# ---------------------------------------------------------------------------------------
# Business outcome, which must never be a failure
# ---------------------------------------------------------------------------------------


def test_known_outcome_returns_business_outcome_not_failure() -> None:
    not_found = observation(text="No matching member was found for that id.", elements=[])
    surface = StubSurface([observation(), observation(), not_found], RESOLVABLE)
    cap = artifact(
        known_outcomes=[
            KnownOutcome(
                code="MEMBER_NOT_FOUND",
                description="No member exists with that id.",
                detect=StateCondition(kind="text_present", value="No matching member"),
                terminal=True,
            )
        ]
    )

    result = make_executor(surface).run(cap, {"member_id": "99"})

    assert result.status is ResultStatus.BUSINESS_OUTCOME
    assert result.status is not ResultStatus.HARD_FAILURE
    assert result.failure_kind is None
    assert result.outcome_code == "MEMBER_NOT_FOUND"


def test_business_outcome_wins_over_a_matching_recovery_rule() -> None:
    """The ordering claim, tested directly: the same screen matches both, outcome wins."""
    both = observation(text="No matching member was found. Please try again.", elements=[])
    surface = StubSurface([observation(), both], RESOLVABLE)
    cap = artifact(
        known_outcomes=[
            KnownOutcome(
                code="MEMBER_NOT_FOUND",
                detect=StateCondition(kind="text_present", value="No matching member"),
            )
        ],
        recovery_rules=[
            RecoveryRule(
                code="TRANSIENT_SLOW_LOAD",
                detect=StateCondition(kind="text_present", value="Please try again"),
                remedy="wait_retry",
                max_attempts=1,
                backoff_ms=0,
            )
        ],
    )

    result = make_executor(surface).run(cap, {"member_id": "99"})
    assert result.status is ResultStatus.BUSINESS_OUTCOME
    assert result.outcome_code == "MEMBER_NOT_FOUND"
    assert all(not t.recoveries for t in result.trace)


# ---------------------------------------------------------------------------------------
# Ranked strategies
# ---------------------------------------------------------------------------------------


def test_second_strategy_resolves_and_the_trace_names_it() -> None:
    degraded = TargetSpec(
        description="Search button",
        strategies=[
            strategy(StrategyKind.ROLE_NAME, "button|Find"),   # stale after a re-skin
            strategy(StrategyKind.TEXT, "Search"),             # still true
        ],
    )
    cap = artifact(
        steps=[Step(id="submit", intent="Run the search", action="click", target=degraded)],
        outputs=[],
    )
    surface = StubSurface([observation()], {"text:Search": "h-search"})

    result = make_executor(surface).run(cap, {"member_id": "88"})

    assert result.status is ResultStatus.SUCCESS
    trace = result.trace[0]
    assert trace.strategy_used == "text:Search"
    assert trace.strategies_tried == ["role_name:button|Find", "text:Search"]


def test_no_strategy_resolves_is_target_not_found_with_debuggable_detail() -> None:
    surface = StubSurface([observation()], {})
    result = make_executor(surface).run(artifact(), {"member_id": "88"})

    assert result.failure_kind is FailureKind.TARGET_NOT_FOUND
    assert result.failed_step_id == "type_id"
    assert result.expected and result.observed
    assert result.trace[-1].strategies_tried == ["label:Member ID"]


# ---------------------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------------------


def test_recovery_fires_then_the_run_succeeds() -> None:
    interstitial = observation(text="Your session is still active. Continue?")
    surface = StubSurface(
        [interstitial, interstitial, observation()],
        {**RESOLVABLE, "role_name:button|Continue": "h-continue"},
    )
    cap = artifact(
        recovery_rules=[
            RecoveryRule(
                code="SESSION_INTERSTITIAL",
                description="Session keep-alive dialog",
                detect=StateCondition(kind="text_present", value="session is still active"),
                remedy="dismiss",
                dismiss_target=TargetSpec(
                    description="Continue button",
                    strategies=[strategy(StrategyKind.ROLE_NAME, "button|Continue")],
                ),
                max_attempts=2,
                backoff_ms=0,
            )
        ]
    )

    result = make_executor(surface).run(cap, {"member_id": "88"})

    assert result.status is ResultStatus.SUCCESS
    fired = [code for t in result.trace for code in t.recoveries]
    assert "SESSION_INTERSTITIAL" in fired
    # The remedy really acted: the dismiss target was clicked.
    assert any(a.handle == "h-continue" for a in surface.actions)


def test_recovery_beyond_max_attempts_is_recovery_exhausted() -> None:
    stuck = observation(text="Your session is still active. Continue?")
    surface = StubSurface([stuck], RESOLVABLE)
    cap = artifact(
        recovery_rules=[
            RecoveryRule(
                code="SESSION_INTERSTITIAL",
                detect=StateCondition(kind="text_present", value="session is still active"),
                remedy="wait_retry",
                max_attempts=2,
                backoff_ms=0,
            )
        ]
    )

    result = make_executor(surface).run(cap, {"member_id": "88"})

    assert result.status is ResultStatus.HARD_FAILURE
    assert result.failure_kind is FailureKind.RECOVERY_EXHAUSTED
    assert result.trace[-1].recoveries.count("SESSION_INTERSTITIAL") == 2
    assert result.caller_should_retry is True


def test_recovery_budget_is_per_step_not_global() -> None:
    """Two steps each get the rule's full ceiling; a global counter would fail the second."""
    banner = observation(text="Your session is still active. Continue?")
    clean = observation()
    # Each step observes twice (before acting, and after). The banner lands on the
    # post-action look of step 1 and again on the post-action look of step 2, so the same
    # rule has to fire once at each step -- which a global ceiling of 1 would refuse.
    surface = StubSurface(
        [clean, banner, clean, clean, clean, banner, clean, clean, clean], RESOLVABLE
    )
    cap = artifact(
        recovery_rules=[
            RecoveryRule(
                code="SESSION_INTERSTITIAL",
                detect=StateCondition(kind="text_present", value="session is still active"),
                remedy="wait_retry",
                max_attempts=1,
                backoff_ms=0,
            )
        ]
    )

    result = make_executor(surface).run(cap, {"member_id": "88"})
    assert result.status is ResultStatus.SUCCESS
    assert sum(len(t.recoveries) for t in result.trace) >= 2


# ---------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------


def test_policy_blocked_step_stops_the_run() -> None:
    off_allowlist = observation(url="https://evil.example.com/members/search")
    surface = StubSurface([off_allowlist], RESOLVABLE)

    result = make_executor(surface).run(artifact(), {"member_id": "88"})

    assert result.status is ResultStatus.HARD_FAILURE
    assert result.failure_kind is FailureKind.POLICY_BLOCKED
    assert result.failed_step_id == "type_id"
    # Blocked BEFORE acting, which is the only thing that makes the gate worth having.
    assert surface.actions == []


def test_cross_origin_child_frame_blocks_before_action() -> None:
    framed = observation(frame_urls=[SEARCH_URL, "https://evil.example/frame"])
    surface = StubSurface([framed], RESOLVABLE)
    result = make_executor(surface).run(artifact(), {"member_id": "88"})
    assert result.failure_kind is FailureKind.POLICY_BLOCKED
    assert "Child frame" in (result.observed or "")
    assert surface.actions == []


def test_path_prefix_match_respects_segment_boundary() -> None:
    engine = PolicyEngine(
        Policy(allowed_origins=[ORIGIN], allowed_path_prefixes=["/app"])
    )
    assert engine.check_navigation(f"{ORIGIN}/app/member").permitted
    assert not engine.check_navigation(f"{ORIGIN}/application/admin").permitted


# ---------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------


def test_checkpoint_not_met_reports_expected_and_observed() -> None:
    blank = observation(text="", title="Loading", elements=[
        UIElement(handle="h-balance", role="cell", name="Available balance")
    ])
    surface = StubSurface([blank], RESOLVABLE)

    result = make_executor(surface).run(artifact(), {"member_id": "88"})

    assert result.status is ResultStatus.HARD_FAILURE
    assert result.failure_kind is FailureKind.CHECKPOINT_NOT_MET
    assert result.expected
    assert result.observed
    assert "text_present" in result.expected


def test_missing_declared_output_is_a_hard_failure() -> None:
    surface = StubSurface([observation()], RESOLVABLE)
    surface.read_value = None
    result = make_executor(surface).run(artifact(), {"member_id": "88"})
    assert result.failure_kind is FailureKind.INVALID_OUTPUT
    assert "balance" in (result.observed or "")


def test_malformed_declared_output_is_a_hard_failure() -> None:
    cap = artifact(outputs=[
        OutputField(name="balance", from_step="read_balance", type=ParamType.MONEY)
    ])
    surface = StubSurface([observation()], RESOLVABLE)
    surface.read_value = "not-money"
    result = make_executor(surface).run(cap, {"member_id": "88"})
    assert result.failure_kind is FailureKind.INVALID_OUTPUT
    assert "malformed" in (result.observed or "")


# ---------------------------------------------------------------------------------------
# Wait / precondition
# ---------------------------------------------------------------------------------------


def test_wait_for_that_never_holds_is_precondition_failed() -> None:
    surface = StubSurface([observation(text="still loading", elements=[])], RESOLVABLE)
    cap = artifact(
        steps=[
            Step(
                id="submit", intent="Run the search", action="click", target=SEARCH_BUTTON,
                wait_for=StateCondition(kind="text_present", value="Results"),
            )
        ],
        outputs=[],
    )

    result = make_executor(surface).run(cap, {"member_id": "88"})
    assert result.failure_kind is FailureKind.PRECONDITION_FAILED
    assert result.failed_step_id == "submit"


# ---------------------------------------------------------------------------------------
# Tenant overrides
# ---------------------------------------------------------------------------------------


def test_tenant_override_is_applied_without_mutating_the_artifact() -> None:
    from capability_system.artifact.schema import TenantOverride

    other = "https://cu-two.example.com"
    cap = artifact(
        steps=[Step(id="open", intent="Open search", action="navigate",
                    literal=f"{ORIGIN}/members/search")],
        outputs=[],
        tenant_overrides=[TenantOverride(tenant_id="cu-two", origin=other)],
    )
    surface = StubSurface([observation(url=f"{other}/members/search")], RESOLVABLE)
    engine = PolicyEngine(Policy(allowed_origins=[other]))
    ex = ReplayExecutor(surface, engine, tenant_id="cu-two")
    ex.wait_timeout_s, ex.poll_interval_s = 0.05, 0.0

    result = ex.run(cap, {"member_id": "88"})

    assert result.status is ResultStatus.SUCCESS
    assert surface.actions[0].text == f"{other}/members/search"
    # The caller's artifact is untouched, so the next tenant's run is unaffected.
    assert cap.steps[0].literal == f"{ORIGIN}/members/search"


# ---------------------------------------------------------------------------------------
# Recorder is duck-typed and optional
# ---------------------------------------------------------------------------------------


def test_recorder_is_called_and_a_broken_recorder_cannot_break_the_run() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.steps: list[str] = []

        def event(self, kind: str, **fields: Any) -> None:
            self.events.append(kind)

        def step(self, index: int, step_id: str, **fields: Any) -> None:
            self.steps.append(step_id)

    recorder = Recorder()
    surface = StubSurface([observation()], RESOLVABLE)
    result = make_executor(surface, recorder=recorder).run(artifact(), {"member_id": "88"})
    assert result.status is ResultStatus.SUCCESS
    assert recorder.steps == ["type_id", "submit", "read_balance"]
    assert "replay_started" in recorder.events

    class BrokenRecorder:
        def event(self, kind: str, **fields: Any) -> None:
            raise RuntimeError("evidence store is down")

        def step(self, index: int, step_id: str, **fields: Any) -> None:
            raise RuntimeError("evidence store is down")

    surface2 = StubSurface([observation()], RESOLVABLE)
    result2 = make_executor(surface2, recorder=BrokenRecorder()).run(
        artifact(), {"member_id": "88"}
    )
    assert result2.status is ResultStatus.SUCCESS


# ---------------------------------------------------------------------------------------
# The constraint the whole module exists to honour
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [executor_module, __import__("capability_system.replay.conditions", fromlist=["x"])]
)
def test_replay_path_contains_no_llm_client(module: Any) -> None:
    """Replay must be deterministic, so no model client may appear on this path at all.

    Asserted against the source text rather than the import graph because the failure mode
    is someone adding a "just for tricky cases" fallback, and that arrives as an import
    long before it arrives as a call.
    """
    source = inspect.getsource(module)
    banned = ["openai", "anthropic", "httpx", "litellm", "langchain", "requests.post", "Client("]
    code_lines = [
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    ]
    # Strip the module docstring, which legitimately discusses models in prose.
    body = "\n".join(code_lines).split('"""', 2)[-1].lower()
    for name in banned:
        assert name.lower() not in body, f"replay path must not reference {name!r}"


# ---------------------------------------------------------------------------------------
# Escalation: the path that turns "stuck" into "a human has it"
# ---------------------------------------------------------------------------------------


class _Broker:
    """In-memory `EscalationBroker`. `auto_resolve` stands in for an operator acting."""

    def __init__(self, auto_resolve: bool = True, on_resolve: Any = None,
                 resolution: Any = None, outputs: dict[str, Any] | None = None) -> None:
        self.requests: dict[str, Any] = {}
        self.auto_resolve = auto_resolve
        self.on_resolve = on_resolve
        self.resolution = resolution
        self.outputs = outputs or {}

    def publish(self, request: Any) -> None:
        self.requests[request.id] = request

    def poll(self, intervention_id: str) -> Any:
        request = self.requests.get(intervention_id)
        if request is not None and self.auto_resolve:
            self.resolve(intervention_id, "handled", self.resolution, self.outputs)
        return self.requests.get(intervention_id)

    def resolve(self, intervention_id: str, note: str = "", resolution: Any = None,
                outputs: dict[str, Any] | None = None) -> None:
        from capability_system.escalation.control import (
            InterventionResolution, InterventionStatus,
        )

        request = self.requests[intervention_id]
        if request.status is not InterventionStatus.RESOLVED and self.on_resolve is not None:
            self.on_resolve()
        request.status = InterventionStatus.RESOLVED
        request.resolution = resolution or InterventionResolution.UNBLOCKED
        request.operator_outputs = dict(outputs or {})
        request.operator_note = note


def _controller(surface: Any, broker: _Broker) -> Any:
    from capability_system.escalation.control import SessionController

    return SessionController("sess-1", broker, surface.observe, poll_interval_s=0.0)


def test_target_not_found_escalates_and_retries_on_the_humans_fresh_state() -> None:
    surface = StubSurface([observation()], {})

    def operator_fixes_the_screen() -> None:
        surface.resolvable.update(RESOLVABLE)

    broker = _Broker(auto_resolve=True, on_resolve=operator_fixes_the_screen)
    ex = make_executor(surface, controller=_controller(surface, broker))

    result = ex.run(artifact(), {"member_id": "88"})

    assert result.status is ResultStatus.SUCCESS
    published = list(broker.requests.values())
    assert published and published[0].reason.value == "target_not_found"
    # The request carries enough for an operator to act without asking anyone.
    assert published[0].goal and published[0].step_intent and published[0].suggested_action
    assert "Available balance" not in published[0].observation_digest
    # Control was handed over and taken back, with a fresh observation on reclaim.
    assert ex.controller is not None and ex.controller.handoffs


def test_escalation_timeout_returns_escalated_not_a_failure() -> None:
    surface = StubSurface([observation()], {})
    broker = _Broker(auto_resolve=False)
    ex = make_executor(surface, controller=_controller(surface, broker))
    ex.escalation_timeout_s = 0.01

    result = ex.run(artifact(), {"member_id": "88"})

    assert result.status is ResultStatus.ESCALATED
    assert result.failure_kind is None
    assert result.escalation_id and result.escalation_id.startswith("iv_")
    assert result.failed_step_id == "type_id"


def test_file_broker_rejects_path_traversal(tmp_path) -> None:
    from capability_system.escalation.control import FileBroker

    broker = FileBroker(tmp_path / "interventions")
    with pytest.raises(ValueError, match="invalid intervention id"):
        broker.poll("../../outside")


def test_operator_performed_read_must_supply_declared_output() -> None:
    from capability_system.escalation.control import InterventionResolution

    cap = artifact()
    cap.steps[-1].risk = RiskClass.RISKY_IRREVERSIBLE
    surface = StubSurface([observation()], RESOLVABLE)
    broker = _Broker(resolution=InterventionResolution.PERFORMED)
    result = make_executor(surface, controller=_controller(surface, broker)).run(
        cap, {"member_id": "88"}
    )
    assert result.failure_kind is FailureKind.INVALID_OUTPUT


def test_operator_performed_read_can_return_declared_output() -> None:
    from capability_system.escalation.control import InterventionResolution

    cap = artifact()
    cap.steps[-1].risk = RiskClass.RISKY_IRREVERSIBLE
    surface = StubSurface([observation()], RESOLVABLE)
    broker = _Broker(
        resolution=InterventionResolution.PERFORMED,
        outputs={"balance": "9,876.54"},
    )
    result = make_executor(surface, controller=_controller(surface, broker)).run(
        cap, {"member_id": "88"}
    )
    assert result.status is ResultStatus.SUCCESS
    assert result.outputs == {"balance": "9,876.54"}
