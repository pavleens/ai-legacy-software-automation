"""Discovery: an LLM drives a real surface once, and the run is compiled into a capability.

TWO PHASES, AND THE SECOND ONE IS THE PROJECT.

Phase one is an ordinary observe / decide / act loop. It is not where the value is. Any
competent implementation of that loop will complete a three-screen bank flow, and a
transcript of it is worthless to a calling agent because it contains ephemeral addresses,
one hard-coded set of inputs, and no contract.

Phase two compiles the successful trajectory into a `CapabilityArtifact`. That is where the
judgment lives, and it turns on one decision:

    THE AGENT ACTS ON EPHEMERAL REFERENCES. THE ARTIFACT STORES SEMANTIC TARGETS.

Playwright's `aria_snapshot(mode="ai")` assigns refs like `e5` to interactable elements,
and they are reassigned on every snapshot. They are exactly right for live discovery and
catastrophic in a stored artifact -- an artifact addressing `e5` would replay against
whatever happened to be fifth next Tuesday. So at compile time every acted-upon ref is
translated into an ordered ladder of stable strategies (role+name, then visible label, then
container-scoped role+name, then exact text, then ordinal). `Surface.stable_target()` owns
that translation; this module owns knowing it must happen.

PARAMETERISATION IS DETERMINISTIC, NOT INFERRED. The caller supplies the goal together
with the concrete values used for the discovery run. Any step whose payload equals one of
those values is bound to that parameter. No second model call, no guessing which literal
was "really" an argument -- a compiler that asks an LLM which of its own constants were
variables is a compiler that produces a different contract each time you run it.

PRIOR ART. The two-phase shape is not novel and the write-up says so. Skyvern's code
caching (AGPL-3.0) and browser-use's workflow-use (AGPL-3.0) both ship record-then-replay,
and PreAct (arXiv 2606.17929) formalises compile-run-to-program with precondition checks
and demotion back to the agent on deviation. Neither project is depended on here: both are
AGPL, and workflow-use's artifact is a bare Pydantic union with `extra="allow"`, which
means it is not actually typed. The contribution here is the artifact contract and the
ref-to-semantic-target compiler, not the loop.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from capability_system.artifact.schema import (
    ApprovalState,
    CapabilityArtifact,
    InputParam,
    OutputField,
    ParamType,
    RiskClass,
    StateCondition,
    Step,
    SurfaceDescriptor,
    TargetSpec,
)
from capability_system.discovery.provider import ModelError, ModelProvider, decode
from capability_system.escalation.control import detect_no_progress
from capability_system.perception.base import Action, ActionKind, Observation
from capability_system.safety.policy import PolicyEngine, redact

SYSTEM_PROMPT = """You operate a legacy back-office banking application by driving its \
user interface, the way a human operator would. You cannot call an API. You see only the \
controls listed to you, addressed by the handle in square brackets.

Rules:
- Choose exactly ONE next action per turn.
- Only ever use a handle that appears in the ELEMENTS list of the CURRENT observation.
- Prefer the smallest action that makes progress. Do not skip ahead.
- Classify every action's risk. An action is risky_irreversible if it commits a change \
that cannot be undone by navigating away: submitting, confirming, transferring, posting, \
authorising, deleting. Searching, reading and navigating are safe_reversible.
- If the goal asks you to read, find or report a VALUE, you must first use action \
"read" with extract_as set to a short snake_case name for that value. Only then answer "done".
- When the goal is achieved, answer with action "done".
- If you cannot make progress with the controls available, answer with action "stuck" and \
explain what you needed and could not find. Do not guess a handle that is not listed.

Reply with ONE JSON object and nothing else."""


class AgentDecision(BaseModel):
    """The model's move. Kept deliberately small.

    A narrow action vocabulary is what makes the recorded artifact auditable: a reviewer
    approving a capability for unattended use against a bank has to be able to read every
    verb it can perform, and six is readable.
    """

    # Explanatory, not load-bearing, so it is optional. Measured: gemma4 returns a bare
    # {"action": "done"} when it believes it is finished, and rejecting a correct decision
    # because the prose was omitted would be the validator failing, not the model.
    reasoning: str = Field(default="", description="Why this action, in one or two sentences.")
    action: Literal["navigate", "click", "type", "select", "read", "done", "stuck"]
    handle: str | None = Field(default=None, description="Handle from the current ELEMENTS list.")
    text: str | None = Field(default=None, description="Text to type, or URL to navigate to.")
    option: str | None = Field(default=None, description="Option label for select.")
    risk: Literal["safe_reversible", "risky_irreversible"] = "safe_reversible"
    extract_as: str | None = Field(
        default=None, description="For read: the output field name this value becomes."
    )


class DiscoveryFailed(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class _Trace(BaseModel):
    """One acted step, retained so phase two can compile it."""

    index: int
    decision: AgentDecision
    handle: str | None
    payload: str | None
    target: TargetSpec | None
    observed_url: str | None


class DiscoveryAgent:
    def __init__(
        self,
        surface: Any,
        provider: ModelProvider,
        policy: PolicyEngine,
        recorder: Any = None,
        max_steps: int = 25,
    ) -> None:
        self.surface = surface
        self.provider = provider
        self.policy = policy
        self.recorder = recorder
        self.max_steps = max_steps

    # -- phase one ---------------------------------------------------------------------

    def discover(
        self,
        goal: str,
        start_url: str,
        capability_id: str,
        inputs: dict[str, str],
        product: str,
        sensitive: Sequence[str] = (),
    ) -> CapabilityArtifact:
        """Drive the surface until the goal is met, then compile what happened.

        `inputs` are the concrete values this run uses. They are the parameterisation key,
        so they must be the real values the model will type.
        """
        verdict = self.policy.check_navigation(start_url)
        if not verdict.permitted:
            raise DiscoveryFailed("policy_blocked", verdict.reason)

        self.surface.act(Action(kind=ActionKind.NAVIGATE, text=start_url))
        traces: list[_Trace] = []
        digests: list[str] = []
        outputs: dict[str, str] = {}
        self._last_action_mutating = True  # the initial navigate counts
        self._last_signature: tuple[str, str | None] | None = None

        for index in range(1, self.max_steps + 1):
            obs: Observation = self.surface.observe()
            # Progress is measured from CONTENT, and only after a MUTATING action.
            #
            # A `read` is not supposed to change anything, so counting it would make two
            # honest consecutive reads indistinguishable from a stuck loop -- which is
            # exactly what happened on the first working run: the model read the balance,
            # read it again, and the detector killed a run that was one step from done.
            # Stuck-detection must only be asked about actions that were meant to move.
            # Progress is measured from CONTENT, not from the URL.
            #
            # On a frameset the top-level URL never changes -- child frames navigate
            # underneath it -- so a url+count digest reports "no progress" through an
            # entire successful flow. Hashing the flattened visible text is what actually
            # distinguishes the search screen from the member detail screen here, and it
            # degrades correctly on a modern SPA for the same reason.
            if self._last_action_mutating:
                digests.append(_digest(obs))

            # Cheap, token-free stuck detection. A model that is looping will keep
            # sounding confident, so progress is measured from the surface, not the prose.
            if detect_no_progress(digests):
                raise DiscoveryFailed("no_progress", f"surface unchanged for 3 observations at {obs.url}")

            decision, raw = self._decide(goal, obs, index, inputs, outputs)
            self._record(index, obs, decision, raw)

            if decision.action == "done":
                if not traces:
                    raise DiscoveryFailed("empty_run", "model declared done without acting")
                return self._compile(
                    goal=goal, capability_id=capability_id, start_url=start_url,
                    inputs=inputs, outputs=outputs, traces=traces, product=product,
                    final=self.surface.observe(), sensitive=set(sensitive),
                )

            if decision.action == "stuck":
                raise DiscoveryFailed("model_stuck", decision.reasoning)

            # Model-independent loop guard. Prompting is advice; this is a fact about the
            # trajectory. An identical (action, handle) repeated is by definition not
            # progress, and on a `read` it cannot become progress no matter how many times
            # it runs. Cheaper and more reliable than trusting the model to notice.
            signature = (decision.action, decision.handle)
            if signature == self._last_signature:
                raise DiscoveryFailed(
                    "repeated_action",
                    f"model repeated {decision.action} on {decision.handle} with no state change",
                )
            self._last_signature = signature

            action, payload = self._to_action(decision, obs)
            risk = RiskClass(decision.risk)
            verdict = self.policy.check_action(
                action, risk=risk, current_url=obs.url, surface_urls=obs.frame_urls
            )
            if not verdict.permitted:
                # Discovery refuses rather than asking a human: a run that needed a person
                # to authorise an irreversible step has not produced an artifact anyone
                # should trust for unattended replay.
                raise DiscoveryFailed("policy_blocked", verdict.reason)

            # COMPILE THE STABLE TARGET BEFORE ACTING. This ordering is not stylistic.
            #
            # Refs are ephemeral AND positional: measured on this surface, `f2e19` is the
            # Member ID *textbox* on the search screen and a Member ID *table cell* on the
            # member detail screen. Acting navigates, refs are reassigned, and resolving the
            # pre-action ref against the post-action page silently yields a DIFFERENT
            # element without raising -- which is the worst failure shape available, because
            # the artifact records a plausible-looking target for the wrong control.
            #
            # An earlier revision of this method compiled after `act()` and produced exactly
            # that: a step whose intent read "click the Search button" carrying a recorded
            # target of `cell|Joined`. Nothing errored. The bug is recorded here because the
            # hazard is the entire reason the artifact stores semantic targets rather than
            # refs, and it still bit the implementation.
            target = self._stable_target(decision.handle)

            result = self.surface.act(action)
            if not result.ok:
                raise DiscoveryFailed("action_failed", result.error or "surface rejected the action")

            self._last_action_mutating = decision.action != "read"
            if decision.action == "read" and decision.extract_as:
                outputs[decision.extract_as] = result.read_value or ""

            traces.append(_Trace(
                index=index,
                decision=decision,
                handle=decision.handle,
                payload=payload,
                target=target,
                observed_url=obs.url,
            ))

        raise DiscoveryFailed("step_budget_exhausted", f"goal not reached in {self.max_steps} steps")

    # -- helpers -----------------------------------------------------------------------

    def _decide(
        self, goal: str, obs: Observation, index: int, inputs: dict[str, str],
        extracted: dict[str, str] | None = None,
    ) -> tuple[AgentDecision, str]:
        known = ", ".join(f"{k}={v!r}" for k, v in inputs.items()) or "none"
        # Without this the model has no evidence its read landed, so it reads again. The
        # loop is only as closed as the feedback it gives back.
        got = ", ".join(f"{k}={v!r}" for k, v in (extracted or {}).items()) or "none yet"
        # A pointed instruction beats a polite one. Measured: gemma4:31b re-read the same
        # cell ten times in a row while the extracted value sat in its context, because
        # "values already extracted" reads as status, not as a directive.
        settled = (
            f"\nYou have ALREADY extracted: {', '.join(sorted(extracted))}. "
            f"Do NOT read those again. If the goal is satisfied, answer with action \"done\".\n"
            if extracted else ""
        )
        rendered = obs.describe()
        user = (
            f"GOAL: {goal}\n"
            f"VALUES YOU MAY USE: {known}\n"
            f"VALUES ALREADY EXTRACTED: {got}\n"
            f"{settled}"
            f"STEP: {index} of at most {self.max_steps}\n\n"
            f"{rendered}\n\n"
            f"What is the next action?"
        )
        try:
            decision, raw = decode(self.provider, AgentDecision, SYSTEM_PROMPT, user)
        except ModelError as exc:
            raise DiscoveryFailed("model_error", str(exc)) from exc
        assert isinstance(decision, AgentDecision)

        # Handles are rendered as "[handle]" for readability, and models routinely copy the
        # brackets back. Measured with gemma4:31b, which returned "[f2|aria-ref=f2e27]" for a
        # handle of "f2|aria-ref=f2e27" -- the right element, the wrong envelope. Normalising
        # is strictly better than tightening the prompt: the prompt is advice, this is a fact
        # about the string. Same posture as the key-alias table in provider.py.
        if decision.handle:
            decision.handle = decision.handle.strip().strip("[]").strip()

        # Handles are composite ("f2|aria-ref=f2e48") and models routinely return only the
        # distinctive tail ("f2e48"), dropping the frame scope. Measured across runs with
        # gemma4:31b: it emits the full handle most of the time and the bare ref sometimes,
        # which makes this nondeterminism in the envelope rather than in the decision.
        # Resolve a partial handle to the unique live handle containing it -- and only if
        # it is unique, because an ambiguous partial is a genuine hallucination risk and
        # must not be guessed at.
        if decision.handle and decision.handle not in _shown_handles(rendered):
            matches = [h for h in _shown_handles(rendered) if h.endswith(decision.handle)]
            if len(matches) == 1:
                decision.handle = matches[0]

        # A hallucinated handle is the most common REAL failure and the cheapest to catch.
        # Checked only after normalisation, so formatting noise never masquerades as one.
        if decision.action in {"click", "type", "select", "read"}:
            # Validate against what the model was actually SHOWN, not against every element
            # that exists. Checking the full set lets a guessed handle pass validation
            # merely because it happens to be real but was never rendered into the prompt,
            # which is how a hallucination becomes an action instead of an error.
            live = _shown_handles(rendered)
            if decision.handle not in live:
                raise DiscoveryFailed(
                    "invalid_handle",
                    f"model chose {decision.handle!r}, which is not in the current observation",
                )
        return decision, raw

    def _to_action(self, decision: AgentDecision, obs: Observation) -> tuple[Action, str | None]:
        kind = ActionKind(decision.action)
        payload = decision.text if kind in {ActionKind.TYPE, ActionKind.NAVIGATE} else decision.option
        return (
            Action(kind=kind, handle=decision.handle, text=decision.text, option=decision.option),
            payload,
        )

    def _stable_target(self, handle: str | None) -> TargetSpec | None:
        """Translate an ephemeral ref into durable strategies. The crux of phase two."""
        if handle is None:
            return None
        strategies = self.surface.stable_target(handle)
        if not strategies:
            raise DiscoveryFailed(
                "uncompilable_step",
                f"no stable target could be derived for {handle!r}; the artifact would not replay",
            )
        return TargetSpec(description=strategies[0].value, strategies=list(strategies))

    def _record(self, index: int, obs: Observation, decision: AgentDecision, raw: str) -> None:
        if self.recorder is None:
            return
        try:
            shot = (
                self.surface.screenshot()
                if getattr(self.recorder, "capture_screenshots", False)
                else None
            )
        except Exception:
            shot = None
        self.recorder.step(
            index, f"step-{index:03d}", observation=obs, action=decision,
            reasoning=decision.reasoning, screenshot_bytes=shot,
        )
        self.recorder.event("decision", step=index, action=decision.action,
                            risk=decision.risk, raw_len=len(raw))

    # -- phase two ---------------------------------------------------------------------

    def _compile(
        self,
        *,
        goal: str,
        capability_id: str,
        start_url: str,
        inputs: dict[str, str],
        outputs: dict[str, str],
        traces: list[_Trace],
        product: str,
        final: Observation,
        sensitive: set[str],
    ) -> CapabilityArtifact:
        """Turn the trajectory into a contract.

        Emitted as DRAFT, always. A capability recorded by a model on one happy path has
        not earned unattended execution against an institution's systems; a human promotes
        it to APPROVED after reading it. That gate is enforced in the replay executor.
        """
        by_value = {v: k for k, v in inputs.items() if v}
        sensitive_values = {
            str(inputs[name]) for name in sensitive if name in inputs and inputs[name]
        }
        safe_goal = redact(goal, sensitive_values)

        # A CAPABILITY MUST BE SELF-CONTAINED. Discovery navigates to the start URL as
        # setup, before the loop, so that navigation is not in the trajectory. Compiling
        # only the trajectory produced an artifact that replayed from about:blank and was
        # blocked by the allowlist at step one -- correct behaviour by the executor, and a
        # capability that could only ever run if something else had already opened the app.
        # The entry point is part of the contract, so it is synthesised here.
        steps: list[Step] = [Step(
            id="s000",
            intent="Open the application at its recorded entry point.",
            action="navigate",
            literal=start_url,
            risk=RiskClass.SAFE_REVERSIBLE,
        )]

        for trace in traces:
            decision = trace.decision
            binds = by_value.get(trace.payload or "")
            steps.append(Step(
                id=f"s{trace.index:03d}",
                intent=redact(decision.reasoning.strip(), sensitive_values)[:280],
                action=decision.action,
                target=trace.target,
                # A payload that matched a supplied value becomes a binding; anything else
                # is a genuine constant of the flow and stays literal.
                literal=None if binds else trace.payload,
                binds_input=binds,
                reads_into=decision.extract_as if decision.action == "read" else None,
                risk=RiskClass(decision.risk),
            ))

        origin = _origin_of(start_url)
        return CapabilityArtifact(
            id=capability_id,
            name=capability_id.replace("_", " ").title(),
            description=safe_goal,
            surface=SurfaceDescriptor(kind="web", product=product, recorded_origin=origin),
            approval=ApprovalState.DRAFT,
            inputs=[
                InputParam(
                    name=name, type=ParamType.STRING, required=True,
                    example=None if name in sensitive else value,
                    sensitive=name in sensitive,
                    description="Caller-supplied capability input.",
                )
                for name, value in inputs.items()
            ],
            outputs=[
                OutputField(
                    name=name, type=ParamType.STRING,
                    description=f"Read from the surface during discovery.",
                    from_step=next(
                        (s.id for s in steps if s.reads_into == name), steps[-1].id
                    ),
                )
                for name in outputs
            ],
            steps=steps,
            # The checkpoint is derived from the end state rather than asked of the model.
            # A success condition the model invents is a success condition the model can
            # also satisfy by accident.
            checkpoint=_checkpoint_for(steps, final, origin),
            allowed_origins=[origin] if origin else [],
            discovered_by=getattr(self.provider, "name", "unknown"),
        )


def _shown_handles(rendered: str) -> set[str]:
    """Handles actually present in the prompt text the model received."""
    import re
    return set(re.findall(r"\[([^\]\s]+)\]", rendered))


def _digest(obs: Observation) -> str:
    """Content fingerprint of an observation, used only for stuck detection."""
    import hashlib
    payload = f"{obs.url}|{obs.title}|{len(obs.elements)}|{obs.text_digest}"
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()


def _checkpoint_for(steps: list[Step], final: Observation, origin: str) -> StateCondition:
    """Derive the success condition from where the run ended up.

    URL is the obvious choice and the wrong one on this class of surface: a frameset
    navigates its child frames while the top-level URL never changes, so a url_matches
    checkpoint is trivially true on every replay and verifies nothing. A checkpoint that
    cannot fail is worse than no checkpoint, because it reports success.

    So prefer the presence of the control the run finished on. It is the one thing known
    to exist in the success state, it was proved resolvable at record time by
    stable_target(), and it actually distinguishes the end screen from the search screen.
    """
    for step in reversed(steps):
        if step.target is not None:
            return StateCondition(kind="element_present", value=step.target.description,
                                  target=step.target)
    return StateCondition(kind="url_matches", value=_escape(final.url or origin))


def _origin_of(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""


def _escape(value: str) -> str:
    import re
    return re.escape(value)
