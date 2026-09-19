"""Human-in-the-loop escalation and control transfer.

THE HARD PART IS NOT DETECTING "STUCK". It is that the human must operate the *same live
session* the automation was using, then hand it back, and both sides must agree at every
moment about who holds control. That is a lease problem, not a UI problem, which is why
the lease lives here and the operator console is deliberately mocked.

THE MECHANISM, AND WHY IT IS HONEST RATHER THAN CLEVER. The automation and the human share
one browser context. Ceding control does not tear anything down and does not open a second
window; the automation simply stops issuing actions while the lease is held by the human,
and the human drives the window that is already open. Locally that means running headed and
letting a person use the visible window. The seam generalises without changing this file:

    local dev      headed browser, the operator uses the window directly
    hosted         the same context exposed over CDP or a co-browsing stream
    desktop        the same OS session surfaced through a remote-control channel

In every case the automation's obligation is identical, which is the point of putting it
behind a lease: stop acting, stay alive, and do not assume anything about the world when
you get the session back.

THE RULE THAT PREVENTS THE WORST BUG. On reclaim the automation MUST re-observe before it
acts. A human who resolves a stuck state has by definition changed the page, and possibly
navigated somewhere entirely different. Resuming against a cached observation is how an
automation clicks "Confirm" on a screen it has never actually seen. `reclaim()` therefore
returns a fresh Observation and the executor is written to require it.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from capability_system.perception.base import Observation


def _safe_url(url: str) -> str:
    """Keep routing context while dropping query parameters and fragments that carry IDs."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


_INTERVENTION_ID = re.compile(r"^iv_[a-f0-9]{12}$")


class ControlOwner(str, Enum):
    AUTOMATION = "automation"
    HUMAN = "human"


class StuckReason(str, Enum):
    """Why the system stopped. Each maps to a different thing the operator must do.

    Routing a request without this is how intervention queues become undifferentiated
    noise: the operator cannot triage, so everything waits the same amount of time.
    """

    TARGET_NOT_FOUND = "target_not_found"                # no strategy resolved
    POLICY_REQUIRES_CONFIRMATION = "policy_requires_confirmation"  # irreversible step
    RECOVERY_EXHAUSTED = "recovery_exhausted"            # bounded retries used up
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"      # discovery ran out of moves
    NO_PROGRESS = "no_progress"                          # state stopped changing
    SURFACE_ERROR = "surface_error"                      # the app itself broke
    UNKNOWN_STATE = "unknown_state"                      # observed something undeclared


class InterventionResolution(str, Enum):
    """How the operator disposed of the request. Determines what the automation does next.

    This distinction is load-bearing and its absence is a bug I shipped and then hit.
    Without it the executor could only ever RETRY the step it escalated on, so a human who
    took the session specifically to perform an irreversible action themselves handed back
    to an automation that immediately tried to perform it again -- and was correctly
    refused by policy a second time, ending a successful handoff in a hard failure.

    "I did it" and "I cleared what was blocking you" are different instructions, and only
    the operator knows which one happened.
    """

    PERFORMED = "performed"    # the human completed this step; skip it and carry on
    UNBLOCKED = "unblocked"    # the obstacle is cleared; retry the step as recorded
    ABORT = "abort"            # do not continue; end the run as escalated


class InterventionStatus(str, Enum):
    OPEN = "open"
    CLAIMED = "claimed"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class InterventionRequest(BaseModel):
    """What the operator receives. Must carry enough context to act without a conversation.

    "Carrying enough context" is a design requirement, not a nicety. An operator who has to
    ask what the automation was trying to do will resolve the session by starting over,
    which destroys the evidence and the session state the handoff existed to preserve.
    """

    id: str = Field(default_factory=lambda: f"iv_{uuid.uuid4().hex[:12]}")
    session_id: str
    capability_id: str | None = None
    capability_version: int | None = None

    goal: str = Field(description="What the run was trying to achieve, in plain language.")
    step_id: str | None = None
    step_intent: str | None = Field(default=None, description="What this step was for.")
    reason: StuckReason
    detail: str = Field(default="", description="Expected versus observed, concretely.")

    current_url: str | None = None
    observation_digest: str = Field(
        default="", description="Trimmed view of the surface at the moment of the stop."
    )
    screenshot_path: str | None = None
    evidence_dir: str | None = None

    suggested_action: str = Field(
        default="", description="What the system believes a human should do next."
    )
    status: InterventionStatus = InterventionStatus.OPEN
    resolution: InterventionResolution | None = Field(
        default=None, description="Set by the operator on resolve. Drives resume behaviour."
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    operator_note: str = ""
    operator_outputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Declared step outputs supplied by the operator when resolution=performed.",
    )


class HandoffRecord(BaseModel):
    """What the human did while holding the lease.

    Reconstructed by diffing the surface across the handoff rather than by instrumenting
    the operator. Instrumenting the human would mean intercepting their input, which on a
    real co-browsing session is both invasive and unreliable. A before/after diff is
    weaker evidence but it is honest evidence, and it is enough to answer the question an
    auditor actually asks: what changed while a person was in control?
    """

    intervention_id: str
    held_from: datetime
    held_until: datetime
    duration_ms: int
    url_before: str | None = None
    url_after: str | None = None
    navigated: bool = False
    elements_before: int = 0
    elements_after: int = 0
    title_before: str = ""
    title_after: str = ""
    operator_note: str = ""


class SessionLease(BaseModel):
    """Single source of truth for who may act on the session."""

    session_id: str
    owner: ControlOwner = ControlOwner.AUTOMATION
    since: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intervention_id: str | None = None


class EscalationBroker(Protocol):
    """How an intervention request reaches a human and how resume is signalled.

    Deliberately narrow. A production broker is a queue plus an operator console; this
    interface is the seam between them and everything above, so swapping one for the
    other touches no other file.
    """

    def publish(self, request: InterventionRequest) -> None: ...

    def poll(self, intervention_id: str) -> InterventionRequest | None: ...

    def resolve(self, intervention_id: str, note: str = "",
                resolution: "InterventionResolution" = ...,
                outputs: dict[str, Any] | None = None) -> None: ...


class FileBroker:
    """A deliberately mocked operator surface, backed by a directory.

    MOCKED ON PURPOSE, AND DOCUMENTED AS SUCH. The brief puts a real-time co-browsing
    operator console out of scope and asks instead for a real handoff mechanism and a
    well-reasoned control-transfer model. So the transport is a JSON file per request that
    a human (or a test) resolves by writing a sentinel. The lease semantics above are
    real; only the delivery channel is stubbed, and it is stubbed at an interface that a
    queue would satisfy unchanged.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def _path(self, intervention_id: str) -> Path:
        if not _INTERVENTION_ID.fullmatch(intervention_id):
            raise ValueError("invalid intervention id")
        return self.root / f"{intervention_id}.json"

    def publish(self, request: InterventionRequest) -> None:
        path = self._path(request.id)
        path.write_text(
            json.dumps(request.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        path.chmod(0o600)

    def poll(self, intervention_id: str) -> InterventionRequest | None:
        path = self._path(intervention_id)
        if not path.exists():
            return None
        return InterventionRequest.model_validate_json(path.read_text(encoding="utf-8"))

    def resolve(
        self,
        intervention_id: str,
        note: str = "",
        resolution: InterventionResolution = InterventionResolution.UNBLOCKED,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        request = self.poll(intervention_id)
        if request is None:
            raise KeyError(intervention_id)
        request.status = InterventionStatus.RESOLVED
        request.resolution = resolution
        request.resolved_at = datetime.now(timezone.utc)
        request.operator_note = note
        request.operator_outputs = dict(outputs or {})
        self.publish(request)


class ControlTransferError(RuntimeError):
    """Raised when an actor tries to act without holding the lease."""


class SessionController:
    """Owns the lease. Every actor asks this before touching the surface."""

    def __init__(
        self,
        session_id: str,
        broker: EscalationBroker,
        observe: Callable[[], Observation],
        poll_interval_s: float = 1.0,
        include_observation_details: bool = False,
    ) -> None:
        self.lease = SessionLease(session_id=session_id)
        self.broker = broker
        self._observe = observe
        self._poll_interval_s = poll_interval_s
        self._include_observation_details = include_observation_details
        self.handoffs: list[HandoffRecord] = []

    # -- invariant --------------------------------------------------------------------

    def assert_automation_may_act(self) -> None:
        """Called before every automated action. The lease is not advisory."""
        if self.lease.owner is not ControlOwner.AUTOMATION:
            raise ControlTransferError(
                f"Session {self.lease.session_id} is held by {self.lease.owner.value}; "
                "automation must not act until control is reclaimed."
            )

    # -- transfer ---------------------------------------------------------------------

    def cede(self, request: InterventionRequest) -> InterventionRequest:
        """Pause automation, publish the request, and hand the lease to a human.

        The browser context is untouched. Nothing is closed, nothing is reloaded, and the
        page the operator sees is the page the automation was looking at.
        """
        self.assert_automation_may_act()
        before = self._observe()
        request.session_id = self.lease.session_id
        request.current_url = before.url if self._include_observation_details else _safe_url(before.url)
        request.observation_digest = (
            before.describe(limit=25)
            if self._include_observation_details
            else f"URL: {_safe_url(before.url)}\nELEMENT COUNT: {len(before.elements)}"
        )
        self.broker.publish(request)

        self.lease = SessionLease(
            session_id=self.lease.session_id,
            owner=ControlOwner.HUMAN,
            intervention_id=request.id,
        )
        self._pending_before = before
        self._held_from = datetime.now(timezone.utc)
        return request

    def await_resume(self, intervention_id: str, timeout_s: float = 300.0) -> bool:
        """Block until the operator marks the request resolved, or time out.

        Timing out does not silently resume. The caller gets False and is expected to end
        the run as ESCALATED, because an unattended automation quietly picking up a session
        a human abandoned mid-edit is exactly the behaviour that makes these systems unsafe.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            request = self.broker.poll(intervention_id)
            if request is not None and request.status is InterventionStatus.RESOLVED:
                return True
            time.sleep(self._poll_interval_s)
        return False

    def reclaim(self, operator_note: str = "") -> tuple[Observation, HandoffRecord]:
        """Take the lease back and re-observe.

        Returns a FRESH observation. Callers are written to use this return value rather
        than anything they held before the handoff; see the module docstring.
        """
        if self.lease.owner is not ControlOwner.HUMAN:
            raise ControlTransferError("Cannot reclaim a session that automation already holds.")

        after = self._observe()
        held_until = datetime.now(timezone.utc)
        before = self._pending_before
        record = HandoffRecord(
            intervention_id=self.lease.intervention_id or "",
            held_from=self._held_from,
            held_until=held_until,
            duration_ms=int((held_until - self._held_from).total_seconds() * 1000),
            url_before=before.url,
            url_after=after.url,
            navigated=before.url != after.url,
            elements_before=len(before.elements),
            elements_after=len(after.elements),
            title_before=before.title,
            title_after=after.title,
            operator_note=operator_note,
        )
        self.handoffs.append(record)
        self.lease = SessionLease(
            session_id=self.lease.session_id, owner=ControlOwner.AUTOMATION
        )
        return after, record


def detect_no_progress(history: list[str], window: int = 3) -> bool:
    """Stuck-detection that does not depend on the model noticing it is stuck.

    If the surface digest has been identical for `window` consecutive observations, the
    loop is not making progress regardless of how confident the model sounds. This is the
    cheapest reliable stuck signal available and it costs no tokens.
    """
    if len(history) < window:
        return False
    return len(set(history[-window:])) == 1
