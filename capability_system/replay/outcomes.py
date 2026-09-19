"""The replay result contract.

The brief asks replay to distinguish three things, and getting this taxonomy right matters
more than any other error-handling code in the system:

    BusinessOutcome     A correct answer the caller must act on. "No such member" is not a
                        bug; it is the answer. Collapsing this into failure is how an
                        automation platform generates alarms nobody reads.

    Recoverable         A condition the system is pre-authorised to handle and retry within
                        a declared ceiling. An interstitial, a transient slow load, a stale
                        session banner. Invisible to the caller except as evidence.

    HardFailure         Something the system cannot safely proceed through. Always carries
                        what step, what was expected, and what was actually observed,
                        because a failure a human cannot debug is barely better than a hang.

A fourth state exists and is not an error either: ESCALATED. Replay stopped and handed the
live session to a human. The caller needs to know the work is in flight, not lost.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResultStatus(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    ESCALATED = "escalated"
    HARD_FAILURE = "hard_failure"


class FailureKind(str, Enum):
    """Why a hard failure happened. Drives the error taxonomy in the evidence log.

    These are the categories a human triaging a broken capability actually needs to sort
    into, because each points at a different fix: re-record, fix the app, widen the
    allowlist, raise a ceiling, or call the vendor.
    """

    TARGET_NOT_FOUND = "target_not_found"          # no strategy resolved. Candidate for re-record.
    CHECKPOINT_NOT_MET = "checkpoint_not_met"      # flow ran, end state wrong.
    PRECONDITION_FAILED = "precondition_failed"    # wait_for never became true.
    POLICY_BLOCKED = "policy_blocked"              # allowlist or risk policy refused the step.
    RECOVERY_EXHAUSTED = "recovery_exhausted"      # recoverable condition outlived its ceiling.
    SURFACE_ERROR = "surface_error"                # the app itself errored.
    INVALID_INPUT = "invalid_input"                # caller supplied args the contract rejects.
    INVALID_OUTPUT = "invalid_output"              # a declared output is missing or malformed.


class StepTrace(BaseModel):
    """What happened at one step. The unit of debuggability."""

    step_id: str
    intent: str
    action: str
    strategy_used: str | None = Field(
        default=None, description="Which ranked strategy resolved the target, if any."
    )
    strategies_tried: list[str] = Field(default_factory=list)
    ok: bool = True
    detail: str = ""
    recoveries: list[str] = Field(
        default_factory=list, description="Recovery rule codes fired during this step."
    )
    elapsed_ms: int = 0


class ReplayResult(BaseModel):
    """The single value a calling agent receives.

    One shape for all four statuses, so a caller never has to guess which branch it is in.
    """

    status: ResultStatus
    capability_id: str
    capability_version: int

    outputs: dict[str, Any] = Field(
        default_factory=dict, description="Declared outputs. Populated only on SUCCESS."
    )

    outcome_code: str | None = Field(
        default=None, description="KnownOutcome.code when status is BUSINESS_OUTCOME."
    )
    outcome_description: str | None = None

    failure_kind: FailureKind | None = None
    failed_step_id: str | None = None
    expected: str | None = Field(default=None, description="What the step required.")
    observed: str | None = Field(default=None, description="What was actually there.")

    escalation_id: str | None = Field(
        default=None, description="Intervention request id when status is ESCALATED."
    )

    trace: list[StepTrace] = Field(default_factory=list)
    evidence_dir: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_ms: int = 0

    @property
    def caller_should_retry(self) -> bool:
        """Whether a calling agent re-invoking this capability could plausibly succeed.

        Deliberately conservative. A business outcome will not change on retry, and a
        target that cannot be found needs a human, not another attempt.
        """
        return self.failure_kind in {FailureKind.SURFACE_ERROR, FailureKind.RECOVERY_EXHAUSTED}

    # Constructors, so callers never assemble an inconsistent result by hand.

    @classmethod
    def success(cls, cap_id: str, version: int, outputs: dict[str, Any], **kw: Any) -> "ReplayResult":
        return cls(status=ResultStatus.SUCCESS, capability_id=cap_id,
                   capability_version=version, outputs=outputs, **kw)

    @classmethod
    def business(cls, cap_id: str, version: int, code: str, description: str, **kw: Any) -> "ReplayResult":
        return cls(status=ResultStatus.BUSINESS_OUTCOME, capability_id=cap_id,
                   capability_version=version, outcome_code=code,
                   outcome_description=description, **kw)

    @classmethod
    def failure(cls, cap_id: str, version: int, kind: FailureKind, step_id: str | None,
                expected: str, observed: str, **kw: Any) -> "ReplayResult":
        return cls(status=ResultStatus.HARD_FAILURE, capability_id=cap_id,
                   capability_version=version, failure_kind=kind, failed_step_id=step_id,
                   expected=expected, observed=observed, **kw)

    @classmethod
    def escalated(cls, cap_id: str, version: int, escalation_id: str, step_id: str | None,
                  reason: str, **kw: Any) -> "ReplayResult":
        return cls(status=ResultStatus.ESCALATED, capability_id=cap_id,
                   capability_version=version, escalation_id=escalation_id,
                   failed_step_id=step_id, observed=reason, **kw)
