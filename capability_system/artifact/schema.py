"""The capability artifact: a typed, versioned contract an AI agent can call.

This is the centre of the system, so the shape is argued rather than assumed.

Three commitments drive it:

1.  **It is a contract, not a recording.** A transcript of "what the model clicked" is
    not reusable. What a calling agent needs is the same thing any function signature
    gives you: typed inputs, typed outputs, a success condition, and a documented set of
    outcomes that are not failures. Steps are an implementation detail of the contract.

2.  **A target is a ranked hypothesis set, never a selector.** Recording one CSS path is
    how replay-many systems rot. Each step carries ordered `TargetStrategy` entries with
    the model's own rationale, so replay degrades through semantic -> structural ->
    positional rather than failing at the first miss. The ordering encodes the robustness
    argument and survives review by a human.

3.  **"Not found" is a result, not a crash.** `known_outcomes` promotes expected business
    states into first-class, declared returns. Without this, a caller cannot tell "the
    member does not exist" from "the automation is broken", which is the difference
    between a useful capability and an alarm.

Storage is JSON. It diffs in review, it needs no runtime, and a human reviewer approving a
capability for unattended use is reading the same bytes the executor runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------------------
# Typing of the capability's public interface
# --------------------------------------------------------------------------------------

class ParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    MONEY = "money"
    DATE = "date"


class InputParam(BaseModel):
    """One argument the calling agent supplies per invocation."""

    name: str
    type: ParamType = ParamType.STRING
    required: bool = True
    description: str = ""
    example: str | None = Field(
        default=None,
        description="Value seen during discovery. Redacted if the param is sensitive.",
    )
    sensitive: bool = Field(
        default=False,
        description="If true the value is never written to artifacts, logs or evidence.",
    )


class OutputField(BaseModel):
    """One value the capability returns."""

    name: str
    type: ParamType = ParamType.STRING
    description: str = ""
    from_step: str = Field(description="Step id whose READ produced this value.")
    required: bool = Field(
        default=True,
        description="A successful replay must return this field unless explicitly optional.",
    )
    sensitive: bool = Field(
        default=True,
        description="If true, evidence redacts this field while the caller still receives it.",
    )


# --------------------------------------------------------------------------------------
# Targeting
# --------------------------------------------------------------------------------------

class StrategyKind(str, Enum):
    """Ordered from most to least durable, per element class.

    ROW_COLUMN leads for CONTENT-NAMED elements only. A table cell's accessible name IS
    its own text, so every name-based rung for a data cell silently encodes the value that
    happened to be on screen during discovery. A capability recorded against member 12345
    then resolves only for a balance of 18,430.09, which is a demo, not a capability.
    ROW_COLUMN addresses the cell structurally instead, by (row key, column) within a
    titled table, and carries no content at all. It is portable for the same reason the
    rest of this ladder is: it is UIA's GridPattern.GetItem and AT-SPI's
    Table.getAccessibleAt.

    ROLE_NAME leads for everything else -- buttons, links, inputs -- because a button's
    accessible name is a label rather than data, and role plus visible label is the one
    thing that tends to survive a vendor re-skin.
    """

    ROW_COLUMN = "row_column"        # (row key, column) in a titled table. Content-free.
    ROLE_NAME = "role_name"          # role=button, name="Search"
    LABEL = "label"                  # form control associated with a visible label
    CONTAINER_ROLE_NAME = "container_role_name"  # disambiguated by labelled ancestor
    TEXT = "text"                    # exact visible text
    ORDINAL = "ordinal"              # nth match of role+name. Brittle, last resort.


class TargetStrategy(BaseModel):
    kind: StrategyKind
    value: str = Field(description="Strategy payload, e.g. 'button|Search' or 'Member ID'.")
    container: str | None = None
    ordinal: int | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(
        default="",
        description="Why this identifies the control. Written during discovery, read by humans.",
    )


class TargetSpec(BaseModel):
    """Ranked strategies for one control. Replay tries them in order."""

    description: str = Field(description="How an operator would name this control.")
    strategies: list[TargetStrategy] = Field(min_length=1)


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------

class RiskClass(str, Enum):
    """Safe/reversible versus risky/irreversible.

    The split is by *effect*, not by verb. Reading a balance and clicking "Search" are
    both reversible. Submitting a transfer is not, even though both are clicks.
    """

    SAFE_REVERSIBLE = "safe_reversible"
    RISKY_IRREVERSIBLE = "risky_irreversible"


class StateCondition(BaseModel):
    """A predicate over an Observation. The unit of waiting and of verification.

    Replay waits on these rather than on sleeps, which is most of what determinism is.
    """

    kind: Literal["element_present", "element_absent", "text_present", "url_matches", "value_equals"]
    value: str
    target: TargetSpec | None = None
    negate: bool = False


class Step(BaseModel):
    id: str
    intent: str = Field(description="Why this step exists, in operator language.")
    action: str = Field(description="ActionKind value: navigate|click|type|select|read|wait_for")
    target: TargetSpec | None = Field(default=None, description="None for navigate.")
    literal: str | None = Field(default=None, description="Static payload, e.g. a URL or fixed text.")
    binds_input: str | None = Field(default=None, description="InputParam name supplying the payload.")
    reads_into: str | None = Field(default=None, description="OutputField name this READ populates.")
    risk: RiskClass = RiskClass.SAFE_REVERSIBLE
    wait_for: StateCondition | None = Field(
        default=None, description="Must hold before the step is considered complete."
    )


# --------------------------------------------------------------------------------------
# Outcomes: the part that separates a capability from a macro
# --------------------------------------------------------------------------------------

class KnownOutcome(BaseModel):
    """A business result the caller must be told about, which is not an error.

    "No such member" is the canonical case. It is a correct answer to a correct question
    and must never surface as a failed automation.
    """

    code: str = Field(description="Stable machine code, e.g. MEMBER_NOT_FOUND.")
    description: str = ""
    detect: StateCondition
    terminal: bool = Field(
        default=True, description="If true, replay stops here and reports this outcome."
    )


class RecoveryRule(BaseModel):
    """A bounded, pre-authorised remedy for a transient or interstitial condition.

    Bounded is the operative word. Each rule declares its own retry ceiling so a
    recoverable condition can never become an unbounded loop against a bank system.
    """

    code: str = Field(description="e.g. SESSION_INTERSTITIAL, TRANSIENT_SLOW_LOAD")
    description: str = ""
    detect: StateCondition
    remedy: Literal["dismiss", "wait_retry", "reload"] = "wait_retry"
    dismiss_target: TargetSpec | None = None
    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_ms: int = Field(default=750, ge=0)


# --------------------------------------------------------------------------------------
# Heterogeneity and multi-tenant reuse
# --------------------------------------------------------------------------------------

class SurfaceDescriptor(BaseModel):
    """What this artifact was recorded against.

    `product` and `product_version` are what make cross-tenant reuse possible: hundreds of
    institutions run the same vendor product at different origins, so the artifact is keyed
    to the product and the origin is supplied per tenant at invocation time.
    """

    kind: Literal["web", "desktop"] = "web"
    product: str = Field(description="Vendor product identity, e.g. 'acme-core-banking'.")
    product_version: str | None = None
    recorded_origin: str | None = Field(
        default=None, description="Origin seen during discovery. Never hard-coded into steps."
    )


class TenantOverride(BaseModel):
    """Per-tenant specialisation of a shared artifact, instead of a per-tenant re-record."""

    tenant_id: str
    origin: str | None = None
    step_target_overrides: dict[str, TargetSpec] = Field(default_factory=dict)
    note: str = ""


class ApprovalState(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"


# --------------------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------------------

class CapabilityArtifact(BaseModel):
    """A versioned, reviewable, agent-invocable capability."""

    schema_version: str = SCHEMA_VERSION
    id: str = Field(description="Stable slug, e.g. 'lookup_member_balance'.")
    version: int = Field(default=1, ge=1)
    name: str
    description: str = Field(description="What a calling agent should expect this to do.")

    surface: SurfaceDescriptor
    approval: ApprovalState = Field(
        default=ApprovalState.DRAFT,
        description="Unattended replay is gated on APPROVED. Discovery always emits DRAFT.",
    )

    inputs: list[InputParam] = Field(default_factory=list)
    outputs: list[OutputField] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)

    checkpoint: StateCondition = Field(
        description="The success condition. Replay that does not reach this has not succeeded."
    )
    known_outcomes: list[KnownOutcome] = Field(default_factory=list)
    recovery_rules: list[RecoveryRule] = Field(default_factory=list)

    allowed_origins: list[str] = Field(
        default_factory=list,
        description="Origin allowlist this capability may act within. Enforced at replay.",
    )
    tenant_overrides: list[TenantOverride] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    discovered_by: str = Field(default="", description="Model identity that recorded it.")
    discovery_evidence: str | None = Field(
        default=None, description="Path to the discovery run's evidence directory."
    )

    def input_map(self) -> dict[str, InputParam]:
        return {p.name: p for p in self.inputs}

    def sensitive_names(self) -> set[str]:
        return (
            {p.name for p in self.inputs if p.sensitive}
            | {p.name for p in self.outputs if p.sensitive}
        )
