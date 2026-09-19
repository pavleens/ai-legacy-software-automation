"""Guardrails: allowlist enforcement, risk classification, redaction.

PRIOR ART, AND WHY THIS SHAPE. This module is a deliberate port of the policy model I
built in `identity-aware-llm-gateway` (https://github.com/pavleens/identity-aware-llm-gateway,
Go, stdlib only): a reverse proxy in front of an LLM provider that enforced a per-identity
allowlist of permitted models, per-identity rate and budget ceilings, server-side API-key
injection so credentials never reached the caller, and per-identity audit logging of what
was read and done. The three ideas that transferred intact:

  1.  The policy is data, not code. It is configured, reviewable, and diffable, and the
      enforcement point does not know what is on the list.
  2.  The enforcement point sits at the boundary the caller cannot route around. There it
      was the proxy; here it is `PolicyEngine.check_action`, which every action passes
      through in both discovery and replay. A guard that callers can bypass is a
      convention, not a gate.
  3.  Deny by default. An action type or origin that is not explicitly permitted is
      refused, rather than permitted because nobody thought to forbid it.

The one thing that does not transfer is the unit of risk. A gateway reasons about
identities; this system reasons about *effects*. Reading a balance and clicking "Search"
are both reversible even though one is a read and one is a click. Submitting a transfer is
not reversible even though it is the same DOM verb as "Search". So risk is classified per
recorded step, at discovery time, with the model's reasoning attached for a human to
review, and never inferred from the verb at replay time.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from capability_system.artifact.schema import RiskClass
from capability_system.perception.base import Action, ActionKind


class Decision(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"


class PolicyVerdict(BaseModel):
    decision: Decision
    reason: str = ""

    @property
    def permitted(self) -> bool:
        return self.decision is Decision.ALLOW


class Policy(BaseModel):
    """Configuration. Deny by default on both axes.

    `allowed_origins` are matched on scheme+host+port. Path prefixes are separate because
    a back-office app routinely hosts an admin console on the same origin as the servicing
    screens, and "this agent may service accounts" should not imply "this agent may
    administer the institution".
    """

    allowed_origins: list[str] = Field(default_factory=list)
    allowed_path_prefixes: list[str] = Field(
        default_factory=lambda: ["/"],
        description="Paths the agent may reach within an allowed origin.",
    )
    allowed_actions: list[ActionKind] = Field(
        default_factory=lambda: [
            ActionKind.NAVIGATE, ActionKind.CLICK, ActionKind.TYPE,
            ActionKind.SELECT, ActionKind.READ, ActionKind.WAIT_FOR,
        ]
    )
    unattended_risky_actions: bool = Field(
        default=False,
        description="If False, a RISKY_IRREVERSIBLE step requires confirmation or escalation.",
    )
    max_steps: int = Field(default=40, ge=1, description="Ceiling on a single run.")

    # Words that, seen in a control's accessible name, mark an action as likely irreversible.
    # Used only as a *discovery-time hint* offered to the model, never as runtime truth.
    irreversible_hints: list[str] = Field(
        default_factory=lambda: [
            "confirm", "submit", "transfer", "pay", "post", "delete", "remove",
            "close account", "authorize", "authorise", "approve", "send", "issue",
        ]
    )


def _origin(url: str) -> str:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


class PolicyEngine:
    """The enforcement point. Everything that touches a surface goes through here."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._origins = {o.rstrip("/") for o in policy.allowed_origins}

    # -- origin / navigation ----------------------------------------------------------

    def check_navigation(self, url: str) -> PolicyVerdict:
        origin = _origin(url)
        if not origin:
            return PolicyVerdict(decision=Decision.BLOCK, reason=f"Unparseable URL: {url!r}")
        if origin not in self._origins:
            return PolicyVerdict(
                decision=Decision.BLOCK,
                reason=f"Origin {origin} is not on the allowlist {sorted(self._origins)}",
            )
        path = urlparse(url).path or "/"
        def path_matches(prefix: str) -> bool:
            normalised = prefix.rstrip("/") or "/"
            return (
                normalised == "/"
                or path == normalised
                or path.startswith(normalised + "/")
            )

        if not any(path_matches(p) for p in self.policy.allowed_path_prefixes):
            return PolicyVerdict(
                decision=Decision.BLOCK,
                reason=f"Path {path!r} is outside permitted prefixes "
                       f"{self.policy.allowed_path_prefixes}",
            )
        return PolicyVerdict(decision=Decision.ALLOW)

    # -- actions ----------------------------------------------------------------------

    def check_action(
        self,
        action: Action,
        risk: RiskClass = RiskClass.SAFE_REVERSIBLE,
        current_url: str | None = None,
        surface_urls: Sequence[str] = (),
    ) -> PolicyVerdict:
        """The single choke point. Called in discovery AND in replay, by design."""
        if action.kind not in self.policy.allowed_actions:
            return PolicyVerdict(
                decision=Decision.BLOCK,
                reason=f"Action type {action.kind.value!r} is not permitted by policy",
            )

        if action.kind is ActionKind.NAVIGATE:
            if not action.text:
                return PolicyVerdict(decision=Decision.BLOCK, reason="NAVIGATE without a URL")
            return self.check_navigation(action.text)

        # Non-navigating actions still must occur on an allowed origin: a redirect or a
        # frame can move the surface somewhere the allowlist never sanctioned.
        if current_url:
            verdict = self.check_navigation(current_url)
            if not verdict.permitted:
                return PolicyVerdict(
                    decision=Decision.BLOCK,
                    reason=f"Surface drifted off-allowlist: {verdict.reason}",
                )

        for frame_url in surface_urls:
            verdict = self.check_navigation(frame_url)
            if not verdict.permitted:
                return PolicyVerdict(
                    decision=Decision.BLOCK,
                    reason=f"Child frame drifted off-allowlist: {verdict.reason}",
                )

        if risk is RiskClass.RISKY_IRREVERSIBLE and not self.policy.unattended_risky_actions:
            return PolicyVerdict(
                decision=Decision.REQUIRE_CONFIRMATION,
                reason="Step is classified irreversible and unattended risky actions are off",
            )

        return PolicyVerdict(decision=Decision.ALLOW)

    def looks_irreversible(self, control_name: str) -> bool:
        """Discovery-time hint only.

        Offered to the model as a prompt signal so it classifies a step deliberately. It is
        never consulted at replay: by then the classification is recorded in the artifact
        and has been through human review. Inferring risk from a label at execution time
        would mean a vendor renaming a button silently changes the safety posture.
        """
        name = control_name.lower()
        return any(h in name for h in self.policy.irreversible_hints)


# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------

# Ordered most specific first, so a card number is not partially eaten by the digit rule.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("SIN", re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[ -.])?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}\b")),
    ("TOKEN", re.compile(r"\b(?:sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9_-]{16,}\b")),
    ("ACCOUNT", re.compile(r"\b\d{9,17}\b")),
]


def redact(text: str, extra_values: set[str] | None = None) -> str:
    """Scrub before anything is written to an artifact, a log, or evidence.

    Two mechanisms, because pattern matching alone is not enough. Patterns catch the
    shapes of regulated data. `extra_values` catches the specific strings this run was
    handed, which is how a value the caller declared `sensitive` gets removed even when it
    looks like ordinary text (a member nickname, a passphrase, a free-text note).
    """
    if not text:
        return text
    out = text
    for value in sorted(extra_values or set(), key=len, reverse=True):
        if value and len(value) >= 3:
            out = out.replace(value, "[REDACTED]")
    for label, pattern in _PATTERNS:
        out = pattern.sub(f"[REDACTED:{label}]", out)
    return out


def redact_mapping(data: dict[str, object], sensitive_keys: set[str]) -> dict[str, object]:
    """Redact by key name as well as by content.

    A field declared sensitive in the capability contract is removed regardless of what it
    contains, because the contract is the authority on what is regulated here, not a regex.
    """
    result: dict[str, object] = {}
    for key, value in data.items():
        if key in sensitive_keys:
            result[key] = "[REDACTED]"
        elif isinstance(value, str):
            result[key] = redact(value)
        else:
            result[key] = value
    return result
