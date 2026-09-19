"""Evaluation of `StateCondition` against an `Observation`.

NO LLM IS INVOLVED HERE, AND THAT IS THE POINT. A capability's waits, its checkpoint, its
known outcomes and its recovery triggers are all the same predicate type, and all of them
are evaluated by this deterministic function. If any one of them needed a model to decide
"did it work?", the capability would not be a contract -- two runs against identical state
could disagree, and a human reviewing the artifact could not predict what the executor
would do.

WHY CONDITIONS ARE SPLIT FROM THE EXECUTOR. The executor owns sequencing, policy and
recovery; this module owns truth about a single moment. Keeping them apart means the whole
outcome taxonomy can be unit-tested against a hand-built `Observation` with no surface, no
browser and no clock, which is what makes the "not found is a business outcome" behaviour
cheap to prove rather than merely asserted in a design doc.

WHY `element_present` GOES THROUGH `surface.resolve()` RATHER THAN SCANNING THE
OBSERVATION. Presence must mean exactly what targeting means. If the condition matched on
its own looser rules, a checkpoint could pass on an element the executor would then fail to
act on, which is the worst possible disagreement to have inside one system. Delegating to
the same ranked-strategy resolver keeps one definition of "this control exists".
"""

from __future__ import annotations

import re
from typing import Any

from capability_system.artifact.schema import StateCondition
from capability_system.perception.base import Observation


def _resolve(condition: StateCondition, surface: Any) -> str | None:
    """Resolve the condition's target, tolerating a condition that declares none.

    A malformed condition (element_present with no target) returns None rather than
    raising: an artifact that is wrong should fail as "condition not met" with a legible
    trace, not as a stack trace out of the middle of a bank run.
    """
    if condition.target is None:
        return None
    if surface is None:
        return None
    return surface.resolve(condition.target.strategies)


def _element_for_handle(obs: Observation, handle: str | None) -> Any:
    if handle is None:
        return None
    for element in obs.elements:
        if element.handle == handle:
            return element
    return None


def _raw(condition: StateCondition, obs: Observation, surface: Any) -> bool:
    """The un-negated truth value. Split out so `negate` is applied in exactly one place."""
    kind = condition.kind

    if kind == "element_present":
        return _resolve(condition, surface) is not None

    if kind == "element_absent":
        return _resolve(condition, surface) is None

    if kind == "text_present":
        # Case-insensitive because vendor screens change capitalisation between releases far
        # more often than they change the sentence, and an outcome code that silently stops
        # matching after a re-skin is worse than one that matches slightly too widely.
        haystack = f"{obs.text_digest}\n{obs.title}"
        return condition.value.lower() in haystack.lower()

    if kind == "url_matches":
        # Regex, not equality: tenants differ by origin and screens carry ids in the path,
        # so an artifact that is reusable across institutions cannot assert a literal URL.
        if obs.url is None:
            return False
        try:
            return re.search(condition.value, obs.url) is not None
        except re.error:
            # A bad pattern in the artifact is a false condition, never an exception.
            return False

    if kind == "value_equals":
        element = _element_for_handle(obs, _resolve(condition, surface))
        if element is None:
            return False
        return (element.value or "") == condition.value

    # Unknown kind: false. pydantic's Literal should make this unreachable, but replay
    # must degrade to "condition not met" rather than crash if an artifact predates a kind.
    return False


def evaluate(condition: StateCondition, obs: Observation, surface: Any = None) -> bool:
    """Return whether `condition` holds over `obs`.

    `surface` is optional so text/url predicates stay testable with no surface at all.
    """
    result = _raw(condition, obs, surface)
    return (not result) if condition.negate else result


def describe(condition: StateCondition) -> str:
    """Human-readable rendering, used to populate `expected` on a failure.

    A failure whose `expected` field says "checkpoint" tells an operator nothing. It has to
    say which predicate over what value, because that string is the whole of what a triaging
    human sees before deciding whether to re-record the capability or call the vendor.
    """
    bits = [condition.kind, repr(condition.value)]
    if condition.target is not None:
        bits.append(f"target={condition.target.description!r}")
    if condition.negate:
        bits.append("(negated)")
    return " ".join(bits)
