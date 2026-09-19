"""Surface abstraction: the seam between "how we perceive and act" and "what we recorded".

The single most consequential decision in this system is what an `Observation` is made of.
If observations are DOM nodes, every artifact we record is secretly a DOM artifact and the
design cannot follow us to a desktop application or to a frameset-era web app with no
semantic markup. So an Observation is a flat list of `UIElement` -- role, name, value,
state -- which is the vocabulary every accessibility layer already speaks:

    web        Chromium accessibility tree (Playwright `page.accessibility` / CDP)
    Windows    UI Automation (UIA) control patterns
    Linux      AT-SPI
    macOS      NSAccessibility

A `handle` is an opaque, surface-private string. The agent and the artifact never parse it.
That is deliberate: it keeps CSS selectors, XPaths, UIA runtime IDs and screen coordinates
all behind the same door, so the replay engine's contract does not change when the surface
technology does.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, Sequence

from pydantic import BaseModel, Field


class ActionKind(str, Enum):
    """The verbs a surface must support.

    Deliberately small. Every additional verb is another thing each surface
    implementation has to get right, and a bank back-office flow is overwhelmingly
    navigate / click / type / select / read.
    """

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    READ = "read"
    WAIT_FOR = "wait_for"


class UIElement(BaseModel):
    """One perceivable control, described the way an operator would describe it."""

    handle: str = Field(description="Opaque, surface-private address. Never parsed by callers.")
    role: str = Field(description="Accessibility role: button, textbox, link, row, heading...")
    name: str = Field(default="", description="Accessible name, the visible label a human reads.")
    value: str | None = Field(default=None, description="Current value for inputs and selects.")
    enabled: bool = True
    visible: bool = True
    # Structural hints kept as plain strings so no surface is forced to invent a DOM.
    container: str | None = Field(
        default=None,
        description="Nearest labelled ancestor (frame title, fieldset legend, table caption).",
    )
    ordinal: int | None = Field(
        default=None,
        description="Position among siblings sharing role+name. Last-resort disambiguator.",
    )


class Observation(BaseModel):
    """A snapshot of what an operator could see and act on right now."""

    url: str | None = Field(default=None, description="Location, where the surface has one.")
    frame_urls: list[str] = Field(
        default_factory=list,
        description="HTTP(S) locations for every actionable frame in this observation.",
    )
    title: str = ""
    elements: list[UIElement] = Field(default_factory=list)
    text_digest: str = Field(
        default="",
        description="Flattened visible text. Used for outcome detection, never for targeting.",
    )
    screenshot_path: str | None = None

    def describe(self, limit: int = 200) -> str:
        """Render for an LLM prompt.

        THE CAP IS A CORRECTNESS CONCERN, NOT A COST ONE. It was 60. The member detail
        screen has 74 elements and the two that mattered -- the Savings label cell and the
        balance cell holding 18,430.09 -- both fell in the omitted tail. The model was
        asked to read a balance it had never been shown, guessed a nearby handle, and
        looped. Truncation that silently removes the answer does not make the task
        cheaper; it makes it impossible.

        200 comfortably covers a dense legacy screen. A surface large enough to exceed it
        needs relevance ranking rather than a blind head-slice, which is named as future
        work in REPORT.md.
        """
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "ELEMENTS:"]
        for el in self.elements[:limit]:
            bits = [f"[{el.handle}]", el.role, repr(el.name)]
            if el.value is not None:
                bits.append(f"value={el.value!r}")
            if not el.enabled:
                bits.append("(disabled)")
            if el.container:
                bits.append(f"in={el.container!r}")
            lines.append("  " + " ".join(bits))
        if len(self.elements) > limit:
            lines.append(f"  ... {len(self.elements) - limit} more elements omitted")
        return "\n".join(lines)


class Action(BaseModel):
    """An instruction to change or read surface state."""

    kind: ActionKind
    handle: str | None = Field(default=None, description="Target element. None for NAVIGATE.")
    text: str | None = Field(default=None, description="Payload for TYPE, url for NAVIGATE.")
    option: str | None = Field(default=None, description="Option label for SELECT.")


class ActionResult(BaseModel):
    ok: bool
    read_value: str | None = None
    error: str | None = None


class Surface(Protocol):
    """What every perceivable, actionable target must implement.

    Implementing this for a desktop application means mapping UIA control patterns onto
    the same six verbs. Nothing above this line has to change.
    """

    def observe(self) -> Observation: ...

    def act(self, action: Action) -> ActionResult: ...

    def resolve(self, strategies: Sequence["object"]) -> str | None:
        """Resolve ordered locator strategies to a live handle, or None.

        Sequence is typed loosely here to avoid importing the artifact schema into the
        perception layer. Replay passes `TargetStrategy` objects.
        """
        ...

    def close(self) -> None: ...
