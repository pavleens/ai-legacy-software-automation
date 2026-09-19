"""Playwright-backed web Surface: perception as *role + accessible name*, never as CSS.

WHY NOT CSS / XPATH
-------------------
The temptation on a browser surface is to perceive the DOM, because the DOM is right
there. Resist it. If an `Observation` is made of DOM nodes then every capability artifact
this system records is secretly a *DOM* artifact, and the design can never follow the
agent onto a desktop application -- which is where a large share of bank back-office work
actually lives.

Role plus accessible name has an exact equivalent on every platform accessibility layer:

    web        Chromium accessibility tree  (role, name)
    Windows    UI Automation                (ControlType, Name)
    Linux      AT-SPI                       (AccessibleRole, accessible-name)
    macOS      NSAccessibility              (AXRole, AXTitle)

So `TargetStrategy(kind=ROLE_NAME, value="button|Search")` is portable text. A CSS
selector is not. That single constraint is what lets `capability_system/artifact` stay
surface-agnostic, and it is why this module never emits, stores, or resolves a selector.

HOW WE READ THE TREE
--------------------
Via `locator.aria_snapshot(mode="ai", boxes=True)`, which returns Chromium's accessibility
tree as YAML. `page.accessibility.snapshot()` was REMOVED in Playwright v1.57 and must not
be used; `mode=` arrived in v1.59 and `boxes=` in v1.60, hence the runtime version gate in
`_require_playwright_version()`.

`mode="ai"` resolves to `visibility:'ariaOrVisible'`, `refs:'interactable'`,
`includeGenericRole:true`, so it reports visible-but-unlabelled controls (which a strict
ARIA walk would drop) and stamps each interactable node with a **ref** such as `f2e19`,
addressable as `page.locator("aria-ref=f2e19")`.

HANDLES ARE EPHEMERAL AND OPAQUE
--------------------------------
`UIElement.handle` is `"f{frame_index}|aria-ref={ref}"`. Only this module ever parses it.

Refs are assigned **by position in the tree at snapshot time** and are reassigned on the
next snapshot of a changed page. This was verified against the mock bank and it is not a
theoretical hazard: on `/app/search`, `f2e19` is the *Member ID textbox*; after clicking
Search, `f2e19` on the same frame is the *"Member ID" table cell* of the member-detail
screen. A ref persisted into an artifact would silently address the wrong element on the
next run, and would do so without erroring.

Therefore:

*   `observe()` hands out refs. That is correct: observation is live and discovery-time.
*   `resolve()` NEVER accepts a ref. It takes `TargetStrategy` objects -- role+name, label,
    container-scoped, text, ordinal -- and re-derives a live handle from the current tree.
*   `stable_target()` is the bridge. Given a live ephemeral handle it compiles the ordered
    ladder of *stable semantic* strategies that identify the same control. Discovery holds
    a ref; the artifact stores only what `stable_target()` returned. That compilation step
    is the whole reason a recorded capability survives to the next run.

THE PROXIMITY-LABEL PASS (the part that makes role+name viable on legacy HTML)
-----------------------------------------------------------------------------
The mock bank -- like the real systems it imitates -- puts each field's label in the table
cell to the *left* of the input, in a `<label>` with no `for=` and no `id=` to point at,
and no `aria-label` anywhere. So Chromium computes an EMPTY accessible name, and
`get_by_role("textbox", name="Member ID")` matches nothing. Verified empirically:

    - cell "Member ID" [ref=f2e17]
    - cell [ref=f2e18]:
      - textbox [ref=f2e19]          <-- no name

`mode="ai"` surfaces the control but does not name it. Naming it is this module's job.
For any interactive control whose accessible name is empty we derive the label the way a
teller reads the screen, in priority order:

  1. the name of the preceding sibling `cell` of the control's enclosing `cell`
     (table-based forms put the label in the cell to the left)
  2. the nearest preceding named/text sibling within the same block
  3. a `<label>` wrapping or immediately preceding the control
  4. the control's `name` or `value` attribute, as a last resort

Rules 1 and 2 are computed **on the accessibility tree**, not the DOM -- "the cell to my
left" is expressible on UIA and AT-SPI too, so the fallback ports with the rest of the
design. Only rules 3 and 4 need the DOM, and they fire for a minority of controls.

A derived name is deliberately NOT presented as if it were an accessible name. It goes in
`UIElement.name` so an operator (and an LLM) can read it, but `stable_target()` leads with
`StrategyKind.LABEL` rather than `ROLE_NAME`, so the distinction survives into the
artifact and replay resolves it through the same proximity pass instead of through a
`get_by_role` name lookup that would fail. Nothing was added to the base contract to carry
this: derived-ness is implicit in which strategy kind resolves the control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from playwright.sync_api import Browser, BrowserContext, Error as PWError, Frame, Page, sync_playwright

from capability_system.artifact.schema import StrategyKind, TargetStrategy
from capability_system.perception.base import (
    Action,
    ActionKind,
    ActionResult,
    Observation,
    UIElement,
)

# `aria_snapshot(boxes=...)` landed in Playwright 1.60; `mode=` in 1.59; and
# `page.accessibility` was removed in 1.57. Anything older silently lacks the API this
# module is built on, so fail loudly at start() rather than mysteriously at snapshot time.
MIN_PLAYWRIGHT = (1, 60)

_HANDLE_RE = re.compile(r"^f(?P<frame>\d+)\|aria-ref=(?P<ref>[A-Za-z0-9]+)$")

# One snapshot line, e.g.  `- textbox "Member ID" [disabled] [ref=f2e19] [box=1,2,3,4]:`
_NODE_RE = re.compile(
    r"^(?P<role>[A-Za-z][A-Za-z0-9_-]*)"
    r'(?:\s+"(?P<name>(?:[^"\\]|\\.)*)")?'
    r"(?P<attrs>(?:\s*\[[^\]]*\])*)"
)
_ATTR_RE = re.compile(r"\[([^\]]*)\]")

#: Roles that an operator can act on. Empty-named members of this set go through the
#: proximity-label pass.
INTERACTIVE_ROLES = frozenset(
    {
        "link", "button", "textbox", "searchbox", "combobox", "listbox", "checkbox",
        "radio", "spinbutton", "slider", "switch", "menuitem", "menuitemcheckbox",
        "menuitemradio", "tab", "option",
    }
)

#: Roles that carry meaning but are not acted on. Reported so an LLM can ground itself and
#: so READ has something to target.
SEMANTIC_ROLES = frozenset(
    {"heading", "cell", "columnheader", "rowheader", "row", "alert", "status", "img"}
)

#: Structural scaffolding. Walked, never reported as elements.
_SKIP_ROLES = frozenset({"generic", "table", "rowgroup", "list", "none", "presentation", "paragraph"})

#: Roles whose accessible name is computed from their own contents, so a TEXT strategy is
#: a meaningful fallback for them.
_TEXT_ADDRESSABLE = frozenset({"link", "button", "cell", "columnheader", "rowheader", "heading"})

#: Cell-ish roles that make up a tabular structure.
CELL_ROLES = frozenset({"cell", "gridcell", "columnheader", "rowheader"})

#: Roles whose accessible name is COMPUTED FROM THEIR OWN CONTENTS, i.e. whose name is the
#: data they display rather than a label for it. Targeting one of these by name records the
#: value that happened to be on screen during discovery, so the capability only ever works
#: for that one record. See `_is_content_named`.
#:
#: Deliberately conservative: on a legacy screen you cannot reliably tell a label cell
#: ("Member Name") from a value cell ("Dolores A. Kettleman") by inspection, so every data
#: cell is treated as potentially value-bearing. Misreading a label as a value only changes
#: which rung is tried first, and both are verified before they are recorded. Misreading a
#: value as a label produces a capability that silently works for exactly one member --
#: an asymmetry that decides the heuristic. `columnheader`/`rowheader`/`heading` are
#: excluded: they are structural labels by definition.
CONTENT_NAMED_ROLES = frozenset({"cell", "gridcell", "row", "listitem"})

#: `StrategyKind.ROW_COLUMN` is the right home for a structural rung, but
#: `capability_system/artifact/schema.py` is owned elsewhere and must not be edited here.
#: Until the member exists this rung rides on CONTAINER_ROLE_NAME with a sentinel-prefixed
#: value, and `_resolve_one` dispatches on the prefix BEFORE the kind so both spellings
#: resolve identically. When the enum member lands this shim becomes a no-op.
_ROW_COLUMN_PREFIX = "@rowcol:"
ROW_COLUMN_KIND = getattr(StrategyKind, "ROW_COLUMN", StrategyKind.CONTAINER_ROLE_NAME)
_ROW_COLUMN_NATIVE = ROW_COLUMN_KIND is not StrategyKind.CONTAINER_ROLE_NAME

_MAX_DIGEST = 4000
_MAX_DERIVED_LABEL = 120


def _collapse(text: str) -> str:
    """Whitespace-collapse the way accessible-name computation does, so our derived names
    compare equal to Chromium's."""
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


def _clean_label(text: str) -> str:
    """Strip the punctuation operators ignore when reading a form label off the screen."""
    return _collapse(text).rstrip(":*").strip()


# ======================================================================================
# Accessibility-tree parsing
# ======================================================================================


@dataclass
class _Node:
    """One node of the parsed ARIA snapshot. Mirrors an a11y-tree node, not a DOM node."""

    role: str
    name: str = ""
    ref: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    text: str = ""          # literal text content for `- text: ...` leaves
    children: list["_Node"] = field(default_factory=list)
    parent: "_Node | None" = None

    @property
    def disabled(self) -> bool:
        return "disabled" in self.attrs

    @property
    def visible(self) -> bool:
        """Boxes are present because we snapshot with `boxes=True`. A zero-area box means
        the node occupies no screen space (collapsed `<option>`s, for instance)."""
        box = self.attrs.get("box")
        if not box:
            return True
        try:
            _x, _y, w, h = (float(p) for p in box.split(","))
        except ValueError:
            return True
        return w > 0 and h > 0

    def label_text(self) -> str:
        """What this node reads as on screen: its accessible name, else its literal text."""
        return self.name or _collapse(self.text)


def _parse_aria_snapshot(yaml_text: str) -> _Node:
    """Parse `aria_snapshot(mode="ai")` YAML into a tree.

    Hand-rolled rather than via PyYAML for two reasons: the payloads are a fixed, tiny
    grammar (`role "name" [attr]...`) that would need regex-parsing out of the YAML scalars
    anyway, and it keeps the perception layer free of an extra runtime dependency.
    """
    root = _Node(role="#root")
    # Stack of (indent_columns, node). Children are more-indented than their parent.
    stack: list[tuple[int, _Node]] = [(-1, root)]

    for raw in yaml_text.splitlines():
        if not raw.strip():
            continue
        stripped = raw.lstrip(" ")
        if not stripped.startswith("- "):
            # Continuation of a YAML block scalar (multi-line literal text). The name we
            # care about is on the opening line, so nothing is lost by skipping it.
            continue
        indent = len(raw) - len(stripped)
        payload = stripped[2:]

        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root

        node = _parse_payload(payload)
        if node is None:
            continue
        node.parent = parent
        parent.children.append(node)
        stack.append((indent, node))

    return root


def _parse_payload(payload: str) -> _Node | None:
    """Turn one `- ` line's payload into a node.

    Handles the four shapes the serializer emits:
        role "name" [ref=x] [box=...]           leaf
        role "name" [ref=x]:                    node with children
        'role "name: with colon" [ref=x]':      single-quoted when the scalar needs it
        text: some literal   /  /url: /app/x    property leaves
    """
    payload = payload.strip()
    if not payload:
        return None

    # Property leaves. `/url:` is the link target; `text:` is a bare text run, which the
    # proximity pass uses as a "preceding visible text" candidate.
    if payload.startswith("/"):
        key, _, val = payload.partition(":")
        return _Node(role="#prop", name=key.strip("/"), text=_unquote(val.strip()))
    if payload.startswith("text:"):
        return _Node(role="#text", text=_unquote(payload[len("text:"):].strip()))

    # YAML single-quotes the whole descriptor when it contains ": ". Unwrap it first so the
    # node regex sees the same grammar either way.
    if payload.startswith("'"):
        end = _find_single_quote_end(payload)
        if end > 0:
            descriptor = payload[1:end].replace("''", "'")
            return _build_node(descriptor)

    m = _NODE_RE.match(payload)
    if not m:
        return None
    return _build_node(payload)


def _find_single_quote_end(s: str) -> int:
    """Index of the closing quote of a YAML single-quoted scalar starting at index 0."""
    i = 1
    while i < len(s):
        if s[i] == "'":
            if i + 1 < len(s) and s[i + 1] == "'":
                i += 2
                continue
            return i
        i += 1
    return -1


def _build_node(descriptor: str) -> _Node | None:
    m = _NODE_RE.match(descriptor.strip())
    if not m:
        return None
    role = m.group("role")
    name = m.group("name") or ""
    name = name.replace('\\"', '"').replace("\\\\", "\\")

    attrs: dict[str, str] = {}
    for chunk in _ATTR_RE.findall(m.group("attrs") or ""):
        key, sep, val = chunk.partition("=")
        attrs[key.strip()] = val.strip() if sep else "true"

    node = _Node(role=role, name=_collapse(name), ref=attrs.get("ref"), attrs=attrs)

    # A node with children is written `... :`; anything after that colon is the node's own
    # literal text, which we keep for the digest and for text-based fallbacks.
    tail = descriptor[m.end():].strip()
    if tail.startswith(":"):
        node.text = _collapse(tail[1:])
    return node


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return _collapse(s)


def _walk(node: _Node) -> Iterable[_Node]:
    """Depth-first, document order. Determinism of element ordering rests on this."""
    for child in node.children:
        yield child
        yield from _walk(child)


# ======================================================================================
# The scan: one pass over every frame, producing the live element index
# ======================================================================================


@dataclass
class _Scanned:
    """A perceived element plus the surface-private bookkeeping callers never see."""

    handle: str
    frame_index: int
    ref: str
    role: str
    name: str
    accessible_name: str     # Chromium's own computation; "" for the hostile controls
    derived: bool            # True when `name` came from the proximity pass
    value: str | None
    enabled: bool
    visible: bool
    container: str | None
    text: str
    ordinal: int = 0         # position among elements sharing role+name, across the page
    # Tabular position. Set for cell-ish elements only; this is what a structural,
    # value-independent target is built from.
    column_index: int | None = None
    column_header: str = ""
    row_index: int | None = None
    row_texts: tuple[str, ...] = ()

    def is_content_named(self) -> bool:
        """True when this element's accessible name IS the data it displays.

        Such an element must never be addressed by name alone: the name is the value that
        happened to be on screen when the capability was recorded.
        """
        return self.role in CONTENT_NAMED_ROLES and bool(self.name)

    def to_ui_element(self) -> UIElement:
        return UIElement(
            handle=self.handle,
            role=self.role,
            name=self.name,
            value=self.value,
            enabled=self.enabled,
            visible=self.visible,
            container=self.container,
            ordinal=self.ordinal,
        )


class WebSurface:
    """A live browser rendered as role/name observations and six verbs.

    Not thread-safe and not re-entrant: it owns one Playwright driver, one browser, one
    context and one page, because a capability run is a single serial conversation with a
    single screen.
    """

    def __init__(
        self,
        headless: bool = True,
        base_url: str | None = None,
        slow_mo_ms: int = 0,
        viewport: tuple[int, int] = (1280, 900),
    ) -> None:
        self.headless = headless
        self.base_url = base_url.rstrip("/") if base_url else None
        self.slow_mo_ms = slow_mo_ms
        self.viewport = viewport

        self._pw: Any = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        # Last scan. `resolve()` and `stable_target()` read it; both refresh it first, so a
        # caller is never resolving against a stale tree.
        self._scan: list[_Scanned] = []
        self._by_handle: dict[str, _Scanned] = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        _require_playwright_version()
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo_ms or 0
        )
        self._context = self._browser.new_context(
            viewport={"width": self.viewport[0], "height": self.viewport[1]}
        )
        self._page = self._context.new_page()

    def close(self) -> None:
        """Tear down in reverse order, swallowing errors.

        Close is called on the failure path too, so it must never raise and mask the
        original exception.
        """
        for closer in (self._context, self._browser):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self._page = None
        self._scan = []
        self._by_handle = {}

    def __enter__(self) -> "WebSurface":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("WebSurface.start() has not been called")
        return self._page

    # ---------------------------------------------------------------- observation

    def observe(self) -> Observation:
        """Snapshot every frame and flatten it into role/name elements.

        Every frame, not just the main one. `/app` is a `<frameset>`: the main frame has no
        `<body>` at all and all content lives in `navfrm` and `mainfrm`. A surface that
        reads only `page.main_frame` perceives an entirely empty screen.
        """
        page = self.page
        elements = self._rescan()
        digest_parts: list[str] = []
        for frame in page.frames:
            text = self._frame_text(frame)
            if text:
                digest_parts.append(text)
        digest = _collapse(" ".join(digest_parts))[:_MAX_DIGEST]

        return Observation(
            url=page.url,
            frame_urls=sorted(
                {
                    frame.url
                    for frame in page.frames
                    if frame.url.startswith(("http://", "https://"))
                }
            ),
            title=self._title(),
            elements=[s.to_ui_element() for s in elements],
            text_digest=digest,
        )

    def _title(self) -> str:
        """Main page title, falling back to the content frame.

        A `<frameset>` shell often carries a generic title (or none) while the frame the
        operator is actually looking at carries the screen name, so the fallback is what
        makes `Observation.title` useful for outcome detection.
        """
        page = self.page
        try:
            title = _collapse(page.title())
        except PWError:
            title = ""
        if title:
            return title
        for frame in page.frames[1:]:
            try:
                sub = _collapse(frame.title())
            except PWError:
                continue
            if sub:
                return sub
        return ""

    def _frame_text(self, frame: Frame) -> str:
        try:
            body = frame.locator("body")
            if body.count() == 0:
                return ""
            return _collapse(body.inner_text(timeout=2000))
        except PWError:
            return ""

    # ---------------------------------------------------------------- the scan

    def _rescan(self) -> list[_Scanned]:
        """Rebuild the live element index. Deterministic: frame order, then document order."""
        page = self.page
        scanned: list[_Scanned] = []

        for frame_index, frame in enumerate(page.frames):
            try:
                body = frame.locator("body")
                if body.count() == 0:
                    # The frameset shell. Real content is in the child frames.
                    continue
                yaml_text = body.aria_snapshot(mode="ai", boxes=True, timeout=5000)
            except PWError:
                continue

            root = _parse_aria_snapshot(yaml_text)
            frame_label = self._frame_label(frame, frame_index)
            scanned.extend(self._scan_frame(frame, frame_index, root, frame_label))

        # Ordinals are global across the observation, which is what StrategyKind.ORDINAL
        # means in an artifact: "the nth control on the screen with this role and name",
        # including across frames. That is precisely the disambiguator the duplicate
        # "Submit" buttons need.
        counts: dict[tuple[str, str], int] = {}
        for s in scanned:
            key = (s.role, s.name)
            s.ordinal = counts.get(key, 0)
            counts[key] = s.ordinal + 1

        self._scan = scanned
        self._by_handle = {s.handle: s for s in scanned}
        return scanned

    def _frame_label(self, frame: Frame, index: int) -> str:
        try:
            title = _collapse(frame.title())
        except PWError:
            title = ""
        return title or frame.name or f"frame{index}"

    def _scan_frame(
        self, frame: Frame, frame_index: int, root: _Node, frame_label: str
    ) -> list[_Scanned]:
        out: list[_Scanned] = []
        for node in _walk(root):
            if node.role.startswith("#") or node.role in _SKIP_ROLES:
                continue
            interactive = node.role in INTERACTIVE_ROLES
            if not interactive and node.role not in SEMANTIC_ROLES:
                continue
            if node.ref is None:
                # No ref means no addressable element (collapsed `<option>`s, for example).
                continue

            accessible = node.name
            name, derived = accessible, False

            if interactive and not accessible:
                derived_name = self._derive_label(frame, node)
                if derived_name:
                    name, derived = derived_name, True
            elif node.role == "row" and not accessible:
                # Legacy data rows carry no accessible name of their own; an operator reads
                # them as the concatenation of their cells, so that is what we report.
                cells = [c.label_text() for c in node.children
                         if c.role in {"cell", "columnheader", "rowheader"}]
                joined = " | ".join(c for c in cells if c)
                if not joined:
                    continue
                name, derived = joined[:_MAX_DERIVED_LABEL], True
            elif not interactive and not accessible and not node.text:
                # An unnamed, textless cell is layout padding. Reporting it is noise.
                continue

            col_index, col_header, row_index, row_texts = self._grid_position(node)
            out.append(
                _Scanned(
                    column_index=col_index,
                    column_header=col_header,
                    row_index=row_index,
                    row_texts=row_texts,
                    handle=f"f{frame_index}|aria-ref={node.ref}",
                    frame_index=frame_index,
                    ref=node.ref,
                    role=node.role,
                    name=name,
                    accessible_name=accessible,
                    derived=derived,
                    value=self._value_of(frame, node) if interactive else None,
                    enabled=not node.disabled,
                    visible=node.visible,
                    container=self._container_of(node, frame_label),
                    text=node.label_text(),
                )
            )
        return out

    def _value_of(self, frame: Frame, node: _Node) -> str | None:
        """Current value of a form control.

        The accessibility snapshot reports names and states but not input values, so this
        is the one place perception touches the DOM -- and it touches it *through the ref*,
        never through a selector, so no CSS enters the resolution path.
        """
        if node.role not in {"textbox", "searchbox", "combobox", "spinbutton", "listbox"}:
            return None
        try:
            return frame.locator(f"aria-ref={node.ref}").evaluate(
                """el => {
                    if (el.tagName === 'SELECT') {
                        const o = el.selectedOptions && el.selectedOptions[0];
                        return o ? o.textContent.trim() : '';
                    }
                    return el.value != null ? String(el.value) : null;
                }""",
                timeout=2000,
            )
        except PWError:
            return None

    def _grid_position(self, node: _Node) -> tuple[int | None, str, int | None, tuple[str, ...]]:
        """Locate a cell within its row and its column within the table's header row.

        This is the raw material for a target that survives a change of record. A cell's
        accessible name is its own contents, so the only value-independent identity it has
        is positional-within-labelled-structure: "the Balance column of the row keyed
        Savings, inside Accounts On File". That concept is not web-specific -- it is
        UIA's GridPattern.GetItem and AT-SPI's Table.getAccessibleAt.
        """
        if node.role not in CELL_ROLES:
            return None, "", None, ()
        row = self._enclosing(node, {"row"})
        if row is None:
            return None, "", None, ()
        cells = [c for c in row.children if c.role in CELL_ROLES]
        if node not in cells:
            return None, "", None, ()
        index = cells.index(node)
        row_texts = tuple(_collapse(c.label_text()) for c in cells)

        table = self._enclosing(row, {"table", "grid", "treegrid"})
        header = self._header_row(table, row, len(cells)) if table is not None else None
        column_header = ""
        if header is not None:
            header_cells = [c for c in header.children if c.role in CELL_ROLES]
            if index < len(header_cells):
                column_header = _collapse(header_cells[index].label_text())

        # Row index among the table's rows of this shape, excluding the header row. This is
        # the fallback identity for a row that has no label-like cell to key on.
        row_index = None
        if table is not None:
            peers = [
                r for r in _walk(table)
                if r.role == "row"
                and len([c for c in r.children if c.role in CELL_ROLES]) == len(cells)
                and r is not header
            ]
            if row in peers:
                row_index = peers.index(row)
        return index, column_header, row_index, row_texts

    @staticmethod
    def _header_row(table: _Node, row: _Node, width: int) -> _Node | None:
        """The table's header row, or None when the table has no header.

        A header row is the first row of the same width as the target's, positioned before
        it, all of whose cells are label-like. The label-like test is what keeps a
        label/value table from nominating its own first data row as a header: in
        `Member Name | Dolores A. Kettleman | Member ID | 12345` the cell `12345` is not a
        label, so the row is rejected and the column falls back to its index. Getting this
        wrong would bake a member's name into the "column header" of every target on the
        screen -- the exact failure this whole change exists to remove.
        """
        for candidate in _walk(table):
            if candidate.role != "row":
                continue
            cells = [c for c in candidate.children if c.role in CELL_ROLES]
            if len(cells) != width:
                continue
            if candidate is row:
                return None  # the target's own row is the first of its shape: no header
            texts = [_collapse(c.label_text()) for c in cells]
            if all(t and _is_label_like(t) for t in texts):
                return candidate
            return None
        return None

    @staticmethod
    def _row_keys(el: "_Scanned") -> list[str]:
        """Stable keys for the row this cell sits in, best first.

        A key must identify the ROW without being the value we are reading, so the cell's
        own text is excluded outright. Among the rest we prefer label-like text over
        numbers, dates and identifiers, because a categorical label ("Savings", "Branch")
        is the part of a row that stays put when the record changes, whereas an account
        number or a date is itself per-record data -- stable for one row, useless for
        generalising to the next member.
        """
        own = _collapse(el.text)
        keys = [
            (i, t) for i, t in enumerate(el.row_texts)
            if t and t != own and i != el.column_index and _is_label_like(t)
        ]
        # Rank by word count, then by column index. Field labels are terse ("Branch",
        # "Member ID"); the data beside them is prose ("Dolores A. Kettleman"). Ranking on
        # length before position matters because the target cell is sometimes itself the
        # leftmost cell, in which case pure left-to-right order would nominate the value
        # sitting next to it as the row key -- keying a row on the very data that changes
        # per record, which is the bug this whole rung exists to avoid.
        keys.sort(key=lambda it: (len(it[1].split()), it[0]))
        seen: set[str] = set()
        return [t for _i, t in keys if not (t in seen or seen.add(t))][:2]

    # ---------------------------------------------------------------- proximity labels

    def _derive_label(self, frame: Frame, node: _Node) -> str:
        """Derive an operator-visible label for a control Chromium could not name.

        See the module docstring for why this exists. Rules 1 and 2 read the accessibility
        tree, so they port to UIA and AT-SPI unchanged; rules 3 and 4 are DOM-only and are
        the genuine last resort.
        """
        # 1. The cell to the left. Table-based forms put the label there, and this is the
        #    rule that names essentially every field in the mock bank.
        cell = self._enclosing(node, {"cell", "columnheader", "rowheader"})
        if cell is not None and cell.parent is not None:
            siblings = cell.parent.children
            idx = siblings.index(cell)
            for prev in reversed(siblings[:idx]):
                text = _clean_label(prev.label_text())
                if text:
                    return text[:_MAX_DERIVED_LABEL]

        # 2. The nearest preceding text or named node inside the same block. Covers forms
        #    laid out with <br> and bare text runs rather than table cells.
        if node.parent is not None:
            siblings = node.parent.children
            idx = siblings.index(node)
            for prev in reversed(siblings[:idx]):
                if prev.role in INTERACTIVE_ROLES:
                    break  # another control: we have walked past our own label
                text = _clean_label(prev.label_text())
                if text:
                    return text[:_MAX_DERIVED_LABEL]

        # 3/4. DOM fallbacks, reached through the ref rather than a selector.
        try:
            derived = frame.locator(f"aria-ref={node.ref}").evaluate(
                """el => {
                    const norm = s => (s || '').replace(/\\u00a0/g, ' ')
                                               .replace(/\\s+/g, ' ').trim();
                    // 3. a <label> wrapping the control, or sitting immediately before it
                    const wrap = el.closest('label');
                    if (wrap && norm(wrap.textContent)) return norm(wrap.textContent);
                    let sib = el.previousElementSibling;
                    while (sib) {
                        if (sib.tagName === 'LABEL' && norm(sib.textContent)) {
                            return norm(sib.textContent);
                        }
                        if (norm(sib.textContent)) break;
                        sib = sib.previousElementSibling;
                    }
                    // 4. the control's own name/value attribute
                    return norm(el.getAttribute('name') || el.getAttribute('value') || '');
                }""",
                timeout=2000,
            )
        except PWError:
            return ""
        return _clean_label(derived or "")[:_MAX_DERIVED_LABEL]

    @staticmethod
    def _enclosing(node: _Node, roles: set[str]) -> _Node | None:
        cur = node.parent
        while cur is not None:
            if cur.role in roles:
                return cur
            cur = cur.parent
        return None

    def _container_of(self, node: _Node, frame_label: str) -> str:
        """Nearest labelled ancestor: `"<frame> / <section heading>"`.

        On this kind of app the section heading is the first cell of the first row of the
        enclosing table -- "Search - Primary Index", "BRANCH CONTEXT". That is what lets
        CONTAINER_ROLE_NAME separate two identically-named buttons without falling all the
        way back to a positional ordinal.
        """
        table = self._enclosing(node, {"table", "grid", "treegrid"})
        section = ""
        while table is not None and not section:
            section = self._section_heading(table)
            if section:
                break
            table = self._enclosing(table, {"table", "grid", "treegrid"})
        return f"{frame_label} / {section}" if section else frame_label

    @staticmethod
    def _section_heading(table: _Node) -> str:
        """The heading of a layout table: the FIRST cell of its first row.

        Deliberately strict about position. Taking the first *non-empty* cell anywhere in
        the first row instead reads a right-aligned footer stamp ("Record read from MBRMAST
        at 09:42:11") as though it were a section title, which is both wrong and unstable
        -- it changes on every page load. An action row whose leading cell is empty has no
        heading, and the correct answer is to climb to the enclosing table rather than to
        invent one. The length cap rejects the outermost layout table, whose single cell
        contains the entire screen.
        """
        for row in _walk(table):
            if row.role != "row":
                continue
            for cell in row.children:
                if cell.role not in {"cell", "columnheader", "rowheader"}:
                    continue
                text = _collapse(cell.label_text())
                return text if 0 < len(text) <= 80 else ""
            return ""
        return ""

    # ---------------------------------------------------------------- resolution

    def resolve(self, strategies: Sequence[TargetStrategy]) -> str | None:
        """Try strategies in order; return the first that identifies exactly ONE element.

        Ambiguity is a failure, not a coin toss. If ROLE_NAME, LABEL or TEXT matches more
        than one visible element this returns None *for that strategy* and moves to the
        next one. Silently taking `.first` is how replay-many systems click the wrong
        "Submit" in production and never find out. CONTAINER_ROLE_NAME narrows by section;
        ORDINAL is the one strategy whose entire job is to pick the nth of several matches,
        which is why it sits last in every ladder `stable_target()` emits.

        Returns a fresh ephemeral handle for the *current* tree, so the caller may act on it
        immediately. It is not valid after the next navigation.
        """
        self._rescan()
        for strategy in strategies:
            try:
                handle = self._resolve_one(strategy)
            except PWError:
                handle = None
            if handle is not None:
                return handle
        return None

    def _resolve_one(self, strategy: TargetStrategy) -> str | None:
        kind = strategy.kind

        # The structural rung is dispatched on its payload BEFORE its kind, so it resolves
        # identically whether it rides on the shim or on a native StrategyKind.ROW_COLUMN.
        if kind is ROW_COLUMN_KIND and strategy.value.startswith(_ROW_COLUMN_PREFIX):
            return self._resolve_row_column(strategy, strategy.value[len(_ROW_COLUMN_PREFIX):])
        if _ROW_COLUMN_NATIVE and kind is ROW_COLUMN_KIND:
            return self._resolve_row_column(strategy, strategy.value)

        if kind is StrategyKind.ROLE_NAME:
            role, name = _split_role_name(strategy.value)
            cands = [
                s for s in self._scan
                if s.role == role and not s.derived and s.accessible_name == name and s.visible
            ]
            # Cross-check against Playwright's own role engine, which is the authority on
            # accessible-name computation. If the two disagree, trust the engine's count.
            if self._role_query_count(role, name) != len(cands):
                return None
            return cands[0].handle if len(cands) == 1 else None

        if kind is StrategyKind.LABEL:
            want = _clean_label(strategy.value)
            cands = [
                s for s in self._scan
                if s.visible and s.role in INTERACTIVE_ROLES and _clean_label(s.name) == want
            ]
            return cands[0].handle if len(cands) == 1 else None

        if kind is StrategyKind.CONTAINER_ROLE_NAME:
            role, name = _split_role_name(strategy.value)
            want = _collapse(strategy.container or "").lower()
            cands = [
                s for s in self._scan
                if s.role == role
                and s.visible
                and _clean_label(s.name) == _clean_label(name)
                and (not want or want in _collapse(s.container or "").lower())
            ]
            return cands[0].handle if len(cands) == 1 else None

        if kind is StrategyKind.TEXT:
            want = _collapse(strategy.value)
            cands = [s for s in self._scan if s.visible and _collapse(s.text) == want]
            return cands[0].handle if len(cands) == 1 else None

        if kind is StrategyKind.ORDINAL:
            role, name = _split_role_name(strategy.value)
            want = _clean_label(name)
            cands = [
                s for s in self._scan
                if s.role == role and s.visible and _clean_label(s.name) == want
            ]
            idx = strategy.ordinal or 0
            return cands[idx].handle if 0 <= idx < len(cands) else None

        return None

    def _resolve_row_column(self, strategy: TargetStrategy, payload: str) -> str | None:
        """Resolve a structural target: `"{row_key}|{column_header}"` + container + column index.

        Nothing in the payload is the value being read, which is the whole point -- the
        same target finds member 12345's savings balance and member 23456's, even though
        the two cells contain different numbers.

        Column matching degrades deliberately. When the recorded column header is present
        it wins, because a header survives a column being inserted to its left; only when
        the table has no header does the rung fall back to `strategy.ordinal` as the column
        index. The row key must appear in the row and must NOT be the candidate's own text,
        so a key can never collapse back into the value it is supposed to be independent of.
        """
        row_key, _, column_header = payload.partition("|")
        row_key = _collapse(row_key)
        column_header = _collapse(column_header)
        want_container = _collapse(strategy.container or "").lower()
        if not row_key:
            return None

        cands = []
        for s in self._scan:
            if s.role not in CELL_ROLES or not s.visible or s.column_index is None:
                continue
            if want_container and want_container not in _collapse(s.container or "").lower():
                continue
            own = _collapse(s.text)
            if row_key.startswith("#") and row_key[1:].isdigit():
                if s.row_index != int(row_key[1:]):
                    continue
            elif row_key == own or row_key not in s.row_texts:
                continue
            if column_header:
                if s.column_header != column_header:
                    continue
            elif strategy.ordinal is None or s.column_index != strategy.ordinal:
                continue
            cands.append(s)
        return cands[0].handle if len(cands) == 1 else None

    def _role_query_count(self, role: str, name: str) -> int:
        """How many visible elements Playwright's role engine matches, across all frames."""
        total = 0
        for frame in self.page.frames:
            try:
                loc = frame.get_by_role(role, name=name, exact=True)  # type: ignore[arg-type]
                for i in range(loc.count()):
                    if loc.nth(i).is_visible():
                        total += 1
            except PWError:
                continue
        return total

    # ---------------------------------------------------------------- ref -> artifact

    def stable_target(self, handle: str) -> list[TargetStrategy]:
        """Compile a live ephemeral handle into the ordered ladder of STABLE strategies.

        This is the seam between discovery and replay, and the reason the artifact outlives
        the session that recorded it. Discovery works with refs because refs are exact and
        cheap; but a ref is a position in a tree snapshot, and positions are reassigned the
        moment the screen changes. Writing one into an artifact produces automation that
        addresses the wrong control on the next run *without erroring* -- the worst
        available failure mode against a bank system.

        So nothing leaves this method except portable semantics. The ladder is ordered
        most- to least-durable, and degrades semantic -> structural -> positional:

          ROLE_NAME             Chromium named the control itself and the name is unique on
                                the screen. Survives a re-skin and has a direct UIA/AT-SPI
                                equivalent. Only emitted for a genuinely accessible name.
          LABEL                 The name came from the proximity pass, so no `get_by_role`
                                name lookup can find it. Replay must re-derive it the same
                                way. Emitting LABEL rather than ROLE_NAME is how the
                                derived-ness of the name survives into the artifact.
          CONTAINER_ROLE_NAME   Same control, narrowed by its section. This is what
                                separates two identically-labelled buttons that live in
                                different parts of the screen, and it stays meaningful
                                after unrelated parts of the page change.
          TEXT                  Exact visible text. Weaker: text is duplicated on purpose in
                                legacy screens, and it is the first thing a translation or a
                                copy edit breaks.
          ORDINAL               "The nth control with this role and name." Brittle by
                                construction -- it is the only strategy that is allowed to
                                resolve an ambiguity, and it is always last.

        Strategies that would be ambiguous *at record time* are omitted rather than
        recorded with a caveat: an ambiguous strategy can only ever waste a replay attempt.
        """
        self._rescan()
        el = self._by_handle.get(handle)
        if el is None:
            return []

        ladder: list[TargetStrategy] = []
        role, name = el.role, el.name

        def keep(strategy: TargetStrategy) -> None:
            """Record a rung only if it resolves, right now, to exactly this element.

            Verifying through `_resolve_one` rather than an ad-hoc uniqueness test means the
            recorder and the replayer can never drift apart: a rung is in the artifact
            precisely because the same code path that replay will run has just proved it
            unambiguous. A rung that is already ambiguous at record time can only ever waste
            a replay attempt, so it is dropped rather than recorded with a caveat.
            """
            try:
                if self._resolve_one(strategy) == handle:
                    ladder.append(strategy)
            except PWError:
                pass

        content_named = el.is_content_named()

        # --- structural rungs, FIRST for content-named elements -----------------------
        # A cell's accessible name is its own contents, so every name- or text-based rung
        # below encodes the record that happened to be on screen during discovery. Those
        # rungs make a capability that works for exactly one member. The structural rung
        # names no value at all, so it generalises to the next record, and it therefore
        # leads the ladder whenever the element is content-named.
        if content_named and el.column_index is not None:
            row_keys = self._row_keys(el)
            if not row_keys and el.row_index is not None:
                # No cell in this row is label-like -- every one of them is per-record data
                # (an account number, a balance, a date). The row still has a structural
                # identity: its position within a titled section. Weaker than a keyed row,
                # because inserting a row above shifts it, but it is value-free, which the
                # rungs below are not.
                row_keys = [f"#{el.row_index}"]
            for row_key in row_keys:
                for column, ordinal, conf in (
                    (el.column_header, el.column_index, 0.8),
                    ("", el.column_index, 0.6),
                ):
                    if not column and el.column_header:
                        why = "index fallback for the same cell, in case the header text changes"
                    elif column:
                        why = f"column identified by its header {column!r}, which survives column moves"
                    else:
                        why = "table has no header row, so the column is identified by index"
                    keep(
                        TargetStrategy(
                            kind=ROW_COLUMN_KIND,
                            value=_row_column_payload(row_key, column),
                            container=el.container,
                            ordinal=ordinal,
                            confidence=conf,
                            rationale=(
                                f"Structural: cell in column #{ordinal} of the row keyed {row_key!r}, "
                                f"within {el.container!r} -- {why}. Carries no value, so the same "
                                "target reads the equivalent cell for a different record."
                            ),
                        )
                    )
                    if el.column_header == "":
                        break  # only one spelling exists when there is no header

            # Never persist a content-derived fallback. If no structural strategy resolves,
            # compilation fails loudly instead of storing a member number or balance in the
            # capability and producing a one-record automation.
            return ladder

        # --- semantic rungs ------------------------------------------------------------
        if not el.derived and el.accessible_name:
            keep(
                TargetStrategy(
                    kind=StrategyKind.ROLE_NAME,
                    value=f"{role}|{el.accessible_name}",
                    confidence=0.2 if content_named else 0.95,
                    rationale=(
                        f"Chromium exposes this as {role} named {el.accessible_name!r}, "
                        "unique on screen. Portable to UI Automation and AT-SPI verbatim."
                    ),
                )
            )
        elif el.derived and name:
            keep(
                TargetStrategy(
                    kind=StrategyKind.LABEL,
                    value=name,
                    confidence=0.85,
                    rationale=(
                        f"No accessible name: the visible label {name!r} is adjacent text with "
                        "no for=/id= pairing, so it is recovered by the proximity pass at replay "
                        "time rather than by a role+name lookup."
                    ),
                )
            )

        if el.container:
            keep(
                TargetStrategy(
                    kind=StrategyKind.CONTAINER_ROLE_NAME,
                    value=f"{role}|{name}",
                    container=el.container,
                    confidence=0.18 if content_named else 0.7,
                    rationale=(
                        f"{role} {name!r} scoped to {el.container!r}; disambiguates duplicate "
                        "labels that sit in different frames or sections."
                    ),
                )
            )

        if el.role in _TEXT_ADDRESSABLE and el.text:
            keep(
                TargetStrategy(
                    kind=StrategyKind.TEXT,
                    value=el.text,
                    confidence=0.15 if content_named else 0.45,
                    rationale=(
                        "Exact visible text; unique now but not robust to copy changes."
                    ),
                )
            )

        keep(
            TargetStrategy(
                kind=StrategyKind.ORDINAL,
                value=f"{role}|{name}",
                ordinal=el.ordinal,
                confidence=0.12 if content_named else 0.25,
                rationale=(
                    f"Positional last resort: match #{el.ordinal} of {role} {name!r} in document "
                    "order across frames. The only strategy permitted to resolve ambiguity."
                ),
            )
        )
        return ladder

    # ---------------------------------------------------------------- action

    def act(self, action: Action) -> ActionResult:
        """Perform one verb.

        Never raises on an ordinary interaction failure. The executor above this layer
        turns an `ActionResult(ok=False, ...)` into a typed outcome and evidence; an
        exception escaping here would bypass that and lose the recovery path.
        """
        try:
            if action.kind is ActionKind.NAVIGATE:
                return self._navigate(action)

            if action.kind is ActionKind.WAIT_FOR:
                # Waiting is expressed as a StateCondition evaluated by the replay engine
                # against successive observations, so the surface verb is a no-op by design.
                return ActionResult(ok=True)

            if not action.handle:
                return ActionResult(ok=False, error=f"{action.kind.value} requires a handle")
            locator = self._locator(action.handle)
            if locator is None:
                return ActionResult(ok=False, error=f"handle not resolvable: {action.handle!r}")

            if action.kind is ActionKind.CLICK:
                locator.click(timeout=5000)
                return ActionResult(ok=True)

            if action.kind is ActionKind.TYPE:
                locator.fill(action.text or "", timeout=5000)
                return ActionResult(ok=True)

            if action.kind is ActionKind.SELECT:
                if action.option is None:
                    return ActionResult(ok=False, error="SELECT requires an option label")
                locator.select_option(label=action.option, timeout=5000)
                return ActionResult(ok=True)

            if action.kind is ActionKind.READ:
                return ActionResult(ok=True, read_value=self._read(action.handle, locator))

            return ActionResult(ok=False, error=f"unsupported action kind: {action.kind}")

        except PWError as exc:
            return ActionResult(ok=False, error=f"{type(exc).__name__}: {_first_line(str(exc))}")
        except Exception as exc:  # defensive: the contract is "never raise"
            return ActionResult(ok=False, error=f"{type(exc).__name__}: {_first_line(str(exc))}")

    def _navigate(self, action: Action) -> ActionResult:
        url = action.text or ""
        if not url:
            return ActionResult(ok=False, error="NAVIGATE requires a url in Action.text")
        if self.base_url and url.startswith("/"):
            url = self.base_url + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
        return ActionResult(ok=True)

    def _read(self, handle: str, locator: Any) -> str:
        """Return the input value for a control, or the visible text for anything else."""
        el = self._by_handle.get(handle)
        if el is not None and el.role in {"textbox", "searchbox", "combobox", "spinbutton"}:
            if el.value is not None:
                return el.value
        try:
            return _collapse(locator.inner_text(timeout=3000))
        except PWError:
            try:
                return _collapse(locator.input_value(timeout=3000))
            except PWError:
                return el.name if el is not None else ""

    def _locator(self, handle: str) -> Any:
        """Turn a handle back into a live locator via its ref, scoped to its frame.

        Handles are opaque to callers, so this is the only place the format is parsed. If
        the handle is not in the current index the tree has moved on since it was issued,
        and the honest answer is None rather than a guess at a same-numbered ref -- refs
        are reassigned by position, so a stale ref that still exists points at a different
        control.
        """
        m = _HANDLE_RE.match(handle)
        if not m:
            return None
        frame_index = int(m.group("frame"))
        ref = m.group("ref")

        if handle not in self._by_handle:
            self._rescan()
            if handle not in self._by_handle:
                return None

        frames = self.page.frames
        if frame_index >= len(frames):
            return None
        try:
            loc = frames[frame_index].locator(f"aria-ref={ref}")
            return loc if loc.count() == 1 else None
        except PWError:
            return None

    # ---------------------------------------------------------------- evidence

    def screenshot(self) -> bytes:
        """Full-page PNG. Evidence for a human reviewer, never an input to targeting."""
        return self.page.screenshot(full_page=True)

    def page_source(self) -> str:
        """Raw HTML of the main document plus every child frame.

        Kept strictly for evidence and post-hoc debugging. Nothing in the targeting path
        reads it -- the moment a selector is derived from page source, the portability
        argument in the module docstring stops being true.
        """
        parts: list[str] = []
        for i, frame in enumerate(self.page.frames):
            try:
                parts.append(f"<!-- frame {i}: {frame.name or '(main)'} {frame.url} -->\n{frame.content()}")
            except PWError:
                continue
        return "\n\n".join(parts)


# ======================================================================================
# helpers
# ======================================================================================


def _row_column_payload(row_key: str, column_header: str) -> str:
    """Payload for a structural rung: `"{row_key}|{column_header}"`.

    `column_header` may be empty, in which case the column is carried by
    `TargetStrategy.ordinal` as a 0-based column index. While the rung rides on
    CONTAINER_ROLE_NAME the payload is sentinel-prefixed so it cannot be mistaken for that
    kind's own `"role|name"` grammar; the prefix disappears once StrategyKind.ROW_COLUMN
    exists.
    """
    body = f"{row_key}|{column_header}"
    return body if _ROW_COLUMN_NATIVE else _ROW_COLUMN_PREFIX + body


def _is_label_like(text: str) -> str | bool:
    """True when text reads as a categorical label rather than as a datum.

    Letters-dominant and not parseable as a number. `Savings` and `Balance (CAD)` pass;
    `18,430.09`, `2004-11-02` and `0041-12345-02` do not. Crude on purpose -- it only has
    to separate "words" from "values", and it is applied to text an operator can read.
    """
    t = _collapse(text)
    if not t:
        return False
    letters = sum(1 for c in t if c.isalpha())
    digits = sum(1 for c in t if c.isdigit())
    return letters >= 2 and letters > digits


def _split_role_name(value: str) -> tuple[str, str]:
    """`"button|Search"` -> `("button", "Search")`. Names may themselves contain `|`."""
    role, _, name = value.partition("|")
    return role.strip(), _collapse(name)


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0][:300] if text else ""


def _require_playwright_version() -> None:
    """Fail loudly on a Playwright too old for the API this module is built on.

    Below 1.57 the removed `page.accessibility` API still exists and is a trap; below 1.60
    `aria_snapshot` lacks `mode=`/`boxes=` and would raise something far less legible than
    this message.
    """
    try:
        from importlib.metadata import version as _version

        raw = _version("playwright")
    except Exception:
        return
    parts = []
    for chunk in raw.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts) < MIN_PLAYWRIGHT:
        raise RuntimeError(
            f"playwright>={MIN_PLAYWRIGHT[0]}.{MIN_PLAYWRIGHT[1]} is required "
            f"(found {raw}). aria_snapshot(mode=, boxes=) is unavailable below it, and "
            "page.accessibility was removed in 1.57."
        )
