"""Integration tests for `WebSurface`, driven against a real mock_bank process.

These are integration tests on purpose. The whole value of this module is in what
Chromium's accessibility tree actually reports for a `<frameset>` app whose labels are
unassociated table cells, and no unit test with a stubbed tree can tell you whether that
is right. So each test launches the real server, drives a real browser, and asserts on
what a real teller would see.

The fixtures start `mock_bank/server.py` on a free port and tear it down; nothing here
touches a shared port or a shared browser, so the file is safe to run on its own.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

playwright = pytest.importorskip("playwright", reason="playwright is not installed")

from capability_system.artifact.schema import StrategyKind, TargetStrategy  # noqa: E402
from capability_system.perception.base import Action, ActionKind  # noqa: E402
from capability_system.perception.web import (  # noqa: E402
    _ROW_COLUMN_PREFIX,
    ROW_COLUMN_KIND,
    WebSurface,
)

PYTHON = os.environ.get("MOCK_BANK_PYTHON", sys.executable)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def bank_url() -> Iterator[str]:
    """Start mock_bank on a free port, wait for /health, guarantee teardown."""
    port = _free_port()
    env = dict(os.environ, PORT=str(port))
    proc = subprocess.Popen(
        [PYTHON, str(REPO_ROOT / "mock_bank" / "server.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode() if proc.stdout else ""
                pytest.skip(f"mock_bank failed to start: {out[:800]}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(0.15)
        else:
            pytest.skip("mock_bank did not become healthy within 30s")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def surface(bank_url: str) -> Iterator[WebSurface]:
    try:
        s = WebSurface(headless=True, base_url=bank_url)
        s.start()
    except Exception as exc:  # no browser binary, missing driver, etc.
        pytest.skip(f"playwright chromium unavailable: {exc}")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def on_app(surface: WebSurface) -> WebSurface:
    """Every test starts from the frameset shell, so tests cannot leak state into each other."""
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app")).ok
    surface.page.wait_for_timeout(300)
    return surface


def _find(obs, role: str, name: str):
    return [e for e in obs.elements if e.role == role and e.name == name]


# ======================================================================================
# 1. Frameset traversal
# ======================================================================================


def test_observe_reads_child_frames_not_just_the_main_frame(on_app: WebSurface) -> None:
    """`/app` is a `<frameset>`: the main frame has no `<body>` and carries no controls.

    A surface that reads only `page.main_frame` observes literally nothing here. Asserting
    that both the Member ID field (content frame) and a nav-frame link are present proves
    the traversal covers `page.frames`, not just the main document.
    """
    obs = on_app.observe()

    assert obs.elements, "observe() saw nothing: the frameset shell was not traversed"

    member_id = _find(obs, "textbox", "Member ID")
    assert member_id, f"Member ID textbox not perceived. Saw:\n{obs.describe()}"

    search_buttons = _find(obs, "button", "Search")
    assert search_buttons, "Search button not perceived in the content frame"

    # Elements must come from more than one frame, and specifically from the nav frame.
    frames_seen = {e.handle.split("|", 1)[0] for e in obs.elements}
    assert len(frames_seen) >= 2, f"only one frame contributed elements: {frames_seen}"
    assert _find(obs, "link", "Member Search"), "nav-frame link not perceived"

    # The frameset's own title is the main page title.
    assert "Meridian" in obs.title


# ======================================================================================
# 2. The proximity-label pass  -- THE important one
# ======================================================================================


def test_proximity_pass_names_controls_the_a11y_tree_leaves_empty(on_app: WebSurface) -> None:
    """The Member ID input has NO for=/id= pairing and NO aria-label.

    Chromium therefore computes an empty accessible name and
    `get_by_role("textbox", name="Member ID")` matches nothing. Verified here both ways:
    the raw role query finds zero, and the surface still names the control "Member ID" by
    reading the cell to its left. This is what makes role+name targeting viable on a
    legacy screen at all.
    """
    # The hostile precondition: Playwright's own role engine cannot find it by label.
    raw_matches = sum(
        f.get_by_role("textbox", name="Member ID", exact=True).count() for f in on_app.page.frames
    )
    assert raw_matches == 0, (
        "precondition broken: the mock app now associates its label with the input, "
        "so this test no longer proves anything"
    )

    obs = on_app.observe()
    [member_id] = _find(obs, "textbox", "Member ID")
    assert member_id.name == "Member ID"
    assert member_id.enabled
    assert member_id.container and "Primary Index" in member_id.container

    # The same pass names the other unassociated controls on the screen.
    assert _find(obs, "checkbox", "Include closed records")
    surname = _find(obs, "textbox", "Surname")
    assert surname and not surname[0].enabled, "disabled secondary-index field mis-reported"
    assert _find(obs, "textbox", "Postal Code")

    # And it names a control in the other frame, i.e. it is not search-page-specific.
    assert _find(obs, "combobox", "Branch")

    # A derived name must resolve via LABEL, and must NOT masquerade as an accessible name.
    assert on_app.resolve([TargetStrategy(kind=StrategyKind.LABEL, value="Member ID")]) is not None
    assert (
        on_app.resolve(
            [TargetStrategy(kind=StrategyKind.ROLE_NAME, value="textbox|Member ID")]
        )
        is None
    ), "a proximity-derived name must not be emitted as a ROLE_NAME match"


def test_proximity_pass_covers_the_subaccount_form(surface: WebSurface, bank_url: str) -> None:
    """Every field of the multi-field form gets an operator-visible name."""
    assert surface.act(
        Action(kind=ActionKind.NAVIGATE, text="/app/subaccount/new?member_id=12345")
    ).ok
    obs = surface.observe()
    names = {(e.role, e.name) for e in obs.elements}
    for expected in [
        ("combobox", "Account Type"),
        ("textbox", "Nickname"),
        ("textbox", "Initial Deposit Amount"),
        ("button", "Continue"),
    ]:
        assert expected in names, f"{expected} missing. Saw:\n{obs.describe(120)}"

    # Both radios share the label cell to their left, so the proximity pass gives them the
    # same name and they are separated by ordinal. That is the honest reading of the
    # screen: the operator sees one "Statement Delivery" control with two options.
    radios = _find(obs, "radio", "Statement Delivery")
    assert len(radios) == 2
    assert sorted(r.ordinal for r in radios) == [0, 1]


# ======================================================================================
# 3 + 3b. Handle stability vs ref ephemerality
# ======================================================================================


def test_handles_are_stable_across_consecutive_observes_of_an_unchanged_page(
    on_app: WebSurface,
) -> None:
    """Two observes of a page that has not changed must agree.

    Refs are assigned by position in the tree, so an unchanged tree yields an identical
    assignment. This is what makes an observe/act loop coherent: the handle an LLM picked
    out of observation N is still that control at the top of turn N+1.
    """
    first = on_app.observe()
    second = on_app.observe()

    assert [e.handle for e in first.elements] == [e.handle for e in second.elements]
    assert [(e.role, e.name, e.ordinal) for e in first.elements] == [
        (e.role, e.name, e.ordinal) for e in second.elements
    ]


def test_refs_are_ephemeral_but_stable_targets_are_not(surface: WebSurface) -> None:
    """The proof that the ref/artifact separation is real.

    The same control, observed on two different loads of the same screen, is reached by a
    handle whose ref number has moved -- and by an identical ladder of stable strategies.
    Concretely, `f2e19` is the Member ID *textbox* on `/app/search` and the Member ID
    *table cell* on `/app/member`: a ref persisted into an artifact would address a
    different element on the next run and would not error while doing it.

    So: handles are for the live loop, `stable_target()` output is for the artifact.
    """
    # Observation A: the search screen inside the frameset (refs are frame-scoped: f2e...).
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app")).ok
    surface.page.wait_for_timeout(300)
    obs_a = surface.observe()
    [field_a] = _find(obs_a, "textbox", "Member ID")
    ladder_a = surface.stable_target(field_a.handle)

    # Observation B: the same screen loaded standalone. Same control, different tree
    # position, therefore a different ref.
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app/search")).ok
    obs_b = surface.observe()
    [field_b] = _find(obs_b, "textbox", "Member ID")
    ladder_b = surface.stable_target(field_b.handle)

    assert field_a.handle != field_b.handle, (
        "handles did not change across contexts; the ephemerality claim needs re-checking"
    )

    def semantic(ladder):
        return [(s.kind, s.value, s.ordinal) for s in ladder]

    assert semantic(ladder_a) == semantic(ladder_b), (
        f"stable ladder drifted:\n  A={semantic(ladder_a)}\n  B={semantic(ladder_b)}"
    )

    # And the ladder is genuinely stable: it leads with LABEL (never ROLE_NAME, because
    # the name was derived), carries no refs, and ends on the positional last resort.
    assert ladder_a[0].kind is StrategyKind.LABEL
    assert ladder_a[0].value == "Member ID"
    assert ladder_a[-1].kind is StrategyKind.ORDINAL
    assert all("aria-ref" not in s.value for s in ladder_a), "a ref leaked into the artifact ladder"
    assert all(s.rationale for s in ladder_a), "every strategy must carry its reasoning"

    # Round trip: the ladder resolves back to the live control.
    assert surface.resolve(ladder_b) == field_b.handle


# ======================================================================================
# 4. Ambiguity
# ======================================================================================


def test_duplicate_submit_buttons_make_role_name_ambiguous(on_app: WebSurface) -> None:
    """Two "Submit" buttons in two frames. ROLE_NAME must refuse; container/ordinal decide.

    Returning None on ambiguity is the whole point. Silently taking the first match is how
    an agent clicks the wrong Submit and nobody finds out until reconciliation.
    """
    obs = on_app.observe()
    submits = _find(obs, "button", "Submit")
    assert len(submits) == 2, f"expected two Submit buttons, saw {len(submits)}"
    containers = [s.container for s in submits]
    assert containers[0] != containers[1], f"containers did not distinguish them: {containers}"

    assert (
        on_app.resolve([TargetStrategy(kind=StrategyKind.ROLE_NAME, value="button|Submit")])
        is None
    ), "ambiguous ROLE_NAME silently picked a match"

    # TEXT is ambiguous for the same reason and must also refuse.
    assert on_app.resolve([TargetStrategy(kind=StrategyKind.TEXT, value="Submit")]) is None

    # CONTAINER_ROLE_NAME disambiguates by section.
    nav_submit = on_app.resolve(
        [
            TargetStrategy(
                kind=StrategyKind.CONTAINER_ROLE_NAME,
                value="button|Submit",
                container="BRANCH CONTEXT",
            )
        ]
    )
    assert nav_submit == submits[0].handle or nav_submit == submits[1].handle
    assert on_app._by_handle[nav_submit].container and "BRANCH CONTEXT" in (
        on_app._by_handle[nav_submit].container or ""
    )

    # ORDINAL is the strategy whose job is to resolve ambiguity, and it is allowed to.
    for i, expected in enumerate(submits):
        got = on_app.resolve(
            [TargetStrategy(kind=StrategyKind.ORDINAL, value="button|Submit", ordinal=i)]
        )
        assert got == expected.handle

    # A ladder degrades: ambiguous first rung, working second rung.
    ladder = [
        TargetStrategy(kind=StrategyKind.ROLE_NAME, value="button|Submit"),
        TargetStrategy(
            kind=StrategyKind.CONTAINER_ROLE_NAME,
            value="button|Submit",
            container="Secondary Index",
        ),
    ]
    assert on_app.resolve(ladder) is not None

    # The recorded ladder for a duplicate name must skip ROLE_NAME entirely.
    kinds = [s.kind for s in on_app.stable_target(submits[0].handle)]
    assert StrategyKind.ROLE_NAME not in kinds, "an ambiguous ROLE_NAME was recorded"
    assert StrategyKind.CONTAINER_ROLE_NAME in kinds


def test_unambiguous_role_name_still_resolves(on_app: WebSurface) -> None:
    """The negative control: ambiguity-refusal must not break the ordinary case."""
    handle = on_app.resolve(
        [TargetStrategy(kind=StrategyKind.ROLE_NAME, value="button|Search")]
    )
    assert handle is not None
    obs = on_app.observe()
    assert handle in {e.handle for e in obs.elements}


# ======================================================================================
# 5. Happy path
# ======================================================================================


def test_full_drive_search_to_member_detail(on_app: WebSurface) -> None:
    """Navigate, type, click, verify -- entirely through role/name targeting."""
    field = on_app.resolve(
        [
            TargetStrategy(kind=StrategyKind.LABEL, value="Member ID"),
            TargetStrategy(kind=StrategyKind.ORDINAL, value="textbox|Member ID", ordinal=0),
        ]
    )
    assert field is not None
    assert on_app.act(Action(kind=ActionKind.TYPE, handle=field, text="12345")).ok

    # The typed value is perceived back, so a StateCondition could assert on it.
    assert _find(on_app.observe(), "textbox", "Member ID")[0].value == "12345"

    search = on_app.resolve(
        [
            TargetStrategy(kind=StrategyKind.ROLE_NAME, value="button|Search"),
            TargetStrategy(
                kind=StrategyKind.CONTAINER_ROLE_NAME,
                value="button|Search",
                container="Primary Index",
            ),
        ]
    )
    assert search is not None
    assert on_app.act(Action(kind=ActionKind.CLICK, handle=search)).ok
    on_app.page.wait_for_timeout(600)

    obs = on_app.observe()
    assert "Dolores A. Kettleman" in obs.text_digest, f"digest was: {obs.text_digest[:400]}"
    assert len(obs.text_digest) <= 4000
    assert _find(obs, "button", "Open Sub-Account"), "member-detail action not perceived"

    # The nav frame is still perceived after the content frame navigated.
    assert _find(obs, "link", "Member Search")


def test_select_and_navigate_verbs(surface: WebSurface) -> None:
    """SELECT by option label, and NAVIGATE resolving a path against base_url."""
    assert surface.act(
        Action(kind=ActionKind.NAVIGATE, text="/app/subaccount/new?member_id=12345")
    ).ok
    handle = surface.resolve([TargetStrategy(kind=StrategyKind.LABEL, value="Account Type")])
    assert handle is not None
    assert surface.act(
        Action(kind=ActionKind.SELECT, handle=handle, option="Savings - High Interest")
    ).ok
    [combo] = _find(surface.observe(), "combobox", "Account Type")
    assert combo.value == "Savings - High Interest"

    # WAIT_FOR is a deliberate no-op at the surface: waiting is a replay-engine concern.
    assert surface.act(Action(kind=ActionKind.WAIT_FOR)).ok


# ======================================================================================
# 6. READ
# ======================================================================================


def test_read_returns_the_savings_balance(surface: WebSurface) -> None:
    """READ the savings balance off the accounts table using role+name only."""
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app/member?member_id=12345")).ok
    obs = surface.observe()

    handle = surface.resolve(
        [
            TargetStrategy(kind=StrategyKind.ROLE_NAME, value="cell|18,430.09"),
            TargetStrategy(kind=StrategyKind.TEXT, value="18,430.09"),
        ]
    )
    assert handle is not None, f"savings balance cell not resolvable. Saw:\n{obs.describe(150)}"

    result = surface.act(Action(kind=ActionKind.READ, handle=handle))
    assert result.ok and result.error is None
    assert result.read_value == "18,430.09"

    # The cell is scoped to its real section heading, not to the footer stamp that also
    # sits in that table's action row -- the stamp changes every load and would make the
    # container useless as a strategy.
    [cell] = [e for e in obs.elements if e.handle == handle]
    assert cell.container == "Member Detail / Accounts On File", cell.container
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app/member?member_id=12345")).ok
    assert (
        surface.resolve(
            [
                TargetStrategy(
                    kind=StrategyKind.CONTAINER_ROLE_NAME,
                    value="cell|18,430.09",
                    container="Accounts On File",
                )
            ]
        )
        is not None
    )

    # READ on a control returns its value rather than its text.
    assert "Savings" in obs.text_digest


# ======================================================================================
# Failure contract
# ======================================================================================


def test_act_never_raises_on_ordinary_failure(on_app: WebSurface) -> None:
    """The executor turns errors into typed outcomes; an exception would bypass that."""
    dead = on_app.act(Action(kind=ActionKind.CLICK, handle="f9|aria-ref=e999"))
    assert dead.ok is False and dead.error

    assert on_app.act(Action(kind=ActionKind.CLICK)).ok is False
    assert on_app.act(Action(kind=ActionKind.NAVIGATE)).ok is False
    assert on_app.act(Action(kind=ActionKind.NAVIGATE, text="http://127.0.0.1:1/nope")).ok is False

    assert on_app.resolve([TargetStrategy(kind=StrategyKind.ROLE_NAME, value="button|Nope")]) is None
    assert on_app.resolve([]) is None
    assert on_app.stable_target("f9|aria-ref=e999") == []


def test_evidence_helpers(on_app: WebSurface) -> None:
    """Screenshot and page source exist for the audit trail, not for targeting."""
    png = on_app.screenshot()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    src = on_app.page_source()
    assert "frameset" in src.lower() and "Member ID" in src


# ======================================================================================
# 7. Structural targets: a capability must not be bound to the record it was recorded on
# ======================================================================================


def _savings_cell(surface: WebSurface, member_id: str):
    assert surface.act(
        Action(kind=ActionKind.NAVIGATE, text=f"/app/member?member_id={member_id}")
    ).ok
    return surface.observe()


def test_content_named_cell_leads_with_a_value_free_structural_target(
    surface: WebSurface,
) -> None:
    """The regression that matters: a READ target that only works for one member.

    A table cell's accessible name IS its contents, so every name- and text-based strategy
    for the savings balance encodes "18,430.09" -- the number that happened to be on screen
    during discovery. Replaying the same capability for member 23456 then fails
    TARGET_NOT_FOUND, because that member's balance is a different number. A capability
    that works for exactly one record is not a capability.

    So the ladder must LEAD with a rung that names no value at all.
    """
    obs = _savings_cell(surface, "12345")
    [cell] = [e for e in obs.elements if e.name == "18,430.09"]
    ladder = surface.stable_target(cell.handle)
    assert ladder, "no ladder produced for the savings balance cell"

    first = ladder[0]
    assert "18,430.09" not in first.value, f"leading rung is value-bound: {first.value!r}"
    assert "18,430.09" not in (first.container or "")
    assert first.kind is ROW_COLUMN_KIND
    assert first.confidence >= 0.8

    # It identifies the cell by POSITION inside a labelled structure: the row keyed by the
    # account type, the column named by its header, inside the titled section.
    payload = first.value
    if payload.startswith(_ROW_COLUMN_PREFIX):
        payload = payload[len(_ROW_COLUMN_PREFIX):]
    row_key, _, column_header = payload.partition("|")
    assert row_key == "Savings"
    assert column_header == "Balance (CAD)"
    assert first.container == "Member Detail / Accounts On File"
    assert first.ordinal == 2

    # No fallback may persist the value. A stale value-bound rung is both a privacy leak
    # and a target that can silently select the wrong record later.
    serialised = json.dumps([s.model_dump(mode="json") for s in ladder])
    assert "18,430.09" not in serialised


def test_structural_target_generalises_to_a_different_member(surface: WebSurface) -> None:
    """The proof: the target recorded against 12345 reads 23456's balance.

    Same rung, different record, different number. This is the difference between a
    capability and a macro.
    """
    obs = _savings_cell(surface, "12345")
    [cell] = [e for e in obs.elements if e.name == "18,430.09"]
    first = surface.stable_target(cell.handle)[0]

    # Member 23456: savings balance 7,905.63, and only two accounts rather than three.
    _savings_cell(surface, "23456")
    handle = surface.resolve([first])
    assert handle is not None, "structural target did not resolve for member 23456"
    result = surface.act(Action(kind=ActionKind.READ, handle=handle))
    assert result.ok and result.read_value == "7,905.63", result

    # Member 45678: a single-row accounts table, so the row's ordinal position differs too.
    _savings_cell(surface, "45678")
    handle = surface.resolve([first])
    assert handle is not None, "structural target did not resolve for member 45678"
    assert surface.act(Action(kind=ActionKind.READ, handle=handle)).read_value == "44,002.77"

    # And the value-bound rungs genuinely do NOT generalise -- which is why they are last.
    _savings_cell(surface, "23456")
    stale = [
        TargetStrategy(kind=StrategyKind.ROLE_NAME, value="cell|18,430.09"),
        TargetStrategy(kind=StrategyKind.TEXT, value="18,430.09"),
    ]
    assert surface.resolve(stale) is None


def test_label_value_pairs_also_get_a_structural_target(surface: WebSurface) -> None:
    """Not only grids. A two-column label/value table keys on the label to its left."""
    obs = _savings_cell(surface, "12345")
    [name_cell] = [e for e in obs.elements if e.name == "Dolores A. Kettleman"]
    first = surface.stable_target(name_cell.handle)[0]
    assert "Dolores" not in first.value
    assert first.kind is ROW_COLUMN_KIND

    _savings_cell(surface, "23456")
    handle = surface.resolve([first])
    assert handle is not None
    assert surface.act(Action(kind=ActionKind.READ, handle=handle)).read_value == "Harold P. Vance"


def test_row_with_no_label_like_cell_falls_back_to_a_row_index(surface: WebSurface) -> None:
    """Every cell of an accounts row is per-record data except the account type.

    Reading the account type itself therefore has no textual row key available. The row
    still has a value-free identity -- its position inside the titled section -- so a
    structural rung is still emitted rather than dropping straight to the value.
    """
    obs = _savings_cell(surface, "12345")
    [type_cell] = [e for e in obs.elements if e.name == "Chequing"]
    first = surface.stable_target(type_cell.handle)[0]
    assert first.kind is ROW_COLUMN_KIND
    assert "Chequing" not in first.value
    payload = first.value
    if payload.startswith(_ROW_COLUMN_PREFIX):
        payload = payload[len(_ROW_COLUMN_PREFIX):]
    assert payload.startswith("#"), payload

    _savings_cell(surface, "23456")
    handle = surface.resolve([first])
    assert handle is not None
    assert surface.act(Action(kind=ActionKind.READ, handle=handle)).read_value == "Chequing"


def test_non_content_named_elements_are_unchanged(surface: WebSurface) -> None:
    """Buttons, links and inputs are named by a label, not by data. Nothing changes.

    This is the guard against over-correcting: the structural rung must not displace
    ROLE_NAME for a control whose name is a stable caption.
    """
    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app/member?member_id=12345")).ok
    obs = surface.observe()
    [button] = [e for e in obs.elements if e.name == "Open Sub-Account"]
    ladder = surface.stable_target(button.handle)
    assert ladder[0].kind is StrategyKind.ROLE_NAME
    assert ladder[0].confidence == 0.95
    assert all("VALUE-BOUND" not in s.rationale for s in ladder)

    assert surface.act(Action(kind=ActionKind.NAVIGATE, text="/app")).ok
    surface.page.wait_for_timeout(300)
    obs = surface.observe()
    [field] = _find(obs, "textbox", "Member ID")
    assert surface.stable_target(field.handle)[0].kind is StrategyKind.LABEL
