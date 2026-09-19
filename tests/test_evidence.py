"""Tests for EvidenceRecorder.

The load-bearing test in this file is `test_sensitive_value_never_hits_disk`. Everything
else checks that the evidence is useful; that one checks that producing it is safe, which
is the constraint that cannot be traded away.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capability_system.evidence.recorder import EvidenceRecorder  # noqa: E402
from capability_system.perception.base import (  # noqa: E402
    Action,
    ActionKind,
    Observation,
    UIElement,
)
from capability_system.replay.outcomes import (  # noqa: E402
    FailureKind,
    ReplayResult,
    StepTrace,
)

SENSITIVE_FIXTURE = "fixture-sensitive-value"
CARD = "4111 1111 1111 1111"


def _observation(n_elements: int = 3, text: str = "ok") -> Observation:
    return Observation(
        url="https://bank.example/members",
        title="Member search",
        elements=[
            UIElement(handle=f"h{i}", role="button", name=f"Control {i}")
            for i in range(n_elements)
        ],
        text_digest=text,
    )


def _all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# -- layout ---------------------------------------------------------------------------


def test_directory_layout(tmp_path: Path) -> None:
    rec = EvidenceRecorder(
        tmp_path, "discovery", "lookup_member_balance",
        synthetic_data=True, capture_screenshots=True,
    )
    step_dir = rec.step(
        1,
        "open_search",
        observation=_observation(),
        action=Action(kind=ActionKind.CLICK, handle="h0"),
        reasoning="The search form is the only entry point to member records.",
        screenshot_bytes=b"\x89PNG\r\n\x1a\nfake",
    )
    run_dir = rec.finish(status="success", result_obj={"capability_id": "lookup_member_balance"})

    assert run_dir == rec.run_dir
    assert run_dir.parent == tmp_path
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "steps").is_dir()

    assert step_dir.name == "001-open-search"
    for fname in ("observation.json", "action.json", "reasoning.txt", "screenshot.png"):
        assert (step_dir / fname).is_file(), fname

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == rec.run_id
    assert manifest["kind"] == "discovery"
    assert manifest["capability_id"] == "lookup_member_balance"
    assert manifest["status"] == "success"
    assert manifest["started_at"].endswith("Z")
    assert manifest["ended_at"].endswith("Z")
    assert manifest["elapsed_ms"] >= 0
    assert manifest["counts"]["steps"] == 1


def test_run_id_is_sortable_and_unique(tmp_path: Path) -> None:
    ids = {EvidenceRecorder(tmp_path, "replay", "c").run_id for _ in range(5)}
    assert len(ids) == 5
    assert all(i == i.strip() and "/" not in i and " " not in i for i in ids)
    # Lexicographic order over the whole id implies chronological order over its prefix.
    stamps = [i.split("-")[0] for i in sorted(ids)]
    assert stamps == sorted(stamps)


def test_reference_does_not_expose_absolute_path(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path / "evidence", "replay", "cap")
    assert rec.reference == f"evidence/{rec.run_id}"
    assert str(tmp_path) not in rec.reference


# -- events.jsonl ---------------------------------------------------------------------


def test_events_jsonl_is_valid_chronological_and_monotonic(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap")
    rec.event("policy_check", decision="allow")
    rec.step(1, "a", observation=_observation())
    rec.event("note", detail="halfway")
    rec.step(2, "b", action=Action(kind=ActionKind.READ, handle="h1"))
    rec.finish(status="success")

    lines = (rec.run_dir / "events.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) >= 6  # run_started + 4 explicit + step events + run_finished
    seqs = [r["seq"] for r in records]
    assert seqs == list(range(1, len(records) + 1))
    timestamps = [r["ts"] for r in records]
    assert timestamps == sorted(timestamps)
    assert all(set(r) >= {"ts", "seq", "kind"} for r in records)
    assert records[0]["kind"] == "run_started"
    assert records[-1]["kind"] == "run_finished"


def test_events_are_flushed_before_finish(tmp_path: Path) -> None:
    """Append-only + per-write flush: a run that never finishes still leaves evidence."""
    rec = EvidenceRecorder(tmp_path, "replay", "cap")
    rec.event("mid_run", detail="process could die right here")
    records = [
        json.loads(line)
        for line in (rec.run_dir / "events.jsonl").read_text().strip().splitlines()
    ]
    assert [r["kind"] for r in records] == ["run_started", "mid_run"]


# -- redaction ------------------------------------------------------------------------


def test_sensitive_value_never_hits_disk(tmp_path: Path) -> None:
    """THE important test: walk every byte under run_dir, the literal must be absent."""
    rec = EvidenceRecorder(
        tmp_path,
        "discovery",
        "transfer_funds",
        sensitive_values={SENSITIVE_FIXTURE},
        sensitive_keys={"passphrase"},
        goal=f"Log in using {SENSITIVE_FIXTURE} and read the balance",
        inputs={"member_id": "M-1", "passphrase": SENSITIVE_FIXTURE},
        model="claude-opus-4",
        synthetic_data=True,
        capture_screenshots=True,
        capture_page_source=True,
    )
    obs = _observation(text=f"Welcome, credential {SENSITIVE_FIXTURE} accepted")
    obs.elements[0].value = SENSITIVE_FIXTURE
    obs.title = f"Session {SENSITIVE_FIXTURE}"

    rec.event("model_call", prompt=f"use {SENSITIVE_FIXTURE}", nested={"deep": [{"k": SENSITIVE_FIXTURE}]})
    rec.step(
        1,
        "login",
        observation=obs,
        action=Action(kind=ActionKind.TYPE, handle="h0", text=SENSITIVE_FIXTURE),
        reasoning=f"Typing the supplied credential {SENSITIVE_FIXTURE} into the password field.",
        screenshot_bytes=b"PNGDATA" + SENSITIVE_FIXTURE.encode() + b"MORE",
    )
    rec.failure_signal(
        observation=obs,
        page_source=f"<input value='{SENSITIVE_FIXTURE}'>",
        screenshot_bytes=SENSITIVE_FIXTURE.encode(),
        note=f"failed while holding {SENSITIVE_FIXTURE}",
    )
    rec.finish(
        status="hard_failure",
        result_obj=ReplayResult.failure(
            "transfer_funds", 1, FailureKind.TARGET_NOT_FOUND, "login",
            expected=f"field containing {SENSITIVE_FIXTURE}", observed=SENSITIVE_FIXTURE,
            trace=[StepTrace(step_id="login", intent="log in", action="type", detail=SENSITIVE_FIXTURE)],
        ),
        extra={"passphrase": SENSITIVE_FIXTURE, "note": SENSITIVE_FIXTURE},
    )

    files = _all_files(rec.run_dir)
    assert files, "no evidence was written"
    needle = SENSITIVE_FIXTURE.encode()
    offenders = [str(p) for p in files if needle in p.read_bytes()]
    assert offenders == [], f"sensitive value leaked into: {offenders}"

    # And the redaction left a visible marker rather than silently deleting content.
    assert "[REDACTED]" in (rec.run_dir / "manifest.json").read_text()


def test_card_number_in_observation_text_is_redacted(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap", synthetic_data=True)
    obs = _observation(text=f"Card on file: {CARD} ending 1111")
    rec.step(1, "read_card", observation=obs)
    rec.finish(status="success")

    raw = (rec.run_dir / "steps" / "001-read-card" / "observation.json").read_text()
    assert CARD not in raw
    assert CARD.replace(" ", "") not in raw
    assert "[REDACTED:CARD]" in raw


def test_sensitive_key_is_redacted_regardless_of_content(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap", sensitive_keys={"pin"})
    rec.event("input_bound", pin="0000", member="Jane")
    rec.finish(status="success")
    text = (rec.run_dir / "events.jsonl").read_text()
    assert '"pin": "[REDACTED]"' in text
    assert "Jane" in text


# -- truncation -----------------------------------------------------------------------


def test_observation_truncation_records_total(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap", element_cap=5)
    rec.step(1, "big_screen", observation=_observation(n_elements=40))
    rec.finish(status="success")

    data = json.loads((rec.run_dir / "steps" / "001-big-screen" / "observation.json").read_text())
    assert len(data["elements"]) == 5
    assert data["elements_total"] == 40
    assert data["elements_truncated"] is True
    assert data["element_cap"] == 5


def test_observation_under_cap_is_not_marked_truncated(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap", element_cap=50)
    rec.step(1, "small", observation=_observation(n_elements=3))
    rec.finish(status="success")
    data = json.loads((rec.run_dir / "steps" / "001-small" / "observation.json").read_text())
    assert data["elements_total"] == 3
    assert data["elements_truncated"] is False
    assert len(data["elements"]) == 3


# -- failure signal -------------------------------------------------------------------


def test_failure_signal_writes_richer_evidence(tmp_path: Path) -> None:
    rec = EvidenceRecorder(
        tmp_path, "replay", "cap", synthetic_data=True,
        capture_screenshots=True, capture_page_source=True,
    )
    rec.failure_signal(
        observation=_observation(),
        screenshot_bytes=b"\x89PNG-bytes",
        page_source="<html><body>error</body></html>",
        note="target not found after all strategies",
    )
    rec.failure_signal(note="second failure")
    rec.finish(status="hard_failure")

    first = rec.run_dir / "failure" / "001"
    assert (first / "observation.json").is_file()
    assert (first / "screenshot.png").read_bytes() == b"\x89PNG-bytes"
    assert "error" in (first / "page_source.txt").read_text()
    assert (first / "note.txt").is_file()
    assert (rec.run_dir / "failure" / "002" / "note.txt").is_file()
    assert json.loads((rec.run_dir / "manifest.json").read_text())["counts"]["failure_signals"] == 2


def test_real_data_mode_omits_pixels_source_and_dynamic_content(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap")
    obs = _observation(text="Customer Jane Doe has $18,430.09")
    obs.url += "?member_id=private-id"
    obs.title = "Jane Doe account"
    obs.elements.append(
        UIElement(handle="cell", role="cell", name="18,430.09", text="18,430.09")
    )
    rec.step(1, "read", observation=obs, screenshot_bytes=b"PNG-sensitive")
    rec.failure_signal(
        observation=obs,
        screenshot_bytes=b"PNG-sensitive",
        page_source="<td>18,430.09</td>",
    )
    rec.finish(status="hard_failure")

    files = {p.name for p in _all_files(rec.run_dir)}
    assert "screenshot.png" not in files
    assert "page_source.txt" not in files
    evidence = "\n".join(
        p.read_text(errors="ignore") for p in _all_files(rec.run_dir)
    )
    assert "Jane Doe" not in evidence
    assert "18,430.09" not in evidence
    assert "private-id" not in evidence
    manifest = json.loads((rec.run_dir / "manifest.json").read_text())
    assert manifest["evidence_policy"]["synthetic_data"] is False


# -- result + context manager ---------------------------------------------------------


def test_result_json_holds_replay_result(tmp_path: Path) -> None:
    result = ReplayResult.business("cap", 2, "MEMBER_NOT_FOUND", "No such member")
    with EvidenceRecorder(tmp_path, "replay", "cap", capability_version=2) as rec:
        rec.finish(status="business_outcome", result_obj=result)
    payload = json.loads((rec.run_dir / "result.json").read_text())
    assert payload["status"] == "business_outcome"
    assert payload["result"]["outcome_code"] == "MEMBER_NOT_FOUND"
    assert payload["capability_version"] == 2


def test_sensitive_output_is_redacted_in_evidence_not_result_object(tmp_path: Path) -> None:
    result = ReplayResult.success("cap", 1, {"balance": "7,905.63"})
    rec = EvidenceRecorder(tmp_path, "replay", "cap", sensitive_keys={"balance"})
    rec.finish(status="success", result_obj=result)
    assert result.outputs["balance"] == "7,905.63"
    stored = (rec.run_dir / "result.json").read_text()
    assert "7,905.63" not in stored
    assert '"balance": "[REDACTED]"' in stored


def test_context_manager_marks_success_when_clean(tmp_path: Path) -> None:
    with EvidenceRecorder(tmp_path, "replay", "cap") as rec:
        rec.event("noop")
    assert json.loads((rec.run_dir / "manifest.json").read_text())["status"] == "completed"


def test_context_manager_records_failure_on_exception(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap")
    with pytest.raises(RuntimeError, match="surface wedged"):
        with rec:
            rec.event("acting")
            raise RuntimeError("surface wedged")

    manifest = json.loads((rec.run_dir / "manifest.json").read_text())
    assert manifest["status"] == "error"
    assert manifest["ended_at"] is not None

    kinds = [
        json.loads(line)["kind"]
        for line in (rec.run_dir / "events.jsonl").read_text().strip().splitlines()
    ]
    assert "exception" in kinds
    assert kinds[-1] == "run_finished"
    assert "surface wedged" in (rec.run_dir / "result.json").read_text()


def test_finish_is_idempotent(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "replay", "cap")
    rec.finish(status="success")
    rec.finish(status="hard_failure")
    assert json.loads((rec.run_dir / "manifest.json").read_text())["status"] == "success"


def test_non_pydantic_objects_do_not_raise(tmp_path: Path) -> None:
    class Plain:
        def __init__(self) -> None:
            self.a = 1
            self.b = "two"

    rec = EvidenceRecorder(tmp_path, "discovery", "cap")
    rec.step(1, "weird", observation=Plain(), action=object())
    rec.finish(status="success", result_obj=Plain())
    assert json.loads((rec.run_dir / "result.json").read_text())["result"]["a"] == 1


def test_step_id_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    rec = EvidenceRecorder(tmp_path, "discovery", "cap")
    step_dir = rec.step(1, "../../etc/passwd", observation=_observation())
    rec.finish(status="success")
    assert rec.run_dir in step_dir.parents
