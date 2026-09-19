"""Run-scoped evidence recording: what the agent did, why, and what it saw when it broke.

WHY THIS EXISTS AT ALL. Discovery is a model improvising against a live banking surface and
replay is unattended automation against the same one. Neither is trustworthy because the
code is careful; they are trustworthy because a human can go back afterwards and check.
That reviewer is not sitting next to the process. They arrive weeks later, with a ticket
number, no browser, no session, and no ability to reproduce the run. So the evidence
directory has to be a complete, self-describing account on its own: the run's identity and
inputs, every decision in order, the observation each decision was made against, and a
richer signal at the point of failure.

THREE DESIGN DECISIONS I WILL DEFEND.

1.  *Append-only, flushed per write.* `events.jsonl` is opened once and every event is
    written and flushed immediately. The obvious alternative -- accumulate a list and
    serialise one tidy JSON document in `finish()` -- produces better-looking output and is
    worthless, because the runs whose evidence you actually need are the ones that did not
    reach `finish()`. A hang, a kill -9, an OOM, a surface that wedges the driver: those are
    exactly the failure modes a reviewer is investigating, and a buffered writer loses all
    of them. JSON Lines rather than JSON because a truncated JSONL file is still parseable
    up to the last complete line, whereas a truncated JSON array is not parseable at all.
    The cost is fsync-less flushes per event, which is nothing next to a browser round trip.

2.  *Redaction at write time, not at read time.* Every string that leaves this module for
    disk goes through `capability_system.safety.policy.redact` first. The tempting
    alternative is to record faithfully and filter when someone views the evidence, which
    is strictly more useful for debugging and is the wrong trade here. This is regulated
    financial data. The moment an unredacted member number is on disk it is in backups, in
    log shipping, in whatever indexes the evidence directory, and outside the control of
    this process forever. A redaction bug at read time leaks; a redaction bug at write time
    only costs debuggability. I would rather lose a detail than lose a member's data, so
    redaction sits at the one choke point everything passes through, and nothing in this
    module writes bytes without going through `_write_json` or `_write_text`.

3.  *Truncation is recorded, never silent.* Observations of a real back-office screen run
    to hundreds of elements. We cap them, but the cap always writes `elements_total`
    alongside, so a reviewer looking at 80 elements knows whether that was the whole screen
    or the first 80 of 412. Evidence that silently omits data is worse than no evidence,
    because it is confidently wrong.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel

from capability_system.safety.policy import redact, redact_mapping

__all__ = ["EvidenceRecorder", "RunKind", "DEFAULT_ELEMENT_CAP"]

RunKind = Literal["discovery", "replay"]

DEFAULT_ELEMENT_CAP = 80
"""Elements kept per observation. Generous enough to contain a realistic screen's
actionable controls, small enough that an evidence directory stays reviewable by hand."""

_REDACTED = "[REDACTED]"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    """ISO8601 with an explicit Z. Evidence gets read across timezones; naive stamps lie."""
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe step directory name.

    Step ids come from a model-authored artifact, so they cannot be trusted to be safe path
    components. Anything outside [a-z0-9-] is collapsed, which also defuses `..` and `/`.
    """
    s = _SLUG_RE.sub("-", (text or "step").lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "step"


def _to_jsonable(obj: Any) -> Any:
    """Normalise anything a caller hands us into JSON-serialisable primitives.

    Pydantic models use `model_dump(mode="json")` so enums, datetimes and nested models
    become primitives the same way they do everywhere else in the system. Non-Pydantic
    objects degrade gracefully rather than raising: an evidence recorder that can crash the
    run it is documenting is a liability, so the worst case here is a `repr`, never an
    exception propagating out of a logging call.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return _iso(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):  # pydantic-like without inheriting BaseModel
        try:
            return _to_jsonable(obj.model_dump(mode="json"))
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "__dict__") and vars(obj):
        return {str(k): _to_jsonable(v) for k, v in vars(obj).items()}
    return repr(obj)


class EvidenceRecorder:
    """Writes one run's evidence directory.

    One instance per run, thread-safe for the event stream because a surface driver and a
    watchdog may both want to record. Cheap to construct; the directory exists and carries a
    manifest from the moment the run starts, so even a run that dies in its first second
    leaves something a reviewer can identify.
    """

    def __init__(
        self,
        root: str | Path,
        kind: RunKind,
        capability_id: str,
        sensitive_values: set[str] | None = None,
        sensitive_keys: set[str] | None = None,
        *,
        capability_version: int | None = None,
        goal: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        model: str | None = None,
        element_cap: int = DEFAULT_ELEMENT_CAP,
        run_id: str | None = None,
        synthetic_data: bool = False,
        capture_screenshots: bool = False,
        capture_page_source: bool = False,
    ) -> None:
        self.kind: RunKind = kind
        self.capability_id = capability_id
        self.capability_version = capability_version
        self.model = model
        self.element_cap = max(1, element_cap)
        self.synthetic_data = synthetic_data
        self.capture_screenshots = capture_screenshots
        self.capture_page_source = capture_page_source

        # The two redaction inputs are kept as instance state rather than passed per call,
        # because "did the caller remember to pass the sensitive set here too?" is precisely
        # the class of mistake that leaks data. There is one set, established at
        # construction, applied to everything.
        self.sensitive_values: set[str] = {v for v in (sensitive_values or set()) if v}
        self.sensitive_keys: set[str] = set(sensitive_keys or set())

        self.run_id = run_id or self._new_run_id()
        self.run_dir = Path(root) / self.run_id
        self.steps_dir = self.run_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = _utc_now()
        self.ended_at: datetime | None = None
        self.status: str = "running"

        self._lock = threading.Lock()
        self._seq = 0
        self._counts: dict[str, int] = {"events": 0, "steps": 0, "failure_signals": 0}
        self._goal = goal
        self._inputs = dict(inputs or {})

        # Opened once for the life of the run, line-buffered, and flushed explicitly after
        # every record. See the module docstring: partial evidence from a killed process is
        # the whole point of this file format.
        self._events_path = self.run_dir / "events.jsonl"
        self._events_fh = self._events_path.open("a", encoding="utf-8")

        self._write_manifest()
        self.event(
            "run_started",
            run_kind=self.kind,
            capability_id=self.capability_id,
            goal=self._goal,
        )

    # -- identity ---------------------------------------------------------------------

    @staticmethod
    def _new_run_id() -> str:
        """Sortable first, unique second.

        `20260919T150723Z-9f3a1c`: lexicographic order equals chronological order, so `ls`
        on the evidence root is already a timeline. The random suffix exists because two
        runs can start inside the same second and evidence must never overwrite evidence.
        """
        return f"{_utc_now().strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"

    @property
    def reference(self) -> str:
        """Portable repository-style reference, never an absolute workstation path."""
        return str(Path(self.run_dir.parent.name) / self.run_dir.name)

    # -- redaction choke point ----------------------------------------------------------

    def _scrub_text(self, text: str) -> str:
        """Single entry point for free text. Patterns plus this run's declared values."""
        return redact(text, self.sensitive_values)

    def _scrub(self, value: Any) -> Any:
        """Recursively redact an already-JSONable structure.

        `redact_mapping` only covers a flat dict of strings, which is not what a serialised
        Observation or ReplayResult looks like. This walks the whole tree so a sensitive key
        buried three levels down in `trace[2].detail` is still caught, and delegates the
        top-level-key rule to `redact_mapping` so there is exactly one definition of
        "sensitive key" in the codebase.
        """
        if isinstance(value, str):
            return self._scrub_text(value)
        if isinstance(value, dict):
            flat = redact_mapping(dict(value), self.sensitive_keys)
            return {k: (v if k in self.sensitive_keys else self._scrub(v))
                    for k, v in flat.items()}
        if isinstance(value, list):
            return [self._scrub(v) for v in value]
        return value

    def _scrub_obj(self, obj: Any) -> Any:
        return self._scrub(_to_jsonable(obj))

    def _scrub_bytes(self, blob: bytes) -> bytes:
        """Byte-level scrub of binary payloads (screenshots).

        A PNG of a screen showing a member number does not contain that number as ASCII, so
        this is not a general defence against pixels -- nothing short of OCR is. What it does
        defend against is the literal appearing in a PNG text chunk, in metadata, or in a
        blob some caller passed through that is not really an image. Corrupting a screenshot
        is an acceptable price for the guarantee "the sensitive literal appears nowhere under
        run_dir"; the reverse trade is not acceptable on regulated data.
        """
        out = blob
        for value in sorted(self.sensitive_values, key=len, reverse=True):
            if len(value) >= 3:
                out = out.replace(value.encode("utf-8", "ignore"), b"[REDACTED]")
        return out

    # -- primitive writers --------------------------------------------------------------

    def _write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._scrub_obj(payload), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path

    def _write_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._scrub_text(text or ""), encoding="utf-8")
        return path

    def _write_bytes(self, path: Path, blob: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._scrub_bytes(blob))
        return path

    # -- manifest -----------------------------------------------------------------------

    def _manifest(self) -> dict[str, Any]:
        elapsed = self.ended_at or _utc_now()
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "goal": self._goal,
            "inputs": self._inputs,
            "model": self.model,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at) if self.ended_at else None,
            "status": self.status,
            "elapsed_ms": int((elapsed - self.started_at).total_seconds() * 1000),
            "counts": dict(self._counts),
            "element_cap": self.element_cap,
            "evidence_policy": {
                "synthetic_data": self.synthetic_data,
                "capture_screenshots": self.capture_screenshots,
                "capture_page_source": self.capture_page_source,
            },
            "redaction": {
                "sensitive_value_count": len(self.sensitive_values),
                "sensitive_keys": sorted(self.sensitive_keys),
            },
        }

    def _write_manifest(self) -> Path:
        """Rewritten on every material change.

        Written at construction and again at finish, so an abandoned run still has a
        manifest saying what it was and that it never reached a terminal status -- which is
        itself the finding a reviewer needs.
        """
        return self._write_json(self.run_dir / "manifest.json", self._manifest())

    # -- public API ---------------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        """Append one redacted record to `events.jsonl` and flush.

        Never raises. A recorder that can break the run it documents would make the system
        less reliable in exchange for observability, which is the wrong direction.
        """
        try:
            with self._lock:
                self._seq += 1
                record = {
                    "ts": _iso(_utc_now()),
                    "seq": self._seq,
                    "kind": kind,
                    **{k: v for k, v in fields.items() if k not in {"ts", "seq", "kind"}},
                }
                line = json.dumps(
                    self._scrub_obj(record), ensure_ascii=False, default=str, sort_keys=False
                )
                self._events_fh.write(line + "\n")
                self._events_fh.flush()
                self._counts["events"] += 1
        except Exception:  # pragma: no cover - defensive
            pass

    def _observation_payload(self, observation: Any) -> dict[str, Any]:
        """Serialise an Observation, capping `elements` and declaring the cap."""
        data = _to_jsonable(observation)
        if not isinstance(data, dict):
            return {"observation": data}
        if not self.synthetic_data:
            def safe_url(value: Any) -> Any:
                if not isinstance(value, str):
                    return value
                parsed = urlsplit(value)
                return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

            data["url"] = safe_url(data.get("url"))
            data["frame_urls"] = [safe_url(url) for url in data.get("frame_urls", [])]
            data["title"] = "[OMITTED: non-synthetic evidence]"
            data["text_digest"] = "[OMITTED: non-synthetic evidence]"
            for element in data.get("elements", []):
                if not isinstance(element, dict):
                    continue
                element["value"] = None
                element["text"] = ""
                element["name"] = "[OMITTED: non-synthetic evidence]"
                element["container"] = None
        elements = data.get("elements")
        if isinstance(elements, list):
            total = len(elements)
            data["elements_total"] = total
            data["elements_truncated"] = total > self.element_cap
            data["element_cap"] = self.element_cap
            if total > self.element_cap:
                data["elements"] = elements[: self.element_cap]
        return data

    def step(
        self,
        index: int,
        step_id: str,
        *,
        observation: Any = None,
        action: Any = None,
        reasoning: str | None = None,
        screenshot_bytes: bytes | None = None,
    ) -> Path:
        """Record one decision point as `steps/NNN-<slug>/`.

        The observation is written as it was *at decision time*, before the action ran. That
        ordering is the reviewable claim: it lets a human ask "given only what the agent
        could see, was this a reasonable action?", which is a different and more useful
        question than "did it work?".
        """
        step_dir = self.steps_dir / f"{index:03d}-{_slug(step_id)}"
        step_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        if observation is not None:
            self._write_json(step_dir / "observation.json", self._observation_payload(observation))
            written.append("observation.json")
        if action is not None:
            self._write_json(step_dir / "action.json", action)
            written.append("action.json")
        if reasoning is not None:
            self._write_text(step_dir / "reasoning.txt", reasoning)
            written.append("reasoning.txt")
        if screenshot_bytes and self.capture_screenshots:
            self._write_bytes(step_dir / "screenshot.png", screenshot_bytes)
            written.append("screenshot.png")

        with self._lock:
            self._counts["steps"] += 1
        self.event(
            "step",
            index=index,
            step_id=step_id,
            dir=step_dir.name,
            files=written,
            action=_to_jsonable(action) if action is not None else None,
        )
        return step_dir

    def failure_signal(
        self,
        *,
        observation: Any = None,
        screenshot_bytes: bytes | None = None,
        page_source: str | None = None,
        note: str = "",
    ) -> None:
        """Capture the richer-than-a-log signal at the moment something went wrong.

        Kept separate from `step` because failure evidence is answering a different question.
        A step record justifies a decision; this one reconstructs a scene. Page source is
        heavy and near-useless on a successful run, so it is collected only here -- that is
        the whole argument for paying for it at all.

        Numbered rather than overwritten: a run can fail, recover, and fail again, and the
        first failure is often the informative one.
        """
        with self._lock:
            self._counts["failure_signals"] += 1
            n = self._counts["failure_signals"]
        fail_dir = self.run_dir / "failure" / f"{n:03d}"
        fail_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        if observation is not None:
            self._write_json(fail_dir / "observation.json", self._observation_payload(observation))
            written.append("observation.json")
        if screenshot_bytes and self.capture_screenshots:
            self._write_bytes(fail_dir / "screenshot.png", screenshot_bytes)
            written.append("screenshot.png")
        if page_source and self.capture_page_source:
            self._write_text(fail_dir / "page_source.txt", page_source)
            written.append("page_source.txt")
        if note:
            self._write_text(fail_dir / "note.txt", note)
            written.append("note.txt")

        self.event("failure_signal", dir=f"failure/{n:03d}", note=note, files=written)

    def finish(
        self,
        *,
        status: str,
        result_obj: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Close the run: write `result.json`, seal the manifest, close the event stream.

        Idempotent, because `__exit__` calls it and a caller may reasonably have called it
        already on the happy path.
        """
        if self.status != "running":
            return self.run_dir

        self.status = status
        self.ended_at = _utc_now()

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": status,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "finished_at": _iso(self.ended_at),
            "result": _to_jsonable(result_obj) if result_obj is not None else None,
        }
        if extra:
            payload["extra"] = _to_jsonable(extra)
        self._write_json(self.run_dir / "result.json", payload)

        self.event("run_finished", status=status, counts=dict(self._counts))
        self._write_manifest()
        try:
            self._events_fh.close()
        except Exception:  # pragma: no cover - defensive
            pass
        return self.run_dir

    # -- context manager ----------------------------------------------------------------

    def __enter__(self) -> "EvidenceRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Never swallows the exception.

        An unhandled crash is a real outcome and the caller's error handling must still see
        it; this only guarantees that the crash is *recorded* before it propagates. The
        exception text is redacted like everything else, because exception messages are one
        of the commonest accidental carriers of input values.
        """
        if exc is not None:
            self.event("exception", exc_type=exc_type.__name__ if exc_type else "Unknown",
                       message=str(exc))
            self.finish(status="error", extra={"exception": f"{exc_type.__name__ if exc_type else 'Unknown'}: {exc}"})
        else:
            self.finish(status=self.status if self.status != "running" else "completed")
        return False
