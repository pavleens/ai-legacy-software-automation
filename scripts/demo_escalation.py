"""End-to-end demonstration of stuck-detection, handoff, and resume.

WHAT IS REAL HERE AND WHAT IS SIMULATED, STATED PLAINLY.

Real: the replay run, the stuck detection, the intervention request and its contents, the
session lease changing hands, the browser context staying alive and on the same page
throughout, the automation re-observing from scratch on reclaim, and the handoff record
produced by diffing the surface across the transfer.

Simulated: the operator. A real operator would take the browser window and fix the
condition by hand. Here a second process plays that part by repairing the underlying fault
and then releasing the lease through the same `FileBroker.resolve()` call a console would
make. The brief puts a real-time co-browsing console out of scope and asks for the handoff
mechanism and the control-transfer model to be real; those are the parts above the broker,
and the broker is the seam.

Run:  python3 scripts/demo_escalation.py --headed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from capability_system.artifact.schema import (  # noqa: E402
    CapabilityArtifact,
    RiskClass,
)
from capability_system.escalation.control import (  # noqa: E402
    ControlOwner,
    FileBroker,
    InterventionRequest,
    SessionController,
    StuckReason,
)
from capability_system.evidence.recorder import EvidenceRecorder  # noqa: E402
from capability_system.perception.web import WebSurface  # noqa: E402
from capability_system.replay.executor import ReplayExecutor  # noqa: E402
from capability_system.safety.policy import Policy, PolicyEngine  # noqa: E402

BANK = "http://127.0.0.1:8099"
INTERVENTIONS = REPO / "evidence" / "interventions"


def fault(name: str) -> None:
    urllib.request.urlopen(f"{BANK}/control/fault/{name}", timeout=5).read()


def operator(intervention_id: str, delay: float = 3.0) -> subprocess.Popen:
    """A stand-in operator: repair the condition, then release the lease."""
    code = (
        "import time,urllib.request,sys;"
        f"time.sleep({delay});"
        f"urllib.request.urlopen('{BANK}/control/fault/clear',timeout=5).read();"
        "sys.path.insert(0,%r);" % str(REPO) +
        "from capability_system.escalation.control import FileBroker,InterventionResolution;"
        f"FileBroker({str(INTERVENTIONS)!r}).resolve({intervention_id!r},"
        "note='authorised and completed the step myself in the live session',"
        "resolution=InterventionResolution.PERFORMED,"
        "outputs={'savings_balance':'18,430.09'})"
    )
    return subprocess.Popen([sys.executable, "-c", code])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headed", action="store_true",
        help="show Chromium so the session handoff can be watched",
    )
    args = parser.parse_args(argv)
    artifact = CapabilityArtifact.model_validate_json(
        (REPO / "capabilities" / "lookup_member_balance.json").read_text()
    )
    recorder = EvidenceRecorder(
        REPO / "evidence", kind="replay", capability_id=artifact.id,
        capability_version=artifact.version, inputs={"member_id": "12345"},
        synthetic_data=True, capture_screenshots=True, capture_page_source=True,
    )

    # WHY AN IRREVERSIBLE STEP RATHER THAN AN INJECTED FAULT.
    #
    # Two earlier versions of this demo armed a server fault and both reported a
    # misleading success: a one-shot fault armed before the run is consumed by the
    # entry-point navigation, and a timed one loses the race because replay finishes in
    # about a second. More importantly, a fault is the LESS representative trigger. In a
    # bank the routine reason an automation stops and asks for a person is not that the
    # app broke; it is that the next step commits something that cannot be undone and
    # policy will not let it happen unattended. So this demonstrates that path.
    #
    # The capability's final step is marked irreversible for this run only. Policy has
    # `unattended_risky_actions=False` by default, so `check_action` returns
    # REQUIRE_CONFIRMATION and the executor escalates before touching the control.
    print("1. marking the final step irreversible so policy must ask a human")
    artifact.steps[-1].risk = RiskClass.RISKY_IRREVERSIBLE

    with WebSurface(headless=not args.headed, slow_mo_ms=500 if args.headed else 0) as surface:
        controller = SessionController(
            session_id=recorder.run_id,
            broker=FileBroker(INTERVENTIONS),
            observe=surface.observe,
            poll_interval_s=0.5,
            include_observation_details=True,
        )
        executor = ReplayExecutor(
            surface=surface,
            policy=PolicyEngine(Policy(allowed_origins=artifact.allowed_origins)),
            recorder=recorder,
            controller=controller,
        )

        print(f"2. lease holder before the run: {controller.lease.owner.value}")

        # The executor escalates on its own when it gets stuck. To make the demonstration
        # legible rather than racing it, the operator process is started the moment the
        # request file appears.
        watcher = _WatchAndRespond(INTERVENTIONS)
        watcher.start()
        result = executor.run(artifact, {"member_id": "12345"})
        watcher.join(timeout=5)

    recorder.finish(status=result.status.value, result_obj=result)

    print(f"3. result: {result.status.value}")
    if result.escalation_id:
        request = FileBroker(INTERVENTIONS).poll(result.escalation_id)
        if request:
            print(f"   intervention {request.id}")
            print(f"   reason      {request.reason.value}")
            print(f"   step        {request.step_id}  ({request.step_intent})")
            print(f"   detail      {request.detail[:110]}")
            print(f"   operator    {request.operator_note or '(unresolved)'}")
    for handoff in controller.handoffs:
        print(f"4. lease held by human for {handoff.duration_ms} ms; "
              f"navigated={handoff.navigated}; "
              f"elements {handoff.elements_before} -> {handoff.elements_after}")
    print(f"5. lease holder after the run: {controller.lease.owner.value}")
    print(f"   evidence: {recorder.run_dir}")
    return 0


class _WatchAndRespond:
    """Starts the stand-in operator as soon as an intervention request is written."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._proc: subprocess.Popen | None = None
        self._thread = None

    def start(self) -> None:
        import threading
        self._seen = {p.name for p in self.directory.glob("iv_*.json")} if self.directory.exists() else set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.directory.exists():
                new = [p for p in self.directory.glob("iv_*.json") if p.name not in self._seen]
                if new:
                    identifier = new[0].stem
                    print(f"   operator notified of {identifier}, taking the session")
                    self._proc = operator(identifier, delay=2.0)
                    return
            time.sleep(0.25)

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._proc:
            self._proc.wait(timeout=timeout)


if __name__ == "__main__":
    raise SystemExit(main())
