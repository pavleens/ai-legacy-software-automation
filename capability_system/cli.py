"""Command line entry point. Three verbs: discover, replay, approve.

The split is the system's thesis made operable. `discover` costs a model call and produces
a DRAFT. `approve` is a human act. `replay` costs nothing and is the only path an AI agent
should ever invoke in production. Anyone can read those three commands and understand the
lifecycle without reading a line of the implementation.

`replay` deliberately takes no model argument and reads no provider environment variable.
If you cannot configure a model on the production path, you cannot accidentally use one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from capability_system.artifact.schema import (
    ApprovalState, CapabilityArtifact, KnownOutcome, RecoveryRule,
)
from capability_system.discovery.agent import DiscoveryAgent, DiscoveryFailed
from capability_system.discovery.provider import select_provider
from capability_system.escalation.control import FileBroker, SessionController
from capability_system.evidence.recorder import EvidenceRecorder
from capability_system.perception.web import WebSurface
from capability_system.replay.executor import ReplayExecutor
from capability_system.replay.outcomes import ResultStatus
from capability_system.safety.policy import Policy, PolicyEngine

REPO = Path(__file__).resolve().parent.parent
CAPABILITIES = REPO / "capabilities"
EVIDENCE = REPO / "evidence"


def _policy(origins: list[str], allow_risky: bool = False) -> PolicyEngine:
    return PolicyEngine(Policy(allowed_origins=origins, unattended_risky_actions=allow_risky))


def _artifact_path(capability_id: str) -> Path:
    return CAPABILITIES / f"{capability_id}.json"


def _load(capability_id: str) -> CapabilityArtifact:
    path = _artifact_path(capability_id)
    if not path.exists():
        sys.exit(f"No capability at {path}. Run `discover` first.")
    return CapabilityArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _save(artifact: CapabilityArtifact) -> Path:
    CAPABILITIES.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(artifact.id)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    inputs = dict(kv.split("=", 1) for kv in args.input)
    sensitive = set(args.sensitive)
    origin = "/".join(args.url.split("/")[:3])

    provider = select_provider(args.provider, args.model)
    if provider.is_remote and not args.allow_remote_model:
        print(
            "Refusing to send UI observations to a remote model. "
            "Use a local Ollama model or pass --allow-remote-model explicitly.",
            file=sys.stderr,
        )
        return 2
    print(f"provider: {provider.name}")

    recorder = EvidenceRecorder(
        EVIDENCE, kind="discovery", capability_id=args.id,
        sensitive_values={inputs[k] for k in sensitive if k in inputs},
        sensitive_keys=sensitive, goal=args.goal, inputs=inputs, model=provider.name,
        synthetic_data=args.synthetic_evidence,
        capture_screenshots=args.synthetic_evidence,
        capture_page_source=args.synthetic_evidence,
    )

    with WebSurface(headless=not args.headed, slow_mo_ms=args.slow_mo) as surface:
        agent = DiscoveryAgent(
            surface=surface, provider=provider,
            policy=_policy([origin], allow_risky=False),
            recorder=recorder, max_steps=args.max_steps,
        )
        try:
            artifact = agent.discover(
                goal=args.goal, start_url=args.url, capability_id=args.id,
                inputs=inputs, product=args.product, sensitive=sorted(sensitive),
            )
        except DiscoveryFailed as exc:
            recorder.failure_signal(observation=surface.observe(), note=str(exc))
            recorder.finish(status="failed", extra={"reason": exc.reason, "detail": exc.detail})
            print(f"discovery failed ({exc.reason}): {exc.detail}", file=sys.stderr)
            print(f"evidence: {recorder.run_dir}", file=sys.stderr)
            return 1

    artifact.discovery_evidence = str(recorder.run_dir.relative_to(REPO))
    path = _save(artifact)
    recorder.finish(status="success", extra={"artifact": str(path), "steps": len(artifact.steps)})

    print(f"recorded  : {path}")
    print(f"steps     : {len(artifact.steps)}")
    print(f"inputs    : {[p.name for p in artifact.inputs]}")
    print(f"outputs   : {[o.name for o in artifact.outputs]}")
    print(f"approval  : {artifact.approval.value}  (replay requires --allow-draft until approved)")
    print(f"evidence  : {recorder.run_dir}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    artifact = _load(args.id)
    inputs = dict(kv.split("=", 1) for kv in args.input)
    origins = artifact.allowed_origins or ["/".join(args.url.split("/")[:3])] if args.url else artifact.allowed_origins

    recorder = EvidenceRecorder(
        EVIDENCE, kind="replay", capability_id=artifact.id,
        capability_version=artifact.version,
        sensitive_values={inputs[p.name] for p in artifact.inputs
                          if p.sensitive and p.name in inputs},
        sensitive_keys=artifact.sensitive_names(), inputs=inputs,
        synthetic_data=args.synthetic_evidence,
        capture_screenshots=args.synthetic_evidence,
        capture_page_source=args.synthetic_evidence,
    )

    with WebSurface(headless=not args.headed, slow_mo_ms=args.slow_mo) as surface:
        controller = None
        if args.escalate:
            controller = SessionController(
                session_id=recorder.run_id,
                broker=FileBroker(EVIDENCE / "interventions"),
                observe=surface.observe,
                include_observation_details=args.synthetic_evidence,
            )
        executor = ReplayExecutor(
            surface=surface, policy=_policy(origins, allow_risky=args.allow_risky),
            recorder=recorder, controller=controller, tenant_id=args.tenant,
        )
        result = executor.run(artifact, inputs, allow_draft=args.allow_draft)

    recorder.finish(status=result.status.value, result_obj=result)

    print(f"status    : {result.status.value}")
    if result.status is ResultStatus.SUCCESS:
        for key, value in result.outputs.items():
            print(f"  {key} = {value}")
    elif result.status is ResultStatus.BUSINESS_OUTCOME:
        print(f"  outcome : {result.outcome_code}  {result.outcome_description or ''}")
    elif result.status is ResultStatus.ESCALATED:
        print(f"  handed to operator, intervention {result.escalation_id}")
    else:
        print(f"  failure : {result.failure_kind.value if result.failure_kind else '?'}")
        print(f"  step    : {result.failed_step_id}")
        print(f"  expected: {result.expected}")
        print(f"  observed: {result.observed}")
    print(f"steps     : {len(result.trace)}")
    print(f"evidence  : {recorder.run_dir}")

    # A business outcome is a correct answer, so it exits 0. Only a hard failure is an
    # error for a shell caller; escalation is "in flight", which is also not success.
    return {ResultStatus.SUCCESS: 0, ResultStatus.BUSINESS_OUTCOME: 0,
            ResultStatus.ESCALATED: 2, ResultStatus.HARD_FAILURE: 1}[result.status]


def cmd_approve(args: argparse.Namespace) -> int:
    """Promote DRAFT to APPROVED. Deliberately a separate, human command.

    There is no automatic promotion on a successful replay. A capability that promotes
    itself by succeeding once is a capability that has never been reviewed, which defeats
    the entire point of the artifact being readable.
    """
    artifact = _load(args.id)
    artifact.approval = ApprovalState.APPROVED
    path = _save(artifact)
    print(f"approved  : {path}")
    print(f"  {len(artifact.steps)} steps, {len(artifact.known_outcomes)} known outcomes, "
          f"{sum(1 for s in artifact.steps if s.risk.value == 'risky_irreversible')} risky")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """Merge human-authored known outcomes and recovery rules into a capability.

    Separate from `discover` on purpose. A discovery run sees one path, so it cannot know
    which OTHER states the application can legitimately reach. Asking a model to enumerate
    them is asking it to invent failure modes, and an invented outcome code is worse than
    an absent one because callers branch on it. This is institutional knowledge, so a human
    supplies it, at the same moment they review the capability for approval.
    """
    artifact = _load(args.id)
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    known = [KnownOutcome.model_validate(o) for o in data.get("known_outcomes", [])]
    rules = [RecoveryRule.model_validate(r) for r in data.get("recovery_rules", [])]

    existing = {o.code for o in artifact.known_outcomes}
    artifact.known_outcomes += [o for o in known if o.code not in existing]
    existing_rules = {r.code for r in artifact.recovery_rules}
    artifact.recovery_rules += [r for r in rules if r.code not in existing_rules]

    # Annotating changes what the capability promises its caller, so it invalidates any
    # prior approval and bumps the version. A reviewer approved the artifact they read.
    artifact.version += 1
    artifact.approval = ApprovalState.DRAFT

    path = _save(artifact)
    print(f"annotated : {path}")
    print(f"  outcomes: {[o.code for o in artifact.known_outcomes]}")
    print(f"  recovery: {[r.code for r in artifact.recovery_rules]}")
    print(f"  version : {artifact.version}  (reset to draft, re-approve to run unattended)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capability", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="LLM drives the surface once and records a capability")
    d.add_argument("--goal", required=True)
    d.add_argument("--url", required=True)
    d.add_argument("--id", required=True, help="capability id, e.g. lookup_member_balance")
    d.add_argument("--product", default="mock-core-banking")
    d.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
    d.add_argument("--sensitive", action="append", default=[], metavar="NAME")
    d.add_argument("--provider", choices=["ollama", "openai", "anthropic"])
    d.add_argument("--model")
    d.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="explicitly permit sending observed UI state to a hosted model",
    )
    d.add_argument("--max-steps", type=int, default=25)
    d.add_argument("--headed", action="store_true")
    d.add_argument("--slow-mo", type=int, default=0)
    d.add_argument(
        "--synthetic-evidence",
        action="store_true",
        help="allow full observations and screenshots; use only with synthetic test data",
    )
    d.set_defaults(func=cmd_discover)

    r = sub.add_parser("replay", help="deterministic replay, no model in the loop")
    r.add_argument("--id", required=True)
    r.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
    r.add_argument("--url", help="override origin, e.g. a different tenant")
    r.add_argument("--tenant", help="apply a recorded tenant override")
    r.add_argument("--allow-draft", action="store_true")
    r.add_argument("--allow-risky", action="store_true",
                   help="permit unattended irreversible steps. Off by default, on purpose.")
    r.add_argument("--escalate", action="store_true", help="enable human handoff")
    r.add_argument("--headed", action="store_true")
    r.add_argument("--slow-mo", type=int, default=0)
    r.add_argument(
        "--synthetic-evidence",
        action="store_true",
        help="allow full observations and screenshots; use only with synthetic test data",
    )
    r.set_defaults(func=cmd_replay)

    a = sub.add_parser("approve", help="human promotion of a draft capability")
    a.add_argument("--id", required=True)
    a.set_defaults(func=cmd_approve)

    n = sub.add_parser("annotate",
                       help="merge human-authored known outcomes and recovery rules")
    n.add_argument("--id", required=True)
    n.add_argument("--file", required=True, metavar="PATH")
    n.set_defaults(func=cmd_annotate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
