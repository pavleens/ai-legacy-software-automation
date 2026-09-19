# Computer-Use Capability System

An LLM drives a hostile legacy UI once. The successful run is compiled into a typed,
versioned capability artifact. Approved artifacts then replay deterministically with no
model in the execution path, bounded recovery, policy enforcement, evidence, and human
handoff.

This is an educational prototype built against a synthetic banking application. It is not
production banking software and must not be connected to real customer systems without a
security, privacy, and operational review.

```text
goal -> discovery model -> draft artifact -> human approval -> deterministic replay
              |                                      |
              +-> evidence                           +-> typed result or handoff
```

## Setup

Requires Python 3.11+.

```bash
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
```

For a reproducible environment, install `requirements.lock` instead. Start the synthetic
target in one terminal:

```bash
python3 mock_bank/server.py
curl -s http://127.0.0.1:8099/health
```

The target deliberately uses framesets, table layout, duplicate labels, and no test IDs or
ARIA annotations. Seed member IDs are synthetic: `12345` and `23456` return different
balances, `99999` is denied, and unknown IDs return a business outcome.

## Model Privacy

Ollama with `qwen3:8b` at a loopback host is the default. This is the only configuration
treated as local inference:

```bash
ollama serve
ollama pull qwen3:8b
```

Hosted Ollama tags such as `gemma4:31b-cloud`, non-loopback Ollama hosts, OpenAI, and
Anthropic are remote. Discovery refuses them unless `--allow-remote-model` is supplied.
That flag is explicit consent to send observed UI state to the configured provider.

```bash
export OPENAI_API_KEY=...  # bring your own key; never commit it
python3 -m capability_system.cli discover ... \
  --provider openai --model gpt-4o-mini --allow-remote-model
```

Replay has no provider option and imports no model client.

## Demo

Run discovery against synthetic data:

```bash
python3 -m capability_system.cli discover \
  --goal "Look up a member and read their current savings balance" \
  --url http://127.0.0.1:8099/app \
  --id lookup_member_balance \
  --product mock-core-banking \
  --input member_id=12345 \
  --sensitive member_id \
  --synthetic-evidence
```

`--synthetic-evidence` permits full observations, screenshots, and failure page source. It
must only be used with generated test data. Without it, dynamic content is minimized and
screenshots/page source are not persisted.

Discovery writes a draft to `capabilities/` and a run directory under `evidence/`. Review
the artifact, then approve it as a separate human action:

```bash
python3 -m capability_system.cli approve --id lookup_member_balance
```

Replay the same artifact with multiple inputs:

```bash
python3 -m capability_system.cli replay \
  --id lookup_member_balance --input member_id=12345 --synthetic-evidence

python3 -m capability_system.cli replay \
  --id lookup_member_balance --input member_id=23456 --synthetic-evidence

python3 -m capability_system.cli replay \
  --id lookup_member_balance --input member_id=00000 --synthetic-evidence
```

The first two return different balances through the same structural target. The last exits
successfully with `MEMBER_NOT_FOUND`: a valid business result is not an automation failure.

Exercise recovery and hard-failure paths:

```bash
curl -s http://127.0.0.1:8099/control/fault/interstitial
python3 -m capability_system.cli replay --id lookup_member_balance \
  --input member_id=12345 --synthetic-evidence

curl -s http://127.0.0.1:8099/control/fault/servererror
python3 -m capability_system.cli replay --id lookup_member_balance \
  --input member_id=12345 --synthetic-evidence

curl -s http://127.0.0.1:8099/control/fault/clear
```

Run the end-to-end handoff demo while the mock target is running:

```bash
python3 scripts/demo_escalation.py --headed
```

The file-backed broker is intentionally a stand-in for an operator console. The session
lease, pause, intervention record, operator disposition, operator-supplied output, fresh
observation on reclaim, and prevention of duplicate irreversible actions are real.

## Safety Properties

- Discovery emits `draft`; only the `approve` command promotes an artifact.
- Every action passes through action, origin, path, frame-origin, and risk policy checks.
- Remote model use requires explicit consent.
- Content-derived table cells compile only to value-free structural targets.
- Inputs and required outputs are validated at runtime.
- Recovery is bounded per rule and step, with an absolute re-entry ceiling.
- Real-data evidence omits screenshots, page source, values, and content-derived labels by
  default. Declared sensitive values are redacted at every write boundary.
- Replay returns `success`, `business_outcome`, `escalated`, or a typed `hard_failure`.

## Tests

```bash
python3 -m pytest -q
```

The suite combines fast executor/evidence unit tests with Chromium integration tests
against the real mock application. It verifies cross-origin frame blocking, exact path
prefix boundaries, remote-provider classification, output contracts, privacy-safe evidence,
structural target generalization, recovery ceilings, and handoff behavior.

Run a repository secret scan before publishing:

```bash
gitleaks dir . --no-banner --redact
```

## Layout

```text
capability_system/
  artifact/       typed, versioned capability contract
  discovery/      provider adapters and observe/decide/act compiler
  perception/     portable surface protocol and Playwright implementation
  replay/         deterministic executor, conditions, typed outcomes
  safety/         policy and redaction
  escalation/     session lease, intervention broker, handoff records
  evidence/       append-only, privacy-aware run recorder
capabilities/     reviewed capability artifacts
evidence/         ten curated synthetic runs and their index
mock_bank/        synthetic legacy target with injectable faults
scripts/          end-to-end handoff demo
tests/            unit and browser integration tests
```

See `REPORT.md` for the design rationale and `SECURITY.md` for disclosure and deployment
boundaries.
