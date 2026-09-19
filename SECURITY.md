# Security Policy

## Scope

This repository is an educational prototype operating against a simulated legacy banking
and finance application containing generated data. It is not approved for real financial
data or production systems.

## Reporting

Please report vulnerabilities privately through GitHub's security advisory feature. Do not
include credentials, customer data, screenshots of real systems, or exploit details in a
public issue.

## Deployment Boundary

Before production use, add authenticated operator access, encrypted evidence storage,
retention and deletion controls, tenant isolation, signed artifact approvals, dependency
and container scanning, rate limits, audit-log export, and an independent security review.

Remote model use can disclose observed UI state. The CLI therefore requires
`--allow-remote-model` for hosted providers and hosted Ollama models. Full screenshots and
page source require `--synthetic-evidence`; never use that flag with real customer data.

Secrets must be provided through the process environment or a secret manager. Never commit
`.env` files, API keys, access tokens, customer identifiers, or production evidence.
