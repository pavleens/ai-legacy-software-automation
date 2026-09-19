# Evidence

Every run in this directory uses generated data from the local `mock_bank` target. The
historical full-fidelity screenshots and observations are committed only because the data
is synthetic; the CLI now requires `--synthetic-evidence` to capture them. The newer
`20260919T231948Z-bbf998` replay demonstrates the privacy-minimized default. Each directory
contains `manifest.json` (run metadata, redacted),
`events.jsonl` (append-only, flushed per event so a crashed run still leaves usable
evidence), `steps/NNN-*/` (observation, action, reasoning, screenshot per step) and
`result.json`.

The discovery runs used the hosted `ollama/gemma4:31b-cloud` model with explicit consent.
That is remote inference, not local inference. Replay runs have no model at all, which is
why their `model` field is empty.

| run | kind | status | member | model |
|---|---|---|---|---|
| `20260919T224820Z-2f1daf` | discovery | **failed** | 12345 | ollama/gemma4:31b-cloud |
| `20260919T224846Z-41c2dd` | discovery | **success** | 12345 | ollama/gemma4:31b-cloud |
| `20260919T224858Z-9291db` | replay | **success** | 12345 | - |
| `20260919T231948Z-bbf998` | replay | **success** | 23456 | - |
| `20260919T224906Z-4d6172` | replay | **business_outcome** | 00000 | - |
| `20260919T224906Z-d3e1d3` | replay | **business_outcome** | 99999 | - |
| `20260919T224913Z-a1169c` | replay | **success** | 12345 | - |
| `20260919T231312Z-cfd901` | replay | **success** | 12345 | - |
| `20260919T225012Z-873685` | replay | **hard_failure** | 12345 | - |
| `20260919T225012Z-cd987f` | replay | **hard_failure** | - | - |

## What each one demonstrates

- **discovery / success** - the required genuine LLM-driven run. The model drove a frameset
  legacy UI with no test IDs, typed a member ID, searched, and read a balance. The compiled
  artifact is `capabilities/lookup_member_balance.json`.
- **discovery / failed** - kept on purpose. The discovery loop is stochastic and a reviewer
  should see what a failed run leaves behind: a failure signal, the observation at the point
  of the stop, and a reason. Replay is deterministic; discovery is not, and pretending
  otherwise would be dishonest.
- **replay / success, member 12345 and 23456** - the same artifact returning two different
  correct balances. This is the proof the recorded targets generalise rather than encoding
  the data that happened to be on screen during discovery.
- **replay / business_outcome, members 00000 and 99999** - `MEMBER_NOT_FOUND` and
  `MEMBER_ACCESS_DENIED`. Both exit 0. A correct answer to a correct question is not a failure.
- **replay / success with a session interstitial injected** - `SESSION_INTERSTITIAL` fired,
  was dismissed within its ceiling, and the run still succeeded. Look for `recovery_fired`
  in `events.jsonl`; the caller was never told, but the auditor can see it.
- **replay / success via human handoff** - policy refused an irreversible step, an
  intervention was published with full context, an operator took the lease on the same live
  session, resolved it as `PERFORMED`, and the automation re-observed and continued without
  re-attempting the irreversible action. The operator returned the declared balance through
  the intervention contract, so the run could not succeed with a missing output. See
  `evidence/interventions/`.
- **replay / hard_failure, invalid_input** - a missing required parameter, rejected against
  the contract before anything touched the surface. Zero steps executed.
- **replay / hard_failure, target_not_found** - a server fault injected mid-flow left the
  surface on an error page. Every rung of the target ladder was tried and reported, with
  what was expected and what was actually observed. This historical trace includes the
  value-bound fallback strategies that exposed the bug; the current compiler refuses to
  persist those fallbacks and the committed capability contains only structural targets.
