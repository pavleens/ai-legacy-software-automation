# mock_bank — Meridian Credit Union Member Services Terminal (mock)

A deliberately hostile, legacy-style stand-in for a credit-union back-office application.
It exists so an LLM-driven browser agent can be exercised against a realistic "no clean
DOM" surface, and so failure paths can be triggered on demand.

Entirely local. No real data, no real credentials, no outbound network calls, no database.
All state is a Python dict and two counters in memory.

## Why this app is deliberately hostile

Modern demo apps are a lie. They ship semantic HTML, stable `id`s, `data-testid` hooks and
a single-purpose button per screen, so an agent that "works" against them has only proved
it can read a well-labelled DOM. The applications that actually run bank back offices were
written closer to 2003 than to 2023, and they look like this one: a `<frameset>` splitting
navigation from content, so the page an agent thinks it is on is really three documents and
the URL bar tells you nothing about what is rendered; nested `<table>` elements doing layout
rather than tabulating data, so visual adjacency and DOM adjacency are unrelated; class names
like `f1 c2` and `btn7` that carry no meaning; no `id` attributes on any interactive control,
no `data-*` hooks of any kind, and `<label>` elements that sit next to their field in a table
cell without a `for=`/`id=` pairing, so the label is legible to a human eye and invisible to
an accessibility-tree lookup. Visible text is duplicated on purpose — two "Submit" buttons in
two different frames, "Search" as a nav link, a section heading and a button — so a naive
text matcher has to disambiguate rather than click the first hit. The point is that a browser
agent should have to ground itself the way a teller does, by reading the screen and reasoning
about position and context, not by querying a hook that a friendly developer left behind.
The fault injection exists for the same reason: session interstitials, timeouts, transient
slow loads, permission walls and ugly 500s are the normal texture of these systems, and an
agent's recovery behaviour is only observable if you can summon them on demand.

## Running

```
/opt/anaconda3/bin/python3.12 mock_bank/server.py          # port 8099
PORT=8123 /opt/anaconda3/bin/python3.12 mock_bank/server.py # override
```

Binds `127.0.0.1` only. Dependency: `flask` (see `requirements.txt`).

```
/opt/anaconda3/bin/python3.12 -m pip install --user flask
```

## Routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | 302 redirect to `/app` |
| GET | `/app` | Frameset shell. Left frame `/app/nav`, right frame `/app/search`. No `<body>`; `<noframes>` fallback only. |
| GET | `/app/nav` | Nav frame. Function links, recent-member links, a branch `<select>` with its own **Submit** button. |
| GET | `/app/search` | Member search. Primary index: "Member ID" field + **Search** button. Secondary index: disabled fields + a second **Submit** button. |
| GET | `/app/member?member_id=<id>` | Member detail: name, branch, status, join date, and a table of accounts (number, type, balance, opened). Carries the **Open Sub-Account** button. |
| GET | `/app/subaccount/new?member_id=<id>` | Sub-account application form: account type `<select>`, nickname text field, initial deposit amount, statement delivery radios, **Continue** button. |
| POST, GET | `/app/subaccount/review` | Validates, then renders the review screen showing every entered value with a **Confirm** button (and an **Amend** button back to the form). Validation failures re-render the form with a visible error. |
| POST, GET | `/app/subaccount/confirm` | Confirmation page with a generated reference number, e.g. `Reference: SA-000123`. The counter starts at `SA-000123` and increments per confirmation. |
| GET | `/health` | `{"ok": true}` |
| GET | `/control/fault/<name>` | Arm a sticky fault for the next `/app` request. 400 + the valid list on an unknown name. |
| GET | `/control/fault/clear` | Disarm the sticky fault. |
| GET | `/control/fault/status` | Report the armed fault and the list of available faults. |

Any unmatched path returns the legacy 404 "Record not found" page.

The intended flow is **search -> detail -> action -> review -> confirmation**:
`/app` -> `/app/search` -> `/app/member?member_id=12345` -> `/app/subaccount/new` ->
`/app/subaccount/review` -> `/app/subaccount/confirm`.

## Seed members

Hard-coded in `MEMBERS` in `server.py`. Entirely fictional.

| Member ID | Name | Branch | Status | Accounts |
|---|---|---|---|---|
| `12345` | Dolores A. Kettleman | 0041 Riverbend | ACTIVE | Chequing 1,284.55 · **Savings 18,430.09** · Term Deposit 5,000.00 |
| `23456` | Harold P. Vance | 0012 Mill Street | ACTIVE | Chequing 342.18 · Savings 7,905.63 |
| `34567` | Priya N. Ramanathan | 0041 Riverbend | ACTIVE | Chequing 9,110.40 · Savings 612.00 · Line of Credit -2,300.00 |
| `45678` | Estate of M. O'Doherty | 0007 Old Post Road | RESTRICTED | Savings 44,002.77 |
| `99999` | — | — | — | Always renders the permission-denied page (HTTP 403). |
| anything else | — | — | — | Always renders the record-not-found page (HTTP 404). |

`12345` is the canonical happy-path member: more than two accounts, including a savings
account with a balance.

## Faults

Every fault can be triggered two ways.

1. **Query param** on any `/app` route: `?fault=<name>` (append with `&` if the URL already
   has a query string, e.g. `/app/member?member_id=12345&fault=servererror`).
2. **Control endpoint**, to fire a fault mid-flow rather than by rewriting a URL:
   `GET /control/fault/<name>` arms it, and it is consumed by the **next** `/app` request
   (one-shot). `GET /control/fault/clear` disarms it without spending it. A query-param
   fault takes precedence over an armed sticky fault.

| Fault | Behaviour | HTTP |
|---|---|---|
| `notfound` | Forces the record-not-found path, "ERR-0044: Record not found." | 404 |
| `interstitial` | Serves "Your session will expire soon" ahead of the target page, with a **Continue** button. The interstitial replays the original method and every submitted field as hidden inputs plus `_ack=1`, so clicking Continue proceeds to the real page with the form data intact. | 200 |
| `slow` | Sleeps 3.5 seconds, then serves the real page — a transient slow load, not an error. | 200 |
| `validation` | The sub-account form rejects with the visible error "ERR-0117: Initial deposit must be greater than zero." Applies on `/app/subaccount/new`, `/review` and `/confirm`. | 200 |
| `servererror` | Ugly legacy 500 page: "ABEND S0C4", module/entry/RC/reason/dump-sequence block. | 500 |
| `permission` | "You are not authorised to view this record." | 403 |
| `timeout` | "Your session has timed out. Please sign in again." | 440 |

Organic (non-injected) versions of several of these still fire on their own: member `99999`
gives the 403, an unknown member ID gives the 404, and submitting a deposit of `0`, a
non-numeric deposit, a blank nickname or no account type produces the real validation error.

Examples:

```
curl "http://127.0.0.1:8099/app/member?member_id=12345&fault=servererror"     # 500
curl "http://127.0.0.1:8099/app/subaccount/new?member_id=12345&fault=validation"
curl "http://127.0.0.1:8099/control/fault/interstitial"                        # arm
curl -X POST -d "member_id=12345&acct_type=Savings - Regular&nickname=Roof&deposit=100" \
     "http://127.0.0.1:8099/app/subaccount/review"                             # interstitial fires here
curl "http://127.0.0.1:8099/control/fault/clear"
```

## Files

```
mock_bank/
  server.py            Flask app, seed data, fault injection, control plane
  requirements.txt     flask
  README.md            this file
  templates/
    frameset.html      the <frameset> shell (no <body>)
    base.html          legacy chrome: header bar, crumb bar, footer, ugly class names
    nav.html           left frame
    search.html        member search (two search sections, duplicate button text)
    member.html        member detail + accounts table + Open Sub-Account
    subaccount_form.html  the multi-field application form
    review.html        review/confirm screen
    confirm.html       confirmation with reference number
    notfound.html      ERR-0044 record not found
    permission.html    ERR-0902 not authorised
    timeout.html       ERR-0301 session timed out
    servererror.html   ABEND S0C4 500 page
    interstitial.html  session-expiry interstitial with request replay
```

## Hostility constraints (enforced, greppable)

- `<frameset>` with two frames for the app shell; `<noframes>` fallback only.
- Table-based page layout throughout, not just for the accounts table.
- Zero `id=` attributes in any template. Zero `data-testid`. Zero `data-*` of any kind.
- Zero `aria-*` attributes. Zero `<script>` tags, zero inline event handlers, zero
  external `http(s)` references.
- `<label>` elements are present and visually adjacent in a table cell, but carry no `for=`,
  so they are never programmatically associated with their control.
- Class names are non-semantic: `f1`, `c2`, `c3`, `hd9`, `sb4`, `rw0`, `rw1`, `btn7`,
  `inp3`, `wrn5`, `ft8`.
- Duplicate visible text on purpose: a **Submit** button in the nav frame and another in the
  content frame; "Search" appears as a nav link, as two section headings, and as a button.
- Every control is still reachable by a human: every input has visible adjacent label text
  and every button has visible text. Accessibility by accident, the way real legacy apps are.
