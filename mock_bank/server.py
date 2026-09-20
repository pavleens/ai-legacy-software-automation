"""
SIMULATED LEGACY SOFTWARE -- Synthetic Financial Workflow (MOCK)

A deliberately difficult legacy-software fixture with a synthetic financial
workflow. Framesets, table layout, no id/data-* hooks. Entirely local, with
no real data, credentials, organizations, or outbound network calls.

Run:  /opt/anaconda3/bin/python3.12 mock_bank/server.py
Port: 8099 (override with PORT env var). Binds 127.0.0.1 only.
"""

import os
import time
from flask import Flask, request, render_template, jsonify, redirect, url_for

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Seed data -- no real members, no real balances.
# ---------------------------------------------------------------------------

MEMBERS = {
    "12345": {
        "name": "Dolores A. Kettleman",
        "branch": "0041 - Riverbend",
        "status": "ACTIVE",
        "joined": "1998-03-14",
        "accounts": [
            {"number": "0041-12345-01", "type": "Chequing", "balance": "1,284.55", "opened": "1998-03-14"},
            {"number": "0041-12345-02", "type": "Savings", "balance": "18,430.09", "opened": "2004-11-02"},
            {"number": "0041-12345-07", "type": "Term Deposit", "balance": "5,000.00", "opened": "2019-06-30"},
        ],
    },
    "23456": {
        "name": "Harold P. Vance",
        "branch": "0012 - Mill Street",
        "status": "ACTIVE",
        "joined": "2011-08-21",
        "accounts": [
            {"number": "0012-23456-01", "type": "Chequing", "balance": "342.18", "opened": "2011-08-21"},
            {"number": "0012-23456-04", "type": "Savings", "balance": "7,905.63", "opened": "2013-01-09"},
        ],
    },
    "34567": {
        "name": "Priya N. Ramanathan",
        "branch": "0041 - Riverbend",
        "status": "ACTIVE",
        "joined": "2020-02-17",
        "accounts": [
            {"number": "0041-34567-01", "type": "Chequing", "balance": "9,110.40", "opened": "2020-02-17"},
            {"number": "0041-34567-02", "type": "Savings", "balance": "612.00", "opened": "2020-02-17"},
            {"number": "0041-34567-09", "type": "Line of Credit", "balance": "-2,300.00", "opened": "2022-09-05"},
        ],
    },
    "45678": {
        "name": "Estate of M. O'Doherty",
        "branch": "0007 - Old Post Road",
        "status": "RESTRICTED",
        "joined": "1979-05-30",
        "accounts": [
            {"number": "0007-45678-01", "type": "Savings", "balance": "44,002.77", "opened": "1979-05-30"},
        ],
    },
    # 99999 is intentionally NOT in MEMBERS -- it is hard-routed to the
    # permission-denied page by lookup_member().
}

RESTRICTED_IDS = {"99999"}

ACCOUNT_TYPES = [
    "Savings - Regular",
    "Savings - High Interest",
    "Chequing - Basic",
    "Term Deposit - 12 Month",
    "Youth Savings",
]

# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

FAULTS = (
    "notfound",
    "interstitial",
    "slow",
    "validation",
    "servererror",
    "permission",
    "timeout",
)

# Sticky fault armed via /control/fault/<name>. Consumed by the next
# /app request so a fault can be triggered mid-flow instead of via URL only.
_sticky = {"fault": None}

# Monotonic reference counter for the confirmation page.
_ref_counter = {"n": 122}


def next_reference():
    _ref_counter["n"] += 1
    return "SA-%06d" % _ref_counter["n"]


def active_fault():
    """Query-param fault wins; otherwise consume the sticky fault (one-shot)."""
    q = (request.args.get("fault") or "").strip().lower()
    if q in FAULTS:
        return q
    s = _sticky["fault"]
    if s:
        _sticky["fault"] = None
        return s
    return None


def replay_fields():
    """Every incoming arg/form field, so the interstitial can replay the request."""
    out = []
    for k, v in request.args.items(multi=True):
        if k in ("fault", "_ack"):
            continue
        out.append((k, v))
    for k, v in request.form.items(multi=True):
        if k in ("fault", "_ack"):
            continue
        out.append((k, v))
    return out


def handle_fault(fault):
    """Return a response for the fault, or None to let the real handler run."""
    if fault is None:
        return None
    if fault == "slow":
        time.sleep(3.5)
        return None
    if fault == "interstitial":
        if request.values.get("_ack") == "1":
            return None
        return render_template(
            "interstitial.html",
            target=request.path,
            method=request.method,
            fields=replay_fields(),
        )
    if fault == "notfound":
        return render_template("notfound.html", member_id=request.values.get("member_id", "")), 404
    if fault == "permission":
        return render_template("permission.html", member_id=request.values.get("member_id", "")), 403
    if fault == "timeout":
        return render_template("timeout.html"), 440
    if fault == "servererror":
        return render_template("servererror.html", path=request.path), 500
    if fault == "validation":
        return "VALIDATION"  # sentinel, handled by the sub-account handlers
    return None


def gate():
    """Run fault handling for a request. Returns (response_or_None, is_validation)."""
    f = active_fault()
    r = handle_fault(f)
    if r == "VALIDATION":
        return None, True
    return r, False


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup_member(member_id):
    """Returns ('ok', member) | ('permission', None) | ('notfound', None)."""
    mid = (member_id or "").strip()
    if mid in RESTRICTED_IDS:
        return "permission", None
    if mid in MEMBERS:
        return "ok", MEMBERS[mid]
    return "notfound", None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def root():
    return redirect(url_for("app_shell"))


@app.route("/app")
def app_shell():
    # Frameset shell. No <body>. Deliberately 1998.
    return render_template("frameset.html")


@app.route("/app/nav")
def app_nav():
    r, _ = gate()
    if r is not None:
        return r
    return render_template("nav.html")


@app.route("/app/search")
def app_search():
    r, _ = gate()
    if r is not None:
        return r
    return render_template("search.html", error=None)


@app.route("/app/member")
def app_member():
    r, _ = gate()
    if r is not None:
        return r
    member_id = (request.args.get("member_id") or "").strip()
    if not member_id:
        return render_template("search.html", error="Member ID is required.")
    state, member = lookup_member(member_id)
    if state == "permission":
        return render_template("permission.html", member_id=member_id), 403
    if state == "notfound":
        return render_template("notfound.html", member_id=member_id), 404
    return render_template("member.html", member_id=member_id, member=member)


@app.route("/app/subaccount/new")
def subaccount_new():
    r, validation = gate()
    if r is not None:
        return r
    member_id = (request.args.get("member_id") or "").strip()
    state, member = lookup_member(member_id)
    if state == "permission":
        return render_template("permission.html", member_id=member_id), 403
    if state == "notfound":
        return render_template("notfound.html", member_id=member_id), 404
    err = "Initial deposit must be greater than zero." if validation else None
    return render_template(
        "subaccount_form.html",
        member_id=member_id,
        member=member,
        types=ACCOUNT_TYPES,
        error=err,
        acct_type="",
        nickname="",
        deposit="",
    )


@app.route("/app/subaccount/review", methods=["POST", "GET"])
def subaccount_review():
    r, validation = gate()
    if r is not None:
        return r
    src = request.form if request.method == "POST" else request.args
    member_id = (src.get("member_id") or "").strip()
    acct_type = (src.get("acct_type") or "").strip()
    nickname = (src.get("nickname") or "").strip()
    deposit = (src.get("deposit") or "").strip()

    state, member = lookup_member(member_id)
    if state == "permission":
        return render_template("permission.html", member_id=member_id), 403
    if state == "notfound":
        return render_template("notfound.html", member_id=member_id), 404

    error = None
    if validation:
        error = "Initial deposit must be greater than zero."
    elif not acct_type:
        error = "Account type must be selected."
    elif not nickname:
        error = "Nickname is a required field."
    else:
        try:
            amt = float(deposit.replace(",", ""))
        except ValueError:
            amt = None
        if amt is None:
            error = "Initial deposit must be a numeric amount."
        elif amt <= 0:
            error = "Initial deposit must be greater than zero."

    if error:
        return render_template(
            "subaccount_form.html",
            member_id=member_id,
            member=member,
            types=ACCOUNT_TYPES,
            error=error,
            acct_type=acct_type,
            nickname=nickname,
            deposit=deposit,
        ), 200

    return render_template(
        "review.html",
        member_id=member_id,
        member=member,
        acct_type=acct_type,
        nickname=nickname,
        deposit=deposit,
    )


@app.route("/app/subaccount/confirm", methods=["POST", "GET"])
def subaccount_confirm():
    r, validation = gate()
    if r is not None:
        return r
    src = request.form if request.method == "POST" else request.args
    member_id = (src.get("member_id") or "").strip()
    acct_type = (src.get("acct_type") or "").strip()
    nickname = (src.get("nickname") or "").strip()
    deposit = (src.get("deposit") or "").strip()

    state, member = lookup_member(member_id)
    if state == "permission":
        return render_template("permission.html", member_id=member_id), 403
    if state == "notfound":
        return render_template("notfound.html", member_id=member_id), 404

    if validation:
        return render_template(
            "subaccount_form.html",
            member_id=member_id,
            member=member,
            types=ACCOUNT_TYPES,
            error="Initial deposit must be greater than zero.",
            acct_type=acct_type,
            nickname=nickname,
            deposit=deposit,
        )

    return render_template(
        "confirm.html",
        member_id=member_id,
        member=member,
        acct_type=acct_type,
        nickname=nickname,
        deposit=deposit,
        reference=next_reference(),
    )


# ---------------------------------------------------------------------------
# Control plane (test harness only -- not part of the simulated app)
# ---------------------------------------------------------------------------

@app.route("/control/fault/clear")
def control_clear():
    _sticky["fault"] = None
    return jsonify({"ok": True, "fault": None})


@app.route("/control/fault/status")
def control_status():
    return jsonify({"ok": True, "fault": _sticky["fault"], "available": list(FAULTS)})


@app.route("/control/fault/<name>")
def control_set(name):
    name = (name or "").strip().lower()
    if name not in FAULTS:
        return jsonify({"ok": False, "error": "unknown fault", "available": list(FAULTS)}), 400
    _sticky["fault"] = name
    return jsonify({"ok": True, "fault": name, "armed_for": "next /app request"})


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.errorhandler(404)
def not_found(e):
    return render_template("notfound.html", member_id=""), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8099"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
