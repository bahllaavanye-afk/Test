"""
Self-Proving Paper Trade — places a REAL Alpaca paper trade, waits for the
fill, closes it, and renders a proof image (order + fill + close + P&L) that
is posted to Slack and uploaded as a CI artifact.

This exists because proof of live trading must be produced by CI against the
real Alpaca paper account — never fabricated. If credentials are missing or
the account can't be reached, it says so and exits cleanly (no fake data).

Uses a 24/7 crypto symbol (BTC/USD) by default so it proves a trade at any
hour, including when US equity markets are closed. Override with SYMBOL.

Env:
  ALPACA_API_KEY, ALPACA_SECRET_KEY   — paper account
  ALPACA_BASE_URL                     — default https://paper-api.alpaca.markets
  SLACK_BOT_TOKEN                     — optional, posts proof to #desk-crypto
  SYMBOL                              — default BTC/USD
  NOTIONAL_USD                        — default 50
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

API_KEY = os.environ.get("ALPACA_API_KEY", "").strip()
API_SECRET = os.environ.get("ALPACA_SECRET_KEY", "").strip()
BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE = "https://data.alpaca.markets"
SYMBOL = os.environ.get("SYMBOL", "BTC/USD")
NOTIONAL = float(os.environ.get("NOTIONAL_USD", "50"))
SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "").strip()
IS_CRYPTO = "/" in SYMBOL


def _headers() -> dict:
    return {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET,
            "Content-Type": "application/json"}


def _req(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_error": f"http_{e.code}", "_body": e.read().decode()[:300]}
    except Exception as e:
        return {"_error": str(e)[:300]}


def get_account() -> dict:
    return _req("GET", f"{BASE}/v2/account")


def get_quote() -> float:
    if IS_CRYPTO:
        d = _req("GET", f"{DATA_BASE}/v1beta3/crypto/us/latest/quotes?symbols={urllib.parse.quote(SYMBOL)}")
        q = (d.get("quotes", {}) or {}).get(SYMBOL, {})
        return float(q.get("ap", 0) or 0)
    d = _req("GET", f"{DATA_BASE}/v2/stocks/{SYMBOL}/quotes/latest")
    return float((d.get("quote", {}) or {}).get("ap", 0) or 0)


def place_order(side: str, qty: float | None = None, notional: float | None = None) -> dict:
    body = {"symbol": SYMBOL, "side": side, "type": "market",
            "time_in_force": "gtc" if IS_CRYPTO else "day"}
    if qty is not None:
        body["qty"] = str(qty)
    else:
        body["notional"] = str(round(notional, 2))
    return _req("POST", f"{BASE}/v2/orders", body)


def get_order(oid: str) -> dict:
    return _req("GET", f"{BASE}/v2/orders/{oid}")


def poll_fill(oid: str, timeout_s: int = 60) -> dict:
    """Poll until the order reaches a terminal state or timeout."""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        last = get_order(oid)
        status = last.get("status")
        if status in ("filled", "partially_filled", "canceled", "rejected", "expired"):
            if status != "partially_filled":
                return last
        time.sleep(2)
    return last


def render_proof(steps: list[dict], out: Path) -> None:
    """Render the trade lifecycle as a proof image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # No matplotlib — write a text proof instead
        out.with_suffix(".txt").write_text(json.dumps(steps, indent=2, default=str))
        return

    fig, ax = plt.subplots(figsize=(11, max(5, 1.1 * len(steps) + 2)))
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 2)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ax.text(0.1, len(steps) + 1.3, "QuantEdge — Live Paper Trade Proof",
            fontsize=18, fontweight="bold", color="#e2e8f0")
    ax.text(0.1, len(steps) + 0.7, f"{SYMBOL}   ·   {ts}   ·   Alpaca paper account",
            fontsize=11, color="#94a3b8")

    for i, s in enumerate(steps):
        y = len(steps) - i - 0.3
        ax.text(0.1, y, f"{s['label']}", fontsize=11, fontweight="bold", color="#7dd3fc")
        ax.text(3.4, y, s["detail"][:80], fontsize=10, color="#e2e8f0",
                family="monospace")

    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def slack_upload(path: Path, comment: str) -> bool:
    if not SLACK_TOKEN.startswith("xoxb-"):
        print("  (no SLACK_BOT_TOKEN — proof saved locally only)", flush=True)
        return False

    def call(method, payload):
        req = urllib.request.Request(f"https://slack.com/api/{method}",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {SLACK_TOKEN}",
                     "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    if not path.exists():
        return False
    size = path.stat().st_size
    s1 = call("files.getUploadURLExternal", {"filename": path.name, "length": size})
    if not s1.get("ok"):
        print(f"  getUploadURL failed: {s1.get('error')}", flush=True)
        return False
    try:
        up = urllib.request.Request(s1["upload_url"], data=path.read_bytes(), method="POST")
        urllib.request.urlopen(up, timeout=30).read()
    except Exception as exc:
        print(f"  upload failed: {exc}", flush=True)
        return False
    done = call("files.completeUploadExternal", {
        "files": [{"id": s1["file_id"], "title": "Paper Trade Proof"}],
        "channel_id": _channel_id("desk-crypto"),
        "initial_comment": comment,
    })
    return done.get("ok", False)


def _channel_id(name: str) -> str:
    req = urllib.request.Request(
        "https://slack.com/api/conversations.list?types=public_channel&limit=200",
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
        for ch in data.get("channels", []):
            if ch["name"] == name:
                return ch["id"]
    except Exception:
        pass
    return name


def main() -> int:
    print(f"Self-proving paper trade — {SYMBOL} ${NOTIONAL}", flush=True)
    if not API_KEY or not API_SECRET:
        print("⚠ ALPACA_API_KEY / ALPACA_SECRET_KEY not set — cannot prove a real trade.", flush=True)
        print("  (This is honest: no credentials → no fabricated proof.)", flush=True)
        return 0

    acct = get_account()
    if acct.get("_error"):
        print(f"✗ Account unreachable: {acct['_error']} {acct.get('_body','')}", flush=True)
        return 0
    equity = float(acct.get("equity", 0))
    print(f"  account equity=${equity:,.2f}  status={acct.get('status')}", flush=True)

    steps: list[dict] = [
        {"label": "1. Account", "detail": f"equity=${equity:,.2f} status={acct.get('status')}"},
    ]

    quote = get_quote()
    steps.append({"label": "2. Quote", "detail": f"{SYMBOL} ask=${quote:,.2f}"})
    print(f"  quote ask=${quote:,.2f}", flush=True)

    # ── Open ─────────────────────────────────────────────────────────────────
    if IS_CRYPTO:
        if quote <= 0:
            print("✗ no quote — aborting", flush=True)
            return 0
        qty = round(NOTIONAL / quote, 6)
        opened = place_order("buy", qty=qty)
    else:
        opened = place_order("buy", notional=NOTIONAL)
    if opened.get("_error") or not opened.get("id"):
        print(f"✗ open order rejected: {opened}", flush=True)
        steps.append({"label": "3. Open", "detail": f"REJECTED {opened.get('_error','')}"})
        render_and_post(steps, success=False)
        return 0
    oid = opened["id"]
    print(f"  ► opened order {oid}", flush=True)

    filled = poll_fill(oid)
    fill_px = filled.get("filled_avg_price")
    fill_qty = filled.get("filled_qty")
    steps.append({"label": "3. Open BUY", "detail":
                  f"id={oid[:8]} status={filled.get('status')} "
                  f"qty={fill_qty} @ ${fill_px}"})
    print(f"  fill: status={filled.get('status')} qty={fill_qty} @ {fill_px}", flush=True)

    # ── Close ────────────────────────────────────────────────────────────────
    time.sleep(2)
    if filled.get("status") == "filled" and fill_qty:
        closed = place_order("sell", qty=float(fill_qty))
        if closed.get("id"):
            close_fill = poll_fill(closed["id"])
            cpx = close_fill.get("filled_avg_price")
            steps.append({"label": "4. Close SELL", "detail":
                          f"id={closed['id'][:8]} status={close_fill.get('status')} @ ${cpx}"})
            # realized P&L
            try:
                pnl = (float(cpx) - float(fill_px)) * float(fill_qty)
                steps.append({"label": "5. Realized P&L", "detail": f"${pnl:+.4f} on {SYMBOL}"})
                print(f"  closed @ {cpx}  P&L=${pnl:+.4f}", flush=True)
            except Exception:
                pass
        else:
            steps.append({"label": "4. Close", "detail": f"close rejected: {closed.get('_error','')}"})
    else:
        steps.append({"label": "4. Close", "detail": "not filled — nothing to close"})

    render_and_post(steps, success=True)
    return 0


def render_and_post(steps: list[dict], success: bool) -> None:
    out = REPO_ROOT / "paper_trade_proof.png"
    render_proof(steps, out)
    print(f"  proof rendered: {out} ({out.stat().st_size if out.exists() else 0} bytes)", flush=True)
    comment_lines = [
        f":receipt: *Live Paper Trade Proof* — {SYMBOL}",
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Alpaca paper account",
    ]
    for s in steps:
        comment_lines.append(f"• {s['label']}: `{s['detail']}`")
    slack_upload(out, "\n".join(comment_lines))


if __name__ == "__main__":
    sys.exit(main())
