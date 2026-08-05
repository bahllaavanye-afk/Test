"""NSE as a *signal input* for the India ETFs/ADRs that can actually execute.

The native-India problem is settled and not re-litigated here: Alpaca has no
route to NSE or BSE, so a `.NS` desk would generate confident signals that can
never become orders. `test_india_desk_coverage.py` fails if a `.NS`/`.BO`/`^`
symbol is added to any desk, for exactly that reason.

This is the intermediate play that keeps the information and drops the venue
problem. NSE trades 03:45-10:00 UTC — a window US regular hours never reaches.
By the time the US session opens, the Indian market has already priced a full
day of India-specific news, and INDA/INFY/HDB *are* placeable. So the Indian
session becomes a prior on the US-listed proxy, not a desk of its own.

    NSE close (10:00 UTC)  ──►  state file  ──►  desk run (13:30+ UTC)
    the information                 here            where it can trade

**What the tilt is, precisely.** A bounded nudge to a signal's confidence at
the desk's threshold gate: positive when the signal agrees with the direction
the Indian session moved, negative when it disagrees. It cannot create a
signal, cannot flip a side, and cannot push confidence outside [0, 1]. A 3%
Nifty day saturates it at ±0.06 — enough to move a marginal signal across a
0.60 bar, not enough to drag a weak one there on its own.

**Why a state file and not a live fetch in the desk.** The desk run is the hot
path; adding a yfinance call per India symbol to it buys a network dependency
and a stall risk for data that changes once a day. The file also makes the
input auditable after the fact and makes staleness *visible* — the consumer
re-checks the age and refuses anything past `MAX_AGE_HOURS`, so a workflow
that quietly stops running degrades to no tilt rather than to yesterday's.

The file is rewritten on every run **even when nothing resolves**, with the
reason recorded. A stale file left in place would keep tilting on dead data,
which is the same green-looking-absence failure this codebase has been paying
down all week, in a new place.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = REPO_ROOT / ".github" / "state" / "india_nse_signal.json"
DISCORD_FILE = Path("/tmp/india_nse_discord.md")

# NSE regular hours are 09:15-15:30 IST = 03:45-10:00 UTC. Daily bars carry a
# session *date* and no time, so the close is reconstructed at this hour.
NSE_CLOSE_UTC_HOUR = 10

# A US session reading the same day's NSE close is 3-10 hours behind it; the
# previous session's close (a Monday US open reading Friday's NSE, when Monday
# is an Indian holiday) is ~30. Past that the "overnight" claim is false.
MAX_AGE_HOURS = float(os.environ.get("INDIA_TILT_MAX_AGE_H", "30") or 30)

# tilt = move_fraction * GAIN, clamped to ±MAX. GAIN 2.0 → a 3% session move
# saturates. Deliberately small: this is a prior, not a signal.
TILT_GAIN = float(os.environ.get("INDIA_TILT_GAIN", "2.0") or 2.0)
TILT_MAX = float(os.environ.get("INDIA_TILT_MAX", "0.06") or 0.06)

# Moves this small are noise and get no tilt at all, so the file does not fill
# with ±0.001 entries that read as signal.
MIN_MOVE_PCT = 0.15

# ── The map from where the information is to where it can be traded ──────────
#
# Single names are the strong link: an ADR is a claim on the *same shares* that
# just traded in Mumbai, so the NSE close is close to a direct quote on the US
# listing. Weight 1.0 for all of them.
ADR_MAP: dict[str, tuple[str, str]] = {
    "INFY.NS":      ("INFY", "Infosys"),
    "HDFCBANK.NS":  ("HDB",  "HDFC Bank"),
    "ICICIBANK.NS": ("IBN",  "ICICI Bank"),
    "WIPRO.NS":     ("WIT",  "Wipro"),
    "DRREDDY.NS":   ("RDY",  "Dr Reddy's Labs"),
}

# The index link is weaker and the weights say how much weaker. INDA and INDY
# track large-cap India, which is what the Nifty 50 *is*. EPI re-weights the
# same universe by earnings, so it drifts. SMIN is small-cap — a different
# beta to a large-cap index, and pretending otherwise would overstate the edge.
INDEX_MAP: dict[str, dict[str, float]] = {
    "^NSEI": {"INDA": 1.0, "INDY": 1.0, "EPI": 0.9, "SMIN": 0.6},
}

# MMYT (MakeMyTrip) is deliberately absent: it is a US-listed Indian company
# with no NSE listing, so there is no Indian session close to read. Leaving it
# out is the honest answer; mapping it to the index would invent a link.


def _session_close_utc(session: date) -> datetime:
    return datetime(session.year, session.month, session.day,
                    NSE_CLOSE_UTC_HOUR, 0, tzinfo=timezone.utc)


def session_move_pct(closes: list[tuple[date, float]]) -> tuple[date, float] | None:
    """(session_date, percent move) from the last two daily closes, or None.

    None means "not enough data to say", which the caller must record as a
    skip with a reason — never as a zero move, which would read as "flat".
    """
    usable = [(d, c) for d, c in closes if c and c > 0]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda x: x[0])
    (_, prev), (day, last) = usable[-2], usable[-1]
    return day, (last / prev - 1.0) * 100.0


def tilt_from_move(move_pct: float, weight: float = 1.0) -> float:
    """Bounded confidence nudge for a `move_pct` Indian session, 0 if noise."""
    if abs(move_pct) < MIN_MOVE_PCT:
        return 0.0
    raw = (move_pct / 100.0) * TILT_GAIN * weight
    return round(max(-TILT_MAX, min(TILT_MAX, raw)), 4)


def build_payload(quotes: dict[str, list[tuple[date, float]]],
                  now: datetime | None = None) -> dict:
    """The state file's contents, from raw closes. Pure — no network, no clock
    surprises: `now` is injectable so the freshness logic is testable.

    Every source symbol lands in exactly one of `tilts` or `skipped`. A symbol
    that silently appeared in neither would be indistinguishable from one that
    was never asked for.
    """
    now = now or datetime.now(timezone.utc)
    tilts: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    sessions: set[str] = set()

    def _record(src: str, targets: dict[str, float], label: str) -> None:
        moved = session_move_pct(quotes.get(src, []))
        if moved is None:
            skipped[src] = "fewer than 2 usable daily closes"
            return
        session, move_pct = moved
        age_h = (now - _session_close_utc(session)).total_seconds() / 3600.0
        if age_h > MAX_AGE_HOURS:
            skipped[src] = (f"last close {session.isoformat()} is {age_h:.0f}h old "
                            f"(> {MAX_AGE_HOURS:.0f}h) — not an overnight read")
            return
        if age_h < -1:
            skipped[src] = f"last close {session.isoformat()} is in the future"
            return
        sessions.add(session.isoformat())
        for us_sym, weight in targets.items():
            tilt = tilt_from_move(move_pct, weight)
            if tilt == 0.0:
                skipped[f"{src}->{us_sym}"] = (
                    f"move {move_pct:+.2f}% below the {MIN_MOVE_PCT}% noise floor")
                continue
            # A single name outranks the index on its own ticker. Nothing maps
            # both ways today, but the rule should not depend on that.
            prior = tilts.get(us_sym)
            if prior and prior.get("kind") == "adr" and label != "adr":
                continue
            tilts[us_sym] = {
                "tilt": tilt,
                "source": src,
                "kind": label,
                "move_pct": round(move_pct, 3),
                "weight": weight,
                "session": session.isoformat(),
            }

    for src, (us_sym, _name) in ADR_MAP.items():
        _record(src, {us_sym: 1.0}, "adr")
    for src, targets in INDEX_MAP.items():
        _record(src, targets, "index")

    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sessions": sorted(sessions),
        "max_age_hours": MAX_AGE_HOURS,
        "tilt_max": TILT_MAX,
        "tilts": tilts,
        "skipped": skipped,
    }


# ── Fetch ────────────────────────────────────────────────────────────────────

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=7d&interval=1d"


def _via_chart_api(symbol: str) -> list[tuple[date, float]]:
    """Yahoo's chart endpoint over plain HTTPS — the same data yfinance wraps.

    Not redundancy for its own sake. yfinance ships its own HTTP stack
    (curl_cffi), which fails behind an egress proxy with an SSL reset while a
    normal `requests` call to the identical URL succeeds. Without this, a
    yfinance transport problem would look exactly like a flat Indian day.
    Daily bars are stamped at the session *open* (03:45 UTC), so the date is
    taken from the timestamp and the close reconstructed by `build_payload`.
    """
    import requests
    r = requests.get(CHART_URL.format(sym=symbol),
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"chart API HTTP {r.status_code}")
    res = r.json()["chart"]["result"][0]
    stamps = res.get("timestamp") or []
    closes = res["indicators"]["quote"][0].get("close") or []
    rows: list[tuple[date, float]] = []
    for ts, close in zip(stamps, closes):
        if close is None:
            continue
        rows.append((datetime.fromtimestamp(ts, tz=timezone.utc).date(), float(close)))
    return rows


def fetch_closes(symbols: list[str]) -> dict[str, list[tuple[date, float]]]:
    """Daily closes per symbol, yfinance first and the chart API as fallback.

    Failures are logged per symbol and never raised — one dead ticker must not
    cost the other five. Which path each symbol took is printed, because a
    silent fallback is how you end up not knowing your primary source died.
    """
    out: dict[str, list[tuple[date, float]]] = {}
    try:
        import yfinance as yf
    except ImportError:
        yf = None
        print("⚠ yfinance not installed — using the chart API for every symbol",
              flush=True)

    for sym in symbols:
        rows: list[tuple[date, float]] = []
        why = ""
        if yf is not None:
            try:
                hist = yf.Ticker(sym).history(period="7d", interval="1d")
                rows = [(idx.date(), float(row["Close"])) for idx, row in hist.iterrows()]
            except Exception as exc:
                why = f"yfinance: {type(exc).__name__}"
        if not rows:
            try:
                rows = _via_chart_api(sym)
                if rows:
                    print(f"  ↩ {sym}: chart-API fallback ({why or 'yfinance returned nothing'})",
                          flush=True)
            except Exception as exc:
                print(f"  ⚠ {sym}: both sources failed — {why or 'yfinance empty'}; "
                      f"chart API: {exc}", flush=True)
                continue
        out[sym] = rows
        if rows:
            print(f"  · {sym}: {len(rows)} bars, last {rows[-1][0]} @ {rows[-1][1]:.2f}",
                  flush=True)
        else:
            print(f"  ⚠ {sym}: no bars from either source", flush=True)
    return out


def discord_body(payload: dict) -> str:
    """The desk post. Says what tilted, and — when nothing did — why."""
    tilts = payload.get("tilts", {})
    lines = ["🇮🇳 **NSE overnight read** — bias for tomorrow's India orders"]
    sess = payload.get("sessions") or []
    lines.append(f"_session: {', '.join(sess) if sess else 'none resolved'}_")
    lines.append("")
    if tilts:
        for us_sym, info in sorted(tilts.items(),
                                   key=lambda kv: -abs(kv[1]["tilt"])):
            arrow = "▲" if info["tilt"] > 0 else "▼"
            lines.append(
                f"{arrow} **{us_sym}** `{info['tilt']:+.3f}` "
                f"← {info['source']} {info['move_pct']:+.2f}% ({info['kind']})")
        lines.append("")
        lines.append(f"Tilt is a confidence nudge capped at ±{payload['tilt_max']}. "
                     f"It cannot create a signal or flip a side.")
    else:
        lines.append("**No tilt this run.** Reasons:")
        for src, why in sorted(payload.get("skipped", {}).items())[:12]:
            lines.append(f"· `{src}` — {why}")
        if not payload.get("skipped"):
            lines.append("· nothing was fetched at all — see the workflow log")
    return "\n".join(lines)


def main() -> int:
    symbols = list(ADR_MAP) + list(INDEX_MAP)
    print(f"Fetching {len(symbols)} NSE symbols…", flush=True)
    quotes = fetch_closes(symbols)
    payload = build_payload(quotes)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    DISCORD_FILE.write_text(discord_body(payload))

    n = len(payload["tilts"])
    print(f"\n✓ {n} tilt(s) written to {STATE_FILE.relative_to(REPO_ROOT)}", flush=True)
    for us_sym, info in sorted(payload["tilts"].items()):
        print(f"  {us_sym:<5} {info['tilt']:+.4f}  ← {info['source']} "
              f"{info['move_pct']:+.2f}%", flush=True)
    for src, why in sorted(payload["skipped"].items()):
        print(f"  skip {src}: {why}", flush=True)
    # Exit 0 even with zero tilts: a flat Indian day is a real, correct answer,
    # and failing the workflow for it would train everyone to ignore red.
    return 0


if __name__ == "__main__":
    sys.exit(main())
