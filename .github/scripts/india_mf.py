"""Indian mutual funds — AMFI NAV feed, investable universe, momentum ranking.

WHY MUTUAL FUNDS ARE NOT A NORMAL DESK
--------------------------------------
Indian MFs price once a day at NAV. There is no intraday quote, no bid/ask, no
partial fill. An order placed before the cutoff transacts at that day's NAV and
settles T+1 (equity) or T+2. So the 50-minute desk cadence is meaningless here:
this runs **once daily** and its output is a ranked list, not a market order.

That also makes a signal-only start honest, unlike the Polymarket desk. A ranked
MF list is directly actionable by a human or by a broker API added later; it is
not a confident signal for an instrument that can never be traded.

DATA (free, no credentials)
---------------------------
AMFI publishes every scheme's NAV daily. Both endpoints verified 2026-08-05:

  https://portal.amfiindia.com/spages/NAVAll.txt          14,238 schemes, 52 AMCs
  .../DownloadNAVHistoryReport_Po.aspx?frmdt=&todt=&mf=   35 dates for one AMC

Note the `www.amfiindia.com` host 302-redirects to `portal.amfiindia.com`; the
redirect must be followed or the payload is a 169-byte HTML stub.

UNIVERSE
--------
Direct + Growth only, deliberately:
  * **Direct** plans carry no distributor commission — typically 0.5-1.0% lower
    expense ratio than the Regular plan of the same scheme. Ranking both would
    fill the list with strictly-worse duplicates.
  * **Growth** option only: IDCW (dividend) options pay out of NAV, so their NAV
    series has discontinuities that look exactly like negative returns.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
STATE_FILE = _REPO / ".github" / "state" / "india_mf.json"

NAV_ALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
NAV_HIST_URL = ("https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"
                "?frmdt={frm}&todt={to}&mf={mf}")
_TIMEOUT = 90
_UA = {"User-Agent": "Mozilla/5.0 (compatible; QuantEdge/1.0)"}

# Keep the persisted history bounded — see the 2026-08-05 agent_memory.json
# entry, where an uncapped state file became 47% of the git repository.
HISTORY_KEEP = 120

# AMFI's `mf=` history parameter, resolved by probe on 2026-08-05 (the codes are
# not published anywhere machine-readable). These are the large AMCs by AUM; the
# ranking is computed within them rather than across all 52, because each AMC is
# one HTTP request and 52 daily requests against a free public endpoint is
# inconsiderate for very little added signal.
AMC_CODES: dict[int, str] = {
    9:  "HDFC",
    22: "SBI",
    21: "Nippon India",
    3:  "Aditya Birla Sun Life",
    53: "Axis",
    28: "UTI",
    27: "Franklin India",
}


@dataclass
class Scheme:
    code: str
    name: str
    isin: str
    nav: float
    date: str
    amc: str
    category: str = "?"

    @property
    def is_direct(self) -> bool:
        return bool(re.search(r"\bdirect\b", self.name, re.I))

    @property
    def is_growth(self) -> bool:
        """Growth option — excludes IDCW/dividend variants.

        Matching on the ABSENCE of IDCW/dividend rather than the presence of
        'growth': many schemes name the growth option implicitly, and requiring
        the word drops them.
        """
        return not re.search(
            r"\bIDCW\b|\bdividend\b|\bdiv\b"
            # AMFI also spells IDCW out in full, and that form defeated the
            # abbreviation match: the 2026-08-05 live run ranked "SBI Innovative
            # Opportunities Fund - Direct Plan - Growth" and "... - Income
            # Distribution cum Capital Withdrawal" as two separate 13.58%
            # entries, plus the same for SBI Automotive. Same portfolio, two
            # slots, and the payout variant is the wrong one to hold.
            r"|income\s+distribution"
            # AMFI TRUNCATES "IDCW" on some rows. Observed 2026-08-05:
            #   Axis Conservative Hybrid - Regular Plan - Half Yearly IDCW  (caught)
            #   Axis Conservative Hybrid - Direct Plan  - Half Yearly       (missed)
            # Same payout option, one spelling. A payout FREQUENCY at the end of
            # the name is the tell: a growth option does not distribute, so it
            # has no frequency. Anchored to the end so an interval fund with
            # "Quarterly" mid-name is not swept up.
            r"|[-\s](annual|half\s*yearly|quarterly|monthly|fortnightly|weekly|daily)"
            r"(\s+option)?\s*$",
            self.name, re.I)


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_navall(text: str) -> list[Scheme]:
    """Parse AMFI's NAVAll.txt.

    Format is sectioned: AMC name on a bare line, then `;`-delimited scheme rows
    beneath it, with blank lines and a 'Open Ended Schemes(...)' category header
    interleaved. The AMC has to be tracked as state while scanning — it is not
    on the scheme row.
    """
    out: list[Scheme] = []
    amc = "?"
    category = "?"
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if ";" not in s:
            # Two kinds of bare line: the AMC, and AMFI's scheme-category header
            # ("Open Ended Schemes(Equity Scheme - Large Cap)"). The category was
            # discarded until 2026-08-05, which made the ranking a sector-rotation
            # readout: the live top 10 was transportation, healthcare, technology
            # and automotive funds — "which sector ran", not "which manager is
            # good". Both are section state; neither is on the scheme row.
            low = s.lower()
            if low.startswith("open ended") or low.startswith("close ended"):
                category = s
            else:
                amc = s
                # NOTE the nesting: the real file is category -> AMC -> rows, so
                # an AMC line must NOT reset the category. An earlier version
                # did, and only 12 of 3,010 schemes ended up categorised — while
                # the unit test passed, because its fixture had them the other
                # way round. The fixture was the thing that was wrong.
            continue
        parts = s.split(";")
        if len(parts) < 6 or parts[0].strip().lower().startswith("scheme code"):
            continue
        try:
            nav = float(parts[4])
        except (ValueError, IndexError):
            continue          # 'N.A.' for schemes that did not price that day
        out.append(Scheme(code=parts[0].strip(), name=parts[3].strip(),
                          isin=parts[1].strip(), nav=nav,
                          date=parts[5].strip(), amc=amc, category=category))
    return out


def investable_universe(schemes: list[Scheme]) -> list[Scheme]:
    """Direct + Growth only, and priced. See the module docstring for why."""
    return [s for s in schemes if s.is_direct and s.is_growth and s.nav > 0]


def parse_history(text: str) -> dict[str, list[tuple[str, float]]]:
    """scheme_code -> [(date, nav)], oldest first."""
    hist: dict[str, list[tuple[str, float]]] = {}
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) < 8 or parts[0].strip().lower().startswith("scheme code"):
            continue
        try:
            nav = float(parts[4])
        except ValueError:
            continue
        hist.setdefault(parts[0].strip(), []).append((parts[-1].strip(), nav))
    for k in hist:
        hist[k].sort(key=lambda x: datetime.strptime(x[0], "%d-%b-%Y"))
    return hist


def fetch_nav_history(amc_code: int, days: int = 90) -> dict[str, list[tuple[str, float]]]:
    to = datetime.now(timezone.utc)
    frm = to - timedelta(days=days)
    url = NAV_HIST_URL.format(frm=frm.strftime("%d-%b-%Y"),
                              to=to.strftime("%d-%b-%Y"), mf=amc_code)
    return parse_history(_get(url))


def total_return_pct(series: list[tuple[str, float]]) -> float | None:
    """Simple NAV-to-NAV return. None when the series cannot support one.

    Growth-option NAV is already total return: no distributions come out of it,
    which is exactly why the universe excludes IDCW variants.
    """
    if len(series) < 2:
        return None
    first, last = series[0][1], series[-1][1]
    if first <= 0:
        return None
    return (last / first - 1.0) * 100.0


def rank_by_momentum(hist: dict[str, list[tuple[str, float]]],
                     names: dict[str, str],
                     min_points: int = 15,
                     top_n: int = 10,
                     investable: set[str] | None = None) -> list[dict]:
    """Rank schemes by trailing NAV return over the fetched window.

    `min_points` guards against a scheme with two stale prints looking like a
    top performer — the same class of error as ranking on a two-trade sample.

    `investable` is not optional in practice. Ranking the raw history returns
    the SAME fund up to four times — Direct/Regular x Growth/IDCW all track one
    portfolio, so they post near-identical returns. Measured on Axis (mf=53),
    the unfiltered top 5 was: Axis IT ETF, then Nifty IT Index Fund as
    Direct-Growth, Direct-IDCW, Regular-Growth and Regular-IDCW — four slots for
    one fund, and two of them the strictly-worse Regular plan. Pass the
    investable code set to get five distinct funds.
    """
    rows = []
    for code, series in hist.items():
        if investable is not None and code not in investable:
            continue
        if len(series) < min_points:
            continue
        ret = total_return_pct(series)
        if ret is None:
            continue
        rows.append({"code": code, "name": names.get(code, "?"),
                     "points": len(series), "return_pct": round(ret, 2),
                     "nav": series[-1][1], "as_of": series[-1][0]})
    rows.sort(key=lambda r: r["return_pct"], reverse=True)
    return rows[:top_n]


def rank_within_categories(hist: dict[str, list[tuple[str, float]]],
                           schemes: list[Scheme],
                           min_points: int = 15,
                           per_category: int = 3,
                           min_peers: int = 5) -> dict[str, list[dict]]:
    """Best funds *within each category*, which is the comparison that means something.

    A flat return ranking is a sector-rotation readout. The 2026-08-05 live top
    10 was transportation/logistics, healthcare, technology and automotive funds
    — that says which sector ran, not which manager is good. A thematic fund
    beating a large-cap fund on 90-day return is not evidence about either.

    `min_peers` drops categories too small to rank: coming first out of two is
    not a percentile. Categories are AMFI's own, so this inherits their taxonomy
    rather than inventing one.
    """
    by_code = {s.code: s for s in schemes}
    buckets: dict[str, list[dict]] = {}
    for code, series in hist.items():
        sch = by_code.get(code)
        if sch is None or len(series) < min_points:
            continue
        ret = total_return_pct(series)
        if ret is None:
            continue
        buckets.setdefault(sch.category, []).append({
            "code": code, "name": sch.name, "category": sch.category,
            "return_pct": round(ret, 2), "nav": series[-1][1],
            "points": len(series), "as_of": series[-1][0],
        })
    out: dict[str, list[dict]] = {}
    for cat, rows in buckets.items():
        if len(rows) < min_peers:
            continue
        rows.sort(key=lambda r: r["return_pct"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
            r["peers"] = len(rows)
        out[cat] = rows[:per_category]
    return dict(sorted(out.items()))


def load_state() -> dict:
    try:
        d = json.loads(STATE_FILE.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict) -> None:
    runs = state.get("runs")
    if isinstance(runs, list):
        state["runs"] = runs[-HISTORY_KEEP:]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> int:
    text = _get(NAV_ALL_URL)
    schemes = parse_navall(text)
    universe = investable_universe(schemes)
    amcs = sorted({s.amc for s in universe})
    print(f"[india_mf] {len(schemes)} schemes, {len(universe)} investable "
          f"(Direct+Growth) across {len(amcs)} AMCs", flush=True)
    if not universe:
        print("[india_mf] EMPTY universe — AMFI format may have changed", flush=True)
        return 1

    as_of = universe[0].date
    names = {s.code: s.name for s in schemes}
    codes = {s.code for s in universe}

    # Ranking window. 90 calendar days is ~60 NAV points, enough for a trailing
    # return that is not one week of noise.
    hist: dict[str, list[tuple[str, float]]] = {}
    for amc_code, label in AMC_CODES.items():
        try:
            got = fetch_nav_history(amc_code, days=90)
            hist.update(got)
            print(f"[india_mf]   {label}: {len(got)} schemes with history", flush=True)
        except Exception as exc:  # noqa: BLE001 — one AMC must not sink the run
            print(f"[india_mf]   {label}: history unavailable ({exc})", flush=True)

    top = rank_by_momentum(hist, names, top_n=10, investable=codes)
    if top:
        print(f"\n[india_mf] top {len(top)} Direct-Growth funds by 90d NAV return "
              f"(absolute — reads as sector rotation, see below):", flush=True)
        for r in top:
            print(f"    {r['return_pct']:>7.2f}%  NAV {r['nav']:>10.4f}  {r['name'][:64]}", flush=True)
    else:
        print("[india_mf] no fund cleared the ranking filters", flush=True)

    # The comparison that actually means something: each fund against its own
    # AMFI category. The absolute list above is dominated by whichever sector
    # ran; this says which manager beat their peers.
    investable_schemes = [s for s in schemes if s.code in codes]
    by_cat = rank_within_categories(hist, investable_schemes, per_category=3)
    if by_cat:
        print(f"\n[india_mf] category leaders ({len(by_cat)} categories with enough peers):",
              flush=True)
        for cat, rows in by_cat.items():
            label = cat.split("(")[-1].rstrip(")") if "(" in cat else cat
            print(f"  {label[:58]}", flush=True)
            for r in rows:
                print(f"    #{r['rank']}/{r['peers']}  {r['return_pct']:>7.2f}%  "
                      f"{r['name'][:56]}", flush=True)
    else:
        print("[india_mf] no category had enough peers to rank", flush=True)

    state = load_state()
    runs = state.get("runs") or []
    runs.append({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "schemes_total": len(schemes),
        "investable": len(universe),
        "amcs": len(amcs),
        "ranked": len(top),
        "top": top[:5],
        "categories_ranked": len(by_cat),
        "category_leaders": {c: r[:1] for c, r in list(by_cat.items())[:12]},
    })
    state["runs"] = runs
    state["as_of"] = as_of
    state["investable"] = len(universe)
    save_state(state)
    print(f"\n[india_mf] NAV as of {as_of} — state written to {STATE_FILE}", flush=True)

    try:
        from india_broker import status_line
        print(f"[india_mf] {status_line()}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
