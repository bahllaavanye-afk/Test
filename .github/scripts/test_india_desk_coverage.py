"""India coverage across the desks, and the honest limit on it.

The desk layer covered 19 countries — Japan, China, Korea, Taiwan, Brazil,
Mexico, Canada, UK, Germany, France, South Africa, Indonesia, Vietnam,
Australia, Singapore, Thailand, Poland, Argentina, Malaysia — and **not India**,
the world's fifth-largest equity market. (Macro/FX already held INDA/EPI/SMIN;
nothing else did.)

**What can actually trade.** Alpaca has no route to NSE or BSE, so native Indian
equities cannot become orders. Every instrument added here is US-listed and
therefore goes through the existing order path with no new broker, no new
credentials and no new venue code:

    INDA  MSCI India          Cboe US      INFY  Infosys        NYSE
    EPI   earnings-weighted   NYSEArca     HDB   HDFC Bank      NYSE
    SMIN  small-cap India     Cboe US      IBN   ICICI Bank     NYSE
    INDY  Nifty 50            NasdaqGM     WIT   Wipro          NYSE
                                           RDY   Dr Reddy's     NYSE
                                           MMYT  MakeMyTrip     NasdaqGS

All eleven were live-verified on 2026-08-05 (5/5 daily bars and a live price
each). PIN was dropped from the candidate list — it returned no close data.

**What deliberately was NOT added.**
- *Native NSE/BSE symbols* (`RELIANCE.NS`, `TCS.NS`, `^NSEI`). Data works fine
  through yfinance — verified, INR, Asia/Kolkata — but there is no execution
  venue. A signal-only India desk would generate confident signals that can
  never become orders, which is precisely the Polymarket situation this repo
  already carries. One of those is a known limitation; two is a pattern.
  Wiring a real Indian broker (Zerodha / Upstox / ICICI Direct) is an operator
  decision, not something to fake with an unexecutable desk.
- *Options desk* — its eight underlyings are all mega-liquid (SPY, QQQ, AAPL…).
  INDA and INFY have listed options but thin chains, and the desk sizes real
  spreads; illiquid chains mean bad fills rather than more coverage.
- *Commodities* — India's gold demand is enormous but expressed through the same
  instruments the desk already trades (GLD). Nothing India-specific to add.
- *Crypto* — already global. *Polymarket* — no venue route at all.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "desk_order_placer.py"

INDIA_ETFS = {"INDA", "EPI", "SMIN", "INDY"}
INDIA_ADRS = {"INFY", "HDB", "IBN", "WIT", "RDY", "MMYT"}
INDIA_ALL = INDIA_ETFS | INDIA_ADRS


def _desks() -> dict[str, list[str]]:
    tree = ast.parse(_SRC.read_text())
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DeskConfig":
            kw = {k.arg: k.value for k in node.keywords}
            if "name" in kw and "symbols" in kw:
                out[ast.literal_eval(kw["name"])] = ast.literal_eval(kw["symbols"])
    return out


@pytest.fixture(scope="module")
def desks() -> dict[str, list[str]]:
    d = _desks()
    assert len(d) >= 9, f"expected the full desk roster, parsed {len(d)}"
    return d


@pytest.mark.parametrize("desk,expected", [
    ("International", INDIA_ETFS),
    ("Equities", INDIA_ADRS),
    ("StatArb", {"INDA", "EPI", "INFY", "WIT"}),
    ("TV Indicators", {"INDA", "INFY"}),
    ("Macro/FX", {"INDA", "EPI", "SMIN"}),
])
def test_each_desk_carries_its_india_instruments(desks, desk, expected):
    got = set(desks[desk]) & INDIA_ALL
    missing = expected - got
    assert not missing, f"{desk} lost India coverage: {sorted(missing)}"


def test_no_desk_has_duplicate_symbols(desks):
    """A duplicate wastes a bar slot and double-counts the symbol in sizing."""
    for name, syms in desks.items():
        dupes = [s for s in set(syms) if syms.count(s) > 1]
        assert not dupes, f"{name} has duplicate symbols: {dupes}"


def test_native_nse_symbols_are_not_in_any_desk(desks):
    """Alpaca cannot route NSE/BSE. A `.NS` symbol would 404 at order time.

    This is the guard against someone "completing" India coverage by pasting in
    RELIANCE.NS — the data resolves, so it looks right, and every order fails.
    """
    for name, syms in desks.items():
        bad = [s for s in syms if s.endswith((".NS", ".BO")) or s.startswith("^")]
        assert not bad, (
            f"{name} contains native Indian/index symbols {bad}, which Alpaca "
            "cannot trade. India exposure must go through US-listed ETFs/ADRs "
            "until a real Indian broker is wired."
        )


def test_india_exposure_spans_both_etfs_and_single_names(desks):
    """ETFs alone cannot express a view on a single Indian bank or IT firm."""
    everything = set().union(*(set(v) for v in desks.values()))
    assert everything & INDIA_ETFS, "no India ETF exposure anywhere"
    assert everything & INDIA_ADRS, "no India single-name exposure anywhere"


def test_the_statarb_pairs_are_present_together(desks):
    """A pair is useless if only one leg is in the universe."""
    sa = set(desks["StatArb"])
    assert {"INDA", "EPI"} <= sa, "the INDA/EPI weighting-spread pair is broken"
    assert {"INFY", "WIT"} <= sa, "the INFY/WIT IT-services pair is broken"


def test_symbol_growth_stays_within_the_batch_budget(desks):
    """Bars are fetched in chunks of 20; each chunk is a request against a
    documented free-tier data limit. India added one chunk (5 -> 6)."""
    distinct = set().union(*(set(v) for v in desks.values()))
    chunks = -(-len(distinct) // 20)
    assert chunks <= 8, (
        f"{len(distinct)} distinct symbols = {chunks} bar requests per run. "
        "Past ~8 the desk starts competing with itself for the free-tier data "
        "limit (see the 2026-07-28 collision measurement)."
    )


def test_the_options_desk_stayed_liquid(desks):
    """Deliberate exclusion — thin India chains would mean bad fills."""
    assert not (set(desks["Options"]) & INDIA_ALL), (
        "India names were added to the Options desk. Its underlyings are all "
        "mega-liquid by design; INDA/INFY chains are thin and the desk sizes "
        "real spreads."
    )
