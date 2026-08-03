"""The crypto desk cannot place an order, and it is arithmetic, not judgement.

`desk_order_placer` admits a signal when `confidence >= desk.confidence_min`
(0.60 for every desk), and gives sub-threshold signals a second chance through a
one-clip-per-desk exploration path floored at 0.45. `crypto_adaptive_trend` is
the only strategy on the Crypto desk still producing signals at all — the rest
depend on data sources that are geo-blocked or rate-limited from GitHub's US
runners (Binance OI: HTTP 451; CoinGecko MVRV: HTTP 429).

Its live confidence is

    confidence = min(|raw_signal| * vol_scalar / 2, 0.95)
    vol_scalar = min(target_vol / max(rv_21, 0.05), 3.0),  target_vol = 0.40

so confidence is a *position size*, not a conviction, and it falls as volatility
rises. With |raw_signal| <= 1 the ceiling is target_vol / (2 * rv_21):

    realized vol (21d, ann)     max attainable confidence
        20%                             0.95
        33%                             0.60   <- the gate, only just
        40%                             0.50
        50%                             0.40   <- below the 0.45 explore floor
        65%                             0.26
        80%                             signal suppressed by min_signal

Crypto's realized vol lives in the 45-80% band, so both gates sit above the
ceiling and the desk is structurally unable to trade.

Measured 2026-08-03, crypto desk run 30782697088:

    [stage] Generate trading signals — signals_generated=16
    · crypto_adaptive_trend/BTC/USD conf=0.37 < 0.60 — skipped
      ... 16 of 16 skipped, confidences 0.23 .. 0.44 ...
    [stage] Apply confidence threshold + top-K filter — passed=0 filtered=16 explored=0
    Done. 0 orders placed across 9 desks.

The highest confidence the desk produced was 0.44, against a 0.45 exploration
floor. It missed even the consolation path by one hundredth.

WHY THIS TEST IS xfail RATHER THAN A FIX
The obvious repair — drop the vol scalar so confidence is |raw_signal| — was
tried and rejected on evidence: `analyze()` computes conviction as
`tanh(composite_raw * 5)`, which saturates almost immediately, so on a
zero-drift random walk at 55% vol it returns 0.83, 0.92 and 0.94 for three
different seeds. That fix trades on noise, which is worse than not trading.

A correct repair has to recalibrate conviction itself (a risk-adjusted score
such as momentum / realized vol, squashed with a gain that actually
discriminates), and the repo's own standard — "walk-forward only; no
in-sample-only backtests are accepted as valid" — means that is a backtested
strategy change, not a patch. Note also that `backtest_signals()` builds its
composite from `.rank(pct=True)` percentiles while `analyze()` uses
`tanh(raw * 5)`: the live and backtested signals are different functions, so
existing backtest results do not describe the behaviour under test here.

Marked `strict=True` deliberately: when someone does recalibrate, this test
starts passing and pytest reports XPASS as a failure, which is the prompt to
delete the marker rather than let the finding rot.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.strategies.manual.crypto_adaptive_trend import CryptoAdaptiveTrendStrategy

_REPO = Path(__file__).resolve().parents[3]
_DESK_PLACER = _REPO / ".github" / "scripts" / "desk_order_placer.py"

# Kept in sync with desk_order_placer's exploration floor; see the block comment
# at "Exploration allocation".
_EXPLORE_FLOOR = 0.45


def _crypto_confidence_min() -> float:
    """Read the Crypto desk's gate from the real config, not a copy of it.

    Hardcoding 0.60 here would let the test keep asserting against a threshold
    the desk no longer uses.
    """
    if not _DESK_PLACER.exists():  # pragma: no cover - backend-only checkouts
        pytest.skip(f"{_DESK_PLACER} not present in this checkout")
    tree = ast.parse(_DESK_PLACER.read_text())
    for node in ast.walk(tree):
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if target is None and isinstance(node, ast.Assign):
            target = node.targets[0]
        if getattr(target, "id", "") != "DESKS":
            continue
        for call in node.value.elts:
            kw = {k.arg: k.value for k in call.keywords}
            if ast.literal_eval(kw["name"]) == "Crypto":
                return float(ast.literal_eval(kw["confidence_min"]))
    raise AssertionError("Crypto desk not found in DESKS — the registry moved")


def _trending_bars(n: int = 300, drift: float = 0.004,
                   vol_ann: float = 0.50, seed: int = 7) -> pd.DataFrame:
    """A strongly trending series at a realistic crypto volatility.

    drift=0.4%/day compounds to roughly +200% over the window — about as
    unambiguous an uptrend as crypto produces. If the gate is unreachable here it
    is unreachable in practice.
    """
    rng = np.random.default_rng(seed)
    daily_vol = vol_ann / np.sqrt(365)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, daily_vol, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": 1e6,
    })


def _confidence(df: pd.DataFrame) -> float | None:
    signal = asyncio.run(CryptoAdaptiveTrendStrategy().analyze(df, "BTC/USD"))
    return None if signal is None else signal.confidence


def test_the_gate_is_below_the_strategys_arithmetic_ceiling():
    """Sanity floor: the config the rest of this file reasons about is real."""
    assert _crypto_confidence_min() > 0, "Crypto desk gate should be a positive threshold"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "crypto_adaptive_trend scales confidence by target_vol/realized_vol, so at "
        "crypto's normal 45-80% vol its ceiling (0.25-0.44) sits below the desk's "
        "0.60 gate. Measured in run 30782697088: 16 signals, all skipped, 0 orders. "
        "Needs a backtested recalibration of conviction, not a patch — see module "
        "docstring."
    ),
)
def test_a_strong_crypto_trend_can_clear_the_order_gate():
    gate = _crypto_confidence_min()
    conf = _confidence(_trending_bars(vol_ann=0.50))
    assert conf is not None, "strategy produced no signal on a strong uptrend"
    assert conf >= gate, (
        f"strongest available trend scored {conf:.3f} against a {gate:.2f} gate — "
        f"the Crypto desk cannot place an order at any signal strength."
    )


@pytest.mark.xfail(
    strict=True,
    reason="same root cause: the ceiling at 50% vol is 0.40, under the 0.45 floor.",
)
def test_a_strong_crypto_trend_can_at_least_reach_the_exploration_floor():
    """Exploration is the fallback that keeps unproven strategies gathering fills.

    It is floored at 0.45 and the desk's best observed signal was 0.44, so even
    the consolation path is shut.
    """
    conf = _confidence(_trending_bars(vol_ann=0.50))
    assert conf is not None and conf >= _EXPLORE_FLOOR


def test_confidence_falls_as_volatility_rises_which_is_the_defect():
    """Pin the mechanism, so a future edit cannot quietly change the diagnosis.

    Identical trend, different volatility: a conviction score should be roughly
    stable, and this one collapses. This is what makes the gate unreachable.
    """
    calm = _confidence(_trending_bars(vol_ann=0.25))
    rough = _confidence(_trending_bars(vol_ann=0.65))
    assert calm is not None and rough is not None
    assert calm > rough + 0.30, (
        f"expected the documented vol contamination (calm={calm}, rough={rough}); "
        f"if this no longer holds, the confidence formula changed and the xfail "
        f"tests above need re-checking rather than trusting."
    )
