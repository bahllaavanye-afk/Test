"""
LLM-driven formulaic alpha factor generation.

Uses the Anthropic QuantEdge AI API to propose novel technical alpha factors,
then validates each factor against historical price data using IC/IR analysis.

Factors that pass validation (IC > 0.02, IR > 0.3) are saved to YAML
in experiments/alpha_mining/results/ for use in the ML feature pipeline.

Usage:
    # From command line:
    python experiments/alpha_mining/llm_alpha_miner.py --symbols SPY QQQ --n-factors 5

    # From Python:
    miner = AlphaMiner()
    factors = miner.generate_factors(n=5)
    results = miner.mine_and_save(["SPY", "QQQ"])
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure we can import project modules when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_IC_THRESHOLD = 0.02      # minimum Information Coefficient to pass
_IR_THRESHOLD = 0.30      # minimum Information Ratio to pass

_GENERATION_PROMPT = """You are a quantitative researcher specialising in alpha factor research.

Propose {n} novel technical alpha factors as Python expressions.
Each factor operates on a pandas DataFrame `df` with columns:
  - open, high, low, close, volume (float64, all > 0)
  - The index is a DatetimeIndex (daily frequency)

Requirements for each factor:
1. Use ONLY pandas/numpy operations (np, pd are available)
2. Use ONLY past data — apply .shift(1) to any forward-looking computation
3. Return a pd.Series aligned to df.index
4. Be different from: RSI, MACD, Bollinger Bands, ATR, momentum (12-1 month), SMA crossover
5. Be computable in a single lambda expression

Examples of acceptable factors:
  - "lambda df: (df['close'].rolling(5).mean() - df['close'].rolling(20).mean()).shift(1)"
  - "lambda df: (df['high'] - df['low']).rolling(10).std().shift(1)"

Output ONLY a JSON array of objects, no other text:
[
  {{"name": "factor_name", "formula": "lambda df: <expression>", "rationale": "one sentence"}},
  ...
]"""


def _safe_eval_factor(formula: str, df: pd.DataFrame) -> pd.Series | None:
    """
    Safely evaluate a lambda formula string against a DataFrame.
    Returns the pd.Series result or None if it fails.
    """
    try:
        fn = eval(formula, {"np": np, "pd": pd})
        result = fn(df)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)
        return result.astype(float)
    except Exception:
        return None


def _compute_ic_ir(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    window: int = 20,
) -> dict:
    """
    Compute Information Coefficient (IC) and Information Ratio (IR).

    IC = Spearman rank correlation between factor values and next-period returns.
    IR = IC.mean() / IC.std()

    Uses rolling windows of size `window` for stable estimation.
    """
    aligned = pd.DataFrame({"factor": factor_values, "ret": forward_returns}).dropna()
    if len(aligned) < window * 2:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ir": 0.0, "n_obs": len(aligned)}

    # Rolling IC (Spearman correlation in each window)
    ic_series = []
    for i in range(window, len(aligned), window // 2):
        chunk = aligned.iloc[max(0, i - window):i]
        if len(chunk) < 5:
            continue
        corr = chunk["factor"].corr(chunk["ret"], method="spearman")
        if not np.isnan(corr):
            ic_series.append(corr)

    if not ic_series:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ir": 0.0, "n_obs": len(aligned)}

    ic_arr = np.array(ic_series)
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr))
    ir = float(ic_mean / (ic_std + 1e-9))

    return {
        "ic_mean": round(ic_mean, 5),
        "ic_std": round(ic_std, 5),
        "ir": round(ir, 4),
        "n_obs": len(aligned),
    }


class AlphaMiner:
    """
    LLM-driven alpha factor miner.

    Generates factor proposals via QuantEdge AI API, evaluates on price history,
    and saves passing factors to YAML files.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model

    def generate_factors(self, n: int = 5) -> list[dict]:
        """
        Generate n alpha factor proposals via the shared free-LLM gateway
        (Groq → DeepSeek → Gemini). Returns list of dicts: [{name, formula, rationale}].
        Falls back to a built-in set if no free provider is configured/reachable.
        """
        try:
            from app.llm.gateway import complete_sync
        except Exception:
            print("free-LLM gateway unavailable — using built-in factor proposals")
            return self._builtin_factors()

        content = complete_sync(
            [{"role": "user", "content": _GENERATION_PROMPT.format(n=n)}],
            max_tokens=1024,
            agent="alpha_miner",
        )
        if not content:
            print("no free LLM provider configured — using built-in factor proposals")
            return self._builtin_factors()

        try:
            # Extract JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            proposals = json.loads(content[start:end])
            parsed = [p for p in proposals if "name" in p and "formula" in p]
            return parsed or self._builtin_factors()
        except Exception as e:
            print(f"LLM factor parsing failed ({e}), using built-in factors")
            return self._builtin_factors()

    def _builtin_factors(self) -> list[dict]:
        """Fallback set of manually crafted alpha factors."""
        return [
            {
                "name": "overnight_gap_normalized",
                "formula": "lambda df: ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).shift(1)",
                "rationale": "Overnight gap normalised by prior close captures opening auction imbalance",
            },
            {
                "name": "volume_price_divergence",
                "formula": "lambda df: (df['close'].pct_change().rolling(5).mean() - df['volume'].pct_change().rolling(5).mean()).shift(1)",
                "rationale": "Divergence between price trend and volume trend signals unsustained moves",
            },
            {
                "name": "close_location_value",
                "formula": "lambda df: ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(10).mean().shift(1)",
                "rationale": "Average close location in daily range — high values suggest buying pressure",
            },
            {
                "name": "intraday_vol_ratio",
                "formula": "lambda df: ((df['high'] - df['low']) / df['close']).rolling(5).mean() / ((df['high'] - df['low']) / df['close']).rolling(20).mean().shift(1)",
                "rationale": "Short-term vs medium-term intraday volatility ratio signals regime changes",
            },
            {
                "name": "price_volume_trend",
                "formula": "lambda df: (df['close'].pct_change() * df['volume']).rolling(10).sum().shift(1)",
                "rationale": "Cumulative price-volume trend over 10 days — positive = sustained buying",
            },
        ]

    def evaluate_factor(
        self,
        formula: str,
        price_data: pd.DataFrame,
        forward_period: int = 1,
    ) -> dict:
        """
        Evaluate a factor formula against price data.

        Args:
            formula:        Python lambda string
            price_data:     OHLCV DataFrame
            forward_period: number of bars ahead for forward return

        Returns:
            dict with keys: ic_mean, ic_std, ir, n_obs, passes
        """
        factor_vals = _safe_eval_factor(formula, price_data)
        if factor_vals is None:
            return {"ic_mean": 0.0, "ic_std": 0.0, "ir": 0.0, "n_obs": 0, "passes": False, "error": "eval_failed"}

        # Forward returns
        close_col = "close" if "close" in price_data.columns else "Close"
        fwd = price_data[close_col].pct_change(forward_period).shift(-forward_period)

        metrics = _compute_ic_ir(factor_vals, fwd)
        passes = (
            abs(metrics["ic_mean"]) >= _IC_THRESHOLD
            and abs(metrics["ir"]) >= _IR_THRESHOLD
        )
        return {**metrics, "passes": passes}

    def mine_and_save(
        self,
        symbols: list[str],
        n_factors: int = 5,
        output_dir: str | None = None,
    ) -> list[dict]:
        """
        Mine factors via LLM, evaluate on historical data for all symbols,
        and save passing factors to YAML.

        Returns list of passing factor dicts.
        """
        out_dir = Path(output_dir) if output_dir else _RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {n_factors} factor proposals via LLM...")
        proposals = self.generate_factors(n_factors)
        print(f"Got {len(proposals)} proposals")

        # Fetch price data for all symbols
        price_datasets: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            print(f"  Fetching {sym}...")
            if _HAS_YF:
                try:
                    df = yf.download(sym, period="2y", interval="1d", auto_adjust=True, progress=False)
                    if df is not None and len(df) > 60:
                        df.columns = [c.lower() for c in df.columns]
                        price_datasets[sym] = df
                except Exception:
                    pass

        if not price_datasets:
            print("No price data available — cannot evaluate factors")
            return []

        passing_factors = []
        for proposal in proposals:
            name = proposal.get("name", "unknown")
            formula = proposal.get("formula", "")
            rationale = proposal.get("rationale", "")
            print(f"  Evaluating: {name}")

            sym_results = []
            for sym, df in price_datasets.items():
                try:
                    metrics = self.evaluate_factor(formula, df)
                    sym_results.append({
                        "symbol": sym,
                        **metrics,
                    })
                except Exception:
                    pass

            if not sym_results:
                continue

            # Average IC across symbols
            avg_ic = float(np.mean([r["ic_mean"] for r in sym_results]))
            avg_ir = float(np.mean([r["ir"] for r in sym_results]))
            passes_count = sum(1 for r in sym_results if r.get("passes", False))

            passes = passes_count >= len(sym_results) * 0.5  # pass on majority of symbols

            print(f"    avg_IC={avg_ic:.4f} avg_IR={avg_ir:.3f} pass_rate={passes_count}/{len(sym_results)} {'✓' if passes else '✗'}")

            factor_record = {
                "name": name,
                "formula": formula,
                "rationale": rationale,
                "avg_ic": round(avg_ic, 5),
                "avg_ir": round(avg_ir, 4),
                "per_symbol": sym_results,
                "passes": passes,
            }

            if passes:
                passing_factors.append(factor_record)
                out_path = out_dir / f"factor_{name}.yaml"
                out_path.write_text(yaml.dump(factor_record, default_flow_style=False))
                print(f"    Saved to {out_path}")

        print(f"\n{len(passing_factors)} / {len(proposals)} factors passed validation")
        return passing_factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Alpha Factor Miner")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"], help="Symbols to evaluate on")
    parser.add_argument("--n-factors", type=int, default=5, help="Number of factors to generate")
    parser.add_argument("--output-dir", default=str(_RESULTS_DIR), help="Output directory for YAML files")
    args = parser.parse_args()

    miner = AlphaMiner()
    results = miner.mine_and_save(args.symbols, n_factors=args.n_factors, output_dir=args.output_dir)
    print(f"\nDone. {len(results)} passing factors saved.")
