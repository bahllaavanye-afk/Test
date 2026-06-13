"""
AutoML Desk — the always-on continuous-learning loop.

Every cycle (default 3 min) it:
  1. Pulls the freshest real OHLCV for each tracked symbol (Redis ring buffer
     first, then a short yfinance window as fallback). Never fabricates data.
  2. Fine-tunes a deep-copied challenger on the newest window (seconds, not days
     — see incremental.py for why).
  3. Scores champion vs challenger on a held-out validation slice.
  4. Promotes the challenger only if it beats the champion by a margin —
     hot-swapping it into the live InferenceService AND persisting to disk so a
     restart keeps the improvement. Otherwise the champion is kept (a backup is
     written before any overwrite so a bad promote can be rolled back).
  5. Records lineage + scores to Redis and broadcasts a status event on the
     agent bus so other desks/agents see the latest model state.

Cold start: if no champion exists yet for a symbol, a fresh small LSTM is trained
on the available window and promoted if it clears the minimum-quality bar.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger

ARTIFACTS_DIR = Path(__file__).parents[3] / "models_artifacts"
STATE_PATH = Path(__file__).parents[3] / "experiments" / "results" / "automl_desk.json"

DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SPY", "QQQ"]
SEQ_LEN = 60
# Hold out the last VAL_FRACTION of sequences for champion/challenger scoring.
VAL_FRACTION = 0.25
MIN_TRAIN_SEQUENCES = 40


@dataclass
class SymbolResult:
    symbol: str
    action: str  # "promoted" | "kept_champion" | "cold_start" | "skipped" | "error"
    reason: str = ""
    champion_score: float | None = None
    challenger_score: float | None = None
    n_val: int = 0


@dataclass
class CycleReport:
    timestamp: str
    symbols_processed: int = 0
    promotions: int = 0
    results: list[SymbolResult] = field(default_factory=list)
    duration_seconds: float = 0.0


class AutoMLDesk:
    def __init__(
        self,
        symbols: list[str] | None = None,
        interval_seconds: int = 180,
        ohlcv_interval: str = "1h",
        fine_tune_epochs: int = 2,
        fine_tune_lr: float = 1e-4,
        min_improvement: float = 0.01,
    ):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.interval_seconds = interval_seconds
        self.ohlcv_interval = ohlcv_interval
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr
        self.min_improvement = min_improvement
        self._running = False
        self.last_report: CycleReport | None = None

    # ------------------------------------------------------------------
    # Data acquisition — real data only.
    # ------------------------------------------------------------------
    async def _fetch_recent(self, symbol: str):
        """
        Return a recent OHLCV DataFrame for `symbol`, or None if no real data is
        available. Tries the Redis ring buffer first (populated by price_feed),
        then falls back to a short yfinance pull. Never returns synthetic data.
        """
        import pandas as pd

        # 1. Redis ring buffer (freshest, lowest latency)
        try:
            from app.redis_client import PriceCache
            cache = PriceCache()
            for exch in ("binance", "alpaca", "yfinance"):
                rows = await cache.get_ohlcv(exch, symbol, self.ohlcv_interval)
                if rows and len(rows) >= SEQ_LEN + MIN_TRAIN_SEQUENCES:
                    df = pd.DataFrame(rows)
                    if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                        return df
        except Exception as e:
            logger.debug("automl_desk: redis ohlcv miss", symbol=symbol, error=str(e))

        # 2. yfinance fallback (short window — enough to fine-tune, fast to fetch)
        try:
            import yfinance as yf
            loop = asyncio.get_running_loop()
            period = "60d" if self.ohlcv_interval.endswith(("m", "h")) else "2y"
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    symbol, period=period, interval=self.ohlcv_interval,
                    auto_adjust=True, progress=False,
                ),
            )
            if df is None or len(df) < SEQ_LEN + MIN_TRAIN_SEQUENCES:
                return None
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            return df
        except Exception as e:
            logger.debug("automl_desk: yfinance fetch failed", symbol=symbol, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Per-symbol incremental update (CPU-bound torch work → executor).
    # ------------------------------------------------------------------
    def _update_symbol_sync(self, symbol: str, df) -> SymbolResult:
        """Synchronous core: build data, fine-tune, validate, promote. Runs in executor."""
        from app.ml.training.incremental import (
            build_supervised, fine_tune, validate_model, should_promote, ValidationScore,
        )
        from app.ml.inference import get_inference_service

        try:
            inference = get_inference_service()
            champion = inference.models.get("lstm")
            champ_scaler = inference.scalers.get("default")

            X, y, fwd, scaler = build_supervised(
                df, seq_len=SEQ_LEN,
                scaler=champ_scaler if champion is not None else None,
            )
            n = len(X)
            if n < MIN_TRAIN_SEQUENCES:
                return SymbolResult(symbol=symbol, action="skipped",
                                    reason=f"only {n} sequences")

            split = int(n * (1 - VAL_FRACTION))
            X_tr, y_tr = X[:split], y[:split]
            X_val, y_val, fwd_val = X[split:], y[split:], fwd[split:]

            # Cold start: no champion yet → train a fresh model on the window.
            if champion is None:
                from app.ml.models.lstm import LSTMPredictor
                n_features = X.shape[-1]
                fresh = LSTMPredictor(n_features=n_features)
                challenger = fine_tune(fresh, X_tr, y_tr,
                                       epochs=max(5, self.fine_tune_epochs), lr=1e-3)
                ch_score = validate_model(challenger, X_val, y_val, fwd_val)
                if should_promote(None, ch_score, min_samples=10):
                    self._promote(symbol, challenger, scaler, inference)
                    return SymbolResult(symbol=symbol, action="cold_start",
                                        reason="trained fresh champion",
                                        challenger_score=round(ch_score.combined, 4),
                                        n_val=ch_score.n)
                return SymbolResult(symbol=symbol, action="skipped",
                                    reason="cold-start model below quality bar",
                                    challenger_score=round(ch_score.combined, 4),
                                    n_val=ch_score.n)

            # Warm path: fine-tune a challenger, compare to champion.
            champ_score = validate_model(champion, X_val, y_val, fwd_val)
            challenger = fine_tune(champion, X_tr, y_tr,
                                   epochs=self.fine_tune_epochs, lr=self.fine_tune_lr)
            ch_score = validate_model(challenger, X_val, y_val, fwd_val)

            if should_promote(champ_score, ch_score, min_improvement=self.min_improvement):
                self._promote(symbol, challenger, scaler, inference)
                return SymbolResult(symbol=symbol, action="promoted",
                                    reason="challenger beat champion",
                                    champion_score=round(champ_score.combined, 4),
                                    challenger_score=round(ch_score.combined, 4),
                                    n_val=ch_score.n)
            return SymbolResult(symbol=symbol, action="kept_champion",
                                reason="no significant improvement",
                                champion_score=round(champ_score.combined, 4),
                                challenger_score=round(ch_score.combined, 4),
                                n_val=ch_score.n)
        except ValueError as e:
            return SymbolResult(symbol=symbol, action="skipped", reason=str(e))
        except Exception as e:  # noqa: BLE001 — one symbol must never kill the cycle
            logger.warning("automl_desk: symbol update failed", symbol=symbol, error=str(e))
            return SymbolResult(symbol=symbol, action="error", reason=str(e))

    def _promote(self, symbol: str, challenger, scaler, inference) -> None:
        """Hot-swap challenger into live inference + persist, backing up the champion."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        champ_path = ARTIFACTS_DIR / "lstm_latest.pt"
        scaler_path = ARTIFACTS_DIR / "scaler_latest.pt"
        if champ_path.exists():
            try:
                shutil.copy2(champ_path, ARTIFACTS_DIR / "lstm_prev.pt")
            except Exception as e:
                logger.debug("automl_desk: champion backup failed", error=str(e))
        try:
            challenger.save(str(champ_path), metadata={
                "init_kwargs": {"n_features": challenger.n_features},
                "promoted_by": "automl_desk",
                "symbol": symbol,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            })
            try:
                scaler.save(str(scaler_path))
            except Exception as e:
                logger.debug("automl_desk: scaler save failed", error=str(e))
            # Live hot-swap — the next prediction uses the improved model.
            inference.models["lstm"] = challenger
            inference.scalers["default"] = scaler
            logger.info("automl_desk: promoted challenger", symbol=symbol)
        except Exception as e:
            logger.warning("automl_desk: promote/persist failed", symbol=symbol, error=str(e))

    # ------------------------------------------------------------------
    # Cycle + loop
    # ------------------------------------------------------------------
    async def run_cycle(self) -> CycleReport:
        import time
        start = time.time()
        report = CycleReport(timestamp=datetime.now(timezone.utc).isoformat())
        loop = asyncio.get_running_loop()

        for symbol in self.symbols:
            df = await self._fetch_recent(symbol)
            if df is None:
                report.results.append(SymbolResult(symbol=symbol, action="skipped",
                                                    reason="no real data available"))
                continue
            result = await loop.run_in_executor(None, self._update_symbol_sync, symbol, df)
            report.results.append(result)
            report.symbols_processed += 1
            if result.action in ("promoted", "cold_start"):
                report.promotions += 1

        report.duration_seconds = round(time.time() - start, 2)
        self.last_report = report
        await self._persist_and_broadcast(report)
        logger.info("automl_desk: cycle complete",
                    processed=report.symbols_processed,
                    promotions=report.promotions,
                    duration_s=report.duration_seconds)
        return report

    async def _persist_and_broadcast(self, report: CycleReport) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(asdict(report), indent=2, default=str))
        except Exception as e:
            logger.debug("automl_desk: state write failed", error=str(e))
        try:
            from app.tasks.agent_bus import get_bus
            bus = get_bus()
            await bus.broadcast_signal(
                {
                    "type": "automl_cycle",
                    "promotions": report.promotions,
                    "processed": report.symbols_processed,
                    "results": [asdict(r) for r in report.results],
                },
                from_agent="automl_desk",
            )
        except Exception as e:
            logger.debug("automl_desk: broadcast failed", error=str(e))

    async def run(self) -> None:
        """Forever loop. Launch via asyncio.create_task()."""
        self._running = True
        logger.info("AutoMLDesk started", symbols=self.symbols, interval=self.interval_seconds)
        while self._running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                logger.info("AutoMLDesk cancelled — shutting down")
                break
            except Exception as e:  # noqa: BLE001
                logger.error("AutoMLDesk cycle crashed", error=str(e))
            if self._running:
                await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


_desk: AutoMLDesk | None = None


def get_automl_desk() -> AutoMLDesk:
    global _desk
    if _desk is None:
        _desk = AutoMLDesk()
    return _desk
