"""
Reinforcement Learning Trader Strategy.

Uses an A3C‑LSTM agent to generate buy/hold/sell signals.
Falls back to RSI‑based signals if no trained model is loaded.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

# Default path where a trained A3C‑LSTM checkpoint is expected.
_DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "checkpoints" / "a3c_lstm_latest.pt"

# Actions emitted by the agent
_BUY = 0
_HOLD = 1
_SELL = 2


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.

    Parameters
    ----------
    series : pd.Series
        Series of price values (typically close prices).
    period : int, optional
        Look‑back period for the RSI calculation, by default 14.

    Returns
    -------
    pd.Series
        RSI values aligned with the input series.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_feature_tensor(df: pd.DataFrame, seq_len: int = 30) -> Optional["torch.Tensor"]:
    """
    Build a tensor of shape ``(1, seq_len, n_features)`` from the last ``seq_len``
    rows of an OHLCV DataFrame.

    The feature set consists of:
    * Returns (pct change of close)
    * Log‑volume (log1p of volume, first difference)
    * Normalised RSI in the range ``[-1, 1]``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``['close', 'volume']`` columns.
    seq_len : int, optional
        Number of rows to include in the tensor, by default 30.

    Returns
    -------
    torch.Tensor | None
        Tensor ready for model inference, or ``None`` if the DataFrame is too
        short to construct a full window.
    """
    if len(df) < seq_len + 1:
        return None

    close = df["close"]
    volume = df["volume"]

    # Simple feature set: returns, log‑volume, RSI normalised to [-1, 1]
    returns = close.pct_change().fillna(0.0)
    log_vol = np.log1p(volume).diff().fillna(0.0)
    rsi_norm = (_rsi(close).fillna(50.0) - 50.0) / 50.0

    window = df.tail(seq_len)
    feat_matrix = np.stack(
        [
            returns.reindex(window.index).fillna(0.0).values,
            log_vol.reindex(window.index).fillna(0.0).values,
            rsi_norm.reindex(window.index).fillna(0.0).values,
        ],
        axis=1,
    )  # (seq_len, 3)

    return torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 3)


class RLTraderStrategy(AbstractStrategy):
    """
    Reinforcement Learning Trader.

    Uses an A3C‑LSTM agent to emit buy/hold/sell signals from recent OHLCV.
    If no trained checkpoint is available, falls back to RSI‑based logic so
    the strategy is always usable even before training completes.
    """

    name = "rl_trader"
    display_name = "RL Trader (A3C-LSTM)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    # Feature dimension expected by the model. Must match training config.
    N_FEATURES = 3
    SEQ_LEN = 30

    _agent: Any

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialise the strategy.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration dictionary. Recognised keys:
            ``model_path`` – path to the A3C‑LSTM checkpoint.
        """
        super().__init__(params)
        self._agent = None
        model_path = self.params.get("model_path", str(_DEFAULT_MODEL_PATH))
        self._model_path = Path(model_path)
        self._try_load_agent()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_agent(self) -> None:
        """
        Attempt to load a trained A3C‑LSTM checkpoint.

        The method silently skips loading if the checkpoint does not exist
        or if any error occurs during import / initialisation.
        """
        if not self._model_path.exists():
            return
        try:
            from app.ml.models.a3c_lstm import A3CLSTMAgent

            self._agent = A3CLSTMAgent.load(str(self._model_path))
            self._agent.eval()
        except Exception:  # pragma: no cover
            self._agent = None

    def _rsi_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Generate a fallback signal based on the RSI indicator.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data for the symbol.
        symbol : str
            Ticker symbol.

        Returns
        -------
        Signal | None
            A ``Signal`` object when the RSI is in oversold/overbought zones,
            otherwise ``None``.
        """
        if len(df) < 15:
            return None
        rsi_val = float(_rsi(df["close"]).iloc[-1])
        close = float(df["close"].iloc[-1])

        if rsi_val < 30:
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=0.55 + (30 - rsi_val) / 100,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=close,
                metadata={"source": "rsi_fallback", "rsi": rsi_val},
            )
        if rsi_val > 70:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=0.55 + (rsi_val - 70) / 100,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=close,
                metadata={"source": "rsi_fallback", "rsi": rsi_val},
            )
        return None

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Produce a single trading signal for the latest market data.

        The method first attempts to run the A3C‑LSTM model; if the model is not
        available or the input window is insufficient, it falls back to the
        RSI‑based signal.

        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV data for the symbol.
        symbol : str
            Ticker symbol.

        Returns
        -------
        Signal | None
            A populated ``Signal`` when confidence exceeds the threshold,
            otherwise ``None``.
        """
        if self._agent is None:
            return self._rsi_signal(data, symbol)

        x = _build_feature_tensor(data, seq_len=self.SEQ_LEN)
        if x is None:
            return self._rsi_signal(data, symbol)

        # Pad or trim feature dimension to match model expectations
        if x.shape[-1] != self._agent.n_features:
            pad_size = self._agent.n_features - x.shape[-1]
            if pad_size > 0:
                x = torch.cat([x, torch.zeros(*x.shape[:2], pad_size)], dim=-1)
            else:
                x = x[..., : self._agent.n_features]

        action = self._agent.select_action(x)
        action_probs, _ = self._agent.forward(x)
        confidence = float(action_probs[0, action].item())
        close = float(data["close"].iloc[-1])

        if action == _BUY and confidence >= self.confidence_threshold:
            return Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=close,
                metadata={"source": "a3c_lstm", "action_probs": action_probs[0].tolist()},
            )
        if action == _SELL and confidence >= self.confidence_threshold:
            return Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                target_price=close,
                metadata={"source": "a3c_lstm", "action_probs": action_probs[0].tolist()},
            )
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Vectorised back‑test over a full DataFrame.

        The method rolls the model over each sliding window (or uses the RSI
        fallback when the model is unavailable).  ``.shift(1)`` is applied to
        avoid look‑ahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV history for a single symbol.

        Returns
        -------
        BacktestSignals
            Named tuple containing ``entries`` and ``exits`` boolean series.
        """
        if self._agent is None or len(df) < self.SEQ_LEN + 1:
            # RSI fallback — vectorised
            rsi_series = _rsi(df["close"]).shift(1)
            entries = (rsi_series < 30).fillna(False)
            exits = (rsi_series > 70).fillna(False)
            return BacktestSignals(entries=entries, exits=exits)

        actions = pd.Series(index=df.index, dtype=int)
        actions[:] = _HOLD

        self._agent.eval()
        with torch.no_grad():
            for i in range(self.SEQ_LEN, len(df)):
                window = df.iloc[i - self.SEQ_LEN : i]
                x = _build_feature_tensor(window, seq_len=self.SEQ_LEN)
                if x is None:
                    continue
                if x.shape[-1] != self._agent.n_features:
                    pad_size = self._agent.n_features - x.shape[-1]
                    if pad_size > 0:
                        x = torch.cat([x, torch.zeros(*x.shape[:2], pad_size)], dim=-1)
                    else:
                        x = x[..., : self._agent.n_features]
                action = self._agent.select_action(x)
                actions.iloc[i] = action

        # Shift by one to prevent look‑ahead bias
        entries = (actions == _BUY).shift(1).fillna(False)
        exits = (actions == _SELL).shift(1).fillna(False)
        return BacktestSignals(entries=entries, exits=exits)