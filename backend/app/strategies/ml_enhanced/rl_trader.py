"""
Reinforcement Learning Trader Strategy.
Uses A3C-LSTM agent to generate buy/hold/sell signals.
Falls back to RSI-based signals if no trained model is loaded.
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


# Default path where a trained A3C-LSTM checkpoint is expected.
_DEFAULT_MODEL_PATH = Path(__file__).parents[3] / "checkpoints" / "a3c_lstm_latest.pt"

# Actions emitted by the agent
_BUY = 0
_HOLD = 1
_SELL = 2


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI fallback signal."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_feature_tensor(df: pd.DataFrame, seq_len: int = 30) -> torch.Tensor | None:
    """
    Build a (1, seq_len, n_features) tensor from the last `seq_len` rows
    of an OHLCV DataFrame.  Returns None if df is too short.
    """
    if len(df) < seq_len + 1:
        return None

    close = df["close"]
    volume = df["volume"]

    # Simple feature set: returns, log-volume, RSI normalised to [-1, 1]
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

    Uses an A3C-LSTM agent to emit buy/hold/sell signals from recent OHLCV.
    If no trained checkpoint is available, falls back to RSI-based logic so
    the strategy is always usable even before training completes.
    """

    name = "rl_trader"
    display_name = "RL Trader (A3C-LSTM)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.60

    # Feature dimension expected by the model.  Must match training config.
    N_FEATURES = 3
    SEQ_LEN = 30

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._agent = None
        model_path = self.params.get("model_path", str(_DEFAULT_MODEL_PATH))
        self._model_path = Path(model_path)
        self._try_load_agent()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_agent(self) -> None:
        """Attempt to load trained A3C-LSTM checkpoint; silently skip if absent."""
        if not self._model_path.exists():
            return
        try:
            from app.ml.models.a3c_lstm import A3CLSTMAgent

            self._agent = A3CLSTMAgent.load(str(self._model_path))
            self._agent.eval()
        except Exception:
            self._agent = None

    def _rsi_signal(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        """Fallback: plain RSI oversold/overbought signal."""
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

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Run A3C-LSTM inference on recent OHLCV; fall back to RSI if model absent.
        """
        if self._agent is None:
            return self._rsi_signal(data, symbol)

        x = _build_feature_tensor(data, seq_len=self.SEQ_LEN)
        if x is None:
            return self._rsi_signal(data, symbol)

        # Pad or trim feature dimension to match model expectations
        if x.shape[-1] != self._agent.n_features:
            # Pad with zeros to match training feature count
            pad_size = self._agent.n_features - x.shape[-1]
            if pad_size > 0:
                x = torch.cat(
                    [x, torch.zeros(*x.shape[:2], pad_size)], dim=-1
                )
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
        Vectorized backtest: roll the model over each window (or RSI if no model).
        Uses .shift(1) to prevent lookahead bias.
        """
        if self._agent is None or len(df) < self.SEQ_LEN + 1:
            # RSI fallback — vectorized
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
                        x = torch.cat(
                            [x, torch.zeros(*x.shape[:2], pad_size)], dim=-1
                        )
                    else:
                        x = x[..., : self._agent.n_features]
                action_probs, _ = self._agent.forward(x)
                actions.iloc[i] = int(action_probs[0].argmax().item())

        # Apply shift(1) — no lookahead
        actions = actions.shift(1).fillna(_HOLD).astype(int)

        entries = actions == _BUY
        exits = actions == _SELL
        short_entries = actions == _SELL
        short_exits = actions == _BUY

        return BacktestSignals(
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
        )
