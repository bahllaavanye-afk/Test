"""Ensemble strategy: pure ML signal from all models combined with additional confirmation filters."""
import pandas as pd
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.inference import get_inference_service


def _validate_dataframe(
    df: pd.DataFrame,
    required_columns: set,
    context: str = "DataFrame",
) -> None:
    """
    Validate that ``df`` is a non‑empty :class:`pandas.DataFrame` containing the
    specified ``required_columns``.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_columns : set
        Column names that must be present in ``df``.
    context : str, optional
        Human‑readable name used in error messages (default ``"DataFrame"``).

    Raises
    ------
    ValueError
        If ``df`` is not a DataFrame, is empty, or is missing required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{context} must be a pandas DataFrame.")
    if df.empty:
        raise ValueError(f"{context} cannot be empty.")
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(sorted(missing))}.")


def _validate_symbol(symbol: str) -> None:
    """
    Validate that ``symbol`` is a non‑empty string.

    Parameters
    ----------
    symbol : str
        The ticker symbol to validate.

    Raises
    ------
    ValueError
        If ``symbol`` is not a string or is empty/whitespace.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non‑empty string.")


class EnsembleStrategy(AbstractStrategy):
    name = "ensemble"
    display_name = "Ensemble ML (LSTM + XGB + Lorentzian)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.70  # higher bar for pure ML
    sma_window = 20  # simple moving average window for confirmation

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Produce a trading signal based on the ML inference combined with
        price‑based confirmation filters.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and volume data. Must contain ``close`` and ``volume`` columns.
        symbol : str
            The ticker symbol for which the signal is generated.

        Returns
        -------
        Signal | None
            A populated :class:`Signal` if all entry conditions are met; otherwise ``None``.

        Raises
        ------
        ValueError
            If ``data`` or ``symbol`` are invalid.
        """
        # Input validation
        _validate_symbol(symbol)
        _validate_dataframe(data, {"close", "volume"}, context="data")

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # Basic ML validation
            if not ml_result or ml_result.get("prediction") == "neutral":
                return None
            if ml_result.get("confidence", 0) < self.confidence_threshold:
                return None

            # Compute SMA and median volume on the latest slice
            recent = data.tail(self.sma_window)
            if recent.empty:
                return None
            sma = recent["close"].mean()
            median_vol = recent["volume"].median()
            latest_close = data["close"].iloc[-1]
            latest_vol = data["volume"].iloc[-1]

            # Directional confirmation
            if ml_result["prediction"] == "up":
                if latest_close <= sma:
                    return None
            else:  # prediction == "down"
                if latest_close >= sma:
                    return None

            # Volume confirmation
            if latest_vol < median_vol:
                return None

            return Signal(
                symbol=symbol,
                side="buy" if ml_result["prediction"] == "up" else "sell",
                confidence=ml_result["confidence"],
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                risk_bucket=self.risk_bucket,
                metadata=ml_result,
            )
        except Exception:
            # In production we would log the exception; for now we silently ignore.
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate entry and exit signals for back‑testing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing historical data and ML inference columns.

        Returns
        -------
        BacktestSignals
            Object with boolean ``entries`` and ``exits`` series.

        Raises
        ------
        ValueError
            If ``df`` is invalid or missing required columns.
        """
        # Input validation
        _validate_dataframe(
            df,
            {"close", "volume", "ml_prediction", "ml_confidence"},
            context="df",
        )

        # Compute rolling SMA and median volume
        sma = df["close"].rolling(window=self.sma_window, min_periods=1).mean()
        median_vol = df["volume"].rolling(window=self.sma_window, min_periods=1).median()

        # Conditions for a valid entry
        is_up = df["ml_prediction"] == "up"
        is_down = df["ml_prediction"] == "down"
        conf_ok = df["ml_confidence"] >= self.confidence_threshold
        price_above_sma = df["close"] > sma
        price_below_sma = df["close"] < sma
        vol_ok = df["volume"] >= median_vol

        long_entry = is_up & conf_ok & price_above_sma & vol_ok
        short_entry = is_down & conf_ok & price_below_sma & vol_ok

        entries = long_entry | short_entry

        # Exit when any of the entry conditions become false for the current side.
        # For simplicity we treat the opposite side as an exit signal.
        exit_long = (~price_above_sma) | (~vol_ok) | (df["ml_prediction"] == "down")
        exit_short = (~price_below_sma) | (~vol_ok) | (df["ml_prediction"] == "up")
        exits = exit_long | exit_short

        # Align boolean Series with BacktestSignals expectations
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return BacktestSignals(entries=entries, exits=exits)