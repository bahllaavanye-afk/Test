"""
Hawkes Self-Exciting Point Process for Order Arrival Rate Modeling
===================================================================
Used to time crypto execution: execute aggressively during high-intensity
periods (liquidity events), use limit orders during quiet periods.

λ(t) = μ + Σ α·exp(-β·(t-t_i)) for t_i < t

Fitted via MLE on historical trade timestamps.

Reference: Hawkes (1971) "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
           Filimonov & Sornette (2012) "Quantifying Reflexivity in Financial Markets"
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HawkesParams:
    mu: float      # baseline intensity (events/second)
    alpha: float   # jump size on each event
    beta: float    # decay rate (1/s)


class HawkesProcess:
    """
    Hawkes self-exciting point process for order arrival rate modeling.

    Fitted via iterative EM-like MLE on historical trade timestamps.
    Used by the SmartOrderRouter to decide market vs limit order submission.

    Parameters:
        beta: decay rate (1/s), controls how quickly excitation dies off.
              Default 1.0 → half-life of ~0.69 seconds.

    Usage:
        hp = HawkesProcess(beta=2.0)
        params = hp.fit(timestamps)   # timestamps in Unix seconds
        intensity = hp.predict_intensity(timestamps, horizon_seconds=30)
        order_type = hp.suggest_execution(intensity, threshold=5.0)
        # 'market' if busy, 'limit' if quiet
    """

    def __init__(self, beta: float = 1.0):
        if not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be a numeric type, got {type(beta)}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = float(beta)
        self.params: HawkesParams | None = None

    def fit(self, timestamps: np.ndarray) -> HawkesParams:
        """
        MLE fit of Hawkes process parameters to trade timestamps (Unix seconds).

        Uses EM-like iterative estimation (Veen & Schoenberg 2008).

        Args:
            timestamps: sorted 1-D array of Unix timestamps in seconds.
                        Must have at least 10 events.

        Returns:
            HawkesParams(mu, alpha, beta) with fitted parameters.
            Returns default stable params if timestamps is too short.
        """
        start_time = time.perf_counter()
        try:
            timestamps = np.asarray(timestamps, dtype=float)
        except Exception as exc:
            logger.error("Failed to convert timestamps to numpy array", exc_info=True)
            raise TypeError("timestamps must be array-like of numeric values") from exc

        if timestamps.ndim != 1:
            logger.error("Timestamps array is not one-dimensional: shape=%s", timestamps.shape)
            raise ValueError("timestamps must be a one-dimensional array")

        signal_count = len(timestamps)

        if signal_count < 10:
            logger.info(
                "Insufficient timestamps; using default parameters",
                extra={"signal_count": signal_count, "execution_time": time.perf_counter() - start_time, "pnl": None},
            )
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        # Sort just in case
        timestamps = np.sort(timestamps)
        T = float(timestamps[-1] - timestamps[0])
        if T < 1e-9:
            logger.warning(
                "Timestamp range too small; using default parameters",
                extra={"signal_count": signal_count, "execution_time": time.perf_counter() - start_time, "pnl": None},
            )
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        n = signal_count
        mu = n / T * 0.5
        alpha = 0.3
        beta = self.beta

        for _ in range(50):  # EM-like iterations
            try:
                # E-step: compute conditional intensities at each event time
                intensities = np.empty(n, dtype=float)
                for i in range(n):
                    ti = timestamps[i]
                    prev_diffs = ti - timestamps[:i]
                    excitation = (
                        alpha * beta * np.sum(np.exp(-beta * prev_diffs))
                        if i > 0 else 0.0
                    )
                    intensities[i] = max(mu + excitation, 1e-10)

                # M-step: update mu and alpha
                inv_int = 1.0 / intensities

                mu_new = mu * np.sum(inv_int) / (T + 1e-10)
                mu_new = float(np.clip(mu_new, 1e-10, n / T))

                if n > 1:
                    excitation_sums = np.array([
                        beta * np.sum(np.exp(-beta * (timestamps[i] - timestamps[:i])))
                        if i > 0 else 0.0
                        for i in range(1, n)
                    ])
                    alpha_new = float(
                        np.sum(inv_int[1:] * excitation_sums) / max(n, 1)
                    )
                    alpha_new = float(np.clip(alpha_new, 0.01, 0.99))
                else:
                    alpha_new = alpha

                mu, alpha = mu_new, alpha_new
            except Exception as exc:
                logger.error("Error during EM iteration", exc_info=True)
                raise RuntimeError("EM iteration failed") from exc

        self.params = HawkesParams(mu=float(mu), alpha=float(alpha), beta=self.beta)

        logger.info(
            "Fit completed",
            extra={
                "signal_count": signal_count,
                "execution_time": time.perf_counter() - start_time,
                "pnl": None,
                "mu": self.params.mu,
                "alpha": self.params.alpha,
                "beta": self.params.beta,
            },
        )
        return self.params

    def predict_intensity(
        self,
        timestamps: np.ndarray,
        horizon_seconds: float = 30.0,
    ) -> float:
        """
        Predict expected number of arrivals in the next horizon_seconds.

        Uses current excitation level from the last 5 minutes of timestamps.

        Args:
            timestamps: recent trade timestamps (Unix seconds), sorted.
            horizon_seconds: prediction window length.

        Returns:
            Expected number of events in [t_last, t_last + horizon_seconds].
        """
        start_time = time.perf_counter()

        if self.params is None:
            logger.debug("Parameters not fitted; returning default intensity 1.0")
            return 1.0

        try:
            timestamps = np.asarray(timestamps, dtype=float)
        except Exception as exc:
            logger.error("Failed to convert timestamps to numpy array", exc_info=True)
            raise TypeError("timestamps must be array-like of numeric values") from exc

        if timestamps.ndim != 1:
            logger.error("Timestamps array is not one-dimensional: shape=%s", timestamps.shape)
            raise ValueError("timestamps must be a one-dimensional array")

        signal_count = len(timestamps)

        if signal_count == 0:
            intensity = float(self.params.mu * horizon_seconds)
            logger.debug("Empty timestamps; returning baseline intensity %f", intensity)
            logger.info(
                "Predicted intensity (baseline)",
                extra={"signal_count": signal_count, "execution_time": time.perf_counter() - start_time, "pnl": None, "intensity": intensity},
            )
            return intensity

        if not isinstance(horizon_seconds, (int, float)):
            logger.error("Invalid horizon_seconds type: %s", type(horizon_seconds))
            raise TypeError("horizon_seconds must be a numeric type")
        if horizon_seconds <= 0:
            logger.error("Non-positive horizon_seconds: %f", horizon_seconds)
            raise ValueError("horizon_seconds must be positive")

        p = self.params
        t_last = float(timestamps[-1])
        recent = timestamps[timestamps > t_last - 300.0]
        carry = float(
            p.alpha * p.beta * np.sum(np.exp(-p.beta * (t_last - recent)))
        )
        lam = p.mu + carry
        intensity = float(lam * horizon_seconds)

        logger.debug(
            "Predicted intensity: mu=%f, carry=%f, lambda=%f, horizon=%f -> intensity=%f",
            p.mu, carry, lam, horizon_seconds, intensity,
        )
        logger.info(
            "Predicted intensity",
            extra={
                "signal_count": signal_count,
                "execution_time": time.perf_counter() - start_time,
                "pnl": None,
                "intensity": intensity,
                "mu": p.mu,
                "carry": carry,
                "lambda": lam,
                "horizon_seconds": horizon_seconds,
            },
        )
        return intensity

    def suggest_execution(
        self,
        intensity: float,
        threshold: float = 5.0,
    ) -> str:
        """
        Recommend order type based on predicted arrival intensity.

        High intensity (many orders arriving) → market order (good liquidity).
        Low intensity (few orders) → limit order (avoid crossing spread).

        Args:
            intensity: predicted arrivals in horizon from predict_intensity().
            threshold: arrivals cutoff between limit and market
        """
        start_time = time.perf_counter()
        if intensity >= threshold:
            decision = "market"
        else:
            decision = "limit"

        logger.info(
            "Execution suggestion",
            extra={
                "intensity": intensity,
                "threshold": threshold,
                "decision": decision,
                "execution_time": time.perf_counter() - start_time,
                "pnl": None,
            },
        )
        return decision

# End of file