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

    _MIN_SWITCH_INTERVAL = 10.0  # seconds: minimum time between order‑type changes

    def __init__(self, beta: float = 1.0):
        if not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be a numeric type, got {type(beta)}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = float(beta)
        self.params: HawkesParams | None = None
        self._last_suggested: str | None = None
        self._last_change_ts: float = 0.0

    def fit(self, timestamps: np.ndarray) -> HawkesParams:
        """
        MLE fit of Hawkes process parameters to trade timestamps (Unix seconds).

        Uses EM-like iterative estimation (Veen & Schoenberg 2008).

        Args:
            timestamps: sorted 1‑D array of Unix timestamps in seconds.
                        Must have at least 10 events.

        Returns:
            HawkesParams(mu, alpha, beta) with fitted parameters.
            Returns default stable params if timestamps is too short.
        """
        try:
            timestamps = np.asarray(timestamps, dtype=float)
        except Exception as exc:
            logger.error("Failed to convert timestamps to numpy array", exc_info=True)
            raise TypeError("timestamps must be array-like of numeric values") from exc

        if timestamps.ndim != 1:
            logger.error("Timestamps array is not one-dimensional: shape=%s", timestamps.shape)
            raise ValueError("timestamps must be a one-dimensional array")

        if len(timestamps) < 10:
            logger.info("Insufficient timestamps (%d); using default parameters", len(timestamps))
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        # Ensure chronological order
        timestamps = np.sort(timestamps)
        T = float(timestamps[-1] - timestamps[0])
        if T < 1e-9:
            logger.warning("Timestamp range too small (T=%.2e); using default parameters", T)
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        n = len(timestamps)
        mu = n / T * 0.5
        alpha = 0.3
        beta = self.beta

        for _ in range(50):  # EM‑like iterations
            try:
                # E‑step: compute conditional intensities at each event time
                intensities = np.empty(n, dtype=float)
                for i in range(n):
                    ti = timestamps[i]
                    prev_diffs = ti - timestamps[:i]
                    excitation = (
                        alpha * beta * np.sum(np.exp(-beta * prev_diffs))
                        if i > 0 else 0.0
                    )
                    intensities[i] = max(mu + excitation, 1e-10)

                # M‑step: update mu and alpha
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
        return self.params

    def predict_intensity(
        self,
        timestamps: np.ndarray,
        horizon_seconds: float = 30.0,
    ) -> float:
        """
        Predict expected number of arrivals in the next ``horizon_seconds``.

        Uses current excitation level from the last 5 minutes of timestamps.

        Args:
            timestamps: recent trade timestamps (Unix seconds), sorted.
            horizon_seconds: prediction window length.

        Returns:
            Expected number of events in ``[t_last, t_last + horizon_seconds]``.
        """
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

        if len(timestamps) == 0:
            intensity = float(self.params.mu * horizon_seconds)
            logger.debug("Empty timestamps; returning baseline intensity %f", intensity)
            return intensity

        if not isinstance(horizon_seconds, (int, float)):
            logger.error("Invalid horizon_seconds type: %s", type(horizon_seconds))
            raise TypeError("horizon_seconds must be a numeric type")
        if horizon_seconds <= 0:
            logger.error("Non‑positive horizon_seconds: %f", horizon_seconds)
            raise ValueError("horizon_seconds must be positive")

        p = self.params
        t_last = float(timestamps[-1])
        recent = timestamps[timestamps > t_last - 300.0]  # last 5 min
        carry = float(p.alpha * p.beta * np.sum(np.exp(-p.beta * (t_last - recent))))
        lam = p.mu + carry
        intensity = float(lam * horizon_seconds)
        logger.debug(
            "Predicted intensity: mu=%f, carry=%f, lambda=%f, horizon=%f -> intensity=%f",
            p.mu, carry, lam, horizon_seconds, intensity,
        )
        return intensity

    def _confirmation_filter(self, intensity: float, mu: float) -> bool:
        """
        Internal helper to decide whether the intensity signal is strong enough.

        Returns ``True`` if the intensity exceeds both an absolute ``threshold``
        and a relative factor over the baseline ``mu``.
        """
        # Absolute threshold handled by ``suggest_execution``; here we enforce a
        # relative condition to avoid false‑positive entries during periods of
        # modest baseline activity.
        relative_factor = 2.0  # require at least double the baseline rate
        return intensity > mu * relative_factor

    def suggest_execution(
        self,
        intensity: float,
        threshold: float = 5.0,
    ) -> str:
        """
        Recommend order type based on predicted arrival intensity.

        * **Market order** – high intensity (many orders arriving) → good liquidity.
        * **Limit order** – low intensity (few orders) → avoid crossing spread.

        The decision is tightened by:
        - requiring intensity to be above ``threshold`` **and** at least twice the
          baseline intensity (``mu``);
        - enforcing a minimum ``_MIN_SWITCH_INTERVAL`` between consecutive
          changes to prevent churning.

        Args:
            intensity: predicted arrivals in horizon from ``predict_intensity``.
            threshold: absolute arrivals cutoff between limit and market.

        Returns:
            ``'market'`` or ``'limit'``.
        """
        if not isinstance(intensity, (int, float)):
            logger.error("Invalid intensity type: %s", type(intensity))
            raise TypeError("intensity must be a numeric type")
        if intensity < 0:
            logger.error("Negative intensity received: %f", intensity)
            raise ValueError("intensity must be non‑negative")

        # Baseline reference for relative comparison
        mu = self.params.mu if self.params else 1.0

        # Determine raw signal
        raw_signal = intensity > threshold and self._confirmation_filter(intensity, mu)

        # Resolve final suggestion with churn protection
        now = time.time()
        desired = "market" if raw_signal else "limit"

        if self._last_suggested is None:
            # First call – set state without restriction
            self._last_suggested = desired
            self._last_change_ts = now
            logger.debug("Initial execution suggestion: %s", desired)
            return desired

        if desired != self._last_suggested:
            time_since_change = now - self._last_change_ts
            if time_since_change < self._MIN_SWITCH_INTERVAL:
                # Keep previous suggestion until the interval expires
                logger.debug(
                    "Switch suppressed (%.2fs < %.2fs); keeping %s",
                    time_since_change,
                    self._MIN_SWITCH_INTERVAL,
                    self._last_suggested,
                )
                return self._last_suggested
            else:
                logger.debug(
                    "Execution suggestion changed from %s to %s after %.2fs",
                    self._last_suggested,
                    desired,
                    time_since_change,
                )
                self._last_suggested = desired
                self._last_change_ts = now
                return desired

        # No change needed
        logger.debug("Execution suggestion unchanged: %s", desired)
        return desired

    # --------------------------------------------------------------------- #
    # Additional helper for external callers that need explicit exit logic
    # --------------------------------------------------------------------- #
    def should_exit_market(
        self,
        intensity: float,
        exit_threshold: float = 3.0,
    ) -> bool:
        """
        Determine whether a previously taken market order should be exited
        (i.e., switch to limit) based on a lower intensity threshold.

        This method can be used by higher‑level strategies to implement a
        graceful exit from aggressive execution when liquidity dries up.

        Args:
            intensity: current predicted intensity.
            exit_threshold: intensity below which the market position should be
                reconsidered.

        Returns:
            ``True`` if the strategy should move to limit orders.
        """
        if not isinstance(intensity, (int, float)):
            raise TypeError("intensity must be numeric")
        if intensity < 0:
            raise ValueError("intensity must be non‑negative")
        # Use a relative component similar to the entry filter
        mu = self.params.mu if self.params else 1.0
        relative_factor = 0.5  # drop below half the baseline to trigger exit
        return intensity < exit_threshold and intensity < mu * relative_factor