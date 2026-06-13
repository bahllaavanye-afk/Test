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

from dataclasses import dataclass

import numpy as np


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
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta
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
        timestamps = np.asarray(timestamps, dtype=float)
        if len(timestamps) < 10:
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        # Sort just in case
        timestamps = np.sort(timestamps)
        T = float(timestamps[-1] - timestamps[0])
        if T < 1e-9:
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        n = len(timestamps)
        mu = n / T * 0.5
        alpha = 0.3
        beta = self.beta

        for _ in range(50):  # EM-like iterations
            # E-step: compute conditional intensities at each event time
            intensities = np.empty(n)
            for i in range(n):
                ti = timestamps[i]
                prev_diffs = ti - timestamps[:i]
                excitation = alpha * beta * np.sum(np.exp(-beta * prev_diffs)) if i > 0 else 0.0
                intensities[i] = max(mu + excitation, 1e-10)

            # M-step: update mu and alpha
            inv_int = 1.0 / intensities

            # mu update: fraction of baseline contribution
            mu_new = mu * np.sum(inv_int) / (T + 1e-10)
            mu_new = float(np.clip(mu_new, 1e-10, n / T))

            # alpha update: excitation contribution
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

            mu = mu_new
            alpha = alpha_new

        self.params = HawkesParams(mu=float(mu), alpha=float(alpha), beta=self.beta)
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
        if self.params is None:
            return 1.0

        timestamps = np.asarray(timestamps, dtype=float)
        if len(timestamps) == 0:
            return float(self.params.mu * horizon_seconds)

        p = self.params
        t_last = float(timestamps[-1])
        # Use only last 5 minutes to compute carry-over excitation
        recent = timestamps[timestamps > t_last - 300.0]
        carry = float(
            p.alpha * p.beta * np.sum(np.exp(-p.beta * (t_last - recent)))
        )
        lam = p.mu + carry
        # Expected arrivals in horizon = λ * horizon (Poisson approximation)
        return float(lam * horizon_seconds)

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
            threshold: arrivals cutoff between limit and market order.

        Returns:
            'market' if intensity > threshold, else 'limit'.
        """
        return "market" if intensity > threshold else "limit"
