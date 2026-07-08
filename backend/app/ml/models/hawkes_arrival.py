"""
Hawkes Self-Exciting Point Process for Order Arrival Rate Modeling
===================================================================

This module provides a lightweight implementation of a Hawkes self‑exciting
point process that is used within the QuantEdge trading platform to model the
arrival rate of market orders.  The intensity λ(t) evolves according to

    λ(t) = μ + Σ α·exp(-β·(t - t_i))   for t_i < t

where

* μ (mu)   – baseline intensity (events per second),
* α (alpha) – jump size added after each event,
* β (beta) – exponential decay rate (1/second).

The process is fitted to historical trade timestamps using an EM‑like maximum
likelihood estimator and can be queried to predict short‑term arrival intensity.
The resulting intensity estimate is then used by the SmartOrderRouter to decide
whether to execute aggressively (market order) or more passively (limit order).

References
----------
* Hawkes, A. G. (1971). “Spectra of Some Self‑Exciting and Mutually Exciting
  Point Processes.” *Biometrika*, 58(1), 83‑90.
* Filimonov, V., & Sornette, D. (2012). “Quantifying Reflexivity in Financial
  Markets.” *Quantitative Finance*, 12(5), 761‑777.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HawkesParams:
    """
    Container for the fitted Hawkes process parameters.

    Attributes
    ----------
    mu : float
        Baseline intensity (events per second).
    alpha : float
        Jump size added after each event.
    beta : float
        Decay rate (1/second).
    """
    mu: float      # baseline intensity (events/second)
    alpha: float   # jump size on each event
    beta: float    # decay rate (1/s)


class HawkesProcess:
    """
    Hawkes self‑exciting point process for order arrival rate modeling.

    The model is fitted via an iterative EM‑like maximum‑likelihood estimator
    on historical trade timestamps.  It is subsequently used by the
    SmartOrderRouter to decide between market and limit order submission.

    Parameters
    ----------
    beta : float, optional
        Decay rate (1/s) that controls how quickly excitation from past events
        dissipates.  The default value of 1.0 corresponds to a half‑life of
        roughly 0.69 seconds.

    Examples
    --------
    >>> hp = HawkesProcess(beta=2.0)
    >>> params = hp.fit(timestamps)                     # timestamps in Unix seconds
    >>> intensity = hp.predict_intensity(timestamps, horizon_seconds=30)
    >>> order_type = hp.suggest_execution(intensity, threshold=5.0)
    >>> # 'market' if busy, 'limit' if quiet
    """

    def __init__(self, beta: float = 1.0) -> None:
        """
        Initialise a HawkesProcess instance.

        Parameters
        ----------
        beta : float
            Positive decay rate (1/s).  Raises ``TypeError`` if ``beta`` is not a
            numeric type and ``ValueError`` if it is non‑positive.
        """
        if not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be a numeric type, got {type(beta)}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta: float = float(beta)
        self.params: Optional[HawkesParams] = None

    def fit(self, timestamps: np.ndarray) -> HawkesParams:
        """
        Fit the Hawkes process parameters to a sequence of trade timestamps.

        The fitting routine follows an EM‑like iterative scheme (Veen & Schoenberg,
        2008) that maximises the likelihood of the observed event times under the
        Hawkes model.

        Parameters
        ----------
        timestamps : np.ndarray
            One‑dimensional array of Unix timestamps (seconds).  The array must
            contain at least ten events; otherwise a default stable parameter set
            is returned.

        Returns
        -------
        HawkesParams
            The estimated ``mu``, ``alpha`` and ``beta`` values.  ``beta`` is fixed
            to the value supplied at construction time.

        Raises
        ------
        TypeError
            If ``timestamps`` cannot be converted to a numeric NumPy array.
        ValueError
            If ``timestamps`` is not one‑dimensional.
        RuntimeError
            If an unexpected error occurs during the EM iteration.
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

        # Ensure timestamps are sorted
        timestamps = np.sort(timestamps)
        T = float(timestamps[-1] - timestamps[0])
        if T < 1e-9:
            logger.warning("Timestamp range too small (T=%.2e); using default parameters", T)
            return HawkesParams(mu=1.0, alpha=0.5, beta=self.beta)

        n = len(timestamps)
        mu = n / T * 0.5
        alpha = 0.3
        beta = self.beta

        for _ in range(50):  # EM-like iterations
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
        Predict the expected number of order arrivals over a future horizon.

        The prediction uses the current excitation level inferred from the most
        recent timestamps (by default the last five minutes).

        Parameters
        ----------
        timestamps : np.ndarray
            Recent trade timestamps (Unix seconds), sorted in ascending order.
        horizon_seconds : float, optional
            Length of the prediction window in seconds.  Must be positive.

        Returns
        -------
        float
            Expected number of arrivals in the interval
            ``[t_last, t_last + horizon_seconds]`` where ``t_last`` is the most
            recent timestamp.

        Raises
        ------
        TypeError
            If ``timestamps`` cannot be converted to a numeric NumPy array or if
            ``horizon_seconds`` is not numeric.
        ValueError
            If ``timestamps`` is not one‑dimensional or if ``horizon_seconds`` is
            non‑positive.
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
            logger.error("Non-positive horizon_seconds: %f", horizon_seconds)
            raise ValueError("horizon_seconds must be positive")

        p = self.params
        t_last = float(timestamps[-1])
        recent = timestamps[timestamps > t_last - 300.0]  # last 5 minutes
        carry = float(
            p.alpha * p.beta * np.sum(np.exp(-p.beta * (t_last - recent)))
        )
        lam = p.mu + carry
        intensity = float(lam * horizon_seconds)
        logger.debug(
            "Predicted intensity: mu=%f, carry=%f, lambda=%f, horizon=%f -> intensity=%f",
            p.mu, carry, lam, horizon_seconds, intensity,
        )
        return intensity

    def suggest_execution(
        self,
        intensity: float,
        threshold: float = 5.0,
    ) -> str:
        """
        Recommend an order execution type based on predicted arrival intensity.

        A high predicted intensity indicates a busy market with abundant liquidity,
        favouring immediate market orders.  Conversely, a low intensity suggests a
        quieter market where passive limit orders are preferable.

        Parameters
        ----------
        intensity : float
            Predicted number of arrivals over the horizon (output of
            :meth:`predict_intensity`).
        threshold : float, optional
            Arrival count that separates the “limit” and “market” regimes.  The
            default value of 5.0 is a heuristic that works well for typical
            crypto trading horizons.

        Returns
        -------
        str
            Either ``"market"`` if ``intensity`` exceeds ``threshold`` or
            ``"limit"`` otherwise.
        """
        if not isinstance(intensity, (int, float)):
            logger.error("Invalid intensity type: %s", type(intensity))
            raise TypeError("intensity must be a numeric type")
        if not isinstance(threshold, (int, float)):
            logger.error("Invalid threshold type: %s", type(threshold))
            raise TypeError("threshold must be a numeric type")
        if threshold <= 0:
            logger.error("Non-positive threshold: %f", threshold)
            raise ValueError("threshold must be positive")

        decision = "market" if intensity >= threshold else "limit"
        logger.debug(
            "Suggested execution: intensity=%f, threshold=%f -> %s",
            intensity, threshold, decision,
        )
        return decision