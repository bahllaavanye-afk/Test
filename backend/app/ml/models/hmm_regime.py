"""
3-state Hidden Markov Model for market regime detection.

States:
  0 = bear   — high volatility, negative drift
  1 = sideways — low volatility, near-zero drift
  2 = bull   — low volatility, positive drift

Uses hmmlearn if available, else falls back to a pure-NumPy Baum-Welch
implementation that converges in 50-100 iterations for financial data.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


# ── Baum-Welch fallback (pure NumPy) ─────────────────────────────────────────


class _BaumWelchHMM:
    """Minimal Gaussian HMM via Baum‑Welch EM. 2‑feature input."""

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self._init_params()

    def _init_params(self) -> None:
        K = self.n_states
        self.pi = np.ones(K) / K
        self.A = np.full((K, K), 1.0 / K)
        # Means: bear=(-0.002,0.02), sideways=(0,0.01), bull=(0.002,0.008)
        self.mu = np.array([[-0.002, 0.020], [0.0, 0.010], [0.002, 0.008]])
        self.sigma2 = np.ones((K, 2)) * 0.0001

    def _gaussian_pdf(self, X: np.ndarray) -> np.ndarray:
        """Return emission probabilities (T, K)."""
        T = len(X)
        K = self.n_states
        B = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            exponent = -0.5 * np.sum(diff ** 2 / np.clip(self.sigma2[k], 1e-10, None), axis=1)
            norm = np.sqrt((2 * np.pi) ** 2 * np.prod(np.clip(self.sigma2[k], 1e-10, None)))
            B[:, k] = np.exp(exponent) / np.clip(norm, 1e-300, None)
        return B + 1e-300

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, K = B.shape
        alpha = np.zeros((T, K))
        alpha[0] = self.pi * B[0]
        scale = np.zeros(T)
        scale[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scale[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scale[t]
        return alpha, scale

    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / (scale[t + 1] or 1e-300)
        return beta

    def fit(self, X: np.ndarray) -> "_BaumWelchHMM":
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = self._gaussian_pdf(X)
            alpha, scale = self._forward(B)
            beta = self._backward(B, scale)
            T, K = alpha.shape
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-300)
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                xi[t] = (alpha[t, :, None] * self.A * (B[t + 1] * beta[t + 1]))
                xi[t] /= xi[t].sum() or 1e-300
            # Update parameters
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True).clip(1e-300)
            self.mu = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma.sum(axis=0)[:, None].clip(1e-300)
            for k in range(K):
                diff = X - self.mu[k]
                self.sigma2[k] = (gamma[:, k, None] * diff ** 2).sum(0) / gamma[:, k].sum().clip(1e-300)
                self.sigma2[k] = np.clip(self.sigma2[k], 1e-8, None)
            ll = np.log(scale).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding → state sequence."""
        B = self._gaussian_pdf(X)
        T, K = B.shape
        viterbi = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        viterbi[0] = np.log(self.pi + 1e-300) + np.log(B[0])
        for t in range(1, T):
            trans = viterbi[t - 1, :, None] + np.log(self.A + 1e-300)
            psi[t] = trans.argmax(axis=0)
            viterbi[t] = trans.max(axis=0) + np.log(B[t])
        states = np.zeros(T, dtype=int)
        states[-1] = viterbi[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


# ── Pydantic schemas for persistence ────────────────────────────────────────


class RegimeDetectorState(BaseModel):
    """
    Serialized representation of :class:`RegimeDetector`.

    The schema validates the content loaded from disk, ensuring that
    critical fields have the expected structure.
    """

    model: Optional[Any] = Field(
        default=None,
        description="Fallback NumPy HMM instance (or None if hmmlearn is used).",
    )
    hmmlearn_model: Optional[Any] = Field(
        default=None,
        description="hmmlearn GaussianHMM instance when the optional dependency is available.",
    )
    use_hmmlearn: bool = Field(
        default=False,
        description="Flag indicating which implementation was used during fitting.",
    )
    state_map: Dict[int, int] = Field(
        default_factory=dict,
        description="Mapping from raw state indices (0‑2) to labelled regimes (0‑bear,1‑sideways,2‑bull).",
        example={0: 1, 1: 0, 2: 2},
    )
    fitted: bool = Field(
        default=False,
        description="Whether the detector has been fitted with data.",
    )

    @validator("state_map")
    def validate_state_map(cls, v: Dict[int, int]) -> Dict[int, int]:
        """Ensure keys and values are within the allowed regime range."""
        allowed = {0, 1, 2}
        if set(v.keys()) - allowed:
            raise ValueError("state_map keys must be in {0,1,2}")
        if set(v.values()) - allowed:
            raise ValueError("state_map values must be in {0,1,2}")
        return v


# ── Public API ────────────────────────────────────────────────────────────────


class RegimeDetector:
    """
    3‑state HMM for market regime detection.

    Fit on a return series; features are ``[return, abs_return]``.
    States are automatically labelled as bear/sideways/bull by drift ordering.
    """

    N_STATES = 3

    def __init__(self) -> None:
        self._model: Optional[_BaumWelchHMM] = None
        self._state_map: dict[int, int] = {0: 0, 1: 1, 2: 2}
        self._use_hmmlearn = False
        self._hmmlearn_model: Optional[Any] = None
        self._fitted = False

    def _build_features(self, returns: np.ndarray) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        return np.column_stack([r, np.abs(r)])

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        X = self._build_features(returns)
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore[import]
            model = GaussianHMM(
                n_components=self.N_STATES,
                covariance_type="diag",
                n_iter=200,
                tol=1e-4,
                random_state=42,
            )
            model.fit(X)
            self._hmmlearn_model = model
            self._use_hmmlearn = True
        except ImportError:
            self._model = _BaumWelchHMM(n_states=self.N_STATES, n_iter=150)
            self._model.fit(X)
            self._use_hmmlearn = False

        # Establish state → regime label by drift ordering
        states = self.predict(returns)
        means = [
            float(returns[states == k].mean()) if (states == k).any() else 0.0
            for k in range(self.N_STATES)
        ]
        order = sorted(range(self.N_STATES), key=lambda k: means[k])
        self._state_map = {raw: label for label, raw in enumerate(order)}
        self._fitted = True
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Raw Viterbi state sequence (0/1/2, unordered)."""
        X = self._build_features(returns)
        if self._use_hmmlearn and self._hmmlearn_model is not None:
            return self._hmmlearn_model.predict(X)
        if self._model is not None:
            return self._model.predict(X)
        raise RuntimeError("Model not fitted — call fit() first")

    def predict_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Labelled regime sequence: 0=bear, 1=sideways, 2=bull."""
        raw = self.predict(returns)
        return np.array([self._state_map.get(int(s), s) for s in raw])

    def current_regime(self, returns: np.ndarray) -> int:
        """Returns the current regime label for the most recent bar."""
        return int(self.predict_regimes(returns)[-1])

    def regime_name(self, regime: int) -> str:
        """Human‑readable name for a regime label."""
        return {0: "bear", 1: "sideways", 2: "bull"}.get(regime, "unknown")

    def save(self, path: str) -> None:
        """Persist the detector to ``path`` using ``pickle``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "hmmlearn_model": self._hmmlearn_model,
                    "use_hmmlearn": self._use_hmmlearn,
                    "state_map": self._state_map,
                    "fitted": self._fitted,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """Load a persisted detector from ``path`` and validate its schema."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Validate loaded data against the Pydantic schema
        validated = RegimeDetectorState(**data)

        det = cls()
        det._model = validated.model
        det._hmmlearn_model = validated.hmmlearn_model
        det._use_hmmlearn = validated.use_hmmlearn
        det._state_map = validated.state_map
        det._fitted = validated.fitted
        return det

# End of file