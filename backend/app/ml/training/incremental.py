"""
Incremental (online) training core for the AutoML desk.

The misconception this addresses: a full from-scratch train of an LSTM/PatchTST
takes hours. But *fine-tuning an already-trained champion on the newest bars*
takes seconds — a handful of low-learning-rate gradient steps on a small recent
window. That is what lets the platform "improve with every second of real data"
without a GPU farm.

This module is deliberately split from the orchestration loop (automl_desk.py)
so the numerics are unit-testable in isolation, with no Redis / network / event
loop involved.

Champion/challenger discipline:
  - The live model is the *champion*.
  - Each cycle we deep-copy it, fine-tune the copy on the newest window → that's
    the *challenger*.
  - Both are scored on a held-out validation slice. The challenger only replaces
    the champion if it beats it by a margin (default 1%). Otherwise we keep the
    champion — fine-tuning on noisy recent data must never silently degrade live
    predictions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationScore:
    accuracy: float
    directional_sharpe: float
    n: int

    @property
    def combined(self) -> float:
        """
        Single comparable scalar for champion/challenger.
        Accuracy anchors it in [0,1]; a scaled Sharpe term rewards models whose
        directional calls actually compound. Sharpe is squashed so a single wild
        fold can't dominate.
        """
        squashed_sharpe = float(np.tanh(self.directional_sharpe / 3.0))
        return self.accuracy + 0.25 * squashed_sharpe


def directional_sharpe(
    probs: np.ndarray,
    forward_returns: np.ndarray,
    annualisation: float = float(np.sqrt(252)),
    eps: float = 1e-9,
) -> float:
    """
    Sharpe of the strategy that takes position sign(prob - 0.5) and earns the
    corresponding forward return. Pure numpy so it tests without torch.
    """
    probs = np.asarray(probs, dtype=float).flatten()
    forward_returns = np.asarray(forward_returns, dtype=float).flatten()
    n = min(len(probs), len(forward_returns))
    if n == 0:
        return 0.0
    probs, forward_returns = probs[:n], forward_returns[:n]
    positions = np.sign(probs - 0.5)
    pnl = positions * forward_returns
    std = pnl.std()
    if std < eps:
        return 0.0
    return float(pnl.mean() / std * annualisation)


def should_promote(
    champion: ValidationScore | None,
    challenger: ValidationScore,
    min_improvement: float = 0.01,
    min_samples: int = 20,
) -> bool:
    """
    Decide whether the challenger replaces the champion.

    Rules:
      - Need at least `min_samples` validation points (don't promote on noise).
      - With no champion (cold start), promote any minimally-valid challenger.
      - Otherwise require combined score to beat the champion by `min_improvement`.
    """
    if challenger.n < min_samples:
        return False
    if champion is None:
        return True
    return challenger.combined >= champion.combined + min_improvement


# ---------------------------------------------------------------------------
# Torch-dependent pieces — imported lazily so the module loads without torch.
# ---------------------------------------------------------------------------

def build_supervised(
    df,
    seq_len: int = 60,
    horizon: int = 1,
    threshold: float = 0.002,
    scaler=None,
):
    """
    Turn a raw OHLCV DataFrame into (X, y, forward_returns, scaler) for training.

    - Engineers features, adds binary direction labels.
    - Fits a FeatureScaler if none supplied (so the desk can cold-start), else
      reuses the champion's scaler for consistency.
    - Returns forward_returns aligned to y so we can compute directional Sharpe.

    Raises ValueError if there is not enough usable data — never fabricates rows.
    """
    from app.ml.features.engineer import (
        engineer_features, create_sequences, add_labels, FEATURE_COLS,
    )
    from app.ml.features.normalization import FeatureScaler

    feat = engineer_features(df, normalize=False)
    feat = add_labels(feat, horizon=horizon, threshold=threshold)
    if len(feat) < seq_len + 10:
        raise ValueError(f"insufficient data: {len(feat)} rows for seq_len={seq_len}")

    active_cols = [c for c in FEATURE_COLS if c in feat.columns]
    if not active_cols:
        raise ValueError("no feature columns present after engineering")

    if scaler is None:
        scaler = FeatureScaler()
        scaler.fit(feat[active_cols])
    feat_scaled = feat.copy()
    feat_scaled[active_cols] = scaler.transform(feat[active_cols])

    X, y = create_sequences(feat_scaled, seq_len=seq_len)
    if X is None or len(X) == 0:
        raise ValueError("no sequences produced")

    # Forward returns aligned to each sequence's label (next-bar pct change).
    fwd = feat["close"].pct_change(horizon).shift(-horizon).fillna(0.0).values
    fwd_aligned = fwd[seq_len:seq_len + len(X)]

    return X, y, fwd_aligned, scaler


def validate_model(model, X, y, forward_returns) -> ValidationScore:
    """Score a model on a validation slice. Lazy-imports torch."""
    import torch

    if not hasattr(model, "eval"):
        raise TypeError("model must be an nn.Module-like object")
    model.eval()
    with torch.no_grad():
        probs = np.asarray(model.predict_proba(X)).flatten()

    y_np = np.asarray(y.numpy() if hasattr(y, "numpy") else y, dtype=float).flatten()
    n = min(len(probs), len(y_np))
    if n == 0:
        return ValidationScore(accuracy=0.0, directional_sharpe=0.0, n=0)
    preds = (probs[:n] > 0.5).astype(int)
    accuracy = float((preds == y_np[:n].astype(int)).mean())
    dsharpe = directional_sharpe(probs[:n], forward_returns[:n])
    return ValidationScore(accuracy=accuracy, directional_sharpe=dsharpe, n=n)


def fine_tune(champion, X_train, y_train, epochs: int = 2, lr: float = 1e-4):
    """
    Deep-copy the champion and fine-tune the copy on the recent window.

    Low LR + few epochs = adaptation without catastrophic forgetting. Returns the
    fine-tuned challenger (the original champion is never mutated).
    """
    import torch
    import torch.nn as nn

    challenger = copy.deepcopy(champion)
    if not hasattr(challenger, "parameters"):
        raise TypeError("champion must expose .parameters() (nn.Module)")

    X_t = X_train if isinstance(X_train, torch.Tensor) else torch.tensor(np.asarray(X_train), dtype=torch.float32)
    y_t = y_train if isinstance(y_train, torch.Tensor) else torch.tensor(np.asarray(y_train), dtype=torch.float32)

    optimizer = torch.optim.AdamW(challenger.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    challenger.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        logits = challenger.forward(X_t)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        loss = criterion(logits, y_t.float())
        loss.backward()
        nn.utils.clip_grad_norm_(challenger.parameters(), 1.0)
        optimizer.step()
    challenger.eval()
    return challenger
