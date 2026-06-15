"""
Custom loss functions for financial ML models.

Key insight: BCE optimises accuracy, not alpha. These losses directly
optimise Sharpe ratio and trading-relevant metrics.

References:
- Sharpe-differentiable loss: Lim et al. (2021) TFT paper Appendix B
- Focal loss: Lin et al. (2017) — handles class imbalance in rare signals
- Sortino-maximising: penalises downside variance only
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpeLoss(nn.Module):
    """
    Differentiable Sharpe ratio loss.

    Instead of binary cross-entropy, directly maximises the Sharpe ratio
    of the predicted portfolio returns.

    portfolio_return[t] = pred[t] * actual_return[t]
    Loss = -Sharpe(portfolio_return)

    Args:
        annualisation: sqrt(bars_per_year) for Sharpe annualisation
        eps: stability floor to avoid division by zero
    """
    def __init__(self, annualisation: float = 252 ** 0.5, eps: float = 1e-6):
        super().__init__()
        self.annualisation = annualisation
        self.eps = eps

    def forward(self, pred: torch.Tensor, actual_returns: torch.Tensor) -> torch.Tensor:
        """
        pred:            (batch,) sigmoid probabilities [0, 1]
        actual_returns:  (batch,) forward returns (e.g. next-day pct change)
        """
        # Convert probability to direction weight: [0,1] → [-1,+1]
        position = 2.0 * pred - 1.0
        portfolio_ret = position * actual_returns
        mean_ret = portfolio_ret.mean()
        std_ret = portfolio_ret.std() + self.eps
        sharpe = mean_ret / std_ret * self.annualisation
        return -sharpe  # minimise negative Sharpe


class SortinoLoss(nn.Module):
    """
    Sortino loss: only penalises downside deviation (unlike Sharpe which
    penalises all volatility). Better for strategies with positive skew.
    """
    def __init__(self, annualisation: float = 252 ** 0.5, eps: float = 1e-6):
        super().__init__()
        self.annualisation = annualisation
        self.eps = eps

    def forward(self, pred: torch.Tensor, actual_returns: torch.Tensor) -> torch.Tensor:
        position = 2.0 * pred - 1.0
        portfolio_ret = position * actual_returns
        mean_ret = portfolio_ret.mean()
        downside = portfolio_ret[portfolio_ret < 0]
        downside_std = downside.std() + self.eps if len(downside) > 1 else torch.tensor(self.eps)
        sortino = mean_ret / downside_std * self.annualisation
        return -sortino


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced trading signal classification.
    Rare profitable signals get higher weight — fixes the common issue
    where models learn to always predict "flat".

    gamma=2.0 is the standard; higher gamma focuses more on hard examples.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class HybridLoss(nn.Module):
    """
    Combines BCE (for classification accuracy) + Sharpe (for alpha).

    Loss = alpha * BCE + (1 - alpha) * SharpeLoss

    Start training with alpha=0.8 (BCE dominant) to stabilise gradients,
    then anneal towards alpha=0.2 to focus on Sharpe.
    """
    def __init__(self, alpha: float = 0.5, annualisation: float = 252 ** 0.5):
        super().__init__()
        self.alpha = alpha
        self.sharpe_loss = SharpeLoss(annualisation=annualisation)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        actual_returns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target.float())
        if actual_returns is None or self.alpha >= 0.999:
            return bce
        pred_proba = torch.sigmoid(pred_logits)
        sharpe = self.sharpe_loss(pred_proba, actual_returns)
        return self.alpha * bce + (1.0 - self.alpha) * sharpe


class LabelSmoothingBCE(nn.Module):
    """
    Binary cross-entropy with label smoothing.
    Prevents overconfident models that are brittle out-of-sample.
    Equivalent to adding noise ε to labels: 1 → 1-ε, 0 → ε.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_target = target.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, smooth_target)


def get_loss(
    loss_name: str = "bce",
    sharpe_alpha: float = 0.5,
    annualisation: float = 252 ** 0.5,
) -> nn.Module:
    """
    Factory: return the correct loss module by name.

    loss_name options:
      "bce"            → standard BCE (baseline)
      "focal"          → focal loss (imbalanced signals)
      "label_smooth"   → BCE + label smoothing (prevents overconfidence)
      "sharpe"         → pure Sharpe maximisation
      "sortino"        → Sortino maximisation (long-vol strategies)
      "hybrid"         → sharpe_alpha * BCE + (1-sharpe_alpha) * Sharpe [RECOMMENDED]
    """
    mapping = {
        "bce":          nn.BCEWithLogitsLoss,
        "focal":        lambda: FocalLoss(),
        "label_smooth": lambda: LabelSmoothingBCE(),
        "sharpe":       lambda: SharpeLoss(annualisation=annualisation),
        "sortino":      lambda: SortinoLoss(annualisation=annualisation),
        "hybrid":       lambda: HybridLoss(alpha=sharpe_alpha, annualisation=annualisation),
    }
    factory = mapping.get(loss_name)
    if factory is None:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose from: {list(mapping)}")
    return factory()
