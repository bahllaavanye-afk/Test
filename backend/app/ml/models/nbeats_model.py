"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.
Oreshkin et al. (2020) — https://arxiv.org/abs/1905.10437

Architecture:
  - Stack of blocks, each with:
    - FC layers for backcast (reconstruction of input)
    - FC layers for forecast (prediction)
  - Generic stack: learned basis functions
  - Trend stack: polynomial basis functions
  - Seasonality stack: Fourier basis functions

For trading: use as a directional classifier — final output is sigmoid(forecast[-1]).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class NBEATSBlock(nn.Module):
    """Single N-BEATS block with backcast and forecast outputs."""

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        basis_function: nn.Module,
        layers: int = 4,
        layer_size: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size if i == 0 else layer_size, layer_size)
            for i in range(layers)
        ])
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor):
        block_input = x
        for layer in self.layers:
            block_input = F.relu(layer(block_input))
        theta = self.basis_parameters(block_input)
        backcast, forecast = self.basis_function(theta)
        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic (learned) basis — no structural assumptions."""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, self.backcast_size:]


class TrendBasis(nn.Module):
    """Polynomial trend basis."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        backcast_grid = torch.linspace(0, 1, backcast_size)
        forecast_grid = torch.linspace(1, 2, forecast_size)
        self.register_buffer("backcast_poly", torch.stack([backcast_grid ** i for i in range(degree + 1)], dim=0))
        self.register_buffer("forecast_poly", torch.stack([forecast_grid ** i for i in range(degree + 1)], dim=0))

    def forward(self, theta: torch.Tensor):
        cut = theta.shape[1] // 2
        backcast = theta[:, :cut] @ self.backcast_poly
        forecast = theta[:, cut:] @ self.forecast_poly
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Fourier seasonality basis."""

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        t_b = torch.linspace(0, 1, backcast_size)
        t_f = torch.linspace(1, 2, forecast_size)
        freq = torch.arange(1, harmonics + 1).float()
        self.register_buffer("backcast_cos", torch.cos(2 * torch.pi * freq.unsqueeze(1) * t_b.unsqueeze(0)))
        self.register_buffer("backcast_sin", torch.sin(2 * torch.pi * freq.unsqueeze(1) * t_b.unsqueeze(0)))
        self.register_buffer("forecast_cos", torch.cos(2 * torch.pi * freq.unsqueeze(1) * t_f.unsqueeze(0)))
        self.register_buffer("forecast_sin", torch.sin(2 * torch.pi * freq.unsqueeze(1) * t_f.unsqueeze(0)))

    def forward(self, theta: torch.Tensor):
        n = theta.shape[1] // 4
        backcast = (theta[:, :n] @ self.backcast_cos + theta[:, n:2 * n] @ self.backcast_sin)
        forecast = (theta[:, 2 * n:3 * n] @ self.forecast_cos + theta[:, 3 * n:4 * n] @ self.forecast_sin)
        return backcast, forecast


class NBEATSPredictor(nn.Module):
    """
    Full N-BEATS model for binary trading signal prediction.

    Uses 3 stacks: Generic + Trend + Seasonality.
    Final output: sigmoid of the last forecast step → probability of up move.

    Compatible interface with LSTMPredictor (same forward signature).
    """

    def __init__(
        self,
        input_size: int = 1800,    # seq_len * n_features (60 * 30); must match flattened input
        forecast_steps: int = 1,   # 1-step ahead
        layer_size: int = 256,
        n_layers: int = 4,
        trend_degree: int = 3,
        harmonics: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.forecast_steps = forecast_steps
        self.dropout = nn.Dropout(dropout)

        # Stack 1: Generic
        generic_basis = GenericBasis(input_size, forecast_steps)
        generic_theta = input_size + forecast_steps

        # Stack 2: Trend
        trend_basis = TrendBasis(trend_degree, input_size, forecast_steps)
        trend_theta = (trend_degree + 1) * 2

        # Stack 3: Seasonality
        seasonal_basis = SeasonalityBasis(harmonics, input_size, forecast_steps)
        seasonal_theta = harmonics * 4

        self.stacks = nn.ModuleList([
            NBEATSBlock(input_size, generic_theta, generic_basis, n_layers, layer_size),
            NBEATSBlock(input_size, trend_theta, trend_basis, n_layers, layer_size),
            NBEATSBlock(input_size, seasonal_theta, seasonal_basis, n_layers, layer_size),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features) OR (batch, input_size)
        Returns: (batch, 1) probability
        """
        if x.dim() == 3:
            batch, seq, feat = x.shape
            x = x.reshape(batch, -1)
            if x.shape[1] != self.input_size:
                raise ValueError(
                    f"NBEATSPredictor input_size mismatch: got {x.shape[1]} "
                    f"(seq_len={seq} * n_features={feat}), expected {self.input_size}. "
                    f"Set input_size={seq * feat} when constructing the model."
                )

        residuals = x
        forecast = torch.zeros(x.shape[0], self.forecast_steps, device=x.device)

        for stack in self.stacks:
            backcast, block_forecast = stack(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        # Final: sigmoid of forecast mean → probability
        return torch.sigmoid(forecast.mean(dim=-1, keepdim=True))
