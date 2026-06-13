"""
Self-supervised pretraining for sequence models (masked-bar reconstruction).

The supervised signal in trading is weak and noisy: a single binary up/down
label per 60-bar window. Self-supervised pretraining squeezes a far richer
signal out of the *same unlabeled bars* — the model learns market structure by
reconstructing randomly masked timesteps before it ever sees a label. The
pretrained encoder then warm-starts the supervised LSTMPredictor, so the
AutoML desk's cold-start begins from a representation that already understands
the data instead of random weights.

This is masked reconstruction (BERT-style) adapted to continuous features:
randomly zero out `mask_ratio` of timesteps, encode the corrupted sequence with
an LSTM, and reconstruct the original features at the masked positions (MSE).
Pure PyTorch, no extra deps. Lazy torch import so the module loads without it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PretrainResult:
    encoder_state: dict        # state_dict of the LSTM encoder (lstm.* keys)
    final_loss: float
    epochs: int
    n_sequences: int


def pretrain_masked(
    X,
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    epochs: int = 10,
    mask_ratio: float = 0.15,
    lr: float = 1e-3,
    seed: int = 0,
) -> PretrainResult:
    """
    Pretrain an LSTM encoder by masked-bar reconstruction on unlabeled sequences.

    X: (N, T, F) tensor/array of feature sequences (the same shape the supervised
       model consumes). Labels are NOT used.
    Returns the encoder's state_dict (keys prefixed `lstm.`) so it can be
    transferred into an LSTMPredictor, plus the final reconstruction loss.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    Xt = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    Xt = Xt.float()
    if Xt.dim() != 3:
        raise ValueError(f"expected (N, T, F), got shape {tuple(Xt.shape)}")
    n, _T, f = Xt.shape
    if f != n_features:
        raise ValueError(f"n_features={n_features} but X has {f} features")
    if n == 0:
        raise ValueError("no sequences to pretrain on")

    dirs = 2 if bidirectional else 1

    class _MaskedReconstructor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size,
                num_layers=num_layers, dropout=0.1 if num_layers > 1 else 0.0,
                bidirectional=bidirectional, batch_first=True,
            )
            # Reconstruct the original features at every timestep.
            self.decoder = nn.Linear(hidden_size * dirs, n_features)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.decoder(out)

    model = _MaskedReconstructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for _ in range(max(1, epochs)):
        # Random mask: which (sample, timestep) positions to hide this epoch.
        mask = (torch.rand(Xt.shape[:2]) < mask_ratio).unsqueeze(-1)  # (N, T, 1)
        corrupted = Xt.masked_fill(mask, 0.0)

        optimizer.zero_grad()
        recon = model(corrupted)
        # Loss only on masked positions — that's the self-supervised target.
        if mask.any():
            mask_f = mask.expand_as(Xt)
            loss = criterion(recon[mask_f], Xt[mask_f])
        else:
            loss = criterion(recon, Xt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    # Extract just the LSTM encoder weights, stripping the module prefix so the
    # keys match an LSTMPredictor's `lstm.*` parameters.
    encoder_state = {
        f"lstm.{k.split('lstm.', 1)[1]}": v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith("lstm.")
    }
    return PretrainResult(
        encoder_state=encoder_state, final_loss=final_loss,
        epochs=max(1, epochs), n_sequences=n,
    )


def transfer_encoder_weights(encoder_state: dict, predictor) -> int:
    """
    Load pretrained encoder weights into an LSTMPredictor's LSTM layer.

    Only tensors whose key AND shape match are copied (so a config mismatch
    degrades to copying what fits rather than crashing). Returns the count of
    tensors transferred.
    """
    import torch

    target = predictor.state_dict()
    transferred = 0
    with torch.no_grad():
        for key, val in encoder_state.items():
            if key in target and target[key].shape == val.shape:
                target[key].copy_(val)
                transferred += 1
    predictor.load_state_dict(target)
    return transferred
