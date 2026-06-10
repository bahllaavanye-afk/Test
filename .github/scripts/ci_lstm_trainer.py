#!/usr/bin/env python3
"""
QuantEdge CI LSTM Trainer
=========================
Self-contained training script for GitHub Actions (CPU only, no app imports).
Downloads 3 years of SPY + BTC-USD daily data via yfinance,
trains a lightweight BiLSTM, evaluates out-of-sample Sharpe,
saves model artifacts, and posts results to Slack #ml-experiments.

Runtime: ~15-20 min on ubuntu-latest (CPU).
Runs weekly (Sunday 02:00 UTC) via lstm-training.yml.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import yfinance as yf
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install numpy yfinance")

SLACK_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "backend/models_artifacts"))

# Symbols to train on: (ticker, asset_type, start_date, end_date)
SYMBOLS: list[tuple[str, str, str, str]] = [
    ("SPY",     "equity", "2021-01-01", "2024-12-31"),
    ("BTC-USD", "crypto", "2021-01-01", "2024-12-31"),
]

# Model hyper-parameters — kept small for CPU training
SEQ_LEN    = 30
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2
EPOCHS     = 40
BATCH_SIZE = 128
LR         = 1e-3
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
PATIENCE   = 8


# ── Technical feature engineering (pure numpy) ──────────────────────────────

def _pct_change(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    result[period:] = (arr[period:] - arr[:-period]) / (np.abs(arr[:-period]) + 1e-10)
    return result


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def _rolling_stat(arr: np.ndarray, window: int, stat: str) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = arr[i - window + 1 : i + 1]
        out[i] = chunk.std() if stat == "std" else chunk.max() if stat == "max" else chunk.min()
    return out


def compute_features(close: np.ndarray, volume: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Compute ~10 normalised technical features. Returns (n, n_features) array."""
    feats: list[np.ndarray] = []

    # Returns: 1d, 5d, 20d
    feats.append(_pct_change(close, 1))
    feats.append(_pct_change(close, 5))
    feats.append(_pct_change(close, 20))

    # RSI-14
    delta = np.diff(close, prepend=close[0])
    gains  = _ema(np.maximum(delta, 0), 14)
    losses = _ema(np.maximum(-delta, 0), 14)
    rs = np.where(losses < 1e-10, 100.0, gains / (losses + 1e-10))
    feats.append((100 - 100 / (1 + rs)) / 100 - 0.5)  # center around 0

    # MACD histogram (normalised by close)
    ema12   = _ema(close, 12)
    ema26   = _ema(close, 26)
    macd    = ema12 - ema26
    signal  = _ema(macd, 9)
    feats.append(macd / (close + 1e-10))
    feats.append((macd - signal) / (close + 1e-10))

    # Bollinger Band z-score
    ma20  = np.convolve(close, np.ones(20) / 20, mode="full")[: len(close)]
    ma20[:19] = np.nan
    std20 = _rolling_stat(close, 20, "std")
    feats.append((close - ma20) / (std20 + 1e-10))

    # ATR-14 normalised
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _ema(tr, 14)
    feats.append(atr / (close + 1e-10))

    # Volume ratio vs 20-day MA
    vol_ma20 = np.convolve(volume, np.ones(20) / 20, mode="full")[: len(volume)]
    vol_ma20[:19] = np.nan
    feats.append(volume / (vol_ma20 + 1e-10) - 1)

    # 52-week price position
    w = min(252, len(close))
    hi52 = _rolling_stat(close, w, "max")
    lo52 = _rolling_stat(close, w, "min")
    feats.append((close - lo52) / (hi52 - lo52 + 1e-10))

    return np.column_stack(feats)  # (n, 10)


def prepare_data(ticker_sym: str, start: str, end: str):
    """Download data, build features/labels, create train/val/test TensorDatasets."""
    print(f"  Downloading {ticker_sym} from {start} to {end}...")
    df = yf.download(ticker_sym, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty or len(df) < 300:
        raise ValueError(f"Only {len(df)} bars for {ticker_sym} — need ≥ 300")

    # Handle MultiIndex columns from yfinance
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.droplevel(1)

    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    print(f"  {len(close)} bars downloaded")

    X_raw = compute_features(close, volume, high, low)

    # Labels: 1 if next-day return > +0.2%, else 0
    fwd_ret = np.full(len(close), np.nan)
    fwd_ret[:-1] = _pct_change(close, 1)[1:]
    y = (fwd_ret > 0.002).astype(np.float32)

    # Drop NaN rows
    valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y))
    X_raw, y = X_raw[valid], y[valid]

    # Normalise using training-set stats (leak-free: fit on train only)
    n_train_raw = int(len(X_raw) * TRAIN_FRAC)
    mean = X_raw[:n_train_raw].mean(axis=0)
    std  = X_raw[:n_train_raw].std(axis=0) + 1e-8
    X_norm = (X_raw - mean) / std

    # Build sequences (seq_len, n_features)
    Xs, ys = [], []
    for i in range(SEQ_LEN, len(X_norm)):
        Xs.append(X_norm[i - SEQ_LEN : i])
        ys.append(y[i])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    n_train = int(len(Xs) * TRAIN_FRAC)
    n_val   = int(len(Xs) * VAL_FRAC)

    train_ds = TensorDataset(torch.from_numpy(Xs[:n_train]),            torch.from_numpy(ys[:n_train]))
    val_ds   = TensorDataset(torch.from_numpy(Xs[n_train : n_train + n_val]), torch.from_numpy(ys[n_train : n_train + n_val]))
    test_ds  = TensorDataset(torch.from_numpy(Xs[n_train + n_val :]),   torch.from_numpy(ys[n_train + n_val :]))

    n_features = Xs.shape[2]
    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    return train_ds, val_ds, test_ds, n_features, scaler


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, layers: int = N_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.head(out).squeeze(-1)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += ((pred > 0.5).float() == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        pred = model(Xb)
        total_loss += criterion(pred, yb).item() * len(yb)
        correct    += ((pred > 0.5).float() == yb).sum().item()
        n          += len(yb)
    return (total_loss / n if n else 0.0), (correct / n if n else 0.0)


@torch.no_grad()
def compute_oos_sharpe(model: nn.Module, test_ds: TensorDataset) -> float:
    """Estimate out-of-sample annualised Sharpe from test predictions."""
    model.eval()
    X, y = test_ds.tensors
    probs   = model(X).numpy()
    signals = np.where(probs > 0.55, 1.0, np.where(probs < 0.45, -1.0, 0.0))
    # Approximate: label 1 = positive day, 0 = negative; scale to ~1% moves
    daily_ret = signals * (y.numpy() * 2 - 1) * 0.01
    if daily_ret.std() < 1e-10:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))


# ── Slack ─────────────────────────────────────────────────────────────────────

def _post_slack(channel: str, text: str) -> None:
    if not SLACK_TOKEN.startswith("xoxb-"):
        print(f"[slack] #{channel}: {text[:120]}")
        return
    payload = json.dumps({"channel": channel, "text": text,
                          "username": "ML Trainer", "icon_emoji": ":brain:"}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {SLACK_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        print(f"[slack] error posting to #{channel}: {e}")


# ── Per-symbol training ───────────────────────────────────────────────────────

def train_symbol(sym: str, asset_type: str, start: str, end: str) -> dict:
    print(f"\n{'─' * 55}")
    print(f"Training LSTM: {sym} ({asset_type})")
    print(f"{'─' * 55}")

    train_ds, val_ds, test_ds, n_features, scaler = prepare_data(sym, start, end)
    print(f"  Features: {n_features}  |  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = LSTMPredictor(n_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == EPOCHS or no_improve >= PATIENCE:
            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                  f"| val_loss={va_loss:.4f} val_acc={va_acc:.3f} | {elapsed:.0f}s")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)

    _, test_acc = evaluate(model, DataLoader(test_ds, batch_size=BATCH_SIZE), criterion)
    sharpe = compute_oos_sharpe(model, test_ds)

    print(f"\n  Best val acc : {best_val_acc:.3f}")
    print(f"  Test acc     : {test_acc:.3f}")
    print(f"  Est OOS Sharpe: {sharpe:.2f}")
    print(f"  Quality gate : OOS Sharpe ≥ 0.8 → {'✅ PASS' if sharpe >= 0.8 else '⚠️  needs GPU tuning'}")

    # Save artifacts
    exp_name = f"lstm_{sym.lower().replace('-', '_')}_1d"
    save_dir  = ARTIFACTS_DIR / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"model_state_dict": best_state or model.state_dict(),
         "n_features": n_features, "hidden": HIDDEN,
         "n_layers": N_LAYERS, "dropout": DROPOUT, "seq_len": SEQ_LEN},
        save_dir / "model.pt",
    )

    metadata = {
        "symbol": sym, "asset_type": asset_type,
        "train_start": start, "train_end": end,
        "n_features": n_features, "seq_len": SEQ_LEN,
        "val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "oos_sharpe": round(sharpe, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scaler": scaler,
        "quality_gate_passed": sharpe >= 0.8 and test_acc >= 0.55,
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Saved → {save_dir}")
    return metadata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("🧠 QuantEdge CI LSTM Trainer")
    print(f"   Artifacts : {ARTIFACTS_DIR}")
    print(f"   Symbols   : {[s[0] for s in SYMBOLS]}")
    print(f"   Max epochs: {EPOCHS}  Batch: {BATCH_SIZE}  Hidden: {HIDDEN}")
    print(f"   Device    : CPU (use Kaggle/Colab for GPU runs)")
    print()

    results: list[dict] = []
    for sym, asset_type, start, end in SYMBOLS:
        try:
            meta = train_symbol(sym, asset_type, start, end)
            results.append(meta)
        except Exception as e:
            print(f"\n  ERROR training {sym}: {e}")
            results.append({"symbol": sym, "error": str(e)})

    # Post summary to Slack #ml-experiments
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f":brain: *LSTM Training Run Complete* | {now_str}", ""]
    for r in results:
        if "error" in r:
            lines.append(f"• *{r['symbol']}*: :red_circle: `{r['error'][:80]}`")
        else:
            sharpe_ok = r["oos_sharpe"] >= 0.8
            acc_ok    = r["test_accuracy"] >= 0.55
            se = ":green_circle:" if sharpe_ok else ":yellow_circle:" if r["oos_sharpe"] >= 0.4 else ":red_circle:"
            ae = ":green_circle:" if acc_ok else ":yellow_circle:"
            gate = "✅ Quality gate passed" if r.get("quality_gate_passed") else "⚠️  Below quality gate (OOS Sharpe ≥ 0.8 required)"
            lines.append(
                f"• *{r['symbol']}* ({r['asset_type']})\n"
                f"  {se} OOS Sharpe: `{r['oos_sharpe']:.2f}` {ae} Test Acc: `{r['test_accuracy']*100:.1f}%` "
                f"Val Acc: `{r['val_accuracy']*100:.1f}%`\n"
                f"  {gate}"
            )

    lines += [
        "",
        "_Models saved to `backend/models_artifacts/` and uploaded as CI artifacts (30-day retention)._",
        "_Next: download artifact and commit to repo, or run on Kaggle GPU for full-quality training._",
        "_Quality gate: OOS Sharpe ≥ 0.8 AND Test Acc ≥ 55% before deployment to InferenceService._",
    ]
    _post_slack("ml-experiments", "\n".join(lines))

    print(f"\n✅ Done — {sum(1 for r in results if 'error' not in r)}/{len(results)} models trained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
