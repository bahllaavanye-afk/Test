"""ML model registry — all model classes importable from here."""
from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.models.ensemble_model import EnsembleModel

# Optional heavy models (torch/sklearn may not be present in all envs)
try:
    from app.ml.models.lstm import LSTMPredictor
except ImportError:
    pass

try:
    from app.ml.models.transformer import TransformerPredictor
except ImportError:
    pass

try:
    from app.ml.models.mamba_trader import MambaTrader
except ImportError:
    pass

try:
    from app.ml.models.itransformer import iTransformerPredictor
except ImportError:
    pass

try:
    from app.ml.models.lorentzian_knn import LorentzianKNN
except ImportError:
    pass
