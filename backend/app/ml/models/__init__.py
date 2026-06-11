"""ML model registry — all model classes importable from here."""
from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.models.ensemble_model import EnsembleModel

# Optional models — require torch/sklearn; gracefully absent if deps not installed
try:
    from app.ml.models.lstm import LSTMPredictor
except ImportError:
    LSTMPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.transformer import TransformerPredictor
except ImportError:
    TransformerPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.mamba_trader import MambaTrader
except ImportError:
    MambaTrader = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.itransformer import iTransformerPredictor
except ImportError:
    iTransformerPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.lorentzian_knn import LorentzianKNN
except ImportError:
    LorentzianKNN = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.gemini_signal import GeminiSignalEngine, get_gemini_engine
except ImportError:
    GeminiSignalEngine = None  # type: ignore[assignment,misc]
    get_gemini_engine = None  # type: ignore[assignment]

try:
    from app.ml.models.patch_tst import PatchTST, PatchEncoder
except ImportError:
    PatchTST = None   # type: ignore[assignment,misc]
    PatchEncoder = None  # type: ignore[assignment,misc]
