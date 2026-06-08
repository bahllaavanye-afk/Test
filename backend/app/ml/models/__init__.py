"""ML model registry — all model classes importable from here."""
import logging
from app.ml.models.base_model import AbstractModel, EvalMetrics
from app.ml.models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)

# Optional heavy models (torch/sklearn may not be present in all envs)
try:
    from app.ml.models.lstm import LSTMPredictor
except ImportError as e:
    logger.warning(f"LSTM model unavailable (missing deps): {e}")
    LSTMPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.transformer import TransformerPredictor
except ImportError as e:
    logger.warning(f"Transformer model unavailable (missing deps): {e}")
    TransformerPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.mamba_trader import MambaTrader
except ImportError as e:
    logger.warning(f"MambaTrader model unavailable (missing deps): {e}")
    MambaTrader = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.itransformer import iTransformerPredictor
except ImportError as e:
    logger.warning(f"iTransformer model unavailable (missing deps): {e}")
    iTransformerPredictor = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.lorentzian_knn import LorentzianKNN
except ImportError as e:
    logger.warning(f"LorentzianKNN model unavailable (missing deps): {e}")
    LorentzianKNN = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.gemini_signal import GeminiSignalEngine, get_gemini_engine
except ImportError as e:
    logger.warning(f"GeminiSignalEngine unavailable (missing deps): {e}")
    GeminiSignalEngine = None  # type: ignore[assignment,misc]
    get_gemini_engine = None  # type: ignore[assignment]
