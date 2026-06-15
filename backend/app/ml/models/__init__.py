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

try:
    from app.ml.models.patch_tst import PatchEncoder, PatchTST
except ImportError as e:
    logger.warning(f"PatchTST unavailable (missing deps): {e}")
    PatchTST = None   # type: ignore[assignment,misc]
    PatchEncoder = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.ssm_model import SSMPredictor, SelectiveSSM
except ImportError as e:
    logger.warning(f"SSMPredictor unavailable (missing deps): {e}")
    SSMPredictor = None  # type: ignore[assignment,misc]
    SelectiveSSM = None  # type: ignore[assignment,misc]

try:
    from app.ml.models.hmm_regime import HMMRegimeModel
except ImportError as e:
    logger.warning(f"HMMRegimeModel unavailable (missing deps): {e}")
    HMMRegimeModel = None  # type: ignore[assignment,misc]
