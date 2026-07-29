"""The CI trainer and the backend's inference model are DIFFERENT networks.

Last tick I recorded three reasons a trained LSTM never reaches inference —
wrong filename, wrong checkpoint schema, never committed — and proposed the
obvious fix: have `ci_lstm_trainer.py` also emit `models_artifacts/lstm_latest.pt`
in `AbstractModel`'s schema. **That fix cannot work**, and this file exists so
nobody ships it.

There are two classes named `LSTMPredictor` and they are not the same network:

    .github/scripts/ci_lstm_trainer.py     backend/app/ml/models/lstm.py
    ─────────────────────────────────      ────────────────────────────────
    self.lstm                              self.lstm
    (no attention)                         self.attention  <- SelfAttention
    self.norm                              self.norm
    self.head:                             self.head:
      Linear(hidden*2, 32)                   Linear(hidden*2, 64)
      GELU, Dropout                          GELU, Dropout
      Linear(32, 1)                          Linear(64, 1)
      Sigmoid                                (logits, no sigmoid)

    __init__(n_features, hidden,           __init__(n_features, hidden_size,
             layers, dropout)                       num_layers, dropout,
                                                    bidirectional)

`load_state_dict` would fail twice over: missing `attention.*` keys, and a
shape mismatch on `head.0.weight` (32×256 vs 64×256). Writing the checkpoint
in the right *wrapper* schema does not help when the tensors inside describe a
different architecture. The two also disagree on output semantics — the trainer
ends in `Sigmoid` and returns probabilities, the backend returns logits — so
even a forced load would silently mis-scale every prediction.

So promotion needs the architectures unified first. The right direction is for
the trainer to import and train `app.ml.models.lstm.LSTMPredictor` rather than
maintain a second definition of the same thing — a duplicate implementation
that reads exactly like the real one is the failure mode this repo keeps
paying for.

The invariant below is conditional on purpose, so it is GREEN today and fires
the moment someone wires promotion without doing the unification:

    if the trainer writes into a path InferenceService reads,
    then the two architectures must match.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TRAINER = _REPO / ".github" / "scripts" / "ci_lstm_trainer.py"
_BACKEND = _REPO / "backend" / "app" / "ml" / "models" / "lstm.py"
_INFERENCE = _REPO / "backend" / "app" / "ml" / "inference.py"


def _class_init(path: Path, name: str):
    tree = ast.parse(path.read_text())
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )
    assert cls is not None, f"class {name} not found in {path}"
    init = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    assert init is not None, f"{name}.__init__ not found in {path}"
    return init


def _submodules(path: Path, name: str) -> dict[str, str]:
    """`self.X = SomeModule(...)` assignments in __init__ — the state_dict prefixes.

    Only CALL-valued assignments count. `self.hidden_size = hidden_size` stores an
    int and contributes no tensors; `self.attention = SelfAttention(...)` does.

    Filtering on an `nn.` prefix instead — my first attempt — silently dropped
    `self.attention` (it is a locally-defined `nn.Module` subclass, not `nn.X`),
    which made the two architectures compare EQUAL and tripped the tripwire below
    on a false positive. Left in the record because it is the same shape of bug
    as everything else in this file: a filter that looked right and quietly
    excluded the one item that mattered.
    """
    out: dict[str, str] = {}
    for node in ast.walk(_class_init(path, name)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    out[target.attr] = ast.unparse(node.value)
    return out


def _inference_paths() -> set[str]:
    """Filenames InferenceService.load_models() reads from models_dir."""
    src = _INFERENCE.read_text()
    return {
        node.value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith((".pt", ".ubj", ".pkl"))
    }


def _trainer_written_names() -> set[str]:
    """String literals the trainer saves to."""
    return {
        node.value
        for node in ast.walk(ast.parse(_TRAINER.read_text()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith((".pt", ".ubj", ".pkl"))
    }


def test_promotion_requires_matching_architectures():
    """The conditional invariant. Green today; fires on a naive promotion."""
    promoted = _trainer_written_names() & _inference_paths()
    if not promoted:
        pytest.skip(
            "trainer does not yet write a path InferenceService reads — "
            "nothing to promote, so nothing to be incompatible"
        )

    trainer = _submodules(_TRAINER, "LSTMPredictor")
    backend = _submodules(_BACKEND, "LSTMPredictor")
    # Plain attributes (ints, bools) are not state_dict entries; submodules are.
    trainer_mods, backend_mods = set(trainer), set(backend)

    missing = backend_mods - trainer_mods
    assert not missing, (
        f"{sorted(promoted)} is now written by the trainer, but the trainer's "
        f"network is missing {sorted(missing)} that the backend's LSTMPredictor "
        f"defines. load_state_dict() will fail on those keys. Unify the "
        f"architectures — have the trainer import "
        f"app.ml.models.lstm.LSTMPredictor — before promoting."
    )
    for mod in sorted(trainer_mods & backend_mods):
        assert trainer[mod] == backend[mod], (
            f"submodule `self.{mod}` differs between the trainer and the "
            f"backend model, so the saved tensors will not fit:\n"
            f"  trainer: {trainer[mod]}\n  backend: {backend[mod]}"
        )


def test_the_divergence_is_still_real_and_still_documented():
    """If the architectures get unified, this file's premise changes.

    Not a assertion that they *must* differ — a tripwire so the docstring above,
    IMPROVEMENTS.md and CONTINUITY.md get corrected in the same change rather
    than silently rotting into another wrong instruction.
    """
    trainer = set(_submodules(_TRAINER, "LSTMPredictor"))
    backend = set(_submodules(_BACKEND, "LSTMPredictor"))
    if trainer == backend:
        pytest.fail(
            "The two LSTMPredictor definitions now have the same submodules. "
            "That is good news — but this test file, IMPROVEMENTS.md and "
            "CONTINUITY.md all state that they diverge. Update them, then "
            "delete this tripwire."
        )
    assert "attention" in backend and "attention" not in trainer, (
        "the divergence changed shape; re-derive it before trusting the notes"
    )


def test_inference_reads_flat_latest_names_not_experiment_subdirs():
    """Pins the filename half of the mismatch, which is easy to 'fix' wrongly."""
    paths = _inference_paths()
    assert "lstm_latest.pt" in paths, (
        "InferenceService no longer reads lstm_latest.pt — the promotion target "
        "moved and the notes describing it are now wrong"
    )
    written = _trainer_written_names()
    assert "model.pt" in written, (
        "the trainer no longer writes model.pt; re-check what it produces"
    )
