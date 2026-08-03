"""`/health/detailed`'s `ml_models` must not call a trained artifact a loaded model.

IMPROVEMENTS.md carried this as a one-line P2: *"`/health/detailed` `ml_models`
count should `rglob`. Otherwise the metric stays 0 even once models land."* The
premise is right — `ci_lstm_trainer.py` saves to
`ARTIFACTS_DIR/lstm_<symbol>_1d/model.pt`, a subdirectory, while the old check
globbed only the top level, so six successful training runs were invisible.

Applying that fix literally would have been worse than the bug. `ok` was
`len(files) > 0`, so switching to `rglob` alone flips `ok: false -> true` the
moment the weekly trainer writes anything, and `/health/detailed` starts
reporting "models loaded" while `app/ml/inference.py` still loads nothing. That
is the green-looking absence this repo keeps paying for — a check that reports
success for work that did not happen.

Inference reads four exact top-level filenames and nothing else. An artifact
sitting in a per-experiment subdirectory is unreachable to it at any depth, and
the gap is not merely a path problem: the CI trainer and `app.ml.models.lstm`
define two different networks under the same name (the backend has a
`SelfAttention` block the trainer lacks, a 64-wide head against the trainer's 32,
logits vs sigmoid), so even a correctly-named artifact would fail
`load_state_dict()`.

So the check reports two numbers — `artifacts_on_disk` (what training produced)
and `count`/`ok` (what inference can load) — and these tests pin the invariant
that the first can never imply the second.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.main import INFERENCE_MODEL_FILES, ml_models_check


def test_nothing_on_disk_reports_not_ok(tmp_path: Path):
    result = ml_models_check(tmp_path)
    assert result["ok"] is False
    assert result["count"] == 0
    assert result["artifacts_on_disk"] == 0
    assert "train models" in result["note"]


def test_a_missing_directory_does_not_raise(tmp_path: Path):
    """Render starts with no models dir at all; the health check must survive it."""
    result = ml_models_check(tmp_path / "does_not_exist")
    assert result["ok"] is False
    assert result["artifacts_on_disk"] == 0


def test_nested_training_artifacts_are_counted(tmp_path: Path):
    """The rglob half of the fix: the trainer's real output path is a subdir.

    `ci_lstm_trainer.py`: `save_dir = ARTIFACTS_DIR / exp_name` then
    `torch.save(..., save_dir / "model.pt")`. A top-level-only glob reports 0
    forever no matter how many models train successfully.
    """
    exp = tmp_path / "lstm_btc_usd_1d"
    exp.mkdir()
    (exp / "model.pt").write_bytes(b"weights")
    assert ml_models_check(tmp_path)["artifacts_on_disk"] == 1


def test_a_nested_artifact_does_not_make_the_check_ok(tmp_path: Path):
    """THE regression this file exists for.

    A trained artifact in a subdirectory is real work and should be visible —
    but it is not loadable, and the health check must not claim otherwise. If
    this ever passes with ok=True, `/health/detailed` has started lying about
    ML readiness.
    """
    exp = tmp_path / "lstm_btc_usd_1d"
    exp.mkdir()
    (exp / "model.pt").write_bytes(b"weights")

    result = ml_models_check(tmp_path)
    assert result["artifacts_on_disk"] == 1, "the artifact should be seen"
    assert result["ok"] is False, (
        "a trained artifact in a subdirectory is NOT loadable by inference.py, "
        "which opens only top-level " + ", ".join(INFERENCE_MODEL_FILES) + ". "
        "Reporting ok=True here would announce ML readiness that does not exist."
    )
    assert result["count"] == 0
    assert "NONE loadable" in result["note"]


def test_promotion_to_a_recognised_name_flips_it_ok(tmp_path: Path):
    """The check must still be capable of going green, or it is just a red light."""
    (tmp_path / "lstm_latest.pt").write_bytes(b"weights")
    result = ml_models_check(tmp_path)
    assert result["ok"] is True
    assert result["count"] == 1
    assert "loadable" in result["note"]


@pytest.mark.parametrize("filename", INFERENCE_MODEL_FILES)
def test_each_inference_filename_is_recognised(filename: str, tmp_path: Path):
    """Guards the name list against drifting from inference.py."""
    (tmp_path / filename).write_bytes(b"x")
    result = ml_models_check(tmp_path)
    assert result["ok"] is True
    assert filename in result["note"]


def test_the_filename_list_matches_inference_py():
    """If inference.py changes which files it opens, this check goes stale silently.

    A health check keyed to filenames nothing reads is another decoy, so read the
    source and compare rather than trusting the constant.
    """
    src = (Path(__file__).resolve().parents[2] / "app" / "ml" / "inference.py").read_text()
    for name in INFERENCE_MODEL_FILES:
        assert f'"{name}"' in src, (
            f"{name} is in INFERENCE_MODEL_FILES but app/ml/inference.py no longer "
            f"opens it — the health check is reporting on a file nothing loads."
        )
