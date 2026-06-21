"""Guard against the class of deploy bug that made 'nothing work' in production:
frontend code that hardcodes a backend URL or silently falls back to localhost /
the orphan `quantedge-api.onrender.com` stub when a VITE_* env var is missing.

These are *configuration* bugs — the app builds fine and every unit/integration test
passes against localhost, so nothing caught them until the production site 404'd.
This test scans the shipped frontend source so the regression can't recur silently.

Allowed single source of truth for endpoint resolution: frontend/src/utils/endpoints.ts
(its localhost value is guarded behind a runtime isLocalHost check, not a `||` fallback).
"""
from __future__ import annotations
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND = _REPO_ROOT / "frontend"

# The bare host is an orphan 3-route stub; the real service is `quantedge-api-9jz0`.
# Anchor on `//` so we match real URLs (https://, wss://) but not documentation prose.
_ORPHAN_STUB = re.compile(r"//quantedge-api\.onrender\.com")
# A hardcoded fallback to a localhost URL, e.g.  || 'ws://localhost:8000'
_LOCALHOST_FALLBACK = re.compile(r"\|\|\s*['\"][a-z]+://localhost")

# Only this module may name a localhost endpoint (behind an isLocalHost guard).
_ENDPOINTS_HELPER = "src/utils/endpoints.ts"


def _shipped_frontend_files() -> list[Path]:
    if not _FRONTEND.is_dir():
        pytest.skip("frontend/ not present in this checkout")
    files: list[Path] = []
    src = _FRONTEND / "src"
    if src.is_dir():
        files += [p for p in src.rglob("*.ts") if not p.name.endswith(".d.ts")]
        files += list(src.rglob("*.tsx"))
    vercel = _FRONTEND / "vercel.json"
    if vercel.is_file():
        files.append(vercel)
    return files


def test_no_orphan_api_stub_in_frontend():
    """No shipped frontend file may point at the bare orphan onrender host."""
    offenders = [
        str(p.relative_to(_REPO_ROOT))
        for p in _shipped_frontend_files()
        if _ORPHAN_STUB.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, (
        "Frontend references the orphan 'quantedge-api.onrender.com' stub "
        f"(use the real '-6orc' service or a VITE_* var) in: {offenders}"
    )


def test_no_hardcoded_localhost_fallback_in_components():
    """Components must resolve endpoints via utils/endpoints.ts, not `|| 'http://localhost'`."""
    offenders = []
    for p in _shipped_frontend_files():
        if p.as_posix().endswith(_ENDPOINTS_HELPER):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if _LOCALHOST_FALLBACK.search(line):
                offenders.append(f"{p.relative_to(_REPO_ROOT)}:{i}")
    assert not offenders, (
        "Hardcoded localhost fallback ships to production — route through "
        f"frontend/src/utils/endpoints.ts instead. Offenders: {offenders}"
    )
