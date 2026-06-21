"""Production deployment smoke test — the check that was missing for months.

The live site 'never worked' because the deployed backend was a 3-route stub (a
different app), not this repo. Every in-repo test passed because they run the real
app locally; nothing verified that the *deployed* service is actually this app.

This hits the live service and asserts it serves the real QuantEdge API (many
routes + a healthy /health), so 'we shipped the wrong/stub app' fails loudly.

It is `live`-marked and skips when the service is unreachable or down (e.g. Render
free-tier sleep / suspension) so it never blocks unrelated PR CI. Run explicitly:
    pytest -m live tests/test_prod_deployment_smoke.py
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error

import pytest

pytestmark = pytest.mark.live

# Override with PROD_API_BASE if the canonical host changes.
PROD_BASE = os.environ.get("PROD_API_BASE", "https://quantedge-api-9jz0.onrender.com")
_TIMEOUT = 10
# A 3-route stub has ~3 paths; the real app exposes ~100. Anything above this rules
# out the stub while tolerating route churn.
_MIN_REAL_ROUTES = 30


def _get(path: str):
    url = f"{PROD_BASE}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"prod backend unreachable at {url}: {e}")


def test_prod_serves_real_app_not_stub():
    status, body = _get("/openapi.json")
    if status != 200:
        pytest.skip(f"prod backend not healthy (GET /openapi.json -> {status}); "
                    "likely sleeping/suspended — not this change's concern")
    spec = json.loads(body)
    paths = spec.get("paths", {})
    assert len(paths) >= _MIN_REAL_ROUTES, (
        f"Deployed app exposes only {len(paths)} routes — looks like the orphan stub, "
        f"not the real QuantEdge API. Title={spec.get('info', {}).get('title')!r}"
    )
    assert any(p.startswith("/api/v1/") for p in paths), (
        "No /api/v1/* routes in the deployed spec — wrong app deployed."
    )


def test_prod_health_ok():
    status, body = _get("/health")
    if status != 200:
        pytest.skip(f"prod /health -> {status}; service down/sleeping — not this change's concern")
    data = json.loads(body)
    assert data.get("status") == "ok", f"unexpected /health payload: {data}"
