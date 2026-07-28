"""Whole-app system test — walk EVERY GET endpoint, assert none 5xx.

The strongest cheap guarantee against 'the website is broken': instead of
hand-picking endpoints (which is how the tearsheet 500 survived), derive the
list from the app's own route table so new endpoints are covered the day
they're added. Auth'd with a real registered user.

PARAMETERISED ROUTES ARE COVERED TOO, as of 2026-07-28. The original docstring
said "path-param routes need fixtures and have their own tests" — but 25 of the
128 GET routes take a path parameter, and several had **no** test at all. That
exclusion is exactly how `/api/v1/scanners/{desk}` shipped returning **500 on
every non-empty result** for all three desks: the producer emitted a 0-100
score and long/short sides against a schema requiring 0-1 and {buy,sell,
neutral,none}. The parameterless twin `/scanners/` passed only because it reads
an empty cache in tests.

Placeholder values are deliberately non-existent ids. The assertion is not
"this returns data" — it is "this does not 5xx". 404/422 on a bogus id is the
correct answer; an unhandled exception is a bug. That distinction is what makes
covering these cheap: no fixtures required.

One value per parameter is NOT enough, and the first draft of this proved it:
with `desk="equity"` the walk passed against the unfixed scanner, because
equity returned an empty result set here and never reached the serialisation
path. Only "polymarket" produced a row. Placeholders are therefore LISTS, and
enum-like params are expanded over every value. Re-verified against the
pre-fix tree, where the walk now reports:

    /api/v1/scanners/{desk} (as /api/v1/scanners/polymarket) → 500:
      2 validation errors for ScanResultOut ...
"""
from __future__ import annotations

import uuid

import pytest

_PASSWORD = "Syst3m!2026xx"

# Endpoints that legitimately need query params or external services and have
# dedicated tests elsewhere. Keep this list SHORT and justified.
_SKIP_PATHS = {
    "/api/v1/auth/google",          # redirects to Google (302 tested elsewhere)
    "/api/v1/auth/google/callback",  # needs OAuth code
}


async def _auth_headers(client) -> dict[str, str]:
    email = f"system_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    if r.status_code == 429:
        # The full CI suite registers many users within a minute and trips the
        # 10/min auth limiter; the demo endpoint is a separate bucket and its
        # user works fine for walking GETs. This is a limiter artifact, not a
        # backend failure — never let it turn the gate red.
        r = await client.post("/api/v1/auth/demo")
        if r.status_code == 429:
            pytest.skip("auth rate-limited in this CI window")
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _parameterless_get_paths() -> list[str]:
    # Derive from the OpenAPI schema, not app.routes: the v1 router is included
    # via a lazy wrapper (_IncludedRouter) that hides APIRoutes from a naive
    # route walk. The schema is the app's own contract — anything served is here.
    from app.main import app

    schema = app.openapi()
    paths = []
    for path, ops in schema.get("paths", {}).items():
        if "get" not in ops or "{" in path or path in _SKIP_PATHS:
            continue
        paths.append(path)
    return sorted(set(paths))


@pytest.mark.asyncio
async def test_no_get_endpoint_returns_5xx(client):
    """Every parameterless GET must respond without a server error.

    4xx is acceptable (auth variants, missing optional services, validation);
    5xx is always a bug. This is the in-repo twin of the live smoke test.
    """
    headers = await _auth_headers(client)
    paths = _parameterless_get_paths()
    assert len(paths) > 60, f"route walk looks broken — only {len(paths)} GET paths found"

    failures: list[str] = []
    for path in paths:
        try:
            r = await client.get(path, headers=headers)
            if r.status_code >= 500:
                failures.append(f"{path} → {r.status_code}: {r.text[:120]}")
        except Exception as exc:  # noqa: BLE001 — an unhandled exception IS a 5xx
            failures.append(f"{path} → raised {type(exc).__name__}: {exc}")

    assert not failures, "Endpoints with server errors:\n" + "\n".join(failures)


def _is_real_server_error(response) -> bool:
    """Distinguish "we crashed" from "the upstream is down and we said so".

    500 is always a bug: FastAPI returns it for an unhandled exception, which
    is what the scanner ValidationError produced.

    502 / 503 are NOT, when they carry a structured `detail` — those are
    deliberate `HTTPException`s for an absent external dependency, and in this
    environment several are expected and correct:

        502 {"detail": "Alpaca bars error: 401 ..."}   no broker credentials
        503 {"detail": "Redis unavailable — ..."}      no Redis

    Failing on those would make the guard environment-dependent, and a guard
    that cries wolf gets deleted. A 502/503 with no `detail` still fails: that
    shape means something escaped rather than being handled.
    """
    if response.status_code < 500:
        return False
    if response.status_code in (502, 503):
        try:
            return "detail" not in response.json()
        except Exception:  # noqa: BLE001 — unparseable body is not a handled error
            return True
    return True


# Placeholders for path params. Real-looking where the route parses the value
# (symbols, uuids), deliberately non-existent everywhere — the assertion is
# "does not 5xx", so a 404 is a pass and no fixtures are needed.
_PATH_PARAM_VALUES = {
    # LISTS, not single values. One value per parameter is not enough: the
    # first draft of this guard used desk="equity" and MISSED the very bug it
    # was written for — /scanners/{desk} only 500'd on "polymarket", because
    # equity returned an empty result set in this environment and never
    # exercised the serialisation path. Enum-like params get every value.
    "desk": ["equity", "crypto", "polymarket"],
    "symbol": ["SPY", "BTC/USD"],
    "underlying": ["SPY"],
    "category": ["momentum"],
    "model_name": ["lstm"],
    "bot_id": ["00000000-0000-0000-0000-000000000000"],
    "account_id": ["00000000-0000-0000-0000-000000000000"],
    "order_id": ["00000000-0000-0000-0000-000000000000"],
    "release_id": ["00000000-0000-0000-0000-000000000000"],
    "run_id": ["00000000-0000-0000-0000-000000000000"],
    "experiment_id": ["00000000-0000-0000-0000-000000000000"],
    "model_id": ["00000000-0000-0000-0000-000000000000"],
}


def _parameterised_get_paths() -> list[tuple[str, str]]:
    """(template, concrete-path) for every GET route taking path params.

    Expands the cartesian product of the placeholder lists, so an enum-like
    param such as {desk} is exercised for every value rather than just one.
    """
    import itertools
    import re

    from app.main import app

    out: list[tuple[str, str]] = []
    for path, ops in app.openapi().get("paths", {}).items():
        if "get" not in ops or "{" not in path or path in _SKIP_PATHS:
            continue
        names = re.findall(r"{(\w+)}", path)
        if any(n not in _PATH_PARAM_VALUES for n in names):
            continue  # unknown placeholder — surfaced by the test below
        for combo in itertools.product(*(_PATH_PARAM_VALUES[n] for n in names)):
            concrete = path
            for n, v in zip(names, combo):
                concrete = concrete.replace(f"{{{n}}}", v)
            out.append((path, concrete))
    return sorted(set(out))


def test_every_path_parameter_has_a_placeholder():
    """A new param name must be added here, not silently skipped.

    Without this, adding `/{portfolio_id}` would quietly drop that route out of
    the 5xx walk — which is precisely how the scanner 500 survived.
    """
    import re

    from app.main import app

    unknown = set()
    for path, ops in app.openapi().get("paths", {}).items():
        if "get" not in ops or "{" not in path or path in _SKIP_PATHS:
            continue
        unknown |= {n for n in re.findall(r"{(\w+)}", path) if n not in _PATH_PARAM_VALUES}
    assert not unknown, (
        f"path params with no placeholder, so their routes are untested: "
        f"{sorted(unknown)} — add them to _PATH_PARAM_VALUES"
    )


@pytest.mark.asyncio
async def test_no_parameterised_get_endpoint_returns_5xx(client):
    """The 25 routes the parameterless walk could never reach.

    Caught here first: /api/v1/scanners/{desk} 500'd on every non-empty result.
    """
    headers = await _auth_headers(client)
    routes = _parameterised_get_paths()
    assert len(routes) >= 20, f"route walk looks broken — only {len(routes)} found"

    failures: list[str] = []
    for template, concrete in routes:
        try:
            r = await client.get(concrete, headers=headers)
            if _is_real_server_error(r):
                failures.append(f"{template} (as {concrete}) → {r.status_code}: {r.text[:160]}")
        except Exception as exc:  # noqa: BLE001 — an unhandled exception IS a 5xx
            failures.append(f"{template} (as {concrete}) → raised {type(exc).__name__}: {exc}")

    assert not failures, "Parameterised endpoints with server errors:\n" + "\n".join(failures)
