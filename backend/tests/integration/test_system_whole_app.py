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

# ── Why these two carry their own timeout ────────────────────────────────────
#
# The suite runs with a global `--timeout=60`. These two sweeps walk EVERY GET
# route, and CI supplies real-looking Alpaca credentials
# (ALPACA_API_KEY="test"), so every broker-backed route attempts a genuine
# outbound call to paper-api.alpaca.markets and waits for it to fail auth.
# Their runtime is therefore network-bound, not compute-bound, and varies with
# runner load and egress latency.
#
# Measured 2026-08-06 with CI's exact env: 14.9s + 11.1s locally, against a
# 60s cap — a 4x margin that CI exhausted. Both timed out on the first CI run
# after improver PR #1510, which made it look like that PR broke them.
# It did not: removing the endpoint it added (`GET /signal_quality`) and
# re-timing gives 14.2s + 10.8s, a difference of ~1.1s.
#
# The cap is raised rather than the network removed ON PURPOSE. Exercising
# these routes WITH credentials present is the point — it proves a broker auth
# failure surfaces as a handled error instead of a 5xx, which is the exact
# class the scanner-500 bug fell into. Stubbing the egress would delete the
# coverage.
#
# BUT THE TIMEOUT IS ONLY HALF THE PROBLEM, so do not read this as "fixed".
# The same egress makes the RESULT non-deterministic, not just the duration:
# on one local run with CI's env these two failed in 5.9s while three
# consecutive runs of the identical command passed in 28-32s. A 5xx from the
# upstream host propagates through the route and the sweep reports it, which
# is indistinguishable here from a 5xx we caused. The real fix is to bound the
# egress — point the broker base URL at a local stub that returns a
# deterministic auth failure, keeping the "handled, not 5xx" coverage while
# removing the third party from the verdict. Logged in IMPROVEMENTS 2026-08-06.
TIMEOUT_NETWORK_BOUND_S = 240

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
@pytest.mark.timeout(TIMEOUT_NETWORK_BOUND_S)
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
            # `_is_real_server_error`, NOT a raw `>= 500`. Its twin below has
            # always used the helper; this one did not, so the two guards
            # applied different criteria to the same question.
            #
            # The helper exists because a 502/503 carrying a structured
            # `detail` is a *handled* upstream outage — `{"detail": "Alpaca
            # bars error: 401 ..."}` when the broker rejects our credentials —
            # while a bare 500 is an unhandled exception, which is the bug this
            # sweep was written for. Failing on the former makes the verdict
            # depend on whether a third party answered, which is exactly the
            # non-determinism measured 2026-08-06: the identical command failed
            # both sweeps in 5.9s once and passed three consecutive times.
            if _is_real_server_error(r):
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
@pytest.mark.timeout(TIMEOUT_NETWORK_BOUND_S)
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


def test_both_5xx_sweeps_use_the_same_criterion():
    """They ask the same question and must answer it the same way.

    Until 2026-08-06 the parameterless sweep used a raw `>= 500` while its
    parameterised twin used `_is_real_server_error`. The helper exists because
    a 502/503 carrying a structured `detail` is a HANDLED upstream outage —
    `{"detail": "Alpaca bars error: 401 ..."}` when the broker rejects our
    credentials — whereas a bare 500 is the unhandled exception this sweep was
    written to catch.

    The consequence was measured: with CI's env the identical command failed
    both sweeps in 5.9s on one run and passed three consecutive times, because
    the verdict depended on whether Alpaca happened to answer. A guard whose
    result a third party decides is not a guard.
    """
    from pathlib import Path
    import re

    src = Path(__file__).read_text()
    for name in ("test_no_get_endpoint_returns_5xx",
                 "test_no_parameterised_get_endpoint_returns_5xx"):
        body = src.split(f"async def {name}(", 1)[1].split("\nasync def ", 1)[0]
        assert "_is_real_server_error(r)" in body, (
            f"{name} does not use the shared criterion")
        assert not re.search(r"if r\.status_code >= 500", body), (
            f"{name} still uses a raw status check alongside the helper")


def test_a_handled_upstream_outage_is_not_counted_as_our_bug():
    """The helper's contract, pinned separately from its callers: a 502/503
    WITH `detail` is deliberate, the same code WITHOUT it means something
    escaped, and 500 is always ours."""
    class _R:
        def __init__(self, code, payload):
            self.status_code, self._p = code, payload
        def json(self):
            if self._p is None:
                raise ValueError("not json")
            return self._p

    assert _is_real_server_error(_R(502, {"detail": "Alpaca bars error: 401"})) is False
    assert _is_real_server_error(_R(503, {"detail": "Redis unavailable"})) is False
    assert _is_real_server_error(_R(502, {"oops": 1})) is True
    assert _is_real_server_error(_R(502, None)) is True
    assert _is_real_server_error(_R(500, {"detail": "anything"})) is True
    assert _is_real_server_error(_R(404, {"detail": "nope"})) is False
