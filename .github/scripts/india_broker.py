"""Indian broker routing — what each option actually requires to run unattended.

QuantEdge runs on GitHub Actions with no human present. That single constraint
decides which Indian broker can work, and it is not the obvious one.

    broker            cost            unattended login?
    ----------------  --------------  ---------------------------------------
    Zerodha Kite      Rs 2,000/month  NO — interactive login every morning
    Upstox            free tier       NO — same daily OAuth redirect
    AngelOne SmartAPI free            YES — TOTP, derivable from a stored seed
    Dhan              free            YES — long-lived access token
    ICICI Breeze      free            NO — daily session key

**Zerodha Kite Connect cannot be automated as asked.** Its OAuth flow issues a
`request_token` only through a browser redirect that a human must complete, and
the resulting `access_token` expires daily (~06:00 IST). There is no documented
programmatic path; automating it means scripting a login page, which breaks
their terms and breaks silently whenever the page changes. A desk built on Kite
would place orders only on days somebody logged in by hand — the "green-looking
absence" this codebase has been paying down all week, in a new place.

So the recommendation is **AngelOne SmartAPI** (free, TOTP) or **Dhan**
(long-lived token). Both authenticate from secrets alone.

NOTHING HERE PLACES AN ORDER YET, deliberately. This module states the contract
and the credential requirements so the decision is informed; wiring live order
routing to an Indian broker moves real money in a real account and is not a
change to make while the operator is away. `configured_broker()` reports which
one — if any — has credentials present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerSpec:
    name: str
    env_vars: tuple[str, ...]
    unattended: bool
    cost: str
    note: str


# Ordered by suitability for unattended operation, best first.
BROKERS: tuple[BrokerSpec, ...] = (
    BrokerSpec(
        name="angelone",
        env_vars=("ANGELONE_API_KEY", "ANGELONE_CLIENT_ID",
                  "ANGELONE_PASSWORD", "ANGELONE_TOTP_SECRET"),
        unattended=True,
        cost="free",
        note="SmartAPI. TOTP is derived from the stored seed, so login needs no human.",
    ),
    BrokerSpec(
        name="dhan",
        env_vars=("DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"),
        unattended=True,
        cost="free",
        note="Access token is long-lived; no daily refresh.",
    ),
    BrokerSpec(
        name="upstox",
        env_vars=("UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_ACCESS_TOKEN"),
        unattended=False,
        cost="free tier",
        note="Daily OAuth redirect — token must be refreshed by hand each morning.",
    ),
    BrokerSpec(
        name="zerodha",
        env_vars=("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"),
        unattended=False,
        cost="Rs 2,000/month",
        note=("Kite Connect. access_token expires daily (~06:00 IST) and the "
              "request_token comes only from a browser redirect. Cannot run "
              "unattended without scripting the login page."),
    ),
)


def configured_broker() -> BrokerSpec | None:
    """The best-suited broker whose credentials are all present, else None.

    'Best-suited' is BROKERS order, not first-found: if both AngelOne and
    Zerodha were configured, the one that can actually run overnight wins.
    """
    for spec in BROKERS:
        if all(os.environ.get(v, "").strip() for v in spec.env_vars):
            return spec
    return None


def missing_for(name: str) -> list[str]:
    """Which env vars a named broker still needs. Empty = ready."""
    for spec in BROKERS:
        if spec.name == name:
            return [v for v in spec.env_vars if not os.environ.get(v, "").strip()]
    raise KeyError(f"unknown broker {name!r}")


def status_line() -> str:
    """One line for the desk log — says what is routable and what is missing."""
    spec = configured_broker()
    if spec is None:
        best = BROKERS[0]
        return (f"India: NO BROKER CONFIGURED — mutual-fund and equity signals are "
                f"research-only. Cheapest unattended path is {best.name} "
                f"({best.cost}); needs {', '.join(missing_for(best.name))}.")
    warn = "" if spec.unattended else "  ⚠️ requires a DAILY manual login — will stall overnight"
    return f"India: broker={spec.name} ({spec.cost}){warn}"


if __name__ == "__main__":
    print(status_line())
    for s in BROKERS:
        miss = missing_for(s.name)
        print(f"  {s.name:<10} unattended={str(s.unattended):<5} {s.cost:<15} "
              f"{'READY' if not miss else 'missing ' + ','.join(miss)}")
