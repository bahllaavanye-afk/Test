"""Hard-budget fail-soft guard for strategies whose analyze() does blocking
network I/O (yfinance) that a plain exception guard cannot time-bound.

The quarantine class this fixes (contract-test audit 2026-07-11): the strategy
IS guarded — offline it eventually returns None — but yfinance's fetch runs
synchronously (and via curl_cffi, which bypasses Python's socket module, so
socket-level network kills don't even reach it). Neither the desk's
asyncio.wait_for nor the contract test's 5s budget can preempt it.

apply_hard_budget() replaces cls.analyze with a version that runs the original
in a detached daemon thread (its own event loop) and abandons it after
STRATEGY_ANALYZE_BUDGET_S seconds (default 3.5, env-tunable — desks give live
universe scans more room). A daemon thread — NOT asyncio.to_thread — because
asyncio.run() joins the default executor on exit, so an abandoned to_thread
worker would hang the caller's loop shutdown for as long as the fetch retries.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time


def apply_hard_budget(cls, name: str) -> None:
    unguarded = cls.analyze

    async def _budgeted_analyze(self, data, symbol: str = "SPY"):
        budget = float(os.getenv("STRATEGY_ANALYZE_BUDGET_S", "3.5"))
        result: dict = {}
        done = threading.Event()

        def _run():
            try:
                result["v"] = asyncio.run(unguarded(self, data, symbol))
            except Exception as exc:  # noqa: BLE001 — handed to the caller below
                result["v"] = exc
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True, name=f"failsoft-{name}").start()
        deadline = time.monotonic() + budget
        while not done.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        if not done.is_set():
            print(f"{name}: analyze fail-soft -> None (budget {budget:.1f}s exceeded; fetch abandoned)")
            return None
        out = result.get("v")
        if isinstance(out, Exception):
            print(f"{name}: analyze fail-soft -> None ({type(out).__name__}: {str(out)[:80]})")
            return None
        return out

    cls.analyze = _budgeted_analyze
