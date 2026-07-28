"""Nothing should quietly become unreachable from the running app.

Coverage measurement (#1113) reported 37 modules at 0%. But 0% coverage does
not mean dead — a module can be exercised only in production. The question
that separates the two is whether the import graph reaches it from a real
entrypoint.

Walking that graph turned up something one-hop analysis misses: a cluster of
nine modules in `app/tasks/` that import *each other* and are reached by
nothing else. `free_llm_router` looked "imported" (by ai_strategy_generator,
research_pipeline, self_improving_loop) — but all three of those are
themselves unreachable. The whole subsystem is orphaned:

    agent_bus              agent_memory           ai_strategy_generator
    free_llm_router        knowledge_loop         research_pipeline
    self_improving_loop    strategy_auction       task_queue

Six of the nine have ZERO references anywhere outside their own file. The only
dynamic-import machinery in `app/` is the strategy registry and a torch
feature-probe, neither of which touches these.

DELIBERATELY NOT WIRED UP. Unlike the risk gate, the exit path or the ML
features — all of which were *supposed* to be running and were switched on —
starting these means the backend begins generating strategies and modifying
itself autonomously. That is a policy decision, not a wiring bug, so it is
reported rather than enacted.

Note the GitHub Actions agent fleet under `.github/scripts/` is a SEPARATE
system and does run; this finding is about the backend only.

This test exists so the orphaned set cannot silently grow.
"""
from __future__ import annotations

import ast
import collections
import pathlib

APP = pathlib.Path(__file__).resolve().parents[2] / "app"

# The real production entrypoints. `static_server` is what Render imports; it
# wraps `main`. `api/v1/router` pulls in every endpoint module; `tasks/scheduler`
# is started from the lifespan.
ENTRYPOINTS = ["main", "static_server", "api/v1/router", "tasks/scheduler"]

# The orphaned agent subsystem, recorded so it stays visible. Shrinking this
# (by wiring a module up or deleting it) is good and the test will say so.
KNOWN_ORPHANED_TASKS = {
    "tasks/agent_bus",
    "tasks/agent_memory",
    "tasks/ai_strategy_generator",
    "tasks/free_llm_router",
    "tasks/knowledge_loop",
    "tasks/research_pipeline",
    "tasks/self_improving_loop",
    "tasks/strategy_auction",
    "tasks/task_queue",
}

# Ceiling on total unreachable modules, from the 2026-07-28 measurement (112).
# A little headroom for legitimately standalone scripts; a real regression
# lands well above it.
MAX_UNREACHABLE = 120


def _module_name(path: pathlib.Path) -> str:
    return str(path.relative_to(APP).with_suffix("")).replace("/__init__", "")


def _import_graph() -> tuple[dict[str, set[str]], set[str]]:
    edges: dict[str, set[str]] = collections.defaultdict(set)
    modules: set[str] = set()
    for path in APP.rglob("*.py"):
        name = _module_name(path)
        modules.add(name)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
                targets.append(node.module[4:])
            elif isinstance(node, ast.Import):
                targets += [a.name[4:] for a in node.names if a.name.startswith("app.")]
            for target in targets:
                edges[name].add(target.replace(".", "/"))
    return edges, modules


def _reachable() -> set[str]:
    edges, modules = _import_graph()
    seen: set[str] = set()
    stack = list(ENTRYPOINTS)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for nxt in edges.get(current, ()):
            candidate = nxt.replace("/__init__", "")
            if candidate in modules and candidate not in seen:
                stack.append(candidate)
    return seen & modules


def _unreachable() -> set[str]:
    _, modules = _import_graph()
    return modules - _reachable()


def test_the_entrypoints_are_real():
    """A typo here would make everything look reachable — or nothing."""
    _, modules = _import_graph()
    missing = [e for e in ENTRYPOINTS if e not in modules]
    assert not missing, f"ENTRYPOINTS names no real module: {missing}"


def test_the_core_of_the_app_is_reachable():
    """Sanity floor: if this drops, the graph walk itself is broken."""
    reachable = _reachable()
    for core in ("risk/manager", "tasks/price_feed", "brokers/alpaca", "bots/engine"):
        assert core in reachable, f"{core} unreachable — the analysis is wrong, not the app"
    assert len(reachable) > 150, f"only {len(reachable)} modules reachable — graph walk broken"


def test_the_orphaned_agent_subsystem_has_not_grown():
    """New orphans in tasks/ must be deliberate, not accidental."""
    orphaned_tasks = {m for m in _unreachable() if m.startswith("tasks/")}
    new = orphaned_tasks - KNOWN_ORPHANED_TASKS
    assert not new, (
        f"new unreachable task modules: {sorted(new)} — either wire them into "
        f"the app or add them to KNOWN_ORPHANED_TASKS with a reason"
    )


def test_known_orphans_that_became_reachable_are_removed_from_the_list():
    """Keeps the list honest in the shrinking direction too."""
    reachable = _reachable()
    now_live = KNOWN_ORPHANED_TASKS & reachable
    assert not now_live, (
        f"these are wired up now — drop them from KNOWN_ORPHANED_TASKS: {sorted(now_live)}"
    )


def test_total_unreachable_modules_stay_bounded():
    unreachable = _unreachable()
    assert len(unreachable) <= MAX_UNREACHABLE, (
        f"{len(unreachable)} unreachable modules exceeds the {MAX_UNREACHABLE} "
        f"ceiling — dead weight is accumulating. Newest: "
        f"{sorted(unreachable)[:10]}"
    )
