"""Twice in one day a test passed here and failed CI on a package CI never installed.

    2026-08-05 morning  test_backtest_covers_crypto.py    ModuleNotFoundError: requests
    2026-08-05 evening  test_order_origin_audit.py        async def functions are not
                                                          natively supported (pytest-asyncio)

Both times the cause was the same and it is not "I forgot": the dev container
has a fat Python environment, CI has a deliberately thin one, and **a green
local run carries no information about the CI environment**. Nothing compared
the two, so the only detector was a red PR — after the push, after the wait.

This test is that comparison. It reads the install line out of `test.yml` and
the imports out of the agent test suite, and fails *here* on the mismatch.

Scope is deliberately narrow: module-level imports in `.github/scripts/test_*.py`
and `.github/tests/`, plus pytest plugins inferred from marker usage. Imports
inside functions, inside `try:`, or behind `pytest.importorskip` are optional by
construction and are not required to be installed — `test_fleet_liveness.py`
opens with `pytest.importorskip("yaml")` precisely so it can be skipped.
"""
from __future__ import annotations

import ast
import re
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
TEST_DIRS = (REPO / ".github" / "scripts", REPO / ".github" / "tests")

# Import name → pip package, where they differ.
PKG_FOR_MODULE = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "PIL": "pillow",
}


def _test_files() -> list[Path]:
    files = sorted((REPO / ".github" / "scripts").glob("test_*.py"))
    files += sorted((REPO / ".github" / "tests").rglob("*.py"))
    return [f for f in files if f.name != Path(__file__).name]


def _install_line() -> str:
    """The `uv pip install` line from the test-agents job."""
    text = WORKFLOW.read_text()
    job = text.split("test-agents:", 1)
    assert len(job) == 2, "the test-agents job was renamed — re-point this test"
    m = re.search(r"uv pip install --system ([^\n]+)", job[1])
    assert m, "test-agents no longer has a `uv pip install --system` line"
    return m.group(1)


def _stdlib() -> set[str]:
    """Standard library only. Local modules are resolved separately, by path —
    folding them in here would also silence any third-party package that
    happened to share a filename with something in `.github/scripts`."""
    return set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)


def _local_modules() -> dict[str, Path]:
    """Importable-by-name modules that live alongside the tests."""
    mods: dict[str, Path] = {}
    for d in TEST_DIRS:
        for p in d.rglob("*.py"):
            mods.setdefault(p.stem, p)
    return mods


def _module_scope_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover — a broken file fails elsewhere
        return []
    mods: list[str] = []
    for node in tree.body:                          # module scope ONLY:
        if isinstance(node, ast.Import):            # try/except, def, if → optional
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module.split(".")[0])
    return mods


def _required_modules() -> dict[str, list[str]]:
    """Third-party module → the files that make it required.

    **Follows local imports.** The `requests` break this morning came from
    `test_backtest_covers_crypto.py` importing `quick_backtest_runner`, a local
    module that imports `requests` at module scope — so the test file itself
    named nothing third-party, and collection still ERRORed in CI. A guard that
    only reads the test files would have missed the very case it exists for
    (it did, on the first draft; the mutation test is what caught that).
    """
    out: dict[str, list[str]] = {}
    std = _stdlib()
    local = _local_modules()
    for f in _test_files():
        seen: set[str] = set()
        queue = [(f, f.name)]
        while queue:
            path, blamed = queue.pop()
            if path in seen:
                continue
            seen.add(path)
            for m in _module_scope_imports(path):
                if m == "pytest":
                    continue
                if m in local:
                    # Local: recurse, but keep blaming the test file, since that
                    # is what a reader has to go and change.
                    queue.append((local[m], blamed))
                elif m not in std:
                    out.setdefault(m, []).append(blamed)
    return out


def test_every_module_the_agent_tests_import_is_installed_by_ci():
    """`requests` broke this exact way this morning: imported at module level by
    a test, absent from the install line, ERROR at collection in CI."""
    line = _install_line()
    missing = {}
    for mod, files in _required_modules().items():
        pkg = PKG_FOR_MODULE.get(mod, mod)
        if not re.search(rf"(?<![\w-]){re.escape(pkg)}(?![\w-])", line):
            missing[pkg] = sorted(set(files))
    assert not missing, (
        f"test-agents installs `{line}` but these are imported at module scope: {missing}")


def _run_command() -> str:
    """The `python -m pytest …` invocation in the test-agents job."""
    job = WORKFLOW.read_text().split("test-agents:", 1)[1]
    m = re.search(r"run: (python -m pytest [^\n]+)", job)
    assert m, "test-agents no longer runs pytest directly"
    return m.group(1)


def test_pytest_plugins_the_tests_rely_on_are_installed():
    """A missing plugin does not error at import — it fails at *run* time with
    'async def functions are not natively supported', which is why this one got
    through CI review and not through CI.

    Two independent sources of evidence, because the two real cases differ:
    `pytest-asyncio` is demanded by a marker in a test file, while
    `pytest-timeout` is demanded by `--timeout=30` on the command line and no
    marker anywhere. Checking only markers misses the second entirely.
    """
    line = _install_line()
    needed: dict[str, list[str]] = {}

    for f in _test_files():
        src = f.read_text()
        for marker, pkg in {"asyncio": "pytest-asyncio",
                            "timeout": "pytest-timeout"}.items():
            if re.search(rf"pytest\.mark\.{marker}\b", src):
                needed.setdefault(pkg, []).append(f.name)

    cmd = _run_command()
    for flag, pkg in {"--timeout": "pytest-timeout",
                      "--asyncio-mode": "pytest-asyncio",
                      "-n ": "pytest-xdist",
                      "--cov": "pytest-cov"}.items():
        if flag in cmd:
            needed.setdefault(pkg, []).append(f"the run command ({flag})")

    missing = {pkg: sorted(set(files)) for pkg, files in needed.items()
               if not re.search(rf"(?<![\w-]){re.escape(pkg)}(?![\w-])", line)}
    assert not missing, (
        f"test-agents installs `{line}` but these need a plugin: {missing}")


def test_the_agent_tests_run_without_pytest_asyncio():
    """The chosen resolution, pinned. `asyncio.run()` inside a sync test needs
    nothing installed; re-adding `@pytest.mark.asyncio` without also adding the
    plugin to `test.yml` puts the same red PR back. Either fix is fine — this
    fails only if someone does neither."""
    line = _install_line()
    if re.search(r"(?<![\w-])pytest-asyncio(?![\w-])", line):
        pytest.skip("pytest-asyncio is installed by CI — the marker is safe to use")
    offenders = [f.name for f in _test_files()
                 if re.search(r"pytest\.mark\.asyncio\b", f.read_text())]
    assert not offenders, (
        f"{offenders} use @pytest.mark.asyncio but CI does not install pytest-asyncio — "
        f"use asyncio.run() in a sync test, or add the plugin to test.yml")
