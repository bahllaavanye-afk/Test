"""Seven scripts computed a REPO_ROOT that pointed at `.github/`.

`.github/scripts/x.py` needs `parents[2]` to reach the repo root. Seven scripts
used `Path(__file__).parent.parent`, which stops at `.github/` — so every path
built from it resolved under a directory that does not exist:

    REPO_ROOT / "backend"     -> .github/backend      (absent)
    REPO_ROOT / "frontend"    -> .github/frontend     (absent)
    REPO_ROOT / "docs"        -> .github/docs         (absent)
    REPO_ROOT / "experiments" -> .github/experiments  (absent)
    REPO_ROOT / ".github"     -> .github/.github      (absent)

It surfaced as the TV Indicator workflow aborting outright:

    [tv-improve] /home/runner/work/Test/Test/.github/backend/app/strategies/
                 manual/tv_indicators.py not found — aborting

`render_auto_fix.py` was the worst of the seven: it writes patched files to
`REPO_ROOT / rel` and runs `git add -A` with `cwd=REPO_ROOT`, so the Render
auto-fixer was operating inside `.github/` rather than the repository.

A missing directory reads as "nothing to do" rather than an error, which is why
six of the seven never announced themselves.

This guard EVALUATES each expression and compares the result to the real repo
root, rather than counting `.parent`s. That matters: `parents[2]` and
`.parent.parent.parent` are the same directory but index differently, so any
count-based rule gets one of them wrong — an earlier draft of this file did
exactly that and reported 29 correct scripts as broken.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[1]

# Names that are meant to be the repository root.
_ROOT_NAMES = {"REPO_ROOT", "ROOT", "REPO"}

# Matches both spellings. `.parent(?!s)` is required: a naive `(?:\.parent)+`
# also matches the `.parent` PREFIX inside `.parents[2]`, which made an early
# version of this check report 29 correct scripts as broken.
_ASSIGN = re.compile(
    r"(\w+)\s*=\s*(Path\(__file__\)(?:\.resolve\(\))?"
    r"(?:(?:\.parent(?!s))+|\.parents\[\d+\]))"
)


def _root_expressions(path: Path) -> list[tuple[str, str]]:
    """(variable, expression) for each root-named Path(__file__) assignment."""
    return [
        (m.group(1), m.group(2))
        for m in _ASSIGN.finditer(path.read_text(errors="ignore"))
        if m.group(1).upper() in _ROOT_NAMES
    ]


def _script_files() -> list[Path]:
    return sorted(p for p in _SCRIPTS.glob("*.py") if not p.name.startswith("test_"))


def test_the_scripts_directory_is_two_levels_below_the_repo_root():
    """The premise the check rests on."""
    assert _SCRIPTS.name == "scripts" and _SCRIPTS.parent.name == ".github"
    assert (_REPO / "backend").is_dir(), _REPO


@pytest.mark.parametrize("script", _script_files(), ids=lambda p: p.name)
def test_repo_root_resolves_to_the_repository_root(script: Path):
    """EVALUATE the expression rather than counting syntax.

    `parents[2]` and `.parent.parent.parent` are the same directory but index
    differently, so any count-based rule gets one of them wrong. Resolving is
    unambiguous.
    """
    for name, expr in _root_expressions(script):
        ns: dict = {"__file__": str(script)}
        exec("from pathlib import Path", ns)  # noqa: S102 — fixed literal
        resolved = Path(eval(expr, ns)).resolve()  # noqa: S307 — matched literal
        assert resolved == _REPO, (
            f"{script.name}: {name} = {expr} resolves to {resolved}, "
            f"not the repo root {_REPO}. Paths built from it will silently "
            f"miss, because the directories it points into do not exist."
        )


def test_the_directories_scripts_reach_for_are_absent_under_dot_github():
    """Why the breakage was silent rather than loud."""
    for d in ("backend", "frontend", "docs", "experiments"):
        assert not (_REPO / ".github" / d).exists(), (
            f".github/{d} now exists, so a wrong REPO_ROOT would resolve to a "
            f"real directory and this guard's failure mode changes"
        )


def test_the_tv_indicator_target_actually_exists():
    """The exact path whose absence aborted the TV Indicator workflow."""
    assert (_REPO / "backend/app/strategies/manual/tv_indicators.py").is_file()


def test_no_script_declares_a_bare_undefined_module_global_named_token():
    """agent_team.py had 65 bare `token` references and no definition.

    One of them crashed the Page Reporter workflow outright:
        NameError: name 'token' is not defined
    main() survived only because its bare `token` sat in the branch taken when
    auth.test SUCCEEDS, and auth.test was failing.
    """
    src = (_SCRIPTS / "agent_team.py").read_text()
    tree = ast.parse(src)
    module_level = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "token" in module_level, (
        "agent_team.py references a bare `token` in 11 entry points; it must be "
        "defined at module level or passed as a parameter"
    )
