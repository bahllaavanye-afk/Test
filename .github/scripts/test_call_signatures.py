"""Agent scripts must call each other with arguments that actually exist.

The Slack→Discord rename left call sites pointing at signatures that had
changed underneath them, and each one raised TypeError on every run:

  * `chat_read_channel(cid, limit=30, oldest=cutoff)` in company_brain.py —
    `oldest` was a Slack server-side parameter that the Discord version never
    grew. Company Brain Sync failed every 15 minutes for weeks.
  * `chat_read_channel(channel_id, thread_ts, limit=15)` in employee_intros.py
    — `thread_ts` was still being passed positionally into a 2-arg function.

`backend/tests/unit/test_dataclass_kwargs.py` catches this class inside
`backend/app`, but nothing watched `.github/scripts/`, which is where the
fleet actually lives and where the improver edits unattended.

Checks calls to helpers defined in this directory: unknown keyword arguments,
and too many positional arguments. Conservative — a function defined more than
once, or one taking **kwargs/*args, is skipped entirely.
"""
from __future__ import annotations

import ast
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parent


def _collect_defs() -> dict[str, list[dict]]:
    defs: dict[str, list[dict]] = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            positional = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
            defs.setdefault(node.name, []).append({
                "positional": positional,
                "accepts": set(positional) | {p.arg for p in a.kwonlyargs},
                "star": a.vararg is not None,
                "kwargs": a.kwarg is not None,
                "decorated": bool(node.decorator_list),
                "is_method": bool(positional) and positional[0] in ("self", "cls"),
                "file": path.name,
                "line": node.lineno,
            })
    return defs


def _checkable(defs: dict[str, list[dict]]) -> dict[str, dict]:
    """Only names we can resolve unambiguously."""
    return {
        name: e[0]
        for name, e in defs.items()
        if len(e) == 1
        and not e[0]["kwargs"]
        and not e[0]["star"]
        and not e[0]["decorated"]
        and not e[0]["is_method"]
    }


def _visible_here(path: pathlib.Path, tree: ast.Module,
                  checkable: dict[str, dict]) -> dict[str, dict]:
    """Names this module can actually resolve.

    Python resolves a bare name against the module's own definitions and its
    imports — NOT against every function in the directory. Matching globally
    produced false positives immediately: `multi_agent_discussion._llm` is its
    own local helper, and `quote` in render_probe_pooler is urllib's, not
    tradier_data's.
    """
    own = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # local name → the name it actually refers to. `from x import llm as _llm`
    # binds _llm to `llm`; resolving the LOCAL name against a same-named
    # function elsewhere is how this checker first mis-flagged _llm.
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # only sibling scripts — stdlib/3rd-party signatures aren't ours
            if (SCRIPTS / f"{node.module.split('.')[-1]}.py").exists():
                for a in node.names:
                    imported[a.asname or a.name] = a.name

    visible = {}
    for local_name, real_name in imported.items():
        if local_name in own:
            continue                      # a local def shadows the import
        info = checkable.get(real_name)
        if info is not None:
            visible[local_name] = info
    for name, info in checkable.items():
        if name in own and info["file"] == path.name:
            visible[name] = info          # own definitions win
    return visible


def _violations() -> list[str]:
    checkable = _checkable(_collect_defs())
    out: list[str] = []

    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visible = _visible_here(path, tree, checkable)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            info = visible.get(node.func.id)
            if info is None:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            if any(k.arg is None for k in node.keywords):
                continue

            for kw in node.keywords:
                if kw.arg not in info["accepts"]:
                    out.append(
                        f"{path.name}:{node.lineno} — {node.func.id}({kw.arg}=...) "
                        f"is not a parameter of {info['file']}:{info['line']}. "
                        f"Accepts: {', '.join(sorted(info['accepts']))}"
                    )
            if len(node.args) > len(info["positional"]):
                out.append(
                    f"{path.name}:{node.lineno} — {node.func.id}() got "
                    f"{len(node.args)} positional args but {info['file']}:{info['line']} "
                    f"takes {len(info['positional'])} ({', '.join(info['positional'])})"
                )
    return out


def test_no_agent_script_calls_a_helper_with_arguments_it_lacks():
    violations = _violations()
    assert not violations, (
        "agent script calls a helper with arguments it does not accept — this "
        "raises TypeError on every run, and these scripts run unattended on a "
        "schedule:\n  " + "\n  ".join(violations)
    )


def test_checker_finds_a_planted_bad_keyword(tmp_path):
    """The check above only means something if it can actually fail."""
    tree = ast.parse('''
def helper(channel, limit=50):
    return []

helper(cid, limit=30, oldest=123)
''')
    defs: dict[str, list[dict]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            positional = [p.arg for p in node.args.args]
            defs[node.name] = [{
                "positional": positional, "accepts": set(positional),
                "star": False, "kwargs": False, "decorated": False,
                "is_method": False, "file": "x.py", "line": node.lineno,
            }]
    info = _checkable(defs)["helper"]
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    bad = [k.arg for k in call.keywords if k.arg not in info["accepts"]]
    assert bad == ["oldest"]


def test_checker_finds_too_many_positionals():
    tree = ast.parse('''
def helper(channel, limit=50):
    return []

helper(cid, thread_ts, limit=15)
''')
    defs: dict[str, list[dict]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            positional = [p.arg for p in node.args.args]
            defs[node.name] = [{
                "positional": positional, "accepts": set(positional),
                "star": False, "kwargs": False, "decorated": False,
                "is_method": False, "file": "x.py", "line": node.lineno,
            }]
    info = _checkable(defs)["helper"]
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    assert len(call.args) == 2 and len(info["positional"]) == 2
    # 2 positionals + limit= keyword would collide on `limit`; the real
    # employee_intros bug was exactly this shape.
    assert "limit" in {k.arg for k in call.keywords}


def test_varargs_and_kwargs_helpers_are_skipped():
    """A helper taking **kwargs can accept anything — never flag it."""
    tree = ast.parse('''
def flexible(a, **kw):
    pass
''')
    defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = [{
                "positional": ["a"], "accepts": {"a"}, "star": False,
                "kwargs": True, "decorated": False, "is_method": False,
                "file": "x.py", "line": 1,
            }]
    assert "flexible" not in _checkable(defs)
