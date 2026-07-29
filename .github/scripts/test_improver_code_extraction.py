"""78% of the improver's failures were a fence it refused to look for.

With the failure counter finally working (`test_improver_failure_counter.py`),
the breakdown became visible for the first time:

    syntax check failed   32
    LLM returned empty     9
                          ──
                          41

32 of 41. The cause is one line of extraction logic:

    if improved.startswith("```"):     # position 0, or nothing happens
        ...strip the fence...

That only unwraps a fence when the response *begins* with one. The cascade's
free providers do not oblige. `llm_common._extract()` deliberately falls back
to `reasoning_content` when `content` is empty, so a reasoning model's
chain-of-thought comes back AS the response — and that is not hypothetical,
it is in `.github/state/agent_status.json` right now, produced by the same
`llm()` helper `improve_file()` calls:

    "The user asks: \"Give a one-sentence status update: ...\" The developer…"
    "We need to respond as algo_agent, one sentence status update, …"

Prepend any of that and `startswith("```")` is False, so the prose AND the
fence went straight into `compile()`. SyntaxError, every time, silently
counted nowhere.

The fix finds a fenced block anywhere in the response and takes the longest
one — responses often carry small illustrative snippets next to the real file,
and the file is the long one.

Note what is deliberately NOT done: no attempt to salvage prose into code. A
response with no fence and no valid Python still fails the syntax check. The
goal is to stop discarding good output, not to start accepting bad output.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent / "continuous_improver.py"

_FILE = 'def f(x):\n    """Doc."""\n    return x + 1\n'


def _load(fn_name: str):
    """Exec one function in isolation — importing the module runs the improver."""
    tree = ast.parse(_SRC.read_text())
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name),
        None,
    )
    assert fn is not None, f"{fn_name}() is gone"
    ns: dict = {}
    exec("import re", ns)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"<{fn_name}>", "exec"), ns)
    return ns[fn_name]


@pytest.fixture(scope="module")
def extract():
    return _load("_extract_code")


def _compiles(src: str) -> bool:
    try:
        compile(src, "<t>", "exec")
        return True
    except SyntaxError:
        return False


# The shapes actually observed from this cascade, verbatim in style.
_PREAMBLES = [
    'The user asks: "Output the complete improved file". The developer wants '
    "docstrings added. Let me rewrite it.\n\n",
    "We need to respond as continuous_improver, improving the file. "
    "Should add type hints.\n\n",
    "Here's the improved file:\n\n",
    "Sure! I've added docstrings.\n\n",
]


@pytest.mark.parametrize("preamble", _PREAMBLES, ids=range(len(_PREAMBLES)))
def test_prose_before_the_fence_no_longer_destroys_the_output(extract, preamble):
    """The exact 32-of-41 failure. Old code returned prose; compile() died."""
    response = f"{preamble}```python\n{_FILE}```\n"
    got = extract(response)
    assert got.strip() == _FILE.strip()
    assert _compiles(got), "extracted text must be valid Python"


def test_a_bare_fence_at_position_zero_still_works(extract):
    """The one case the old code did handle — must not regress."""
    assert extract(f"```python\n{_FILE}```").strip() == _FILE.strip()
    assert extract(f"```\n{_FILE}```").strip() == _FILE.strip()


def test_trailing_prose_after_the_fence_is_dropped(extract):
    response = f"```python\n{_FILE}```\n\nI added a docstring and a type hint."
    got = extract(response)
    assert got.strip() == _FILE.strip()
    assert "I added a docstring" not in got


def test_the_longest_block_wins_over_illustrative_snippets(extract):
    """Models explain with small snippets and then give the file."""
    response = (
        "First, note the change:\n\n```python\nreturn x + 1\n```\n\n"
        f"Here is the complete file:\n\n```python\n{_FILE}```\n"
    )
    got = extract(response)
    assert got.strip() == _FILE.strip(), "picked a snippet instead of the file"


def test_an_unterminated_fence_returns_the_partial_code(extract):
    """A truncated response should fail the syntax check, not be mangled."""
    partial = 'def f(x):\n    """Doc.\n'
    got = extract(f"Here you go:\n\n```python\n{partial}")
    assert got.strip() == partial.strip()
    assert not _compiles(got), (
        "a truncated file must still fail the syntax check — this test exists "
        "to confirm the fix does not paper over real truncation"
    )


def test_no_fence_at_all_is_passed_through_unchanged(extract):
    """Some providers return bare code. Others return pure prose."""
    assert extract(_FILE).strip() == _FILE.strip()
    prose = "I cannot improve this file because it is already optimal."
    assert extract(prose).strip() == prose
    assert not _compiles(extract(prose)), (
        "pure prose must still be rejected downstream — the fix must not start "
        "accepting bad output, only stop discarding good output"
    )


def test_language_tag_variants(extract):
    for tag in ("python", "py", ""):
        assert extract(f"prose\n\n```{tag}\n{_FILE}```").strip() == _FILE.strip(), tag


def test_the_old_startswith_check_is_gone():
    """Pin the actual defect so it cannot be reintroduced.

    Checked over the AST, not the text. A substring search hit the quotation of
    the old line inside `_extract_code`'s own docstring and failed on correct
    code — the classic weak-test trap this repo keeps rediscovering, and worth
    one comment here since the fix explains itself only in prose.
    """
    tree = ast.parse(_SRC.read_text())
    for fn_name in ("improve_file", "_extract_code"):
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == fn_name
        )
        # Docstrings are Constants, not Calls — they cannot trip this.
        calls = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "startswith"
        ]
        assert not calls, (
            f"{fn_name}() calls .startswith() again — the position-0-only fence "
            f"check is back. Any prose preamble makes it False and the whole "
            f"response goes to compile() as-is."
        )

    improve = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "improve_file"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_extract_code"
        for n in ast.walk(improve)
    ), "improve_file() must route its response through _extract_code()"
