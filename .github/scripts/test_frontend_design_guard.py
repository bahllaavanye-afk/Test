"""Guards against the frontend-design-agent committing truncated components.

The agent's LLM occasionally returns a cut-off file (max_tokens), and the old
syntax_check_tsx (export + import + <> + len>100) happily passed a file that
ended at `return (`. That truncated RegimeIndicator.tsx and AppShell.tsx and
broke `npm run build` twice, blocking auto-merge fleet-wide. These tests pin
the stronger validation so a truncated response is skipped, not committed.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_SCRIPTS = Path(__file__).parent


def _load_agent():
    # stub the LLM layer so importing the agent needs no network/keys
    stub = types.ModuleType("llm_common")
    stub.llm = lambda *a, **k: None
    stub.slack_post = lambda *a, **k: None
    stub.memory_write = lambda *a, **k: None
    sys.modules["llm_common"] = stub
    spec = importlib.util.spec_from_file_location("fda_under_test", _SCRIPTS / "frontend_design_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


fda = _load_agent()

_GOOD = """import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

export const Widget = () => {
  const { data, isLoading } = useQuery({ queryKey: ['x'], queryFn: () => api.get('/x') })
  if (isLoading) {
    return (<div className="p-3">loading</div>)
  }
  return (
    <div className="p-3" style={{ color: '#00c853' }}>
      <span>{data?.value}</span>
    </div>
  )
}

export default Widget
"""


def test_good_file_passes():
    assert fda.syntax_check_tsx(_GOOD, _GOOD) is True


def test_default_export_identifier_ending_passes():
    # ends with `export default Widget` — a bare identifier, no semicolon
    assert _GOOD.rstrip().endswith("Widget")
    assert fda.syntax_check_tsx(_GOOD, _GOOD) is True


def test_truncated_at_return_paren_is_rejected():
    truncated = "\n".join(_GOOD.split("\n")[:6]) + "\n  if (isLoading) {\n    return ("
    assert fda.syntax_check_tsx(truncated, _GOOD) is False


def test_truncated_mid_numeric_literal_is_rejected():
    # the exact f36f06b failure shape: `refetchInterval: 60_`
    truncated = "import a from 'b'\nexport const W = () => {\n  useQuery({ refetchInterval: 60_"
    assert fda.syntax_check_tsx(truncated, _GOOD) is False


def test_large_shrink_is_rejected():
    # balanced but <60% of the original → response was cut off
    small = "import a from 'b'\nexport const W = () => (<div/>)\n"
    big = small + ("// filler line to make the original much longer\n" * 40)
    assert fda.syntax_check_tsx(small, big) is False


def test_unbalanced_brackets_helper():
    assert fda._brackets_balanced("const x = () => { return (<a/>) }") is True
    assert fda._brackets_balanced("const x = () => { return (") is False
    # braces inside strings/templates must not trip it
    assert fda._brackets_balanced("const s = `a{b}c`; const t = '}}}'; f()") is True
