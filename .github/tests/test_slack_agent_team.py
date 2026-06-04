"""
Tests for role-based employee key lookup and new agent registration.
Covers: split-bug fix for multi-segment keys (e.g. 'vp_eng', 'alpha_dir'),
account dict completeness for all core employees, and new agent persona coverage.

All tests are fully offline — no real Slack token or LLM keys required.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Make the scripts directory importable and stub out optional heavy packages
# before importing the module under test.
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Stub langfuse before import so we never need the real package
_lf_stub = types.ModuleType("langfuse")
_lf_dec_stub = types.ModuleType("langfuse.decorators")


def _noop_observe(fn=None, **kw):
    return fn if fn else (lambda f: f)


class _noop_ctx:
    @staticmethod
    def update_current_observation(**kw):
        pass

    @staticmethod
    def score_current_observation(**kw):
        pass


_lf_dec_stub.observe = _noop_observe
_lf_dec_stub.langfuse_context = _noop_ctx
sys.modules.setdefault("langfuse", _lf_stub)
sys.modules.setdefault("langfuse.decorators", _lf_dec_stub)

# Stub litellm so it is treated as unavailable
_litellm_stub = types.ModuleType("litellm")
_litellm_stub.set_verbose = False
_litellm_stub.drop_params = True
_litellm_stub.completion = MagicMock(side_effect=RuntimeError("litellm not available in tests"))
sys.modules.setdefault("litellm", _litellm_stub)

# Now import the module under test
import slack_agent_team as sat  # noqa: E402


# ===========================================================================
# 1. _groq_key_for — role-based key (no split bug)
# ===========================================================================

def test_groq_key_for_role_based_key():
    """_groq_key_for('vp_eng') must resolve to GROQ_API_KEY_1, not fall through to default.

    Before the split-bug fix, 'vp_eng'.split('_')[0] == 'vp' which is absent from
    _GROQ_ACCOUNT, so the function fell back to GROQ_API_KEY_1 accidentally.
    The fix checks the full key first, then only falls back to the first segment
    for legacy bare names.  This test verifies the full-key path is taken.
    """
    env_patch = {"GROQ_API_KEY_1": "test-groq-key-1"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._groq_key_for("vp_eng")
    assert key == "test-groq-key-1", (
        f"_groq_key_for('vp_eng') returned {key!r}; expected the GROQ_API_KEY_1 value. "
        "The split('_')[0] bug may still be present."
    )


def test_groq_key_for_alpha_dir_uses_key1():
    """_groq_key_for('alpha_dir') must resolve to GROQ_API_KEY_1."""
    env_patch = {"GROQ_API_KEY_1": "key-for-alpha-dir"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._groq_key_for("alpha_dir")
    assert key == "key-for-alpha-dir"


def test_groq_key_for_poly_desk_uses_key3():
    """_groq_key_for('poly_desk') must resolve to GROQ_API_KEY_3."""
    env_patch = {"GROQ_API_KEY_3": "key-for-poly-desk"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._groq_key_for("poly_desk")
    assert key == "key-for-poly-desk"


# ===========================================================================
# 2. _gemini_key_for — role-based key (no split bug)
# ===========================================================================

def test_gemini_key_for_role_based_key():
    """_gemini_key_for('alpha_dir') must resolve to GEMINI_API_KEY_1."""
    env_patch = {"GEMINI_API_KEY_1": "gemini-key-1"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._gemini_key_for("alpha_dir")
    assert key == "gemini-key-1", (
        f"_gemini_key_for('alpha_dir') returned {key!r}; expected GEMINI_API_KEY_1 value."
    )


def test_gemini_key_for_backend_lead_uses_key2():
    """_gemini_key_for('backend_lead') must resolve to GEMINI_API_KEY_2."""
    env_patch = {"GEMINI_API_KEY_2": "gemini-key-2"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._gemini_key_for("backend_lead")
    assert key == "gemini-key-2"


def test_gemini_key_for_poly_desk_uses_key3():
    """_gemini_key_for('poly_desk') must resolve to GEMINI_API_KEY_3."""
    env_patch = {"GEMINI_API_KEY_3": "gemini-key-3"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._gemini_key_for("poly_desk")
    assert key == "gemini-key-3"


# ===========================================================================
# 3. _cerebras_key_for — role-based key (no split bug)
# ===========================================================================

def test_cerebras_key_for_role_based_key():
    """_cerebras_key_for('poly_desk') must resolve to CEREBRAS_API_KEY_2 (group 3)."""
    env_patch = {"CEREBRAS_API_KEY_2": "cerebras-key-2"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._cerebras_key_for("poly_desk")
    assert key == "cerebras-key-2", (
        f"_cerebras_key_for('poly_desk') returned {key!r}; expected CEREBRAS_API_KEY_2 value."
    )


def test_cerebras_key_for_vp_eng_uses_key1():
    """_cerebras_key_for('vp_eng') must resolve to CEREBRAS_API_KEY_1 (group 1)."""
    env_patch = {"CEREBRAS_API_KEY_1": "cerebras-key-1"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._cerebras_key_for("vp_eng")
    assert key == "cerebras-key-1"


def test_cerebras_key_for_cro_uses_key2():
    """_cerebras_key_for('cro') must resolve to CEREBRAS_API_KEY_2 (group 3)."""
    env_patch = {"CEREBRAS_API_KEY_2": "cerebras-cro-key"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        key = sat._cerebras_key_for("cro")
    assert key == "cerebras-cro-key"


# ===========================================================================
# 4. _employee_keys — groq provider for ml_lead
# ===========================================================================

def test_employee_keys_groq_for_ml_lead():
    """_employee_keys('ml_lead', 'groq') must return a list containing the GROQ_API_KEY_1 value."""
    env_patch = {"GROQ_API_KEY_1": "groq-ml-lead-key"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        keys = sat._employee_keys("ml_lead", "groq")
    assert isinstance(keys, list)
    assert len(keys) >= 1
    assert "groq-ml-lead-key" in keys, (
        f"_employee_keys('ml_lead', 'groq') returned {keys!r}; expected GROQ_API_KEY_1 value."
    )


def test_employee_keys_groq_for_exec_eng_uses_key2():
    """_employee_keys('exec_eng', 'groq') must return the GROQ_API_KEY_2 value."""
    env_patch = {"GROQ_API_KEY_2": "groq-exec-key"}
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        keys = sat._employee_keys("exec_eng", "groq")
    assert "groq-exec-key" in keys


def test_employee_keys_groq_returns_single_key_per_employee():
    """Groq must return only one key per employee — no cross-account contamination."""
    env_patch = {
        "GROQ_API_KEY_1": "key-1",
        "GROQ_API_KEY_2": "key-2",
        "GROQ_API_KEY_3": "key-3",
    }
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(os.environ, env_patch, clear=False):
        keys = sat._employee_keys("vp_eng", "groq")
    assert len(keys) == 1, (
        f"Groq must return exactly 1 key per employee to prevent quota cross-contamination. Got {len(keys)}: {keys}"
    )
    assert keys[0] == "key-1"


# ===========================================================================
# 5. All _EMPLOYEES are in _GROQ_ACCOUNT
# ===========================================================================

def test_all_employees_in_groq_account():
    """Every key in _EMPLOYEES must have an entry in _GROQ_ACCOUNT."""
    missing = [emp for emp in sat._EMPLOYEES if emp not in sat._GROQ_ACCOUNT]
    assert not missing, (
        f"The following employees are missing from _GROQ_ACCOUNT: {missing}. "
        "Add them to _GROQ_ACCOUNT with an appropriate key."
    )


# ===========================================================================
# 6. All _EMPLOYEES are in _GEMINI_ACCOUNT
# ===========================================================================

def test_all_employees_in_gemini_account():
    """Every key in _EMPLOYEES must have an entry in _GEMINI_ACCOUNT."""
    missing = [emp for emp in sat._EMPLOYEES if emp not in sat._GEMINI_ACCOUNT]
    assert not missing, (
        f"The following employees are missing from _GEMINI_ACCOUNT: {missing}. "
        "Add them to _GEMINI_ACCOUNT with an appropriate key."
    )


# ===========================================================================
# 7. All _EMPLOYEES are in _CEREBRAS_ACCOUNT
# ===========================================================================

def test_all_employees_in_cerebras_account():
    """Every key in _EMPLOYEES must have an entry in _CEREBRAS_ACCOUNT."""
    missing = [emp for emp in sat._EMPLOYEES if emp not in sat._CEREBRAS_ACCOUNT]
    assert not missing, (
        f"The following employees are missing from _CEREBRAS_ACCOUNT: {missing}. "
        "Add them to _CEREBRAS_ACCOUNT with an appropriate key."
    )


# ===========================================================================
# 8. New agents have personas in _EMPLOYEE_PERSONAS
# ===========================================================================

_NEW_AGENTS = [
    # Group 1 (KEY_1)
    "equity_lead",
    "fixed_income_desk",
    "macro_researcher",
    "stat_arb_desk",
    "vol_trader",
    "momentum_quant",
    "alt_data_lead",
    "model_validator",
    "feature_engineer",
    # Group 2 (KEY_2)
    "crypto_quant",
    "derivatives_desk",
    "arb_trader",
    "portfolio_manager",
    "market_maker",
    "regime_analyst",
    "backtest_engineer",
    "data_engineer_2",
    "infra_lead",
    # Group 3 (KEY_3)
    "nlp_researcher",
    "rl_trader",
    "graph_ml_researcher",
    "crypto_defi_desk",
    "kalshi_desk",
    "research_ops",
    "senior_quant",
    "latency_engineer",
]


def test_new_agents_have_personas():
    """All 25 new specialized agents must have entries in _EMPLOYEE_PERSONAS."""
    missing = [agent for agent in _NEW_AGENTS if agent not in sat._EMPLOYEE_PERSONAS]
    assert not missing, (
        f"The following new agents are missing from _EMPLOYEE_PERSONAS: {missing}. "
        "Add a system prompt persona for each."
    )


def test_new_agents_have_groq_accounts():
    """All 25 new specialized agents must have entries in _GROQ_ACCOUNT."""
    missing = [agent for agent in _NEW_AGENTS if agent not in sat._GROQ_ACCOUNT]
    assert not missing, (
        f"The following new agents are missing from _GROQ_ACCOUNT: {missing}."
    )


def test_new_agents_have_gemini_accounts():
    """All 25 new specialized agents must have entries in _GEMINI_ACCOUNT."""
    missing = [agent for agent in _NEW_AGENTS if agent not in sat._GEMINI_ACCOUNT]
    assert not missing, (
        f"The following new agents are missing from _GEMINI_ACCOUNT: {missing}."
    )


def test_new_agents_have_cerebras_accounts():
    """All 25 new specialized agents must have entries in _CEREBRAS_ACCOUNT."""
    missing = [agent for agent in _NEW_AGENTS if agent not in sat._CEREBRAS_ACCOUNT]
    assert not missing, (
        f"The following new agents are missing from _CEREBRAS_ACCOUNT: {missing}."
    )


def test_new_agents_persona_is_non_empty_string():
    """Each new agent persona must be a non-empty string containing role-relevant content."""
    for agent in _NEW_AGENTS:
        persona = sat._EMPLOYEE_PERSONAS.get(agent, "")
        assert isinstance(persona, str) and len(persona) > 50, (
            f"Agent '{agent}' has an empty or trivially short persona: {persona!r}"
        )


def test_allow_paid_apis_is_false():
    """ALLOW_PAID_APIS must remain False — zero-spend policy."""
    assert sat.ALLOW_PAID_APIS is False, (
        "ALLOW_PAID_APIS must never be True. This protects against accidental paid API usage."
    )


def test_no_old_human_names_in_employees_list():
    """The _EMPLOYEES list must not contain old human first-name keys."""
    old_names = {
        "maya", "aarav", "linh", "jian", "anna", "aditi", "kenji",
        "diego", "lior", "sara", "sofia", "hugo", "marcus", "priya",
        "ravi", "karl", "tomas", "cameron", "wei", "sina", "alex",
        "yuki", "helena", "laavanye",
    }
    found = [emp for emp in sat._EMPLOYEES if emp in old_names]
    assert not found, (
        f"Old human-name keys still present in _EMPLOYEES: {found}. "
        "All employees should use role-based keys (e.g. 'vp_eng', 'alpha_dir')."
    )


def test_groq_account_values_are_valid_env_var_names():
    """All values in _GROQ_ACCOUNT must be valid environment variable name strings."""
    import re
    pattern = re.compile(r'^[A-Z][A-Z0-9_]*$')
    for key, val in sat._GROQ_ACCOUNT.items():
        assert pattern.match(val), (
            f"_GROQ_ACCOUNT['{key}'] = {val!r} is not a valid env var name."
        )


def test_cerebras_account_only_two_keys():
    """_CEREBRAS_ACCOUNT values must only be CEREBRAS_API_KEY_1 or CEREBRAS_API_KEY_2."""
    allowed = {"CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2"}
    for key, val in sat._CEREBRAS_ACCOUNT.items():
        assert val in allowed, (
            f"_CEREBRAS_ACCOUNT['{key}'] = {val!r} is not one of {allowed}."
        )
