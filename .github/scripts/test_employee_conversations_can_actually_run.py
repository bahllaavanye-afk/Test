"""An hourly workflow reported success 0/47 for as long as it has existed.

`employee_conversation_runner.py` opened with a pre-flight key check and, when it
failed, `sys.exit(0)` — a clean exit, so the step passed, the workflow went
green, and it posted "👥 Employee Conversations — run complete / Responded:
0/47" to #engineering every hour.

Production log, run at 2026-08-05 04:14 UTC on sha 071df8db:

    No LLM keys available — skipping real conversations

The guard kept its own list of seven env vars:

    GROQ, DEEPSEEK, SAMBANOVA, CEREBRAS, HYPERBOLIC, TOGETHER, GEMINI

`llm_common._PROVIDERS` has **eight**. The missing one is NVIDIA
(`NVIDIA_AGENTS_API_KEYS`, alt `NVIDIA_NIM_API_KEY`) — and that is the provider
this workflow actually supplies. Its six other mappings point at UNSUFFIXED
secrets that are empty: the repo's populated free-tier secrets are the `_1`
variants, which is why `agent-status-check`, `multi-agent-discussion` and
`claude-chat` (all mapping `secrets.GEMINI_API_KEY_1`) get real replies while
this one never did.

So a guard with its own copy of the provider list vetoed a cascade that had a
usable provider. The fix is not to add NVIDIA to the list — it is to stop
keeping a second list. `_any_llm_key()` now asks `llm_common` what it can reach,
so the guard cannot disagree with the thing it guards.

`test_the_guard_sees_a_provider_the_old_list_omitted` is the one that fails
against the old code.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent
_WF = _DIR.parent / "workflows" / "employee-conversations.yml"
sys.path.insert(0, str(_DIR))

_PROVIDER_ENVS = ("GROQ_API_KEY", "DEEPSEEK_API_KEY", "SAMBANOVA_API_KEY",
                  "CEREBRAS_API_KEY", "HYPERBOLIC_API_KEY", "TOGETHER_API_KEY",
                  "GEMINI_API_KEY", "NVIDIA_AGENTS_API_KEYS", "NVIDIA_NIM_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _PROVIDER_ENVS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _any_llm_key():
    """Import the guard fresh — the module exits at import when it returns False."""
    path = _DIR / "employee_conversation_runner.py"
    body = path.read_text().split("if not _any_llm_key():")[0]
    # `__file__` is used at module level (sys.path.insert(0, dirname(__file__))),
    # so the exec namespace has to provide it — otherwise the prelude NameErrors
    # and every guard test fails for a reason that has nothing to do with keys.
    ns: dict = {"__file__": str(path), "__name__": "employee_conversation_runner"}
    exec(compile(body, str(path), "exec"), ns)  # noqa: S102
    return ns["_any_llm_key"]()


def test_no_keys_at_all_still_reports_unavailable(clean_env):
    """The guard must keep working — this is not 'always return True'."""
    assert _any_llm_key() is False, (
        "with every provider env var unset the guard claimed keys were "
        "available. It has stopped checking anything."
    )


def test_the_guard_sees_a_provider_the_old_list_omitted(clean_env):
    """THE regression test. NVIDIA was absent from the hand-maintained list.

    This is the exact production condition: only the NVIDIA key is populated,
    which is what employee-conversations.yml supplies.
    """
    clean_env.setenv("NVIDIA_AGENTS_API_KEYS", "nvapi-xxxx")
    assert _any_llm_key() is True, (
        "the guard still cannot see NVIDIA_AGENTS_API_KEYS — the one provider "
        "this workflow is actually given. It will exit(0) and the run stays a "
        "no-op reporting 0/47."
    )


def test_the_alt_env_var_counts_too(clean_env):
    """llm_common's NVIDIA entry has key_env_alt: NVIDIA_NIM_API_KEY."""
    clean_env.setenv("NVIDIA_NIM_API_KEY", "nvapi-yyyy")
    assert _any_llm_key() is True, "key_env_alt is not being honoured"


@pytest.mark.parametrize("var", ["GROQ_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"])
def test_the_original_providers_still_count(clean_env, var):
    clean_env.setenv(var, "sk-test")
    assert _any_llm_key() is True, f"{var} no longer satisfies the guard"


def test_disabled_is_not_a_key(clean_env):
    """`ANTHROPIC_API_KEY: "disabled"` is the repo's idiom for off."""
    clean_env.setenv("GROQ_API_KEY", "disabled")
    assert _any_llm_key() is False, (
        "the literal string 'disabled' was accepted as a usable key"
    )


def test_the_guard_does_not_keep_its_own_provider_list():
    """The root cause was a second copy of the list, not a missing entry.

    Re-adding a hardcoded list would pass every test above while drifting again
    the next time a provider is added to llm_common.
    """
    src = (_DIR / "employee_conversation_runner.py").read_text()
    assert "_LLM_KEY_VARS" not in src, (
        "a hand-maintained provider list is back. It will drift from "
        "llm_common._PROVIDERS again — that drift is why this workflow was a "
        "no-op."
    )
    assert "_PROVIDERS" in src and "_has_key" in src, (
        "the guard no longer derives from llm_common, so it can once more "
        "disagree with the cascade it is guarding."
    )


def test_it_degrades_open_not_closed(clean_env, monkeypatch):
    """If llm_common cannot be imported, run rather than silently skip.

    Exiting 0 on an import error reproduces the original failure: a green,
    silent no-op. Attempting the run surfaces a real error instead.
    """
    src = (_DIR / "employee_conversation_runner.py").read_text()
    guard = src.split("if not _any_llm_key():")[0]
    assert "return True" in guard.split("except")[-1], (
        "the except branch no longer fails open — an import problem would make "
        "this workflow silently do nothing again."
    )


# ── the workflow half ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("var", ["GROQ_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"])
def test_the_workflow_maps_the_populated_secret_variants(var):
    """The repo's real free-tier secrets are the `_1` names."""
    wf = _WF.read_text()
    m = re.search(rf"^\s*{var}:\s*(.+)$", wf, re.M)
    assert m, f"{var} is no longer passed to the runner at all"
    value = m.group(1)
    assert f"secrets.{var}_1" in value, (
        f"{var} maps only `secrets.{var}` — the unsuffixed secret, which is "
        f"empty. That is why this workflow had no free provider. Working "
        f"workflows map `secrets.{var}_1`."
    )


def test_the_mapping_keeps_the_unsuffixed_fallback():
    """`X_1 || X` cannot regress if the unsuffixed secret is later populated."""
    wf = _WF.read_text()
    assert "secrets.GROQ_API_KEY_1 || secrets.GROQ_API_KEY" in wf, (
        "the fallback was dropped. Pinning to `_1` alone breaks the workflow "
        "if the secrets are ever renamed back."
    )


def test_the_nvidia_keys_are_still_supplied():
    """They are what made the guard's omission matter."""
    wf = _WF.read_text()
    assert "NVIDIA_AGENTS_API_KEYS: ${{ secrets.NVIDIA_AGENTS_API_KEYS }}" in wf
    assert "NVIDIA_NIM_API_KEY: ${{ secrets.NVIDIA_NIM_API_KEY }}" in wf


def test_the_workflow_yaml_still_parses():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_WF.read_text())
    assert doc, "employee-conversations.yml no longer parses"
    steps = doc["jobs"]["employee-conversations"]["steps"]
    run_step = next(s for s in steps if "employee_conversation_runner.py" in str(s.get("run", "")))
    assert run_step["env"]["GROQ_API_KEY"], "the runner step lost its key env"


# ── call pacing (first real run returned 1/47) ────────────────────────────────

def _spacing(monkeypatch, **env):
    """Import the module with a key present, then evaluate the pacing helper."""
    monkeypatch.setenv("GEMINI_API_KEY", "x")   # so the module-level guard passes
    for k in _PROVIDER_ENVS:
        if k != "GEMINI_API_KEY":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("EMPLOYEE_CALL_SPACING_S", raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import employee_conversation_runner as runner
    return runner._call_spacing_seconds()


def test_spacing_leaves_headroom_below_geminis_declared_rpm(monkeypatch):
    """The first real run fired 47 calls in ~20s and got 1 response.

    llm_common declares `"rpm_free": 15` for Gemini — 47 calls in 20s is
    ~140/min against it, so the 429 wall was arithmetic, not an outage.

    But `60/15 = 4.0s` is exactly 100% of the quota, which assumes this workflow
    is the only consumer. 42 workflows map `secrets.GEMINI_API_KEY_1` and a
    dozen run on schedules, so the quota is fleet-wide: a 100%-utilisation plan
    walks straight back into 429s the moment anything else fires.
    """
    spacing = _spacing(monkeypatch)
    assert spacing == pytest.approx(60.0 / (15 * 0.6)), (
        f"expected 60/(15*0.6)=6.67s for a Gemini-only key set, got {spacing}. "
        "Either the derivation changed or the fleet-quota margin was dropped."
    )
    assert spacing > 60.0 / 15, (
        "spacing is at or below 100% of the declared RPM, leaving nothing for "
        "the dozen other scheduled workflows sharing this key."
    )


def test_spacing_widens_when_only_a_slower_provider_is_available(monkeypatch):
    """It must be derived, not hardcoded — that is the whole point."""
    fast = _spacing(monkeypatch, TOGETHER_API_KEY="x")   # rpm_free 60
    assert fast == pytest.approx(60.0 / (60 * 0.6)), (
        f"expected 60/(60*0.6)=1.67s with Together available, got {fast}. The "
        "interval is not being derived from llm_common's rpm_free table."
    )
    assert fast < _spacing(monkeypatch), "more headroom must mean less waiting"


def test_the_quota_share_is_a_real_margin_not_a_no_op(monkeypatch):
    """Guards the specific mutation of setting the share back to 1.0."""
    import employee_conversation_runner as runner
    assert 0 < runner._QUOTA_SHARE < 1.0, (
        f"_QUOTA_SHARE={runner._QUOTA_SHARE} does not reserve any quota for the "
        "rest of the fleet."
    )


def test_a_keyless_environment_still_paces(monkeypatch):
    """Fail safe: unknown capacity means assume the observed floor, not zero."""
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    import employee_conversation_runner as runner
    for k in _PROVIDER_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("EMPLOYEE_CALL_SPACING_S", raising=False)
    assert runner._call_spacing_seconds() == pytest.approx(60.0 / (15 * 0.6)), (
        "with no keys visible the spacing collapsed. A zero interval reproduces "
        "the 1/47 run."
    )


def test_the_override_wins(monkeypatch):
    assert _spacing(monkeypatch, EMPLOYEE_CALL_SPACING_S="1.5") == pytest.approx(1.5)


def test_a_malformed_override_falls_back_instead_of_crashing(monkeypatch):
    assert _spacing(monkeypatch, EMPLOYEE_CALL_SPACING_S="fast") == pytest.approx(60.0 / (15 * 0.6))


def test_a_negative_override_cannot_go_below_zero(monkeypatch):
    assert _spacing(monkeypatch, EMPLOYEE_CALL_SPACING_S="-5") == 0.0


def test_the_full_roster_fits_inside_the_workflow_timeout(monkeypatch):
    """Pacing that exceeds the job timeout trades one silent failure for another."""
    spacing = _spacing(monkeypatch)
    yaml = pytest.importorskip("yaml")   # module-level import is not available here
    doc = yaml.safe_load(_WF.read_text())
    (job,) = doc["jobs"].values()
    budget_s = job["timeout-minutes"] * 60
    import employee_conversation_runner as runner
    roster = len(runner._EMPLOYEE_PERSONAS)
    sleep_s = roster * spacing
    assert sleep_s < budget_s * 0.5, (
        f"{roster} employees x {spacing}s = {sleep_s/60:.1f} min of pure sleep "
        f"against a {job['timeout-minutes']}-min timeout, leaving too little for "
        "the calls themselves. Raise the timeout or cap EMPLOYEE_RUNNER_LIMIT."
    )


def test_the_loop_sleeps_between_calls_not_before_the_first(monkeypatch):
    """A leading sleep adds dead time to every run for no benefit."""
    src = (_DIR / "employee_conversation_runner.py").read_text()
    loop = src[src.index("for i, emp_key in enumerate(emp_keys):"):]
    loop = loop[:loop.index("employee_provider_prompt")]
    assert "if i:" in loop and "time.sleep(spacing)" in loop, (
        "the inter-employee sleep is unguarded, so the run waits before doing "
        "any work at all."
    )
