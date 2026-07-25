"""
Safety and correctness checks for all .github/scripts/*.py files.
Tests run without network access — they only inspect source code.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"

# ── helpers ──────────────────────────────────────────────────────────────────

def _src(name: str) -> str:
    p = SCRIPTS_DIR / name
    assert p.exists(), f"Script {name} not found at {p}"
    return p.read_text(encoding="utf-8")


def _all_scripts() -> list[str]:
    return [p.name for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("test_")]


# ── undefined 'msg' variable bug (was crashing apply_and_push / git_commit_and_push) ──

class TestUndefinedMsgBug:
    """The apply_and_push / git_commit_and_push functions must not reference an
    undefined 'msg' variable. This was a critical NameError in production."""

    def test_backend_team_no_undefined_msg(self):
        src = _src("backend_team.py")
        m = re.search(r"def apply_and_push.*?(?=\ndef |\Z)", src, re.DOTALL)
        assert m, "apply_and_push not found in backend_team.py"
        fn_body = m.group(0)
        # If the standalone word 'msg' is used, it must be defined in the function body
        if re.search(r"\bmsg\b", fn_body):
            assert re.search(r"\bmsg\s*=", fn_body), (
                "backend_team.py apply_and_push uses 'msg' without defining it"
            )

    def test_frontend_team_no_undefined_msg(self):
        src = _src("frontend_team.py")
        m = re.search(r"def apply_and_push.*?(?=\ndef |\Z)", src, re.DOTALL)
        assert m, "apply_and_push not found in frontend_team.py"
        fn_body = m.group(0)
        if re.search(r"\bmsg\b", fn_body):
            assert re.search(r"\bmsg\s*=", fn_body), (
                "frontend_team.py apply_and_push uses 'msg' without defining it"
            )

    def test_render_autofix_no_undefined_msg(self):
        src = _src("render_auto_fix.py")
        m = re.search(r"def git_commit_and_push.*?(?=\ndef |\Z)", src, re.DOTALL)
        assert m, "git_commit_and_push not found in render_auto_fix.py"
        fn_body = m.group(0)
        if re.search(r"\bmsg\b", fn_body):
            # msg must be defined OR the function parameter 'reason' must be used instead
            has_msg_assign = bool(re.search(r"\bmsg\s*=", fn_body))
            has_reason = "reason" in fn_body
            assert has_msg_assign or not re.search(r'git.*commit.*"\bmsg\b"', fn_body), (
                "render_auto_fix.py git_commit_and_push uses undefined 'msg'"
            )
            _ = has_reason  # reason param is the intended fix

    def test_render_autofix_uses_reason_param(self):
        src = _src("render_auto_fix.py")
        m = re.search(r"def git_commit_and_push\(reason.*?\).*?(?=\ndef |\Z)", src, re.DOTALL)
        assert m, "git_commit_and_push(reason) not found"
        fn_body = m.group(0)
        assert "reason" in fn_body, (
            "git_commit_and_push has 'reason' param but never uses it"
        )


# ── apply_fix vs apply_patches naming consistency ────────────────────────────

class TestFunctionNaming:
    def test_render_autofix_no_apply_patches(self):
        src = _src("render_auto_fix.py")
        # apply_patches is not defined — only apply_fix is
        assert "def apply_patches" not in src, (
            "render_auto_fix.py defines apply_patches which conflicts with apply_fix"
        )
        # _gemini_autofix must call apply_fix, not apply_patches
        gemini_m = re.search(r"def _gemini_autofix.*?(?=\ndef |\Z)", src, re.DOTALL)
        if gemini_m:
            body = gemini_m.group(0)
            assert "apply_patches" not in body, (
                "_gemini_autofix calls undefined apply_patches — should call apply_fix"
            )


# ── ANTHROPIC_API_KEY disabled guard ─────────────────────────────────────────

class TestAnthropicDisabledGuard:
    """Scripts that call anthropic.Anthropic(api_key=...) must guard against
    the key being 'disabled' (security policy sets it to that string)."""

    SCRIPTS_WITH_ANTHROPIC = [
        "backend_team.py",
        "frontend_team.py",
        "render_auto_fix.py",
    ]

    @pytest.mark.parametrize("script_name", SCRIPTS_WITH_ANTHROPIC)
    def test_script_checks_for_disabled(self, script_name):
        src = _src(script_name)
        # Must check for "disabled" string explicitly
        assert "disabled" in src, (
            f"{script_name} calls Anthropic API but does not check for "
            "ANTHROPIC_API_KEY == 'disabled' — will crash in CI where key is set to 'disabled'"
        )

    @pytest.mark.parametrize("script_name", SCRIPTS_WITH_ANTHROPIC)
    def test_script_has_free_fallback(self, script_name):
        src = _src(script_name)
        # Must have at least one free LLM fallback (Gemini or Groq)
        has_gemini = "gemini" in src.lower()
        has_groq = "groq" in src.lower()
        assert has_gemini or has_groq, (
            f"{script_name} has no free LLM fallback (Gemini/Groq) — "
            "will produce no output when Anthropic is disabled"
        )


# ── ALLOW_PAID_APIS guard ─────────────────────────────────────────────────────

class TestPaidApiGuard:
    SCRIPTS_REQUIRING_GUARD = [
        "multi_agent_discussion.py",
        "continuous_improver.py",
        "agent_team.py",
    ]

    @pytest.mark.parametrize("script_name", SCRIPTS_REQUIRING_GUARD)
    def test_allow_paid_apis_guard_present(self, script_name):
        p = SCRIPTS_DIR / script_name
        if not p.exists():
            pytest.skip(f"{script_name} not found")
        src = p.read_text()
        assert "ALLOW_PAID_APIS" in src, (
            f"{script_name} is missing ALLOW_PAID_APIS guard"
        )


# ── Branch constant ──────────────────────────────────────────────────────────

class TestBranchConstant:
    SCRIPTS_WITH_BRANCH = [
        "backend_team.py",
        "frontend_team.py",
        "render_auto_fix.py",
        "strategy_generator.py",
    ]
    EXPECTED_BRANCH = "main"

    @pytest.mark.parametrize("script_name", SCRIPTS_WITH_BRANCH)
    def test_correct_branch(self, script_name):
        p = SCRIPTS_DIR / script_name
        if not p.exists():
            pytest.skip(f"{script_name} not found")
        src = p.read_text()
        if "BRANCH" in src:
            assert self.EXPECTED_BRANCH in src, (
                f"{script_name} has BRANCH constant but not '{self.EXPECTED_BRANCH}'"
            )


# ── Safe-to-fix / safe-to-modify path lists ──────────────────────────────────

class TestSafeFileLists:
    def test_backend_team_safe_to_fix_no_strategy_files(self):
        src = _src("backend_team.py")
        m = re.search(r"SAFE_TO_FIX\s*=\s*\[(.*?)\]", src, re.DOTALL)
        assert m, "SAFE_TO_FIX not found in backend_team.py"
        block = m.group(1)
        # Strategy files must never be auto-fixed (too risky)
        assert "strategies/" not in block, (
            "SAFE_TO_FIX in backend_team.py includes strategy files — too risky"
        )
        assert "ml/" not in block, (
            "SAFE_TO_FIX in backend_team.py includes ML model files — too risky"
        )

    def test_render_autofix_safe_to_modify_no_strategy_files(self):
        src = _src("render_auto_fix.py")
        m = re.search(r"SAFE_TO_MODIFY\s*=\s*\[(.*?)\]", src, re.DOTALL)
        assert m, "SAFE_TO_MODIFY not found in render_auto_fix.py"
        block = m.group(1)
        assert "strategies/" not in block, (
            "SAFE_TO_MODIFY in render_auto_fix.py includes strategy files"
        )


# ── Slack removal guard ──────────────────────────────────────────────────────

class TestSlackStaysRemoved:
    """Slack was removed completely on 2026-07-25 (user directive).

    The autonomous improver edits these scripts unattended, so this guard stops
    a Slack path being reintroduced. That matters beyond tidiness: the old
    `slack_post` helper returned {} whenever SLACK_BOT_TOKEN was unset, so after
    the quota died every message from 27 scripts was silently discarded.
    """

    def test_no_script_calls_the_slack_api(self):
        offenders = [
            name for name in _all_scripts()
            if "slack.com/api" in (SCRIPTS_DIR / name).read_text(errors="ignore")
        ]
        assert not offenders, (
            f"these scripts still call the Slack API: {sorted(offenders)} — "
            "post via notify.post() (Discord) instead"
        )

    def test_no_script_reads_a_slack_token(self):
        offenders = [
            name for name in _all_scripts()
            if "SLACK_BOT_TOKEN" in (SCRIPTS_DIR / name).read_text(errors="ignore")
        ]
        assert not offenders, (
            f"these scripts still read SLACK_BOT_TOKEN: {sorted(offenders)}"
        )

# ── Python syntax validity of all scripts ────────────────────────────────────

class TestScriptSyntax:
    @pytest.mark.parametrize("script_name", _all_scripts())
    def test_script_parses_as_valid_python(self, script_name):
        src = _src(script_name)
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{script_name} has a syntax error: {e}")


# ── claude_conversations.py specific ─────────────────────────────────────────

class TestClaudeConversations:
    def test_get_or_create_channel_defined(self):
        src = _src("claude_conversations.py")
        assert "def get_or_create_channel" in src, (
            "claude_conversations.py must define get_or_create_channel() "
            "so missing channels are auto-created"
        )

    def test_channel_list_paginated(self):
        src = _src("claude_conversations.py")
        m = re.search(r"def get_channel_id.*?(?=\ndef |\Z)", src, re.DOTALL)
        assert m, "get_channel_id not found"
        body = m.group(0)
        assert "cursor" in body, (
            "get_channel_id must paginate with cursor — "
            "a workspace with >200 channels will silently miss channels otherwise"
        )

    def test_all_employee_channels_defined(self):
        src = _src("claude_conversations.py")
        assert "CHANNEL_EMPLOYEES" in src, "CHANNEL_EMPLOYEES not found in claude_conversations.py"
        # Count top-level keys: lines that are '    "channel-name": {' (4-space indent, string key, dict value)
        channel_keys = re.findall(r'^\s{4}"([\w-]+)"\s*:\s*\{', src, re.MULTILINE)
        assert len(channel_keys) >= 10, (
            f"CHANNEL_EMPLOYEES only has {len(channel_keys)} top-level keys — expected at least 10"
        )


# ── slack_bootstrap.py must include all CHANNEL_EMPLOYEES channels ────────────

class TestDiscordChannelCoverage:
    """Every channel the employee router targets must actually be created.

    Was TestSlackBootstrap against slack_bootstrap.py; repointed to
    discord_setup_channels.py when Slack was removed (2026-07-25). The invariant
    is unchanged and still the one that matters: if CHANNEL_EMPLOYEES references
    a channel the setup script never creates, those messages land nowhere.
    """

    def test_setup_includes_required_channels(self):
        setup_src = _src("discord_setup_channels.py")
        conv_src = _src("claude_conversations.py")

        m = re.search(r"CHANNEL_EMPLOYEES\s*=\s*\{(.*?)\n\}", conv_src, re.DOTALL)
        if not m:
            pytest.skip("CHANNEL_EMPLOYEES not found in claude_conversations.py")
        employee_channels = set(re.findall(r'"([\w-]+)"\s*:', m.group(1)))

        # STRUCTURE is {category: [channel, ...]} — collect every listed channel.
        struct = re.search(r"STRUCTURE:[^=]*=\s*\{(.*?)\n\}", setup_src, re.DOTALL)
        assert struct, "STRUCTURE not found in discord_setup_channels.py"
        discord_channels = set(re.findall(r'"([\w-]+)"', struct.group(1)))

        missing = employee_channels - discord_channels
        assert not missing, (
            "discord_setup_channels.py STRUCTURE is missing channels that "
            f"CHANNEL_EMPLOYEES references: {sorted(missing)}"
        )

# ── workflow YAML checks ──────────────────────────────────────────────────────

class TestWorkflowYamls:
    WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
    EXPECTED_BRANCH = "main"

    def _get_workflow_files(self) -> list[Path]:
        return list(self.WORKFLOWS_DIR.glob("*.yml"))

    def test_no_anthropic_key_leaked_in_workflows(self):
        """No workflow must pass a real ANTHROPIC_API_KEY — must always be 'disabled'."""
        violations = []
        for f in self._get_workflow_files():
            src = f.read_text()
            lines = src.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                if "ANTHROPIC_API_KEY:" in line and '"disabled"' not in line and "secrets." not in line:
                    violations.append(f"{f.name}:{i}: {stripped}")
        assert not violations, (
            f"These workflows pass ANTHROPIC_API_KEY without 'disabled' guard:\n"
            + "\n".join(violations)
        )

    def test_no_duplicate_ref_keys_in_workflows(self):
        """Duplicate 'ref:' keys in a single 'with:' block cause silent branch override."""
        violations = []
        for f in self._get_workflow_files():
            src = f.read_text()
            # Find with: blocks and count ref: occurrences
            with_blocks = re.findall(r"with:\s*\n((?:[ \t]+\S.*\n?)*)", src)
            for block in with_blocks:
                ref_count = len(re.findall(r"^\s+ref:", block, re.MULTILINE))
                if ref_count > 1:
                    violations.append(f"{f.name}: {ref_count} ref: keys in one with: block")
        assert not violations, (
            "These workflows have duplicate ref: keys (last one silently wins):\n"
            + "\n".join(violations)
        )

    def test_scheduled_workflows_checkout_correct_branch(self):
        """Workflows triggered by schedule must checkout the working branch, not main."""
        violations = []
        for f in self._get_workflow_files():
            src = f.read_text()
            # Only check scheduled workflows
            if "schedule:" not in src:
                continue
            # Skip workflows intentionally targeting main (deploy notifications)
            if "slack-on-deploy" in f.name:
                continue
            # Check that the correct branch is referenced somewhere in checkout
            if "actions/checkout" in src and self.EXPECTED_BRANCH not in src:
                violations.append(f.name)
        assert not violations, (
            f"These scheduled workflows don't reference branch '{self.EXPECTED_BRANCH}':\n"
            + "\n".join(violations)
        )
