"""
Comprehensive pytest tests for the QuantEdge Slack agent team system.
Tests cover: summon detection, cooldown/dedup, Slack history dedup,
daily reminder, onboarding, provider cascade, employee provider,
answer_agent_summons, state mutation, and channel agent identity.

All tests are fully offline — no real Slack token or LLM keys required.
"""

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Make the scripts directory importable and stub out optional heavy packages
# before importing the module under test.
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Stub langfuse before import so we never need the real package
_lf_stub = types.ModuleType("langfuse")
_lf_dec_stub = types.ModuleType("langfuse.decorators")

def _noop_observe(fn=None, **kw):
    return fn if fn else (lambda f: f)

class _noop_ctx:
    @staticmethod
    def update_current_observation(**kw): pass
    @staticmethod
    def score_current_observation(**kw): pass

_lf_dec_stub.observe = _noop_observe
_lf_dec_stub.langfuse_context = _noop_ctx
sys.modules.setdefault("langfuse", _lf_stub)
sys.modules.setdefault("langfuse.decorators", _lf_dec_stub)

# Stub litellm so it is treated as unavailable (module-level flag _LITELLM_AVAILABLE = False)
_litellm_stub = types.ModuleType("litellm")
_litellm_stub.set_verbose = False
_litellm_stub.drop_params = True
_litellm_stub.completion = MagicMock(side_effect=RuntimeError("litellm not available in tests"))
sys.modules.setdefault("litellm", _litellm_stub)

# Now import the module under test
import slack_agent_team as sat  # noqa: E402


# ===========================================================================
# 1. Agent summon detection (_match_agent_summon)
# ===========================================================================

def test_match_at_agent_prefix():
    result = sat._match_agent_summon("@agent what is the Sharpe ratio?")
    assert result is not None
    assert "Sharpe ratio" in result


def test_match_at_quant_prefix():
    result = sat._match_agent_summon("@quant explain momentum")
    assert result is not None
    assert "momentum" in result


def test_match_ask_colon_prefix():
    result = sat._match_agent_summon("ask: how does Kelly work?")
    assert result is not None
    assert "Kelly" in result


def test_match_double_question_mark_prefix():
    result = sat._match_agent_summon("?? explain walk-forward")
    assert result is not None
    assert "walk-forward" in result


def test_match_agent_mid_sentence():
    # "hello @agent can you help" — @agent appears mid-message
    result = sat._match_agent_summon("hello @agent can you help")
    assert result is not None


def test_match_real_slack_mention_format():
    # Real Slack format: <@U1234567> what is your status?
    result = sat._match_agent_summon("<@U1234567> what is your status?", bot_user_id="U1234567")
    assert result is not None
    assert "status" in result.lower()


def test_match_real_slack_mention_with_username():
    # <@U1234567|username> help me
    result = sat._match_agent_summon("<@U1234567|username> help me", bot_user_id="U1234567")
    assert result is not None
    assert "help" in result.lower()


def test_match_different_bot_user_id():
    # Different bot user ID: <@UABC123> test  with bot_user_id="UABC123"
    result = sat._match_agent_summon("<@UABC123> test", bot_user_id="UABC123")
    assert result is not None
    assert "test" in result.lower()


def test_match_empty_message_returns_none():
    assert sat._match_agent_summon("") is None


def test_match_no_trigger_returns_none():
    assert sat._match_agent_summon("good morning team") is None


def test_match_trigger_with_no_question_returns_none():
    # Just the trigger with nothing after it
    assert sat._match_agent_summon("@agent") is None
    assert sat._match_agent_summon("@agent   ") is None


def test_match_trigger_with_very_short_question_at_start():
    # Exact boundary: question stripped must be non-empty for start-of-message triggers
    # "@agent X" — X is short (1 char) but for start-matching, any non-empty question passes
    result = sat._match_agent_summon("@agent X")
    # The function only requires len > 0 for start-of-message; short is ok there
    # Mid-message requires len > 5. Start-message: returns "X"
    assert result == "X"


def test_match_slack_mention_wrong_user_id():
    # Mention is for a different bot — should NOT match as a @mention-based summon
    result = sat._match_agent_summon("<@UOTHER> help me", bot_user_id="U1234567")
    # The Slack mention won't match, but "help me" alone has no literal trigger
    # so result depends on whether any literal trigger appears — none here
    assert result is None


# ===========================================================================
# 2. Cooldown / dedup (_already_posted)
# ===========================================================================

def test_already_posted_fresh_state_returns_false_and_mutates():
    state = {}
    result = sat._already_posted(state, "engineering", "agent_reminder", 86000)
    assert result is False
    # Must have mutated state
    assert "post_dedup" in state
    assert "engineering:agent_reminder" in state["post_dedup"]
    assert isinstance(state["post_dedup"]["engineering:agent_reminder"], float)


def test_already_posted_within_cooldown_returns_true():
    state = {}
    # First call sets the timestamp
    sat._already_posted(state, "engineering", "agent_reminder", 86000)
    # Second call with fresh-enough timestamp should be True
    result = sat._already_posted(state, "engineering", "agent_reminder", 86000)
    assert result is True


def test_already_posted_after_cooldown_expires_returns_false():
    state = {}
    sat._already_posted(state, "engineering", "agent_reminder", 86000)
    # Backdate the stored timestamp to simulate expiry
    state["post_dedup"]["engineering:agent_reminder"] = time.time() - 90000
    result = sat._already_posted(state, "engineering", "agent_reminder", 86000)
    assert result is False


def test_already_posted_different_content_key_is_independent():
    state = {}
    sat._already_posted(state, "engineering", "agent_reminder", 86000)
    # Different content_key should not be affected
    result = sat._already_posted(state, "engineering", "other_key", 86000)
    assert result is False


def test_already_posted_different_channel_is_independent():
    state = {}
    sat._already_posted(state, "engineering", "agent_reminder", 86000)
    # Different channel should not be affected
    result = sat._already_posted(state, "help", "agent_reminder", 86000)
    assert result is False


# ===========================================================================
# 3. Slack history dedup (_slack_channel_has_recent_bot_post)
# ===========================================================================

def test_slack_history_dedup_found_bot_message():
    """Returns True when a matching bot message exists in history."""
    with patch.object(sat, "get_channel_id", return_value="C123"), \
         patch.object(sat, "slack_call", return_value={
             "ok": True,
             "messages": [
                 {
                     "text": "Welcome to QuantEdge — this is our guide",
                     "bot_id": "BTEST",
                     "username": "QuantEdge Agent Team",
                 }
             ]
         }):
        result = sat._slack_channel_has_recent_bot_post("token", "help", "Welcome to QuantEdge", hours=23.0)
    assert result is True


def test_slack_history_dedup_no_matching_message():
    """Returns False when no matching message in history."""
    with patch.object(sat, "get_channel_id", return_value="C123"), \
         patch.object(sat, "slack_call", return_value={
             "ok": True,
             "messages": [
                 {
                     "text": "Unrelated message about something else",
                     "bot_id": "BTEST",
                 }
             ]
         }):
        result = sat._slack_channel_has_recent_bot_post("token", "help", "Welcome to QuantEdge", hours=23.0)
    assert result is False


def test_slack_history_dedup_channel_id_none():
    """Returns False when channel ID cannot be resolved."""
    with patch.object(sat, "get_channel_id", return_value=None):
        result = sat._slack_channel_has_recent_bot_post("token", "nonexistent", "Some snippet", hours=23.0)
    assert result is False


def test_slack_history_dedup_slack_call_not_ok():
    """Returns False when Slack API returns ok=False."""
    with patch.object(sat, "get_channel_id", return_value="C123"), \
         patch.object(sat, "slack_call", return_value={"ok": False, "error": "channel_not_found"}):
        result = sat._slack_channel_has_recent_bot_post("token", "help", "Welcome to QuantEdge", hours=23.0)
    assert result is False


def test_slack_history_dedup_only_matches_bot_messages():
    """Only matches messages with bot_id or known bot username — not human messages."""
    with patch.object(sat, "get_channel_id", return_value="C123"), \
         patch.object(sat, "slack_call", return_value={
             "ok": True,
             "messages": [
                 {
                     # Human message containing the snippet — no bot_id, not a known bot username
                     "text": "Welcome to QuantEdge everyone!",
                     "user": "U_HUMAN_123",
                 }
             ]
         }):
        result = sat._slack_channel_has_recent_bot_post("token", "help", "Welcome to QuantEdge", hours=23.0)
    assert result is False


def test_slack_history_dedup_matches_known_bot_username():
    """Matches messages from known _BOT_USERNAMES even without bot_id."""
    known_username = next(iter(sat._BOT_USERNAMES))
    with patch.object(sat, "get_channel_id", return_value="C123"), \
         patch.object(sat, "slack_call", return_value={
             "ok": True,
             "messages": [
                 {
                     "text": "Your free AI team is on 24/7",
                     "username": known_username,
                 }
             ]
         }):
        result = sat._slack_channel_has_recent_bot_post(
            "token", "engineering", "Your free AI team is on 24/7", hours=23.0
        )
    assert result is True


# ===========================================================================
# 4. post_daily_agent_reminder — no alpha-research spam
# ===========================================================================

def test_post_daily_reminder_posts_to_engineering_and_help():
    """When not already posted and no recent history, posts to engineering and help."""
    state = {}
    mock_slack = MagicMock(return_value={"ok": True})

    with patch.object(sat, "_already_posted", return_value=False), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=False), \
         patch.object(sat, "slack_call", mock_slack):
        sat.post_daily_agent_reminder("xoxb-test", state)

    # Extract all channels posted to (slack_call is called with positional args:
    # slack_call(token, method, payload) — c[0] holds positional args)
    channels_posted = {c[0][2]["channel"] for c in mock_slack.call_args_list
                       if len(c[0]) >= 3 and isinstance(c[0][2], dict) and c[0][2].get("channel")}
    assert "engineering" in channels_posted
    assert "help" in channels_posted
    # Must NOT post to alpha-research
    assert "alpha-research" not in channels_posted


def test_post_daily_reminder_skips_when_already_posted():
    """When _already_posted returns True, slack_call is NOT called for that channel."""
    state = {}
    mock_slack = MagicMock(return_value={"ok": True})

    with patch.object(sat, "_already_posted", return_value=True), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=False), \
         patch.object(sat, "slack_call", mock_slack):
        sat.post_daily_agent_reminder("xoxb-test", state)

    # No posts should have been made (all channels skipped by _already_posted)
    assert mock_slack.call_count == 0


def test_post_daily_reminder_skips_when_recent_history_found():
    """When _slack_channel_has_recent_bot_post returns True, slack_call is NOT called and state is updated."""
    state = {}
    mock_slack = MagicMock(return_value={"ok": True})

    with patch.object(sat, "_already_posted", return_value=False), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=True), \
         patch.object(sat, "slack_call", mock_slack):
        sat.post_daily_agent_reminder("xoxb-test", state)

    # No actual chat.postMessage calls
    assert mock_slack.call_count == 0
    # State should be updated with the dedup entry
    post_dedup = state.get("post_dedup", {})
    assert "engineering:agent_reminder" in post_dedup or "help:agent_reminder" in post_dedup


# ===========================================================================
# 5. post_engineer_onboarding — weekly + history dedup
# ===========================================================================

def test_onboarding_skips_when_same_week_in_state():
    """If state already has the current week, returns immediately without Slack calls."""
    from datetime import datetime, timezone
    current_week = datetime.now(timezone.utc).strftime("%Y-W%W")
    state = {"onboarding_posted_week": current_week}
    mock_slack = MagicMock()

    with patch.object(sat, "slack_call", mock_slack), \
         patch.object(sat, "get_channel_id", return_value="C_HELP"):
        sat.post_engineer_onboarding("xoxb-test", state)

    assert mock_slack.call_count == 0


def test_onboarding_skips_when_history_has_recent_post():
    """With empty state + history check True, marks state and does not post."""
    state = {}
    mock_slack = MagicMock(return_value={"ok": True})

    with patch.object(sat, "get_channel_id", return_value="C_HELP"), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=True), \
         patch.object(sat, "slack_call", mock_slack):
        sat.post_engineer_onboarding("xoxb-test", state)

    # No actual post
    assert mock_slack.call_count == 0
    # State should be marked
    from datetime import datetime, timezone
    current_week = datetime.now(timezone.utc).strftime("%Y-W%W")
    assert state.get("onboarding_posted_week") == current_week


def test_onboarding_posts_when_state_empty_and_no_history():
    """With empty state and no recent history, posts to #help and marks state."""
    state = {}
    mock_slack = MagicMock(return_value={"ok": True})

    with patch.object(sat, "get_channel_id", return_value="C_HELP"), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=False), \
         patch.object(sat, "slack_call", mock_slack):
        sat.post_engineer_onboarding("xoxb-test", state)

    # Should have posted (at least one slack_call)
    assert mock_slack.call_count >= 1
    # Check that a chat.postMessage was issued with the help channel
    calls_to_post = [c for c in mock_slack.call_args_list
                     if c[0][1] == "chat.postMessage"]
    assert len(calls_to_post) >= 1

    # State should now have the week set
    from datetime import datetime, timezone
    current_week = datetime.now(timezone.utc).strftime("%Y-W%W")
    assert state.get("onboarding_posted_week") == current_week


# ===========================================================================
# 6. Provider cascade order (call_best_agent)
# ===========================================================================

def test_call_best_agent_uses_github_models_first():
    """When call_github_models returns a valid string, it is used as the response."""
    valid_response = "This is a detailed answer about the Sharpe ratio and momentum strategies."
    with patch.object(sat, "call_github_models", return_value=valid_response) as mock_gh:
        result = sat.call_best_agent("test question")
    mock_gh.assert_called_once()
    assert result == valid_response.strip()


def test_call_best_agent_falls_through_when_github_models_returns_none():
    """When call_github_models returns None, falls through to call_gemini."""
    gemini_response = "This is a quality gemini answer about walk-forward validation in quant strategies."
    with patch.object(sat, "call_github_models", return_value=None), \
         patch.object(sat, "call_gemini", return_value=gemini_response) as mock_gemini:
        result = sat.call_best_agent("test question")
    mock_gemini.assert_called_once()
    assert result == gemini_response.strip()


def test_call_best_agent_returns_none_when_all_exhausted():
    """When all providers return None, returns None."""
    with patch.object(sat, "call_github_models", return_value=None), \
         patch.object(sat, "call_gemini", return_value=None), \
         patch.object(sat, "_groq_key_shared", return_value=None), \
         patch.object(sat, "_try_openai_compat", return_value=None), \
         patch.object(sat, "_employee_keys", return_value=[]):
        result = sat.call_best_agent("test question")
    assert result is None


# ===========================================================================
# 7. Employee provider (employee_provider_prompt)
# ===========================================================================

def _make_quality_response():
    return (
        "The LSTM in backend/app/ml/models/lstm_model.py shows 1.8 val_sharpe vs 3.2 "
        "train_sharpe — reduce d_model from 64->32 and increase dropout 0.1->0.25. "
        "XGBoost on multi-asset shows better OOS stability at 1.4 Sharpe."
    )


def test_employee_provider_maya_high_quality():
    """Maya with score >= 7 returns the result."""
    good_response = _make_quality_response()
    with patch.object(sat, "call_best_agent_for_task", return_value=(good_response, "GitHub Models")), \
         patch.object(sat, "score_agent_output", return_value=(8, "Specific and actionable")), \
         patch.object(sat, "check_for_hallucination", return_value=False):
        result_text, provider = sat.employee_provider_prompt("maya", "What is CI health?")
    assert result_text is not None
    assert "LSTM" in result_text or "lstm" in result_text.lower() or good_response[:30] in result_text


def test_employee_provider_aarav_high_quality():
    """Aarav with score >= 7 returns the result."""
    good_response = _make_quality_response()
    with patch.object(sat, "call_best_agent_for_task", return_value=(good_response, "GitHub Models")), \
         patch.object(sat, "score_agent_output", return_value=(8, "Good quality")), \
         patch.object(sat, "check_for_hallucination", return_value=False):
        result_text, provider = sat.employee_provider_prompt("aarav", "Review strategy performance")
    assert result_text is not None


def test_employee_provider_sara_high_quality():
    """Sara with score >= 7 returns the result."""
    good_response = _make_quality_response()
    with patch.object(sat, "call_best_agent_for_task", return_value=(good_response, "GitHub Models")), \
         patch.object(sat, "score_agent_output", return_value=(8, "Good quality")), \
         patch.object(sat, "check_for_hallucination", return_value=False):
        result_text, provider = sat.employee_provider_prompt("sara", "Compare models")
    assert result_text is not None


def test_employee_provider_low_quality_triggers_retry():
    """When score < 7, it retries with an enhanced prompt."""
    good_response = _make_quality_response()
    better_response = good_response + " Additional specifics: d_model=32."
    call_count = {"n": 0}

    def mock_call_best(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ("Too generic, no specifics provided.", "GitHub Models")
        else:
            return (better_response, "Gemini")

    def mock_score(text, *args, **kwargs):
        if "Too generic" in text:
            return (4, "Generic output")
        return (8, "Specific")

    with patch.object(sat, "call_best_agent_for_task", side_effect=mock_call_best), \
         patch.object(sat, "score_agent_output", side_effect=mock_score), \
         patch.object(sat, "check_for_hallucination", return_value=False):
        result_text, provider = sat.employee_provider_prompt("maya", "Analyze CI")

    # Should have been called at least twice (initial + retry)
    assert call_count["n"] >= 2


def test_employee_provider_returns_none_when_no_result():
    """Returns (None, None) when call_best_agent_for_task returns None."""
    with patch.object(sat, "call_best_agent_for_task", return_value=(None, "exhausted")):
        result_text, provider = sat.employee_provider_prompt("maya", "What is CI health?")
    assert result_text is None
    assert provider is None


# ===========================================================================
# 8. answer_agent_summons — no template spam
# ===========================================================================

def _make_summon(channel="alpha-research", thread_ts="1234567.000", question="What is Sharpe?"):
    return {
        "channel_name": channel,
        "channel_id": "C_TEST",
        "thread_ts": thread_ts,
        "user": "U_HUMAN",
        "question": question,
    }


def test_answer_summons_posts_llm_answer():
    """When call_best_agent returns a real answer, posts it (not a template)."""
    real_answer = (
        "The Sharpe ratio in backend/app/risk/kelly.py is computed as annualized return "
        "divided by annualized volatility. Our momentum strategy shows 1.6 Sharpe on SPY 1d walk-forward."
    )
    summons = [_make_summon()]
    state = {"replied_to": []}

    mock_post = MagicMock(return_value={"ok": True, "ts": "9999999.000"})
    with patch.object(sat, "call_best_agent", return_value=real_answer), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        count = sat.answer_agent_summons("xoxb-test", summons, state)

    assert count == 1
    posted_text = mock_post.call_args[0][2]
    assert "Sharpe" in posted_text or real_answer[:30] in posted_text
    # Confirm ts is recorded
    assert "1234567.000" in state["replied_to"]


def test_answer_summons_skips_already_replied_ts():
    """If ts is already in state['replied_to'], skip it."""
    summons = [_make_summon(thread_ts="ALREADY_REPLIED")]
    state = {"replied_to": ["ALREADY_REPLIED"]}

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value="Some answer"), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        count = sat.answer_agent_summons("xoxb-test", summons, state)

    assert count == 0
    mock_post.assert_not_called()


def test_answer_summons_posts_api_limit_fallback_when_llm_fails():
    """When call_best_agent returns None, posts the API-limit fallback message, not an onboarding template."""
    summons = [_make_summon()]
    state = {"replied_to": []}

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value=None), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    mock_post.assert_called_once()
    fallback_text = mock_post.call_args[0][2]
    # Should contain an API-limit message, NOT onboarding boilerplate
    assert "free-tier" in fallback_text.lower() or "api limit" in fallback_text.lower() or "throttled" in fallback_text.lower()
    # Definitely should NOT be the onboarding guide
    assert "Welcome to QuantEdge" not in fallback_text
    assert "1234567.000" in state["replied_to"]


def test_answer_summons_records_ts_in_replied_to():
    """Verifies replied_to state mutation."""
    summons = [_make_summon(thread_ts="TS_NEW_1"), _make_summon(thread_ts="TS_NEW_2")]
    state = {"replied_to": []}
    good_answer = "This is a detailed answer about momentum and Sharpe ratios in our strategy files."

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value=good_answer), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    assert "TS_NEW_1" in state["replied_to"]
    assert "TS_NEW_2" in state["replied_to"]


# ===========================================================================
# 9. State mutation is correct
# ===========================================================================

def test_already_posted_stores_correct_key_format():
    """_already_posted stores state['post_dedup']['channel:key'] = timestamp."""
    state = {}
    sat._already_posted(state, "engineering", "my_key", 3600)
    store = state["post_dedup"]
    assert "engineering:my_key" in store
    ts_stored = store["engineering:my_key"]
    assert abs(ts_stored - time.time()) < 5  # within 5 seconds of now


def test_answer_summons_appends_to_replied_to():
    """answer_agent_summons appends ts to state['replied_to']."""
    summons = [_make_summon(thread_ts="MUT_TS_1")]
    state = {}  # no replied_to key initially
    good_answer = "Detailed LSTM analysis in backend/app/ml/models/lstm_model.py — val_sharpe 1.8 vs train 3.2."

    with patch.object(sat, "call_best_agent", return_value=good_answer), \
         patch.object(sat, "post_to_slack", return_value={"ok": True}), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    assert "replied_to" in state
    assert "MUT_TS_1" in state["replied_to"]


def test_onboarding_sets_state_week_key():
    """post_engineer_onboarding sets state['onboarding_posted_week']."""
    state = {}
    with patch.object(sat, "get_channel_id", return_value="C_HELP"), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=False), \
         patch.object(sat, "slack_call", return_value={"ok": True}):
        sat.post_engineer_onboarding("xoxb-test", state)

    from datetime import datetime, timezone
    current_week = datetime.now(timezone.utc).strftime("%Y-W%W")
    assert state.get("onboarding_posted_week") == current_week


def test_already_posted_then_reads_same_value():
    """After _already_posted mutates state, subsequent call reads the stored value correctly."""
    state = {}
    # First call: sets timestamp
    first_result = sat._already_posted(state, "help", "onboarding", 3600)
    assert first_result is False

    # Manually verify stored timestamp
    stored_ts = state["post_dedup"]["help:onboarding"]
    assert abs(stored_ts - time.time()) < 5

    # Second call: should return True (still within cooldown)
    second_result = sat._already_posted(state, "help", "onboarding", 3600)
    assert second_result is True


# ===========================================================================
# 10. Channel agent identity (_CHANNEL_AGENT_IDENTITY)
# ===========================================================================

def test_identity_alpha_research():
    name, emoji = sat._CHANNEL_AGENT_IDENTITY.get("alpha-research", ("QuantEdge Agent", ":robot_face:"))
    assert name == "Alpha Research Director"
    assert emoji == ":chart_with_upwards_trend:"


def test_identity_engineering():
    name, emoji = sat._CHANNEL_AGENT_IDENTITY.get("engineering", ("QuantEdge Agent", ":robot_face:"))
    assert name == "VP Engineering"
    assert emoji == ":woman_office_worker:"


def test_identity_desk_crypto():
    name, emoji = sat._CHANNEL_AGENT_IDENTITY.get("desk-crypto", ("QuantEdge Agent", ":robot_face:"))
    assert name == "Crypto desk bot"
    assert emoji == ":coin:"


def test_identity_unknown_channel_falls_back_to_default():
    """Unknown channels fall back to the default identity in answer_agent_summons."""
    name, emoji = sat._CHANNEL_AGENT_IDENTITY.get("totally-unknown-channel-xyz", ("QuantEdge Agent", ":robot_face:"))
    assert name == "QuantEdge Agent"
    assert emoji == ":robot_face:"


def test_identity_used_in_answer_summons():
    """answer_agent_summons calls post_to_slack with the correct agent identity for the channel."""
    summons = [_make_summon(channel="desk-crypto")]
    state = {"replied_to": []}
    answer = "Funding rate carry on Binance via CCXT in backend/app/strategies/manual/triangular_arb.py shows 0.8% annualized per day."

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value=answer), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    mock_post.assert_called_once()
    kwargs = mock_post.call_args[1]
    assert kwargs.get("username") == "Crypto desk bot"
    assert kwargs.get("icon_emoji") == ":coin:"


def test_identity_used_for_engineering_channel():
    """answer_agent_summons uses VP Engineering identity for #engineering."""
    summons = [_make_summon(channel="engineering")]
    state = {"replied_to": []}
    answer = "CI test coverage in backend/tests/ is at 87%. The failing test is test_risk_engine.py line 45."

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value=answer), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    mock_post.assert_called_once()
    kwargs = mock_post.call_args[1]
    assert kwargs.get("username") == "VP Engineering"
    assert kwargs.get("icon_emoji") == ":woman_office_worker:"


# ===========================================================================
# Additional edge-case / integration tests
# ===========================================================================

def test_match_agent_summon_triggers_list():
    """All _AGENT_SUMMON_TRIGGERS are handled by _match_agent_summon."""
    for trig in sat._AGENT_SUMMON_TRIGGERS:
        msg = f"{trig}explain the Sharpe ratio in detail"
        result = sat._match_agent_summon(msg)
        assert result is not None, f"Trigger '{trig}' should have matched but returned None"


def test_bot_usernames_are_set():
    """_BOT_USERNAMES is a non-empty set."""
    assert isinstance(sat._BOT_USERNAMES, (set, frozenset))
    assert len(sat._BOT_USERNAMES) > 0


def test_channel_agent_identity_dict_completeness():
    """Key channels present in _CHANNEL_AGENT_IDENTITY."""
    required = ["engineering", "alpha-research", "desk-crypto", "help", "ml-experiments"]
    for ch in required:
        assert ch in sat._CHANNEL_AGENT_IDENTITY, f"#{ch} missing from _CHANNEL_AGENT_IDENTITY"


def test_already_posted_zero_cooldown_always_false():
    """With zero cooldown, _already_posted always returns False (time.time() - 0 > 0)."""
    state = {}
    result = sat._already_posted(state, "ch", "key", cooldown_seconds=0)
    # time.time() - stored_ts will be >= 0; if exactly 0, might pass. But typically > 0.
    # The second call will have the stored ts equal to now, so time.time()-ts ~= 0 < 0 is False.
    # Actually cooldown=0 means any existing ts >= 0 seconds old passes, so second call returns False too.
    # The first call always sets it and returns False.
    assert result is False


def test_answer_summons_alpha_research_uses_correct_identity():
    """#alpha-research summons use Alpha Research Director identity."""
    summons = [_make_summon(channel="alpha-research")]
    state = {"replied_to": []}
    answer = "Momentum in backend/app/strategies/manual/momentum.py shows 1.6 Sharpe on SPY walk-forward 2021-2024."

    mock_post = MagicMock(return_value={"ok": True})
    with patch.object(sat, "call_best_agent", return_value=answer), \
         patch.object(sat, "post_to_slack", mock_post), \
         patch.object(sat, "_build_summon_context", return_value=""):
        sat.answer_agent_summons("xoxb-test", summons, state)

    kwargs = mock_post.call_args[1]
    assert kwargs.get("username") == "Alpha Research Director"
    assert kwargs.get("icon_emoji") == ":chart_with_upwards_trend:"


def test_post_daily_reminder_never_posts_to_alpha_research():
    """post_daily_agent_reminder never posts to #alpha-research regardless of state."""
    state = {}
    posted_channels = []

    def capture_slack(token, method, payload):
        if method == "chat.postMessage":
            posted_channels.append(payload.get("channel", ""))
        return {"ok": True}

    with patch.object(sat, "_already_posted", return_value=False), \
         patch.object(sat, "_slack_channel_has_recent_bot_post", return_value=False), \
         patch.object(sat, "slack_call", side_effect=capture_slack):
        sat.post_daily_agent_reminder("xoxb-test", state)

    assert "alpha-research" not in posted_channels


def test_call_best_agent_for_task_github_models_first():
    """call_best_agent_for_task tries GitHub Models first before any other provider."""
    valid = "Detailed answer about LSTM walk-forward validation and backtest_signals() in FastAPI."
    call_order = []

    def record_gh(*args, **kwargs):
        call_order.append("github_models")
        return valid

    with patch.object(sat, "call_github_models", side_effect=record_gh), \
         patch.object(sat, "call_gemini_with_key", MagicMock(return_value=None)):
        result, provider = sat.call_best_agent_for_task("quant", "test prompt")

    assert call_order[0] == "github_models"
    assert result is not None
    assert provider == "GitHub Models"
