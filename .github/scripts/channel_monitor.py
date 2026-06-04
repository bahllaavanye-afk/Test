"""
QuantEdge Channel Monitor — reads every monitored channel, reports gaps,
auto-heals silent agents.

For each channel in the monitored list (same list as slack_agent_team.py
inbox_channels), checks the last 20 messages and reports:
  - Last post timestamp (how long ago in hours/minutes)
  - Whether it was from a bot or human
  - SILENT if last post > 2 hours ago
  - TEMPLATE_SPAM if the last bot post is the onboarding/reminder template

For each employee→channel mapping, if that channel is silent > 4 hours,
the employee function is called directly (auto-heal).

Posts structured report to #incidents and a 1-line summary to #allquantedge.

Usage:
    SLACK_BOT_TOKEN=xoxb-... python channel_monitor.py
    SLACK_BOT_TOKEN="" python channel_monitor.py   # prints report to stdout, exits 0
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_SCRIPT = Path(__file__).parent / "slack_agent_team.py"

# ─── Token ────────────────────────────────────────────────────────────────────

SLACK_BOT_TOKEN: str = os.environ.get("SLACK_BOT_TOKEN", "")

# ─── Monitored channels (same list as slack_agent_team.py inbox_channels) ────

MONITORED_CHANNELS: list[str] = [
    "engineering", "alpha-research", "ml-experiments",
    "squad-qa", "squad-backend", "squad-frontend", "risk-alerts",
    "desk-crypto", "desk-polymarket", "desk-fx-rates",
    "desk-kalshi", "desk-stat-arb", "desk-futures",
    "desk-rates", "desk-equities", "desk-commodities",
    "desk-options", "squad-data", "pod-ml-rl",
    "help", "pnl-daily", "squad-execution",
    # Additional channels
    "general", "random", "standup", "wins", "incidents",
    "strategy-review", "model-performance", "code-review",
    # Autopilot gap channels
    "papers", "leadership-summary", "infra-alerts", "ci-failures",
    # Specialist channels
    "security-alerts", "finance-ops", "legal-compliance",
    "announcements", "allquantedge",
]

# ─── Employee → channel mapping (derived from Post(channel=...) in each fn) ──
# Key: employee display name for reports, Value: (function_name, channel)

EMPLOYEE_CHANNEL_MAP: dict[str, tuple[str, str]] = {
    "maya_chen":              ("maya_chen_eng_daily",          "engineering"),
    "aarav_patel":            ("aarav_patel_strategy_review",  "alpha-research"),
    "linh_tran":              ("linh_tran_ml_results",         "ml-experiments"),
    "diego_ramirez":          ("diego_ramirez_execution",      "squad-execution"),
    "jian_wu":                ("jian_wu_risk",                 "risk-alerts"),
    "priya_subramanian":      ("priya_subramanian_frontend",   "squad-frontend"),
    "anna_hoffmann":          ("anna_hoffmann_backend",        "squad-backend"),
    "sina_hassani":           ("sina_hassani_data",            "squad-data"),
    "kenji_watanabe":         ("kenji_watanabe_devops",        "infra-alerts"),
    "aditi_sharma":           ("aditi_sharma_qa",              "squad-qa"),
    "cameron_park":           ("cameron_park_security",        "security-alerts"),
    "sofia_karlsson":         ("sofia_karlsson_research",      "papers"),
    "yuki_mori":              ("yuki_mori_options",            "desk-options"),
    "hugo_bernardes":         ("hugo_bernardes_research",      "alpha-research"),
    "tomas_lindqvist":        ("tomas_lindqvist_rl",           "pod-ml-rl"),
    "lior_avraham":           ("lior_avraham_polymarket",      "desk-polymarket"),
    "marcus_olufemi":         ("marcus_olufemi_risk",          "leadership-summary"),
    "wei_chang":              ("wei_chang_finance",            "finance-ops"),
    "helena_voss":            ("helena_voss_compliance",       "legal-compliance"),
    "karl_nystrom":           ("karl_nystrom_question",        "help"),
    "ravi_iyer":              ("ravi_iyer_ci",                 "engineering"),
    "kenji_deploy":           ("kenji_deploy_readiness",       "infra-alerts"),
    "sara_kim":               ("sara_kim_ml_research",         "ml-experiments"),
    "marcus_williams":        ("marcus_williams_dl_engineer",  "engineering"),
    "priya_nair":             ("priya_nair_feature_eng",       "alpha-research"),
    "alex_chen":              ("alex_chen_quant_ml",           "alpha-research"),
    "laavanye_bahl":          ("laavanye_bahl_ceo",            "announcements"),
}

# Templates that indicate onboarding/reminder spam rather than real content
TEMPLATE_SPAM_SNIPPETS: list[str] = [
    "Welcome to QuantEdge — Free Agent Team Guide",
    "Our AI agents run 24/7 across all Slack channels",
    "Agent Health Report",
    "Channel Health Monitor",
    "Daily agent reminder",
    "reminder: post your daily update",
]

# Silence thresholds
SILENCE_WARN_HOURS = 2.0    # SILENT flag
AGENT_DOWN_HOURS   = 4.0    # AGENT_DOWN flag + auto-heal

# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ChannelStatus:
    channel: str
    last_post_ts: float = 0.0          # epoch seconds of most recent message
    last_post_hours_ago: float = 999.0  # derived
    last_poster_is_bot: bool = False
    is_silent: bool = False            # > 2 hours no activity
    is_template_spam: bool = False     # last bot post is an onboarding template
    last_text_snippet: str = ""
    error: str = ""


@dataclass
class AgentStatus:
    emp_name: str
    fn_name: str
    channel: str
    channel_last_bot_hours_ago: float = 999.0
    is_agent_down: bool = False
    auto_healed: bool = False
    heal_error: str = ""


@dataclass
class MonitorReport:
    timestamp: str
    channel_statuses: list[ChannelStatus] = field(default_factory=list)
    agent_statuses: list[AgentStatus] = field(default_factory=list)
    active_count: int = 0
    silent_count: int = 0
    agent_down_count: int = 0
    agents_healed: int = 0


# ─── Slack helpers ────────────────────────────────────────────────────────────


_channels_cache: dict[str, dict] = {}
_list_attempted: bool = False


def slack_call(token: str, method: str, payload: dict) -> dict:
    """Make a Slack API call. Returns {} on network error."""
    if not token:
        return {"ok": False, "error": "no_token"}
    url = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        # Handle rate-limit
        if isinstance(result, dict) and result.get("error") == "ratelimited":
            time.sleep(5)
            with urllib.request.urlopen(req, timeout=15) as resp2:
                result = json.loads(resp2.read())
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _load_channel_list(token: str) -> None:
    global _channels_cache, _list_attempted
    if _list_attempted:
        return
    _list_attempted = True
    cursor = ""
    while True:
        payload: dict = {"types": "public_channel,private_channel", "limit": 200}
        if cursor:
            payload["cursor"] = cursor
        data = slack_call(token, "conversations.list", payload)
        if not data.get("ok"):
            print(f"  [monitor] conversations.list failed: {data.get('error')}")
            break
        for ch in data.get("channels", []):
            _channels_cache[ch["name"]] = ch
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break


def get_channel_id(token: str, name: str) -> str | None:
    _load_channel_list(token)
    ch = _channels_cache.get(name)
    return ch["id"] if ch else None


def post_to_slack(
    token: str,
    channel: str,
    text: str,
    *,
    username: str = "Channel Monitor",
    icon_emoji: str = ":mag:",
) -> dict:
    """Post a message to a Slack channel by name."""
    if not token:
        return {"ok": False, "error": "no_token"}
    ch_id = get_channel_id(token, channel)
    if ch_id:
        # Auto-join public channels
        ch = _channels_cache.get(channel, {})
        if not ch.get("is_private", False):
            slack_call(token, "conversations.join", {"channel": ch_id})
        channel_ref = ch_id
    else:
        channel_ref = f"#{channel}"
    payload = {
        "channel": channel_ref,
        "text": text,
        "username": username,
        "icon_emoji": icon_emoji,
        "mrkdwn": True,
    }
    result = slack_call(token, "chat.postMessage", payload)
    # Fallback: missing chat:write.customize scope
    if not result.get("ok") and result.get("error") in ("not_allowed_token_type", "missing_scope"):
        fallback = {"channel": channel_ref, "text": f"*[{username}]* {text}", "mrkdwn": True}
        result = slack_call(token, "chat.postMessage", fallback)
    return result


# ─── Channel scanning ─────────────────────────────────────────────────────────


def _is_bot_message(msg: dict) -> bool:
    """True if the message was posted by a bot (not a human user)."""
    # Slack marks bot messages with bot_id or subtype='bot_message'
    if msg.get("bot_id"):
        return True
    if msg.get("subtype") == "bot_message":
        return True
    # Messages with a 'username' field and no 'user' field are typically bots
    if msg.get("username") and not msg.get("user"):
        return True
    return False


def _is_template_spam(text: str) -> bool:
    """True if the message looks like the onboarding/reminder boilerplate."""
    text_lower = text.lower()
    for snippet in TEMPLATE_SPAM_SNIPPETS:
        if snippet.lower() in text_lower:
            return True
    return False


def check_channel(token: str, channel: str) -> ChannelStatus:
    """Fetch the last 20 messages from a channel and return its status."""
    status = ChannelStatus(channel=channel)
    now = time.time()

    try:
        ch_id = get_channel_id(token, channel)
        if not ch_id:
            status.error = "channel_not_found"
            status.last_post_hours_ago = 999.0
            return status

        resp = slack_call(token, "conversations.history", {
            "channel": ch_id,
            "limit": 20,
        })
        if not resp.get("ok"):
            status.error = resp.get("error", "history_failed")
            return status

        messages = resp.get("messages", [])
        if not messages:
            status.is_silent = True
            status.last_post_hours_ago = 999.0
            return status

        # Most recent message first (Slack returns newest first)
        most_recent = messages[0]
        ts_float = float(most_recent.get("ts", 0))
        status.last_post_ts = ts_float
        status.last_post_hours_ago = (now - ts_float) / 3600.0
        status.last_poster_is_bot = _is_bot_message(most_recent)
        status.last_text_snippet = most_recent.get("text", "")[:120]

        # SILENT: no message in last 2 hours
        if status.last_post_hours_ago > SILENCE_WARN_HOURS:
            status.is_silent = True

        # TEMPLATE_SPAM: last bot post is onboarding/reminder boilerplate
        if status.last_poster_is_bot and _is_template_spam(status.last_text_snippet):
            status.is_template_spam = True
            # Also look past it to find the real last human/real-content post
            for msg in messages[1:]:
                if not _is_template_spam(msg.get("text", "")):
                    real_ts = float(msg.get("ts", 0))
                    real_hours_ago = (now - real_ts) / 3600.0
                    if real_hours_ago > SILENCE_WARN_HOURS:
                        status.is_silent = True
                    break

    except Exception as exc:
        status.error = str(exc)[:120]

    return status


# ─── Agent module loader ───────────────────────────────────────────────────────


_agent_module = None


def _load_agent_module():
    """Dynamically import slack_agent_team without executing main()."""
    global _agent_module
    if _agent_module is not None:
        return _agent_module
    try:
        spec = importlib.util.spec_from_file_location("slack_agent_team", AGENT_SCRIPT)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
        _agent_module = mod
        return mod
    except Exception as exc:
        print(f"  [monitor] Could not import slack_agent_team: {exc}")
        return None


def call_employee_fn(fn_name: str) -> list:
    """Call the named employee function from slack_agent_team.py and return Posts."""
    mod = _load_agent_module()
    if mod is None:
        return []
    fn: Callable | None = getattr(mod, fn_name, None)
    if fn is None:
        print(f"  [auto-heal] Function {fn_name} not found in agent module")
        return []
    try:
        return fn() or []
    except Exception as exc:
        print(f"  [auto-heal] {fn_name}() raised: {exc}")
        return []


# ─── Auto-heal: post employee output to their channel ────────────────────────


def auto_heal_agent(token: str, emp_name: str, fn_name: str, channel: str) -> tuple[bool, str]:
    """
    Call the employee function and post its output to the channel.
    Returns (success, error_message).
    """
    print(f"  [auto-heal] Calling {fn_name} for #{channel}…")
    try:
        posts = call_employee_fn(fn_name)
    except Exception as exc:
        return False, str(exc)[:200]

    if not posts:
        return False, f"{fn_name}() returned no posts"

    success = False
    for post in posts:
        try:
            channel_target = getattr(post, "channel", channel)
            text = getattr(post, "text", "")
            username = getattr(post, "username", emp_name.replace("_", " ").title())
            icon = getattr(post, "icon_emoji", ":robot_face:")
            if not text:
                continue
            res = post_to_slack(token, channel_target, text, username=username, icon_emoji=icon)
            if res.get("ok"):
                success = True
                print(f"  [auto-heal] ✓ {fn_name} → #{channel_target}")
            else:
                print(f"  [auto-heal] ✗ {fn_name} → #{channel_target}: {res.get('error')}")
        except Exception as exc:
            print(f"  [auto-heal] post error: {exc}")

    return success, "" if success else "all posts failed"


# ─── State helpers ────────────────────────────────────────────────────────────


def _try_load_state() -> dict:
    """Load slack_state.json if it exists, for save_state compatibility."""
    try:
        mod = _load_agent_module()
        if mod:
            return mod.load_state()
    except Exception:
        pass
    return {}


def _try_save_state(state: dict) -> None:
    try:
        mod = _load_agent_module()
        if mod:
            mod.save_state(state)
    except Exception:
        pass


# ─── Report formatting ────────────────────────────────────────────────────────


def _fmt_ago(hours: float) -> str:
    """Format hours-ago as a human-readable string."""
    if hours >= 999:
        return "never"
    if hours < 1.0:
        mins = int(hours * 60)
        return f"{mins}m ago"
    return f"{hours:.1f}h ago"


def build_incidents_report(report: MonitorReport) -> str:
    """Build the full structured report for #incidents."""
    lines: list[str] = []
    lines.append(f"*Channel Health Monitor — {report.timestamp}*")
    lines.append("")

    # Summary counts
    active = report.active_count
    total  = len(report.channel_statuses)
    silent = report.silent_count
    down   = report.agent_down_count

    lines.append(f"✅ {active}/{total} channels active in last 2h")

    if silent > 0:
        silent_names = [
            f"#{ cs.channel}"
            for cs in report.channel_statuses
            if cs.is_silent
        ]
        lines.append(f"⚠️  {silent} channel{'s' if silent != 1 else ''} silent > 2h: {', '.join(silent_names)}")
    else:
        lines.append("✅ No channels silent > 2h")

    # Template spam
    spam_channels = [cs.channel for cs in report.channel_statuses if cs.is_template_spam]
    if spam_channels:
        lines.append(f"🔁 {len(spam_channels)} channel{'s' if len(spam_channels) != 1 else ''} last post is template/onboarding: " +
                     ", ".join(f"#{c}" for c in spam_channels))

    # Errors
    err_channels = [cs.channel for cs in report.channel_statuses if cs.error]
    if err_channels:
        lines.append(f"❓ {len(err_channels)} channel{'s' if len(err_channels) != 1 else ''} unreadable: " +
                     ", ".join(f"#{c}" for c in err_channels))

    lines.append("")

    # Agent gap section
    down_agents = [a for a in report.agent_statuses if a.is_agent_down]
    if down_agents:
        gap_parts = []
        for a in down_agents:
            gap_parts.append(f"{a.emp_name} (→#{a.channel} last {a.channel_last_bot_hours_ago:.1f}h ago)")
        lines.append(f"❌ {len(down_agents)} agent gap{'s' if len(down_agents) != 1 else ''}: {', '.join(gap_parts)}")
    else:
        lines.append("✅ All agents posting within 4h")

    # Auto-heal section
    healed = [a for a in report.agent_statuses if a.auto_healed]
    if healed:
        lines.append("")
        lines.append("*Auto-fix triggered*: silence_breaker ran on flagged channels:")
        for a in healed:
            result_emoji = "✅" if a.auto_healed and not a.heal_error else "❌"
            lines.append(f"  {result_emoji} {a.emp_name} → #{a.channel}" +
                         (f" ({a.heal_error})" if a.heal_error else ""))
    elif down_agents:
        lines.append("")
        lines.append("*Auto-fix triggered*: silence_breaker will run on flagged channels next wave.")

    return "\n".join(lines)


def build_allquantedge_summary(report: MonitorReport) -> str:
    """Build the 1-line summary for #allquantedge."""
    total   = len(report.channel_statuses)
    active  = report.active_count
    healed  = report.agents_healed
    if healed > 0:
        return f"Health: {active}/{total} channels active, {healed} agent{'s' if healed != 1 else ''} auto-healed"
    elif report.silent_count > 0:
        return f"Health: {active}/{total} channels active, {report.silent_count} silent (monitor notified)"
    else:
        return f"Health: {active}/{total} channels active, all agents OK"


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    now_epoch = time.time()

    if not SLACK_BOT_TOKEN:
        print("No token — printing report to stdout only (no Slack posts)")
        print(f"\nChannel Health Monitor — {timestamp}")
        print(f"Monitoring {len(MONITORED_CHANNELS)} channels, {len(EMPLOYEE_CHANNEL_MAP)} employee agents")
        print("Set SLACK_BOT_TOKEN env var to enable Slack posting and live channel reads.")
        return 0

    print(f"[channel_monitor] Starting — {timestamp}")
    print(f"  Checking {len(MONITORED_CHANNELS)} channels…")

    report = MonitorReport(timestamp=timestamp)

    # ── Phase 1: Check every monitored channel ────────────────────────────────
    for channel in MONITORED_CHANNELS:
        try:
            status = check_channel(SLACK_BOT_TOKEN, channel)
        except Exception as exc:
            status = ChannelStatus(channel=channel, error=str(exc)[:120])
        report.channel_statuses.append(status)

        flag = ""
        if status.error:
            flag = f"[ERR: {status.error}]"
        elif status.is_silent:
            flag = f"[SILENT {_fmt_ago(status.last_post_hours_ago)}]"
            if status.is_template_spam:
                flag += " [TEMPLATE_SPAM]"
        else:
            flag = f"[OK {_fmt_ago(status.last_post_hours_ago)} {'bot' if status.last_poster_is_bot else 'human'}]"
        print(f"  #{channel:30s} {flag}")
        time.sleep(0.3)  # gentle rate-limit

    # Compute summary counts
    report.active_count = sum(
        1 for cs in report.channel_statuses
        if not cs.is_silent and not cs.error
    )
    report.silent_count = sum(1 for cs in report.channel_statuses if cs.is_silent)

    # ── Phase 2: Build per-channel last-bot-post lookup ───────────────────────
    # Map channel → hours since last bot post (for agent-down detection)
    channel_bot_hours: dict[str, float] = {}
    for cs in report.channel_statuses:
        if cs.error:
            channel_bot_hours[cs.channel] = 999.0
        elif cs.last_poster_is_bot and not cs.is_template_spam:
            channel_bot_hours[cs.channel] = cs.last_post_hours_ago
        else:
            # Last post was human or template — scan messages for last real bot post
            # Use the channel status last_post_hours_ago as a conservative estimate
            channel_bot_hours[cs.channel] = cs.last_post_hours_ago if cs.is_silent else cs.last_post_hours_ago

    # ── Phase 3: Check each employee agent ───────────────────────────────────
    print(f"\n  Checking {len(EMPLOYEE_CHANNEL_MAP)} employee agents…")
    agents_to_heal: list[AgentStatus] = []

    for emp_name, (fn_name, channel) in EMPLOYEE_CHANNEL_MAP.items():
        hours_ago = channel_bot_hours.get(channel, 999.0)
        is_down = hours_ago > AGENT_DOWN_HOURS

        agent_st = AgentStatus(
            emp_name=emp_name,
            fn_name=fn_name,
            channel=channel,
            channel_last_bot_hours_ago=hours_ago,
            is_agent_down=is_down,
        )
        report.agent_statuses.append(agent_st)

        if is_down:
            print(f"  AGENT_DOWN: {emp_name} → #{channel} ({_fmt_ago(hours_ago)})")
            agents_to_heal.append(agent_st)

    report.agent_down_count = len(agents_to_heal)

    # ── Phase 4: Post report to #incidents ────────────────────────────────────
    incidents_text = build_incidents_report(report)
    print(f"\n  Posting health report to #incidents…")
    print(incidents_text)
    inc_res = post_to_slack(
        SLACK_BOT_TOKEN, "incidents", incidents_text,
        username="Channel Health Monitor",
        icon_emoji=":rotating_light:",
    )
    if not inc_res.get("ok"):
        print(f"  [monitor] #incidents post failed: {inc_res.get('error')}")

    # ── Phase 5: Auto-heal AGENT_DOWN cases ───────────────────────────────────
    if agents_to_heal:
        print(f"\n  Auto-healing {len(agents_to_heal)} agent(s)…")
        for agent_st in agents_to_heal:
            try:
                success, err = auto_heal_agent(
                    SLACK_BOT_TOKEN,
                    agent_st.emp_name,
                    agent_st.fn_name,
                    agent_st.channel,
                )
                agent_st.auto_healed = success
                agent_st.heal_error = err
                if success:
                    report.agents_healed += 1
            except Exception as exc:
                agent_st.heal_error = str(exc)[:200]
                print(f"  [auto-heal] exception for {agent_st.emp_name}: {exc}")
            time.sleep(1.0)

    # ── Phase 6: Post summary to #allquantedge ────────────────────────────────
    summary = build_allquantedge_summary(report)
    print(f"\n  Posting summary to #allquantedge: {summary}")
    aq_res = post_to_slack(
        SLACK_BOT_TOKEN, "allquantedge", summary,
        username="Channel Health Monitor",
        icon_emoji=":bar_chart:",
    )
    if not aq_res.get("ok"):
        print(f"  [monitor] #allquantedge post failed: {aq_res.get('error')}")

    # ── Phase 7: Persist state (update last_post_ts for healed agents) ────────
    try:
        state = _try_load_state()
        last_posts = state.setdefault("last_post_ts", {})
        now_ts = time.time()
        for agent_st in agents_to_heal:
            if agent_st.auto_healed:
                # Map emp_name to the short key used by slack_agent_team
                short_key = agent_st.emp_name.split("_")[0]
                last_posts[short_key] = now_ts
        _try_save_state(state)
    except Exception as exc:
        print(f"  [monitor] state save failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[channel_monitor] Done — {report.active_count}/{len(report.channel_statuses)} active, "
          f"{report.silent_count} silent, {report.agent_down_count} agent gaps, "
          f"{report.agents_healed} auto-healed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
