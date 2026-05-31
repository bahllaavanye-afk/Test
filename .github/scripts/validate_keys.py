#!/usr/bin/env python3
"""
Validate all configured API keys by making minimal test calls.
Posts results to #agent-api-usage on Slack.
Run: python .github/scripts/validate_keys.py
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error
from datetime import datetime, timezone

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _http_post(url: str, headers: dict, body: dict, timeout: int = 8) -> tuple[int, dict]:
    """Returns (status_code, response_dict)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}

def test_groq(key: str, label: str) -> dict:
    status, resp = _http_post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
    )
    ok = status == 200 and "choices" in resp
    return {"provider": label, "status": "✅" if ok else f"❌ {status}", "ok": ok}

def test_cerebras(key: str) -> dict:
    status, resp = _http_post(
        "https://api.cerebras.ai/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {"model": "qwen-3-32b", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
    )
    ok = status == 200 and "choices" in resp
    return {"provider": "Cerebras", "status": "✅" if ok else f"❌ {status}", "ok": ok}

def test_sambanova(key: str) -> dict:
    status, resp = _http_post(
        "https://api.sambanova.ai/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {"model": "Meta-Llama-3.3-70B-Instruct", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
    )
    ok = status == 200 and "choices" in resp
    return {"provider": "SambaNova", "status": "✅" if ok else f"❌ {status}", "ok": ok}

def test_gemini(key: str, label: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    data = json.dumps({"contents": [{"parts": [{"text": "Say OK"}]}]}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = json.loads(r.read())
            ok = "candidates" in resp
            return {"provider": label, "status": "✅" if ok else "❌ bad response", "ok": ok}
    except urllib.error.HTTPError as e:
        return {"provider": label, "status": f"❌ {e.code}", "ok": False}
    except Exception as e:
        return {"provider": label, "status": f"❌ {e}", "ok": False}

def test_openrouter(key: str) -> dict:
    status, resp = _http_post(
        "https://openrouter.ai/api/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
         "HTTP-Referer": "https://github.com/bahllaavanye-afk/Test"},
        {"model": "meta-llama/llama-3.3-70b-instruct:free",
         "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
    )
    ok = status == 200 and "choices" in resp
    return {"provider": "OpenRouter", "status": "✅" if ok else f"❌ {status}", "ok": ok}

def test_github_models(token: str) -> dict:
    status, resp = _http_post(
        "https://models.inference.ai.azure.com/chat/completions",
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5}
    )
    ok = status == 200 and "choices" in resp
    return {"provider": "GitHub Models", "status": "✅" if ok else f"❌ {status}", "ok": ok}

def test_slack(token: str) -> dict:
    req = urllib.request.Request(
        "https://slack.com/api/auth.test",
        headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = json.loads(r.read())
            ok = resp.get("ok", False)
            return {"provider": "Slack Bot", "status": "✅" if ok else f"❌ {resp.get('error')}", "ok": ok}
    except Exception as e:
        return {"provider": "Slack Bot", "status": f"❌ {e}", "ok": False}

def post_results(token: str, results: list[dict]) -> None:
    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    lines = [
        f"*:key: API Key Validation — {_today()}*",
        f"*{passed} passed / {failed} failed*",
        "```",
    ]
    for r in results:
        lines.append(f"{r['status']}  {r['provider']}")
    lines.append("```")
    if failed:
        lines.append(f"\n:warning: {failed} key(s) need attention — check GitHub Secrets")
    else:
        lines.append(":white_check_mark: All keys operational")

    data = json.dumps({
        "text": "\n".join(lines),
        "username": "Key Validator",
        "icon_emoji": ":key:",
    }).encode()

    # Find channel
    req = urllib.request.Request(
        "https://slack.com/api/conversations.list?limit=200",
        headers={"Authorization": f"Bearer {token}"}
    )
    ch_id = None
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            resp = json.loads(r.read())
            for ch in resp.get("channels", []):
                if ch.get("name") == "agent-api-usage":
                    ch_id = ch["id"]
                    break
    except Exception:
        pass

    if ch_id:
        post_data = json.dumps({"channel": ch_id, "text": "\n".join(lines),
                                "username": "Key Validator", "icon_emoji": ":key:"}).encode()
        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=post_data,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    # Always print to stdout
    print("\n".join(lines))

def main():
    g = os.environ.get
    results = []

    # Slack
    slack_token = g("SLACK_BOT_TOKEN", "")
    if slack_token:
        results.append(test_slack(slack_token))
    else:
        results.append({"provider": "Slack Bot", "status": "⚠️ not set", "ok": False})

    # Groq accounts
    for label, env in [("Groq Account 1", "GROQ_API_KEY_1"), ("Groq Account 2", "GROQ_API_KEY_2"), ("Groq Account 3", "GROQ_API_KEY_3")]:
        key = g(env, "") or (g("GROQ_API_KEY", "") if env == "GROQ_API_KEY_1" else "")
        if key:
            results.append(test_groq(key, label))
        else:
            results.append({"provider": label, "status": "⚠️ not set", "ok": False})

    # Cerebras
    key = g("CEREBRAS_API_KEY", "")
    if key:
        results.append(test_cerebras(key))
    else:
        results.append({"provider": "Cerebras", "status": "⚠️ not set", "ok": False})

    # SambaNova
    key = g("SAMBANOVA_API_KEY", "")
    if key:
        results.append(test_sambanova(key))
    else:
        results.append({"provider": "SambaNova", "status": "⚠️ not set", "ok": False})

    # Gemini
    for label, env in [("Gemini Account 1", "GEMINI_API_KEY_1"), ("Gemini Account 2", "GEMINI_API_KEY_2"), ("Gemini Account 3", "GEMINI_API_KEY_3")]:
        key = g(env, "") or (g("GEMINI_API_KEY", "") if env == "GEMINI_API_KEY_1" else "")
        if key:
            results.append(test_gemini(key, label))
        else:
            results.append({"provider": label, "status": "⚠️ not set", "ok": False})

    # OpenRouter
    key = g("OPENROUTER_API_KEY", "")
    if key:
        results.append(test_openrouter(key))
    else:
        results.append({"provider": "OpenRouter", "status": "⚠️ not set", "ok": False})

    # GitHub Models
    gh_token = g("GH_TOKEN", "") or g("GITHUB_TOKEN", "")
    if gh_token:
        results.append(test_github_models(gh_token))
    else:
        results.append({"provider": "GitHub Models", "status": "⚠️ not set", "ok": False})

    post_results(slack_token, results)

    failed = sum(1 for r in results if not r["ok"])
    return 1 if failed else 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
