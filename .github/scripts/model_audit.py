"""Audit configured LLM model IDs against each provider's live /models list.

Providers retire model IDs without warning (Cerebras killed llama-3.3-70b,
NVIDIA killed nemotron-70b) and the cascade then 404s silently until someone
notices the brain is dumber. This script asks every configured provider for
its live model list and reports any configured ID that no longer exists.

Exit code 0 always (auditing must not break CI) — drift is reported via
stdout/step summary; the workflow files an `agent-fix-needed` issue so the
weekly queue worker repairs llm_common automatically.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_common  # noqa: E402


def _get_json(url: str, headers: dict[str, str]) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 QuantEdge-Audit", **headers})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f"    fetch failed: {str(exc)[:100]}", file=sys.stderr)
        return None


def _openai_model_ids(base_chat_url: str, key: str) -> set[str] | None:
    models_url = base_chat_url.replace("/chat/completions", "/models")
    data = _get_json(models_url, {"Authorization": f"Bearer {key}"})
    if not isinstance(data, dict):
        return None
    return {m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)}


def main() -> int:
    findings: list[str] = []
    checked: list[str] = []

    for p in llm_common._PROVIDERS:
        name, fmt = p["name"], p.get("fmt")
        model = p.get("model")
        key = os.environ.get(p.get("key_env", ""), "") or os.environ.get(p.get("key_env_alt", ""), "")
        if fmt != "openai" or not model:
            continue  # gemini embeds its model in the URL; a 404 there pages via brain-health
        if not key:
            print(f"[{name}] no key — skipped")
            continue
        print(f"[{name}] checking '{model}'…")
        live = _openai_model_ids(p["url"], key)
        if live is None:
            print(f"[{name}] models endpoint unreachable — skipped")
            continue
        checked.append(name)
        if model not in live:
            close = sorted(m for m in live if model.split("/")[-1][:8].lower() in m.lower())[:5]
            findings.append(
                f"**{name}**: configured model `{model}` is NOT in the live model list. "
                f"Nearby candidates: {', '.join(f'`{c}`' for c in close) or 'none obvious'}"
            )
            print(f"[{name}] ❌ '{model}' missing from live list")
        else:
            print(f"[{name}] ✅ '{model}' live")

    # OpenRouter slugs (one call covers all)
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        data = _get_json("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {or_key}"})
        if isinstance(data, dict):
            live = {m.get("id", "") for m in data.get("data", [])}
            checked.append("openrouter")
            for slug in llm_common._OPENROUTER_MODELS:
                if slug in live:
                    print(f"[openrouter] ✅ '{slug}' live")
                else:
                    findings.append(f"**openrouter**: slug `{slug}` is NOT in the live catalog.")
                    print(f"[openrouter] ❌ '{slug}' missing")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("## 🔎 LLM model-ID audit\n\n")
            fh.write(f"Providers checked: {', '.join(checked) or 'none (no keys reachable)'}\n\n")
            if findings:
                fh.write("### ❌ Drift found\n\n" + "\n".join(f"- {f}" for f in findings) + "\n")
            else:
                fh.write("✅ Every configured model ID is live.\n")

    with open(os.environ.get("AUDIT_OUT", "/tmp/model_audit.md"), "w") as fh:
        if findings:
            fh.write(
                "The weekly model audit found configured LLM model IDs that the provider "
                "no longer serves. Update `.github/scripts/llm_common.py` (or the matching "
                "env override) to a live ID per the candidates below, and verify with a "
                "1-token call.\n\n" + "\n".join(f"- {f}" for f in findings) + "\n"
            )
    print(f"\n{len(findings)} drift finding(s) across {len(checked)} provider(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
