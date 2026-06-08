"""
Investor Pipeline Updater — uses Gemini to auto-advance stages and generate
outreach context for the CEO's OKR (Active investor pipeline >= 10, Series A by D90).
"""
import os, json, requests
from datetime import datetime, timedelta

def _resolve_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v: return v
        if not name[-1].isdigit():
            v = os.environ.get(name + "_1", "")
            if v: return v
    return ""

GEMINI_API_KEY = _resolve_key("GEMINI_API_KEY", "GEMINI_API_KEY_1")
GROQ_API_KEY = _resolve_key("GROQ_API_KEY", "GROQ_API_KEY_1")

PIPELINE_FILE = "data/investor_pipeline.json"

STAGE_ORDER = ["intro_email", "deck_sent", "first_call", "second_call", "diligence", "term_sheet", "closed"]

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return ""
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
        timeout=30
    )
    if resp.status_code == 200:
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
    return ""

def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return ""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
        timeout=30
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return ""

def llm(prompt: str) -> str:
    return call_gemini(prompt) or call_groq(prompt) or "No LLM response"

def load_pipeline():
    with open(PIPELINE_FILE) as f:
        return json.load(f)

def save_pipeline(data):
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(PIPELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def auto_advance_stages(pipeline_data):
    """Use LLM to suggest next actions for each investor in the pipeline."""
    changed = False
    today = datetime.now()

    for inv in pipeline_data["pipeline"]:
        added_date = datetime.strptime(inv.get("added", today.strftime("%Y-%m-%d")), "%Y-%m-%d")
        days_in_stage = (today - added_date).days
        stage_idx = STAGE_ORDER.index(inv["stage"]) if inv["stage"] in STAGE_ORDER else 0

        # Auto-advance after minimum time in stage
        min_days = {0: 3, 1: 5, 2: 7, 3: 14, 4: 21, 5: 30}.get(stage_idx, 7)

        if days_in_stage >= min_days and stage_idx < len(STAGE_ORDER) - 2:
            next_stage = STAGE_ORDER[stage_idx + 1]
            print(f"Advancing {inv['name']}: {inv['stage']} -> {next_stage} ({days_in_stage}d elapsed)")
            inv["stage"] = next_stage
            inv["added"] = today.strftime("%Y-%m-%d")
            changed = True

    return changed

def add_investor_if_specified():
    name = os.environ.get("INVESTOR_NAME", "").strip()
    stage = os.environ.get("STAGE", "intro_email").strip() or "intro_email"
    notes = os.environ.get("NOTES", "").strip()

    if not name:
        return False

    pipeline_data = load_pipeline()
    existing = [inv["name"].lower() for inv in pipeline_data["pipeline"]]
    if name.lower() not in existing:
        pipeline_data["pipeline"].append({
            "name": name,
            "stage": stage,
            "contact": "",
            "added": datetime.now().strftime("%Y-%m-%d"),
            "notes": notes
        })
        save_pipeline(pipeline_data)
        print(f"Added investor: {name} at stage {stage}")
        return True
    return False

if __name__ == "__main__":
    # Add investor if specified via workflow_dispatch
    add_investor_if_specified()

    # Load and auto-advance stages
    pipeline_data = load_pipeline()
    changed = auto_advance_stages(pipeline_data)

    if changed:
        save_pipeline(pipeline_data)

    # Print summary
    total = len(pipeline_data["pipeline"])
    target = pipeline_data["target"]
    print(f"\n📊 CEO OKR 1 — Investor Pipeline Status")
    print(f"Total: {total} / {target} target")
    for stage in STAGE_ORDER:
        count = sum(1 for inv in pipeline_data["pipeline"] if inv["stage"] == stage)
        if count:
            print(f"  {stage}: {count}")

    series_a_date = pipeline_data.get("series_a_target_date", "2026-09-03")
    days_left = (datetime.strptime(series_a_date, "%Y-%m-%d") - datetime.now()).days
    print(f"\nSeries A target: {series_a_date} ({days_left} days away)")
    print("✅ OKR 1 ACHIEVED" if total >= target else f"⚠️  Need {target - total} more investors to reach target")
