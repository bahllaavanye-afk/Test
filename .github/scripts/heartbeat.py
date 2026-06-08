import json, subprocess, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
MEMORY_FILE = REPO_ROOT / ".github" / "state" / "agent_memory.json"

def main():
    d = json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {"version": 1}
    d["last_heartbeat"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metrics = d.setdefault("platform_metrics", {})
    metrics["heartbeat_count"] = metrics.get("heartbeat_count", 0) + 1
    try:
        result = subprocess.run(["git", "log", "--oneline", "--since=1 day ago"], capture_output=True, text=True, cwd=REPO_ROOT)
        metrics["commits_today"] = len(result.stdout.strip().splitlines())
    except Exception:
        pass
    MEMORY_FILE.write_text(json.dumps(d, indent=2))
    print(f"Heartbeat #{metrics['heartbeat_count']} at {d['last_heartbeat']}")

if __name__ == "__main__":
    main()
