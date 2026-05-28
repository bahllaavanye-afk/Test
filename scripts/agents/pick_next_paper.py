"""
Pick the next paper from experiments/research_queue.yaml.

Usage:
    python pick_next_paper.py                     # prints "<id>|<title>|<url>"
    python pick_next_paper.py --mark-implemented 007
    python pick_next_paper.py --list              # show all pending papers
    python pick_next_paper.py --status            # summary counts by status

Picker rules (from queue file header):
  1. Skip anything not 'pending'.
  2. Prefer highest expected_sharpe.
  3. Tie-break by id ascending.
"""

import argparse
import sys
import os

try:
    import yaml
except ImportError:
    # yaml may not be installed; fall back to a minimal YAML-subset parser
    yaml = None

QUEUE_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "experiments", "research_queue.yaml"
)


def _load_queue(path: str) -> dict:
    path = os.path.realpath(path)
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    # Minimal fallback: only used if PyYAML is not installed
    raise RuntimeError(
        "PyYAML is required. Run: pip install pyyaml"
    )


def _save_queue(data: dict, path: str) -> None:
    path = os.path.realpath(path)
    if yaml is None:
        raise RuntimeError("PyYAML required to save queue.")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _pick(papers: list[dict]) -> dict | None:
    pending = [p for p in papers if p.get("status") == "pending"]
    if not pending:
        return None
    # Sort: descending sharpe, then ascending id (string sort is fine for zero-padded ids)
    pending.sort(key=lambda p: (-float(p.get("expected_sharpe", 0)), p.get("id", "999")))
    return pending[0]


def cmd_pick(args):
    data = _load_queue(QUEUE_FILE)
    papers = data.get("queue", [])
    paper = _pick(papers)
    if paper is None:
        print("", end="")  # empty output signals queue exhausted
        sys.exit(0)
    pid = paper.get("id", "?")
    title = paper.get("title", "Unknown")
    url = paper.get("url", "")
    print(f"{pid}|{title}|{url}")


def cmd_mark_implemented(paper_id: str):
    data = _load_queue(QUEUE_FILE)
    papers = data.get("queue", [])
    found = False
    for p in papers:
        if p.get("id") == paper_id:
            p["status"] = "implemented"
            found = True
            break
    if not found:
        print(f"Paper {paper_id} not found in queue.", file=sys.stderr)
        sys.exit(1)
    data["queue"] = papers
    _save_queue(data, QUEUE_FILE)
    print(f"Marked paper {paper_id} as implemented.")


def cmd_list():
    data = _load_queue(QUEUE_FILE)
    papers = data.get("queue", [])
    pending = [p for p in papers if p.get("status") == "pending"]
    pending.sort(key=lambda p: (-float(p.get("expected_sharpe", 0)), p.get("id", "999")))
    print(f"{'ID':<5} {'Sharpe':>7}  {'Title'}")
    print("-" * 70)
    for p in pending:
        print(f"{p.get('id','?'):<5} {p.get('expected_sharpe',0):>7.2f}  {p.get('title','?')}")


def cmd_status():
    data = _load_queue(QUEUE_FILE)
    papers = data.get("queue", [])
    counts: dict[str, int] = {}
    for p in papers:
        s = p.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status:<15} {count:>3}")


def main():
    parser = argparse.ArgumentParser(description="Research queue picker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mark-implemented", metavar="ID", help="Mark a paper as implemented")
    group.add_argument("--list", action="store_true", help="List all pending papers by priority")
    group.add_argument("--status", action="store_true", help="Print queue status summary")
    args = parser.parse_args()

    if args.mark_implemented:
        cmd_mark_implemented(args.mark_implemented)
    elif args.list:
        cmd_list()
    elif args.status:
        cmd_status()
    else:
        cmd_pick(args)


if __name__ == "__main__":
    main()
