"""A persist step that gates on `git diff --quiet` cannot see a NEW file.

The strategy trimmer computed a correct trim and threw it away. Run
30457733119 (2026-07-29 13:48 UTC), verbatim:

    [TRIM] avellaneda: cumulative return -7.9% ≤ -5.0% over 10 trades
    trimmed total: 1 | newly trimmed this run: 1
    No trim changes.

Three lines, and the third contradicts the first two. `strategy_trims.json`
had never been committed, so it was UNTRACKED, and `git diff --quiet -- <path>`
compares the working tree against the index for TRACKED paths only — an
untracked file is invisible to it, so the gate reported "no change", the commit
was skipped, and the file died with the runner. The desk then read a trims file
that did not exist and `avellaneda_stoikov_mm` kept trading after being retired
for losing 7.9%.

The same gate sat in `system-status.yml`, whose own header promises "commits a
fresh SYSTEM_STATUS.md so the repo always shows live truth". `SYSTEM_STATUS.md`
has never existed in this repository. Every run since it was written probed the
brain, Slack, Alpaca and the backend, rendered a report, and discarded it.

This is the *first write* that breaks — which is exactly the write that matters,
because it is the one that creates the state other jobs read. Once a file is
tracked the unstaged form starts working, which is why the bug survives review:
it looks correct, and for every file that already exists it IS correct.

The fix is to stage first and diff the INDEX (`git add -- f` then
`git diff --cached --quiet -- f`), which sees an addition as a change.

`test_git_really_cannot_see_untracked_in_diff_quiet` builds a real repository
and demonstrates the semantics rather than asserting them, so this file does
not rest on my description of what git does.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = sorted((_REPO / ".github" / "workflows").glob("*.yml"))

# `git add` forms that deliberately stage only already-tracked changes. For
# these the unstaged gate and the staging agree, so there is no blind spot.
_TRACKED_ONLY_ADD = {"-u", "--update", "-A", "--all", "."}


def _tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout
    return set(out.split())


def _steps(text: str) -> list[str]:
    """Split a workflow into step-sized chunks on `- name:` boundaries."""
    bounds = [m.start() for m in re.finditer(r"^\s*- name:", text, re.M)]
    if not bounds:
        return [text]
    bounds.append(len(text))
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _staged_paths(chunk: str) -> set[str]:
    """Paths this chunk explicitly stages, resolving simple `f="..."` vars."""
    variables = dict(re.findall(r'^\s*(\w+)="([^"]+)"', chunk, re.M))
    found = set()
    for raw in re.findall(r"git add\s+([^\n;&|]+)", chunk):
        for arg in raw.split():
            arg = arg.strip().strip('"').strip("'")
            if arg == "--" or arg in _TRACKED_ONLY_ADD:
                continue
            if arg.startswith("$"):
                arg = variables.get(arg.lstrip("${").rstrip("}"), "")
            if arg and not arg.startswith("-"):
                found.add(arg)
    return found


def _unstaged_gates(chunk: str) -> list[str]:
    """`git diff ... --quiet` commands that omit --cached (the blind form)."""
    return [
        cmd
        for cmd in re.findall(r"git diff[^\n;&|]*--quiet[^\n;&|]*", chunk)
        if "--cached" not in cmd and "--staged" not in cmd
    ]


def _persist_steps():
    tracked = _tracked_paths()
    for wf in _WORKFLOWS:
        text = wf.read_text()
        for chunk in _steps(text):
            if "git commit" not in chunk:
                continue
            for path in _staged_paths(chunk):
                if path not in tracked:
                    yield wf.name, chunk, path


def test_no_persist_step_gates_an_untracked_path_on_an_unstaged_diff():
    """The class, swept across every workflow — catch the next one for free."""
    broken = []
    for name, chunk, path in _persist_steps():
        for gate in _unstaged_gates(chunk):
            broken.append(f"  {name}: stages untracked {path!r} behind `{gate.strip()}`")
    assert not broken, (
        "These persist steps gate on a diff that cannot see the file they "
        "stage. The file is untracked, so the gate reports 'no change' and the "
        "work is silently discarded:\n" + "\n".join(sorted(set(broken)))
        + "\n\nUse: git add -- \"$f\" && git diff --cached --quiet -- \"$f\""
    )


@pytest.mark.parametrize("workflow,path", [
    ("strategy-trim.yml", ".github/state/strategy_trims.json"),
    ("system-status.yml", "SYSTEM_STATUS.md"),
])
def test_the_two_known_losses_stay_fixed(workflow, path):
    """Nail the specific regressions, not just the shape."""
    text = (_REPO / ".github" / "workflows" / workflow).read_text()
    chunk = next((c for c in _steps(text) if "git commit" in c and "git add" in c), None)
    assert chunk is not None, f"{workflow} no longer has a persist step"
    assert path in _staged_paths(chunk), f"{workflow} no longer stages {path}"
    assert not _unstaged_gates(chunk), (
        f"{workflow} is back to gating {path} on an unstaged `git diff --quiet`. "
        f"{path} is created by the job, so the gate never fires and the result "
        f"is thrown away with the runner."
    )
    assert "--cached" in chunk, f"{workflow} must diff the index, not the worktree"


def test_git_really_cannot_see_untracked_in_diff_quiet(tmp_path):
    """Demonstrate the semantics in a real repo instead of asserting them.

    If this ever fails, git changed and the whole premise above needs revisiting
    — which is precisely why it is checked rather than taken on faith.
    """
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True
        )

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    (tmp_path / "seed").write_text("seed\n")
    git("add", "seed")
    git("commit", "-qm", "seed")

    new = tmp_path / "state.json"
    new.write_text('{"avellaneda": "trimmed"}\n')

    # The blind form: exit 0 means "no difference" — the new file is invisible.
    assert git("diff", "--quiet", "--", "state.json").returncode == 0, (
        "git now reports untracked files in `git diff --quiet`"
    )

    # The fix: stage it, then diff the index.
    git("add", "--", "state.json")
    assert git("diff", "--cached", "--quiet", "--", "state.json").returncode != 0, (
        "staging then diffing the index must report the addition as a change"
    )
