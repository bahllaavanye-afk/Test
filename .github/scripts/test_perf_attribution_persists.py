"""The P&L attribution file was recomputed and thrown away on every run.

`fill_tracker.py` fetches filled orders, attributes them back to strategies via
the `client_order_id` encoding, and writes cumulative stats to

    backend/performance_log/strategy_performance.json

`fill-tracking.yml` ran it on schedule and reported success — but never
committed the output. Actions runners are ephemeral, so the file existed for
the length of the job and vanished. It has never been in the repository.

THREE consumers read that exact path, and all three were therefore inert:

  strategy_trimmer.py     load_perf() -> {} -> no strategy ever evaluated,
                          so .github/state/strategy_trims.json is never written
  strategy_auto_tuner.py  prints "not found — no data to tune from" and stops
  desk_order_placer.py    _trimmed_strategies() reads the trims file that the
                          trimmer never produces

So the file-based retirement path was dead end-to-end, at the source rather
than at the consumer. (The desk's OTHER pruning mechanism — attribution weights
from /api/v1/leaderboard/live, which sets weight 0.0 and skips the order with a
`✂ pruned by attribution` line — is live and unaffected. That distinction
matters: losing strategies WERE being stopped; the redundant file-based trimmer
was the part that never worked.)

Same shape as the dead `run_desk()` in test_no_dead_desk_path.py: machinery
that runs on schedule, exits zero, and produces nothing that outlives the job.
A workflow that writes a file it does not commit is indistinguishable from one
that does — until something tries to read it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_WF = Path(__file__).resolve().parents[1] / "workflows" / "fill-tracking.yml"
_SCRIPTS = Path(__file__).resolve().parent
_PERF_PATH = "backend/performance_log/strategy_performance.json"


@pytest.fixture(scope="module")
def wf() -> str:
    assert _WF.is_file(), f"missing {_WF}"
    return _WF.read_text()


def test_the_workflow_commits_the_file_it_writes(wf):
    assert _PERF_PATH in wf, (
        "fill-tracking.yml never references the file its script writes, so the "
        "attribution data cannot outlive the runner"
    )
    assert "git commit" in wf and "git push" in wf, (
        "the tracker's output is written to an ephemeral runner and discarded"
    )


# ── the producer must keep up with its consumers ─────────────────────────────
# Committing the file was necessary but not sufficient: it also has to be
# RECENT. `workflow_run: [CI]` looks like a frequent trigger and is not one —
# agent-branch CI is dispatched by auto-pr.yml with GITHUB_TOKEN, and GitHub
# does not create new workflow runs from GITHUB_TOKEN-triggered events, so the
# chain is suppressed for exactly the branches that generate the fills.
# Measured 2026-07-29: one firing in three hours across many CI passes. The
# cron is the real schedule.

def _yaml_on(src: str) -> dict:
    """The parsed `on:` block. YAML 1.1 turns the key `on` into True."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(src)
    return doc.get("on", doc.get(True)) or {}


def _cron_lines(src: str) -> list[str]:
    return re.findall(r'cron:\s*"([^"]+)"', src)


def test_the_tracker_runs_at_least_as_often_as_the_trimmer_reads(wf):
    """A once-a-day producer feeding a 4x-a-day consumer starves it."""
    trim = (_WF.parent / "strategy-trim.yml").read_text()
    trim_crons = _cron_lines(trim)
    assert trim_crons, "trimmer has no cron — this comparison is meaningless"

    def per_day(cron: str) -> float:
        minute, hour, dom, month, dow = cron.split()
        runs = 24 / int(hour.split("/")[1]) if hour.startswith("*/") else (
            1 if hour.isdigit() else 24
        )
        days = 5 / 7 if dow == "1-5" else 1
        return runs * days

    tracker_rate = max(per_day(c) for c in _cron_lines(wf))
    trimmer_rate = max(per_day(c) for c in trim_crons)
    assert tracker_rate >= trimmer_rate, (
        f"fill-tracking runs {tracker_rate:g}x/day but strategy-trim reads it "
        f"{trimmer_rate:g}x/day — most trimmer runs would see stale attribution"
    )


def test_the_tracker_runs_every_day_not_just_weekdays(wf):
    """The crypto desks trade 24/7; weekday-only left weekend fills unscored."""
    for cron in _cron_lines(wf):
        assert cron.split()[4] in ("*", "?"), (
            f"cron {cron!r} restricts the day-of-week, but fills accrue daily"
        )


def test_the_tracker_lands_before_the_trimmer_reads(wf):
    """Ordering within the shared 6h slot: produce, then consume."""
    trim = (_WF.parent / "strategy-trim.yml").read_text()
    t_min = int(_cron_lines(wf)[0].split()[0])
    c_min = int(_cron_lines(trim)[0].split()[0])
    assert t_min < c_min, (
        f"fill-tracking fires at :{t_min:02d} and strategy-trim at :{c_min:02d} — "
        f"the trimmer would read the previous cycle's data"
    )


# ── the cron alone does not deliver ──────────────────────────────────────────
# Measured 2026-07-29: the 06:11 slot was DROPPED, not delayed — 85 minutes
# past with no run recorded, while the rest of the fleet fired normally. The
# previous cron (0 22) had started 62 minutes late. GitHub silently drops
# scheduled runs under load, and a dropped run never appears in the run list,
# so a schedule is not a delivery guarantee. The crypto desk fires
# "7,27,47 * * * *" 24/7 and is cron-actored — so it escapes the GITHUB_TOKEN
# suppression that makes the CI chain useless here — and is used as the
# reliability anchor, with a freshness gate keeping the effective rate at ~4x/day.

def test_it_chains_off_a_reliable_cron_actored_workflow(wf):
    triggers = _yaml_on(wf)
    chained = triggers.get("workflow_run", {}).get("workflows", [])
    assert any("Crypto 24/7" in w for w in chained), (
        "fill-tracking relies on its own cron, which has been observed dropped "
        f"outright. Chain it off a frequently-firing cron-actored workflow. "
        f"Currently chained to: {chained}"
    )


def test_the_freshness_gate_exists_so_the_chain_does_not_run_it_72x_a_day(wf):
    assert "generated_at" in wf, (
        "no freshness gate — chaining off a 20-minute workflow without one "
        "would run the tracker ~72x/day and commit as often"
    )
    assert "steps.due.outputs.run" in wf, "gate computed but not applied to any step"


def test_every_expensive_step_is_gated(wf):
    """A gate applied to some steps but not others still pays the cost."""
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(wf)["jobs"]["track-fills"]["steps"]
    must_gate = ("Run fill tracker", "Persist strategy performance",
                 "Install dependencies", "Set up Python 3.11")
    for s in steps:
        if s.get("name") in must_gate:
            assert s.get("if") == "steps.due.outputs.run == 'true'", (
                f"step {s.get('name')!r} is not gated on the freshness check"
            )


def test_the_gate_window_is_under_the_cron_period(wf):
    """A window >= the cron period would let a scheduled run skip itself."""
    m = re.search(r"MAX_AGE_S\s*=\s*(\d+)\s*\*\s*3600", wf)
    assert m, "freshness window not found"
    hours = int(m.group(1))
    # For "*/N" on the HOUR field the period is N hours. (24/N is runs-per-day,
    # which is what this line said at first — it made a 5h window look like it
    # exceeded a 4h period. Caught by this test failing on a correct workflow.)
    cron_hours = int(_cron_lines(wf)[0].split()[1].split("/")[1])
    assert hours < cron_hours, (
        f"gate window {hours}h >= cron period {cron_hours}h — a scheduled run "
        f"could find the artifact 'fresh' and skip, defeating its own schedule"
    )


def test_the_gate_fails_open(wf):
    """Missing or unparseable artifact must RUN, not skip.

    The entire defect being fixed is 'never produced anything', so a gate that
    fails closed would recreate it in a new form.
    """
    body = wf.split("python3 - <<'PY'", 1)[1].split("PY", 1)[0]
    assert "except Exception" in body and 'print("run=true")' in body, (
        "the freshness gate must default to running when it cannot read the "
        "artifact's timestamp"
    )


def test_it_has_write_permission(wf):
    """A commit step without contents: write fails at push time, not at parse."""
    m = re.search(r"^permissions:\s*$(.*?)^\S", wf, re.MULTILINE | re.DOTALL)
    assert m, "no permissions block — the default may be read-only"
    assert "contents: write" in m.group(1)


def test_the_commit_is_conditional(wf):
    """An unconditional commit fails the job on a no-op run."""
    assert "git diff --quiet" in wf, "commit must be skipped when nothing changed"


def test_the_commit_does_not_retrigger_ci(wf):
    """This workflow triggers on CI completion — an unmarked commit would loop."""
    body = wf.split("Persist strategy performance", 1)[1]
    assert "[skip ci]" in body, (
        "fill-tracking runs on workflow_run:[CI]; a commit without [skip ci] "
        "risks a self-sustaining loop"
    )


def test_the_push_retries(wf):
    """Concurrent state-bot pushes to main are routine in this repo."""
    body = wf.split("Persist strategy performance", 1)[1]
    assert "for i in" in body and "sleep" in body, "push should back off and retry"


# ── the consumers must keep agreeing on the path ─────────────────────────────

@pytest.mark.parametrize("script", ["strategy_trimmer.py", "strategy_auto_tuner.py", "fill_tracker.py"])
def test_producer_and_consumers_use_the_same_path(script):
    """A silent path divergence here would restore the exact original bug."""
    src = (_SCRIPTS / script).read_text()
    assert '"performance_log"' in src or "performance_log" in src, script
    assert "strategy_performance.json" in src, script


def test_the_trimmer_still_fails_soft_on_a_missing_file():
    """Until the first commit lands, the file is absent — that must not crash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "trimmer_under_test", _SCRIPTS / "strategy_trimmer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.load_perf() == {} or isinstance(mod.load_perf(), dict)
    assert isinstance(mod.load_trims(), dict)


# ── the commit must work for a file that has NEVER been committed ────────────
# The 2026-07-29 08:53 run succeeded, wrote the artifact, printed
# "No attribution changes.", and left the repo empty. Cause: the step tested
#     ! git diff --quiet -- <path>
# BEFORE staging, and `git diff` does not see UNTRACKED files. So for a file
# that has never been committed it reported "no change" and skipped the commit
# — the condition could never make the FIRST commit, the only one that
# mattered. These tests run the SHIPPED shell against a real repo.

def _persist_script(wf: str) -> str:
    body = wf.split("- name: Persist strategy performance", 1)[1]
    body = body.split("run: |", 1)[1]
    lines = []
    for line in body.split("\n"):
        if line.strip() and not line.startswith(" " * 10):
            break
        lines.append(line[10:] if line.startswith(" " * 10) else line)
    return "\n".join(lines)


def _run_persist(wf: str, tmp_path, content: str | None, precommit: str | None = None):
    """Execute the real persist step in a throwaway repo. Returns (stdout, committed)."""
    import subprocess
    repo = tmp_path / "repo"
    (repo / "backend" / "performance_log").mkdir(parents=True)
    sh = lambda *a: subprocess.run(a, cwd=repo, capture_output=True, text=True)
    sh("git", "init", "-q", "-b", "main")
    sh("git", "config", "user.email", "t@t"); sh("git", "config", "user.name", "t")
    (repo / "seed").write_text("x")
    sh("git", "add", "seed"); sh("git", "commit", "-qm", "seed")
    f = repo / "backend/performance_log/strategy_performance.json"
    if precommit is not None:
        f.write_text(precommit)
        sh("git", "add", "-A"); sh("git", "commit", "-qm", "prior")
    if content is not None:
        f.write_text(content)
    before = sh("git", "rev-parse", "HEAD").stdout.strip()
    script = _persist_script(wf).replace("$GITHUB_WORKSPACE", str(repo))
    # the push has no remote here; drop it so we test staging/commit only
    script = "\n".join(l for l in script.split("\n") if "git push" not in l)
    out = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True)
    after = sh("git", "rev-parse", "HEAD").stdout.strip()
    return out.stdout + out.stderr, before != after


def test_a_brand_new_artifact_is_committed(wf, tmp_path):
    """The regression. Untracked file must still produce a commit."""
    out, committed = _run_persist(wf, tmp_path, content='{"generated_at": "2026-01-01T00:00:00Z"}')
    assert committed, (
        "a never-before-committed artifact was NOT committed — this is the "
        f"untracked-file bug returning. Output:\n{out}"
    )


def test_an_updated_artifact_is_committed(wf, tmp_path):
    out, committed = _run_persist(
        wf, tmp_path, content='{"generated_at": "2026-01-02T00:00:00Z"}',
        precommit='{"generated_at": "2026-01-01T00:00:00Z"}')
    assert committed, f"a modified artifact was not committed:\n{out}"


def test_an_unchanged_artifact_is_not_committed(wf, tmp_path):
    same = '{"generated_at": "2026-01-01T00:00:00Z"}'
    out, committed = _run_persist(wf, tmp_path, content=same, precommit=same)
    assert not committed, f"committed an unchanged artifact:\n{out}"
    assert "No attribution changes" in out


def test_a_missing_artifact_warns_and_does_not_fail(wf, tmp_path):
    """The tracker can legitimately produce nothing; that must not fail the job."""
    out, committed = _run_persist(wf, tmp_path, content=None)
    assert not committed
    assert "nothing to persist" in out or "::warning" in out, out


# ── consumers must stay coupled to the producer's schedule ───────────────────
# Moving the tracker's cron in a previous change silently desynchronised
# strategy-auto-tune, whose comment still claimed "30 min after fill tracker
# completes" while it actually read data 4h19m old. Nothing failed — nothing
# tied the schedules together. These tests do.

_TUNE = _WF.parent / "strategy-auto-tune.yml"
_TRIM = _WF.parent / "strategy-trim.yml"


def _slot_minutes(cron: str) -> tuple[int, set[int]]:
    """(minute, set of hours) for a cron using only literals and */N."""
    minute, hour = cron.split()[0], cron.split()[1]
    if hour == "*":
        hours = set(range(24))
    elif hour.startswith("*/"):
        hours = set(range(0, 24, int(hour[2:])))
    else:
        hours = {int(h) for h in hour.split(",")}
    return int(minute), hours


def test_the_auto_tuner_reads_shortly_after_a_producer_slot():
    """Its whole design is 'run just after fresh attribution lands'."""
    p_min, p_hours = _slot_minutes(_cron_lines(_WF.read_text())[0])
    t_min, t_hours = _slot_minutes(_cron_lines(_TUNE.read_text())[0])
    lags = []
    for th in sorted(t_hours):
        prior = max((h for h in p_hours if h <= th), default=None)
        if prior is None:
            continue
        lags.append((th - prior) * 60 + (t_min - p_min))
    assert lags, "auto-tuner never runs after a producer slot"
    assert min(lags) >= 0, "auto-tuner fires BEFORE the producer in its hour"
    assert min(lags) <= 90, (
        f"auto-tuner runs {min(lags)} min after the freshest attribution slot — "
        f"its schedule has drifted from the producer's"
    )


@pytest.mark.parametrize("wf_path", ["strategy-auto-tune.yml", "strategy-trim.yml"])
def test_consumers_run_every_day_not_just_weekdays(wf_path):
    """Crypto trades 24/7; a weekday-only consumer ignores weekend fills."""
    src = (_WF.parent / wf_path).read_text()
    for cron in _cron_lines(src):
        assert cron.split()[4] in ("*", "?"), (
            f"{wf_path} cron {cron!r} restricts day-of-week, but fills accrue daily"
        )


def test_the_trimmer_still_reads_after_the_producer():
    """The relationship that was already correct — pinned so it stays that way."""
    p_min, p_hours = _slot_minutes(_cron_lines(_WF.read_text())[0])
    c_min, c_hours = _slot_minutes(_cron_lines(_TRIM.read_text())[0])
    assert p_hours == c_hours, "trimmer and tracker no longer share slot hours"
    assert c_min > p_min, "trimmer would read the previous cycle's attribution"
