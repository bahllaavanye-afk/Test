"""Workflow files must not contain duplicate keys.

PyYAML silently accepts duplicate mapping keys (last one wins). GitHub Actions
does NOT — it refuses to parse the file, creates a run with **zero jobs**, and
marks it `failure`. There is no error in any log, because no job ever starts.

That disabled five workflows completely, including the entire ML pipeline:

    agent-health-monitor · channel-monitor · daily-employee-review
    model-audit · run-experiments-agent

All five carried `OPENROUTER_API_KEY` twice in the same `env:` block. They had
failed 100% of runs for weeks. The tell was in the API rather than the logs:
GitHub reported their workflow `name` as the FILE PATH
(`.github/workflows/model-audit.yml`) instead of the declared name, which is
its fallback when it cannot parse the file.

`yaml.safe_load` passing is NOT evidence a workflow is valid. This check uses a
loader that rejects duplicates the way Actions does.
"""
from __future__ import annotations

import collections
import pathlib

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / "workflows"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that raises on duplicate mapping keys."""


def _no_duplicates(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate keys: {sorted(map(str, dupes))}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _workflow_files() -> list[pathlib.Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def test_workflows_exist():
    assert _workflow_files(), f"no workflow files found under {WORKFLOWS}"


def test_no_workflow_has_duplicate_keys():
    offenders = []
    for path in _workflow_files():
        try:
            yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
        except ValueError as exc:
            offenders.append(f"{path.name}: {exc}")
        except yaml.YAMLError as exc:
            offenders.append(f"{path.name}: unparseable — {str(exc)[:120]}")

    assert not offenders, (
        "GitHub Actions refuses a workflow with duplicate keys and creates a run "
        "with ZERO jobs — no log, no error, the workflow simply never runs:\n  "
        + "\n  ".join(offenders)
    )


def test_every_workflow_declares_jobs_and_triggers():
    """A workflow with no jobs or no triggers can never do anything."""
    broken = []
    for path in _workflow_files():
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            broken.append(f"{path.name}: unparseable — {str(exc)[:80]}")
            continue
        if not isinstance(doc, dict):
            broken.append(f"{path.name}: not a mapping")
            continue
        # `on:` is parsed as the boolean True by YAML 1.1
        triggers = doc.get("on", doc.get(True))
        if not triggers:
            broken.append(f"{path.name}: no triggers")
        if not doc.get("jobs"):
            broken.append(f"{path.name}: no jobs")

    assert not broken, "workflow(s) that cannot run:\n  " + "\n  ".join(broken)


def test_the_strict_loader_actually_rejects_duplicates():
    """The check above only means something if it can fail."""
    src = """
jobs:
  build:
    env:
      KEY: a
      KEY: b
"""
    assert yaml.safe_load(src) is not None, "SafeLoader tolerates this — that is the trap"
    with pytest.raises(ValueError, match="duplicate keys"):
        yaml.load(src, Loader=_StrictLoader)
