# Playbook: copy Options Alpha bots (full fidelity)

In any Cowork / Claude-in-Chrome session, after YOU log into OA yourself, say:
> Follow docs/playbooks/OA_BOT_COPY.md

## Extraction spec v2 (per bot) — capture EVERYTHING visible
Open each bot's DETAIL page (the list view lacks legs/settings). Extract:

```json
{"name","bot_id","account","bot_group","template","bot_version","notes",
 "settings":{
   "safeguards":{"allocation","daily_positions","position_limit","day_trading"},
   "scan_speeds":{"automations","exit_options"},
   "symbols","activity_alerts":[]},
 "automations":[
   {"name","type","trigger":{"type","schedule"},
    "flow":[
      {"node":"Decision","label","yes_if":[]},
      {"node":"Action","label","position":{
        "type","symbol","expiration",
        "legs":[{"side","type","strike_rule","delta","dte","ratio"}],
        "position_size","price",
        "exit_options":{"tp_pct","sl_pct","time"},
        "entry_criteria":{},
        "tags"}}]}],
 "exit_management","positions_tab":{"open_positions","closed_positions_sample":[],"note"},
 "source_url"}
```

Rules: capture every Decision node's criteria verbatim; every leg's strike
rule as written ("$0.01 below underlying price" etc.); read-only — never
click anything that changes bot state.

## What the importer does with it
- Supported settings map directly (trigger schedule → time_window, TP/SL/time
  exits, allocation → size_pct, position limits → no_position guard).
- OA features WITHOUT an engine equivalent yet (reward/risk gates, decision
  recipes, scan speeds, alert prefs) are preserved verbatim under the
  template's "oa_meta" key + summarized in its description — nothing is
  silently dropped, and each such feature is a candidate engine upgrade for
  the improvement queue.

## Import (no hand-coding)
Save the array as oa_dump.json in repo root, then:
  python scripts/import_oa_bots.py oa_dump.json
Or paste the JSON into a GitHub issue titled "[oa-import] ..." (or into
Claude chat) — pipeline converts, tests, PRs through the automerge gate.

## Keeping it current afterwards
OA_SESSION_COOKIE repo secret is set — oa-scout diffs daily and auto-files
new bots. When the cookie expires the scout posts the auth-wall warning to
Discord; re-paste to resume.
