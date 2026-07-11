# Playbook: copy Options Alpha bots (one line to run it)

In any Cowork / Claude-in-Chrome session, after YOU log into OA yourself, say:
> Follow docs/playbooks/OA_BOT_COPY.md

## Extraction spec (per bot)
Visit each bot in the library + leaderboard top list. Extract to JSON:
{"name","symbol","structure","legs":[{"side","type","delta","dte","ratio"}],
 "trigger":{"interval","time_window"},"conditions":[],"exits":{"tp_pct","sl_pct","time"},
 "allocation_pct","no_position_guard","source_url"}

## Import (no hand-coding)
Save array to oa_dump.json in repo root, then:
  python scripts/import_oa_bots.py oa_dump.json
→ generates backend/app/bots/templates.py entries (1m / 2.5% / no_position
defaults), dedupes against existing oa_* names, runs the template tests, and
opens a PR via the normal branch flow. (Importer script: see IMPROVEMENTS.md
queue item — if not yet built, paste the JSON into a GitHub issue titled
"[oa-import] ..." and the employees implement it from there, as done for the
original 15.)

## Keeping it current afterwards
Add OA_SESSION_COOKIE repo secret (DevTools → Application → Cookies) —
oa-scout then diffs daily and auto-files new bots. When the cookie expires the
scout posts the auth-wall warning to Discord; re-paste to resume.
