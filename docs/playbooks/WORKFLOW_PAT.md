# Playbook: WORKFLOW_PAT (improver PRs self-merge)
1. github.com → Settings → Developer settings → Tokens (classic) → Generate:
   scopes `repo` + `workflow`, 90-day expiry.
2. Repo → Settings → Secrets → Actions → new secret `WORKFLOW_PAT`.
3. Task for teams: auto-pr.yml + continuous-improvement.yml switch to
   `${{ secrets.WORKFLOW_PAT || secrets.GITHUB_TOKEN }}` when present —
   PAT-created PRs trigger CI, so the automerge gate can land them unattended.
