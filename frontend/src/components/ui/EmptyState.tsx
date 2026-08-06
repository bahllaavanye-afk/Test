/**
 * Why a panel is empty — instead of an empty panel, or worse, advice that cannot work.
 *
 * Audited 2026-08-06. Several pages already had empty states, and the problem
 * was not silence but MISDIRECTION:
 *
 *   Experiments → "Click 'Train Model' above to queue your first training run"
 *   Comparison  → "run a strategy comparison in Backtest Lab"
 *
 * Both tell the operator to do something that cannot help. The backend runs on
 * an ephemeral SQLite fallback, so rows are wiped on every redeploy; and ML
 * training needs PyTorch, which is deliberately excluded from the deployment
 * image. Following either instruction produces the same empty table again.
 *
 * A panel that blames the user for a platform-level cause is the UI version of
 * the defect this codebase keeps finding: output that does not depend on the
 * thing it claims to report.
 */

// STATUS OF EACH REASON, 2026-08-06. Two are proven against live evidence; two
// are not, and saying so matters in a component whose job is stating true causes.
//
//   ephemeral-db          USED + verified — /health/detailed reports
//                         database.fallback = "sqlite", database_primary.ok = false.
//   ml-runtime            USED + verified — torch sits in an optional pyproject
//                         group and `import torch` fails in the deploy env.
//   no-rows-yet           unused. Harmless, but unexercised.
//   subsystem-unreachable UNUSED AND UNPROVEN. The one subsystem recorded as
//                         unreachable (the agent dashboard) turned out to be
//                         working: routes registered, skill_library.json holding
//                         200 skills, task_registry genuinely empty. Applying this
//                         reason on the strength of that entry would have rendered
//                         a false cause. Before anyone uses it, re-check that the
//                         subsystem is actually unreachable.
export type EmptyReason =
  | 'ephemeral-db'
  | 'no-rows-yet'
  | 'subsystem-unreachable'
  | 'ml-runtime'

const COPY: Record<EmptyReason, { title: string; detail: string }> = {
  'ephemeral-db': {
    title: 'No stored history yet',
    detail:
      'The backend is on its ephemeral SQLite fallback, so records are wiped on every redeploy. ' +
      'History persists here once the durable Postgres database is connected.',
  },
  'no-rows-yet': {
    title: 'Nothing recorded yet',
    detail: 'This fills in as the desks trade. Empty because there is no data — not because it failed.',
  },
  'subsystem-unreachable': {
    title: 'Subsystem not reporting',
    detail:
      'The endpoints behind this panel return no data. A known gap, not a failure of the page itself.',
  },
  'ml-runtime': {
    title: 'ML runtime unavailable',
    detail:
      'Inference needs PyTorch, deliberately excluded from the deployment image to stay inside the free ' +
      'tier. Nothing is broken — there is no runtime here to report on.',
  },
}

export default function EmptyState({
  reason,
  note,
  className = '',
}: {
  reason: EmptyReason
  /** Optional page-specific sentence, appended after the shared explanation. */
  note?: string
  className?: string
}) {
  const copy = COPY[reason]
  return (
    <div
      data-testid="empty-state"
      data-reason={reason}
      className={`px-5 py-8 text-center ${className}`}
    >
      <p className="text-sm text-[#e8e8e8]">{copy.title}</p>
      <p className="text-xs text-[#888888] mt-1 max-w-xl mx-auto leading-relaxed">{copy.detail}</p>
      {note && <p className="text-xs text-[#555] mt-2 max-w-xl mx-auto">{note}</p>}
    </div>
  )
}
