import { HISTORY_LIMIT } from '../api'
import type { AssessmentSummary, CaseSummary, Health } from '../api'
import { BarList, EmptyState, OutcomeBadge, Panel, StatTile } from '../components/ui'
import { OUTCOMES, OUTCOME_LABELS, OUTCOME_TONE, TRIGGER_LABELS } from '../format'
import { formatRecordedTime } from '../utils/dateFormat'
import type { Navigate } from '../routes'

type Props = { health: Health | null; cases: CaseSummary[]; history: AssessmentSummary[]; loaded: boolean; navigate: Navigate }

// Every figure here is a count of API data (demo cases and runs stored in this database), not an evaluation metric.
export function Overview({ health, cases, history, loaded, navigate }: Props) {
  const assessed = cases.filter((c) => c.latest_outcome)
  const runsLabel = history.length >= HISTORY_LIMIT ? `${HISTORY_LIMIT}+` : String(history.length)
  const simulated = history.filter((a) => a.recommended_action).length
  const ambiguous = assessed.filter((c) => c.latest_outcome === 'AMBIGUOUS').length

  const outcomeItems = OUTCOMES.map((outcome) => ({
    key: outcome,
    label: OUTCOME_LABELS[outcome],
    value: assessed.filter((c) => c.latest_outcome === outcome).length,
    tone: OUTCOME_TONE[outcome],
  })).filter((item) => item.value > 0)

  const triggerItems = [...new Set(cases.map((c) => c.trigger_type))].sort().map((trigger) => ({
    key: trigger,
    label: TRIGGER_LABELS[trigger] ?? trigger,
    value: cases.filter((c) => c.trigger_type === trigger).length,
  }))

  return (
    <div className="page">
      <div className="stats">
        <StatTile label="Demo cases" value={loaded ? cases.length : '…'} note="Held-out Phase 8 test split" />
        <StatTile label="Cases assessed" value={loaded ? `${assessed.length} / ${cases.length}` : '…'}
          note="At least one stored run" />
        <StatTile label="Assessment runs stored" value={loaded ? runsLabel : '…'} note={`Database: ${health?.database_backend ?? '—'}`} />
        <StatTile label="Simulated responses" value={loaded ? simulated : '…'} note="Runs with a recommended action" />
        <StatTile label="Ambiguous cases" value={loaded ? ambiguous : '…'} note="Latest outcome; preserved, not resolved" />
      </div>

      <div className="grid-2">
        <Panel title="Latest outcome per assessed case">
          {assessed.length === 0 ? (
            <EmptyState title="No cases assessed yet">Run an assessment from Case Monitoring to populate this view.</EmptyState>
          ) : (
            <>
              <BarList items={outcomeItems} total={assessed.length} unit="cases" />
              <p className="footnote">Counts of stored pipeline outcomes for the {assessed.length} assessed demo cases.
                These are not detection-performance metrics.</p>
            </>
          )}
        </Panel>
        <Panel title={`Telecom trigger mix (${cases.length} cases)`}>
          {cases.length === 0 ? <EmptyState title="No cases loaded" /> : (
            <>
              <BarList items={triggerItems} total={cases.length} unit="cases" />
              <p className="footnote">Trigger event type of each demo case. A SIM/eSIM change alone is not evidence of fraud.</p>
            </>
          )}
        </Panel>
      </div>

      <div className="grid-overview">
        <Panel title="Recent activity" actions={<a className="link" href="#/history">View all</a>}>
          {history.length === 0 ? <EmptyState title="No assessment runs yet" /> : (
            <table className="table">
              <thead><tr><th>Time</th><th>Case</th><th>Outcome</th><th>Simulated response</th></tr></thead>
              <tbody>
                {history.slice(0, 8).map((a) => (
                  <tr key={a.id} className="clickable" onClick={() => navigate('assessment', a.user_id, a.id)}>
                    <td className="mono">{formatRecordedTime(a.created_at)}</td>
                    <td className="mono">{a.user_id}</td>
                    <td><OutcomeBadge outcome={a.outcome} /></td>
                    <td>{a.recommended_action ?? <span className="muted">None</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>
        <Panel title="System status">
          <dl className="status-list">
            <dt>API</dt><dd>{health ? health.status : 'unreachable'}</dd>
            <dt>Reference model</dt><dd>{health?.model_id ?? '—'}</dd>
            <dt>Model SHA-256</dt><dd className="mono">{health?.model_sha256 ? `${health.model_sha256.slice(0, 16)}…` : '—'}</dd>
            <dt>Database</dt><dd>{health?.database_backend ?? '—'}</dd>
            <dt>Feature window</dt><dd>T to T+{health?.prediction_window_minutes ?? '—'} min (inclusive)</dd>
            <dt>Risk score</dt><dd>Not defined (UNDETERMINED)</dd>
          </dl>
        </Panel>
      </div>
    </div>
  )
}
