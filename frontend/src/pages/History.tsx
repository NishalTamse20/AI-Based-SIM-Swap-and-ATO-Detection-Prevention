import { useState } from 'react'
import { HISTORY_LIMIT } from '../api'
import type { AssessmentSummary } from '../api'
import { EmptyState, OutcomeBadge, Panel } from '../components/ui'
import { OUTCOMES, OUTCOME_LABELS } from '../format'
import { formatRecordedTime } from '../utils/dateFormat'
import type { Navigate } from '../routes'

export function History({ history, loaded, navigate }: { history: AssessmentSummary[]; loaded: boolean; navigate: Navigate }) {
  const [outcome, setOutcome] = useState('ALL')
  const [search, setSearch] = useState('')
  const term = search.trim().toUpperCase()
  const rows = history.filter((a) => (outcome === 'ALL' || a.outcome === outcome) && (!term || a.user_id.includes(term)))

  return (
    <div className="page">
      <Panel className="toolbar-panel">
        <div className="toolbar">
          <input type="search" placeholder="Search user id" value={search} onChange={(e) => setSearch(e.target.value)} />
          <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
            <option value="ALL">All outcomes</option>
            {OUTCOMES.map((o) => <option key={o} value={o}>{OUTCOME_LABELS[o]}</option>)}
          </select>
          <span className="muted small toolbar-count">
            {rows.length} of {history.length}{history.length >= HISTORY_LIMIT ? '+' : ''} stored runs
          </span>
        </div>
      </Panel>
      <Panel>
        {!loaded ? <EmptyState title="Loading history…" /> : rows.length === 0 ? (
          <EmptyState title={history.length === 0 ? 'No assessment runs stored yet' : 'No runs match these filters'} />
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr><th>Run</th><th>Time</th><th>Case</th><th>Outcome</th><th className="hide-md">Evidence status</th><th>Simulated response</th><th>Risk level</th></tr>
              </thead>
              <tbody>
                {rows.map((a) => (
                  <tr key={a.id} className="clickable" onClick={() => navigate('assessment', a.user_id, a.id)}>
                    <td className="mono">#{a.id}</td>
                    <td className="mono">{formatRecordedTime(a.created_at)}</td>
                    <td className="mono strong">{a.user_id}</td>
                    <td><OutcomeBadge outcome={a.outcome} /></td>
                    <td className="mono small hide-md">{a.evidence_status}</td>
                    <td>{a.recommended_action ?? <span className="muted">None</span>}</td>
                    <td className="muted">{a.risk_level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
