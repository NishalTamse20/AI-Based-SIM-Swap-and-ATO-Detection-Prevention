import { useMemo, useState } from 'react'
import type { CaseSummary } from '../api'
import { EmptyState, OutcomeBadge, Panel } from '../components/ui'
import { OUTCOMES, OUTCOME_LABELS, TRIGGER_LABELS, outcomeLabel } from '../format'
import { SIMULATED_CLOCK_NOTE, formatSimulatedTime } from '../utils/dateFormat'
import type { Navigate } from '../routes'

type SortKey = 'user_id' | 'trigger_type' | 'trigger_time' | 'event_count' | 'latest_outcome'
const PAGE_SIZE = 25

export function CaseMonitoring({ cases, loaded, navigate }: { cases: CaseSummary[]; loaded: boolean; navigate: Navigate }) {
  const [search, setSearch] = useState('')
  const [trigger, setTrigger] = useState('ALL')
  const [outcome, setOutcome] = useState('ALL')
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'user_id', dir: 1 })
  const [page, setPage] = useState(0)

  const triggers = useMemo(() => [...new Set(cases.map((c) => c.trigger_type))].sort(), [cases])

  const rows = useMemo(() => {
    const term = search.trim().toUpperCase()
    const filtered = cases.filter((c) =>
      (!term || c.user_id.includes(term) || c.account_id.includes(term)) &&
      (trigger === 'ALL' || c.trigger_type === trigger) &&
      (outcome === 'ALL' || (outcome === 'NONE' ? !c.latest_outcome : c.latest_outcome === outcome)))
    const value = (c: CaseSummary) => (sort.key === 'latest_outcome' ? outcomeLabel(c.latest_outcome) : c[sort.key])
    return [...filtered].sort((a, b) => (value(a) > value(b) ? 1 : value(a) < value(b) ? -1 : 0) * sort.dir)
  }, [cases, search, trigger, outcome, sort])

  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const current = Math.min(page, pages - 1)
  const visible = rows.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE)

  const header = (key: SortKey, label: string) => (
    <th className="sortable" onClick={() => setSort((s) => ({ key, dir: s.key === key ? (-s.dir as 1 | -1) : 1 }))}>
      {label}{sort.key === key && <span className="sort">{sort.dir === 1 ? ' ▲' : ' ▼'}</span>}
    </th>
  )

  return (
    <div className="page">
      <Panel className="toolbar-panel">
        <div className="toolbar">
          <input type="search" placeholder="Search user or account id" value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0) }} />
          <select value={trigger} onChange={(e) => { setTrigger(e.target.value); setPage(0) }}>
            <option value="ALL">All triggers</option>
            {triggers.map((t) => <option key={t} value={t}>{TRIGGER_LABELS[t] ?? t}</option>)}
          </select>
          <select value={outcome} onChange={(e) => { setOutcome(e.target.value); setPage(0) }}>
            <option value="ALL">All outcomes</option>
            <option value="NONE">Not assessed</option>
            {OUTCOMES.map((o) => <option key={o} value={o}>{OUTCOME_LABELS[o]}</option>)}
          </select>
          <span className="muted small toolbar-count">{rows.length} of {cases.length} held-out synthetic cases · <span title={SIMULATED_CLOCK_NOTE}>times in IST</span></span>
        </div>
      </Panel>

      <Panel>
        {!loaded ? <EmptyState title="Loading cases…" /> : rows.length === 0 ? (
          <EmptyState title="No cases match these filters" />
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  {header('user_id', 'User')}
                  <th className="hide-md">Account</th>
                  {header('trigger_type', 'Trigger')}
                  {header('trigger_time', 'Trigger time')}
                  {header('event_count', 'Events')}
                  {header('latest_outcome', 'Latest outcome')}
                  <th className="right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((c) => (
                  <tr key={c.user_id}>
                    <td className="mono strong">{c.user_id}</td>
                    <td className="mono muted hide-md">{c.account_id}</td>
                    <td><span className={`trigger-tag trigger-${c.trigger_type}`}>{c.trigger_type}</span></td>
                    <td className="mono">{formatSimulatedTime(c.trigger_time)}</td>
                    <td>{c.event_count}</td>
                    <td><OutcomeBadge outcome={c.latest_outcome} /></td>
                    <td className="right nowrap">
                      <button className="btn ghost" onClick={() => navigate('investigation', c.user_id)}>Investigate</button>
                      <button className="btn" onClick={() => navigate('assessment', c.user_id)}>Assess</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="pager">
          <button className="btn ghost" disabled={current === 0} onClick={() => setPage(current - 1)}>Previous</button>
          <span className="muted small">Page {current + 1} of {pages}</span>
          <button className="btn ghost" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>Next</button>
        </div>
      </Panel>
    </div>
  )
}
