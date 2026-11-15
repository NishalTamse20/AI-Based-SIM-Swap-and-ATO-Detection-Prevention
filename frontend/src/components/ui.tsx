import type { ReactNode } from 'react'
import { OUTCOME_TONE, formatValue, outcomeLabel } from '../format'

export function OutcomeBadge({ outcome }: { outcome: string | null }) {
  const tone = outcome ? OUTCOME_TONE[outcome] ?? 'plain' : 'none'
  return <span className={`badge tone-${tone}`}>{outcomeLabel(outcome)}</span>
}

export function Panel({ title, actions, children, className = '' }:
  { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          {title && <h3>{title}</h3>}
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function StatTile({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  )
}

export type BarItem = { key: string; label: string; value: number; tone?: string }

// Horizontal bars with direct labels; each row is labelled in text, so colour is never the only cue.
export function BarList({ items, total, unit }: { items: BarItem[]; total: number; unit: string }) {
  const max = Math.max(1, ...items.map((item) => item.value))
  return (
    <ul className="bars" role="list">
      {items.map((item) => {
        const share = total > 0 ? Math.round((item.value / total) * 100) : 0
        return (
          <li key={item.key} title={`${item.label}: ${item.value} ${unit} (${share}%)`}>
            <div className="bar-row-label">
              <span>{item.label}</span>
              <span className="bar-count">{item.value}<span className="muted"> · {share}%</span></span>
            </div>
            <div className="bar-track">
              <div className={`bar-fill tone-fill-${item.tone ?? 'accent'}`} style={{ width: `${(item.value / max) * 100}%` }} />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

export function KeyValueTable({ data, emptyText = '—' }: { data: Record<string, unknown>; emptyText?: string }) {
  return (
    <table className="kv">
      <tbody>
        {Object.entries(data).map(([key, value]) => (
          <tr key={key}>
            <th>{key}</th>
            <td className={value === null ? 'muted' : ''}>{value === null ? emptyText : formatValue(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children && <div className="muted">{children}</div>}
    </div>
  )
}

export function CasePicker({ caseIds, value, onChange }:
  { caseIds: string[]; value: string | null; onChange: (userId: string) => void }) {
  return (
    <label className="case-picker">
      <span className="muted small">Case</span>
      <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
        <option value="" disabled>Select a demo case…</option>
        {caseIds.map((id) => <option key={id} value={id}>{id}</option>)}
      </select>
    </label>
  )
}
