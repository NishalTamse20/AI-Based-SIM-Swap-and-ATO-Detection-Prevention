import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CaseEvent, CaseEvents, CaseSummary } from '../api'
import { CasePicker, EmptyState, OutcomeBadge, Panel } from '../components/ui'
import { PHASE_LABELS, TRIGGER_LABELS, formatOffset, secondsBetween } from '../format'
import { SIMULATED_CLOCK_NOTE, formatSimulatedDate, formatSimulatedTime } from '../utils/dateFormat'
import type { Navigate } from '../routes'

type Props = { userId: string | null; cases: CaseSummary[]; navigate: Navigate }

function EventNode({ event, triggerTime }: { event: CaseEvent; triggerTime: string }) {
  const offset = secondsBetween(triggerTime, event.timestamp)
  const details = [
    event.device_id && `device ${event.device_id}`,
    event.channel && `channel ${event.channel}`,
    event.transaction_type && `type ${event.transaction_type}`,
    event.amount && `amount ${event.amount}`,
    event.beneficiary_id && `beneficiary ${event.beneficiary_id}`,
  ].filter(Boolean)
  return (
    <li className={`tl-item tl-${event.phase}`}>
      <div className="tl-time">
        <span className="mono">{event.phase === 'PRE_TRIGGER' ? formatSimulatedDate(event.timestamp) : formatOffset(offset)}</span>
      </div>
      <div className="tl-dot" />
      <div className="tl-card">
        <div className="tl-head">
          <span className={`cat cat-${event.event_category}`}>{event.event_category}</span>
          <strong>{event.event_type}</strong>
          {event.status && <span className={`tl-status ${event.status === 'FAILED' || event.status === 'FAILURE' ? 'failed' : ''}`}>{event.status}</span>}
        </div>
        <div className="tl-meta">
          <span className="mono">{event.event_id}</span>
          <span className="mono">{formatSimulatedTime(event.timestamp)}</span>
          {details.map((d) => <span key={d as string}>{d}</span>)}
        </div>
      </div>
    </li>
  )
}

export function Investigation({ userId, cases, navigate }: Props) {
  const [data, setData] = useState<CaseEvents | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showBaseline, setShowBaseline] = useState(false)

  useEffect(() => {
    setData(null)
    setError(null)
    setShowBaseline(false)
    if (userId) api.events(userId).then(setData).catch((e) => setError(String(e)))
  }, [userId])

  const summary = cases.find((c) => c.user_id === userId)
  const picker = <CasePicker caseIds={cases.map((c) => c.user_id)} value={userId} onChange={(id) => navigate('investigation', id)} />

  if (!userId) {
    return (
      <div className="page">
        <Panel actions={picker}>
          <EmptyState title="Select a case to investigate">
            Choose one of the 200 held-out demo cases, or open one from Case Monitoring.
          </EmptyState>
        </Panel>
      </div>
    )
  }

  const events = data?.events ?? []
  const byPhase = (phase: CaseEvent['phase']) => events.filter((e) => e.phase === phase)
  const baseline = byPhase('PRE_TRIGGER')
  const trigger = byPhase('TRIGGER')
  const inWindow = byPhase('FEATURE_WINDOW')
  const after = byPhase('AFTER_WINDOW')
  const triggerEvent = trigger[0]

  return (
    <div className="page">
      <Panel className="case-header" actions={picker}>
        <div className="case-header-grid">
          <div>
            <div className="eyebrow">Case</div>
            <div className="case-id mono">{userId}</div>
            <div className="muted small mono">{summary?.account_id}</div>
          </div>
          <div>
            <div className="eyebrow">Telecom trigger</div>
            {triggerEvent ? (
              <>
                <span className={`trigger-tag trigger-${triggerEvent.event_type}`}>{triggerEvent.event_type}</span>
                <div className="muted small">{TRIGGER_LABELS[triggerEvent.event_type] ?? ''} at {formatSimulatedTime(triggerEvent.timestamp)}</div>
              </>
            ) : <span className="muted">—</span>}
          </div>
          <div>
            <div className="eyebrow">Events by phase</div>
            <div className="phase-counts">
              <span>{baseline.length} baseline</span>
              <span className="hl">{inWindow.length} in window</span>
              <span>{after.length} after</span>
            </div>
          </div>
          <div>
            <div className="eyebrow">Latest outcome</div>
            <OutcomeBadge outcome={summary?.latest_outcome ?? null} />
          </div>
          <div className="case-header-cta">
            <button className="btn primary" onClick={() => navigate('assessment', userId)}>Open ATO assessment →</button>
          </div>
        </div>
      </Panel>

      {error && <div className="alert">{error}</div>}

      <Panel title="Event timeline" actions={
        <span className="legend">
          <span className="lg lg-trigger">Trigger</span>
          <span className="lg lg-window">Feature window (T to T+{data?.prediction_window_minutes ?? 15} min)</span>
          <span className="lg lg-after">After window</span>
        </span>
      }>
        {!data ? <EmptyState title="Loading events…" /> : (
          <ol className="timeline">
            <li className="tl-group">
              <button className="btn ghost small" onClick={() => setShowBaseline((v) => !v)}>
                {showBaseline ? 'Hide' : 'Show'} baseline history ({baseline.length} events before T)
              </button>
              <span className="muted small">Used only to build the pre-trigger behavioural baseline. {SIMULATED_CLOCK_NOTE}</span>
            </li>
            {showBaseline && baseline.map((e) => <EventNode key={e.event_id} event={e} triggerTime={data.trigger_time} />)}
            {trigger.map((e) => <EventNode key={e.event_id} event={e} triggerTime={data.trigger_time} />)}
            <li className="tl-group tl-window-label"><span>{PHASE_LABELS.FEATURE_WINDOW}: evidence used by rules, model and temporal features</span></li>
            {inWindow.length === 0 && <li className="tl-group muted small">No events in the feature window.</li>}
            {inWindow.map((e) => <EventNode key={e.event_id} event={e} triggerTime={data.trigger_time} />)}
            {after.length > 0 && (
              <li className="tl-group tl-after-label"><span>{PHASE_LABELS.AFTER_WINDOW}: not seen by rules or model</span></li>
            )}
            {after.map((e) => <EventNode key={e.event_id} event={e} triggerTime={data.trigger_time} />)}
          </ol>
        )}
      </Panel>
    </div>
  )
}
