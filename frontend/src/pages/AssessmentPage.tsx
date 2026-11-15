import { useEffect, useState } from 'react'
import { api } from '../api'
import type { AssessmentDetail, AssessmentSummary, CaseSummary } from '../api'
import { CasePicker, EmptyState, KeyValueTable, Panel } from '../components/ui'
import { OUTCOME_TONE, formatOffset, outcomeLabel } from '../format'
import { formatRecordedTime, withSimulatedTimes } from '../utils/dateFormat'
import type { Navigate } from '../routes'

type Props = {
  userId: string | null
  assessmentId: number | null
  cases: CaseSummary[]
  history: AssessmentSummary[]
  modelLoaded: boolean
  navigate: Navigate
  onAssessed: () => Promise<void>
}

// Places each non-null time_to_* value at its real offset on the 0..window axis. Null means not observed.
function TemporalStrip({ temporal, windowMinutes }: { temporal: Record<string, number | null>; windowMinutes: number }) {
  const windowSeconds = windowMinutes * 60
  const markers = Object.entries(temporal).filter(([key]) => key.startsWith('time_to_'))
  return (
    <div className="tstrip">
      <div className="tstrip-axis">
        <span>T</span><span>T+{windowMinutes / 2}m</span><span>T+{windowMinutes}m</span>
      </div>
      {markers.map(([key, value]) => {
        const name = key.replace(/^time_to_/, '').replace(/_seconds$/, '').replace(/_/g, ' ')
        return (
          <div key={key} className="tstrip-row" title={value === null ? `${name}: not observed in window` : `${name}: ${value} s after trigger`}>
            <span className="tstrip-label">{name}</span>
            <div className="tstrip-track">
              {value !== null && (
                <span className="tstrip-marker" style={{ left: `${Math.min(Math.max(value / windowSeconds, 0), 1) * 100}%` }} />
              )}
            </div>
            <span className={`tstrip-value mono ${value === null ? 'muted' : ''}`}>
              {value === null ? '—' : formatOffset(value)}
            </span>
          </div>
        )
      })}
      <p className="footnote">— = not observed in the window. Missing evidence is not evidence of safety.</p>
    </div>
  )
}

export function AssessmentPage({ userId, assessmentId, cases, history, modelLoaded, navigate, onAssessed }: Props) {
  const [detail, setDetail] = useState<AssessmentDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // A run id in the URL wins; otherwise show this case's latest stored run, if any.
  const latestStoredId = userId ? history.find((a) => a.user_id === userId)?.id ?? null : null
  const targetId = assessmentId ?? latestStoredId

  useEffect(() => {
    setError(null)
    if (!userId || targetId === null) {
      setDetail(null)
      return
    }
    if (detail?.id === targetId) return
    setLoading(true)
    api.assessment(targetId)
      .then((d) => {
        if (d.user_id !== userId) throw new Error(`run ${targetId} belongs to ${d.user_id}, not ${userId}`)
        setDetail(d)
      })
      .catch((e) => { setDetail(null); setError(String(e)) })
      .finally(() => setLoading(false))
    // detail is intentionally excluded: it is the result of this effect, not an input.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, targetId])

  const run = async () => {
    if (!userId) return
    setBusy(true)
    setError(null)
    try {
      const result = await api.assess(userId)
      setDetail(result)
      await onAssessed()
      navigate('assessment', userId, result.id)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const picker = <CasePicker caseIds={cases.map((c) => c.user_id)} value={userId} onChange={(id) => navigate('assessment', id)} />
  const runButton = (
    <button className="btn primary" onClick={run} disabled={!userId || busy || !modelLoaded}>
      {busy ? 'Assessing…' : detail ? 'Run new assessment' : 'Run ATO assessment'}
    </button>
  )

  if (!userId) {
    return (
      <div className="page">
        <Panel actions={picker}>
          <EmptyState title="Select a case to assess">Choose one of the 200 held-out demo cases.</EmptyState>
        </Panel>
      </div>
    )
  }

  const response = detail?.response
  const rule = response?.component_evidence.rule_evidence
  const model = response?.component_evidence.model_evidence
  const temporal = response?.component_evidence.temporal_evidence
  const windowMinutes = Number(response?.provenance.prediction_window_minutes ?? 15)
  const tone = response ? OUTCOME_TONE[response.outcome] ?? 'plain' : 'plain'

  return (
    <div className="page">
      <Panel className="assess-bar">
        <div className="assess-bar-inner">
          {picker}
          <div className="assess-bar-meta muted small">
            {detail ? <>Showing stored run <span className="mono">#{detail.id}</span> · {formatRecordedTime(detail.created_at)}</> : 'No stored run for this case yet.'}
          </div>
          <div className="assess-bar-actions">
            <button className="btn ghost" onClick={() => navigate('investigation', userId)}>View timeline</button>
            {runButton}
          </div>
        </div>
      </Panel>

      {error && <div className="alert">{error}</div>}

      {!response ? (
        <Panel>
          <EmptyState title={loading ? 'Loading stored assessment…' : 'Not assessed yet'}>
            {!loading && 'Run the single-case pipeline: baseline → T to T+15 features → rule → reference model → evidence → explainable response.'}
          </EmptyState>
        </Panel>
      ) : (
        <>
          <section className={`outcome-hero tone-hero-${tone}`}>
            <div className="hero-main">
              <div className="eyebrow">Outcome</div>
              <div className="hero-title">{outcomeLabel(response.outcome)}</div>
              <code>{response.outcome}</code>
            </div>
            <div className="hero-cell">
              <div className="eyebrow">Evidence status</div>
              <div className="hero-value">{response.evidence_status}</div>
            </div>
            <div className="hero-cell">
              <div className="eyebrow">Simulated response</div>
              <div className="hero-value">{response.recommended_action ?? 'No recommendation'}</div>
            </div>
            <div className="hero-cell">
              <div className="eyebrow">Risk score / level</div>
              <div className="hero-value">{response.risk_score_0_100 === null ? 'Not defined' : response.risk_score_0_100} / {response.risk_level}</div>
            </div>
          </section>

          <Panel title="Explanation">
            <p className="explanation">{response.explanation}</p>
            <div className="chips">{response.reason_codes.map((code) => <span key={code} className="chip">{code}</span>)}</div>
          </Panel>

          <div className="grid-2">
            <Panel title="Rule evidence · Phase 7 multi-signal rule">
              {rule ? (
                <>
                  <div className="evidence-line">
                    <span className="eyebrow">Rule outcome</span>
                    <strong>{rule.prediction}</strong>
                  </div>
                  <div className="evidence-line">
                    <span className="eyebrow">Matched rule</span>
                    <span className="mono">{rule.matched_rules.join(', ')}</span>
                  </div>
                  <p className="muted small">{rule.explanation}</p>
                  <div className="signals">
                    {Object.entries(rule.feature_evidence).map(([key, value]) => (
                      <div key={key} className={`signal ${Number(value) > 0 ? 'on' : ''}`}>
                        <span className="signal-name">{key.replace(/_/g, ' ')}</span>
                        <span className="signal-value mono">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : <EmptyState title="Rule evidence missing" />}
            </Panel>

            <Panel title="Model evidence · Phase 8 reference Random Forest">
              {model ? (
                <>
                  <div className="evidence-line">
                    <span className="eyebrow">Model prediction</span>
                    <strong>{model.prediction}</strong>
                    <span className="muted small">{model.prediction === 1 ? '(risk indicator)' : '(no indicator)'}</span>
                  </div>
                  <div className="evidence-line">
                    <span className="eyebrow">Uncalibrated model output</span>
                    <strong className="mono">{model.probability_ato.toFixed(2)}</strong>
                  </div>
                  <div className="caveat">
                    Random Forest <code>predict_proba</code> output. It is <strong>not calibrated</strong>, is not a
                    probability of account takeover, and is not used as a risk score.
                  </div>
                  <p className="muted small">Model id: <span className="mono">{String(response.provenance.model_id ?? '—')}</span></p>
                </>
              ) : <EmptyState title="Model evidence missing" />}
            </Panel>

            <Panel title={`Temporal evidence · time after trigger (T to T+${windowMinutes} min)`}>
              {temporal ? (
                <>
                  <TemporalStrip temporal={temporal} windowMinutes={windowMinutes} />
                  <KeyValueTable data={Object.fromEntries(Object.entries(temporal).filter(([k]) => !k.startsWith('time_to_')))} />
                </>
              ) : <EmptyState title="Temporal evidence missing" />}
            </Panel>

            <Panel title="Simulated response">
              <div className={`response-box tone-border-${tone}`}>
                <div className="eyebrow">Recommended action</div>
                <div className="response-action">{response.recommended_action ?? 'No recommendation'}</div>
              </div>
              <ul className="response-notes">
                <li>Prototype output only: no real account is blocked, challenged or changed.</li>
                <li>Only corroborated or model-only risk indicators produce a recommendation.</li>
                <li>Ambiguous, conflicting, insufficient or no-indicator evidence produces none, and is not treated as legitimate or safe.</li>
              </ul>
            </Panel>
          </div>

          <details className="panel advanced">
            <summary>Advanced details · feature snapshot ({Object.keys(detail.feature_snapshot).length} fields)</summary>
            <KeyValueTable data={detail.feature_snapshot} />
          </details>
          <details className="panel advanced">
            <summary>Advanced details · pre-trigger baseline profile</summary>
            <KeyValueTable data={withSimulatedTimes(detail.baseline_profile)} />
          </details>
          <details className="panel advanced">
            <summary>Advanced details · provenance</summary>
            <KeyValueTable data={{
              stored_run_id: detail.id,
              pipeline_assessment_id: detail.pipeline_assessment_id,
              created_at_ist: formatRecordedTime(detail.created_at),
              created_at_stored_utc: detail.created_at,
              ...Object.fromEntries(Object.entries(response.provenance).map(([k, v]) => [k, Array.isArray(v) ? v.join(', ') : v])),
            }} />
          </details>
        </>
      )}
    </div>
  )
}
