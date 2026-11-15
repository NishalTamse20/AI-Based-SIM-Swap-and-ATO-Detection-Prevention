// Types mirror backend/app/main.py responses. All data is synthetic; actions are simulated.

export type Health = {
  status: string
  database_backend: string
  model_loaded: boolean
  model_error: string | null
  model_id: string | null
  model_sha256: string | null
  prediction_window_minutes: number
  disclaimer: string
}

export type CaseSummary = {
  user_id: string
  account_id: string
  trigger_type: string
  trigger_time: string
  event_count: number
  latest_outcome: string | null
}

export type CaseEvent = {
  event_id: string
  event_category: string
  event_type: string
  timestamp: string
  device_id: string | null
  channel: string | null
  status: string | null
  amount: string | null
  beneficiary_id: string | null
  transaction_type: string | null
  source: string | null
  phase: 'PRE_TRIGGER' | 'TRIGGER' | 'FEATURE_WINDOW' | 'AFTER_WINDOW'
}

export type CaseEvents = {
  user_id: string
  trigger_time: string
  prediction_window_minutes: number
  events: CaseEvent[]
}

export type AssessmentSummary = {
  id: number
  user_id: string
  created_at: string
  outcome: string
  evidence_status: string
  recommended_action: string | null
  risk_level: string
  risk_score_0_100: number | null
}

type Scalar = string | number | boolean | null

export type ExplainableResponse = {
  assessment_id: string
  identifiers: Record<string, string>
  outcome: string
  evidence_status: string
  reason_codes: string[]
  explanation: string
  risk_score_0_100: number | null
  risk_level: string
  recommended_action: string | null
  component_evidence: {
    rule_evidence: {
      prediction: string
      matched_rules: string[]
      feature_evidence: Record<string, Scalar>
      explanation: string
    } | null
    model_evidence: { prediction: number; probability_ato: number } | null
    temporal_evidence: Record<string, number | null> | null
  }
  provenance: Record<string, Scalar | string[]>
}

export type AssessmentDetail = AssessmentSummary & {
  pipeline_assessment_id: string
  pipeline_version: string
  model_id: string | null
  model_sha256: string | null
  response: ExplainableResponse
  feature_snapshot: Record<string, Scalar>
  temporal_evidence: Record<string, number | null>
  baseline_profile: Record<string, Scalar>
  disclaimer: string
}

// The backend returns the newest runs first; the UI flags when this cap is reached.
export const HISTORY_LIMIT = 1000

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      // keep the status text
    }
    throw new Error(`${response.status}: ${detail}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/health'),
  cases: () => request<CaseSummary[]>('/api/cases'),
  events: (userId: string) => request<CaseEvents>(`/api/cases/${userId}/events`),
  assess: (userId: string) => request<AssessmentDetail>(`/api/cases/${userId}/assess`, { method: 'POST' }),
  assessments: (limit = HISTORY_LIMIT) => request<AssessmentSummary[]>(`/api/assessments?limit=${limit}`),
  assessment: (id: number) => request<AssessmentDetail>(`/api/assessments/${id}`),
}
