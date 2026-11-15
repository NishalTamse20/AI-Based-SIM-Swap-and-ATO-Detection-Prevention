// Display helpers only. Outcome meaning and wording come from the backend pipeline unchanged.

export const OUTCOMES = [
  'CORROBORATED_RISK_INDICATOR',
  'MODEL_ONLY_RISK_INDICATOR',
  'CONFLICTING_EVIDENCE',
  'AMBIGUOUS',
  'INSUFFICIENT_EVIDENCE',
  'NO_ATO_RISK_INDICATOR',
] as const

export const OUTCOME_LABELS: Record<string, string> = {
  CORROBORATED_RISK_INDICATOR: 'Corroborated risk indicator',
  MODEL_ONLY_RISK_INDICATOR: 'Model-only risk indicator',
  CONFLICTING_EVIDENCE: 'Conflicting evidence',
  AMBIGUOUS: 'Ambiguous',
  INSUFFICIENT_EVIDENCE: 'Insufficient evidence',
  NO_ATO_RISK_INDICATOR: 'No configured risk indicator',
}

// Tone drives colour: risk (red), caution (amber), neutral (violet), plain (slate). Never "safe"/green.
export const OUTCOME_TONE: Record<string, 'risk' | 'caution' | 'neutral' | 'plain'> = {
  CORROBORATED_RISK_INDICATOR: 'risk',
  MODEL_ONLY_RISK_INDICATOR: 'caution',
  CONFLICTING_EVIDENCE: 'caution',
  AMBIGUOUS: 'neutral',
  INSUFFICIENT_EVIDENCE: 'neutral',
  NO_ATO_RISK_INDICATOR: 'plain',
}

export const PHASE_LABELS: Record<string, string> = {
  PRE_TRIGGER: 'Baseline history',
  TRIGGER: 'Telecom trigger (T)',
  FEATURE_WINDOW: 'T to T+15 min',
  AFTER_WINDOW: 'After window (not used)',
}

export const TRIGGER_LABELS: Record<string, string> = {
  SIM_SWAP: 'SIM swap',
  SIM_REPLACEMENT: 'SIM replacement',
  ESIM_CHANGE: 'eSIM change',
}

export function outcomeLabel(outcome: string | null): string {
  if (!outcome) return 'Not assessed'
  return OUTCOME_LABELS[outcome] ?? outcome
}

// Clock-time display lives in utils/dateFormat.ts. These two work on elapsed time only.

export function formatOffset(seconds: number): string {
  const sign = seconds < 0 ? '−' : '+'
  const abs = Math.abs(Math.round(seconds))
  const minutes = Math.floor(abs / 60)
  const rest = String(abs % 60).padStart(2, '0')
  return `T${sign}${minutes}:${rest}`
}

export function secondsBetween(fromIso: string, toIso: string): number {
  return (new Date(toIso).getTime() - new Date(fromIso).getTime()) / 1000
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number' && !Number.isInteger(value)) return value.toFixed(4)
  return String(value)
}
