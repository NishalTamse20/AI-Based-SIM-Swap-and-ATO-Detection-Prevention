// Date/time display in Indian Standard Time. Presentation only: API values, sorting and
// temporal calculations keep using the original ISO strings.
//
// Two kinds of timestamps reach the UI, with different origins:
// 1. Synthetic event clocks (event timestamp, trigger_time, baseline_start, trigger_cutoff) are
//    timezone-naive simulation values (simulator/scenarios/scenario_runner.py builds them from
//    datetime(2026, 1, 1) with no zone). Their clock is kept as-is and labelled IST; no offset is applied.
// 2. Assessment run times (created_at) are written by the backend as UTC (backend/app/db.py,
//    datetime.now(timezone.utc)). SQLite drops the offset on read, PostgreSQL keeps it; either
//    way the instant is converted to Asia/Kolkata.

export const DISPLAY_TIMEZONE = 'Asia/Kolkata'
export const SIMULATED_CLOCK_NOTE = 'Event times are synthetic simulation clocks shown as IST; no offset is applied.'

const IST_OFFSET = '+05:30'
const HAS_OFFSET = /(Z|[+-]\d{2}:?\d{2})$/i
const NAIVE_ISO = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/

const dateTimeFormatter = new Intl.DateTimeFormat('en-IN', {
  timeZone: DISPLAY_TIMEZONE,
  hour12: true,
  day: '2-digit',
  month: 'short',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})

const dateFormatter = new Intl.DateTimeFormat('en-IN', {
  timeZone: DISPLAY_TIMEZONE,
  day: '2-digit',
  month: 'short',
  year: 'numeric',
})

function parts(formatter: Intl.DateTimeFormat, date: Date): Record<string, string> {
  return Object.fromEntries(formatter.formatToParts(date).map((p) => [p.type, p.value]))
}

/** "03 Jan 2026, 09:05 AM IST" for an absolute instant. */
function formatInstant(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const p = parts(dateTimeFormatter, date)
  return `${p.day} ${p.month} ${p.year}, ${p.hour}:${p.minute} ${p.dayPeriod.toUpperCase()} IST`
}

/** A naive synthetic clock pinned to IST, so the IST rendering shows the same clock unchanged. */
function simulatedInstant(iso: string): string {
  return HAS_OFFSET.test(iso) ? iso : `${iso.replace(' ', 'T')}${IST_OFFSET}`
}

/** Synthetic event clock: naive values keep their clock and are labelled IST. */
export function formatSimulatedTime(iso: string): string {
  return formatInstant(simulatedInstant(iso))
}

/** Synthetic event date only, e.g. "03 Jan 2026" (used where a full time does not fit). */
export function formatSimulatedDate(iso: string): string {
  const date = new Date(simulatedInstant(iso))
  if (Number.isNaN(date.getTime())) return iso
  const p = parts(dateFormatter, date)
  return `${p.day} ${p.month} ${p.year}`
}

/** Assessment run time: recorded in UTC by the backend, converted to IST. */
export function formatRecordedTime(iso: string): string {
  return formatInstant(HAS_OFFSET.test(iso) ? iso : `${iso.replace(' ', 'T')}Z`)
}

/** Formats any naive ISO strings in a snapshot (e.g. baseline_start) as synthetic IST clocks. */
export function withSimulatedTimes(data: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(data).map(([key, value]) =>
    [key, typeof value === 'string' && NAIVE_ISO.test(value) ? formatSimulatedTime(value) : value]))
}
