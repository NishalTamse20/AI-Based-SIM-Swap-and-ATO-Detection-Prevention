import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import type { AssessmentSummary, CaseSummary, Health } from './api'
import { Sidebar } from './components/Sidebar'
import { AssessmentPage } from './pages/AssessmentPage'
import { CaseMonitoring } from './pages/CaseMonitoring'
import { History } from './pages/History'
import { Investigation } from './pages/Investigation'
import { Overview } from './pages/Overview'
import { PAGE_TITLES, parseRoute, routeHref } from './routes'
import type { Navigate, Route } from './routes'
import './App.css'

function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash))
  const [health, setHealth] = useState<Health | null>(null)
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [history, setHistory] = useState<AssessmentSummary[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastCase, setLastCase] = useState<string | null>(route.userId)

  const refresh = useCallback(async () => {
    try {
      const [nextCases, nextHistory] = await Promise.all([api.cases(), api.assessments()])
      setCases(nextCases)
      setHistory(nextHistory)
      setError(null)
    } catch (e) {
      setError(`Backend unreachable: ${e}`)
    } finally {
      setLoaded(true)
    }
  }, [])

  useEffect(() => {
    const onHash = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onHash)
    api.health().then(setHealth).catch((e) => setError(`Backend unreachable: ${e}`))
    refresh()
    return () => window.removeEventListener('hashchange', onHash)
  }, [refresh])

  useEffect(() => {
    if (route.userId) setLastCase(route.userId)
    window.scrollTo(0, 0)
  }, [route.page, route.userId])

  const navigate = useCallback<Navigate>((page, userId, assessmentId) => {
    window.location.hash = routeHref(page, userId, assessmentId)
  }, [])

  const caseIds = cases.map((c) => c.user_id)
  const knownCase = route.userId && (!loaded || caseIds.includes(route.userId)) ? route.userId : null
  const unknownCase = loaded && cases.length > 0 && route.userId && !caseIds.includes(route.userId)

  return (
    <div className="shell">
      <Sidebar active={route.page} health={health} lastCase={lastCase} />
      <div className="workspace">
        <header className="topbar">
          <div>
            <div className="crumb">SIM-Swap ATO · Research prototype</div>
            <h1>{PAGE_TITLES[route.page]}</h1>
          </div>
          {health && (
            <div className="topbar-status">
              <span className={`status-chip ${health.model_loaded ? 'ok' : 'bad'}`}>
                <i /> {health.model_loaded ? 'Reference model loaded' : 'Model not loaded'}
              </span>
              <span className={`status-chip ${health.database_backend === 'postgresql' ? 'ok' : 'warn'}`}>
                <i /> DB: {health.database_backend}
              </span>
              <span className="status-chip">Window T → T+{health.prediction_window_minutes} min</span>
            </div>
          )}
        </header>

        <div className="notice">
          Synthetic data only. Outcomes are research indicators, not fraud verdicts; all responses are simulated.
          No validated risk score or level exists, and the model output is uncalibrated.
        </div>
        {health?.model_error && <div className="alert">Model error: {health.model_error}</div>}
        {error && <div className="alert">{error}</div>}
        {unknownCase && (
          <div className="alert">{route.userId} is not one of the 200 held-out demo cases.</div>
        )}

        <main className="content">
          {route.page === 'overview' && (
            <Overview health={health} cases={cases} history={history} loaded={loaded} navigate={navigate} />
          )}
          {route.page === 'cases' && <CaseMonitoring cases={cases} loaded={loaded} navigate={navigate} />}
          {route.page === 'investigation' && (
            <Investigation userId={knownCase} cases={cases} navigate={navigate} />
          )}
          {route.page === 'assessment' && (
            <AssessmentPage userId={knownCase} assessmentId={route.assessmentId} cases={cases} history={history}
              modelLoaded={!!health?.model_loaded} navigate={navigate} onAssessed={refresh} />
          )}
          {route.page === 'history' && <History history={history} loaded={loaded} navigate={navigate} />}
        </main>
      </div>
    </div>
  )
}

export default App
