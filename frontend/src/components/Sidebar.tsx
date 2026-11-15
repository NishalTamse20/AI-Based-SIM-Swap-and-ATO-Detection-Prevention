import type { Health } from '../api'
import { routeHref } from '../routes'
import type { PageKey } from '../routes'

const NAV: { page: PageKey; label: string; icon: string; needsCase?: boolean }[] = [
  { page: 'overview', label: 'Overview', icon: 'M3 3h7v7H3zM14 3h7v4h-7zM14 11h7v10h-7zM3 14h7v7H3z' },
  { page: 'cases', label: 'Case Monitoring', icon: 'M4 5h16M4 10h16M4 15h16M4 20h10' },
  { page: 'investigation', label: 'Investigation', icon: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM16.5 16.5 21 21', needsCase: true },
  { page: 'assessment', label: 'ATO Assessment', icon: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z', needsCase: true },
  { page: 'history', label: 'History', icon: 'M12 7v5l3 2M3.5 12a8.5 8.5 0 1 0 2.5-6M3 4v4h4' },
]

export function Sidebar({ active, health, lastCase }: { active: PageKey; health: Health | null; lastCase: string | null }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2l9 4v6c0 5.5-3.8 9.3-9 10-5.2-.7-9-4.5-9-10V6z" /></svg>
        <div>
          <div className="brand-name">SIM-ATO Guard</div>
          <div className="brand-sub">Analyst console</div>
        </div>
      </div>
      <nav>
        {NAV.map((item) => (
          <a key={item.page} href={routeHref(item.page, item.needsCase ? lastCase : null)}
            className={active === item.page ? 'active' : ''}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d={item.icon} /></svg>
            <span>{item.label}</span>
          </a>
        ))}
      </nav>
      <div className="sidebar-foot">
        <div className="foot-label">Active case</div>
        <div className="foot-value">{lastCase ?? 'None selected'}</div>
        <div className="foot-label">Model</div>
        <div className="foot-value mono">{health?.model_sha256 ? health.model_sha256.slice(0, 12) : '—'}</div>
      </div>
    </aside>
  )
}
