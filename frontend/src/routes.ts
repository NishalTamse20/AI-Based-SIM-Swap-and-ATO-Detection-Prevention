// Hash routes: #/page[/USER_ID][?run=ID]. No router dependency is needed for five pages.

export type PageKey = 'overview' | 'cases' | 'investigation' | 'assessment' | 'history'

export type Route = { page: PageKey; userId: string | null; assessmentId: number | null }

export type Navigate = (page: PageKey, userId?: string | null, assessmentId?: number | null) => void

const PAGES: PageKey[] = ['overview', 'cases', 'investigation', 'assessment', 'history']

export const PAGE_TITLES: Record<PageKey, string> = {
  overview: 'Security Overview',
  cases: 'Case Monitoring',
  investigation: 'Investigation',
  assessment: 'ATO Assessment',
  history: 'Assessment History',
}

export function parseRoute(hash: string): Route {
  const [path, query = ''] = hash.replace(/^#\/?/, '').split('?')
  const [page, userId] = path.split('/')
  const run = new URLSearchParams(query).get('run')
  return {
    page: PAGES.includes(page as PageKey) ? (page as PageKey) : 'overview',
    userId: userId || null,
    assessmentId: run ? Number(run) : null,
  }
}

export function routeHref(page: PageKey, userId?: string | null, assessmentId?: number | null): string {
  return `#/${page}${userId ? `/${userId}` : ''}${assessmentId ? `?run=${assessmentId}` : ''}`
}
