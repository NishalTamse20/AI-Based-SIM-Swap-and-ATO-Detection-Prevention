# Presentation Demo Runbook

End-to-end demo of the single-case assessment pipeline:
React dashboard (Vite, port 5173) → FastAPI (port 8010) → `ml.pipeline.assess_case` with the fixed Phase 8 reference model → result stored in PostgreSQL (SQLite fallback).

The API adds no ML or decision logic. Outcomes, reason codes and simulated actions come from the existing pipeline unchanged; `risk_score_0_100` stays null and `risk_level` stays `UNDETERMINED`.

## Prerequisites (one-time)

- `.venv` with `backend/requirements.txt` installed. The numpy/pandas/scikit-learn/joblib pins must not change, or the reference model refuses to load.
- `ml/artifacts/phase8_reference_model.joblib` and `.json` present. They are git-ignored; copy them to the presentation laptop, or rebuild with `python -m ml.models.reference_model`.
- `frontend/node_modules` installed (`npm install` in `frontend/`).

### PostgreSQL (optional; SQLite fallback otherwise)

1. Install (you set the `postgres` superuser password during setup): `winget install PostgreSQL.PostgreSQL.17`
2. Create the database: `& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -c "CREATE DATABASE sim_ato_demo;"`
3. Before starting the API, in the same terminal: `$env:DATABASE_URL = "postgresql+psycopg://postgres:<your-password>@127.0.0.1:5432/sim_ato_demo"`

Tables (`demo_cases`, `assessments`) are created automatically at API startup. If `DATABASE_URL` is unset or unreachable, the API logs a warning and uses `backend/demo_fallback.db`. The dashboard header always shows the active backend (`DB: postgresql` or `DB: sqlite-fallback`).

## Start (two terminals, from the repository root)

```
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8010
```

```
npm --prefix frontend run dev
```

Open http://localhost:5173. Port 8000 is not used because Splunk occupies it on the development laptop.

## Pre-flight check

- http://127.0.0.1:8010/api/health shows `"status": "ok"`, `"model_loaded": true` and the intended `database_backend`.
- The case list shows "200 of 200 held-out synthetic cases".

## Suggested walkthrough (about 5 minutes)

The example users were chosen by joining evaluation labels after assessment (reporting only); the dashboard never shows or uses labels.

| Step | Case | Expected outcome | Point to make |
|---|---|---|---|
| 1 | `USR-000003` (SIM_SWAP) | `CORROBORATED_RISK_INDICATOR`, simulated Protect/Security Review | New device and failed logins within T+15; rule M2 and model agree. Simulated action only. |
| 2 | `USR-000785` (SIM_REPLACEMENT) | `NO_ATO_RISK_INDICATOR`, no recommendation | No configured rule matched and the model is negative. This is not "legitimate". |
| 3 | `USR-000632` (SIM_REPLACEMENT) | `AMBIGUOUS`, model prediction 1 | Recovery-pattern evidence is ambiguous; the positive model output is kept but does not override it. |
| 4 | `USR-000033` (SIM_REPLACEMENT) | `AMBIGUOUS` | Disclosed limitation: suspicious SIM replacement looks the same as legitimate recovery within T+15, so it is not flagged. Tick "Show pre-trigger baseline history" and point out post-window events marked "After window (not used)". |
| 5 | Assessment history tab | Stored rows | Each run is persisted with pipeline version and model SHA-256 for provenance. |

## Fallback if something fails on the day

- Postgres down → the API falls back to SQLite automatically; nothing else changes.
- Model fails to load → `/api/health` shows `degraded` with `model_error`; the "Run assessment" button is disabled. Check library versions and the artifact files.
- Frontend issue → use http://127.0.0.1:8010/docs (FastAPI Swagger UI) and call `POST /api/cases/{user_id}/assess` directly.

## Statements to avoid

- Do not call outputs fraud verdicts or call `NO_ATO_RISK_INDICATOR` "legitimate".
- Do not present the model output as a probability or a risk score; no validated 0–100 score or Low/Medium/High level exists.
- Do not claim real telecom/banking integration or real-world performance; all data is synthetic and actions are simulated.
