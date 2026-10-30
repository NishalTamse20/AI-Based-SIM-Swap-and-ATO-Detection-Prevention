# Single-Case Assessment Pipeline

## Purpose

Assess one SIM/eSIM-triggered case end to end by chaining the existing, tested research components. This is the core that a later API and dashboard will call. It has no FastAPI, database or frontend dependency, adds no decision rule, and produces no risk score.

## Flow

| Step | Component (reused unchanged) | Output |
|---|---|---|
| 0 | Input validation (`ml/pipeline/assessment.py`) | One user/account, exactly one TELECOM trigger, canonical event names (`CATEGORY_EVENT_TYPES`). `ato_label` and `scenario_type` are blanked before any component runs. |
| 1 | Phase 5 `build_baseline_profiles` + `validate_baseline_cutoffs` | Profile from pre-trigger history plus the trigger only. |
| 2 | Phase 6 `build_feature_rows` | 25 features over the inclusive T to T+15 window; target is `None` for unlabelled input. |
| 3 | Phase 9 `build_temporal_feature_rows` | 13 temporal values (numbers or `None`), passed as temporal evidence. |
| 4 | Phase 7 `multi_signal_prediction` | `NO_ATO_RISK_INDICATOR`, `ATO_RISK_INDICATOR` or `AMBIGUOUS_INSUFFICIENT_EVIDENCE`. |
| 5 | Phase 8 reference model (`ml/models/reference_model.py`, `predict_case`) | `{prediction, probability_ato}` on the 21 Phase 8 columns. Probability is uncalibrated. |
| 6 | Phase 10.2 `assemble_evidence` | Evidence status: `AMBIGUOUS`, `CONFLICTING`, `INCOMPLETE` or `ASSEMBLED`. |
| 7 | Phase 11.1 `build_explainable_response` | Outcome, reason codes, explanation, simulated recommendation; `risk_score_0_100 = null`, `risk_level = UNDETERMINED`. |

Entry point: `assess_case(events, model=..., model_metadata=..., assessment_id=None)` returns `response`, `feature_snapshot`, `temporal_evidence` and `baseline_profile`. `load_events_by_user` / `load_case_events` read cases from an event CSV.

## Reference model

The Phase 8 Random Forest is reproduced exactly (same `features.csv` SHA-256, row order, stratified 80/20 split with seed 42, parameters, and numpy 2.4.6 / pandas 3.0.6 / scikit-learn 1.9.1 / Python 3.11.9). Its 200 test-split predictions and probabilities equal the saved Phase 8 outputs with zero difference. The artifact lives in `ml/artifacts/` (git-ignored) and is rebuilt with `python -m ml.models.reference_model`, which refuses to save without exact parity or to overwrite. Loading checks the file hash, library versions and feature list before unpickling.

## Outcome mapping (existing logic)

| Rule | Model | Outcome | Simulated action |
|---|---|---|---|
| M1 ambiguous recovery | any | `AMBIGUOUS` | none |
| M2 risk indicator | 1 | `CORROBORATED_RISK_INDICATOR` | Protect/Security Review |
| M2 risk indicator | 0 | `CONFLICTING_EVIDENCE` | none |
| M3 no match | 1 | `MODEL_ONLY_RISK_INDICATOR` | Step-Up Authentication |
| M3 no match | 0 | `NO_ATO_RISK_INDICATOR` (not "legitimate") | none |
| model missing | — | `INSUFFICIENT_EVIDENCE` | none |

## Verified behaviour

Parity (all exact): Phase 5 profiles, Phase 6 features and Phase 9 temporal values for all 1,000 legacy synthetic users; Phase 7 rule outcomes for all 1,000; rule, model prediction and probability for the 200 Phase 8 test users. Labels and `scenario_type` changed or removed give identical results; extra post-trigger events never change the baseline.

Outcomes on the 200 Phase 8 test users (labels joined only after assessment):

| Scenario type | Rule / model evidence | Outcome |
|---|---|---|
| LEGITIMATE_ESIM_CHANGE (24), LEGITIMATE_SIM_REPLACEMENT (28) | M3 / 0 | `NO_ATO_RISK_INDICATOR` |
| LEGITIMATE_ESIM_DEVICE_RECOVERY (29), LEGITIMATE_SIM_AUTH_RECOVERY (22) | M1 / 0 | `AMBIGUOUS` |
| LEGITIMATE_SIM_DEVICE_RECOVERY (22) | M1 / 1 (12), 0 (10) | `AMBIGUOUS` |
| SUSPICIOUS_SIM_REPLACEMENT (21) | M1 / 1 (12), 0 (9) | `AMBIGUOUS` |
| SIM_ESIM_AUTH_ANOMALY (25), SUSPICIOUS_ESIM_CHANGE (29) | M2 / 1 | `CORROBORATED_RISK_INDICATOR` |

No legitimate-labelled test case received a simulated action. Suspicious SIM replacement cases are not flagged: within T+15 they match the same ambiguous recovery pattern as legitimate SIM device recovery, so the pipeline reports them as `AMBIGUOUS` with no recommendation. This is consistent with the earlier finding that their distinguishing events fall after the T+15 cutoff.

## Decisions

1. Suspicious SIM-replacement cases stay `AMBIGUOUS` when the T..T+15 evidence cannot separate them from legitimate recovery. They are not relabelled or forced into a risk outcome.
2. The initial demo is restricted to the 200 Phase 8 test-split users (listed in the reference model metadata).
3. `assess_case` keeps its internal return structure (`response`, `feature_snapshot`, `temporal_evidence`, `baseline_profile`). A future API should expose only the fields it needs.

## Input validation

Every event must have a non-empty `event_id` and `timestamp`; otherwise `ValueError` names the missing field. `event_id` is the schema's unique event identifier (the dataset validator already rejects duplicates dataset-wide), so a repeated `event_id` within a case is rejected as a duplicate record and cannot be double-counted by the baseline. Distinct events with different ids are still counted normally.

## Limitations

- Outcomes are fully determined by the legacy scenario template; this shows template dependence, not generalisation to unseen attacks.
- 800 of the 1,000 users were in the reference model's training split; their model evidence is in-sample, so the demo uses the 200 test-split users.
- Model loading checks file integrity (hash, versions, feature list) against local metadata; it does not authenticate the artifact's origin. Only load artifacts built locally with `python -m ml.models.reference_model`.
- The model probability is uncalibrated and is not a risk probability; there is no score or level (Phase 11.2).
- Synthetic data only; outputs are prototype indicators and simulated recommendations, not fraud verdicts or account actions.
- The artifact is a version-bound pickle; it only loads under the recorded library versions.
