# Phase 7 Rule-Based Baselines

## Scope and inputs

The baselines consume the Phase 6 feature rows at the frozen T+15 snapshot. They return one result per input row with `prediction`, `matched_rules`, `feature_evidence`, and `explanation`. Identifiers may be copied for traceability but do not participate in decisions.

Neither baseline reads `ato_label`, `scenario_type`, PaySim fraud fields, or transaction features. No thresholds, weights, or transaction signals are used. ATO-risk indicators are research outputs, not confirmation of fraud.

## Telecom context reference baseline

| Rule | Exact condition | Prediction | Evidence and explanation |
|---|---|---|---|
| T0 — Trigger context only | `telecom_event_present = 1` and (`sim_change = 1` or `esim_change = 1`) | `NO_ATO_RISK_INDICATOR` | Include all three telecom fields. A SIM/eSIM event is contextual and does not itself indicate ATO. |
| T1 — No trigger context | T0 does not match | `NO_ATO_RISK_INDICATOR` | Include all three telecom fields. No trigger is represented, so the context-only reference raises no indicator. |

## Multi-signal baseline

Rules are evaluated in order; M1 takes precedence.

| Rule | Exact condition | Prediction | Evidence and explanation |
|---|---|---|---|
| M1 — Ambiguous device/authentication/recovery | `new_device = 1`, `authentication_anomaly = 1`, and `password_reset = 1` | `AMBIGUOUS_INSUFFICIENT_EVIDENCE` | Include `new_device`, `authentication_anomaly`, `password_reset`, `failed_login_count`, `device_change`, and `device_deviation`. The combination can occur during authorized recovery and is insufficient to resolve ATO risk. |
| M2 — Device/authentication corroboration | M1 does not match; `new_device = 1`, `authentication_anomaly = 1`, `password_reset = 0`, and (`device_change = 1` or `device_deviation = 1`) | `ATO_RISK_INDICATOR` | Include the same six multi-signal fields. Explain that the combination is an indicator, not confirmed fraud. |
| M3 — No configured combination | Neither M1 nor M2 matches | `NO_ATO_RISK_INDICATOR` | Include the same six multi-signal fields. Explain that no configured combination matched; this does not establish legitimate activity. |

The result uses three distinct outcomes so an overlapping legitimate/suspicious feature pattern is not forced into a binary class. `matched_rules` contains the single selected rule identifier.

## Evaluation on the Development Dataset

Evaluation used the current 1,000-row feature dataset. `ato_label` and `scenario_type` were joined only after the baselines produced predictions, for evaluation and scenario grouping; neither was supplied to the rules.

### Multi-signal outcome cross-tab

| Rule outcome | Legitimate (`ato_label = 0`) | Suspicious (`ato_label = 1`) | Total |
|---|---:|---:|---:|
| ATO-risk indicator | 0 | 250 | 250 |
| Ambiguous / insufficient evidence | 375 | 125 | 500 |
| No ATO-risk indicator | 250 | 0 | 250 |
| **Total** | **625** | **375** | **1,000** |

The ambiguity rate is **500 / 1,000 (50%)**. Ambiguous cases remain a separate outcome and are not assigned to either binary class.

### Conditional binary metrics

Binary metrics below apply only to the **500 determinate cases**. Ambiguous cases were excluded from the binary confusion matrix; they were not mapped to legitimate or suspicious. These figures are not overall performance metrics.

Confusion matrix rows are actual legitimate/suspicious; columns are predicted no-indicator/ATO-risk:

|  | Predicted no-indicator | Predicted ATO-risk |
|---|---:|---:|
| Actual legitimate | TN 250 | FP 0 |
| Actual suspicious | FN 0 | TP 250 |

| Metric | Result |
|---|---:|
| TP | 250 |
| FP | 0 |
| TN | 250 |
| FN | 0 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| FPR | 0.00 |
| FNR | 0.00 |

**Precision, Recall, and F1 of 1.00 apply only to the 500 determinate cases. They do not describe all 1,000 rows.** Half the dataset received an ambiguous outcome.

### Telecom context baseline

The context-only baseline returned `NO_ATO_RISK_INDICATOR` for all rows. Its confusion matrix, with the same row/column convention, is:

|  | Predicted no-indicator | Predicted ATO-risk |
|---|---:|---:|
| Actual legitimate | 625 | 0 |
| Actual suspicious | 375 | 0 |

Recall is 0.00, FNR is 1.00, FPR is 0.00, and precision is undefined because the baseline produced no positive predictions.

### Scenario types by multi-signal outcome

- **ATO-risk indicator:** `SUSPICIOUS_ESIM_CHANGE` (125) and `SIM_ESIM_AUTH_ANOMALY` (125).
- **Ambiguous / insufficient evidence:** `LEGITIMATE_SIM_DEVICE_RECOVERY` (125), `LEGITIMATE_ESIM_DEVICE_RECOVERY` (125), `LEGITIMATE_SIM_AUTH_RECOVERY` (125), and `SUSPICIOUS_SIM_REPLACEMENT` (125).
- **No ATO-risk indicator:** `LEGITIMATE_SIM_REPLACEMENT` (125) and `LEGITIMATE_ESIM_CHANGE` (125).

The telecom context baseline returned no indicator for all eight scenario types (125 rows each).

## Limitations

The synthetic scenario construction creates strong separability in some cases: the two suspicious scenarios classified as ATO-risk have device/authentication feature combinations that do not appear in the legitimate scenarios in this dataset. Conversely, suspicious SIM replacement and legitimate recovery scenarios share T+15 feature patterns, so they remain ambiguous. The conditional metrics therefore reflect this constructed dataset and the rules' determinate subset; they are not evidence of equivalent performance on independent or real-world data. No rules were tuned after this evaluation.

The complete project test suite passed after the Phase 7 evaluation: **37 passed**.
