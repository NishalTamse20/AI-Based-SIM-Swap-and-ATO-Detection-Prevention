# Phase 9 Temporal Correlation

## Research objective

Evaluate whether the timing and ordering of device, authentication, and recovery events after a SIM/eSIM trigger provide additional detection value beyond the Phase 8 Random Forest inputs. The output is an experimental comparison on the current synthetic dataset, not a claim about production fraud detection.

## Prediction boundary

Let `T` be the timestamp of the user's sole telecom trigger (`SIM_SWAP`, `SIM_REPLACEMENT`, or `ESIM_CHANGE`). The primary observation boundary is the inclusive interval `T <= timestamp <= T+15 minutes`. An event exactly at T+15 minutes is included; any later event is excluded. This is the project's frozen synthetic research-design choice, not an industry-standard timing threshold.

Time-to-event and count features can include a matching event at T (giving a time difference of zero). Sequence features require strictly increasing timestamps after T; equal timestamps do not establish which event came first. No universal timing threshold is applied.

## Temporal feature definitions

All features below are calculated independently per synthetic user from event rows in the observation interval. `event_category` is the broad category and `event_type` is the specific event.

| Feature | Definition | Missing/value semantics |
|---|---|---|
| `time_to_new_device_seconds` | Seconds from T to the earliest `DEVICE/NEW_DEVICE` | Empty when absent |
| `time_to_failed_login_seconds` | Seconds from T to the earliest `AUTHENTICATION/FAILED_LOGIN` | Empty when absent |
| `time_to_password_reset_seconds` | Seconds from T to the earliest `RECOVERY/PASSWORD_RESET` | Empty when absent |
| `time_to_recovery_seconds` | Seconds from T to the earliest event with category `RECOVERY` | Empty when absent |
| `time_to_auth_anomaly_seconds` | Seconds from T to the earliest `AUTHENTICATION/FAILED_LOGIN` or `AUTHENTICATION/MFA_FAILURE`, matching Phase 6 anomaly semantics | Empty when absent |
| `post_trigger_failed_login_count` | Count of `AUTHENTICATION/FAILED_LOGIN` | Zero when no such row is observed |
| `post_trigger_device_event_count` | Count of all `DEVICE` rows plus `BEHAVIOUR/DEVICE_DEVIATION`, the device-related semantics already used by Phase 6 | Zero when no such row is observed |
| `post_trigger_recovery_event_count` | Count of all events with category `RECOVERY` | Zero when no such row is observed |
| `telecom_to_device_sequence` | 1 when a `DEVICE/NEW_DEVICE` has timestamp strictly after T in-window | Otherwise 0 |
| `telecom_to_auth_sequence` | 1 when an authentication anomaly (`FAILED_LOGIN` or `MFA_FAILURE`) has timestamp strictly after T in-window | Otherwise 0 |
| `telecom_to_recovery_sequence` | 1 when a `RECOVERY` event has timestamp strictly after T in-window | Otherwise 0 |
| `device_before_auth_sequence` | 1 when the earliest in-window `DEVICE/NEW_DEVICE` is strictly earlier than the earliest authentication anomaly | Otherwise 0 |
| `auth_before_recovery_sequence` | 1 when the earliest in-window authentication anomaly is strictly earlier than the earliest recovery event | Otherwise 0 |

The proposed `temporal_sequence_score` is omitted. No project methodology defines weights for such a score.

## Leakage controls

- The generator reads only `user_id`, `account_id`, `event_id`, `event_category`, `event_type`, and `timestamp`.
- `ato_label`, `scenario_type`, `paysim_isFraud`, `paysim_isFlaggedFraud`, and Phase 7 predictions are not read to create temporal features.
- Only the existing event categories/types are used; no new event type is introduced.
- The temporal feature artifact contains identifiers and temporal inputs only, not labels or scenario metadata.
- The model comparison joins `ato_label` only as the target. Phase 8 identifiers remain traceability fields, not model inputs. Scenario metadata is not used by the comparison.
- Each model uses the same Phase 8 stratified 80/20 row split, seed 42, and fixed Random Forest configuration. No hyperparameter search or custom probability threshold is used.
- Temporal missing times stay empty in the CSV. Within the Phase 9 Random Forest pipeline only, a median imputer is fit on training rows and adds missingness indicators. Test rows are transformed using that fitted preprocessor; this does not alter the temporal CSV or Phase 8 pipeline/results.
- The experiment is a separate, read-only comparison entry point and does not write Phase 8 outputs.

## Limitations

These are temporal features derived from synthetic event sequences. They describe observed event timing/order and do not establish causality or confirmed account takeover. Equal timestamps cannot establish event order. The 15-minute boundary is a frozen project design choice. Results remain limited by the synthetic scenario templates and the current dataset; no real-user or production generalization is implied.

## Reproducible comparison

Run `python -m ml.features.temporal_features` to generate `data/processed/temporal_features.csv`, then `python -m ml.models.temporal_random_forest_experiment` to print Phase 8 versus Phase 9 metrics. The experiment reuses Phase 8's stratified split helper and exact classifier parameters. The comparison reports Precision, Recall, F1, FPR, FNR, ROC-AUC, average precision (PR-AUC), and the confusion matrix for each model.

### Development comparison result

On the current 1,000-row synthetic development set, the shared split contained 800 training rows (500 label 0, 300 label 1) and 200 test rows (125 label 0, 75 label 1). The Phase 8 result reproduced its recorded held-out metrics. The Phase 9 augmented model returned Precision 1.000000, Recall 1.000000, F1 1.000000, FPR 0.000000, FNR 0.000000, ROC-AUC 1.000000, and average precision 1.000000, with confusion matrix `[[125, 0], [0, 75]]`. Phase 8 on those rows returned Precision 0.846154, Recall 0.880000, F1 0.862745, FPR 0.096000, FNR 0.120000, ROC-AUC 0.974347, and average precision 0.960169, with confusion matrix `[[113, 12], [9, 66]]`.

These are single-split results on the synthetic development scenarios, not research-performance claims. The perfect Phase 9 result warrants particular caution: temporal patterns may strongly encode how the synthetic scenarios were constructed. It does not establish generalization to unseen scenarios or real users. No Phase 8 result files were rewritten by the comparison.
