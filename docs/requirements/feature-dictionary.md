# Feature Dictionary

## Row and Prediction Scope

The Phase 6 output contains one row per synthetic user/account. The primary prediction snapshot is fixed at 15 minutes after the SIM/eSIM trigger. Let `T` be the trigger timestamp: primary features may use trigger and post-trigger events with timestamps `T <= timestamp <= T + 15 minutes`; events after `T + 15 minutes` are excluded. Baseline features come from the Phase 5 profile built strictly before `T`.

This 15-minute window is a synthetic research-design choice for this project, not an industry-standard threshold. Phase 15 may evaluate 5-, 10-, 15-, and 30-minute windows as a sensitivity experiment; those windows do not replace the frozen primary window.

`user_id`, `account_id`, and `event_id` are identifiers. In the output, `event_id` identifies the SIM/eSIM trigger. `ato_label` is retained separately as the supervised target and is not an input feature. `scenario_type` and PaySim fraud flags are excluded from the feature output.

## Input Features

| Feature | Source field/event | Meaning | Data type | Leakage considerations |
|---|---|---|---|---|
| `sim_change` | `event_category=TELECOM`; `event_type=SIM_SWAP` or `SIM_REPLACEMENT` | Whether a SIM swap or replacement is observed by the prediction snapshot | Integer, 0/1 | Derived from observed telecom events; not a label |
| `esim_change` | `event_category=TELECOM`; `event_type=ESIM_CHANGE` | Whether an eSIM change is observed by the snapshot | Integer, 0/1 | Derived from observed telecom events; not a label |
| `telecom_event_present` | Telecom event rows | Whether a SIM/eSIM trigger is observed by the snapshot | Integer, 0/1 | Trigger presence alone does not represent ATO |
| `new_device` | `DEVICE/NEW_DEVICE` | Whether a new device event is observed by the snapshot | Integer, 0/1 | Uses observed events only |
| `device_change` | `DEVICE/DEVICE_CHANGE` | Whether a device change event is observed by the snapshot | Integer, 0/1 | Uses observed events only |
| `device_deviation` | `BEHAVIOUR/DEVICE_DEVIATION` or `DEVICE` event with `status=DEVIATION` | Whether a device deviation signal is present | Integer, 0/1 | Uses event/status values; no behavioural threshold is introduced |
| `known_device` | Observed non-telecom `device_id` values and Phase 5 `known_device_ids` | 1 when at least one post-trigger device is observed and all such IDs occur in the pre-trigger known-device set | Integer, 0/1 | Known-device set is pre-trigger only; trigger device itself is excluded from this comparison |
| `failed_login_count` | `AUTHENTICATION/FAILED_LOGIN` | Number of failed login events observed by the snapshot | Integer, count | Uses only observed events up to the snapshot |
| `authentication_anomaly` | `AUTHENTICATION/FAILED_LOGIN`, `AUTHENTICATION/MFA_FAILURE` | Whether either documented failure event is observed | Integer, 0/1 | No additional anomaly rule or threshold is applied |
| `password_reset` | `RECOVERY/PASSWORD_RESET` | Whether a password reset event is observed | Integer, 0/1 | Uses observed recovery events only |
| `account_recovery` | `RECOVERY/ACCOUNT_RECOVERY` | Whether an account recovery event is observed | Integer, 0/1 | Uses observed recovery events only |
| `recovery_event_present` | Any `RECOVERY` event | Whether a recovery event is observed | Integer, 0/1 | Uses observed recovery events only |
| `new_beneficiary` | `ACCOUNT/NEW_BENEFICIARY` | Whether a new beneficiary event is observed | Integer, 0/1 | Uses observed account events only |
| `account_change` | Any `ACCOUNT` event | Whether an account event is observed | Integer, 0/1 | Uses observed account events only |
| `transaction_amount` | `TRANSACTION/amount` | Sum of observed transaction amounts through the snapshot | Decimal amount | PaySim fraud flags are not read |
| `transaction_count` | `TRANSACTION` event rows | Number of transactions observed through the snapshot; zero means no transaction row was observed in the 15-minute window | Integer, count | Uses observed transaction events only; zero is an observed count, not an imputed amount |
| `transaction_type` | `TRANSACTION/transaction_type` | Sorted, distinct observed PaySim transaction types joined with `\|` | String, categorical | Source fraud flags are excluded |
| `transaction_deviation` | Observed transaction `amount`; Phase 5 `transaction_amount_mean` | Mean observed post-trigger transaction amount minus pre-trigger baseline mean; empty when either side is unavailable | Decimal amount difference | No threshold; only events through the prediction snapshot and pre-trigger profile values are used |
| `baseline_event_count` | Phase 5 `baseline_event_count` | Number of events in the pre-trigger baseline | Integer, count | Pre-trigger profile only |
| `baseline_event_frequency` | Phase 5 `event_frequency_per_day` | Pre-trigger events per observed baseline day | Decimal rate | Pre-trigger profile only |
| `baseline_transaction_count` | Phase 5 `transaction_count` | Number of pre-trigger transactions | Integer, count | Pre-trigger profile only |
| `baseline_transaction_amount_mean` | Phase 5 `transaction_amount_mean` | Mean pre-trigger transaction amount | Decimal amount | Pre-trigger profile only |
| `baseline_transaction_amount_min` | Phase 5 `transaction_amount_min` | Minimum pre-trigger transaction amount | Decimal amount | Pre-trigger profile only |
| `baseline_transaction_amount_max` | Phase 5 `transaction_amount_max` | Maximum pre-trigger transaction amount | Decimal amount | Pre-trigger profile only |
| `baseline_successful_login_count` | Phase 5 `successful_login_count` | Count of successful pre-trigger LOGIN events | Integer, count | Pre-trigger profile only |

## Target and Excluded Fields

`ato_label` is output as the target column only; it is not part of `FEATURE_COLUMNS`. The generator does not read `scenario_type`, `paysim_isFraud`, or `paysim_isFlaggedFraud` when calculating inputs. Identifiers are output for traceability and are not model inputs. No temporal correlation or sequence features are included.

When no transaction occurs by the 15-minute cutoff, `transaction_amount`, `transaction_type`, and `transaction_deviation` remain empty because those values are unavailable. `transaction_count` remains zero because it is the observed number of transaction rows in the window.
