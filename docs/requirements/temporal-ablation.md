# Phase 9B Temporal Feature Ablation

## Objective

Estimate which existing Phase 9 temporal feature groups account for the observed difference between the Phase 8 baseline and Phase 9 model under the established unseen-scenario evaluation. This is a synthetic research validation experiment. It does not change the temporal feature definitions, prediction boundary, model configuration, or evaluation methodology.

## Five fixed experiments

All experiments use the same Phase 8 feature set as their base and the same leave-one-scenario-type-out folds:

| Experiment | Inputs beyond the Phase 8 base |
|---|---|
| BASELINE | None |
| TIMING | Group 1 |
| COUNTS | Group 2 |
| SEQUENCES | Group 3 |
| ALL_TEMPORAL | Groups 1, 2, and 3 |

### Group 1 — Timing

- time_to_new_device_seconds
- time_to_failed_login_seconds
- time_to_password_reset_seconds
- time_to_recovery_seconds
- time_to_auth_anomaly_seconds

### Group 2 — Counts

- post_trigger_failed_login_count
- post_trigger_device_event_count
- post_trigger_recovery_event_count

### Group 3 — Sequences

- telecom_to_device_sequence
- telecom_to_auth_sequence
- telecom_to_recovery_sequence
- device_before_auth_sequence
- auth_before_recovery_sequence

temporal_sequence_score remains excluded. No feature is added, removed, redefined, or weighted for this experiment.

## Fixed evaluation methodology

Use leave-one-scenario-type-out evaluation over all eight existing scenario types. For each fold, every row of the held-out type is excluded from training and receives one out-of-fold prediction. Repeat for each type. Each of the five experiments uses those exact same train/test indices. Aggregate all out-of-fold predictions and report Precision, Recall, F1, FPR, FNR, ROC-AUC, average precision (PR-AUC), and a confusion matrix in [0, 1] class order. Also report applicable metrics and FP/FN/correct counts by held-out type.

Every estimator uses the unchanged Phase 8 RandomForestClassifier configuration: n_estimators=100, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=balanced, random_state=42, and n_jobs=-1. There is no hyperparameter or threshold tuning. Experiments with temporal inputs fit the existing median imputer and missingness indicators on each training fold only. BASELINE uses the Phase 8 classifier directly.

## Leakage controls

- scenario_type is used only to define folds and label post-prediction evaluation records. It is not an ML input.
- ato_label is the target only. It is not included in any experiment's X.
- user_id, account_id, and event_id are traceability fields only.
- PaySim fraud fields and Phase 7 predictions are excluded.
- Each experiment's input columns are the fixed Phase 8 feature list plus only the specified temporal group(s).
- Temporal inputs come from the existing T-through-T+15 inclusive feature artifact. The dataset and Phase 9 feature artifact are read-only; no raw future events are used to construct inputs.
- Test labels are used only after prediction for metric calculation.
- The output path is guarded against overlap with Phase 8 and Phase 9 output locations.

## Interpretation limitations

The current eight synthetic scenario types each contain one ato_label class. Consequently, each held-out fold is one-class: ROC-AUC is unavailable for every individual fold, and average precision is unavailable for legitimate-only folds. Undefined metrics are null, not estimated or fabricated. Pooled out-of-fold metrics include both classes.

Temporal features can encode the synthetic scenario templates directly. Repeated values or sequences may explain a model's scores without representing broadly generalizable behavior. Group ablation is descriptive for this fixed dataset and split; it does not establish causal effects, feature importance outside these templates, or real-world detection performance. Pooled AUC metrics also combine probabilities from separately fitted fold models, whose score scales are not necessarily calibrated across folds.

## Reproducible outputs

Run python -m ml.models.temporal_ablation_experiment. The script reuses the existing feature artifacts and leave-one-scenario-type-out fold builder, then writes only data/processed/temporal_ablation/metrics.json, predictions.csv, scenario_error_summary.csv, and run_metadata.json. Metadata records the feature groups, fold definitions, model parameters, input hashes, and software versions.

## Development run results

The existing 1,000-row feature artifacts and the eight shared 125-row folds were used. Each scenario type had only one target class, so ROC-AUC is unavailable for every individual fold; average precision is unavailable for each legitimate-only fold. Per-fold metrics are recorded as null where undefined in metrics.json.

| Experiment | Precision | Recall | F1 | FPR | FNR | ROC-AUC | Average precision | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BASELINE | 0.342920 | 0.413333 | 0.374849 | 0.475200 | 0.586667 | 0.363213 | 0.332043 | [[328, 297], [220, 155]] |
| TIMING | 1.000000 | 0.666667 | 0.800000 | 0.000000 | 0.333333 | 1.000000 | 1.000000 | [[625, 0], [125, 250]] |
| COUNTS | 0.299043 | 0.333333 | 0.315259 | 0.468800 | 0.666667 | 0.363693 | 0.353869 | [[332, 293], [250, 125]] |
| SEQUENCES | 0.466418 | 0.666667 | 0.548847 | 0.457600 | 0.333333 | 0.435324 | 0.379315 | [[339, 286], [125, 250]] |
| ALL_TEMPORAL | 1.000000 | 0.666667 | 0.800000 | 0.000000 | 0.333333 | 1.000000 | 1.000000 | [[625, 0], [125, 250]] |

The TIMING and ALL_TEMPORAL results are identical on all reported pooled metrics and confusion counts. TIMING alone therefore reproduces the apparent Phase 9 improvement on this dataset and split. COUNTS alone is below the Phase 8 F1; SEQUENCES alone is above its F1 but below TIMING/ALL_TEMPORAL.

Scenario-level FP/FN/correct counts:

| Held-out scenario type | Rows (legitimate/suspicious) | BASELINE | TIMING | COUNTS | SEQUENCES | ALL_TEMPORAL |
|---|---:|---:|---:|---:|---:|---:|
| LEGITIMATE_ESIM_CHANGE | 125 (125/0) | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 |
| LEGITIMATE_ESIM_DEVICE_RECOVERY | 125 (125/0) | 84/0/41 | 0/0/125 | 83/0/42 | 81/0/44 | 0/0/125 |
| LEGITIMATE_SIM_AUTH_RECOVERY | 125 (125/0) | 88/0/37 | 0/0/125 | 85/0/40 | 80/0/45 | 0/0/125 |
| LEGITIMATE_SIM_DEVICE_RECOVERY | 125 (125/0) | 125/0/0 | 0/0/125 | 125/0/0 | 125/0/0 | 0/0/125 |
| LEGITIMATE_SIM_REPLACEMENT | 125 (125/0) | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 |
| SIM_ESIM_AUTH_ANOMALY | 125 (0/125) | 0/95/30 | 0/0/125 | 0/125/0 | 0/0/125 | 0/0/125 |
| SUSPICIOUS_ESIM_CHANGE | 125 (0/125) | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 | 0/0/125 |
| SUSPICIOUS_SIM_REPLACEMENT | 125 (0/125) | 0/125/0 | 0/125/0 | 0/125/0 | 0/125/0 | 0/125/0 |

Cells show FP/FN/correct. The SUSPICIOUS_SIM_REPLACEMENT failure remains in every experiment, including TIMING and ALL_TEMPORAL.

## Interpretation of the ablation

Timing appears responsible for most of the pooled improvement in this fixed experiment: Timing alone matches All Temporal, while Counts alone does not improve F1 over the Phase 8 baseline and Sequences alone produces a smaller F1 increase. This is an observed ablation pattern, not a causal attribution or a feature-selection decision.

The result remains strongly dependent on synthetic template construction. The same timing group yields perfect pooled ranking metrics but still misses all rows from one suspicious scenario type; every fold is one-class, and pooled probabilities come from separately fitted models. These findings do not demonstrate robust temporal generalization or real-world performance.
