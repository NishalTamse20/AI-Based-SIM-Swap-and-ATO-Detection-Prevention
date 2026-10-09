# Unseen-Scenario Generalization Evaluation

## Purpose

The Phase 9 random row split can place accounts generated from the same synthetic scenario templates in both training and test data. The resulting score may reflect recognition of repeated templates rather than generalization to a scenario type absent from training. This evaluation holds out one complete scenario_type at a time.

This is an evaluation of the current synthetic scenario set. It is not evidence of real-user or production performance.

## Leave-one-scenario-type-out method

For each unique scenario_type, all rows of that type are excluded from training and form that fold's test set. The fixed Phase 8 model is fit on every remaining scenario type. The process repeats once for each type; each row therefore receives exactly one out-of-fold prediction.

scenario_type comes only from the TELECOM trigger row and is used as a grouping key and after-prediction evaluation metadata. It is never part of Model A or Model B input matrices. The split is not stratified: holding out scenario types is the evaluation design. Model A and Model B use identical train/test indices for every fold.

### Models

- **Model A:** the exact 21 Phase 8 feature columns.
- **Model B:** those same 21 columns plus the 13 Phase 9 temporal feature columns.

Both use the Phase 8 RandomForestClassifier configuration unchanged: 100 trees, Gini criterion, unrestricted depth, minimum split 2, minimum leaf 1, square-root feature sampling, bootstrap enabled, class_weight="balanced", random_state=42, and n_jobs=-1. There is no hyperparameter or decision-threshold tuning. Model B's median imputer and missingness indicators are fit separately using each training fold only; the temporal feature CSV preserves its original missing values.

## Leakage and data controls

- ato_label is used only as the supervised target and for metrics after the fold model has generated predictions.
- scenario_type is only the grouping variable and evaluation metadata; it is excluded from X.
- user_id, account_id, and event_id provide traceability and are excluded from X.
- PaySim isFraud/isFlaggedFraud fields and Phase 7 predictions are excluded.
- Model A uses only the Phase 8 feature list; Model B uses that list and the existing Phase 9 temporal feature list. No raw event activity is loaded as model input.
- The only event file data read by this experiment is scenario_type from each user's TELECOM trigger row for fold assignment and grouping.
- Phase 9 inputs come from the already-generated T through T+15 inclusive temporal snapshot. The experiment does not regenerate or extend that snapshot, and no event after T+15 is introduced.
- Test labels are not used in fitting, preprocessing, or model selection.

## Metrics and error analysis

Aggregated metrics are calculated from the complete set of out-of-fold predictions: Precision, Recall, F1, FPR, FNR, ROC-AUC, average precision (PR-AUC), and a confusion matrix with class order [0, 1]. Results are also reported for each held-out scenario type. Fold-level metrics with zero denominators are null. ROC-AUC is unavailable when a fold contains only one class; average precision is unavailable when a fold has no positive examples. Confusion matrices are still reported with both class labels.

The scenario error table reports test row count, legitimate and suspicious counts, false positives, false negatives, correct predictions, and total errors for both models. The temporal feature audit gives missingness, occurrence counts, and numeric summaries by scenario type. It is descriptive and does not use labels to fit models.

## Current data limitation

The current data contains eight scenario types with 125 rows each. Each scenario type contains a single ato_label class: five legitimate types and three suspicious types. Consequently, individual held-out folds are one-class tests. ROC-AUC is unavailable in every individual fold; PR-AUC is unavailable for legitimate-only folds. Fold-level error metrics remain interpretable only where their denominators exist. Aggregated out-of-fold metrics combine the predictions across all held-out types and contain both classes.

The perfect label-to-template association is a design limitation. Holding out a type tests transfer across the existing templates, but does not separate class generalization from the scenario construction. Temporal feature differences by type are associations with those templates, not causal evidence. No statistical-significance procedure is defined or applied here.

## Interpretation rules

Compare Model A and Model B only on the same aggregated out-of-fold rows and identical scenario folds. State the exact metric differences. Do not interpret a higher score as proof that temporal correlation improves real-world detection. Results apply only to these synthetic templates and this fixed experiment.

## Reproducibility

Run python -m ml.models.unseen_scenario_experiment. It reads the existing data/processed/features.csv and data/processed/temporal_features.csv, uses Phase 8's feature list, model factory, and metric implementation, then writes only these Phase 9 validation artifacts under data/processed/unseen_scenario/: metrics.json, predictions.csv, scenario_error_summary.csv, feature_audit.csv, and run_metadata.json. Input file hashes, feature lists, fold membership, model parameters, and runtime package versions are recorded in metadata.

## Development run findings

The current artifacts contain 1,000 rows across eight types (125 rows each): five legitimate-only types and three suspicious-only types. The complete out-of-fold metrics were:

| Metric | Model A: Phase 8 | Model B: Phase 8 + Phase 9 | Model B minus A |
|---|---:|---:|---:|
| Precision | 0.342920 | 1.000000 | +0.657080 |
| Recall | 0.413333 | 0.666667 | +0.253333 |
| F1 | 0.374849 | 0.800000 | +0.425151 |
| FPR | 0.475200 | 0.000000 | -0.475200 |
| FNR | 0.586667 | 0.333333 | -0.253333 |
| ROC-AUC | 0.363213 | 1.000000 | +0.636787 |
| Average precision | 0.332043 | 1.000000 | +0.667957 |

Confusion matrices use rows actual [0, 1] and columns predicted [0, 1]:

- Model A: [[328, 297], [220, 155]]
- Model B: [[625, 0], [125, 250]]

The random-split Phase 9 F1 of 1.000 did not persist: the unseen-type aggregate F1 is 0.800. On this exact pooled out-of-fold comparison, Model B has higher F1 than Model A by 0.425151. However, Model B misses all 125 rows of SUSPICIOUS_SIM_REPLACEMENT. Its aggregate ROC-AUC and average precision are 1.000 even though its default class predictions miss that entire type; probabilities rank the pooled classes perfectly while remaining below the default prediction boundary for those rows. The AUC values pool scores from separately trained fold models and should be interpreted cautiously because score scales need not be calibrated across folds.

| Held-out scenario type | Rows (legitimate/suspicious) | Model A FP/FN/correct | Model B FP/FN/correct |
|---|---:|---:|---:|
| LEGITIMATE_ESIM_CHANGE | 125 (125/0) | 0 / 0 / 125 | 0 / 0 / 125 |
| LEGITIMATE_ESIM_DEVICE_RECOVERY | 125 (125/0) | 84 / 0 / 41 | 0 / 0 / 125 |
| LEGITIMATE_SIM_AUTH_RECOVERY | 125 (125/0) | 88 / 0 / 37 | 0 / 0 / 125 |
| LEGITIMATE_SIM_DEVICE_RECOVERY | 125 (125/0) | 125 / 0 / 0 | 0 / 0 / 125 |
| LEGITIMATE_SIM_REPLACEMENT | 125 (125/0) | 0 / 0 / 125 | 0 / 0 / 125 |
| SIM_ESIM_AUTH_ANOMALY | 125 (0/125) | 0 / 95 / 30 | 0 / 0 / 125 |
| SUSPICIOUS_ESIM_CHANGE | 125 (0/125) | 0 / 0 / 125 | 0 / 0 / 125 |
| SUSPICIOUS_SIM_REPLACEMENT | 125 (0/125) | 0 / 125 / 0 | 0 / 125 / 0 |

Per-fold ROC-AUC is unavailable for all eight folds because each held-out type has one class. Per-fold average precision is unavailable on the five legitimate-only folds because they contain no positive examples. Other denominator-dependent values are recorded as null in metrics.json. In the suspicious-only folds, average precision is defined and is reported, but ROC-AUC remains undefined.

The temporal audit shows strong template association: within each scenario type, the temporal profiles are largely constant. For example, both simple legitimate telecom-only types have all five time-to-event values missing and zero post-trigger counts/sequences. Device/recovery templates have repeated 300-second first-device, 480-second first-failed-login, and 600- or 720-second recovery timings; suspicious SIM replacement uses 300, 600, and 900 seconds for first device, failed login, and recovery. These are observed generated values, not thresholds or causal evidence.

Conclusion: Model B performs better on pooled metrics in this leave-one-type-out run, but the result is not sufficient to claim that temporal correlation improves generalization beyond the existing synthetic templates. The label-pure templates, eight-type scope, constant within-type patterns, one-class folds, pooled cross-fold probability scores, and complete failure on one suspicious type materially limit that interpretation.
