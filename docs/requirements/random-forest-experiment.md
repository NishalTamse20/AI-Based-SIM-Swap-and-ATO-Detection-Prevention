# Phase 8 Random Forest Experiment

## Inputs and target

The model uses exactly these 21 Phase 6 features:

```text
sim_change, esim_change, telecom_event_present, new_device, device_change,
device_deviation, known_device, failed_login_count, authentication_anomaly,
password_reset, account_recovery, recovery_event_present, new_beneficiary,
account_change, baseline_event_count, baseline_event_frequency,
baseline_transaction_count, baseline_transaction_amount_mean,
baseline_transaction_amount_min, baseline_transaction_amount_max,
baseline_successful_login_count
```

The supervised target is `ato_label` (`0` legitimate scenario, `1` simulated ATO scenario), kept separate from X. Identifiers (`user_id`, `account_id`, `event_id`) are only for traceability. `scenario_type` is only joined after prediction for error analysis. PaySim fraud flags and Phase 7 predictions are excluded from X.

The post-trigger transaction features (`transaction_amount`, `transaction_count`, `transaction_type`, `transaction_deviation`) are excluded because transactions are unavailable in the current T+15 prediction window. Pre-trigger transaction summaries remain included as baseline features, per the feature dictionary. `telecom_event_present` is constant in the current dataset and contributes no current split information, but remains in the fixed schema-defined input list.

## Split and model

Use one stratified 80/20 train/test split with `random_state=42`. With the current dataset, expected counts are 800 training rows (500 legitimate, 300 suspicious) and 200 test rows (125 legitimate, 75 suspicious). Fit preprocessing and the model on training rows only. No resampling is used.

Use this fixed `RandomForestClassifier` configuration without tuning:

```text
n_estimators=100
criterion="gini"
max_depth=None
min_samples_split=2
min_samples_leaf=1
max_features="sqrt"
bootstrap=True
class_weight="balanced"
random_state=42
n_jobs=-1
```

Balanced class weights are computed from the training labels. Predictions use the estimator's default `predict` behavior; no custom probability threshold is selected.

## Evaluation

Report Precision, Recall, F1, FPR, FNR, ROC-AUC, average precision as PR-AUC, and a confusion matrix on the held-out test rows. Confusion matrix labels are `[0, 1]`, with rows actual and columns predicted. Undefined metrics are recorded as null when their denominator or required class is absent.

Calculate held-out permutation importance with average precision scoring, 10 repeats, and `random_state=42`. Separately refit the same configuration on the same training split after removing only `device_deviation`; report the same test metrics and the ablated-minus-full metric differences. These are diagnostics, not tuning.

## Phase 7 comparison

Generate both Phase 7 baselines for the exact same held-out rows. Report telecom context results over all test rows. Keep the multi-signal `AMBIGUOUS_INSUFFICIENT_EVIDENCE` outcome separate and report its rate/coverage. Calculate its binary metrics only on determinate rows. Also evaluate the Random Forest on that same determinate subset to provide a like-for-like comparison; separately report its full-test metrics.

## Scenario error analysis

Only after fitting and predicting, join `scenario_type` from the trigger-event data using identifiers. Report Random Forest TP/FP/TN/FN and error totals by scenario type, along with Phase 7 ambiguity and determinate errors. `scenario_type` is never a model input.

## Reproducibility and limitations

Outputs record source-file SHA-256 hashes, exact input columns, fixed split user IDs/counts, model parameters, random seeds, evaluation treatment, and Python/pandas/NumPy/scikit-learn versions. CSV/JSON outputs are written to `data/processed/random_forest/`.

The random row split places repeated synthetic scenario templates in both train and test. Results therefore measure performance on held-out accounts drawn from the existing scenario construction, not generalization to unseen scenario designs or real users. Permutation importance and feature ablation are used to identify reliance on scenario-specific signals such as `device_deviation`; they do not remove this limitation.

## Development Run Results

The recorded run used the current 1,000-row development feature set. Stratification produced 800 training rows (500 legitimate, 300 suspicious) and 200 held-out test rows (125 legitimate, 75 suspicious).

### Random Forest held-out metrics

| Metric | Result |
|---|---:|
| Precision | 0.846154 |
| Recall | 0.880000 |
| F1 | 0.862745 |
| FPR | 0.096000 |
| FNR | 0.120000 |
| ROC-AUC | 0.974347 |
| PR-AUC (average precision) | 0.960169 |

Confusion matrix uses rows actual `[0, 1]` and columns predicted `[0, 1]`:

```text
[[113, 12],
 [  9, 66]]
```

### Held-out permutation importance

The five largest mean average-precision decreases under permutation were:

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| `device_deviation` | 0.170720 | 0.026911 |
| `failed_login_count` | 0.043667 | 0.008635 |
| `password_reset` | 0.022139 | 0.004105 |
| `recovery_event_present` | 0.021158 | 0.003926 |
| `esim_change` | 0.013143 | 0.002763 |

The full importance output for all 21 inputs is `data/processed/random_forest/feature_importance.csv`.

### `device_deviation` ablation

Removing only `device_deviation`, while preserving the split and model configuration, produced Precision 0.833333, Recall 0.866667, F1 0.849673, FPR 0.104000, FNR 0.133333, ROC-AUC 0.974400, and PR-AUC 0.960247. Its confusion matrix was `[[112, 13], [10, 65]]`. Relative to the full model, F1 decreased by 0.013072; ROC-AUC and PR-AUC increased slightly by 0.000053 and 0.000079, respectively. The ablation indicates some dependence on `device_deviation` for class predictions, while probability ranking changed very little in these metrics.

### Same-test-row Phase 7 comparison

The telecom context baseline predicted no indicator for all 200 test rows: confusion matrix `[[125, 0], [75, 0]]`, Recall 0.00, FNR 1.00, FPR 0.00, ROC-AUC 0.50, and PR-AUC 0.375. These two ranking metrics use the baseline's binary output as its score because it provides no probability score. Precision is undefined because there were no positive predictions.

The Phase 7 multi-signal baseline returned 54 ATO-risk indicators, 94 ambiguous outcomes, and 52 no-indicator outcomes. Ambiguity was 47% of this test set and determinate coverage was 53%. On the 106 determinate rows, its confusion matrix was `[[52, 0], [0, 54]]`. The Random Forest evaluated on those same 106 rows had the same confusion matrix. Ambiguous rows were excluded from both binary comparisons, not assigned to either class. Full-test Random Forest metrics above include all 200 rows.

### Scenario-level errors

| Scenario type (evaluation only) | Test rows | Random Forest errors |
|---|---:|---:|
| `LEGITIMATE_SIM_DEVICE_RECOVERY` | 22 | 12 FP |
| `SUSPICIOUS_SIM_REPLACEMENT` | 21 | 9 FN |
| Each of the other six scenario types | 22–29 | 0 |

The complete scenario-by-scenario error counts and Phase 7 outcomes are in `data/processed/random_forest/scenario_error_summary.csv`; row-level held-out predictions are in `predictions.csv`.

### Interpretation limits

These are results on a single seeded split of a synthetic development dataset, not research performance claims. The split contains repeated scenario templates in both partitions, and `device_deviation` is strongly associated with some generated suspicious sequences. The two main error-bearing scenario types also have overlapping T+15 patterns: legitimate device recovery produced false positives, while suspicious SIM replacement produced false negatives. The Phase 7 comparison is conditional on its 53% determinate coverage for the multi-signal baseline. These results do not establish generalization to unseen scenario designs or real users.

The saved artifacts are reproducible with the source hashes and environment versions in `data/processed/random_forest/run_metadata.json`. Rerunning the experiment with the same inputs and environment reproduced identical output files.
