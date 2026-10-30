# Feature Attribution Audit (v3 Leave-One-Family-Out Models)

## Purpose

Measure which inputs the v3 leave-one-family-out (LOFO) Random Forest models depend on, test whether the designed device-novelty difference dominates, and test whether the post-trigger transaction columns add information. Results describe these fitted models on synthetic controlled cohorts (v3, seed 42); they are not real-world findings and do not show causation.

## Method

- **Same models.** The 40 fold models (Model A and Model B × 20 folds) were not persisted, so each was refit with the unchanged Phase 8 configuration on the identical LOFO folds. Before any attribution, refit predictions and `predict_proba` values were compared with `data/processed/family_holdout_v3/*/predictions.csv`: **0 prediction mismatches and 0 probability mismatches out of 400 per model per cohort**. The run stops on any mismatch.
- **Held-out permutation importance** (Phase 8 precedent: average precision, 10 repeats, `random_state=42`) plus accuracy. Within each held-out family's 20 cases, one unit's columns are shuffled; the fold model re-predicts; the drop versus unshuffled is recorded. Pooled drops use all 400 out-of-fold predictions; per-family drops are averaged over repeats.
- **Grouped units** are shuffled jointly because correlated columns split credit when shuffled singly:
  - `device_group` = `new_device`, `device_change`, `known_device`
  - `transaction_group` (Model B) = `transaction_count`, `transaction_amount`, `transaction_deviation`, three `transaction_type_*` one-hots
- **Why this method:** it works with the fitted Random Forest unchanged, is evaluated on held-out families (matching the evaluation), and follows the project precedent. Impurity (MDI) importance was not used: it is computed on training data and favours continuous, high-cardinality columns such as `transaction_amount`.
- **Device-group ablation (measured diagnostic, not tuning):** both models refit on the same folds and settings with only the device group removed.
- **Correlation diagnostics** (descriptive only): agreement between predictions and the `new_device` column, and accuracy within `new_device` = 0/1.
- Code: `ml/models/feature_attribution_experiment.py`. Outputs: `data/processed/feature_attribution_v3/{predictive,negative_control}/` — `feature_attribution.csv` (51 units), `feature_attribution_per_family.csv` (1,020 rows), `metrics.json`, `run_metadata.json`.

## Important clarification about "new_device"

The design-audit baseline previously called `new_device` (lookup accuracy 0.677) was computed on the generator's **device-slot version** (new vs known device), not on the `new_device` feature column. In families whose device event is `DEVICE_CHANGE`, a new device appears as `device_change = 1` and `new_device = 0`. In every family, the version is reflected directly by `known_device`. The designed device-novelty signal is therefore spread across the three device columns, which is why `known_device` and `device_group` rank highest below while the `new_device` column alone ranks low.

## Measured results — predictive cohort

Pooled drop = unshuffled minus shuffled (mean ± SD over 10 repeats). Per-family = median [min, max] across 20 held-out families.

| Model | Rank | Unit | Pooled AP drop | Pooled accuracy drop | Per-family accuracy drop |
|---|---|---|---|---|---|
| A | 1 | device_group | 0.136 ± 0.022 | 0.122 ± 0.024 | 0.125 [0.020, 0.250] |
| A | 2 | known_device | 0.062 ± 0.012 | 0.076 ± 0.014 | 0.085 [−0.015, 0.165] |
| A | 3 | authentication_anomaly | 0.049 ± 0.017 | 0.010 ± 0.013 | 0.013 [−0.055, 0.080] |
| A | 4 | new_beneficiary | 0.034 ± 0.006 | 0.019 ± 0.006 | 0.020 [−0.085, 0.065] |
| A | 5 | recovery_event_present | 0.028 ± 0.011 | 0.027 ± 0.011 | 0.015 [−0.040, 0.180] |
| A | 7 | new_device (column) | 0.008 ± 0.004 | 0.000 ± 0.005 | 0.000 [−0.055, 0.060] |
| B | 1 | device_group | 0.127 ± 0.015 | 0.126 ± 0.013 | 0.133 [−0.030, 0.245] |
| B | 2 | known_device | 0.068 ± 0.016 | 0.101 ± 0.009 | 0.113 [−0.005, 0.190] |
| B | 3 | authentication_anomaly | 0.038 ± 0.018 | 0.006 ± 0.013 | 0.005 [−0.075, 0.100] |
| B | 10 | transaction_group | 0.006 | 0.010 | see per-family CSV |

Model B transaction columns individually: `transaction_type_PAYMENT` 0.008 AP / 0.017 accuracy, `_TRANSFER` 0.008 / 0.005, `_CASH_OUT` 0.007 / 0.004; `transaction_amount` −0.001 / 0.012, `transaction_deviation` −0.009 / 0.011, `transaction_count` −0.007 / 0.001.

**Device-group ablation (measured):**

| Model | Accuracy full → ablated (95% CI of ablated) | ROC-AUC change | Recall change | FPR change | Per-family accuracy change, median [min, max] |
|---|---|---|---|---|---|
| A | 0.645 → 0.545 (0.496–0.593) | −0.143 | −0.140 | +0.060 | −0.10 [−0.30, +0.15] |
| B | 0.660 → 0.573 (0.524–0.620) | −0.110 | −0.130 | +0.045 | −0.10 [−0.25, +0.15] |

## Measured results — negative control

- Every unit's pooled drop is small: the largest are `device_group` 0.018 AP / 0.020 accuracy (A) and `account_change` 0.019 AP (B), with `transaction_group` 0.012 AP (B). Per-family drops straddle zero.
- `baseline_transaction_amount_min` shows a pooled accuracy drop of 0.034 (A) / 0.023 (B) with a very wide per-family range (up to 0.335). The negative-control labels are independent of all features by construction, so this is consistent with noise around a chance-level model.
- Device-group ablation: A 0.520 → 0.492 (0.444–0.541), B 0.525 → 0.512 (0.464–0.561); both intervals include 0.5.

## Correlation diagnostics (descriptive, not attribution)

| Cohort | Model | Prediction = `new_device` column | Accuracy when `new_device` = 0 / 1 |
|---|---|---|---|
| Predictive | A | 0.635 | 0.654 (n=240) / 0.631 (n=160) |
| Predictive | B | 0.600 | 0.667 / 0.650 |
| Negative control | A | 0.435 | 0.521 (n=242) / 0.519 (n=158) |
| Negative control | B | 0.415 | 0.579 / 0.443 |

## Findings, labelled

1. **Measured:** in the predictive cohort the device group is the dominant unit for both models (pooled accuracy drop ≈ 0.12–0.13, AP drop ≈ 0.13–0.14, about 2–3× the next unit), and removing it lowers accuracy by about 0.09–0.10. Within the group, `known_device` carries most of the single-column importance; the `new_device` column alone contributes little.
2. **Measured:** the models also use authentication anomaly, new beneficiary and recovery presence, but with much smaller drops. With the device group removed, Model B remains at 0.573 (interval excludes 0.5); Model A at 0.545 (interval includes 0.5).
3. **Measured:** the transaction group adds little in Model B (pooled AP drop 0.006, accuracy 0.010). This agrees with the earlier B-vs-A difference (+0.015 accuracy). The type one-hots carry what little there is; amount, deviation and count are near zero or negative.
4. **Inference:** the earlier observation that the models do not exceed the single device-version lookup (0.677) is consistent with finding 1. The design-audit lookup uses the generator's version directly, which the model can only reconstruct from three columns whose meaning differs by family; this may partly explain why the models fall slightly below it. Not measured here.
5. **Measured:** the negative control shows no unit with a material drop and an ablation within chance; no leakage signal was found.
6. **Assumption (not a finding):** `p_new_device` = 0.5 (legitimate) vs 0.9 (suspicious) is a provisional synthetic design parameter. The dominance in finding 1 reflects this chosen parameter, not any real-world rate.

## Limitations

- Synthetic data from one generator seed (42); seed-to-seed variance is unknown.
- Families are combinations of shared building blocks; held-out families are not truly novel.
- Per-family drops are computed on 20 cases and are noisy; negative per-family drops occur.
- Permutation importance can understate correlated features shuffled singly and can create unrealistic rows; grouped results are reported to limit this.
- Probabilities are uncalibrated; AP and ROC-AUC are ranking summaries only.
- Attribution describes model dependence on this data; it does not establish causation or real-world detection effectiveness.

## Verification

- `.venv\Scripts\python.exe -m pytest -q tests/ml/test_feature_attribution_experiment.py` — 4 passed.
- `.venv\Scripts\python.exe -m pytest -q tests/ml/test_family_holdout_experiment.py` — 6 passed.
- Input SHA-256 checks before/after the run: v3 `features.csv` and all `family_holdout_v3` outputs unchanged. Full suite not run.
