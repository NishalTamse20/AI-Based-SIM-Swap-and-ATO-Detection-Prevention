# Leave-One-Family-Out Evaluation (v3 Controlled SIM-Replacement Cohorts)

## Purpose

Measure whether the fixed Phase 8 Random Forest recovers the designed v3 patterns when an entire template family is held out, and confirm that the negative-control cohort stays near chance.

This is an evaluation of synthetic controlled cohorts (v3, generator seed 42). It is not evidence of real-user or production detection performance.

## Inputs

- `data/processed/controlled_sim_replacement_v3/predictive/` and `.../negative_control/` (`features.csv`, `case_metadata.csv`, `design_audit.json`).
- `template_family_id` comes only from `case_metadata.csv` and is used as the grouping key and for after-prediction reporting. It is never a model input.

## Method

- 20 folds per cohort. Each fold trains on 19 families (380 cases) and predicts the held-out family (20 cases, 10 per label). The held-out family never appears in training (tested).
- Both cohorts use the identical procedure; results are reported separately.
- **Model A:** the exact 21 Phase 8 feature columns (no post-trigger transaction columns).
- **Model B:** Model A plus `transaction_count`, `transaction_amount`, `transaction_deviation` and a one-hot `transaction_type` (PAYMENT / TRANSFER / CASH_OUT). Missing transaction amount/deviation remain missing in the data; a median imputer with missingness indicators is fit on each training fold only (reused `_phase9_pipeline`). The one-hot columns are all zero when no transaction is observed.
- Both models use the Phase 8 `RandomForestClassifier` configuration unchanged (100 trees, `class_weight="balanced"`, `random_state=42`) and the default `predict` decision rule. No thresholds or parameters were tuned.
- Excluded from inputs: labels, `scenario_type`, `template_family_id`, IDs, timestamps, case-design columns (`recipe_source`, `is_anchor`, `components_json`, intent) and PaySim fraud flags.
- Code: `ml/models/family_holdout_experiment.py` (reuses Phase 8 `make_classifier`/`binary_metrics` and Phase 9 `build_scenario_folds`/`_phase9_pipeline`).
- Outputs: `data/processed/family_holdout_v3/{predictive,negative_control}/` — `metrics.json`, `predictions.csv`, `family_error_summary.csv`, `run_metadata.json`.

## Coverage and metric validity

- The Random Forest predicts every case: coverage 400/400 per cohort, 0 abstentions, 0 undetermined cases.
- Every held-out family contains both classes (10/10), so per-family precision, recall, F1, FPR, FNR and ROC-AUC are all defined. No per-family metric was undefined in either cohort.
- The lookup baselines from the design audit do abstain on unseen or tied feature vectors; abstentions are counted as 0.5 there.
- `predict_proba` outputs are uncalibrated; ROC-AUC/AP are ranking summaries only.

## Pooled out-of-fold results

Wilson 95% intervals are computed from the observed counts.

| Cohort | Model | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR | Accuracy (95% CI) | ROC-AUC | AP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive | A | 135 | 77 | 123 | 65 | 0.637 | 0.675 | 0.655 | 0.385 | 0.325 | 0.645 (0.597–0.690) | 0.713 | 0.696 |
| Predictive | B | 138 | 74 | 126 | 62 | 0.651 | 0.690 | 0.670 | 0.370 | 0.310 | 0.660 (0.612–0.705) | 0.723 | 0.681 |
| Negative control | A | 104 | 96 | 104 | 96 | 0.520 | 0.520 | 0.520 | 0.480 | 0.480 | 0.520 (0.471–0.569) | 0.521 | 0.517 |
| Negative control | B | 111 | 101 | 99 | 89 | 0.524 | 0.555 | 0.539 | 0.505 | 0.445 | 0.525 (0.476–0.573) | 0.537 | 0.525 |

## Per-family spread (median, min–max across 20 held-out families)

| Cohort | Model | Recall | FPR | F1 |
|---|---|---|---|---|
| Predictive | A | 0.70 (0.30–0.90) | 0.35 (0.10–0.80) | 0.668 (0.40–0.842) |
| Predictive | B | 0.70 (0.50–0.90) | 0.40 (0.10–0.80) | 0.667 (0.50–0.90) |
| Negative control | A | 0.50 (0.20–0.90) | 0.50 (0.30–0.80) | 0.524 (0.222–0.762) |
| Negative control | B | 0.60 (0.10–1.00) | 0.50 (0.20–0.90) | 0.53 (0.125–0.87) |

Each family's TP/FP/TN/FN for both models is in `family_error_summary.csv`; every FP/FN row is in `predictions.csv`. With 10 cases per class, one case moves a per-family recall or FPR by 10 points.

## Comparison with the v3 design audit

| Cohort | Full-vector lookup (abstain = 0.5) | Device-version lookup (design audit `new_device` component) | Model A accuracy | Model B accuracy |
|---|---|---|---|---|
| Predictive | 0.580 (222/400 covered) | 0.677 | 0.645 | 0.660 |
| Negative control | 0.495 (252/400 covered) | 0.463 | 0.520 | 0.525 |

- **Baseline definition (correction):** the 0.677 lookup is the design audit's `new_device` *component*, which measures the generator's **device version** (new vs known device). It is not a lookup on the `new_device` feature column alone. Depending on the scenario family, new-device information appears in different feature columns: families whose device event is `NEW_DEVICE` set `new_device = 1`, families whose device event is `DEVICE_CHANGE` set `device_change = 1` (with `new_device = 0`), and in every family the known-device version sets `known_device = 1`. The device version is therefore represented across `new_device`, `device_change` and `known_device`, not by any single column. See `docs/requirements/feature-attribution-evaluation.md`.
- **Predictive:** both models beat the full-vector lookup, but **neither exceeds the device-version lookup (0.677)**. The model results are therefore consistent with recovering mainly the designed device-version difference, which is a provisional synthetic assumption (below), rather than a combination of signals. Feature importance was not computed in this run, so this is an inference, not a measured attribution; the later attribution audit measured the three device columns jointly as the dominant group.
- **Model B vs A:** +0.015 accuracy (6 more correct cases of 400). The intervals overlap almost entirely; there is no evidence here that the transaction columns add signal.
- **Negative control:** both models are at chance (accuracy 0.520 / 0.525, ROC-AUC 0.521 / 0.537; the 95% intervals include 0.5). This is the expected null result: no label leakage was detected through the generator or the evaluation.

### Theoretical ceiling (added after the evaluation)

Exact Bayes accuracy ceilings for the predictive design, computed from the manifest with equal priors and **conditional on the realized seed-42 skeleton sets** (`data/processed/bayes_ceiling_v3/v3_default_ceiling.json`; method in `docs/requirements/interaction-cohort-design.md`):

| Ceiling | Value | Compare with |
|---|---|---|
| Version-level upper bound (oracle sees skeleton and every present version) | 0.685 | — not reachable from features |
| Feature-level, Model A columns | 0.660 | Model A accuracy 0.645 |
| Feature-level, Model B columns | 0.679 | Model B accuracy 0.660 |

- The feature-level ceiling is the relevant comparison: it is the Bayes ceiling for the stated feature representation and population (family-dependent encodings and missing-slot merging included). It is not an achievable target for leave-one-family-out models, which cannot learn a held-out family's encodings from that family.
- Ceilings are expectations over the generator's version draws. On this seed's realized sample, applying the exact Bayes decision rule to the saved feature rows gives 0.693 (Model A columns) and 0.703 (Model B columns), about 1–1.4 standard errors above the expected ceilings, consistent with the favourable realized device draw.
- The negative control's theoretical ceiling is 0.5 by construction; its observed finite-sample accuracy is a separate quantity.

## Required disclosures and limitations

- `p_new_device` = 0.5 (legitimate) vs 0.9 (suspicious) is a **provisional synthetic design assumption, not an empirical rate**. The other version probabilities, the 30% anchor share and the slot quotas are also provisional.
- All results come from synthetic data generated with seed 42. No other seeds were run; seed-to-seed variance is unknown.
- Families are combinations of shared building blocks; a held-out family is not truly novel.
- Eight features are constant in both cohorts (`sim_change`, `esim_change`, `telecom_event_present`, `device_deviation` and four fixed-history baseline counts).
- Probabilities are uncalibrated; no risk score, band or threshold is validated by this evaluation.
- These results do not establish real-world detection effectiveness.

## Verification

- `.venv\Scripts\python.exe -m pytest -q tests/ml/test_family_holdout_experiment.py` — 6 passed.
- `.venv\Scripts\python.exe -m pytest -q tests/ml/test_unseen_scenario_experiment.py tests/ml/test_random_forest_experiment.py` — 13 passed.
- Full suite not run.
