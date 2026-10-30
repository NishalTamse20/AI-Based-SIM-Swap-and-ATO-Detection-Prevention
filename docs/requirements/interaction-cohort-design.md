# Interaction Cohort Design (v3, coordinated mixture)

Status: **design and Step 1 code only.** No interaction cohort, ceiling output, LOFO evaluation or attribution has been produced yet. Each later step needs separate approval.

## Purpose

The v3 default cohort makes every label signal an independent per-slot probability difference, so it cannot test whether models use *combinations* of signals. The interaction cohort keeps everything identical except the event-version model: each latent signal has the same marginal rate for both labels, and the label differs only in how signals co-occur.

## Mechanism (provisional synthetic assumption)

Latent risk versions: D = new device, A = auth failure, C = `NEW_BENEFICIARY`, T = high-risk transaction. All four are drawn for every variant case; the label-neutral skeleton decides which are observed.

- Legitimate (and negative-control `profile_A`): independent, P(s risky) = q_s.
- Suspicious (and negative-control `profile_B`): with probability ρ all four are risky; otherwise independent with P(s risky) = r_s = (q_s − ρ)/(1 − ρ).
- q = (0.7, 0.5, 0.5, 0.6) — the label-average of the v3 default rates — and ρ = 0.25, so r = (3/5, 1/3, 1/3, 7/15). These were fixed before any ceiling or model result and are not tuned. They are not empirical rates, and the coordinated pattern is not a validated fraud pattern.
- Unchanged from v3: families, skeletons, quotas, anchors, ordering, timestamps, amounts, history, the `SIM_REPLACEMENT` trigger. eSIM and temporal coupling are excluded.
- Manifest: `simulator/scenarios/sim_replacement_v3_interaction_manifest.json` (`version_model: "coordinated_mixture"`). The default v3 manifest has no `version_model` key and follows the original code path.

## Mathematical properties (exact)

- Marginals: P(s risky | S) = ρ + (1 − ρ) r_s = q_s = P(s risky | L) for every slot, independent of the skeleton. Anchors have fixed versions identical for both labels.
- Joints: for any observed set O with |O| ≥ 2, P(all of O risky | S) = ρ + (1 − ρ) Π r_s > Π q_s = P(all of O risky | L). Pairwise covariance is ρ(1 − q_s)(1 − q_t)/(1 − ρ) under S (equivalently ρ(1 − ρ)(1 − r_s)(1 − r_t)) and 0 under L.
- The mechanism is AND-like; a linear threshold can represent it. It tests joint use of signals, not non-linearity specifically.

Verified by exact `Fraction` tests. A separate fixed-seed 200,000-draw sanity check confirms the sampler's rates and pairwise co-occurrences within ±0.005 of the exact values.

## Finite-sample behaviour

With 7 variant cases per family/label cell, realized rates will differ from q by chance. These deviations are **reported, not forced**: `design_audit.json` (interaction mode only) contains per-slot latent rates by label (pooled and per family), rate differences, pairwise co-occurrence, all-four-risky rates and coordinated-component counts.

## Negative control

Recipe sources, skeleton assignment and versions are drawn from the seed and case/family identity only; labels come from `label_seed` and are attached last. By construction P(events, features | label) = P(events, features), so the **theoretical** Bayes ceiling is exactly 0.5. This assumes the seed-derived hash streams behave as independent and labels are balanced per family. **Observed** accuracy on the finite negative-control sample can differ from 0.5 by chance and is a separate quantity.

## Bayes ceilings (computed before evaluation)

Both use equal priors and the population of 20 families × 10 cell positions (3 anchors + 7 skeletons), and are **conditional on the realized seed-42 skeleton sets**. Code: `simulator/generators/sim_replacement_v3_ceiling.py`.

- **Version-level ceiling (upper bound):** an oracle sees each case's position and the version of every present slot. Σ_cases (1/N) Σ_x ½ max(P(x | L, O), P(x | S, O)); anchors contribute ½.
- **Feature-level ceiling:** the Bayes ceiling for the specified feature representation and population. Every (family, position, latent combination) is pushed through the generator's `_token_event` mapping and the real Phase 5/6 builders; ceiling = ½ Σ_f max(P(f | L), P(f | S)). Family-dependent encodings (`new_device` vs `device_change`; `failed_login_count` only in `FAILED_LOGIN` families) and missing-slot merging (auth success ≡ absent auth slot) come from the real code. Reported for the Model A and Model B column sets.
- The feature-level ceiling is **not** an achievable target for leave-one-family-out models, which cannot learn a held-out family's encodings from that family.

Excluded columns and why (audited against `_build_history`, `_generate_events` and `build_feature_rows`):
- `baseline_*` derive only from history built from seed/case identity; `transaction_amount` comes from a shared pool assigned by seed/case rank; `transaction_deviation` is amount minus the baseline mean. None reads labels, intents, skeletons or versions. Tests confirm history, profiles and per-case amounts are identical across label seeds and across the two version models.
- Phase 6 features in these column sets do not use event order or timestamps (tested by shuffling).
- If the hash-stream independence assumption failed, these columns could carry incidental information; this is a documented limitation.

Planned outputs (Step 2, not yet produced): `data/processed/bayes_ceiling_v3/v3_default_ceiling.json`, `data/processed/bayes_ceiling_v3/v3_interaction_ceiling_preregistration.json`.

## Evaluation protocol (Step 4, not yet run)

The existing `family_holdout_experiment` and `feature_attribution_experiment` CLIs are reused unchanged via `--cohort-root`/`--holdout-root`/`--output-root`: 20 family folds, the fixed Phase 8 configuration, train-fold-only imputation, no threshold tuning. Latent bits and the coordination flag live only in `components_json`, which is excluded from model inputs.

## Limitations

- Synthetic data, one seed; ceilings are conditional on the realized skeletons.
- The coupling is global, so a held-out family does not test an unseen interaction mechanism.
- The joint pattern does not establish or represent real-world fraud behaviour.
- q and ρ are provisional assumptions; they are not chosen for, or validated by, model accuracy.
- Probabilities from models are uncalibrated; no real-world detection claim.
