# Phase 10.1 — Hybrid Risk Engine Specification

**Status:** design proposal only; no engine, score, risk levels, or action policy is implemented by this document.

## Purpose and scope

Specify a future, explainable research component that could combine the existing Phase 7 rules, Phase 8 Random Forest evidence, and Phase 9 temporal evidence after a SIM/eSIM trigger. It operates on synthetic data and the frozen inclusive T-to-T+15-minute snapshot. A SIM/eSIM event is context, never proof of ATO. Outputs are research recommendations, not account actions. Live telecom/banking integration, APIs, databases, deployment, and frontend work are out of scope.

## Existing behavior and available inputs

The inspected files do not expose a unified risk-engine input/output interface or a fitted model serving interface.

| Source | Existing behavior and fields |
|---|---|
| Phase 7 rules | generate_rule_baselines produces separate telecom_context and multi_signal result lists. Each result can include identifiers, prediction, matched_rules, feature_evidence, and explanation. Telecom inputs are telecom_event_present, sim_change, esim_change. Multi-signal inputs are new_device, authentication_anomaly, password_reset, failed_login_count, device_change, device_deviation. Its three outcomes are NO_ATO_RISK_INDICATOR, ATO_RISK_INDICATOR, and AMBIGUOUS_INSUFFICIENT_EVIDENCE; no-indicator does not certify legitimacy. |
| Phase 8 Random Forest | extract_xy selects the fixed 21 Phase 8 feature columns and ato_label as target; IDs are excluded from X. run_experiment evaluates a stratified random split and writes held-out predictions, including rf_probability_ato. This is an experimental output, not an online prediction API. The inspected implementation does not calibrate the probability. |
| Phase 9 temporal model | run_temporal_comparison compares Phase 8 inputs with the same inputs plus 13 temporal features using the fixed random split and train-only median imputation/missingness indicators. It returns aggregate metrics, not per-user runtime assessments. The unseen-scenario experiment emits out-of-fold predictions for evaluation; these are fold-specific research outputs, not a fitted serving model. |

Phase 8 X includes the 21 existing fields: telecom (sim_change, esim_change, telecom_event_present); device (new_device, device_change, device_deviation, known_device); authentication/recovery/account (failed_login_count, authentication_anomaly, password_reset, account_recovery, recovery_event_present, new_beneficiary, account_change); and pre-trigger baseline fields (baseline_event_count, baseline_event_frequency, baseline_transaction_count, baseline_transaction_amount_mean, baseline_transaction_amount_min, baseline_transaction_amount_max, baseline_successful_login_count). Post-trigger transaction features are not model inputs. Phase 9 adds the already-defined time-to-event, count, and sequence fields; it does not extend T+15.

scenario_type and PaySim fraud flags are not model inputs. ato_label is the research target only. The Phase 9 unseen-scenario evaluation found strong synthetic-template dependence and missed all 125 SUSPICIOUS_SIM_REPLACEMENT cases. Phase 9B found that timing alone matched all temporal features on its pooled metrics; counts alone did not improve F1 over baseline, and sequences alone had a smaller F1 increase. These results do not establish real-world performance.

## Proposed output schema

The following is a future design contract, not an existing output:

| Field | Proposed meaning |
|---|---|
| assessment_id | Unique assessment trace ID |
| user_id, account_id, event_id | Existing traceability identifiers, not model evidence |
| prediction_window | Frozen window identifier, T_TO_T_PLUS_15_MINUTES |
| decision_status | PROVISIONAL, INSUFFICIENT_EVIDENCE, or CONFLICTING_EVIDENCE |
| component_evidence | Separate rule outcomes/evidence and model prediction/probability, with model and rule versions |
| risk_score_0_100 | Nullable until a validated mapping is selected |
| risk_level | LOW, MEDIUM, HIGH, or UNDETERMINED; no boundaries are defined here |
| recommended_action | Nullable; when determined, one of Allow, Step-Up Authentication, Protect/Security Review |
| reason_codes, explanation | Human-readable source evidence and reason for uncertainty |
| audit | Feature-snapshot reference/hash, component versions, score/level policy version, and assessment time |

Current feature CSVs contain IDs but do not expose a unified snapshot ID or trigger timestamp as an assessment field. Those would need to be supplied or deliberately omitted by a future interface; they must not be fabricated.

## Candidate scoring approaches

1. **Deterministic rule score.** Transparent and reproducible, but needs explicitly defined evidence semantics and validated weights/boundaries. Phase 7 currently returns categorical outcomes and evidence, not points or calibrated severity.
2. **Calibrated model probability mapped to a score.** A mapping such as 100 times a calibrated ATO probability is interpretable only after calibration has been implemented and evaluated. The current predict_proba values are not calibrated probabilities by claim, and Phase 9 fold-specific scores are evaluation outputs.
3. **Hybrid decision policy.** Keep model evidence and rule outcomes separate, then apply explicit policy branches. This avoids pretending heterogeneous evidence is additive, but each branch and its treatment of ambiguity still needs a prespecified, held-out evaluation.

**Decision unresolved:** current evidence does not justify selecting one scoring approach or combining arbitrary rule points with Random Forest probabilities. Before choosing, implement candidate policies as research-only alternatives, compare them on the same held-out scenario folds, preserve ambiguous outcomes, and report calibration/decision behavior and scenario errors. Do not select a design by tuning on the final evaluation results.

## Double-counting, missing evidence, and conflicts

Phase 7 rules are built from Phase 6 features that also appear in the Random Forest inputs. Phase 9 timing/count/sequence fields describe many of those same events. Adding a rule score to an RF score would therefore count overlapping evidence more than once. A future hybrid should retain component evidence separately and either use one validated combined model or define policy rules whose incremental effect is evaluated; it must not sum correlated signals by default.

Keep missingness distinct from observed absence. In particular, a missing time-to-event value is not an event at time zero. Preserve the Phase 9 train-only preprocessing behavior for model use and retain missingness in audit evidence. Treat Phase 7 ambiguity, unavailable model output, or material disagreement between components as uncertain/conflicting; do not silently convert these to legitimate, suspicious, or a numeric midpoint. If no validated score can be produced, leave the score null and level UNDETERMINED. A SIM/eSIM-only signal must not raise an ATO conclusion.

## Score, levels, and recommended actions

A 0–100 scale and Low/Medium/High levels can be defined later as explicit experimental design choices, not industry standards. First decide whether the number represents calibrated probability, a validated rule index, or an ordinal ranking; these meanings are not interchangeable. If probability semantics are chosen, evaluate calibration on data separate from model fitting and report calibration results before mapping it to 0–100. If an ordinal score is chosen, document that it is not a probability.

Only after that choice, prespecify two boundaries for Low/Medium/High using a documented research decision process and freeze them before final held-out evaluation. This specification assigns no numeric values. A proposed qualitative mapping is Low → Allow, Medium → Step-Up Authentication, High → Protect/Security Review. The levels and mapping remain unvalidated; for undetermined or conflicting evidence, return no action recommendation until a conservative policy is explicitly validated. All actions are recommendations in the synthetic prototype, not actual account controls.

## Auditability, reproducibility, testing, and limitations

A future implementation should record the input snapshot/window, missingness, rule IDs and evidence, raw versus calibrated model output, model/rule/policy versions, score and level definitions, and the recommended action with its rationale. Reproducibility requires fixed model/preprocessing configuration, random seeds, input hashes, and identical evaluation folds when comparing candidates.

Tests should cover schema and identifier traceability; T+15 cutoff; target/scenario/PaySim exclusion; rule ambiguity preservation; missing and contradictory signals; deterministic scoring; calibration behavior if introduced; frozen level boundaries; action mapping; and no double counting. Evaluate every policy on the same unseen-scenario folds, including the SUSPICIOUS_SIM_REPLACEMENT failure.

Evidence remains limited to synthetic templates. Current scenario types are label-pure, temporal profiles strongly reflect template construction, and Phase 9 missed 125 suspicious SIM-replacement cases. Calibration, score boundaries, hybrid-policy benefit, action costs, and generalization to real users are unresolved. No causal or real-world effectiveness claim is supported.
