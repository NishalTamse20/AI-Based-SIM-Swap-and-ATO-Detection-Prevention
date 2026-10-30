# Phase 10.1 — Hybrid Risk Engine Specification

**Status:** Phase 10 remains a research specification; Phase 11.1 implements only a downstream evidence explainer and conservative simulated response mapper. No numeric score or calibrated risk level is implemented.

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

## Phase 11.1 implemented behavior

`ml.risk.explainable_response.build_explainable_response` accepts an explicit
record from the Phase 10.2 evidence assembler. It keeps rule, model, temporal,
and provenance fields separate, emits stable reason codes, and rejects records
whose component-derived evidence status is inconsistent. It does not read the
saved experiment CSVs or provide model serving.

| Assembled evidence | Explainable outcome | Simulated recommendation |
|---|---|---|
| Rule `ATO_RISK_INDICATOR` and model prediction `1` | `CORROBORATED_RISK_INDICATOR` | Protect/Security Review |
| Rule `NO_ATO_RISK_INDICATOR` and model prediction `1` | `MODEL_ONLY_RISK_INDICATOR` | Step-Up Authentication |
| Rule `NO_ATO_RISK_INDICATOR` and model prediction `0` | `NO_ATO_RISK_INDICATOR` | No recommendation |
| Rule `AMBIGUOUS_INSUFFICIENT_EVIDENCE` | `AMBIGUOUS` | No recommendation |
| Rule risk indicator and model prediction `0` | `CONFLICTING_EVIDENCE` | No recommendation |
| Required rule, model, or temporal component missing | `INSUFFICIENT_EVIDENCE` | No recommendation |

These mappings are deterministic prototype design choices, not validated
security actions. The absence of a configured rule indicator or a negative
model prediction is not a legitimacy finding, so no `Allow`/normal-access
recommendation is inferred. An ambiguous rule remains ambiguous even when the
model predicts risk. Missing temporal components and individual missing
temporal values remain explicit and are not treated as safe evidence. Temporal
features are not added as another vote because Phase 9 Model B already uses
them.

Each output retains component evidence and provenance, includes reason codes
for the outcomes and missing temporal values, sets `risk_score_0_100` to null,
and sets `risk_level` to `UNDETERMINED`. No weights, thresholds, calibrated
probability claims, or numeric score are defined. Selecting an `Allow` action,
risk levels, or score requires a separately specified policy and held-out
evaluation. Recommendations are simulated only; there are no live account
actions or telecom/banking integrations.

Tests cover rule/model agreement, model-only risk, no-indicator, ambiguity,
missing evidence, conflicting evidence, deterministic output, and preservation
of input evidence. Evaluation must continue to expose the synthetic-template
limitation and the 125 missed `SUSPICIOUS_SIM_REPLACEMENT` cases; this component
does not change or repair those results.

## Phase 11.2 — Scoring decision and validation plan

**Decision:** do not produce a 0–100 score or Low/Medium/High level from the
current evidence. Keep `risk_score_0_100` null and `risk_level` as
`UNDETERMINED`.

The saved Random Forest probability outputs have not been calibrated. The
unseen-scenario evaluation holds out eight label-pure scenario types, so
per-fold ROC-AUC is undefined; pooled probabilities come from separately
trained fold models and are not established as comparable or calibrated. The
Phase 10.3 policy's perfect determinate-only metrics cover 500 of 1,000 rows
(50% coverage) and do not validate a continuous score. The temporal model's
125 false negatives on `SUSPICIOUS_SIM_REPLACEMENT` also remain material.
These results are synthetic and show scenario-template dependence.

Before any score is implemented, resolve and document:

- Whether a score means calibrated ATO probability or a non-probabilistic
  ordinal index. The existing results support neither a calibrated probability
  mapping nor additive rule/model/temporal points.
- The training-only method for producing that quantity and the independent
  validation design. Current label-pure leave-one-scenario-type-out folds do
  not provide per-fold mixed-class calibration evidence.
- If levels are retained, the process for choosing and freezing Low/Medium/
  High boundaries and their response mapping before final evaluation.
- How ambiguous, missing, conflicting, and no-indicator outcomes remain
  explicit without being converted to legitimate activity.

Validation must use a prespecified score meaning and method, fit any
calibration only on training data, and assess it on an independent held-out
set with sufficient class and scenario variation. Report calibration
reliability and discrimination only where mathematically supported, along
with coverage and scenario-level errors. Keep all 125
`SUSPICIOUS_SIM_REPLACEMENT` cases visible. Freeze any level boundaries and
response policy before evaluation; do not choose them to improve the reported
test result. Until those decisions and evidence exist, score and level outputs
remain null/undetermined and no real-world validity is implied.
