# Phase 10.3 — Hybrid Policy Evaluation

## Scope and evaluation inputs

This evaluation applies two fixed categorical policies to the existing eight
leave-one-scenario-type-out (LOSO) out-of-fold predictions. It does not fit or
tune models. Phase 7 `multi_signal` results are regenerated deterministically
from frozen Phase 6 features. Saved `model_a` is the Phase 8 reference; saved
`model_b` is the sole learned signal used by both policies. Model B already
uses Phase 8 features plus all 13 Phase 9 temporal features. The Phase 9B
`ALL_TEMPORAL` and `BASELINE` rows are checked against Model B and Model A;
these models are not independent votes.

Inputs are aligned on the complete `(user_id, account_id, event_id)` key.
Duplicates, unmatched keys, row-count discrepancies, fold mismatches, or label
misalignment stop the evaluation. Policy outcomes are constructed before
`ato_label` and scenario evaluation metadata are attached. Null temporal
feature values are preserved; they are not interpreted as safe or absent
evidence. A missing Model B prediction or rule output yields insufficient
evidence (a missing row in a source artifact instead fails exact-key
validation).

## Fixed decision tables

`R` is the Phase 7 outcome; `M` is the saved Phase 9 Model B prediction. A
missing model prediction or rule output is `MISSING`. A Phase 7 ambiguous
outcome remains ambiguous regardless of model output. A rule risk indicator
with model prediction 0 is conflicting. `NO_ATO_RISK_INDICATOR` means no
configured rule matched, never that the account is legitimate.

### Policy 1 — RF-led, rule-context preserving

| R | M = 1 | M = 0 | M missing |
|---|---|---|---|
| `ATO_RISK_INDICATOR` | `ATO_RISK_INDICATOR` (corroborated) | `CONFLICTING_EVIDENCE` | `INSUFFICIENT_EVIDENCE` |
| `NO_ATO_RISK_INDICATOR` | `MODEL_ONLY_RISK_INDICATOR` | `NO_ATO_RISK_INDICATOR` | `INSUFFICIENT_EVIDENCE` |
| `AMBIGUOUS_INSUFFICIENT_EVIDENCE` | ambiguous | ambiguous | ambiguous |
| Rule missing | insufficient | insufficient | insufficient |

### Policy 2 — rule corroboration required

| R | M = 1 | M = 0 | M missing |
|---|---|---|---|
| `ATO_RISK_INDICATOR` | `ATO_RISK_INDICATOR` (corroborated) | `CONFLICTING_EVIDENCE` | `INSUFFICIENT_EVIDENCE` |
| `NO_ATO_RISK_INDICATOR` | `AMBIGUOUS_INSUFFICIENT_EVIDENCE` | `NO_ATO_RISK_INDICATOR` | `INSUFFICIENT_EVIDENCE` |
| `AMBIGUOUS_INSUFFICIENT_EVIDENCE` | ambiguous | ambiguous | ambiguous |
| Rule missing | insufficient | insufficient | insufficient |

No points, probabilities, feature values, or temporal events are added or
counted as extra votes. Temporal evidence is already represented in Model B.

## Metrics and interpretation

Full-set Phase 8 and Phase 9 reference metrics are reported separately. For
each policy, determinate outcomes are the policy risk-indicator outcomes and
`NO_ATO_RISK_INDICATOR`; coverage is determinate rows divided by all expected
joined rows. Ambiguous, conflicting, and insufficient outcomes are excluded
from determinate metrics and reported by count and actual-label composition.
For determinate-case metric calculation only, risk-indicator outcomes map to
binary 1 and `NO_ATO_RISK_INDICATOR` maps to binary 0. This is an evaluation
convention, not a legitimate-account classification. Model A and Model B are
also evaluated on exactly each policy's determinate rows. Per-scenario
coverage and errors are reported.

Policy outcomes have no continuous score, so policy ROC-AUC and average
precision are unavailable. Per-fold ROC-AUC is undefined because each held-out
scenario fold contains one class. Pooled reference ROC-AUC and average
precision are descriptive only: each probability comes from a separately
trained fold model, and the cross-fold probability scales are uncalibrated and
may not be comparable.

The evaluation exports all 125 `SUSPICIOUS_SIM_REPLACEMENT` rows and requires
that the saved Phase 9 Model B prediction remains negative for all 125. This
preserves the documented 125 false negatives; the policy does not tune around
or conceal them. These results describe synthetic scenario templates and are
not real-world performance claims.

## Outputs and reproducibility

The run writes `metrics.json`, `predictions.csv`,
`scenario_error_summary.csv`, `suspicious_sim_replacement_cases.csv`, and
`run_metadata.json` under `data/processed/hybrid_policy_evaluation/` only.
Metadata records source hashes, row counts, the unchanged eight fold
assignments, policy definitions, and the fact that no model was fitted.
Predictions retain identifiers, Phase 7 output, Phase 8/9 predictions and
probabilities, both policy outcomes/errors, evaluation-only labels/scenario
fields, and the 13 raw temporal values with their nulls preserved.
