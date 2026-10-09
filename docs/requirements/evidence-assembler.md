# Phase 10.2 — Evidence Assembler

## Scope

`ml.risk.evidence_assembler.assemble_evidence` packages explicitly supplied
Phase 7 rule output, model output, temporal evidence, trace identifiers, and
provenance. It does not read experiment CSVs or imply those artifacts are a
live inference interface. It does not calculate a score, risk level, or action.

## Record

The returned record contains `assessment_id`, `identifiers`, `evidence_status`,
and separate `rule_evidence`, `model_evidence`, `temporal_evidence`, and
`provenance` fields. Missing components are `None`; missing temporal values
remain `None`. Phase 7 outcomes, including
`AMBIGUOUS_INSUFFICIENT_EVIDENCE`, are preserved. A disagreement between a
rule `ATO_RISK_INDICATOR` and model prediction `0` is marked `CONFLICTING`;
neither component is overwritten or converted into a fused label. A rule
`NO_ATO_RISK_INDICATOR` means no configured rule matched; it is not treated as
a negative/legitimate classification and does not conflict by itself with a
model prediction of `1`. A rule ambiguous outcome takes precedence and remains
`AMBIGUOUS`. Any absent rule, model, or temporal component otherwise yields
`INCOMPLETE`. Model probabilities are passed through as supplied and are not
claimed to be calibrated.

Rule predictions and evidence structure, binary model predictions, optional
finite probabilities in `[0, 1]`, non-empty string identifiers, and temporal
numeric-or-`None` values are validated. Invalid values raise `ValueError`;
they are not silently reinterpreted as absent or negative evidence.

## Limitations

This is an evidence packaging utility, not a trained model serving layer or a
validated hybrid risk engine. It introduces no weights, thresholds, score, or
automated response. Phase 9 evidence remains subject to synthetic-template
dependence; the unseen-scenario evaluation missed all 125
`SUSPICIOUS_SIM_REPLACEMENT` cases. That result is not repaired or hidden by
this assembler. Any future fused policy requires separate held-out evaluation
and must preserve ambiguity and missingness.
