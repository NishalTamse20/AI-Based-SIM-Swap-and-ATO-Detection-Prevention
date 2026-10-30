"""Generate isolated SIM_REPLACEMENT v2 negative-control and predictive cohorts.

No model fitting or evaluation is performed here. Evaluation metadata is kept
in a separate case manifest; Phase 6 receives only its established event and
baseline schemas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

from ml.features.feature_generator import FEATURE_COLUMNS, OUTPUT_COLUMNS, build_feature_rows
from simulator.baseline.behavioural_baseline import (
    BASELINE_FIELDS,
    build_baseline_profiles,
    validate_baseline_cutoffs,
)
from simulator.config.generator_config import DEFAULT_PAYSIM_PATH
from simulator.generators.event_generator import EVENT_FIELDS, make_event
from simulator.generators.paysim_loader import sample_transactions
from simulator.validation.dataset_validator import CATEGORY_EVENT_TYPES


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "simulator" / "scenarios" / "sim_replacement_v2_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "controlled_sim_replacement_v2"
CASES_PER_FAMILY = 20
CASES_PER_LABEL_PER_FAMILY = 10
FAMILY_COUNT = 20
TOTAL_CASES = CASES_PER_FAMILY * FAMILY_COUNT
HISTORY_EVENTS_PER_KIND = 5
HISTORY_WINDOW_DAYS = 30
OBSERVATION_MINUTES = 15
_STREAM_OFFSETS = {"events": 1009, "labels": 2017, "paysim_sample": 3023,
                   "paysim_allocation": 4051}


def _stream_seed(seed: int, name: str) -> int:
    return int(seed) + _STREAM_OFFSETS[name]


def _stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def _load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _event_signature(rows: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    trigger = next(row for row in rows if row["event_category"] == "TELECOM")
    trigger_time = datetime.fromisoformat(trigger["timestamp"])
    return tuple(
        (row["event_category"], row["event_type"], row["status"])
        for row in sorted(rows, key=lambda row: datetime.fromisoformat(row["timestamp"]))
        if row["event_category"] != "TELECOM"
    )


def _case_specs(mode: str, manifest: dict[str, Any], seed: int,
                label_seed: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in {"negative_control", "predictive"}:
        raise ValueError("mode must be 'negative_control' or 'predictive'")
    families = manifest.get("families", [])
    if len(families) != FAMILY_COUNT:
        raise ValueError(f"manifest must define exactly {FAMILY_COUNT} template families")
    definitions = manifest.get("event_definitions", {})
    normalized_recipe_family: dict[tuple[str, ...], str] = {}
    for family in families:
        family_id = str(family["template_family_id"])
        for intent in ("legitimate", "suspicious"):
            recipe = family.get(intent, [])
            if len(recipe) != 4 or any(name not in definitions for name in recipe):
                raise ValueError(f"{family_id}/{intent} must define four supported events")
            normalized = tuple(sorted(recipe))
            prior = normalized_recipe_family.get(normalized)
            if prior is not None and prior != family_id:
                raise ValueError(f"near-duplicate event multiset crosses families: {prior}/{family_id}")
            normalized_recipe_family[normalized] = family_id
    event_rng = random.Random(_stream_seed(seed, "events"))
    labels_rng = random.Random(_stream_seed(seed if label_seed is None else label_seed, "labels"))
    case_metadata: list[dict[str, Any]] = []
    event_plans: list[dict[str, Any]] = []

    # Opaque case identity is derived from family and slot, not from its label.
    for family in families:
        family_id = str(family["template_family_id"])
        if not family_id:
            raise ValueError("empty template_family_id")
        for slot in range(CASES_PER_FAMILY):
            case_id = f"{mode.upper()}-{family_id}-{slot + 1:02d}"
            case_metadata.append({"case_id": case_id, "template_family_id": family_id,
                                  "slot": slot})

    # The label stream creates a balanced, shuffled schedule independently for
    # each family. Negative-control event plans do not read this schedule.
    labels_by_case: dict[str, int] = {}
    intent_by_case: dict[str, str] = {}
    family_by_id = {str(f["template_family_id"]): f for f in families}
    for family_id in sorted(family_by_id):
        ids = [c["case_id"] for c in case_metadata if c["template_family_id"] == family_id]
        if mode == "predictive":
            intent_schedule = (["legitimate"] * CASES_PER_LABEL_PER_FAMILY
                               + ["suspicious"] * CASES_PER_LABEL_PER_FAMILY)
            labels_rng.shuffle(intent_schedule)
            intent_by_case.update(zip(ids, intent_schedule))
            labels_by_case.update({case_id: int(intent == "suspicious")
                                   for case_id, intent in zip(ids, intent_schedule)})
        else:
            label_schedule = [0] * CASES_PER_LABEL_PER_FAMILY + [1] * CASES_PER_LABEL_PER_FAMILY
            labels_rng.shuffle(label_schedule)
            labels_by_case.update(zip(ids, label_schedule))

    for metadata in sorted(case_metadata, key=lambda c: c["case_id"]):
        family = family_by_id[metadata["template_family_id"]]
        label = labels_by_case[metadata["case_id"]]
        if mode == "predictive":
            intent = intent_by_case[metadata["case_id"]]
            selected_names = list(family[intent])
        else:
            intent = None
            # Label-independent sequence draw: neither the label nor its
            # schedule is consulted to choose this event sequence.
            selected_names = list(event_rng.choice((family["legitimate"], family["suspicious"])))
        if not selected_names:
            raise ValueError(f"empty event recipe for {metadata['template_family_id']}")
        event_specs = []
        for event_name in selected_names:
            if event_name not in definitions:
                raise ValueError(f"undefined event type in manifest: {event_name}")
            event_specs.append({"event_type": event_name, **definitions[event_name]})
        metadata.update({"scenario_type": "SUSPICIOUS_SIM_REPLACEMENT" if label else
                         "LEGITIMATE_SIM_REPLACEMENT", "ato_label": label,
                         "declared_scenario_intent": intent})
        event_plans.append({**metadata, "event_specs": event_specs})
    return case_metadata, event_plans


def _generate_events(event_plans: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(_stream_seed(seed, "events"))
    all_events: list[dict[str, Any]] = []
    trigger_context: list[dict[str, Any]] = []
    for case_index, plan in enumerate(sorted(event_plans, key=lambda p: p["case_id"]), start=1):
        case_id = plan["case_id"]
        # Stable identifiers do not encode label; case ordering is lexical and
        # stable even if the manifest's family list is reordered.
        identity = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12].upper()
        user_id, account_id = f"USR-SIMREP2-{identity}", f"ACC-SIMREP2-{identity}"
        known_device, new_device = f"DEV-SIMREP2-{identity}-K", f"DEV-SIMREP2-{identity}-N"
        trigger_time = datetime(2026, 1, 1) + timedelta(days=case_index - 1)
        trigger_id = f"EVT-SIMREP2-{identity}-00"
        trigger = make_event(event_id=trigger_id, user_id=user_id, account_id=account_id,
                             event_category="TELECOM", event_type="SIM_REPLACEMENT",
                             timestamp=trigger_time, device_id=known_device,
                             scenario_type=None, ato_label=None, status="SUCCESS")
        trigger_context.append({"case_id": case_id, "template_family_id": plan["template_family_id"],
                                "user_id": user_id, "account_id": account_id,
                                "trigger_event_id": trigger_id, "trigger_time": trigger_time,
                                "known_device_id": known_device})
        elapsed_seconds = sorted(rng.sample(range(1, OBSERVATION_MINUTES * 60 + 1),
                                            len(plan["event_specs"])))
        case_events = [trigger]
        for index, (spec, offset) in enumerate(zip(plan["event_specs"], elapsed_seconds), start=1):
            case_events.append(make_event(
                event_id=f"EVT-SIMREP2-{identity}-{index:02d}", user_id=user_id,
                account_id=account_id, event_category=spec["event_category"],
                event_type=spec["event_type"], timestamp=trigger_time + timedelta(seconds=offset),
                device_id=new_device, scenario_type=None, ato_label=None, status=spec["status"],
            ))
        case_events.sort(key=lambda row: (row["timestamp"], row["event_id"]))
        all_events.extend(case_events)
    return all_events, trigger_context


def _build_history(trigger_context: list[dict[str, Any]], paysim_path: str | Path,
                   seed: int) -> list[dict[str, Any]]:
    # Sampling and case-to-bundle assignment have separate deterministic RNGs.
    sample_seed = _stream_seed(seed, "paysim_sample")
    allocation_rng = random.Random(_stream_seed(seed, "paysim_allocation"))
    cases = sorted(trigger_context, key=lambda row: _stable_rank(seed, row["case_id"]))
    transactions = sample_transactions(paysim_path, len(cases) * HISTORY_EVENTS_PER_KIND, sample_seed)
    bundle_order = list(range(len(cases)))
    allocation_rng.shuffle(bundle_order)
    bundles = [transactions[i * HISTORY_EVENTS_PER_KIND:(i + 1) * HISTORY_EVENTS_PER_KIND]
               for i in range(len(cases))]
    bundle_for_case = {cases[bundle_order[i]]["case_id"]: bundles[i] for i in range(len(cases))}

    history: list[dict[str, Any]] = []
    for case in sorted(trigger_context, key=lambda row: row["case_id"]):
        trigger_time = case["trigger_time"]
        identity = hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()[:12].upper()
        bundle = iter(bundle_for_case[case["case_id"]])
        for index in range(10):
            timestamp = trigger_time - timedelta(days=HISTORY_WINDOW_DAYS * (1 - index / 10))
            is_login = index % 2 == 0
            history.append(make_event(
                event_id=f"EVT-SIMREP2-HIST-{identity}-{index + 1:02d}",
                user_id=case["user_id"], account_id=case["account_id"],
                event_category="AUTHENTICATION" if is_login else "TRANSACTION",
                event_type="LOGIN" if is_login else "TRANSACTION", timestamp=timestamp,
                device_id=case["known_device_id"], scenario_type=None, ato_label=None,
                status="SUCCESS", transaction=None if is_login else next(bundle),
            ))
    return sorted(history, key=lambda row: (row["user_id"], row["timestamp"], row["event_id"]))


def _baseline_audit(features: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {row["user_id"]: row for row in cases}
    fields = ("baseline_event_count", "baseline_event_frequency", "baseline_transaction_count",
              "baseline_transaction_amount_mean", "baseline_transaction_amount_min",
              "baseline_transaction_amount_max", "baseline_successful_login_count")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        case = metadata[row["user_id"]]
        grouped[(case["template_family_id"], int(case["ato_label"]))].append(row)
    audit: list[dict[str, Any]] = []
    for (family_id, label), rows in sorted(grouped.items()):
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] not in (None, "")]
            audit.append({"template_family_id": family_id, "ato_label": label,
                          "feature": field, "count": len(values),
                          "missing_count": len(rows) - len(values),
                          "min": min(values) if values else None,
                          "mean": mean(values) if values else None,
                          "median": median(values) if values else None,
                          "max": max(values) if values else None})
    return audit


def validate_experiment(result: dict[str, Any]) -> None:
    """Validate cohort schema, family balance, cutoff, and cross-family signatures."""
    cases, events = result["cases"], result["events"]
    if len(cases) != TOTAL_CASES:
        raise ValueError(f"expected {TOTAL_CASES} cases, got {len(cases)}")
    keys = ("case_id", "user_id", "account_id", "event_id")
    for key in keys:
        values = [str(row[key]) for row in cases]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {key} in case metadata")
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ids = [str(row["event_id"]) for row in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate event_id")
    if any(tuple(row.keys()) != EVENT_FIELDS for row in events):
        raise ValueError("event rows differ from canonical EVENT_FIELDS")
    for row in events:
        if row["event_type"] not in CATEGORY_EVENT_TYPES.get(row["event_category"], ()):
            raise ValueError(f"invalid event_type {row['event_category']}/{row['event_type']}")
        datetime.fromisoformat(str(row["timestamp"]))
        by_user[str(row["user_id"])].append(row)
    if len(by_user) != TOTAL_CASES:
        raise ValueError("event users do not match case count")
    case_by_user = {row["user_id"]: row for row in cases}
    if set(case_by_user) != set(by_user):
        raise ValueError("event users and case metadata users differ")
    labels_by_family: dict[str, Counter[int]] = defaultdict(Counter)
    signature_family: dict[tuple[tuple[str, str, str], ...], str] = {}
    multiset_family: dict[tuple[tuple[str, str, str], ...], str] = {}
    for case in cases:
        family, label = case["template_family_id"], int(case["ato_label"])
        labels_by_family[family][label] += 1
        rows = by_user[case["user_id"]]
        if any(row["account_id"] != case["account_id"] for row in rows):
            raise ValueError(f"account join mismatch for {case['case_id']}")
        if any(row["scenario_type"] != case["scenario_type"] or row["ato_label"] != label for row in rows):
            raise ValueError(f"label/scenario join mismatch for {case['case_id']}")
        triggers = [r for r in rows if r["event_category"] == "TELECOM"]
        if len(triggers) != 1 or triggers[0]["event_type"] != "SIM_REPLACEMENT":
            raise ValueError(f"invalid SIM_REPLACEMENT trigger in {case['case_id']}")
        trigger_time = datetime.fromisoformat(triggers[0]["timestamp"])
        times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        if times != sorted(times) or any(t < trigger_time or t > trigger_time + timedelta(minutes=15) for t in times):
            raise ValueError(f"non-chronological or out-of-window events in {case['case_id']}")
        if not any(row["event_category"] == "RECOVERY" for row in rows):
            raise ValueError(f"recovery-like overlap missing in {case['case_id']}")
        signature = _event_signature(rows)
        previous = signature_family.get(signature)
        if previous is not None and previous != family:
            raise ValueError(f"exact normalized event signature crosses families: {previous}/{family}")
        signature_family[signature] = family
        multiset_signature = tuple(sorted(signature))
        previous_multiset_family = multiset_family.get(multiset_signature)
        if previous_multiset_family is not None and previous_multiset_family != family:
            raise ValueError(f"near-duplicate event multiset crosses families: {previous_multiset_family}/{family}")
        multiset_family[multiset_signature] = family
    if len(labels_by_family) != FAMILY_COUNT or any(c != Counter({0: 10, 1: 10}) for c in labels_by_family.values()):
        raise ValueError("each family must contain exactly 10 cases per label")

    for profile in result["baseline_profiles"]:
        case = case_by_user.get(profile["user_id"])
        if case is None or profile["account_id"] != case["account_id"] or profile["trigger_event_id"] != case["event_id"]:
            raise ValueError("baseline profile metadata does not join to case/trigger")
        trigger = next(row for row in by_user[case["user_id"]] if row["event_category"] == "TELECOM")
        if profile["trigger_cutoff"] != trigger["timestamp"]:
            raise ValueError("baseline profile cutoff does not match trigger")
    cutoff_errors = validate_baseline_cutoffs(result["baseline_input"], result["baseline_profiles"])
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))
    feature_by_user = {row["user_id"]: row for row in result["features"]}
    if len(feature_by_user) != TOTAL_CASES or set(feature_by_user) != set(case_by_user):
        raise ValueError("feature rows do not join one-to-one with metadata")
    for user_id, row in feature_by_user.items():
        if row["ato_label"] != case_by_user[user_id]["ato_label"]:
            raise ValueError("feature target does not align with case label")
        forbidden = {"template_family_id", "scenario_type", "paysim_isFraud", "paysim_isFlaggedFraud"}
        if forbidden.intersection(row):
            raise ValueError("evaluation metadata or PaySim flags entered Phase 6 feature output")


def generate_experiment(mode: str, *, output_dir: str | Path,
                        paysim_path: str | Path = DEFAULT_PAYSIM_PATH,
                        seed: int = 42, label_seed: int | None = None,
                        manifest_path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Generate one v2 experiment into its own directory and return its artifacts."""
    manifest = _load_manifest(manifest_path)
    cases, event_plans = _case_specs(mode, manifest, seed, label_seed)
    events, trigger_context = _generate_events(event_plans, seed)
    history = _build_history(trigger_context, paysim_path, seed)
    triggers = [row for row in events if row["event_category"] == "TELECOM"]
    baseline_input = history + triggers
    profiles = build_baseline_profiles(baseline_input)
    cutoff_errors = validate_baseline_cutoffs(baseline_input, profiles)
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))

    plan_by_case = {row["case_id"]: row for row in event_plans}
    context_by_case = {row["case_id"]: row for row in trigger_context}
    user_to_case = {row["user_id"]: row["case_id"] for row in trigger_context}
    final_cases: list[dict[str, Any]] = []
    for case in cases:
        plan, context = plan_by_case[case["case_id"]], context_by_case[case["case_id"]]
        trigger_id = context["trigger_event_id"]
        final_case = {"case_id": case["case_id"], "template_family_id": case["template_family_id"],
                      "user_id": context["user_id"], "account_id": context["account_id"],
                      "event_id": trigger_id, "scenario_type": plan["scenario_type"],
                      "ato_label": plan["ato_label"],
                      "declared_scenario_intent": plan["declared_scenario_intent"]}
        final_cases.append(final_case)
        for row in events:
            if row["user_id"] == context["user_id"]:
                row["scenario_type"], row["ato_label"] = final_case["scenario_type"], final_case["ato_label"]
    feature_rows = build_feature_rows(events, profiles)
    audit = _baseline_audit(feature_rows, final_cases)
    result = {"mode": mode, "cases": final_cases, "events": events,
              "historical_event_input": history, "baseline_input": baseline_input,
              "baseline_profiles": profiles, "features": feature_rows,
              "baseline_feature_audit": audit}
    validate_experiment(result)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(manifest_path).open("r", encoding="utf-8") as source:
        manifest_text = source.read()
    (output / "template_manifest.json").write_text(manifest_text, encoding="utf-8")
    _write_csv(output / "case_metadata.csv", final_cases,
               ("case_id", "template_family_id", "user_id", "account_id", "event_id", "scenario_type", "ato_label", "declared_scenario_intent"))
    _write_csv(output / "events.csv", events, EVENT_FIELDS)
    _write_csv(output / "historical_event_input.csv", history, EVENT_FIELDS)
    _write_csv(output / "baseline_profiles.csv", profiles, BASELINE_FIELDS)
    _write_csv(output / "features.csv", feature_rows, OUTPUT_COLUMNS)
    audit_fields = ("template_family_id", "ato_label", "feature", "count", "missing_count", "min", "mean", "median", "max")
    _write_csv(output / "baseline_feature_audit.csv", audit, audit_fields)
    metadata = {"mode": mode, "seed": seed, "label_seed": seed if label_seed is None else label_seed,
                "event_seed": _stream_seed(seed, "events"),
                "paysim_sample_seed": _stream_seed(seed, "paysim_sample"),
                "paysim_allocation_seed": _stream_seed(seed, "paysim_allocation"),
                "cases": len(final_cases), "events": len(events), "history_events": len(history),
                "profiles": len(profiles), "feature_rows": len(feature_rows),
                "families": FAMILY_COUNT, "cases_per_label_per_family": CASES_PER_LABEL_PER_FAMILY,
                "inclusive_prediction_window_minutes": OBSERVATION_MINUTES,
                "feature_columns": list(FEATURE_COLUMNS),
                "pay_sim_fraud_fields_used_as_target_or_feature": False,
                "evaluation_metadata_is_model_input": False,
                "interpretation": "Synthetic controlled experiment; cohort size and family count do not establish generalization."}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result
