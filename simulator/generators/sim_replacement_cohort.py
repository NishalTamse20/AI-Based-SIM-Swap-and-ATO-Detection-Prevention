"""Generate an isolated, manifest-driven SIM_REPLACEMENT cohort.

The family/case manifest is evaluation metadata and is returned separately
from event rows so it cannot silently enter the Phase 6/9 feature matrices.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from simulator.generators.event_generator import EVENT_FIELDS, make_event


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "sim_replacement_template_manifest.json"
OBSERVATION_MINUTES = 15
EVENT_TYPES_BY_CATEGORY = {
    "TELECOM": {"SIM_SWAP", "SIM_REPLACEMENT", "ESIM_CHANGE"},
    "DEVICE": {"NEW_DEVICE", "DEVICE_CHANGE", "DEVICE_FINGERPRINT_CHANGE"},
    "AUTHENTICATION": {"LOGIN", "FAILED_LOGIN", "MFA_FAILURE", "MFA_SUCCESS", "PASSWORD_CHANGE"},
    "RECOVERY": {"PASSWORD_RESET", "ACCOUNT_RECOVERY", "RECOVERY_CONTACT_CHANGE"},
}


def load_template_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load the declarative family definitions without modifying the manifest."""
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _case_signature(case_events: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    trigger = next(row for row in case_events if row["event_category"] == "TELECOM")
    trigger_time = datetime.fromisoformat(trigger["timestamp"])
    trigger_device = trigger["device_id"]
    return tuple(
        (
            row["event_category"],
            row["event_type"],
            (datetime.fromisoformat(row["timestamp"]) - trigger_time).total_seconds(),
            "trigger_device" if row["device_id"] == trigger_device else "other_device",
            row["status"],
        )
        for row in sorted(case_events, key=lambda item: datetime.fromisoformat(item["timestamp"]))
    )


def validate_cohort(cohort: dict[str, Any]) -> None:
    """Raise ValueError for invalid labels, identity, schema timing, or leakage signatures."""
    cases = cohort.get("cases")
    events = cohort.get("events")
    if not isinstance(cases, list) or not isinstance(events, list) or not cases:
        raise ValueError("cohort must contain non-empty cases and events lists")

    case_ids = [case.get("case_id") for case in cases]
    user_ids = [case.get("user_id") for case in cases]
    account_ids = [case.get("account_id") for case in cases]
    if any(not value for value in case_ids + user_ids + account_ids):
        raise ValueError("case_id, user_id, and account_id are required")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case identifiers must be unique")
    if len(set(user_ids)) != len(user_ids) or len(set(account_ids)) != len(account_ids):
        raise ValueError("user and account identifiers must be unique per case")

    labels_by_family: dict[str, set[int]] = defaultdict(set)
    for case in cases:
        family_id = str(case.get("template_family_id", ""))
        if not family_id:
            raise ValueError(f"template_family_id missing for case {case['case_id']}")
        labels_by_family[family_id].add(int(case["ato_label"]))
    if any(labels != {0, 1} for labels in labels_by_family.values()):
        raise ValueError("every template family must contain both ato_label values 0 and 1")

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_ids: list[str] = []
    for event in events:
        if tuple(event) != EVENT_FIELDS:
            raise ValueError("event row does not match the existing event schema")
        required = {"event_id", "user_id", "account_id", "event_category", "event_type", "timestamp", "device_id", "scenario_type", "ato_label"}
        if not required.issubset(event):
            raise ValueError(f"event missing required schema fields: {sorted(required - set(event))}")
        event_ids.append(str(event["event_id"]))
        by_user[str(event["user_id"])].append(event)
        try:
            datetime.fromisoformat(str(event["timestamp"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamp for event {event['event_id']}") from exc
        supported = EVENT_TYPES_BY_CATEGORY.get(str(event["event_category"]))
        if supported is None or event["event_type"] not in supported:
            raise ValueError(f"unsupported event category/type: {event['event_category']}/{event['event_type']}")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event identifiers must be unique")
    if set(by_user) != set(user_ids):
        raise ValueError("event users must match case users exactly")

    case_by_user = {str(case["user_id"]): case for case in cases}
    family_signatures: dict[tuple[tuple[Any, ...], ...], str] = {}
    for user_id, case_events in by_user.items():
        case = case_by_user[user_id]
        if any(row["account_id"] != case["account_id"] for row in case_events):
            raise ValueError(f"account identifier mismatch for case {case['case_id']}")
        if any(row["scenario_type"] != case["scenario_type"] or row["ato_label"] != case["ato_label"] for row in case_events):
            raise ValueError(f"event label/scenario metadata mismatch for case {case['case_id']}")
        triggers = [row for row in case_events if row["event_category"] == "TELECOM"]
        if len(triggers) != 1 or triggers[0]["event_type"] != "SIM_REPLACEMENT":
            raise ValueError(f"case {case['case_id']} must have exactly one SIM_REPLACEMENT trigger")
        trigger_time = datetime.fromisoformat(triggers[0]["timestamp"])
        if case["scenario_type"] not in {"LEGITIMATE_SIM_REPLACEMENT", "SUSPICIOUS_SIM_REPLACEMENT"}:
            raise ValueError(f"unsupported SIM_REPLACEMENT scenario_type in {case['case_id']}")
        expected_label = 0 if case["scenario_type"] == "LEGITIMATE_SIM_REPLACEMENT" else 1
        if case["ato_label"] != expected_label:
            raise ValueError(f"label does not match declared scenario intent for {case['case_id']}")
        ordered = sorted(case_events, key=lambda row: datetime.fromisoformat(row["timestamp"]))
        if ordered != case_events:
            raise ValueError(f"events are not chronological for case {case['case_id']}")
        cutoff = trigger_time + timedelta(minutes=OBSERVATION_MINUTES)
        if any(datetime.fromisoformat(row["timestamp"]) < trigger_time for row in case_events):
            raise ValueError(f"pre-trigger event found in controlled cohort case {case['case_id']}")
        if any(datetime.fromisoformat(row["timestamp"]) > cutoff for row in case_events):
            raise ValueError(f"event exceeds inclusive T+15 window for case {case['case_id']}")
        types = {(row["event_category"], row["event_type"]) for row in case_events}
        if not {
            ("DEVICE", "NEW_DEVICE"),
            ("AUTHENTICATION", "FAILED_LOGIN"),
            ("RECOVERY", "PASSWORD_RESET"),
        }.issubset(types):
            raise ValueError(f"recovery-like overlap missing for case {case['case_id']}")

        family_id = str(case.get("template_family_id", ""))
        if not family_id:
            raise ValueError(f"template_family_id missing for case {case['case_id']}")
        signature = _case_signature(case_events)
        previous_family = family_signatures.get(signature)
        if previous_family is not None and previous_family != family_id:
            raise ValueError(
                f"exact normalized event signature appears in multiple families: {previous_family}, {family_id}"
            )
        family_signatures[signature] = family_id

def generate_sim_replacement_cohort(
    *,
    start_time: datetime = datetime(2026, 1, 1),
    manifest_path: str | Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Generate event rows plus a separate case/family metadata manifest.

    Ground-truth labels are copied from the manifest's declared scenario
    intent. Event patterns are shared within each family across both labels.
    """
    manifest = load_template_manifest(manifest_path)
    events: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    case_index = 0
    for family in manifest["families"]:
        family_id = str(family["template_family_id"])
        for intent in family["cases"]:
            case_index += 1
            case_id = str(intent["case_id"])
            scenario_type = str(intent["scenario_type"])
            label = int(intent["ato_label"])
            user_id = f"USR-SIMREP-{case_index:04d}"
            account_id = f"ACC-SIMREP-{case_index:04d}"
            known_device = f"DEV-SIMREP-{case_index:04d}-01"
            new_device = f"DEV-SIMREP-{case_index:04d}-02"
            trigger_time = start_time + timedelta(days=case_index - 1)
            case_events = [make_event(
                event_id=f"EVT-SIMREP-{case_index:04d}-00",
                user_id=user_id,
                account_id=account_id,
                event_category="TELECOM",
                event_type="SIM_REPLACEMENT",
                timestamp=trigger_time,
                device_id=known_device,
                scenario_type=scenario_type,
                ato_label=label,
                status="SUCCESS",
            )]
            for sequence_index, event_spec in enumerate(family["post_trigger_events"], start=1):
                device_id = known_device if event_spec["device_role"] == "known" else new_device
                case_events.append(make_event(
                    event_id=f"EVT-SIMREP-{case_index:04d}-{sequence_index:02d}",
                    user_id=user_id,
                    account_id=account_id,
                    event_category=str(event_spec["event_category"]),
                    event_type=str(event_spec["event_type"]),
                    timestamp=trigger_time + timedelta(minutes=int(event_spec["offset_minutes"])),
                    device_id=device_id,
                    scenario_type=scenario_type,
                    ato_label=label,
                    status=str(event_spec["status"]),
                ))
            events.extend(case_events)
            cases.append({
                "case_id": case_id,
                "template_family_id": family_id,
                "user_id": user_id,
                "account_id": account_id,
                "event_id": case_events[0]["event_id"],
                "scenario_type": scenario_type,
                "ato_label": label,
            })

    cohort = {"manifest_version": manifest["manifest_version"], "cases": cases, "events": events}
    validate_cohort(cohort)
    return cohort

