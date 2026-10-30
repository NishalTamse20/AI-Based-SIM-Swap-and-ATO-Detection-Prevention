import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pytest

from ml.features.feature_generator import FEATURE_COLUMNS
from simulator.generators.event_generator import EVENT_FIELDS
from simulator.generators.sim_replacement_v2 import (
    CASES_PER_LABEL_PER_FAMILY,
    FAMILY_COUNT,
    OBSERVATION_MINUTES,
    _event_signature,
    _load_manifest,
    generate_experiment,
    validate_experiment,
)
from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.validation.dataset_validator import CATEGORY_EVENT_TYPES


@pytest.fixture
def paysim_csv(tmp_path):
    path = tmp_path / "paysim.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        for i in range(2200):
            writer.writerow({
                "step": str(i), "type": "CASH_OUT" if i % 2 else "TRANSFER",
                "amount": str(10 + i), "nameOrig": f"C{i}",
                "oldbalanceOrg": str(10000 + i), "newbalanceOrig": str(9000 + i),
                "nameDest": f"D{i}", "oldbalanceDest": str(2000 + i),
                "newbalanceDest": str(2010 + i), "isFraud": str(i % 2),
                "isFlaggedFraud": str((i // 2) % 2),
            })
    return path


def _rows(result, user_id):
    return [row for row in result["events"] if row["user_id"] == user_id]


def _history_digest(result):
    return hashlib.sha256(json.dumps(result["historical_event_input"], sort_keys=True).encode()).hexdigest()


def _feature_inputs(result):
    return [{key: row[key] for key in ("user_id", "account_id", "event_id", *FEATURE_COLUMNS)}
            for row in result["features"]]


def test_manifest_families_are_explicit_and_cross_family_signatures_are_unique():
    manifest = _load_manifest()
    assert len(manifest["families"]) == FAMILY_COUNT
    signatures = {}
    normalized_bags = {}
    event_defs = manifest["event_definitions"]
    for family in manifest["families"]:
        assert len(family["legitimate"]) == len(family["suspicious"]) == 4
        for intent in ("legitimate", "suspicious"):
            names = family[intent]
            assert any(event_defs[name]["event_category"] == "RECOVERY" for name in names)
            assert any(event_defs[name]["event_category"] == "DEVICE" for name in names)
            assert any(event_defs[name]["event_category"] == "AUTHENTICATION" and
                       event_defs[name]["status"] == "FAILURE" for name in names)
            signature = tuple((event_defs[name]["event_category"], name,
                               event_defs[name]["status"]) for name in names)
            assert signature not in signatures or signatures[signature] == family["template_family_id"]
            signatures[signature] = family["template_family_id"]
            bag = tuple(sorted(names))
            assert bag not in normalized_bags or normalized_bags[bag] == family["template_family_id"]
            normalized_bags[bag] = family["template_family_id"]
    assert "reviewed" in manifest["near_duplicate_review"].lower()


def test_both_experiments_meet_composition_schema_cutoff_and_join_requirements(tmp_path, paysim_csv):
    for mode in ("negative_control", "predictive"):
        result = generate_experiment(mode, output_dir=tmp_path / mode, paysim_path=paysim_csv, seed=42)
        validate_experiment(result)
        assert len(result["cases"]) == 400
        family_labels = defaultdict(Counter)
        for case in result["cases"]:
            family_labels[case["template_family_id"]][case["ato_label"]] += 1
        assert len(family_labels) == 20
        assert all(count == Counter({0: 10, 1: 10}) for count in family_labels.values())
        assert len(result["features"]) == 400
        assert len(result["baseline_profiles"]) == 400

        case_by_user = {row["user_id"]: row for row in result["cases"]}
        profiles = {row["user_id"]: row for row in result["baseline_profiles"]}
        history_by_user = defaultdict(list)
        for row in result["historical_event_input"]:
            history_by_user[row["user_id"]].append(row)
            assert row["scenario_type"] is None and row["ato_label"] is None
        assert set(history_by_user) == set(case_by_user) == set(profiles)
        for user_id, rows in history_by_user.items():
            case = case_by_user[user_id]
            trigger = next(row for row in _rows(result, user_id) if row["event_category"] == "TELECOM")
            cutoff = datetime.fromisoformat(trigger["timestamp"])
            assert all(datetime.fromisoformat(row["timestamp"]) < cutoff for row in rows)
            assert all(row["device_id"] == trigger["device_id"] for row in rows)
            assert sum(row["event_type"] == "LOGIN" and row["status"] == "SUCCESS" for row in rows) == 5
            assert sum(row["event_category"] == "TRANSACTION" for row in rows) == 5
            assert profiles[user_id]["account_id"] == case["account_id"]
            assert profiles[user_id]["trigger_event_id"] == case["event_id"]
            assert profiles[user_id]["trigger_cutoff"] == trigger["timestamp"]
        for user_id, rows in ((u, _rows(result, u)) for u in case_by_user):
            trigger = next(row for row in rows if row["event_category"] == "TELECOM")
            trigger_time = datetime.fromisoformat(trigger["timestamp"])
            assert rows == sorted(rows, key=lambda row: datetime.fromisoformat(row["timestamp"]))
            assert all(trigger_time <= datetime.fromisoformat(row["timestamp"]) <=
                       trigger_time + timedelta(minutes=OBSERVATION_MINUTES) for row in rows)
        assert all(tuple(row.keys()) == EVENT_FIELDS for row in result["events"])
        assert not ({"scenario_type", "template_family_id", "paysim_isFraud", "paysim_isFlaggedFraud"}
                    & set(result["features"][0]))
        assert set(result["features"][0]) == {
            "user_id", "account_id", "event_id", *FEATURE_COLUMNS, "ato_label"
        }
        for name in ("template_manifest.json", "case_metadata.csv", "events.csv",
                     "historical_event_input.csv", "baseline_profiles.csv", "features.csv",
                     "baseline_feature_audit.csv", "run_metadata.json"):
            assert (tmp_path / mode / name).is_file()


def test_negative_control_labels_are_independent_of_sequences_and_history(tmp_path, paysim_csv):
    first = generate_experiment("negative_control", output_dir=tmp_path / "nc1",
                                paysim_path=paysim_csv, seed=31, label_seed=5)
    second = generate_experiment("negative_control", output_dir=tmp_path / "nc2",
                                 paysim_path=paysim_csv, seed=31, label_seed=29)
    assert _history_digest(first) == _history_digest(second)
    assert first["baseline_profiles"] == second["baseline_profiles"]
    rows1 = [{k: v for k, v in row.items() if k not in {"ato_label", "scenario_type"}}
             for row in first["events"]]
    rows2 = [{k: v for k, v in row.items() if k not in {"ato_label", "scenario_type"}}
             for row in second["events"]]
    assert rows1 == rows2
    assert _feature_inputs(first) == _feature_inputs(second)
    assert any(a["ato_label"] != b["ato_label"]
               for a, b in zip(first["cases"], second["cases"]))


def test_predictive_labels_change_intent_patterns_but_not_history(tmp_path, paysim_csv):
    first = generate_experiment("predictive", output_dir=tmp_path / "p1",
                                paysim_path=paysim_csv, seed=17, label_seed=3)
    second = generate_experiment("predictive", output_dir=tmp_path / "p2",
                                 paysim_path=paysim_csv, seed=17, label_seed=23)
    assert _history_digest(first) == _history_digest(second)
    assert first["baseline_profiles"] == second["baseline_profiles"]
    assert all(case["ato_label"] == (case["declared_scenario_intent"] == "suspicious")
               for case in first["cases"])
    signatures1 = {case["case_id"]: _event_signature(_rows(first, case["user_id"]))
                   for case in first["cases"]}
    signatures2 = {case["case_id"]: _event_signature(_rows(second, case["user_id"]))
                   for case in second["cases"]}
    assert any(signatures1[key] != signatures2[key] for key in signatures1)
    # Every event type is represented under both declared intents in the recipes.
    manifest = _load_manifest()
    names = {name for family in manifest["families"] for intent in ("legitimate", "suspicious")
             for name in family[intent]}
    assert names
    for name in names:
        assert any(name in family["legitimate"] for family in manifest["families"])
        assert any(name in family["suspicious"] for family in manifest["families"])


def test_deterministic_rerun_and_raw_paysim_immutability(tmp_path, paysim_csv):
    before = hashlib.sha256(paysim_csv.read_bytes()).hexdigest()
    first = generate_experiment("predictive", output_dir=tmp_path / "run1",
                                paysim_path=paysim_csv, seed=42)
    second = generate_experiment("predictive", output_dir=tmp_path / "run2",
                                 paysim_path=paysim_csv, seed=42)
    assert first["events"] == second["events"]
    assert first["historical_event_input"] == second["historical_event_input"]
    assert first["baseline_profiles"] == second["baseline_profiles"]
    assert first["features"] == second["features"]
    output_names = {path.name for path in (tmp_path / "run1").iterdir()}
    assert output_names == {path.name for path in (tmp_path / "run2").iterdir()}
    assert all((tmp_path / "run1" / name).read_bytes() == (tmp_path / "run2" / name).read_bytes()
               for name in output_names)
    assert hashlib.sha256(paysim_csv.read_bytes()).hexdigest() == before


def test_validation_rejects_cross_family_exact_duplicate_signature(tmp_path, paysim_csv):
    result = generate_experiment("predictive", output_dir=tmp_path / "predictive",
                                 paysim_path=paysim_csv, seed=42)
    by_user = defaultdict(list)
    for row in result["events"]:
        by_user[row["user_id"]].append(row)
    cases = result["cases"]
    family_a = cases[0]
    family_b = next(case for case in cases if case["template_family_id"] != family_a["template_family_id"])
    signature_a = _event_signature(by_user[family_a["user_id"]])
    signature_b = _event_signature(by_user[family_b["user_id"]])
    # Construct a malformed cohort by replacing one case's post-trigger recipe.
    source_rows = by_user[family_a["user_id"]]
    target_rows = by_user[family_b["user_id"]]
    source_post = [row for row in source_rows if row["event_category"] != "TELECOM"]
    target_post = [row for row in target_rows if row["event_category"] != "TELECOM"]
    for target, source in zip(target_post, source_post):
        target["event_category"] = source["event_category"]
        target["event_type"] = source["event_type"]
        target["status"] = source["status"]
    signature_b = _event_signature(target_rows)
    assert signature_a == signature_b
    with pytest.raises(ValueError, match="signature crosses families"):
        validate_experiment(result)


def test_manifest_and_generated_events_use_canonical_event_types(tmp_path, paysim_csv):
    manifest = _load_manifest()
    for event_type, definition in manifest["event_definitions"].items():
        assert event_type in CATEGORY_EVENT_TYPES[definition["event_category"]]
    result = generate_experiment("predictive", output_dir=tmp_path / "predictive",
                                 paysim_path=paysim_csv, seed=42)
    account_types = {row["event_type"] for row in result["events"] if row["event_category"] == "ACCOUNT"}
    assert account_types == {"NEW_BENEFICIARY", "PROFILE_CHANGE"}
    assert any(row["new_beneficiary"] == 1 for row in result["features"])


@pytest.mark.parametrize("event_type", ["BENEFICIARY_ADDED", "CONTACT_CHANGED"])
def test_validation_rejects_non_canonical_event_type(tmp_path, paysim_csv, event_type):
    result = generate_experiment("predictive", output_dir=tmp_path / "predictive",
                                 paysim_path=paysim_csv, seed=42)
    account_row = next(row for row in result["events"] if row["event_category"] == "ACCOUNT")
    account_row["event_type"] = event_type
    with pytest.raises(ValueError, match=f"invalid event_type ACCOUNT/{event_type}"):
        validate_experiment(result)
