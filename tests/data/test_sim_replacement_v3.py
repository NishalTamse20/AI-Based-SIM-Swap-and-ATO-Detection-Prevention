import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pytest

from ml.features.feature_generator import FEATURE_COLUMNS, OUTPUT_COLUMNS
from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.generators.sim_replacement_v3 import (
    CASES_PER_LABEL_PER_FAMILY,
    EVALUATION_ONLY_COLUMNS,
    FAMILY_COUNT,
    WINDOW_DISCRETE_FEATURES,
    _load_manifest,
    generate_experiment,
    validate_experiment,
    validate_manifest,
)
from simulator.validation.dataset_validator import CATEGORY_EVENT_TYPES


EXPECTED_FILES = {"template_manifest.json", "case_metadata.csv", "events.csv",
                  "historical_event_input.csv", "baseline_profiles.csv", "features.csv",
                  "baseline_feature_audit.csv", "run_metadata.json", "design_audit.json"}
LABEL_COLUMNS = {"ato_label", "scenario_type"}
NEUTRAL_SLOTS = ("recovery", "transaction", "account")


@pytest.fixture(scope="module")
def paysim_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("paysim") / "paysim.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        for i in range(5000):
            writer.writerow({
                "step": str(i), "type": ("CASH_OUT", "TRANSFER", "PAYMENT")[i % 3],
                "amount": str(10 + i), "nameOrig": f"C{i}",
                "oldbalanceOrg": str(10000 + i), "newbalanceOrig": str(9000 + i),
                "nameDest": f"D{i}", "oldbalanceDest": str(2000 + i),
                "newbalanceDest": str(2010 + i), "isFraud": str(i % 2),
                "isFlaggedFraud": str((i // 2) % 2),
            })
    return path


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory, paysim_csv):
    root = tmp_path_factory.mktemp("v3")
    return {mode: generate_experiment(mode, output_dir=root / mode, paysim_path=paysim_csv, seed=42)
            for mode in ("predictive", "negative_control")} | {"root": root}


def _rows_by_user(result):
    by_user = defaultdict(list)
    for row in result["events"]:
        by_user[row["user_id"]].append(row)
    return by_user


def _without_labels(rows):
    return [{k: v for k, v in row.items() if k not in LABEL_COLUMNS} for row in rows]


def _feature_inputs(result):
    return [{key: row[key] for key in ("user_id", "account_id", "event_id", *FEATURE_COLUMNS)}
            for row in result["features"]]


def _components(case):
    return json.loads(case["components_json"])


def _cells(result):
    cells = defaultdict(list)
    for case in result["cases"]:
        cells[(case["template_family_id"], case["ato_label"])].append(case)
    return cells


def test_manifest_uses_canonical_events_bounded_versions_and_consistent_quotas():
    manifest = _load_manifest()
    for name, category in manifest["event_categories"].items():
        assert name in CATEGORY_EVENT_TYPES[category]
    for profile in manifest["design_assumptions"]["version_probabilities"].values():
        assert all(0.1 <= p <= 0.9 for p in profile.values())
    assert "not empirical" in manifest["design_assumptions"]["status"]
    assert len({f["template_family_id"] for f in manifest["families"]}) == FAMILY_COUNT


def test_manifest_rejects_identical_family_definitions():
    manifest = copy.deepcopy(_load_manifest())
    manifest["families"][1] = dict(manifest["families"][0], template_family_id="F02")
    with pytest.raises(ValueError, match="identical family definitions"):
        validate_manifest(manifest)


def test_manifest_rejects_quotas_that_remove_presence_or_absence():
    manifest = copy.deepcopy(_load_manifest())
    manifest["design_assumptions"]["skeleton_quotas_per_family"]["recovery"] = 7
    with pytest.raises(ValueError, match="both present and absent"):
        validate_manifest(manifest)


def test_both_cohorts_write_expected_files_and_balanced_families(cohorts):
    for mode in ("predictive", "negative_control"):
        assert {p.name for p in (cohorts["root"] / mode).iterdir()} == EXPECTED_FILES
        counts = Counter((c["template_family_id"], c["ato_label"]) for c in cohorts[mode]["cases"])
        assert len(counts) == 2 * FAMILY_COUNT and set(counts.values()) == {CASES_PER_LABEL_PER_FAMILY}
        with (cohorts["root"] / mode / "features.csv").open(encoding="utf-8") as source:
            assert tuple(csv.DictReader(source).fieldnames) == OUTPUT_COLUMNS
    assert not EVALUATION_ONLY_COLUMNS.intersection(FEATURE_COLUMNS)


def test_predictive_labels_follow_intent_and_anchors_are_shared(cohorts):
    result = cohorts["predictive"]
    by_user = _rows_by_user(result)
    anchors = defaultdict(lambda: defaultdict(set))
    for case in result["cases"]:
        assert case["ato_label"] == int(case["declared_scenario_intent"] == "suspicious")
        if case["is_anchor"]:
            recipe = tuple(sorted(r["event_type"] for r in by_user[case["user_id"]] if r["event_category"] != "TELECOM"))
            anchors[case["template_family_id"]][case["ato_label"]].add(recipe)
    assert len(anchors) == FAMILY_COUNT
    for by_label in anchors.values():
        assert by_label[0] == by_label[1] and len(by_label[0]) == 1


def test_predictive_skeletons_event_counts_and_neutral_presence_match_across_labels(cohorts):
    result = cohorts["predictive"]
    by_user = _rows_by_user(result)
    cells = _cells(result)
    for family in {c["template_family_id"] for c in result["cases"]}:
        signatures = []
        for label in (0, 1):
            members = cells[(family, label)]
            comps = [_components(c) for c in members]
            signatures.append({
                "skeletons": sorted(json.dumps(c["skeleton"], sort_keys=True) for c in comps),
                "event_counts": sorted(len(by_user[c["user_id"]]) - 1 for c in members),
                "presence": {slot: sum(bool(c["skeleton"][slot]) for c in comps) for slot in NEUTRAL_SLOTS},
                "auth_slot": sum(c["skeleton"]["auth_k"] > 0 for c in comps),
            })
        assert signatures[0] == signatures[1]
        assert signatures[0]["presence"] == {"recovery": 7, "transaction": 4, "account": 3}
        assert signatures[0]["auth_slot"] == 4


def test_event_count_carries_no_label_signal_in_either_cohort(cohorts):
    predictive = cohorts["predictive"]["design_audit"]["component_diagnostics"]["window_event_count"]
    assert predictive["label_0"]["counts"] == predictive["label_1"]["counts"]
    assert predictive["single_feature_lofo_lookup"]["covered"] == 0
    negative = cohorts["negative_control"]["design_audit"]["component_diagnostics"]["window_event_count"]
    assert negative["label_0"]["n"] == negative["label_1"]["n"]


def test_every_predictive_cell_has_neutral_slot_presence_and_absence(cohorts):
    assert cohorts["predictive"]["design_audit"]["family_label_cells_lacking_variation"] == []


def test_predictive_families_overlap_and_none_is_perfectly_separable(cohorts):
    audit = cohorts["predictive"]["design_audit"]
    assert audit["overlap"]["families_perfectly_separable"] == 0
    assert all(f["ambiguous_share"] > 0 for f in audit["overlap"]["per_family"].values())
    assert audit["lookup_feature_columns"] == list(WINDOW_DISCRETE_FEATURES)


def test_event_order_follows_slot_rules_and_varies(cohorts):
    for mode in ("predictive", "negative_control"):
        result = cohorts[mode]
        by_user = _rows_by_user(result)
        orders = set()
        for case in result["cases"]:
            order = _components(case)["slot_order"]
            orders.add(tuple(order))
            assert order[0] == "device"
            first_auth = min(i for i, t in enumerate(order) if t in {"auth", "login"})
            assert all(i > first_auth for i, t in enumerate(order) if t in {"account", "transaction"})
            rows = by_user[case["user_id"]]
            trigger = next(r for r in rows if r["event_category"] == "TELECOM")
            window = sorted((r for r in rows if r["event_category"] != "TELECOM"), key=lambda r: r["timestamp"])
            assert len(window) == len(order)
            assert window[0]["event_category"] in {"DEVICE", "AUTHENTICATION"}
            start = datetime.fromisoformat(trigger["timestamp"])
            assert all(start < datetime.fromisoformat(r["timestamp"]) <= start + timedelta(minutes=15)
                       for r in window)
        transaction_positions = {o.index("transaction") for o in orders if "transaction" in o}
        assert len(transaction_positions) > 1


@pytest.mark.parametrize("offset_seconds, accepted", [(900, True), (901, False)])
def test_window_boundary_is_inclusive_at_t_plus_15(cohorts, offset_seconds, accepted):
    result = copy.deepcopy(cohorts["predictive"])
    case = result["cases"][0]
    rows = [r for r in result["events"] if r["user_id"] == case["user_id"]]
    trigger = next(r for r in rows if r["event_category"] == "TELECOM")
    last = max((r for r in rows if r["event_category"] != "TELECOM"), key=lambda r: r["timestamp"])
    last["timestamp"] = (datetime.fromisoformat(trigger["timestamp"])
                         + timedelta(seconds=offset_seconds)).isoformat(timespec="seconds")
    if accepted:
        validate_experiment(result)
    else:
        with pytest.raises(ValueError, match="out-of-window"):
            validate_experiment(result)


def test_negative_control_artifacts_are_invariant_to_label_seed(tmp_path, paysim_csv):
    first = generate_experiment("negative_control", output_dir=tmp_path / "nc1",
                                paysim_path=paysim_csv, seed=31, label_seed=5)
    second = generate_experiment("negative_control", output_dir=tmp_path / "nc2",
                                 paysim_path=paysim_csv, seed=31, label_seed=29)
    assert _without_labels(first["events"]) == _without_labels(second["events"])
    assert first["historical_event_input"] == second["historical_event_input"]
    assert first["baseline_profiles"] == second["baseline_profiles"]
    assert _feature_inputs(first) == _feature_inputs(second)
    recipe = ("case_id", "recipe_source", "is_anchor", "components_json", "declared_scenario_intent")
    assert [{k: c[k] for k in recipe} for c in first["cases"]] == \
           [{k: c[k] for k in recipe} for c in second["cases"]]
    for name in ("historical_event_input.csv", "baseline_profiles.csv"):
        assert (tmp_path / "nc1" / name).read_bytes() == (tmp_path / "nc2" / name).read_bytes()
    assert any(a["ato_label"] != b["ato_label"] for a, b in zip(first["cases"], second["cases"]))


def test_negative_control_uses_each_skeleton_once_per_profile(cohorts):
    by_family = defaultdict(Counter)
    for case in cohorts["negative_control"]["cases"]:
        comps = _components(case)
        if not case["is_anchor"]:
            by_family[case["template_family_id"]][(comps["skeleton_index"], case["recipe_source"])] += 1
    for counts in by_family.values():
        assert set(counts) == {(i, s) for i in range(7) for s in ("profile_A", "profile_B")}
        assert set(counts.values()) == {1}


def test_seed_changes_events_but_not_label_schedule(tmp_path, paysim_csv):
    first = generate_experiment("negative_control", output_dir=tmp_path / "a",
                                paysim_path=paysim_csv, seed=31, label_seed=5)
    second = generate_experiment("negative_control", output_dir=tmp_path / "b",
                                 paysim_path=paysim_csv, seed=32, label_seed=5)
    assert _without_labels(first["events"]) != _without_labels(second["events"])
    assert [c["ato_label"] for c in first["cases"]] == [c["ato_label"] for c in second["cases"]]


def test_transaction_amount_is_independent_of_label_and_type(tmp_path, paysim_csv):
    first = generate_experiment("predictive", output_dir=tmp_path / "p1",
                                paysim_path=paysim_csv, seed=17, label_seed=3)
    second = generate_experiment("predictive", output_dir=tmp_path / "p2",
                                 paysim_path=paysim_csv, seed=17, label_seed=23)
    amounts = []
    for result in (first, second):
        by_user = _rows_by_user(result)
        amounts.append({c["case_id"]: r["amount"] for c in result["cases"]
                        for r in by_user[c["user_id"]] if r["event_category"] == "TRANSACTION"})
    shared = set(amounts[0]) & set(amounts[1])
    assert shared
    assert all(amounts[0][k] == amounts[1][k] for k in shared)


def test_missing_transaction_policy(cohorts):
    for mode in ("predictive", "negative_control"):
        result = cohorts[mode]
        for row in result["features"]:
            if row["transaction_count"] == 0:
                assert row["transaction_amount"] is None and row["transaction_deviation"] is None
                assert row["transaction_type"] in (None, "")
            else:
                assert row["transaction_amount"] is not None and row["transaction_type"]
        summary = result["design_audit"]["feature_summary"]
        assert summary["all_missing"] == [] and "recovery_event_present" not in summary["constant"]
        assert result["design_audit"]["interpretation"].startswith("Synthetic")


def test_validation_rejects_non_canonical_event_type(cohorts):
    result = copy.deepcopy(cohorts["predictive"])
    row = next(r for r in result["events"] if r["event_category"] == "ACCOUNT")
    row["event_type"] = "BENEFICIARY_ADDED"
    with pytest.raises(ValueError, match="invalid event_type ACCOUNT/BENEFICIARY_ADDED"):
        validate_experiment(result)


def test_validation_rejects_imputed_missing_transaction_amount(cohorts):
    result = copy.deepcopy(cohorts["predictive"])
    row = next(r for r in result["features"] if r["transaction_count"] == 0)
    row["transaction_amount"] = 0
    with pytest.raises(ValueError, match="must stay missing"):
        validate_experiment(result)


def test_validation_rejects_label_dependent_skeletons(cohorts):
    result = copy.deepcopy(cohorts["predictive"])
    case = next(c for c in result["cases"] if not c["is_anchor"] and c["ato_label"] == 1)
    comps = _components(case)
    comps["skeleton"]["recovery"] = not comps["skeleton"]["recovery"]
    tokens = ["device"] + ["auth"] * comps["skeleton"]["auth_k"] + ["recovery"] * comps["skeleton"]["recovery"] \
        + ["login"] + ["account"] * comps["skeleton"]["account"] + ["transaction"] * comps["skeleton"]["transaction"]
    comps["slot_order"] = tokens
    case["components_json"] = json.dumps(comps, sort_keys=True)
    with pytest.raises(ValueError, match="slot order|differ by label"):
        validate_experiment(result)


def test_deterministic_rerun_and_raw_paysim_immutability(tmp_path, paysim_csv):
    before = hashlib.sha256(paysim_csv.read_bytes()).hexdigest()
    first = generate_experiment("predictive", output_dir=tmp_path / "r1", paysim_path=paysim_csv, seed=9)
    second = generate_experiment("predictive", output_dir=tmp_path / "r2", paysim_path=paysim_csv, seed=9)
    for name in EXPECTED_FILES:
        assert (tmp_path / "r1" / name).read_bytes() == (tmp_path / "r2" / name).read_bytes()
    assert hashlib.sha256(paysim_csv.read_bytes()).hexdigest() == before
    assert json.loads((tmp_path / "r1" / "run_metadata.json").read_text())["version"] == "v3"
    assert first["cases"] == second["cases"]
