import copy
import csv
import itertools
import json
from collections import defaultdict
from fractions import Fraction

import pytest

from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.generators.sim_replacement_v3 import (
    INTERACTION_SLOTS,
    MANIFEST_PATH,
    _latent_draw,
    _load_manifest,
    _versions_from_latent,
    generate_experiment,
    validate_manifest,
)
from simulator.generators.sim_replacement_v3_ceiling import coordinated_distribution, independent_distribution


INTERACTION_MANIFEST = MANIFEST_PATH.with_name("sim_replacement_v3_interaction_manifest.json")
LABEL_COLUMNS = {"ato_label", "scenario_type"}


@pytest.fixture(scope="module")
def paysim_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("paysim") / "paysim.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        for i in range(5000):
            writer.writerow({
                "step": str(i), "type": ("CASH_OUT", "TRANSFER", "PAYMENT")[i % 3],
                "amount": str(10 + i), "nameOrig": f"C{i}", "oldbalanceOrg": str(10000 + i),
                "newbalanceOrig": str(9000 + i), "nameDest": f"D{i}", "oldbalanceDest": str(2000 + i),
                "newbalanceDest": str(2010 + i), "isFraud": str(i % 2), "isFlaggedFraud": "0",
            })
    return path


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory, paysim_csv):
    root = tmp_path_factory.mktemp("interaction")
    return {mode: generate_experiment(mode, output_dir=root / mode, paysim_path=paysim_csv, seed=42,
                                      manifest_path=INTERACTION_MANIFEST)
            for mode in ("predictive", "negative_control")} | {"root": root}


def _model():
    return _load_manifest(INTERACTION_MANIFEST)["interaction_model"]


def _exact(model):
    rho = Fraction(str(model["rho"]))
    q = {s: Fraction(str(model["q"][s])) for s in INTERACTION_SLOTS}
    r = {s: (q[s] - rho) / (1 - rho) for s in INTERACTION_SLOTS}
    return rho, q, r


def _without_labels(rows):
    return [{k: v for k, v in row.items() if k not in LABEL_COLUMNS} for row in rows]


def test_interaction_manifest_validates_and_rejects_inconsistent_parameters():
    manifest = _load_manifest(INTERACTION_MANIFEST)
    assert manifest["version_model"] == "coordinated_mixture"
    assert "not empirical" in manifest["interaction_model"]["status"]
    bad_r = copy.deepcopy(manifest)
    bad_r["interaction_model"]["r"]["device"] = 0.61
    with pytest.raises(ValueError, match="must equal"):
        validate_manifest(bad_r)
    bad_q = copy.deepcopy(manifest)
    bad_q["interaction_model"]["q"]["auth"] = 0.2
    with pytest.raises(ValueError, match="must exceed rho"):
        validate_manifest(bad_q)


def test_exact_marginals_match_across_labels_while_joints_differ():
    rho, q, r = _exact(_model())
    legit = independent_distribution(q)
    susp = coordinated_distribution(r, rho)
    assert sum(legit.values()) == 1 and sum(susp.values()) == 1
    for i, slot in enumerate(INTERACTION_SLOTS):
        assert rho + (1 - rho) * r[slot] == q[slot]
        assert sum(p for bits, p in susp.items() if bits[i]) == q[slot]
        assert sum(p for bits, p in legit.items() if bits[i]) == q[slot]
    for (i, a), (j, b) in itertools.combinations(enumerate(INTERACTION_SLOTS), 2):
        joint_s = sum(p for bits, p in susp.items() if bits[i] and bits[j])
        joint_l = sum(p for bits, p in legit.items() if bits[i] and bits[j])
        assert joint_l == q[a] * q[b]
        # Cov_S = rho (1 - rho)(1 - r_a)(1 - r_b) = rho (1 - q_a)(1 - q_b) / (1 - rho) > 0.
        assert joint_s - q[a] * q[b] == rho * (1 - q[a]) * (1 - q[b]) / (1 - rho)
    all_risky = (1, 1, 1, 1)
    assert susp[all_risky] == rho + (1 - rho) * r["device"] * r["auth"] * r["account"] * r["transaction"]
    assert susp[all_risky] > legit[all_risky]


def test_fixed_seed_200000_draw_sanity_check():
    manifest = _load_manifest(INTERACTION_MANIFEST)
    rho, q, r = _exact(manifest["interaction_model"])
    exact = {"legitimate": independent_distribution(q), "suspicious": coordinated_distribution(r, rho)}
    n = 200_000
    for intent in ("legitimate", "suspicious"):
        draws = [_latent_draw(manifest, intent, 42, f"SANITY-{i}") for i in range(n)]
        for slot in INTERACTION_SLOTS:
            assert abs(sum(d["risky"][slot] for d in draws) / n - float(q[slot])) < 0.005
        for (i, a), (j, b) in itertools.combinations(enumerate(INTERACTION_SLOTS), 2):
            expected = float(sum(p for bits, p in exact[intent].items() if bits[i] and bits[j]))
            observed = sum(d["risky"][a] and d["risky"][b] for d in draws) / n
            assert abs(observed - expected) < 0.005
        coordinated = sum(d["coordinated"] for d in draws) / n
        assert (abs(coordinated - float(rho)) < 0.005) if intent == "suspicious" else coordinated == 0


def test_coordination_only_in_suspicious_and_profile_b_and_versions_follow_latent(cohorts):
    for mode in ("predictive", "negative_control"):
        for case in cohorts[mode]["cases"]:
            components = json.loads(case["components_json"])
            if case["is_anchor"]:
                assert "latent_risky" not in components
                continue
            if components["coordinated"]:
                assert case["recipe_source"] in {"suspicious", "profile_B"}
                assert all(components["latent_risky"].values())
            assert components["versions"] == _versions_from_latent(components["latent_risky"], components["skeleton"])


def test_interaction_diagnostics_are_reported_not_forced(cohorts):
    diagnostics = cohorts["predictive"]["design_audit"]["interaction_diagnostics"]
    assert diagnostics["pooled_latent_risky_rates"]["label_0"]["n"] == 140
    assert set(diagnostics["pooled_rate_difference_label1_minus_label0"]) == set(INTERACTION_SLOTS)
    assert diagnostics["coordinated_component_count_by_label"]["label_0"] == 0
    assert len(diagnostics["pairwise_cooccurrence_by_label"]) == 6
    assert "interaction_diagnostics" in cohorts["negative_control"]["design_audit"]


def test_negative_control_artifacts_are_invariant_to_label_seed(tmp_path, paysim_csv):
    first = generate_experiment("negative_control", output_dir=tmp_path / "a", paysim_path=paysim_csv,
                                seed=31, label_seed=5, manifest_path=INTERACTION_MANIFEST)
    second = generate_experiment("negative_control", output_dir=tmp_path / "b", paysim_path=paysim_csv,
                                 seed=31, label_seed=29, manifest_path=INTERACTION_MANIFEST)
    assert _without_labels(first["events"]) == _without_labels(second["events"])
    assert first["historical_event_input"] == second["historical_event_input"]
    assert first["baseline_profiles"] == second["baseline_profiles"]
    keep = ("case_id", "recipe_source", "is_anchor", "components_json", "declared_scenario_intent")
    assert [{k: c[k] for k in keep} for c in first["cases"]] == [{k: c[k] for k in keep} for c in second["cases"]]
    assert any(a["ato_label"] != b["ato_label"] for a, b in zip(first["cases"], second["cases"]))


def test_history_and_amounts_do_not_depend_on_labels_or_version_model(tmp_path, paysim_csv):
    runs = [generate_experiment("predictive", output_dir=tmp_path / name, paysim_path=paysim_csv, seed=17,
                                label_seed=label_seed, manifest_path=manifest)
            for name, label_seed, manifest in (("i3", 3, INTERACTION_MANIFEST), ("i23", 23, INTERACTION_MANIFEST),
                                               ("d3", 3, MANIFEST_PATH))]
    for other in runs[1:]:
        assert other["historical_event_input"] == runs[0]["historical_event_input"]
        assert other["baseline_profiles"] == runs[0]["baseline_profiles"]
    amounts = []
    for result in runs:
        by_user = defaultdict(list)
        for row in result["events"]:
            by_user[row["user_id"]].append(row)
        amounts.append({c["case_id"]: r["amount"] for c in result["cases"]
                        for r in by_user[c["user_id"]] if r["event_category"] == "TRANSACTION"})
    for other in amounts[1:]:
        shared = set(amounts[0]) & set(other)
        assert shared and all(amounts[0][k] == other[k] for k in shared)


def test_default_manifest_path_has_no_interaction_artifacts(tmp_path, paysim_csv):
    assert "version_model" not in _load_manifest()
    result = generate_experiment("predictive", output_dir=tmp_path / "default", paysim_path=paysim_csv, seed=42)
    assert all("latent_risky" not in json.loads(c["components_json"]) for c in result["cases"])
    assert "interaction_diagnostics" not in result["design_audit"]
    assert "version_model" not in json.loads((tmp_path / "default" / "run_metadata.json").read_text())
