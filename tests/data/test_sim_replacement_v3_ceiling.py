import csv
import json
import random
from datetime import datetime, timedelta
from fractions import Fraction

import pytest

from ml.features.feature_generator import build_feature_rows
from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.generators.sim_replacement_v3 import (
    INTERACTION_SLOTS,
    MANIFEST_PATH,
    _load_manifest,
    generate_experiment,
)
from simulator.generators.sim_replacement_v3_ceiling import (
    MODEL_A_KEY_COLUMNS,
    MODEL_B_KEY_COLUMNS,
    compute_ceilings,
    coordinated_distribution,
    feature_key,
    feature_level_ceiling,
    independent_distribution,
    latent_distribution,
    marginal,
    per_position_ceiling,
    pooled_ceiling,
    version_level_ceiling,
)


INTERACTION_MANIFEST = MANIFEST_PATH.with_name("sim_replacement_v3_interaction_manifest.json")
F = Fraction


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


def test_single_slot_independent_case_matches_hand_value():
    # 0.5 + |0.9 - 0.5| / 2 = 0.7
    item = (F(1), {1: F(1, 2), 0: F(1, 2)}, {1: F(9, 10), 0: F(1, 10)})
    assert per_position_ceiling([item]) == F(7, 10)
    assert pooled_ceiling([item]) == F(7, 10)


def test_single_observed_slot_under_mixture_is_exactly_half():
    manifest = _load_manifest(INTERACTION_MANIFEST)
    legit, susp = latent_distribution(manifest, "legitimate"), latent_distribution(manifest, "suspicious")
    for slot in INTERACTION_SLOTS:
        assert per_position_ceiling([(F(1), marginal(legit, (slot,)), marginal(susp, (slot,)))]) == F(1, 2)


def test_two_slot_mixture_matches_hand_value_seven_twelfths():
    # q = 1/2, rho = 1/4 -> r = 1/3; P_S = (1/3, 1/6, 1/6, 1/3) for (00, 01, 10, 11).
    q = {s: F(1, 2) for s in INTERACTION_SLOTS}
    rho = F(1, 4)
    r = {s: (q[s] - rho) / (1 - rho) for s in INTERACTION_SLOTS}
    legit = marginal(independent_distribution(q), ("device", "auth"))
    susp = marginal(coordinated_distribution(r, rho), ("device", "auth"))
    assert susp == {(0, 0): F(1, 3), (0, 1): F(1, 6), (1, 0): F(1, 6), (1, 1): F(1, 3)}
    assert legit == {k: F(1, 4) for k in susp}
    assert per_position_ceiling([(F(1), legit, susp)]) == F(7, 12)


def test_auth_success_merging_lowers_feature_level_below_version_level():
    # Device always observed (1/2 vs 9/10); auth slot present in half the positions (3/10 vs 7/10).
    def dev(p):
        return {1: p, 0: 1 - p}

    def joint(pd, pa):
        return {(d, a): (pd if d else 1 - pd) * (pa if a else 1 - pa) for d in (0, 1) for a in (0, 1)}

    version_items = [(F(1, 2), joint(F(1, 2), F(3, 10)), joint(F(9, 10), F(7, 10))),
                     (F(1, 2), dev(F(1, 2)), dev(F(9, 10)))]
    assert per_position_ceiling(version_items) == F(18, 25)

    # Observation is (device, anomaly). An absent auth slot also yields anomaly 0, so pooling
    # merges it with the success version under the same key.
    def absent(pd):
        return {(1, 0): pd, (0, 0): 1 - pd}

    feature_items = [(F(1, 2), joint(F(1, 2), F(3, 10)), joint(F(9, 10), F(7, 10))),
                     (F(1, 2), absent(F(1, 2)), absent(F(9, 10)))]
    assert pooled_ceiling(feature_items) == F(7, 10)


def test_negative_control_theory_identical_mixtures_give_half():
    mixture = {("a",): F(3, 10), ("b",): F(7, 10)}
    assert pooled_ceiling([(F(1), mixture, mixture)]) == F(1, 2)


@pytest.mark.parametrize("manifest_path", [MANIFEST_PATH, INTERACTION_MANIFEST])
def test_enumerated_feature_mapping_reproduces_generated_features(tmp_path, paysim_csv, manifest_path):
    manifest = _load_manifest(manifest_path)
    families = {f["template_family_id"]: f for f in manifest["families"]}
    for mode in ("predictive", "negative_control"):
        result = generate_experiment(mode, output_dir=tmp_path / mode, paysim_path=paysim_csv, seed=42,
                                     manifest_path=manifest_path)
        features = {row["user_id"]: row for row in result["features"]}
        for case in result["cases"]:
            components = json.loads(case["components_json"])
            expected = feature_key(families[case["template_family_id"]], components["skeleton"],
                                   components["versions"], manifest, MODEL_B_KEY_COLUMNS)
            actual = tuple(str(features[case["user_id"]][c]) for c in MODEL_B_KEY_COLUMNS)
            assert actual == expected


def test_key_features_ignore_event_order_and_timestamps(tmp_path, paysim_csv):
    result = generate_experiment("predictive", output_dir=tmp_path / "p", paysim_path=paysim_csv, seed=42)
    rng = random.Random(7)
    for case in result["cases"][:40]:
        rows = [dict(r) for r in result["events"] if r["user_id"] == case["user_id"]]
        trigger = next(r for r in rows if r["event_category"] == "TELECOM")
        window = [r for r in rows if r["event_category"] != "TELECOM"]
        start = datetime.fromisoformat(trigger["timestamp"])
        offsets = sorted(rng.sample(range(1, 901), len(window)))
        rng.shuffle(window)
        for row, offset in zip(window, offsets):
            row["timestamp"] = (start + timedelta(seconds=offset)).isoformat(timespec="seconds")
        profile = [p for p in result["baseline_profiles"] if p["user_id"] == case["user_id"]]
        original = next(r for r in result["features"] if r["user_id"] == case["user_id"])
        shuffled = build_feature_rows([trigger, *window], profile)[0]
        assert all(original[c] == shuffled[c] for c in MODEL_B_KEY_COLUMNS)


def test_default_v3_ceilings_regression_and_ordering():
    manifest = _load_manifest()
    version = version_level_ceiling(manifest, 42)
    feature_a = feature_level_ceiling(manifest, 42, MODEL_A_KEY_COLUMNS)
    feature_b = feature_level_ceiling(manifest, 42, MODEL_B_KEY_COLUMNS)
    assert abs(float(version) - 0.685) < 0.0005
    assert F(1, 2) <= feature_a <= feature_b <= version


def test_interaction_ceilings_ordering_and_single_versions_at_half():
    result = compute_ceilings(INTERACTION_MANIFEST, 42)
    assert all(v == 0.5 for v in result["single_latent_version_ceiling_if_always_observed"].values())
    version = result["version_level_ceiling_upper_bound"]["value"]
    feature_a = result["feature_level_ceiling"]["model_a_columns"]["value"]
    feature_b = result["feature_level_ceiling"]["model_b_columns"]["value"]
    assert 0.5 <= feature_a <= feature_b <= version
    assert any("not an achievable target" in a for a in result["assumptions"])
