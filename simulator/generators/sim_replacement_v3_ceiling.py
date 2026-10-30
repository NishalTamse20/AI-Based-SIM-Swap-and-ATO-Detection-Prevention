"""Exact Bayes accuracy ceilings for v3 cohort designs, computed before any labels are drawn.

Two ceilings are reported, both with equal class priors and conditional on the realized
seed-specific skeleton sets (20 families x 10 cell positions: 3 anchors + 7 skeletons):

* version-level: an oracle observes each case's cell position (skeleton) and the version of
  every present slot. This is an upper bound on any feature representation.
* feature-level: the Bayes ceiling for the specified feature representation and population.
  Each (family, position, latent version combination) is pushed through the generator's own
  event mapping and the Phase 5/6 builders, so family-dependent encodings and missing-slot
  merging (e.g. an auth success is indistinguishable from an absent auth slot) come from the
  real code. It is not an achievable target for leave-one-family-out models, which cannot
  learn a held-out family's encodings from that family.

Excluded columns (audited against the generator, history builder and feature builder):
``baseline_*`` come only from history built by ``_build_history`` from seed/case identity;
``transaction_amount`` is assigned by seed/case rank from a shared pool; and
``transaction_deviation`` is amount minus the baseline mean. None reads labels, intents,
skeletons or versions, so (treating the seed-derived hash streams as independent) they are
independent of the label and of the included features and cannot change either ceiling.
Event order and timestamps are not used by the Phase 6 features in these column sets.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

from ml.features.feature_generator import build_feature_rows
from ml.models.random_forest_experiment import FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS
from simulator.baseline.behavioural_baseline import build_baseline_profiles
from simulator.generators.event_generator import make_event
from simulator.generators.sim_replacement_v3 import (
    INTERACTION_SLOTS,
    _family_skeletons,
    _load_manifest,
    _slot_tokens,
    _token_event,
    _versions_from_latent,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "bayes_ceiling_v3"
HALF = Fraction(1, 2)
MODEL_A_KEY_COLUMNS = tuple(c for c in PHASE8_FEATURE_COLUMNS if not c.startswith("baseline_"))
MODEL_B_KEY_COLUMNS = (*MODEL_A_KEY_COLUMNS, "transaction_count", "transaction_type")
INDEPENDENT_KEYS = {"device": "p_new_device", "auth": "p_auth_failure",
                    "account": "p_new_beneficiary", "transaction": "p_high_risk_transaction"}
ASSUMPTIONS = (
    "Equal class priors; population is the 20 families x 10 cell positions (3 anchors + 7 skeletons) "
    "of the realized seed-specific skeleton sets, each weighted equally; the ceiling is conditional on those skeletons.",
    "baseline_*, transaction_amount and transaction_deviation are excluded: they are derived only from seed/case "
    "identity streams (history bundles, amount pool rank) that never read labels, intents, skeletons or versions; "
    "seed-derived hash streams are treated as independent.",
    "Phase 6 features in these column sets do not depend on event order or timestamps within T..T+15.",
    "Feature-level ceiling is the Bayes ceiling for the stated feature representation and population; it is not an "
    "achievable target for leave-one-family-out models.",
    "All probabilities are provisional synthetic design parameters, not empirical rates.",
)


def _fraction(value: Any) -> Fraction:
    return Fraction(str(value))


def per_position_ceiling(items: list[tuple[Fraction, dict[Any, Fraction], dict[Any, Fraction]]]) -> Fraction:
    """Oracle knows each item's position: sum_w w * 1/2 * sum_x max(P_L(x), P_S(x))."""
    return sum((weight * HALF * sum(max(p_l.get(k, 0), p_s.get(k, 0)) for k in set(p_l) | set(p_s))
                for weight, p_l, p_s in items), Fraction(0))


def pooled_ceiling(items: list[tuple[Fraction, dict[Any, Fraction], dict[Any, Fraction]]]) -> Fraction:
    """Classifier sees only the observation: 1/2 * sum_x max(sum_w w P_L(x), sum_w w P_S(x))."""
    pooled_l: dict[Any, Fraction] = {}
    pooled_s: dict[Any, Fraction] = {}
    for weight, p_l, p_s in items:
        for key, value in p_l.items():
            pooled_l[key] = pooled_l.get(key, Fraction(0)) + weight * value
        for key, value in p_s.items():
            pooled_s[key] = pooled_s.get(key, Fraction(0)) + weight * value
    return HALF * sum((max(pooled_l.get(k, 0), pooled_s.get(k, 0)) for k in set(pooled_l) | set(pooled_s)),
                      Fraction(0))


def independent_distribution(rates: dict[str, Fraction]) -> dict[tuple[int, ...], Fraction]:
    """Joint over INTERACTION_SLOTS bits with independent per-slot risky rates."""
    distribution = {}
    for bits in itertools.product((0, 1), repeat=len(INTERACTION_SLOTS)):
        probability = Fraction(1)
        for slot, bit in zip(INTERACTION_SLOTS, bits):
            probability *= rates[slot] if bit else 1 - rates[slot]
        distribution[bits] = probability
    return distribution


def coordinated_distribution(r: dict[str, Fraction], rho: Fraction) -> dict[tuple[int, ...], Fraction]:
    """rho * 1[all risky] + (1 - rho) * independent(r)."""
    base = independent_distribution(r)
    return {bits: (1 - rho) * p + (rho if all(bits) else 0) for bits, p in base.items()}


def latent_distribution(manifest: dict[str, Any], intent: str) -> dict[tuple[int, ...], Fraction]:
    if manifest.get("version_model", "independent") == "independent":
        params = manifest["design_assumptions"]["version_probabilities"][intent]
        return independent_distribution({s: _fraction(params[INDEPENDENT_KEYS[s]]) for s in INTERACTION_SLOTS})
    model = manifest["interaction_model"]
    if intent in model["coordinated_intents"]:
        rho = _fraction(model["rho"])
        r = {s: (_fraction(model["q"][s]) - rho) / (1 - rho) for s in INTERACTION_SLOTS}
        return coordinated_distribution(r, rho)
    return independent_distribution({s: _fraction(model["q"][s]) for s in INTERACTION_SLOTS})


def marginal(distribution: dict[tuple[int, ...], Fraction], slots: tuple[str, ...]) -> dict[tuple[int, ...], Fraction]:
    index = [INTERACTION_SLOTS.index(s) for s in slots]
    out: dict[tuple[int, ...], Fraction] = {}
    for bits, p in distribution.items():
        key = tuple(bits[i] for i in index)
        out[key] = out.get(key, Fraction(0)) + p
    return out


def cell_positions(manifest: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Label-neutral positions: per family, the anchors then the family's skeletons."""
    anchors = manifest["design_assumptions"]["anchor_cases_per_label_per_family"]
    anchor_skeleton = {k: manifest["anchor_skeleton"][k] for k in ("auth_k", "recovery", "account", "transaction")}
    positions = []
    for family in manifest["families"]:
        family_id = family["template_family_id"]
        positions += [{"family": family, "skeleton": anchor_skeleton, "is_anchor": True}] * anchors
        positions += [{"family": family, "skeleton": s, "is_anchor": False}
                      for s in _family_skeletons(manifest, seed, family_id)]
    return positions


def _observed_slots(skeleton: dict[str, Any]) -> tuple[str, ...]:
    present = {"device": True, "auth": bool(skeleton["auth_k"]), "account": bool(skeleton["account"]),
               "transaction": bool(skeleton["transaction"])}
    return tuple(s for s in INTERACTION_SLOTS if present[s])


def version_level_ceiling(manifest: dict[str, Any], seed: int) -> Fraction:
    positions = cell_positions(manifest, seed)
    weight = Fraction(1, len(positions))
    p_legit, p_susp = latent_distribution(manifest, "legitimate"), latent_distribution(manifest, "suspicious")
    items = []
    for position in positions:
        if position["is_anchor"]:
            items.append((weight, {"anchor": Fraction(1)}, {"anchor": Fraction(1)}))
            continue
        slots = _observed_slots(position["skeleton"])
        items.append((weight, marginal(p_legit, slots), marginal(p_susp, slots)))
    return per_position_ceiling(items)


def feature_key(family: dict[str, Any], skeleton: dict[str, Any], versions: dict[str, Any],
                manifest: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    """Feature vector produced by the generator's event mapping and the real Phase 5/6 builders."""
    trigger_time = datetime(2026, 1, 1)
    known, new = "DEV-CEIL-K", "DEV-CEIL-N"
    device = new if versions["device"] == "new" else known
    common = {"user_id": "USR-CEIL", "account_id": "ACC-CEIL", "scenario_type": None}
    trigger = make_event(event_id="EVT-CEIL-00", event_category="TELECOM", event_type="SIM_REPLACEMENT",
                         timestamp=trigger_time, device_id=known, ato_label=0, status="SUCCESS", **common)
    history = [make_event(event_id="EVT-CEIL-H1", event_category="AUTHENTICATION", event_type="LOGIN",
                          timestamp=trigger_time - timedelta(days=1), device_id=known, ato_label=None,
                          status="SUCCESS", **common)]
    window = []
    for index, token in enumerate(_slot_tokens(skeleton), start=1):
        name = _token_event(token, family, manifest, versions)
        category = manifest["event_categories"][name]
        row = make_event(event_id=f"EVT-CEIL-{index:02d}", event_category=category, event_type=name,
                         timestamp=trigger_time + timedelta(seconds=index), device_id=device, ato_label=0,
                         status="FAILURE" if name == family["auth_failure_event"] else "SUCCESS",
                         amount="100" if category == "TRANSACTION" else None, **common)
        if category == "TRANSACTION":
            row["transaction_type"] = manifest["transaction_type_mapping"][name]
        window.append(row)
    profiles = build_baseline_profiles(history + [trigger])
    row = build_feature_rows([trigger, *window], profiles)[0]
    return tuple(str(row[c]) for c in columns)


def feature_level_ceiling(manifest: dict[str, Any], seed: int, columns: tuple[str, ...]) -> Fraction:
    positions = cell_positions(manifest, seed)
    weight = Fraction(1, len(positions))
    p_legit, p_susp = latent_distribution(manifest, "legitimate"), latent_distribution(manifest, "suspicious")
    anchor_versions = {"device": manifest["anchor_skeleton"]["device_version"], "auth": None,
                       "account": None, "transaction": None}
    cache: dict[tuple[str, str, str], tuple[str, ...]] = {}

    def _key(family: dict[str, Any], skeleton: dict[str, Any], versions: dict[str, Any]) -> tuple[str, ...]:
        cache_key = (family["template_family_id"], json.dumps(skeleton, sort_keys=True), json.dumps(versions, sort_keys=True))
        if cache_key not in cache:
            cache[cache_key] = feature_key(family, skeleton, versions, manifest, columns)
        return cache[cache_key]

    items = []
    for position in positions:
        family, skeleton = position["family"], position["skeleton"]
        if position["is_anchor"]:
            key = _key(family, skeleton, anchor_versions)
            items.append((weight, {key: Fraction(1)}, {key: Fraction(1)}))
            continue
        p_l: dict[tuple[str, ...], Fraction] = {}
        p_s: dict[tuple[str, ...], Fraction] = {}
        for bits in p_legit:
            versions = _versions_from_latent(dict(zip(INTERACTION_SLOTS, map(bool, bits))), skeleton)
            key = _key(family, skeleton, versions)
            p_l[key] = p_l.get(key, Fraction(0)) + p_legit[bits]
            p_s[key] = p_s.get(key, Fraction(0)) + p_susp[bits]
        items.append((weight, p_l, p_s))
    return pooled_ceiling(items)


def compute_ceilings(manifest_path: str | Path, seed: int = 42) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    version_level = version_level_ceiling(manifest, seed)
    feature_a = feature_level_ceiling(manifest, seed, MODEL_A_KEY_COLUMNS)
    feature_b = feature_level_ceiling(manifest, seed, MODEL_B_KEY_COLUMNS)
    single = {}
    p_legit, p_susp = latent_distribution(manifest, "legitimate"), latent_distribution(manifest, "suspicious")
    for slot in INTERACTION_SLOTS:
        single[slot] = float(per_position_ceiling([(Fraction(1), marginal(p_legit, (slot,)), marginal(p_susp, (slot,)))]))
    return {
        "manifest": str(Path(manifest_path).resolve().relative_to(ROOT)) if Path(manifest_path).resolve().is_relative_to(ROOT)
        else str(manifest_path),
        "version_model": manifest.get("version_model", "independent"), "seed": seed,
        "version_level_ceiling_upper_bound": {"value": float(version_level), "exact": str(version_level)},
        "feature_level_ceiling": {
            "model_a_columns": {"value": float(feature_a), "exact": str(feature_a), "columns": list(MODEL_A_KEY_COLUMNS)},
            "model_b_columns": {"value": float(feature_b), "exact": str(feature_b), "columns": list(MODEL_B_KEY_COLUMNS)}},
        "single_latent_version_ceiling_if_always_observed": single,
        "assumptions": list(ASSUMPTIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact v3 Bayes ceilings before model evaluation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing {args.output}")
    result = compute_ceilings(args.manifest, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("version_level_ceiling_upper_bound", "feature_level_ceiling")}, indent=2))


if __name__ == "__main__":
    main()
