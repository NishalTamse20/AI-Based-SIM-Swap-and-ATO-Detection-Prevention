"""Generate v3 controlled SIM_REPLACEMENT cohorts from shared slot skeletons.

Each family has a fixed set of label-neutral slot skeletons (which slots are
present and how many events they emit). In the predictive cohort both labels
reuse the same skeletons and the declared intent only chooses event versions
(e.g. new vs known device, failure vs success). Anchors are shared by both
labels. Ordering and timestamps depend only on seed, case_id and slot tokens.
The negative control draws skeletons, sources and versions without reading
labels. All probabilities and quotas are provisional synthetic design
parameters, not empirical rates; outputs do not establish real-world performance.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
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
from simulator.generators.sim_replacement_v2 import (
    _baseline_audit,
    _build_history,
    _stable_rank,
    _write_csv,
)
from simulator.validation.dataset_validator import CATEGORY_EVENT_TYPES, PAYSIM_TRANSACTION_TYPES


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "simulator" / "scenarios" / "sim_replacement_v3_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "controlled_sim_replacement_v3"
FAMILY_COUNT = 20
CASES_PER_LABEL_PER_FAMILY = 10
CASES_PER_FAMILY = 2 * CASES_PER_LABEL_PER_FAMILY
TOTAL_CASES = CASES_PER_FAMILY * FAMILY_COUNT
OBSERVATION_MINUTES = 15
WINDOW_PAYSIM_SEED_OFFSET = 5051
MAX_ORDER_ATTEMPTS = 1000
FAMILY_SLOTS = {
    "device_event": "DEVICE", "auth_failure_event": "AUTHENTICATION",
    "auth_success_event": "AUTHENTICATION", "recovery_event": "RECOVERY",
    "high_risk_transaction_event": "TRANSACTION",
}
VERSION_KEYS = ("p_new_device", "p_auth_failure", "p_new_beneficiary", "p_high_risk_transaction")
# "independent" (default, manifest key absent) is the original v3 path; "coordinated_mixture"
# matches every latent version's marginal across labels and differs only in co-occurrence.
VERSION_MODELS = ("independent", "coordinated_mixture")
INTERACTION_SLOTS = ("device", "auth", "account", "transaction")
INTERACTION_RISKY = {"device": "new", "auth": "failure", "account": "risk", "transaction": "high_risk"}
AUTH_TOKENS = {"auth", "login"}
# Discrete in-window features for overlap/lookup diagnostics. Continuous
# amounts and baseline columns are excluded so exact-vector lookup is meaningful.
WINDOW_DISCRETE_FEATURES = tuple(
    c for c in FEATURE_COLUMNS
    if not c.startswith("baseline_") and c not in {"transaction_amount", "transaction_deviation"}
)
EVALUATION_ONLY_COLUMNS = {"template_family_id", "scenario_type", "declared_scenario_intent",
                           "recipe_source", "is_anchor", "components_json",
                           "paysim_isFraud", "paysim_isFlaggedFraud"}
CASE_FIELDS = ("case_id", "template_family_id", "user_id", "account_id", "event_id",
               "scenario_type", "ato_label", "declared_scenario_intent", "recipe_source",
               "is_anchor", "components_json")
INTERPRETATION = ("Synthetic controlled experiment: measures whether designed patterns are "
                  "recoverable under family hold-out. Families are combinations of shared "
                  "building blocks, so a held-out family is not truly novel. Not real-world "
                  "detection performance.")
MISSING_VALUE_POLICY = ("transaction_count=0 is an observed count. transaction_amount, "
                        "transaction_type and transaction_deviation are left missing (empty) "
                        "when no transaction is observed in T..T+15; they are not imputed. "
                        "Model-side handling must be decided and approved at training time.")


def _case_rng(seed: int, stage: str, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stage}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject objective manifest defects; similarity is reported, not thresholded."""
    categories = manifest["event_categories"]
    for name, category in categories.items():
        if name not in CATEGORY_EVENT_TYPES.get(category, ()):
            raise ValueError(f"invalid event_type {category}/{name}")
    for key, category in (("risk_account_event", "ACCOUNT"), ("benign_account_event", "ACCOUNT"),
                          ("low_risk_transaction_event", "TRANSACTION")):
        if categories.get(manifest[key]) != category:
            raise ValueError(f"{key} must be a {category} event")
    families = manifest["families"]
    if len(families) != FAMILY_COUNT:
        raise ValueError(f"manifest must define exactly {FAMILY_COUNT} template families")
    specs: dict[tuple[str, ...], str] = {}
    for family in families:
        for slot, category in FAMILY_SLOTS.items():
            if categories.get(family[slot]) != category:
                raise ValueError(f"{family['template_family_id']}/{slot} is not a {category} event")
        spec = tuple(family[slot] for slot in FAMILY_SLOTS)
        if spec in specs:
            raise ValueError(f"identical family definitions: {specs[spec]}/{family['template_family_id']}")
        specs[spec] = family["template_family_id"]

    assumptions = manifest["design_assumptions"]
    version_model = manifest.get("version_model", "independent")
    if version_model not in VERSION_MODELS:
        raise ValueError(f"unknown version_model {version_model!r}")
    if version_model == "independent":
        profiles = assumptions["version_probabilities"]
        if set(profiles) != {"legitimate", "suspicious"}:
            raise ValueError("version_probabilities must define legitimate and suspicious")
        for key in VERSION_KEYS:
            if any(not 0.1 <= profiles[name][key] <= 0.9 for name in profiles):
                raise ValueError(f"{key} must stay within [0.1, 0.9] so both labels can produce any version")
    else:
        _validate_interaction_model(manifest["interaction_model"])
    mapping = manifest["transaction_type_mapping"]
    if any(categories.get(name) != "TRANSACTION" or value not in PAYSIM_TRANSACTION_TYPES
           for name, value in mapping.items()):
        raise ValueError("transaction_type_mapping must map TRANSACTION events to PaySim types")

    anchors = assumptions["anchor_cases_per_label_per_family"]
    variants = assumptions["variant_skeletons_per_family"]
    if anchors + variants != CASES_PER_LABEL_PER_FAMILY:
        raise ValueError("anchors plus variant skeletons must fill each family/label cell")
    anchor = manifest["anchor_skeleton"]
    quotas = assumptions["skeleton_quotas_per_family"]
    auth_quota = quotas["auth_k1"] + quotas["auth_k2"]
    # Including anchors, every label-neutral slot must be both present and absent in each cell.
    expected = {"device": CASES_PER_LABEL_PER_FAMILY, "login": CASES_PER_LABEL_PER_FAMILY}
    for slot, quota in (("recovery", quotas["recovery"]), ("transaction", quotas["transaction"]),
                        ("account", quotas["account"]), ("auth", auth_quota)):
        if not 0 <= quota <= variants:
            raise ValueError(f"{slot} quota must fit within {variants} variant skeletons")
        anchor_present = bool(anchor["auth_k"]) if slot == "auth" else bool(anchor[slot])
        expected[slot] = quota + anchors * anchor_present
        if not 0 < expected[slot] < CASES_PER_LABEL_PER_FAMILY:
            raise ValueError(f"{slot} must be both present and absent in every family/label cell")
    if expected != assumptions["presence_per_label_cell_including_anchors"]:
        raise ValueError("presence_per_label_cell_including_anchors does not match the quotas")
    sources = assumptions["negative_control_recipe_sources_per_family"]
    if sources != {"anchor": 2 * anchors, "profile_A": variants, "profile_B": variants}:
        raise ValueError("negative-control sources must be 2x anchors plus each skeleton once per profile")


def _validate_interaction_model(model: dict[str, Any]) -> None:
    """Require equal marginals by construction: q_s > rho and r_s = (q_s - rho) / (1 - rho)."""
    if tuple(model["slots"]) != INTERACTION_SLOTS or model["risky_version"] != INTERACTION_RISKY:
        raise ValueError("interaction_model slots/risky versions must match the generator")
    if model["independent_intents"] != ["legitimate"] or model["coordinated_intents"] != ["suspicious"]:
        raise ValueError("interaction_model must keep legitimate independent and suspicious coordinated")
    rho = model["rho"]
    if not 0 < rho < 1:
        raise ValueError("rho must lie strictly between 0 and 1")
    for slot in INTERACTION_SLOTS:
        q, r = model["q"][slot], model["r"][slot]
        if not rho < q < 1:
            raise ValueError(f"q[{slot}] must exceed rho so r stays a probability")
        if abs(r - (q - rho) / (1 - rho)) > 1e-12:
            raise ValueError(f"r[{slot}] must equal (q - rho) / (1 - rho)")


def _latent_draw(manifest: dict[str, Any], intent: str, seed: int, case_id: str) -> dict[str, Any]:
    """Coordinated-mixture draw of all four latent risk versions (fixed uniform order)."""
    model = manifest["interaction_model"]
    rng = _case_rng(seed, "versions", case_id)
    u_couple = rng.random()
    uniforms = [rng.random() for _ in INTERACTION_SLOTS]
    coordinated = intent in model["coordinated_intents"] and u_couple < model["rho"]
    rates = model["r"] if intent in model["coordinated_intents"] else model["q"]
    risky = {slot: coordinated or u < rates[slot] for slot, u in zip(INTERACTION_SLOTS, uniforms)}
    return {"risky": risky, "coordinated": coordinated}


def _family_skeletons(manifest: dict[str, Any], seed: int, family_id: str) -> list[dict[str, Any]]:
    """Build the family's label-neutral skeletons from exact quotas."""
    assumptions = manifest["design_assumptions"]
    variants = assumptions["variant_skeletons_per_family"]
    quotas = assumptions["skeleton_quotas_per_family"]
    rng = _case_rng(seed, "skeletons", family_id)

    def _quota(present: int) -> list[bool]:
        values = [True] * present + [False] * (variants - present)
        rng.shuffle(values)
        return values

    auth_k = [1] * quotas["auth_k1"] + [2] * quotas["auth_k2"]
    auth_k += [0] * (variants - len(auth_k))
    rng.shuffle(auth_k)
    recovery, account, transaction = _quota(quotas["recovery"]), _quota(quotas["account"]), _quota(quotas["transaction"])
    return [{"auth_k": auth_k[i], "recovery": recovery[i], "account": account[i],
             "transaction": transaction[i]} for i in range(variants)]


def _slot_tokens(skeleton: dict[str, Any]) -> list[str]:
    tokens = ["device"] + ["auth"] * skeleton["auth_k"]
    tokens += ["recovery"] * skeleton["recovery"] + ["login"]
    tokens += ["account"] * skeleton["account"] + ["transaction"] * skeleton["transaction"]
    return tokens


def _order_valid(order: list[str]) -> bool:
    if not order or order[0] != "device":
        return False
    first_auth = min(i for i, token in enumerate(order) if token in AUTH_TOKENS)
    return all(i > first_auth for i, token in enumerate(order) if token in {"account", "transaction"})


def _slot_order(skeleton: dict[str, Any], seed: int, case_id: str) -> list[str]:
    """Random valid order from seed, case_id and slot tokens only (never versions/labels)."""
    tokens = _slot_tokens(skeleton)
    rng = _case_rng(seed, "order", case_id)
    rest = tokens[1:]
    for _ in range(MAX_ORDER_ATTEMPTS):
        rng.shuffle(rest)
        order = ["device", *rest]
        if _order_valid(order):
            return order
    raise ValueError(f"could not find a valid slot order for {case_id}")


def _versions(manifest: dict[str, Any], source: str, skeleton: dict[str, Any],
              seed: int, case_id: str) -> dict[str, Any]:
    assumptions = manifest["design_assumptions"]
    if source == "anchor":
        return {"device": manifest["anchor_skeleton"]["device_version"], "auth": None,
                "account": None, "transaction": None}
    name = assumptions["negative_control_profile_parameters"].get(source, source)
    if manifest.get("version_model", "independent") == "coordinated_mixture":
        return _versions_from_latent(_latent_draw(manifest, name, seed, case_id)["risky"], skeleton)
    params = assumptions["version_probabilities"][name]
    rng = _case_rng(seed, "versions", case_id)
    # Every uniform is drawn in a fixed order so the stream never depends on outcomes.
    u_device, u_auth, u_account, u_transaction = (rng.random() for _ in range(4))
    return {"device": "new" if u_device < params["p_new_device"] else "known",
            "auth": ("failure" if u_auth < params["p_auth_failure"] else "success") if skeleton["auth_k"] else None,
            "account": ("risk" if u_account < params["p_new_beneficiary"] else "benign") if skeleton["account"] else None,
            "transaction": ("high_risk" if u_transaction < params["p_high_risk_transaction"] else "low_risk")
            if skeleton["transaction"] else None}


def _versions_from_latent(risky: dict[str, bool], skeleton: dict[str, Any]) -> dict[str, Any]:
    """Map latent risk bits to the versions of slots present in the skeleton."""
    return {"device": "new" if risky["device"] else "known",
            "auth": ("failure" if risky["auth"] else "success") if skeleton["auth_k"] else None,
            "account": ("risk" if risky["account"] else "benign") if skeleton["account"] else None,
            "transaction": ("high_risk" if risky["transaction"] else "low_risk") if skeleton["transaction"] else None}


def _token_event(token: str, family: dict[str, Any], manifest: dict[str, Any],
                 versions: dict[str, Any]) -> str:
    if token == "device":
        return family["device_event"] if versions["device"] == "new" else family["auth_success_event"]
    if token == "auth":
        return family["auth_failure_event"] if versions["auth"] == "failure" else family["auth_success_event"]
    if token == "login":
        return family["auth_success_event"]
    if token == "recovery":
        return family["recovery_event"]
    if token == "account":
        return manifest["risk_account_event"] if versions["account"] == "risk" else manifest["benign_account_event"]
    return (family["high_risk_transaction_event"] if versions["transaction"] == "high_risk"
            else manifest["low_risk_transaction_event"])


def _case_plans(mode: str, manifest: dict[str, Any], seed: int, label_seed: int) -> list[dict[str, Any]]:
    if mode not in {"negative_control", "predictive"}:
        raise ValueError("mode must be 'negative_control' or 'predictive'")
    assumptions = manifest["design_assumptions"]
    anchor_skeleton = {k: manifest["anchor_skeleton"][k] for k in ("auth_k", "recovery", "account", "transaction")}
    anchors_per_label = assumptions["anchor_cases_per_label_per_family"]
    plans: list[dict[str, Any]] = []
    for family in manifest["families"]:
        family_id = family["template_family_id"]
        case_ids = [f"V3-{mode.upper()}-{family_id}-{slot + 1:02d}" for slot in range(CASES_PER_FAMILY)]
        skeletons = _family_skeletons(manifest, seed, family_id)
        # Labels come from label_seed only; in negative_control nothing below reads them.
        labels = [0] * CASES_PER_LABEL_PER_FAMILY + [1] * CASES_PER_LABEL_PER_FAMILY
        _case_rng(label_seed, "labels", family_id).shuffle(labels)
        label_by_case = dict(zip(case_ids, labels))
        assignment: dict[str, tuple[str, dict[str, Any], int | None]] = {}
        if mode == "predictive":
            for label in (0, 1):
                members = sorted(cid for cid in case_ids if label_by_case[cid] == label)
                rng = _case_rng(seed, "assign", f"{family_id}:{label}")
                rng.shuffle(members)
                intent = "suspicious" if label else "legitimate"
                for cid in members[:anchors_per_label]:
                    assignment[cid] = ("anchor", anchor_skeleton, None)
                for index, cid in enumerate(members[anchors_per_label:]):
                    assignment[cid] = (intent, skeletons[index], index)
        else:
            slots = [("anchor", anchor_skeleton, None)] * assumptions["negative_control_recipe_sources_per_family"]["anchor"]
            slots += [(source, skeleton, index) for index, skeleton in enumerate(skeletons)
                      for source in ("profile_A", "profile_B")]
            _case_rng(seed, "assign", family_id).shuffle(slots)
            assignment = dict(zip(case_ids, slots))
        for case_id in case_ids:
            label = label_by_case[case_id]
            source, skeleton, skeleton_index = assignment[case_id]
            versions = _versions(manifest, source, skeleton, seed, case_id)
            order = _slot_order(skeleton, seed, case_id)
            components = {"skeleton": skeleton, "skeleton_index": skeleton_index,
                          "versions": versions, "slot_order": order}
            if manifest.get("version_model", "independent") == "coordinated_mixture" and source != "anchor":
                # Evaluation-only record of the latent draw; components_json never enters model inputs.
                name = assumptions["negative_control_profile_parameters"].get(source, source)
                latent = _latent_draw(manifest, name, seed, case_id)
                components["latent_risky"] = latent["risky"]
                components["coordinated"] = latent["coordinated"]
            plans.append({
                "case_id": case_id, "template_family_id": family_id, "ato_label": label,
                "scenario_type": "SUSPICIOUS_SIM_REPLACEMENT" if label else "LEGITIMATE_SIM_REPLACEMENT",
                "declared_scenario_intent": ("suspicious" if label else "legitimate") if mode == "predictive" else None,
                "recipe_source": source, "is_anchor": int(source == "anchor"), "components": components,
                "event_names": [_token_event(t, family, manifest, versions) for t in order],
            })
    return sorted(plans, key=lambda plan: plan["case_id"])


def _window_transaction_pool(history: list[dict[str, Any]], paysim_path: str | Path,
                             seed: int) -> list[dict[str, str]]:
    """Sample in-window amounts disjoint from history rows, independent of label and type."""
    used = {(row["paysim_step"], row["paysim_nameOrig"], row["paysim_nameDest"], row["amount"])
            for row in history if row["event_category"] == "TRANSACTION"}
    sample = sample_transactions(paysim_path, TOTAL_CASES + len(used), seed + WINDOW_PAYSIM_SEED_OFFSET)
    pool, seen = [], set(used)
    for row in sample:
        key = (row["step"], row["nameOrig"], row["nameDest"], row["amount"])
        if key not in seen:
            seen.add(key)
            pool.append(row)
    if len(pool) < TOTAL_CASES:
        raise ValueError("not enough PaySim rows disjoint from history for in-window transactions")
    return pool[:TOTAL_CASES]


def _generate_events(plans: list[dict[str, Any]], manifest: dict[str, Any], seed: int,
                     paysim_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    families = {family["template_family_id"]: family for family in manifest["families"]}
    categories = manifest["event_categories"]
    trigger_context: list[dict[str, Any]] = []
    for case_index, plan in enumerate(plans):
        identity = hashlib.sha256(plan["case_id"].encode("utf-8")).hexdigest()[:12].upper()
        trigger_context.append({
            "case_id": plan["case_id"], "template_family_id": plan["template_family_id"],
            "user_id": f"USR-SIMREP3-{identity}", "account_id": f"ACC-SIMREP3-{identity}",
            "trigger_event_id": f"EVT-SIMREP3-{identity}-00",
            "trigger_time": datetime(2026, 1, 1) + timedelta(days=case_index),
            "known_device_id": f"DEV-SIMREP3-{identity}-K", "identity": identity})

    history = _build_history(trigger_context, paysim_path, seed)
    for row in history:
        row["event_id"] = row["event_id"].replace("EVT-SIMREP2-HIST-", "EVT-SIMREP3-HIST-", 1)
    pool = _window_transaction_pool(history, paysim_path, seed)
    ranked = sorted(plans, key=lambda plan: _stable_rank(seed, f"window_amount:{plan['case_id']}"))
    amount_by_case = {plan["case_id"]: pool[index]["amount"] for index, plan in enumerate(ranked)}

    events: list[dict[str, Any]] = []
    for plan, context in zip(plans, trigger_context):
        trigger_time, identity = context["trigger_time"], context["identity"]
        family = families[plan["template_family_id"]]
        device = (f"DEV-SIMREP3-{identity}-N" if plan["components"]["versions"]["device"] == "new"
                  else context["known_device_id"])
        events.append(make_event(event_id=context["trigger_event_id"], user_id=context["user_id"],
                                 account_id=context["account_id"], event_category="TELECOM",
                                 event_type="SIM_REPLACEMENT", timestamp=trigger_time,
                                 device_id=context["known_device_id"], scenario_type=None,
                                 ato_label=None, status="SUCCESS"))
        names = plan["event_names"]
        # Offsets depend on seed, case_id and the label-neutral event count only.
        offsets = sorted(_case_rng(seed, "timing", plan["case_id"]).sample(
            range(1, OBSERVATION_MINUTES * 60 + 1), len(names)))
        for index, (name, offset) in enumerate(zip(names, offsets), start=1):
            category = categories[name]
            is_transaction = category == "TRANSACTION"
            row = make_event(
                event_id=f"EVT-SIMREP3-{identity}-{index:02d}", user_id=context["user_id"],
                account_id=context["account_id"], event_category=category, event_type=name,
                timestamp=trigger_time + timedelta(seconds=offset), device_id=device,
                scenario_type=None, ato_label=None,
                status="FAILURE" if name == family["auth_failure_event"] else "SUCCESS",
                amount=amount_by_case[plan["case_id"]] if is_transaction else None,
            )
            if is_transaction:
                # Only the PaySim amount is used; the type follows the drawn event, so
                # PaySim's own type (which correlates with amount) never enters the row.
                row["transaction_type"] = manifest["transaction_type_mapping"][name]
            events.append(row)
    return events, history, trigger_context


def generate_experiment(mode: str, *, output_dir: str | Path,
                        paysim_path: str | Path = DEFAULT_PAYSIM_PATH,
                        seed: int = 42, label_seed: int | None = None,
                        manifest_path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Generate one v3 cohort into its own directory and return its artifacts."""
    label_seed = seed if label_seed is None else label_seed
    manifest = _load_manifest(manifest_path)
    plans = _case_plans(mode, manifest, seed, label_seed)
    events, history, trigger_context = _generate_events(plans, manifest, seed, paysim_path)
    triggers = [row for row in events if row["event_category"] == "TELECOM"]
    baseline_input = history + triggers
    profiles = build_baseline_profiles(baseline_input)
    cutoff_errors = validate_baseline_cutoffs(baseline_input, profiles)
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))

    # Labels are attached only after events, history and profiles exist.
    context_by_case = {row["case_id"]: row for row in trigger_context}
    plan_by_user = {context_by_case[plan["case_id"]]["user_id"]: plan for plan in plans}
    for row in events:
        plan = plan_by_user[row["user_id"]]
        row["scenario_type"], row["ato_label"] = plan["scenario_type"], plan["ato_label"]
    cases = [{"case_id": plan["case_id"], "template_family_id": plan["template_family_id"],
              "user_id": context_by_case[plan["case_id"]]["user_id"],
              "account_id": context_by_case[plan["case_id"]]["account_id"],
              "event_id": context_by_case[plan["case_id"]]["trigger_event_id"],
              "scenario_type": plan["scenario_type"], "ato_label": plan["ato_label"],
              "declared_scenario_intent": plan["declared_scenario_intent"],
              "recipe_source": plan["recipe_source"], "is_anchor": plan["is_anchor"],
              "components_json": json.dumps(plan["components"], sort_keys=True)}
             for plan in plans]
    features = build_feature_rows(events, profiles)
    result = {"mode": mode, "manifest": manifest, "cases": cases, "events": events,
              "historical_event_input": history, "baseline_input": baseline_input,
              "baseline_profiles": profiles, "features": features,
              "baseline_feature_audit": _baseline_audit(features, cases)}
    validate_experiment(result)
    result["design_audit"] = design_audit(result)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "template_manifest.json").write_text(
        Path(manifest_path).read_text(encoding="utf-8"), encoding="utf-8")
    _write_csv(output / "case_metadata.csv", cases, CASE_FIELDS)
    _write_csv(output / "events.csv", events, EVENT_FIELDS)
    _write_csv(output / "historical_event_input.csv", history, EVENT_FIELDS)
    _write_csv(output / "baseline_profiles.csv", profiles, BASELINE_FIELDS)
    _write_csv(output / "features.csv", features, OUTPUT_COLUMNS)
    _write_csv(output / "baseline_feature_audit.csv", result["baseline_feature_audit"],
               ("template_family_id", "ato_label", "feature", "count", "missing_count",
                "min", "mean", "median", "max"))
    metadata = {"version": "v3", "mode": mode, "seed": seed, "label_seed": label_seed,
                "window_paysim_seed": seed + WINDOW_PAYSIM_SEED_OFFSET,
                "cases": len(cases), "events": len(events), "history_events": len(history),
                "profiles": len(profiles), "feature_rows": len(features),
                "families": FAMILY_COUNT, "cases_per_label_per_family": CASES_PER_LABEL_PER_FAMILY,
                "inclusive_prediction_window_minutes": OBSERVATION_MINUTES,
                "feature_columns": list(FEATURE_COLUMNS),
                "evaluation_only_columns": sorted(EVALUATION_ONLY_COLUMNS),
                "design_assumptions": manifest["design_assumptions"],
                "ordering_rules": manifest["ordering_rules"],
                "missing_value_policy": MISSING_VALUE_POLICY,
                "pay_sim_fraud_fields_used_as_target_or_feature": False,
                "evaluation_metadata_is_model_input": False,
                "interpretation": INTERPRETATION}
    if manifest.get("version_model", "independent") != "independent":
        metadata["version_model"] = manifest["version_model"]
        metadata["interaction_model"] = manifest["interaction_model"]
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "design_audit.json").write_text(
        json.dumps(result["design_audit"], indent=2, default=str) + "\n", encoding="utf-8")
    return result


def _cell_signature(cases: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    components = [json.loads(c["components_json"]) for c in cases]
    return {"skeletons": sorted(json.dumps(c["skeleton"], sort_keys=True) for c in components),
            "event_counts": sorted(counts[c["user_id"]] for c in cases),
            "anchors": sum(c["is_anchor"] for c in cases)}


def validate_experiment(result: dict[str, Any]) -> None:
    """Validate schema, window, ordering, joins, skeleton equivalence and missing values."""
    cases, events, manifest = result["cases"], result["events"], result["manifest"]
    assumptions = manifest["design_assumptions"]
    if len(cases) != TOTAL_CASES:
        raise ValueError(f"expected {TOTAL_CASES} cases, got {len(cases)}")
    for key in ("case_id", "user_id", "account_id", "event_id"):
        if len({row[key] for row in cases}) != len(cases):
            raise ValueError(f"duplicate {key} in case metadata")
    if len({row["event_id"] for row in events}) != len(events):
        raise ValueError("duplicate event_id")
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if tuple(row.keys()) != EVENT_FIELDS:
            raise ValueError("event rows differ from canonical EVENT_FIELDS")
        if row["event_type"] not in CATEGORY_EVENT_TYPES.get(row["event_category"], ()):
            raise ValueError(f"invalid event_type {row['event_category']}/{row['event_type']}")
        if row["event_category"] == "TRANSACTION":
            try:
                Decimal(str(row["amount"]))
            except InvalidOperation as error:
                raise ValueError(f"invalid transaction amount in {row['event_id']}") from error
            if row["transaction_type"] not in PAYSIM_TRANSACTION_TYPES:
                raise ValueError(f"invalid transaction_type in {row['event_id']}")
        by_user[row["user_id"]].append(row)
    if set(by_user) != {row["user_id"] for row in cases}:
        raise ValueError("event users and case metadata users differ")

    window_counts: dict[str, int] = {}
    anchors: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    family_labels: dict[str, Counter[int]] = defaultdict(Counter)
    for case in cases:
        family, label = case["template_family_id"], int(case["ato_label"])
        family_labels[family][label] += 1
        rows = by_user[case["user_id"]]
        if any(row["account_id"] != case["account_id"] or row["ato_label"] != label
               or row["scenario_type"] != case["scenario_type"] for row in rows):
            raise ValueError(f"account/label join mismatch for {case['case_id']}")
        triggers = [row for row in rows if row["event_category"] == "TELECOM"]
        if len(triggers) != 1 or triggers[0]["event_type"] != "SIM_REPLACEMENT":
            raise ValueError(f"invalid SIM_REPLACEMENT trigger in {case['case_id']}")
        trigger_time = datetime.fromisoformat(triggers[0]["timestamp"])
        window = sorted((row for row in rows if row["event_category"] != "TELECOM"),
                        key=lambda row: (row["timestamp"], row["event_id"]))
        times = [datetime.fromisoformat(row["timestamp"]) for row in window]
        if [r["event_id"] for r in window] != sorted(r["event_id"] for r in window) or any(
                t <= trigger_time or t > trigger_time + timedelta(minutes=OBSERVATION_MINUTES) for t in times):
            raise ValueError(f"non-chronological or out-of-window events in {case['case_id']}")
        components = json.loads(case["components_json"])
        order = components["slot_order"]
        if not _order_valid(order) or sorted(order) != sorted(_slot_tokens(components["skeleton"])) \
                or len(order) != len(window):
            raise ValueError(f"invalid slot order in {case['case_id']}")
        window_counts[case["user_id"]] = len(window)
        if result["mode"] == "predictive":
            if case["declared_scenario_intent"] != ("suspicious" if label else "legitimate"):
                raise ValueError(f"label does not match declared intent in {case['case_id']}")
        elif case["declared_scenario_intent"] is not None or case["recipe_source"] not in \
                assumptions["negative_control_recipe_sources_per_family"]:
            raise ValueError(f"negative-control case carries an intent in {case['case_id']}")
        if case["is_anchor"]:
            anchors[family].add(tuple(sorted(row["event_type"] for row in window)))
    if len(family_labels) != FAMILY_COUNT or any(
            c != Counter({0: CASES_PER_LABEL_PER_FAMILY, 1: CASES_PER_LABEL_PER_FAMILY})
            for c in family_labels.values()):
        raise ValueError("each family must contain exactly 10 cases per label")
    if len(anchors) != FAMILY_COUNT or any(len(recipes) != 1 for recipes in anchors.values()):
        raise ValueError("anchor cases within a family must share one event-type multiset")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[case["template_family_id"]].append(case)
    for family, members in by_family.items():
        if result["mode"] == "predictive":
            cells = [_cell_signature([c for c in members if c["ato_label"] == label], window_counts)
                     for label in (0, 1)]
            if cells[0] != cells[1] or cells[0]["anchors"] != assumptions["anchor_cases_per_label_per_family"]:
                raise ValueError(f"skeletons, event counts or anchors differ by label in {family}")
        elif sum(c["is_anchor"] for c in members) != assumptions["negative_control_recipe_sources_per_family"]["anchor"]:
            raise ValueError(f"anchor quota violated in {family}")

    if any(row["scenario_type"] or row["ato_label"] is not None for row in result["historical_event_input"]):
        raise ValueError("history rows carry labels")
    cutoff_errors = validate_baseline_cutoffs(result["baseline_input"], result["baseline_profiles"])
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))
    case_by_user = {row["user_id"]: row for row in cases}
    features = result["features"]
    if len(features) != TOTAL_CASES or {row["user_id"] for row in features} != set(case_by_user):
        raise ValueError("feature rows do not join one-to-one with metadata")
    for row in features:
        if set(row) != set(OUTPUT_COLUMNS) or EVALUATION_ONLY_COLUMNS.intersection(row):
            raise ValueError("feature rows differ from OUTPUT_COLUMNS or carry evaluation metadata")
        if row["ato_label"] != case_by_user[row["user_id"]]["ato_label"]:
            raise ValueError("feature target does not align with case label")
        if row["transaction_count"] == 0:
            if any(row[c] not in (None, "") for c in
                   ("transaction_amount", "transaction_type", "transaction_deviation")):
                raise ValueError("transaction features must stay missing when no transaction is observed")
        elif row["transaction_amount"] is None or not row["transaction_type"]:
            raise ValueError("observed transaction is missing amount or type")


def _rate_table(values_by_label: dict[int, list[Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for label in (0, 1):
        values = values_by_label.get(label, [])
        table[f"label_{label}"] = {"n": len(values), "counts": dict(sorted(Counter(map(str, values)).items()))}
    return table


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": mean(values), "median": median(values),
            "min": min(values), "max": max(values)}


def _lofo_lookup(keys: dict[str, Any], labels: dict[str, int], families: dict[str, str]) -> dict[str, Any]:
    """Leave-one-family-out majority-label lookup; unseen or tied keys abstain."""
    per_family: dict[str, Any] = {}
    correct = covered = total = 0
    for held_out in sorted(set(families.values())):
        votes: dict[Any, Counter[int]] = defaultdict(Counter)
        for user, key in keys.items():
            if families[user] != held_out:
                votes[key][labels[user]] += 1
        f_correct = f_covered = f_total = 0
        for user, key in keys.items():
            if families[user] != held_out:
                continue
            f_total += 1
            counts = votes.get(key)
            if not counts or counts[0] == counts[1]:
                continue
            f_covered += 1
            f_correct += int((counts[1] > counts[0]) == bool(labels[user]))
        per_family[held_out] = {"cases": f_total, "covered": f_covered, "correct": f_correct}
        correct, covered, total = correct + f_correct, covered + f_covered, total + f_total
    return {"pooled": {"cases": total, "covered": covered, "correct": correct,
                       "accuracy_on_covered": correct / covered if covered else None,
                       "accuracy_abstain_as_half": (correct + 0.5 * (total - covered)) / total},
            "per_family": per_family}


def design_audit(result: dict[str, Any]) -> dict[str, Any]:
    """Report-first diagnostics of overlap, shortcuts, ordering, coverage and similarity."""
    manifest, cases, features = result["manifest"], result["cases"], result["features"]
    labels = {c["user_id"]: int(c["ato_label"]) for c in cases}
    families = {c["user_id"]: c["template_family_id"] for c in cases}
    components = {c["user_id"]: json.loads(c["components_json"]) for c in cases}
    feature_by_user = {row["user_id"]: row for row in features}
    vectors = {u: tuple(str(feature_by_user[u][c]) for c in WINDOW_DISCRETE_FEATURES) for u in labels}

    overlap: dict[str, Any] = {}
    for family in sorted(set(families.values())):
        users = [u for u in labels if families[u] == family]
        by_vector: dict[tuple[str, ...], Counter[int]] = defaultdict(Counter)
        for u in users:
            by_vector[vectors[u]][labels[u]] += 1
        ambiguous = sum(1 for u in users if len(by_vector[vectors[u]]) == 2)
        overlap[family] = {"distinct_vectors": len(by_vector),
                           "ambiguous_share": ambiguous / len(users),
                           "within_family_lookup_ceiling": sum(max(v.values()) for v in by_vector.values()) / len(users),
                           "perfectly_separable": all(len(v) == 1 for v in by_vector.values())}

    window_rows = defaultdict(list)
    trigger_time: dict[str, datetime] = {}
    for row in result["events"]:
        if row["event_category"] == "TELECOM":
            trigger_time[row["user_id"]] = datetime.fromisoformat(row["timestamp"])
        else:
            window_rows[row["user_id"]].append(row)

    def _relative_position(u: str, token: str) -> float | None:
        order = components[u]["slot_order"]
        return order.index(token) / (len(order) - 1) if token in order else None

    derived = {
        "transaction_present": {u: int(components[u]["skeleton"]["transaction"]) for u in labels},
        "transaction_type": {u: feature_by_user[u]["transaction_type"] or "MISSING" for u in labels},
        "new_device": {u: int(components[u]["versions"]["device"] == "new") for u in labels},
        "auth_slot_present": {u: int(components[u]["skeleton"]["auth_k"] > 0) for u in labels},
        "auth_failure_version": {u: int(components[u]["versions"]["auth"] == "failure") for u in labels},
        "failed_login_count": {u: feature_by_user[u]["failed_login_count"] for u in labels},
        "account_event_present": {u: int(components[u]["skeleton"]["account"]) for u in labels},
        "new_beneficiary": {u: feature_by_user[u]["new_beneficiary"] for u in labels},
        "recovery_present": {u: int(components[u]["skeleton"]["recovery"]) for u in labels},
        "window_event_count": {u: len(window_rows[u]) for u in labels},
        "is_anchor": {c["user_id"]: c["is_anchor"] for c in cases},
        "second_slot_token": {u: components[u]["slot_order"][1] for u in labels},
    }
    component_diagnostics: dict[str, Any] = {}
    for name, values in derived.items():
        by_label: dict[int, list[Any]] = defaultdict(list)
        for u, value in values.items():
            by_label[labels[u]].append(value)
        component_diagnostics[name] = {**_rate_table(by_label),
                                       "single_feature_lofo_lookup": _lofo_lookup(values, labels, families)["pooled"]}

    def _numeric_by_label(getter) -> dict[str, Any]:
        return {f"label_{label}": _numeric_summary(
            [v for u in labels if labels[u] == label and (v := getter(u)) is not None]) for label in (0, 1)}

    def _transaction_offset(u: str) -> float | None:
        rows = [r for r in window_rows[u] if r["event_category"] == "TRANSACTION"]
        return (datetime.fromisoformat(rows[0]["timestamp"]) - trigger_time[u]).total_seconds() if rows else None

    def _last_offset(u: str) -> float:
        return max((datetime.fromisoformat(r["timestamp"]) - trigger_time[u]).total_seconds()
                   for r in window_rows[u])

    amount_by_type: dict[str, list[float]] = defaultdict(list)
    for u in labels:
        if feature_by_user[u]["transaction_amount"] is not None:
            amount_by_type[feature_by_user[u]["transaction_type"]].append(float(feature_by_user[u]["transaction_amount"]))
    numeric_diagnostics = {
        "transaction_amount": _numeric_by_label(
            lambda u: float(a) if (a := feature_by_user[u]["transaction_amount"]) is not None else None),
        "transaction_deviation": _numeric_by_label(
            lambda u: float(d) if (d := feature_by_user[u]["transaction_deviation"]) is not None else None),
        "transaction_offset_seconds": _numeric_by_label(_transaction_offset),
        "transaction_relative_position": _numeric_by_label(lambda u: _relative_position(u, "transaction")),
        "account_relative_position": _numeric_by_label(lambda u: _relative_position(u, "account")),
        "last_event_offset_seconds": _numeric_by_label(_last_offset),
        "transaction_amount_by_type": {t: _numeric_summary(v) for t, v in sorted(amount_by_type.items())},
    }

    coverage: dict[str, Any] = {}
    cells_lacking: list[str] = []
    for family in sorted(set(families.values())):
        entry = {}
        for label in (0, 1):
            users = [u for u in labels if families[u] == family and labels[u] == label]
            cell = {"cases": len(users)}
            for slot in ("recovery_present", "transaction_present", "account_event_present", "auth_slot_present"):
                cell[slot] = sum(derived[slot][u] for u in users)
                if cell[slot] in (0, len(users)):
                    cells_lacking.append(f"{family}/label_{label}/{slot}")
            cell["event_counts"] = dict(sorted(Counter(derived["window_event_count"][u] for u in users).items()))
            entry[f"label_{label}"] = cell
        coverage[family] = entry

    specs = {f["template_family_id"]: {f[slot] for slot in FAMILY_SLOTS} for f in manifest["families"]}
    similarity = {}
    for family, options in specs.items():
        others = {other: len(options & o) / len(options | o) for other, o in specs.items() if other != family}
        nearest = max(others, key=others.get)
        similarity[family] = {"max_jaccard": others[nearest], "nearest_family": nearest}
    cross_family = {}
    for family in sorted(set(families.values())):
        elsewhere = {vectors[u] for u in labels if families[u] != family}
        own = [vectors[u] for u in labels if families[u] == family]
        cross_family[family] = sum(v in elsewhere for v in own) / len(own)

    all_values = {c: {str(row[c]) for row in features} for c in FEATURE_COLUMNS}
    extra: dict[str, Any] = {}
    if manifest.get("version_model", "independent") != "independent":
        extra["interaction_diagnostics"] = _interaction_diagnostics(cases)
    return {
        "mode": result["mode"], "interpretation": INTERPRETATION,
        "design_assumptions": manifest["design_assumptions"],
        "ordering_rules": manifest["ordering_rules"],
        "missing_value_policy": MISSING_VALUE_POLICY,
        "lookup_feature_columns": list(WINDOW_DISCRETE_FEATURES),
        "feature_summary": {
            "all_missing": [c for c in FEATURE_COLUMNS if all_values[c] <= {"None", ""}],
            "constant": [c for c in FEATURE_COLUMNS if len(all_values[c]) == 1 and not all_values[c] <= {"None", ""}],
            "missing_counts": {c: sum(row[c] is None for row in features) for c in FEATURE_COLUMNS},
        },
        "overlap": {"per_family": overlap,
                    "families_perfectly_separable": sum(v["perfectly_separable"] for v in overlap.values()),
                    "mean_ambiguous_share": mean(v["ambiguous_share"] for v in overlap.values())},
        "lofo_lookup_baseline": _lofo_lookup(vectors, labels, families),
        "component_diagnostics": component_diagnostics,
        "numeric_diagnostics": numeric_diagnostics,
        "family_coverage": coverage,
        "family_label_cells_lacking_variation": cells_lacking,
        "family_similarity": similarity,
        "cross_family_vector_overlap": cross_family,
        **extra,
    }


def _interaction_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Finite-sample latent rates and co-occurrence by label (report only; nothing is forced)."""
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family_label: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        components = json.loads(case["components_json"])
        if "latent_risky" in components:
            by_label[int(case["ato_label"])].append(components)
            by_family_label[(case["template_family_id"], int(case["ato_label"]))].append(components)

    def _rates(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        return {"n": n, **{slot: sum(c["latent_risky"][slot] for c in group) / n if n else None
                           for slot in INTERACTION_SLOTS}}

    pairs = [(a, b) for i, a in enumerate(INTERACTION_SLOTS) for b in INTERACTION_SLOTS[i + 1:]]
    pooled = {f"label_{label}": _rates(by_label[label]) for label in (0, 1)}
    differences = {slot: pooled["label_1"][slot] - pooled["label_0"][slot]
                   for slot in INTERACTION_SLOTS if pooled["label_0"]["n"] and pooled["label_1"]["n"]}
    families = sorted({family for family, _ in by_family_label})
    per_family = {family: {f"label_{label}": _rates(by_family_label[(family, label)]) for label in (0, 1)}
                  for family in families}
    family_gaps = {slot: [per_family[f]["label_1"][slot] - per_family[f]["label_0"][slot] for f in families
                          if per_family[f]["label_0"]["n"] and per_family[f]["label_1"]["n"]]
                   for slot in INTERACTION_SLOTS}
    return {
        "kind": "finite-sample latent version diagnostics (variant cases only; anchors excluded)",
        "pooled_latent_risky_rates": pooled,
        "pooled_rate_difference_label1_minus_label0": differences,
        "per_family_rate_difference_range": {slot: {"min": min(v), "max": max(v)} if v else None
                                             for slot, v in family_gaps.items()},
        "per_family_latent_risky_rates": per_family,
        "pairwise_cooccurrence_by_label": {
            f"{a}&{b}": {f"label_{label}": (sum(c["latent_risky"][a] and c["latent_risky"][b] for c in by_label[label])
                                            / len(by_label[label])) if by_label[label] else None
                         for label in (0, 1)} for a, b in pairs},
        "all_four_risky_by_label": {
            f"label_{label}": (sum(all(c["latent_risky"].values()) for c in by_label[label]) / len(by_label[label]))
            if by_label[label] else None for label in (0, 1)},
        "coordinated_component_count_by_label": {f"label_{label}": sum(c["coordinated"] for c in by_label[label])
                                                 for label in (0, 1)},
    }
