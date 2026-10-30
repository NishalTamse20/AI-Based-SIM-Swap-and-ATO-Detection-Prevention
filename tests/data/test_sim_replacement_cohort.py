from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from simulator.generators.sim_replacement_cohort import (
    generate_sim_replacement_cohort,
    validate_cohort,
)
from simulator.generators.event_generator import EVENT_FIELDS


def test_manifest_families_contain_both_declared_labels_and_recovery_overlap():
    cohort = generate_sim_replacement_cohort()
    validate_cohort(cohort)
    assert all(tuple(row) == EVENT_FIELDS for row in cohort["events"])
    by_family = {}
    by_user = {}
    for case in cohort["cases"]:
        by_family.setdefault(case["template_family_id"], set()).add(case["ato_label"])
        by_user[case["user_id"]] = case
    assert by_family and all(labels == {0, 1} for labels in by_family.values())
    for case in cohort["cases"]:
        rows = [row for row in cohort["events"] if row["user_id"] == case["user_id"]]
        types = {(row["event_category"], row["event_type"]) for row in rows}
        assert {("DEVICE", "NEW_DEVICE"), ("AUTHENTICATION", "FAILED_LOGIN"),
                ("RECOVERY", "PASSWORD_RESET")} <= types
        assert all(row["scenario_type"] == case["scenario_type"] for row in rows)
        assert all(row["ato_label"] == case["ato_label"] for row in rows)


def test_case_and_event_identifiers_are_unique():
    cohort = generate_sim_replacement_cohort()
    assert len({case["case_id"] for case in cohort["cases"]}) == len(cohort["cases"])
    assert len({row["event_id"] for row in cohort["events"]}) == len(cohort["events"])


def test_timestamps_are_valid_chronological_and_within_inclusive_t_plus_15():
    cohort = generate_sim_replacement_cohort()
    grouped = {}
    for row in cohort["events"]:
        datetime.fromisoformat(row["timestamp"])
        grouped.setdefault(row["user_id"], []).append(row)
    for rows in grouped.values():
        times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        assert times == sorted(times)
        trigger = times[0]
        assert all(trigger <= value <= trigger + timedelta(minutes=15) for value in times)
    assert any((datetime.fromisoformat(row["timestamp"]) - trigger).total_seconds() == 900
               for rows in grouped.values()
               for trigger in [datetime.fromisoformat(rows[0]["timestamp"])]
               for row in rows)


def test_exact_normalized_event_signatures_cannot_cross_families():
    cohort = generate_sim_replacement_cohort()
    duplicate = deepcopy(cohort)
    source_case = next(case for case in duplicate["cases"]
                       if case["template_family_id"] == "SIM_REPLACEMENT_FAMILY_02")
    other_case = next(case for case in duplicate["cases"]
                      if case["template_family_id"] == "SIM_REPLACEMENT_FAMILY_03"
                      and case["ato_label"] == source_case["ato_label"])
    source_rows = [row for row in duplicate["events"] if row["user_id"] == source_case["user_id"]]
    target_rows = [row for row in duplicate["events"] if row["user_id"] == other_case["user_id"]]
    target_trigger = datetime.fromisoformat(target_rows[0]["timestamp"])
    source_trigger = datetime.fromisoformat(source_rows[0]["timestamp"])
    for target, source in zip(target_rows, source_rows):
        target["event_category"] = source["event_category"]
        target["event_type"] = source["event_type"]
        target["status"] = source["status"]
        source_offset = datetime.fromisoformat(source["timestamp"]) - source_trigger
        target["timestamp"] = (target_trigger + source_offset).isoformat(timespec="seconds")
        target["device_id"] = target_rows[0]["device_id"] if source["device_id"] == source_rows[0]["device_id"] else "DUPLICATE-OTHER"
    with pytest.raises(ValueError, match="exact normalized event signature"):
        validate_cohort(duplicate)


def test_preflight_rejects_missing_class_and_out_of_window_event():
    cohort = generate_sim_replacement_cohort()
    missing_class = deepcopy(cohort)
    missing_class["cases"] = [case for case in missing_class["cases"]
                              if not (case["template_family_id"] == "SIM_REPLACEMENT_FAMILY_01"
                                      and case["ato_label"] == 1)]
    missing_user = missing_class["cases"][-1]["user_id"]
    missing_class["events"] = [row for row in missing_class["events"] if row["user_id"] != missing_user]
    with pytest.raises(ValueError, match="both ato_label values"):
        validate_cohort(missing_class)

    late = deepcopy(cohort)
    late["events"][-1]["timestamp"] = (datetime.fromisoformat(late["events"][-1]["timestamp"])
                                        + timedelta(minutes=1)).isoformat(timespec="seconds")
    with pytest.raises(ValueError, match=r"exceeds inclusive T\+15"):
        validate_cohort(late)

