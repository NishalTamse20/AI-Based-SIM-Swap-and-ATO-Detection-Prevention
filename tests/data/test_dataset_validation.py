import csv

from simulator.config.generator_config import GeneratorConfig
from simulator.generators.dataset_generator import generate_dataset
from simulator.validation.dataset_validator import validate_dataset


def _make_dataset(tmp_path, paysim_fixture):
    path = tmp_path / "events.csv"
    generate_dataset(GeneratorConfig(8, 42, paysim_fixture, path))
    return path


def test_generated_dataset_passes_validation(tmp_path, paysim_fixture):
    path = _make_dataset(tmp_path, paysim_fixture)
    result = validate_dataset(path)
    assert result.is_valid, result.errors
    assert result.event_count > 0
    assert result.user_count == 8
    assert result.scenario_count == 8
    assert result.history_event_count == 80


def test_validation_detects_duplicate_id_invalid_type_and_missing_required_value(tmp_path, paysim_fixture):
    path = _make_dataset(tmp_path, paysim_fixture)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    rows[1]["event_id"] = rows[0]["event_id"]
    rows[2]["event_type"] = "NUMBER_PORT"
    rows[3]["device_id"] = ""
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = validate_dataset(path)
    assert not result.is_valid
    assert any("duplicate event_id" in error for error in result.errors)
    assert any("invalid event_type" in error for error in result.errors)
    assert any("missing required values" in error for error in result.errors)


def test_validation_rejects_labelled_pretrigger_history(tmp_path, paysim_fixture):
    path = _make_dataset(tmp_path, paysim_fixture)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    history = next(row for row in rows if not row["scenario_type"])
    history["scenario_type"] = "LEGITIMATE_SIM_REPLACEMENT"
    history["ato_label"] = "0"
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = validate_dataset(path)
    assert any("pre-trigger history contains scenario/ATO labels" in error for error in result.errors)


def test_validation_detects_missing_column_and_invalid_timestamp(tmp_path, paysim_fixture):
    path = _make_dataset(tmp_path, paysim_fixture)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0]["timestamp"] = "not-a-timestamp"
    reduced_fields = [field for field in fields if field != "event_id"]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=reduced_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = validate_dataset(path)
    assert any("missing required columns: event_id" in error for error in result.errors)
    assert any("invalid timestamp" in error for error in result.errors)
