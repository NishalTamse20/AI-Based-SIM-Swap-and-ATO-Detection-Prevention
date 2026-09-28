import csv

from simulator.generators.paysim_loader import PAYSIM_FIELDS, sample_transactions


def test_loader_samples_source_fields_without_using_flags_as_target(tmp_path):
    source = tmp_path / "paysim.csv"
    rows = [
        dict(step=str(i), type="TRANSFER", amount="12.50", nameOrig=f"C{i}", oldbalanceOrg="20",
             newbalanceOrig="7.5", nameDest=f"M{i}", oldbalanceDest="0", newbalanceDest="12.5",
             isFraud=str(i % 2), isFlaggedFraud=str((i + 1) % 2))
        for i in range(4)
    ]
    with source.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    sample = sample_transactions(source, 3, seed=42)
    assert len(sample) == 3
    assert set(sample[0]) == set(PAYSIM_FIELDS)
    assert "ato_label" not in sample[0]
    assert source.read_bytes() == before


def test_loader_rejects_request_larger_than_input(tmp_path):
    import pytest

    source = tmp_path / "paysim.csv"
    with source.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
    with pytest.raises(ValueError, match="only found 0"):
        sample_transactions(source, 1, seed=42)
