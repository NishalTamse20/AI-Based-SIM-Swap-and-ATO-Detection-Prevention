import csv

import pytest

from simulator.generators.paysim_loader import PAYSIM_FIELDS


@pytest.fixture
def paysim_fixture(tmp_path):
    path = tmp_path / "paysim.csv"
    rows = [
        {"step": str(index + 1), "type": "TRANSFER", "amount": str(100 + index),
         "nameOrig": f"C{index}", "oldbalanceOrg": "500", "newbalanceOrig": "400",
         "nameDest": f"M{index}", "oldbalanceDest": "0", "newbalanceDest": "100",
         "isFraud": "0", "isFlaggedFraud": "0"}
        for index in range(100)
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
