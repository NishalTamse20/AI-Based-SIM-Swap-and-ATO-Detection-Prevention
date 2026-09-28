"""Bounded-memory sampling of financial transaction rows from PaySim."""

import csv
import random
from pathlib import Path


PAYSIM_FIELDS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)


def sample_transactions(path: str | Path, sample_size: int, seed: int) -> list[dict[str, str]]:
    """Reservoir-sample rows without loading or modifying the raw CSV."""
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    if sample_size == 0:
        return []

    reservoir: list[dict[str, str]] = []
    rng = random.Random(seed)
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = set(PAYSIM_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PaySim CSV is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader):
            if row_number < sample_size:
                reservoir.append({field: row[field] for field in PAYSIM_FIELDS})
            else:
                replacement_index = rng.randrange(row_number + 1)
                if replacement_index < sample_size:
                    reservoir[replacement_index] = {field: row[field] for field in PAYSIM_FIELDS}
    if len(reservoir) < sample_size:
        raise ValueError(f"Requested {sample_size} PaySim rows, but only found {len(reservoir)}")
    return reservoir
