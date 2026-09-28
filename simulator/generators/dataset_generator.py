"""Generate balanced scenarios and export their events as CSV."""

import csv
import random
from pathlib import Path
from typing import Any

from simulator.config.generator_config import GeneratorConfig
from simulator.generators.event_generator import EVENT_FIELDS
from simulator.generators.paysim_loader import sample_transactions
from simulator.scenarios.scenario_runner import SCENARIOS, SCENARIO_TYPES, TRANSACTION_SCENARIOS, run_scenario


def generate_dataset(config: GeneratorConfig) -> list[dict[str, Any]]:
    """Generate a balanced dataset of chronological scenario event rows."""
    per_scenario = config.scenario_count // len(SCENARIO_TYPES)
    transaction_count = (
        config.scenario_count * config.history_length
        + per_scenario * len(TRANSACTION_SCENARIOS)
    )
    pay_sim_rows = iter(sample_transactions(config.paysim_path, transaction_count, config.random_seed))

    schedule = [scenario_type for scenario_type in SCENARIO_TYPES for _ in range(per_scenario)]
    random.Random(config.random_seed).shuffle(schedule)
    events: list[dict[str, Any]] = []
    for scenario_index, scenario_type in enumerate(schedule, start=1):
        historical_transactions = [next(pay_sim_rows) for _ in range(config.history_length)]
        transaction = next(pay_sim_rows) if scenario_type in TRANSACTION_SCENARIOS else None
        events.extend(run_scenario(
            scenario_index,
            scenario_type,
            transaction,
            historical_transactions,
            config.history_length,
            config.history_window_days,
        ))

    events.sort(key=lambda event: (event["user_id"], event["timestamp"]))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_path).open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=EVENT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(events)
    return events
