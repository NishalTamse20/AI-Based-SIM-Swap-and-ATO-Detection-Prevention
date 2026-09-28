"""Validate the generated event dataset and create baseline profiles."""

import argparse
from pathlib import Path

from simulator.baseline.behavioural_baseline import DEFAULT_BASELINE_PATH, build_profiles_from_csv
from simulator.config.generator_config import DEFAULT_OUTPUT_PATH
from simulator.validation.dataset_validator import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate event data and build pre-trigger profiles.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()

    validation = validate_dataset(args.input)
    if not validation.is_valid:
        for error in validation.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    profiles = build_profiles_from_csv(args.input, args.output)
    print(f"Validation passed: {validation.event_count} events, {validation.user_count} users, {validation.scenario_count} scenarios")
    print(f"Pre-trigger events validated: {validation.history_event_count}")
    print("Baseline cutoff verified: pre-trigger events only; post-trigger events excluded")
    print(f"Baseline profiles written: {len(profiles)} at {args.output}")


if __name__ == "__main__":
    main()
