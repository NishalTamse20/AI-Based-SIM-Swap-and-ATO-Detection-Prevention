"""Configuration and defaults for Phase 4 dataset generation."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAYSIM_PATH = PROJECT_ROOT / "data" / "raw" / "paysim" / "PS_20174392719_1491204439457_log.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "ato_synthetic_events.csv"


@dataclass(frozen=True)
class GeneratorConfig:
    """Settings for reproducible synthetic scenario generation."""

    scenario_count: int = 1000
    random_seed: int = 42
    paysim_path: Path = DEFAULT_PAYSIM_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    history_length: int = 5
    history_window_days: int = 30

    def __post_init__(self) -> None:
        if self.scenario_count <= 0:
            raise ValueError("scenario_count must be greater than zero")
        if self.history_length <= 0:
            raise ValueError("history_length must be greater than zero")
        if self.history_window_days <= 0:
            raise ValueError("history_window_days must be greater than zero")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic SIM/eSIM ATO scenario events.")
    parser.add_argument("--scenario-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paysim-path", type=Path, default=DEFAULT_PAYSIM_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--history-window-days", type=int, default=30)
    args = parser.parse_args()
    from simulator.generators.dataset_generator import generate_dataset

    config = GeneratorConfig(
        scenario_count=args.scenario_count,
        random_seed=args.seed,
        paysim_path=args.paysim_path,
        output_path=args.output_path,
        history_length=args.history_length,
        history_window_days=args.history_window_days,
    )
    events = generate_dataset(config)
    print(f"Generated {len(events)} events from {config.scenario_count} scenarios at {config.output_path}")
