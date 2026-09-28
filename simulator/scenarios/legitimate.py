"""Templates for legitimate SIM/eSIM changes."""

LEGITIMATE_SIM_REPLACEMENT = "LEGITIMATE_SIM_REPLACEMENT"
LEGITIMATE_ESIM_CHANGE = "LEGITIMATE_ESIM_CHANGE"


def scenario_definitions() -> list[tuple[str, int, str, str]]:
    return [
        (LEGITIMATE_SIM_REPLACEMENT, 0, "TELECOM", "SIM_REPLACEMENT"),
        (LEGITIMATE_ESIM_CHANGE, 0, "TELECOM", "ESIM_CHANGE"),
    ]
