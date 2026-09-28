"""Templates for the three suspicious scenarios in the specification."""

SUSPICIOUS_SIM_REPLACEMENT = "SUSPICIOUS_SIM_REPLACEMENT"
SUSPICIOUS_ESIM_CHANGE = "SUSPICIOUS_ESIM_CHANGE"
SIM_ESIM_AUTH_ANOMALY = "SIM_ESIM_AUTH_ANOMALY"


def scenario_definitions() -> list[tuple[str, int, str, str]]:
    return [
        (SUSPICIOUS_SIM_REPLACEMENT, 1, "TELECOM", "SIM_REPLACEMENT"),
        (SUSPICIOUS_ESIM_CHANGE, 1, "TELECOM", "ESIM_CHANGE"),
        (SIM_ESIM_AUTH_ANOMALY, 1, "TELECOM", "SIM_SWAP"),
    ]
