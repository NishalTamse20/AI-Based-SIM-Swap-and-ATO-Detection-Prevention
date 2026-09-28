"""Create synthetic device identifiers without linking PaySim identities."""


def known_device_id(scenario_index: int) -> str:
    return f"DEV-{scenario_index:06d}-01"


def new_device_id(scenario_index: int) -> str:
    return f"DEV-{scenario_index:06d}-02"
