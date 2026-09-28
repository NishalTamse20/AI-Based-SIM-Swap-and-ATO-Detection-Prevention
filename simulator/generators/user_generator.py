"""Create synthetic user and account identities."""


def generate_identity(scenario_index: int) -> dict[str, str]:
    """Return unique synthetic user and account IDs for one scenario."""
    identifier = f"{scenario_index:06d}"
    return {"user_id": f"USR-{identifier}", "account_id": f"ACC-{identifier}"}
