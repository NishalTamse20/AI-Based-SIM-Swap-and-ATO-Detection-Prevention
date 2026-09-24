# Unified Event Schema

## 1. Purpose

The unified event schema provides a common representation for telecom, device, authentication, recovery, account and transaction events.

## 2. Event Categories

### Telecom

- SIM_SWAP
- SIM_REPLACEMENT
- ESIM_CHANGE

### Device

- NEW_DEVICE
- DEVICE_CHANGE
- DEVICE_FINGERPRINT_CHANGE

### Authentication

- LOGIN
- FAILED_LOGIN
- MFA_FAILURE
- MFA_SUCCESS
- PASSWORD_CHANGE

### Recovery

- PASSWORD_RESET
- ACCOUNT_RECOVERY
- RECOVERY_CONTACT_CHANGE

### Account

- NEW_BENEFICIARY
- BENEFICIARY_CHANGE
- PROFILE_CHANGE

### Transaction

- TRANSACTION
- TRANSFER
- WITHDRAWAL

### Behavioural

- LOGIN_TIME_DEVIATION
- LOCATION_DEVIATION
- DEVICE_DEVIATION
- TRANSACTION_PATTERN_DEVIATION

## 3. Common Event Fields

Every event should support the following common structure:

| Field | Description |
|---|---|
| event_id | Unique event identifier |
| user_id | Synthetic user identifier |
| event_type | Type of event |
| timestamp | Event timestamp |
| device_id | Associated device identifier |
| ip_address | Associated IP address |
| location | Event location |
| channel | Source/channel of event |
| status | Event result/status |
| amount | Transaction amount where applicable |
| beneficiary_id | Beneficiary reference where applicable |
| source | Event source |
| metadata | Additional event-specific information |

Not every event requires every field.

## 4. Example: SIM Swap Event

```json
{
  "event_id": "EVT-00001",
  "user_id": "USR-001",
  "event_type": "SIM_SWAP",
  "timestamp": "2026-09-24T10:30:00",
  "device_id": "DEV-102",
  "ip_address": "192.168.1.10",
  "location": "Pune",
  "channel": "telecom",
  "status": "SUCCESS",
  "source": "synthetic"
}