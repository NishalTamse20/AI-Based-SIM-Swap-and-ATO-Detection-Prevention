# Master Data Dictionary

## 1. Identity

| Field | Type | Description |
|---|---|---|
| user_id | string | Synthetic user identifier |
| account_id | string | Synthetic account identifier |
| event_id | string | Unique event identifier |

PaySim identifiers such as nameOrig/nameDest remain source identifiers
and are not treated as real customer identities.

---

## 2. Common Event Fields

| Field | Type | Description |
|---|---|---|
| timestamp | datetime | Event time |
| event_type | string | Unified event category |
| source | string | Event source |
| status | string | Event outcome/status |

Supported event_type values:

- TELECOM
- DEVICE
- AUTHENTICATION
- RECOVERY
- ACCOUNT
- TRANSACTION
- BEHAVIOUR

---

## 3. Telecom Fields

| Field | Type | Description |
|---|---|---|
| telecom_event | string | SIM/eSIM event |
| telecom_status | string | Event status |

Supported telecom_event values:

- SIM_SWAP
- SIM_REPLACEMENT
- ESIM_CHANGE

A telecom event is a contextual signal and does not by itself indicate ATO.

---

## 4. Device Fields

| Field | Type | Description |
|---|---|---|
| device_id | string | Synthetic device identifier |
| device_change | boolean | Whether device changed |
| device_deviation | boolean | Whether device differs from established behaviour |

---

## 5. Authentication Fields

| Field | Type | Description |
|---|---|---|
| auth_type | string | Authentication mechanism |
| auth_status | string | Authentication outcome |
| failed_attempts | integer | Number of failed authentication attempts |

Example auth_type values:

- PASSWORD
- OTP
- MFA
- BIOMETRIC

---

## 6. Recovery Fields

| Field | Type | Description |
|---|---|---|
| recovery_type | string | Recovery operation |
| recovery_status | string | Recovery outcome |

Example recovery_type values:

- PASSWORD_RESET
- ACCOUNT_RECOVERY
- DEVICE_RECOVERY

---

## 7. Account Fields

| Field | Type | Description |
|---|---|---|
| account_action | string | Account-level action |
| beneficiary_id | string | Synthetic beneficiary identifier |
| beneficiary_new | boolean | Whether beneficiary is new |

Example account_action:

- BENEFICIARY_ADD
- PROFILE_CHANGE
- PASSWORD_CHANGE

---

## 8. Transaction Fields

PaySim source mapping:

| PaySim field | Unified field / usage |
|---|---|
| step | transaction temporal position |
| type | transaction_type |
| amount | amount |
| nameOrig | source account reference |
| oldbalanceOrg | source balance before transaction |
| newbalanceOrig | source balance after transaction |
| nameDest | destination reference |
| oldbalanceDest | destination balance before transaction |
| newbalanceDest | destination balance after transaction |
| isFraud | PaySim reference label only |
| isFlaggedFraud | PaySim reference indicator only |

Unified transaction fields:

| Field | Type | Description |
|---|---|---|
| transaction_type | string | PaySim transaction type |
| amount | float | Transaction amount |
| source_balance_before | float | Source balance before transaction |
| source_balance_after | float | Source balance after transaction |
| destination_balance_before | float | Destination balance before transaction |
| destination_balance_after | float | Destination balance after transaction |
| beneficiary_id | string | Project beneficiary reference |

Supported PaySim transaction types:

- PAYMENT
- TRANSFER
- CASH_OUT
- CASH_IN
- DEBIT

---

## 9. Behavioural Fields

| Field | Type | Description |
|---|---|---|
| behaviour_deviation | float | Deviation from established behaviour |
| transaction_deviation | float | Deviation in transaction behaviour |
| device_deviation | boolean | Device deviation indicator |
| authentication_deviation | boolean | Authentication deviation indicator |

The exact calculation of behavioural features will be defined during
Phase 5/6 and must not be invented in the raw dataset.

---

## 10. Temporal Fields

| Field | Type | Description |
|---|---|---|
| hours_since_telecom_event | float | Time since relevant SIM/eSIM event |
| events_after_telecom | integer | Relevant events after telecom event |
| temporal_window | string | Research correlation window |

Temporal features must only use information available at prediction time.

---

## 11. Ground Truth

| Field | Type | Description |
|---|---|---|
| ato_label | integer | Project research target |
| scenario_type | string | Synthetic scenario category |

Values:

ato_label:
- 0 = legitimate scenario
- 1 = simulated ATO scenario

scenario_type examples:

- LEGITIMATE_SIM_CHANGE
- SUSPICIOUS_SIM_CHANGE
- SUSPICIOUS_ESIM_CHANGE
- DEVICE_AUTH_ANOMALY
- RECOVERY_TRANSACTION_ANOMALY

PaySim isFraud is NOT the project ato_label.

---

## 12. Leakage Rules

The following must never be used as model input:

- ato_label
- scenario_type
- future events
- future transactions
- label-generation variables that directly reveal the label
- PaySim isFraud when training the project ATO model
- PaySim isFlaggedFraud when training the project ATO model

PaySim fraud fields may remain available for separate source-data
analysis but are not the project target.

---

## 13. Identity Rules

All project identities are synthetic.

The generator creates:

- user_id
- account_id
- device_id
- beneficiary_id

PaySim source identifiers are retained only where necessary for
traceability during dataset construction.