# Dataset Strategy

## 1\. Objective

The dataset will combine financial transaction history with simulated
telecom, device, authentication, recovery and behavioural events to
evaluate SIM/eSIM-triggered account takeover risk.

## 2\. Data Sources

### PaySim

Role:

* Financial transaction history
* Transaction behaviour
* Transaction amount patterns
* Sender/receiver relationships
* Financial fraud reference data

PaySim does NOT provide:

* SIM swap events
* eSIM events
* Device-change events
* Authentication events
* Account recovery events

### Synthetic Project Data

Synthetic data will represent:

* SIM\_SWAP
* SIM\_REPLACEMENT
* ESIM\_CHANGE
* Device changes
* Authentication events
* Password/account recovery
* Beneficiary changes
* Behavioural deviations
* Temporal relationships between events

## 3\. Unified Dataset

The final research dataset will contain:

1. User/account identity
2. Telecom events
3. Device events
4. Authentication events
5. Recovery events
6. Account events
7. Transaction events
8. Behavioural features
9. Temporal features
10. Research ground-truth label

## 4\. Identity Strategy

All project identities will be synthetic.

PaySim identifiers will not be treated as real customer identities.

A project-level synthetic user/account identifier will be used to
associate events belonging to the same simulated account.

## 5\. Ground Truth

The final ATO label will represent the simulated research scenario.

A SIM/eSIM event alone must NOT automatically produce an ATO label.

The label will be based on the complete simulated event sequence
defined by the scenario generator.

## 6\. Data Leakage Prevention

Features that directly reveal the target label must not be used as
model input.

Future events must not be used to predict an earlier event.

Training and test data must be separated appropriately.

## 7\. Research Dataset Requirements

The final dataset must support:

* Legitimate SIM/eSIM change scenarios
* Suspicious SIM/eSIM change scenarios
* Post-SIM authentication anomalies
* Device changes
* Recovery anomalies
* Transaction anomalies
* Behavioural deviations
* Temporal event sequences



