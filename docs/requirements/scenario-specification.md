# Scenario Specification

## Shared Pre-trigger Baseline History

Every scenario includes normal synthetic activity before its SIM/eSIM trigger. The development default is a 30-day pre-trigger window containing five successful LOGIN events and five successful TRANSACTION events. Historical activity uses the same known device and is generated independently of the post-trigger scenario and its ground-truth label. No post-trigger event is part of this history.

## Scenario 1 — Legitimate SIM Replacement

SIM_REPLACEMENT
→ known device
→ normal authentication
→ no abnormal recovery
→ normal transaction behaviour
→ ato_label = 0

## Scenario 2 — Legitimate eSIM Change

ESIM_CHANGE
→ known device or expected device
→ normal authentication
→ normal behaviour
→ ato_label = 0

## Scenario 3 — Suspicious SIM Replacement

SIM_REPLACEMENT
→ new/unrecognized device
→ authentication anomaly
→ recovery/password change
→ new beneficiary
→ unusual transaction
→ ato_label = 1

## Scenario 4 — Suspicious eSIM Change

ESIM_CHANGE
→ device deviation
→ authentication anomaly
→ behavioural deviation
→ suspicious transaction activity
→ ato_label = 1

## Scenario 5 — SIM/eSIM + Authentication Anomaly

SIM_SWAP or ESIM_CHANGE
→ new device
→ repeated authentication failures
→ abnormal behaviour
→ ato_label = 1

## Scenario 6 — Legitimate SIM Replacement with Device Migration

The synthetic user initiates an authorized SIM replacement and migrates to a new device.

SIM_REPLACEMENT
→ NEW_DEVICE
→ FAILED_LOGIN
→ PASSWORD_RESET
→ successful LOGIN on the new device
→ ato_label = 0

## Scenario 7 — Legitimate eSIM Migration with Authentication Recovery

The synthetic user initiates an authorized eSIM migration to a new device and completes authentication recovery.

ESIM_CHANGE
→ NEW_DEVICE
→ FAILED_LOGIN
→ PASSWORD_RESET
→ successful LOGIN on the new device
→ ato_label = 0

## Scenario 8 — Legitimate SIM Change with Repeated Login Difficulty

The synthetic user initiates an authorized SIM change, encounters repeated login failures, completes a password reset, and successfully logs in on the new device.

SIM_SWAP
→ NEW_DEVICE
→ repeated FAILED_LOGIN events
→ PASSWORD_RESET
→ successful LOGIN on the new device
→ ato_label = 0

For Scenarios 6–8, events are generated in the listed order within the frozen T+15-minute prediction window. Their legitimate labels represent the explicitly authorized synthetic scenario, not the absence of suspicious events. Each retains the shared normal pre-trigger history above.

## Important Rule

A SIM/eSIM event alone must never determine ato_label.

The label represents the complete simulated scenario.
