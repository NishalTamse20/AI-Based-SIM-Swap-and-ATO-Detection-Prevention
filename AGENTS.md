# SIM-Swap ATO Project Rules

## Project Goal

AI-based detection of potential account takeover following SIM/eSIM
events using multi-signal temporal correlation.

The project is a research/experimental prototype, not a production
fraud-blocking system.

## Core Principle

A SIM/eSIM change is a contextual trigger, not proof of account takeover.

The system must correlate SIM/eSIM events with other signals such as:

- device activity
- authentication activity
- account recovery
- account changes
- behavioural deviations
- transaction activity
- temporal relationships

## Core Telecom Events

Only these are core telecom events:

- SIM_SWAP
- SIM_REPLACEMENT
- ESIM_CHANGE

Number Porting / MNP is out of the current implementation scope.

## Data

PaySim is used only as a financial transaction source.

PaySim does NOT contain:

- SIM swap events
- SIM replacement events
- eSIM events
- device events
- authentication events
- recovery events

PaySim fields `isFraud` and `isFlaggedFraud` must not be used as the
project's `ato_label`.

Synthetic project data provides:

- SIM/eSIM events
- device events
- authentication events
- recovery events
- account events
- behavioural events
- temporal relationships

## Ground Truth

`ato_label` represents the simulated research scenario.

A SIM/eSIM event alone must never determine `ato_label`.

Keep:

- `ato_label`
- `scenario_type`

as separate fields.

## Research Comparisons

The project will compare:

1. SIM-only baseline
2. Multi-signal rule-based approach
3. Random Forest
4. Hybrid approach with temporal correlation

## Research Metrics

Potential evaluation metrics include:

- Precision
- Recall
- F1
- False Positive Rate
- False Negative Rate
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Error Analysis

Do not invent or report results before experiments are actually run.

## Do Not Invent

Never invent:

- dataset statistics
- model results
- accuracy
- precision/recall/F1
- risk thresholds
- risk weights
- real-world integrations
- research results
- production claims

## Out of Scope

Do not add without explicit approval:

- Number Porting / MNP subsystem
- real telecom integration
- real banking integration
- customer PII
- OTP interception
- production account blocking
- blockchain
- GNN
- GIS
- cash-out prediction
- Kafka/streaming infrastructure
- unnecessary microservices
- production deployment infrastructure

## Technology

Primary stack:

- Python
- FastAPI
- PostgreSQL
- React
- pandas
- NumPy
- scikit-learn
- Random Forest
- pytest

XGBoost is optional and requires explicit justification.

## Development Rules

1. Read the existing project documentation before modifying architecture.
2. Do not redesign the architecture without a clear technical reason.
3. Do not change the research methodology without explicit approval.
4. Prefer simple, maintainable implementation over unnecessary complexity.
5. Keep components modular and testable.
6. Use deterministic random seeds for synthetic data generation.
7. Keep raw PaySim data unchanged.
8. Prevent target leakage and future-event leakage.
9. Add tests for new functionality.
10. Run tests after significant changes.
11. Do not commit `.venv`.
12. Do not commit the raw PaySim dataset.
13. Do not create large generated datasets unnecessarily during development.

## Current Roadmap

Phase 0 — Freeze Research Scope & Requirements — DONE

Phase 1 — Research & Real-Case Analysis — DONE

Phase 2 — System Architecture & Data Specification — DONE

Phase 3 — Dataset Strategy & Unified Event Schema — DONE

Phase 4 — Synthetic Scenario & Data Generator — CURRENT

Phase 5 — Dataset Validation & Behavioural Baseline

Phase 6 — Feature Engineering

Phase 7 — Rule-Based Baseline

Phase 8 — Random Forest Model

Phase 9 — Temporal Correlation

Phase 10 — Hybrid Risk Engine

Phase 11 — Explainable Risk & Adaptive Response

Phase 12 — FastAPI Backend + PostgreSQL

Phase 13 — React Research Dashboard

Phase 14 — Security & Testing

Phase 15 — Research Experiments & Error Analysis

Phase 16 — Final Prototype Integration & Demo

Phase 17 — Research Paper + Final Report + Viva

## Current Task Rule

When working on a phase, implement only the current phase unless
explicitly instructed otherwise.

For Phase 4, do not implement:

- ML training
- Random Forest
- feature engineering
- rule engine
- risk scoring
- FastAPI
- PostgreSQL
- React
- dashboard

First complete and test the synthetic dataset generator.